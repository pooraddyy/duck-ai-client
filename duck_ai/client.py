from __future__ import annotations

import base64
import json
import logging
import random
import threading
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

import httpx

from ._challenge import make_fe_signals, solve_challenge
from ._durable import generate_jwk
from .exceptions import (
    APIError,
    ChallengeError,
    ConversationLimitError,
    DuckChatError,
    RateLimitError,
)
from .models import (
    History,
    ImagePart,
    Message,
    ModelType,
    Role,
    image_generation,
    model_supports_web_search,
    resolve_effort,
    resolve_model,
    vision_capable_default,
)

log = logging.getLogger("duck_ai")

_BASE = "https://duck.ai"
_DDG_BASE = "https://duckduckgo.com"
_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.6 Safari/605.1.15"
)
# duck.ai rotates this string. Library users can override via `fe_version=`.
_DEFAULT_FE_VERSION = "serp_20260424_180649_ET-0bdc33b2a02ebf8f235def65d887787f694720a1"
# Tool channels duck.ai exposes via `metadata.toolChoice`. The non-WebSearch
# channels stay off by default (matches the duck.ai web client behaviour).
# `WebSearch` is opt-in per call via the `web_search=True` argument.
_TOOL_CHOICE_OFF = {
    "NewsSearch": False,
    "VideosSearch": False,
    "LocalSearch": False,
    "WeatherForecast": False,
}

# duck.ai routes the `image-generation` model to a different endpoint that
# accepts both pure prompt-to-image and image+prompt (image edit) payloads.
_CHAT_PATH = "/duckchat/v1/chat"
_IMAGES_PATH = "/duckchat/v1/images"

class DuckChat:
    def __init__(
        self,
        model: Union[ModelType, str] = "gpt4",
        *,
        effort: Optional[str] = None,
        user_agent: Optional[str] = None,
        fe_version: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 0.6,
        warm_session: bool = True,
        aggressive_warm: bool = True,
    ):
        self.model = resolve_model(model)
        self.effort = effort
        self.user_agent = user_agent or _DEFAULT_UA
        self.fe_version = fe_version or _DEFAULT_FE_VERSION
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.backoff_base = max(0.0, float(backoff_base))
        self.aggressive_warm = aggressive_warm
        self.history = History(model=self.model)
        self._jwk: Optional[Dict[str, Any]] = None
        self._jwk_lock = threading.Lock()
        self._owns_client = client is None
        self._client = client or httpx.Client(
            http2=False,
            timeout=httpx.Timeout(timeout, connect=15.0, read=timeout),
            headers={
                "User-Agent": self.user_agent,
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": f"{_BASE}/",
                "Origin": _BASE,
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Ch-Ua": '"Not.A/Brand";v="99", "Chromium";v="136"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"macOS"',
            },
            follow_redirects=True,
        )
        self._warmed = not warm_session
        self._pending_hash: Optional[str] = None
        self._seeded = False
        if warm_session:
            try:
                self._warm()
            except Exception as e:
                # Warm-up is best-effort; the retry layer (if enabled) will
                # cover any residual cold-start failure.
                log.debug("warm-up failed: %s", e)
        if warm_session and aggressive_warm:
            try:
                self._seed_session()
            except Exception as e:
                # Seed is also best-effort. If the network/duck.ai is down
                # we'll still surface a real error on the user's first call.
                log.debug("session seed failed: %s", e)

    # ------------------------------------------------------------------ ctx
    def __enter__(self) -> "DuckChat":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            try:
                self._client.close()
            except Exception:
                pass

    def reset(self) -> None:
        self.history.clear()

    # ----------------------------------------------------------------- warm
    def _warm(self) -> None:
        # Browser-faithful warm-up:
        #   1. Visit the duck.ai homepage so we collect any Set-Cookie headers
        #      it issues (session, AB-test, A/A buckets, etc).
        #   2. Set DuckDuckGo's cookie-consent + locale cookies on both
        #      duck.ai and duckduckgo.com domains. Without these the chat
        #      endpoint frequently 418s on the very first call.
        #   3. Visit the chat-mode SERP URL (`/?q=...&ia=chat`) which is what
        #      a real browser hits when a user clicks "AI Chat" — this seeds
        #      the second-tier cookies the chat endpoint inspects.
        if self._warmed:
            return
        try:
            html_headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            }
            self._client.get(f"{_BASE}/", headers=html_headers, timeout=10.0)
            for k, v in (
                ("5", "1"),         # cookie-consent acknowledged
                ("ah", "wt-wt"),    # region: world-wide
                ("dcs", "1"),       # duckchat enabled
                ("dcm", "3"),       # duckchat model picker dismissed
                ("isRecentChatOn", "1"),
            ):
                for dom in (".duck.ai", ".duckduckgo.com"):
                    try:
                        self._client.cookies.set(k, v, domain=dom)
                    except Exception:
                        pass
            self._client.get(
                f"{_DDG_BASE}/?q=DuckDuckGo+AI+Chat&ia=chat&duckai=1",
                headers=html_headers,
                timeout=10.0,
            )
        finally:
            self._warmed = True

    def _seed_session(self) -> None:
        # Make a single tiny throwaway chat call internally so that
        # `_pending_hash` is populated with a server-issued, freshly-rotated
        # challenge. After this seed, every user-facing call uses the
        # rotated hash chain — which is what gives us first-attempt success
        # without leaning on the retry loop.
        if self._seeded:
            return
        msgs = [Message(role=Role.User.value, content="hi").to_dict()]
        # Seed with the SAME model the user picked. duck.ai's challenge
        # rotation appears to bind the next hash to the model used in the
        # request that produced it; using a different model on the seed
        # yields a hash the chat endpoint rejects on the user's first call.
        payload = self._build_payload(msgs, model=self.model, can_use_tools=False)
        # Use a small internal retry budget here so the seed itself is
        # robust on a true cold start. This budget is independent of
        # `self.max_retries` (which is the user-facing retry budget).
        prev = self.max_retries
        self.max_retries = 4
        try:
            try:
                for _ in self._stream_with_retry(payload):
                    # We only need to drive the stream far enough to capture
                    # the rotated challenge header from the response; we
                    # don't care about the model output.
                    if self._pending_hash:
                        break
            except Exception as e:
                # A failed seed must not leave a poisoned `_pending_hash`
                # behind; the user's first real call would inherit it and
                # 418 immediately. Drop it and let /status be hit fresh.
                self._pending_hash = None
                log.debug("session seed stream failed: %s", e)
        finally:
            self.max_retries = prev
            self._seeded = True

    # ------------------------------------------------------------ challenge
    def _fetch_challenge_header(self) -> str:
        # Reuse the rotating header captured from the previous chat response;
        # this is exactly how the duck.ai web client behaves and avoids the
        # "first call works, second 418s" trap (and vice versa).
        pending = getattr(self, "_pending_hash", None)
        if pending:
            self._pending_hash = None
            return solve_challenge(pending, self.user_agent)
        r = self._client.get(
            f"{_BASE}/duckchat/v1/status",
            headers={
                "Accept": "*/*",
                "x-vqd-accept": "1",
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
                "Referer": f"{_BASE}/",
                "Origin": _BASE,
            },
            timeout=self.timeout,
        )
        if r.status_code == 429:
            raise RateLimitError(r.text)
        if r.status_code >= 400:
            raise APIError(
                f"status endpoint failed: {r.status_code}",
                r.status_code,
                r.text,
            )
        challenge = r.headers.get("x-vqd-hash-1")
        if not challenge:
            raise DuckChatError("server did not return x-vqd-hash-1 challenge")
        return solve_challenge(challenge, self.user_agent)

    def _get_jwk(self) -> Dict[str, Any]:
        if self._jwk is None:
            with self._jwk_lock:
                if self._jwk is None:
                    self._jwk = generate_jwk()
        return self._jwk

    # ------------------------------------------------------------- payload
    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        can_use_tools: bool = True,
        effort: Optional[str] = None,
        web_search: bool = False,
    ) -> Dict[str, Any]:
        m = resolve_model(model or self.model)
        tool_choice: Dict[str, bool] = dict(_TOOL_CHOICE_OFF)
        # Only flip WebSearch on for models that actually expose it; sending
        # it for an unsupported model just gets it silently ignored, but it
        # also makes the request signature look unusual to the anti-abuse
        # heuristics, so we keep it off there.
        if web_search and model_supports_web_search(m):
            tool_choice["WebSearch"] = True
        payload: Dict[str, Any] = {
            "model": m,
            "metadata": {"toolChoice": tool_choice},
            "messages": messages,
            "canUseTools": can_use_tools,
        }
        eff = resolve_effort(m, effort if effort is not None else self.effort)
        if eff is not None:
            payload["reasoningEffort"] = eff
        payload["canUseApproxLocation"] = None
        # The /images endpoint does not accept the durableStream public key.
        # Only attach it for the regular chat endpoint.
        if m != image_generation:
            payload["durableStream"] = {
                "messageId": str(uuid.uuid4()),
                "conversationId": str(uuid.uuid4()),
                "publicKey": self._get_jwk(),
            }
        return payload

    @staticmethod
    def _endpoint_for(model: str) -> str:
        # `image-generation` routes to /duckchat/v1/images for both pure
        # text-to-image and image-edit (text + image input) requests.
        return _IMAGES_PATH if model == image_generation else _CHAT_PATH

    @staticmethod
    def _has_image(messages: List[Dict[str, Any]]) -> bool:
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "image":
                        return True
        return False

    # ----------------------------------------------------- HTTP + SSE iter
    def _chat_stream(self, payload: Dict[str, Any]):
        hash_header = self._fetch_challenge_header()
        path = self._endpoint_for(payload.get("model", self.model))
        return self._client.stream(
            "POST",
            f"{_BASE}{path}",
            content=json.dumps(payload),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "x-vqd-hash-1": hash_header,
                "x-fe-signals": make_fe_signals(),
                "x-fe-version": self.fe_version,
                "Referer": f"{_BASE}/",
                "Origin": _BASE,
            },
        )

    @staticmethod
    def _raise_for_status(resp: "httpx.Response") -> None:
        try:
            resp.read()
            body = resp.text
        except Exception:
            body = ""
        if resp.status_code == 418:
            raise ChallengeError(f"server rejected challenge: {body[:200]}")
        if resp.status_code == 429:
            if "ERR_CONVERSATION_LIMIT" in body:
                raise ConversationLimitError(body)
            raise RateLimitError(body)
        raise APIError(
            f"chat failed: HTTP {resp.status_code}", resp.status_code, body
        )

    @staticmethod
    def _iter_sse(resp: "httpx.Response") -> Iterator[str]:
        for raw in resp.iter_lines():
            if not raw:
                continue
            if not raw.startswith("data:"):
                continue
            data = raw[5:].lstrip()
            if data:
                yield data

    # ---------------------------------------------------------- retry core
    def _attempt_stream(
        self, payload: Dict[str, Any]
    ) -> Iterator[dict]:
        with self._chat_stream(payload) as resp:
            if resp.status_code != 200:
                self._raise_for_status(resp)
            # Capture the next challenge header for the following request.
            new_hash = resp.headers.get("x-vqd-hash-1")
            if new_hash:
                self._pending_hash = new_hash
            saw_any = False
            for data in self._iter_sse(resp):
                if data == "[DONE]":
                    return
                if (
                    data.startswith("[CHAT_TITLE")
                    or data.startswith("[LIMIT")
                    or data.startswith("[PING")
                ):
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if obj.get("action") == "error":
                    msg = (
                        obj.get("type")
                        or obj.get("error")
                        or json.dumps(obj)
                    )
                    if obj.get("status") == 429:
                        if msg == "ERR_CONVERSATION_LIMIT":
                            raise ConversationLimitError(msg)
                        raise RateLimitError(msg)
                    if msg in ("ERR_CHALLENGE", "ERR_INVALID_CHALLENGE"):
                        raise ChallengeError(str(msg))
                    raise APIError(str(msg), obj.get("status"), data)
                saw_any = True
                yield obj
            if not saw_any:
                # Empty stream — duck.ai sometimes does this when the
                # challenge was accepted but the model rejected the request.
                # Treat as transient.
                raise APIError("empty stream from duck.ai", 200, "")

    def _stream_with_retry(
        self, payload: Dict[str, Any]
    ) -> Iterator[dict]:
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            yielded = False
            try:
                for item in self._attempt_stream(payload):
                    yielded = True
                    yield item
                return
            except (ChallengeError, httpx.RemoteProtocolError, httpx.ReadError) as e:
                # The cached `_pending_hash` (if any) is what produced this
                # rejection — drop it so the next attempt fetches a fresh
                # challenge from /status instead of replaying the bad one.
                self._pending_hash = None
                if yielded:
                    # Mid-stream failure: retrying would replay the chunks
                    # we already emitted, which makes downstream consumers
                    # (e.g. Telegram bots that edit a message with the
                    # cumulative text) send the same text twice and trip
                    # "Message is not modified". Surface the error instead
                    # of silently producing duplicate output.
                    raise
                last_exc = e  # transient: refresh challenge and retry
            except APIError as e:
                # Retry on 5xx and on the synthetic empty-stream APIError.
                if e.status_code is None or e.status_code >= 500:
                    if yielded:
                        raise
                    last_exc = e
                else:
                    raise
            except RateLimitError as e:
                # 429s without conversation-limit semantics may clear if we
                # back off; ConversationLimitError is a subclass of
                # RateLimitError but we already raised it specifically.
                if isinstance(e, ConversationLimitError):
                    raise
                if yielded:
                    raise
                last_exc = e
            except httpx.TimeoutException as e:
                if yielded:
                    raise
                last_exc = e
            # Backoff with jitter before the next attempt.
            if attempt < self.max_retries - 1:
                delay = self.backoff_base * (2**attempt) + random.uniform(0, 0.25)
                log.debug(
                    "duck.ai retry %d/%d after %.2fs",
                    attempt + 2,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise DuckChatError("exhausted retries with no specific error")

    # ----------------------------------------------------------- public API
    def stream(
        self,
        prompt: Union[str, List[Union[str, ImagePart, dict]]],
        *,
        remember: bool = True,
        model: Optional[Union[ModelType, str]] = None,
        effort: Optional[str] = None,
        web_search: bool = False,
    ) -> Iterator[str]:
        if remember:
            self.history.add_user(prompt)
            messages = self.history.to_messages()
        else:
            messages = [Message(role=Role.User.value, content=prompt).to_dict()]
        is_mm = self._has_image(messages)
        if model is not None:
            use_model = resolve_model(model)
        elif is_mm:
            # Only re-route to a vision model if the user's choice can't see.
            from .models import model_supports_vision

            use_model = (
                self.model
                if model_supports_vision(self.model)
                else vision_capable_default()
            )
        else:
            use_model = self.model
        # duck.ai's challenge rotation binds the next `x-vqd-hash-1` to the
        # model used in the request that produced it. The session seed and
        # any previous chat call cached a hash bound to `self.model`. If
        # this call uses a different model (e.g. an image upload routed to
        # a vision-capable model), replaying that cached hash makes the
        # server return 418 ERR_CHALLENGE. Drop it so we fetch a fresh
        # challenge from /status for this request.
        if use_model != self.model:
            self._pending_hash = None
        payload = self._build_payload(
            messages,
            model=use_model,
            effort=effort,
            web_search=web_search,
        )
        collected: List[str] = []
        for obj in self._stream_with_retry(payload):
            chunk = obj.get("message") or ""
            if chunk:
                collected.append(chunk)
                yield chunk
        if remember and collected:
            self.history.add_assistant("".join(collected))

    def ask(
        self,
        prompt: Union[str, List[Union[str, ImagePart, dict]]],
        *,
        remember: bool = True,
        model: Optional[Union[ModelType, str]] = None,
        effort: Optional[str] = None,
        web_search: bool = False,
    ) -> str:
        return "".join(
            self.stream(
                prompt,
                remember=remember,
                model=model,
                effort=effort,
                web_search=web_search,
            )
        )

    def ask_with_image(
        self,
        prompt: str,
        image: Union[str, bytes, ImagePart],
        *,
        mime_type: str = "image/png",
        remember: bool = True,
        model: Optional[Union[ModelType, str]] = None,
        effort: Optional[str] = None,
        web_search: bool = False,
    ) -> str:
        part = self._coerce_image(image, mime_type)
        return self.ask(
            [prompt, part],
            remember=remember,
            model=model,
            effort=effort,
            web_search=web_search,
        )

    @staticmethod
    def _coerce_image(
        image: Union[str, bytes, ImagePart], mime_type: str
    ) -> ImagePart:
        if isinstance(image, ImagePart):
            return image
        if isinstance(image, bytes):
            return ImagePart.from_bytes(image, mime_type=mime_type)
        if isinstance(image, str):
            if image.startswith("data:"):
                return ImagePart(image=image, mime_type=mime_type)
            return ImagePart.from_path(image)
        raise TypeError(f"unsupported image type: {type(image).__name__}")

    def generate_image(
        self,
        prompt: str,
        *,
        save_to: Optional[str] = None,
    ) -> bytes:
        return self._run_image_request(
            content=prompt,
            save_to=save_to,
        )

    def edit_image(
        self,
        prompt: str,
        image: Union[str, bytes, ImagePart],
        *,
        mime_type: str = "image/png",
        save_to: Optional[str] = None,
    ) -> bytes:
        # Image edit on duck.ai is the same /duckchat/v1/images endpoint as
        # image generation, but the user message carries both a text caption
        # and an `image` content part. The server returns the edited image
        # in the same `partial-image` / `generated-image` SSE channel.
        part = self._coerce_image(image, mime_type)
        return self._run_image_request(
            content=[prompt, part],
            save_to=save_to,
        )

    def _run_image_request(
        self,
        *,
        content: Union[str, List[Union[str, ImagePart, dict]]],
        save_to: Optional[str],
    ) -> bytes:
        messages = [Message(role=Role.User.value, content=content).to_dict()]
        # The image-generation endpoint is bound to its own model and does
        # not share the rotated hash chain with the chat endpoint, so drop
        # any cached pending hash before we route here.
        if self.model != image_generation:
            self._pending_hash = None
        payload = self._build_payload(
            messages,
            model=image_generation,
            can_use_tools=True,
        )
        partials: List[str] = []
        final: Optional[str] = None
        for obj in self._stream_with_retry(payload):
            role = obj.get("role") or ""
            result = obj.get("result") or ""
            if role == "partial-image" and result:
                partials.append(result)
            elif role in ("generated-image", "image") and result:
                final = result
        b64 = final if final else "".join(partials)
        if not b64:
            raise DuckChatError("image generation returned no data")
        if "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        if save_to:
            with open(save_to, "wb") as f:
                f.write(data)
        return data
