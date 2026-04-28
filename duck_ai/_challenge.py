from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

from .exceptions import ChallengeError

_STUBS_PATH = os.path.join(os.path.dirname(__file__), "_stubs.js")
_STUBS_TEMPLATE: Optional[str] = None
_STUBS_LOCK = threading.Lock()
_RUNTIME_LOCK = threading.Lock()

def _load_stubs() -> str:
    global _STUBS_TEMPLATE
    if _STUBS_TEMPLATE is None:
        with _STUBS_LOCK:
            if _STUBS_TEMPLATE is None:
                with open(_STUBS_PATH, "r", encoding="utf-8") as f:
                    _STUBS_TEMPLATE = f.read()
    return _STUBS_TEMPLATE

def _b64_sha256(s: str) -> str:
    return base64.b64encode(hashlib.sha256(s.encode("utf-8")).digest()).decode(
        "ascii"
    )

def _has_minirar() -> bool:
    try:
        import py_mini_racer  # noqa: F401

        return True
    except Exception:
        return False

def _extract_html_inputs(js: str) -> List[str]:
    found: List[str] = []
    seen = set()
    for m in re.finditer(r"""(['"])(<[^'"]{1,400}?)\1""", js):
        s = m.group(2)
        if s in seen:
            continue
        seen.add(s)
        found.append(s)
    return found

def _serialize_etree(node) -> str:
    out = ""
    for child in node:
        tag = child.tag
        if not isinstance(tag, str):
            continue
        if "}" in tag:
            tag = tag.split("}", 1)[1]
        attrs = "".join(f' {k}="{v}"' for k, v in child.attrib.items())
        out += f"<{tag}{attrs}>"
        if child.text:
            out += child.text
        out += _serialize_etree(child)
        out += f"</{tag}>"
        if child.tail:
            out += child.tail
    return out

def _normalize_html(s: str) -> Tuple[str, int]:
    try:
        import html5lib  # type: ignore

        frag = html5lib.parseFragment(
            s, treebuilder="etree", namespaceHTMLElements=False
        )
        inner = _serialize_etree(frag)
        count = sum(1 for e in frag.iter() if isinstance(e.tag, str)) - 1
        if count < 0:
            count = 0
        return inner, count
    except Exception:
        return s, 0

def _build_html_lookup(js: str) -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    for s in _extract_html_inputs(js):
        norm, count = _normalize_html(s)
        lookup[s] = {"html": norm, "count": count}
    return lookup

def solve_challenge(challenge_b64: str, user_agent: str) -> str:
    if not _has_minirar():
        raise ChallengeError(
            "JavaScript engine not available. Install with: pip install mini-racer"
        )
    try:
        from py_mini_racer import MiniRacer
    except Exception as e:  # pragma: no cover - import guarded above
        raise ChallengeError(f"failed to import mini-racer: {e}") from e

    try:
        js = base64.b64decode(challenge_b64).decode("utf-8", errors="replace")
    except Exception as e:
        raise ChallengeError(f"challenge is not valid base64: {e}") from e

    html_lookup = _build_html_lookup(js)
    stubs = _load_stubs().replace("__DDG_REAL_UA__", json.dumps(user_agent))
    stubs = stubs.replace("__DDG_HTML_LOOKUP__", json.dumps(html_lookup))

    # MiniRacer isolates are not thread-safe; serialise solves.
    with _RUNTIME_LOCK:
        ctx = MiniRacer()
        try:
            ctx.eval(stubs)
            ctx.eval(
                "(%s).then(function(v){__R=v;}).catch(function(e){__E=String((e&&e.stack)||e);});"
                % js
            )
            for _ in range(100):
                if ctx.execute("__R !== null || __E !== null"):
                    break
                time.sleep(0.02)
            err = ctx.execute("__E")
            if err:
                raise ChallengeError(f"challenge JS error: {err}")
            res = ctx.execute("__R")
        finally:
            try:
                # Free V8 resources so we don't leak isolates between calls.
                del ctx
            except Exception:
                pass

    if not isinstance(res, dict):
        raise ChallengeError("challenge returned no result")
    client_hashes = list(res.get("client_hashes") or [])
    if not client_hashes:
        raise ChallengeError("challenge returned empty client_hashes")
    client_hashes[0] = user_agent
    res["client_hashes"] = [_b64_sha256(t) for t in client_hashes]
    payload = json.dumps(res, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")

def make_fe_signals(*, duration_ms: int = 3000) -> str:
    # Build a realistic event timeline so duck.ai's anti-bot heuristics
    # see a "real user" interaction, not an empty signal envelope.
    # Browser sends keypress/focus events with relative timestamps; we
    # mimic that pattern with reasonable jitter.
    import random as _r

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - duration_ms
    events = []
    t = _r.randint(80, 180)
    events.append({"name": "onboarding_impression_1", "delta": t})
    t += _r.randint(120, 260)
    events.append({"name": "onboarding_impression_2", "delta": t})
    t += _r.randint(200, 500)
    events.append({"name": "startNewChat", "delta": t})
    n_keys = _r.randint(6, 18)
    for _ in range(n_keys):
        t += _r.randint(40, 180)
        events.append({"name": "user_input", "delta": t})
    t += _r.randint(120, 350)
    events.append({"name": "user_submit", "delta": t})
    payload = {
        "start": start_ms,
        "events": events,
        "end": max(t + _r.randint(20, 90), duration_ms),
    }
    return base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
