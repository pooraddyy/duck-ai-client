# p2d-duck

Free, no-API-key Python client for **DuckDuckGo AI Chat** ([duck.ai](https://duck.ai)).

- **Single sync client** — built on `httpx`. No async twin to drift out of sync.
- **Auto-retry on challenge failures** — fresh `x-vqd-hash-1` solve, exponential backoff with jitter. Works on the first call, not the third.
- **Session warm-up** — cookies are seeded on construction so the very first chat request looks like a real browser session.
- 6 chat models + image generation: GPT-4o mini, GPT-5 mini, Claude Haiku 4.5, Llama 4 Scout, Mistral Small, GPT-OSS 120B.
- **Reasoning effort switching** (`fast` vs `reasoning`).
- **Image generation** (`image-generation` model).
- **Image edit** (caption + source image → edited image, same model).
- **Image upload / multimodal** (vision-capable chats).
- **Web search tool** opt-in per call on supported models (GPT-4o mini, GPT-5 mini, Claude Haiku 4.5).
- Built-in solver for the `x-vqd-hash-1` JS challenge via `mini-racer`.
- CLI: `p2d-duck`.
- No account, no API key, no server, no fee.

## Install

```bash
pip install p2d-duck
```

> The Python import name is still `duck_ai`, e.g. `from duck_ai import DuckChat`.

Latest release: [**v1.1.0**](https://github.com/pooraddyy/p2d-duck/releases/tag/v1.1.0) — see the full changelog on the release page.

## Quickstart

```python
from duck_ai import DuckChat, gpt4

with DuckChat(model=gpt4) as duck:
    print(duck.ask("Explain quantum tunneling in one sentence."))
```

Switch models by passing an alias:

```python
from duck_ai import DuckChat

with DuckChat(model="claude") as duck:
    print(duck.ask("Hi Claude!"))
```

You can pass any of: `"gpt4"`, `"gpt5_mini"`, `"claude"`, `"llama"`, `"mistral"`, `"gpt-oss"`, `"image"`, or any raw model id.

## Streaming

```python
from duck_ai import DuckChat

with DuckChat() as duck:
    for chunk in duck.stream("Write a 4-line haiku about ducks."):
        print(chunk, end="", flush=True)
```

## Multi-turn conversation

`DuckChat` keeps history automatically. Use `duck.reset()` to start fresh.

```python
from duck_ai import DuckChat, claude

with DuckChat(model=claude) as duck:
    duck.ask("My name is Alice. Remember it.")
    print(duck.ask("What is my name?"))
    duck.reset()
```

## Reasoning vs Fast mode

```python
from duck_ai import DuckChat, gpt5_mini, claude

with DuckChat(model=gpt5_mini, effort="fast") as duck:
    print(duck.ask("Quick: 2+2?"))

with DuckChat(model=claude, effort="reasoning") as duck:
    print(duck.ask("Solve: I speak without a mouth..."))
```

| Model | Supports `fast` / `reasoning`? | Default effort |
|---|---|---|
| `gpt4` (gpt-4o-mini) | no | — |
| `gpt5_mini` | yes | `minimal` |
| `claude` (Haiku 4.5) | yes | `low` |
| `gpt_oss` (gpt-oss 120B) | yes | `low` |
| `llama` (Llama 4 Scout) | no | — |
| `mistral` (Small 2603) | no | — |

## Image generation

```python
from duck_ai import DuckChat, image_generation

with DuckChat(model=image_generation) as duck:
    duck.generate_image(
        "a cute rubber duck wearing a wizard hat, digital art",
        save_to="duck_wizard.jpg",
    )
```

## Image edit

Edit an existing image with a caption. Same `image-generation` model, same
endpoint as image generation, but the user message carries both a text
caption and an `ImagePart`:

```python
from duck_ai import DuckChat, image_generation

with DuckChat(model=image_generation) as duck:
    duck.edit_image(
        "make the duck wear a tiny chef hat",
        "duck_wizard.jpg",
        save_to="duck_chef.jpg",
    )
```

## Web search

Web search is an opt-in tool, off by default. Pass `web_search=True` per call.
It is only sent to models that actually support it (GPT-4o mini, GPT-5 mini,
Claude Haiku 4.5); the flag is silently ignored for the others.

```python
from duck_ai import DuckChat, gpt5_mini, model_supports_web_search

assert model_supports_web_search(gpt5_mini)

with DuckChat(model=gpt5_mini) as duck:
    print(duck.ask(
        "What did Apple announce at WWDC this year?",
        web_search=True,
    ))
```

CLI:

```bash
p2d-duck -m gpt5_mini chat "Latest SpaceX launch?" --web-search
```

## Image upload (multimodal)

```python
from duck_ai import DuckChat, ImagePart

with DuckChat() as duck:
    print(duck.ask_with_image("What is in this image?", "photo.jpg"))

    print(duck.ask([
        "Compare these two images:",
        ImagePart.from_path("a.png"),
        ImagePart.from_path("b.png"),
    ]))
```

If your selected model has no vision capability, multimodal requests are
automatically routed to a vision-capable model (`gpt-5-mini`).

## CLI

```bash
p2d-duck                                       # interactive REPL
p2d-duck chat "Hello, who are you?"
p2d-duck -m claude chat "Hi Claude!"
p2d-duck -m gpt5_mini -e reasoning chat "Solve x^2 - 5x + 6 = 0"
p2d-duck chat "Describe this" --image cat.jpg
p2d-duck image "a watercolor moon over a lake" -o moon.jpg
p2d-duck edit "make the cat wear sunglasses" --image cat.jpg -o cat_cool.jpg
p2d-duck -m claude chat "Top news today" --web-search
p2d-duck models                                # list known models
```

> The legacy `duck-ai` command is also installed for backwards compatibility,
> so existing scripts keep working.

## Reliability — what changed

The previous version of this client raised on the very first 418
`ERR_CHALLENGE`, leaving callers to retry manually 2-3 times. This rewrite:

1. **Warms the HTTP session** by hitting the duck.ai homepage on construction
   so cookies are present before the first chat request.
2. **Wraps every chat call in a retry loop**. On `ChallengeError`,
   `RemoteProtocolError`, transient `RateLimitError`, dropped streams, or an
   empty SSE response, it re-fetches the `x-vqd-hash-1` challenge and tries
   again with exponential backoff + jitter.
3. **Treats `ConversationLimitError` as terminal** so we don't burn retries on
   a permanent failure.
4. **Refuses to fall back to a fake RSA key** for durable streams. If
   `cryptography` isn't installed we raise immediately instead of sending a
   garbage public key the server will reject.

You can tune retries with `DuckChat(max_retries=4, backoff_base=0.6)`.

## Models

```python
from duck_ai import (
    DuckChat,
    gpt4, gpt5_mini, claude, llama, mistral, gpt_oss, image_generation,
)
```

| Alias | Resolved model id |
|---|---|
| `gpt4` / `gpt4o_mini` | `gpt-4o-mini` |
| `gpt5` / `gpt5_mini` | `gpt-5-mini` |
| `claude` / `claude_haiku` | `claude-haiku-4-5` |
| `llama` / `llama4_scout` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` |
| `mistral` / `mistral_small` | `mistral-small-2603` |
| `gpt_oss` / `gpt_oss_120b` | `tinfoil/gpt-oss-120b` |
| `image_generation` / `image` | `image-generation` |

You can also pass any model string directly: `DuckChat(model="gpt-4o-mini")`.

## How it works

DuckDuckGo's AI Chat backend (`duck.ai/duckchat/v1/*`) requires a per-request
proof-of-work challenge encoded in the `x-vqd-hash-1` header. The server returns
an obfuscated JavaScript snippet that must be evaluated against a browser-like
environment to compute valid client hashes.

`p2d-duck` ships with:

1. A minimal browser-DOM JavaScript shim (`_stubs.js`).
2. An embedded V8 isolate via [`mini-racer`](https://pypi.org/project/mini-racer/)
   to execute the challenge.
3. SHA-256 hashing of the resulting fingerprint values.
4. A real RSA-OAEP public key for resumable streams (durable streams).

No external Node.js install is required.

## Exceptions

| Exception | When |
|---|---|
| `DuckChatError` | Generic error; base class. |
| `ChallengeError` | Couldn't solve the JS challenge. |
| `RateLimitError` | HTTP 429 from the server. |
| `ConversationLimitError` | Too many turns in one session (terminal). |
| `APIError` | Any other non-200 response (`.status_code`, `.body`). |

If you hit `HTTP 418 ERR_CHALLENGE` *after* the retry budget is exhausted,
your IP is being throttled by duck.ai's anti-abuse system. Wait 30-60 seconds
between consecutive requests.

## License

MIT. See [LICENSE](LICENSE).

## Disclaimer

This is an **unofficial** reverse-engineered client. It is not affiliated
with or endorsed by DuckDuckGo. Use at your own risk and respect
[duck.ai](https://duck.ai)'s terms of service. The DuckDuckGo backend may
change at any time and break this library.
