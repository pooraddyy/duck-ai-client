from __future__ import annotations

import argparse
import sys

from . import DuckChat, __version__, gpt4, image_generation, list_models

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="p2d-duck",
        description="Free DuckDuckGo AI Chat (duck.ai) client. No API key required.",
    )
    p.add_argument(
        "-v", "--version", action="version", version=f"p2d-duck {__version__}"
    )
    p.add_argument(
        "-m",
        "--model",
        default=gpt4,
        help="Model name or alias (e.g. gpt4, gpt5_mini, claude, llama, image)",
    )
    p.add_argument(
        "-e",
        "--effort",
        default=None,
        help="Reasoning effort: 'fast' or 'reasoning' (reasoning models only)",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (collect full response first)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Maximum retry attempts per request (default 4)",
    )
    sub = p.add_subparsers(dest="cmd")

    chat = sub.add_parser("chat", help="Interactive chat (default)")
    chat.add_argument(
        "prompt", nargs="*", help="One-shot prompt; omit for REPL"
    )
    chat.add_argument(
        "--image", help="Attach an image (path, data URL, or http(s) URL)"
    )

    image = sub.add_parser("image", help="Generate an image")
    image.add_argument("prompt", nargs="+", help="Image prompt")
    image.add_argument(
        "-o", "--output", default="duck.jpg", help="Output file path"
    )

    sub.add_parser("models", help="List known model ids")

    return p

def _img(spec: str):
    from .models import ImagePart

    if spec.startswith("data:"):
        return ImagePart(image=spec)
    if spec.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(spec) as r:
            data = r.read()
            mt = r.headers.get_content_type() or "image/png"
            return ImagePart.from_bytes(data, mime_type=mt)
    return ImagePart.from_path(spec)

def _make_client(args) -> DuckChat:
    return DuckChat(
        model=args.model, effort=args.effort, max_retries=args.retries
    )

def _run_chat(args: argparse.Namespace) -> int:
    duck = _make_client(args)
    if args.prompt:
        prompt = " ".join(args.prompt)
        try:
            if args.image:
                if args.no_stream:
                    print(duck.ask_with_image(prompt, args.image))
                else:
                    for chunk in duck.stream([prompt, _img(args.image)]):
                        print(chunk, end="", flush=True)
                    print()
            else:
                if args.no_stream:
                    print(duck.ask(prompt))
                else:
                    for chunk in duck.stream(prompt):
                        print(chunk, end="", flush=True)
                    print()
        finally:
            duck.close()
        return 0
    print(f"p2d-duck {__version__} ({args.model}) - type /reset, /quit")
    try:
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("/q", "/quit", "/exit"):
                break
            if line in ("/r", "/reset"):
                duck.reset()
                print("(history cleared)")
                continue
            try:
                print("ai> ", end="", flush=True)
                if args.no_stream:
                    print(duck.ask(line))
                else:
                    for chunk in duck.stream(line):
                        print(chunk, end="", flush=True)
                    print()
            except Exception as e:
                print(f"\n[error] {e}")
    finally:
        duck.close()
    return 0

def _run_image(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt)
    duck = DuckChat(model=image_generation, max_retries=args.retries)
    try:
        data = duck.generate_image(prompt, save_to=args.output)
        print(f"saved {len(data)} bytes -> {args.output}")
    finally:
        duck.close()
    return 0

def _run_models(_args: argparse.Namespace) -> int:
    for m in list_models():
        print(m)
    return 0

def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "image":
        return _run_image(args)
    if args.cmd == "models":
        return _run_models(args)
    return _run_chat(args)

if __name__ == "__main__":
    sys.exit(main())
