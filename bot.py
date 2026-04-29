"""
Telegram bot wrapper around p2d-duck (DuckDuckGo AI Chat).

Features
--------
- Plain text messages -> chat with the currently selected model
- /image <prompt>     -> generate an image and send it back
- Photo with caption  -> vision (ask_with_image), caption is the question
- Reply to a photo with /edit <prompt> -> image edit (uses the image-generation model)
- /model [name]       -> show or switch model (any alias accepted by DuckChat)
- /effort [fast|reasoning]
- /history [on|off]   -> toggle multi-turn memory (default OFF, matches the library)
- /search [on|off]    -> toggle WebSearch tool for chat (model-dependent)
- /reset              -> clear chat history for this user
- /help, /start

History, effort, model, and web_search are tracked per Telegram chat id.

Run
---
    pip install p2d-duck "python-telegram-bot>=21,<22"
    python bot.py

The bot token is HARDCODED below. Replace the placeholder before running.
DO NOT commit a real token to a public repo.
"""

from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from duck_ai import DuckChat, image_generation, list_models
from duck_ai.models import ImagePart


# -------------------------------------------------------------------- config
# Replace this placeholder with your real BotFather token before running.
BOT_TOKEN = "PUT_YOUR_BOT_TOKEN_HERE"

# Default chat model (alias accepted by DuckChat; e.g. "gpt4", "claude", "llama").
DEFAULT_MODEL = "gpt4"

# Maximum size (bytes) of an image we will download from Telegram.
MAX_IMAGE_BYTES = 8 * 1024 * 1024

# Telegram message hard limit; we split long replies safely below this.
TG_MSG_LIMIT = 3500


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("p2d-duck-bot")


# -------------------------------------------------------------------- state
@dataclass
class Session:
    """Per-Telegram-chat state. One DuckChat instance per chat id."""

    model: str = DEFAULT_MODEL
    effort: Optional[str] = None
    history: bool = False
    web_search: bool = False
    duck: Optional[DuckChat] = field(default=None, repr=False)

    def client(self) -> DuckChat:
        # Recreate the underlying client whenever the user switches model
        # or effort, so we don't carry stale settings.
        if (
            self.duck is None
            or self.duck.model != self.model
            or self.duck.effort != self.effort
        ):
            if self.duck is not None:
                try:
                    self.duck.close()
                except Exception:
                    pass
            self.duck = DuckChat(
                model=self.model,
                effort=self.effort,
                history=self.history,
            )
        else:
            # Sync the toggle in case it changed without rebuilding.
            if self.history:
                self.duck.enable_history()
            else:
                self.duck.disable_history()
        return self.duck

    def close(self) -> None:
        if self.duck is not None:
            try:
                self.duck.close()
            except Exception:
                pass
            self.duck = None


_sessions: Dict[int, Session] = {}


def _session(update: Update) -> Session:
    cid = update.effective_chat.id
    sess = _sessions.get(cid)
    if sess is None:
        sess = Session()
        _sessions[cid] = sess
    return sess


# ---------------------------------------------------------------- utilities
async def _typing(update: Update) -> None:
    try:
        await update.effective_chat.send_action(ChatAction.TYPING)
    except Exception:
        pass


async def _send_long(update: Update, text: str) -> None:
    """Send a message, chunking if it exceeds Telegram's hard limit."""
    text = text or "(empty response)"
    if len(text) <= TG_MSG_LIMIT:
        await update.effective_message.reply_text(text)
        return
    # Try to split on paragraph boundaries first, then on lines, then hard.
    chunks = []
    remaining = text
    while len(remaining) > TG_MSG_LIMIT:
        cut = remaining.rfind("\n\n", 0, TG_MSG_LIMIT)
        if cut < 1000:
            cut = remaining.rfind("\n", 0, TG_MSG_LIMIT)
        if cut < 1000:
            cut = TG_MSG_LIMIT
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    chunks.append(remaining)
    for c in chunks:
        await update.effective_message.reply_text(c)


# ------------------------------------------------------------ command handlers
HELP_TEXT = (
    "p2d-duck telegram bot\n\n"
    "Just send a message to chat with the current model.\n\n"
    "Commands:\n"
    "  /model [name]        Show or switch model (gpt4, gpt5_mini, claude, llama, mistral, gpt-oss)\n"
    "  /effort [fast|reasoning|off]   Reasoning effort\n"
    "  /history [on|off]    Toggle multi-turn memory (default off)\n"
    "  /search  [on|off]    Toggle WebSearch tool for chat\n"
    "  /image <prompt>      Generate an image\n"
    "  /edit <prompt>       Reply to a photo with this to edit it\n"
    "  /reset               Clear this chat's history\n"
    "  /models              List known model ids\n"
    "  /status              Show current settings\n"
    "  /help                Show this help\n\n"
    "Send a photo with a caption to ask a question about that image."
)


async def cmd_start(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(HELP_TEXT)


async def cmd_help(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(HELP_TEXT)


async def cmd_status(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = _session(update)
    await update.effective_message.reply_text(
        "model:   {m}\n"
        "effort:  {e}\n"
        "history: {h}\n"
        "search:  {w}".format(
            m=s.model,
            e=s.effort or "(default)",
            h="on" if s.history else "off",
            w="on" if s.web_search else "off",
        )
    )


async def cmd_models(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text("\n".join(list_models()))


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = _session(update)
    if not ctx.args:
        await update.effective_message.reply_text(f"current model: {s.model}")
        return
    new_model = ctx.args[0].strip()
    s.model = new_model
    s.close()  # force rebuild on next call
    await update.effective_message.reply_text(f"model set to: {new_model}")


async def cmd_effort(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = _session(update)
    if not ctx.args:
        await update.effective_message.reply_text(
            f"current effort: {s.effort or '(default)'}"
        )
        return
    val = ctx.args[0].strip().lower()
    if val in ("off", "default", "none", ""):
        s.effort = None
    else:
        s.effort = val
    s.close()
    await update.effective_message.reply_text(
        f"effort set to: {s.effort or '(default)'}"
    )


async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = _session(update)
    if not ctx.args:
        await update.effective_message.reply_text(
            f"history is: {'on' if s.history else 'off'}"
        )
        return
    val = ctx.args[0].strip().lower()
    if val in ("on", "1", "true", "enable"):
        s.history = True
        if s.duck is not None:
            s.duck.enable_history()
        await update.effective_message.reply_text("history: on")
    elif val in ("off", "0", "false", "disable"):
        s.history = False
        if s.duck is not None:
            s.duck.disable_history()
        await update.effective_message.reply_text("history: off (cleared)")
    else:
        await update.effective_message.reply_text("usage: /history on|off")


async def cmd_search(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = _session(update)
    if not ctx.args:
        await update.effective_message.reply_text(
            f"web_search is: {'on' if s.web_search else 'off'}"
        )
        return
    val = ctx.args[0].strip().lower()
    if val in ("on", "1", "true", "enable"):
        s.web_search = True
        await update.effective_message.reply_text("web_search: on")
    elif val in ("off", "0", "false", "disable"):
        s.web_search = False
        await update.effective_message.reply_text("web_search: off")
    else:
        await update.effective_message.reply_text("usage: /search on|off")


async def cmd_reset(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = _session(update)
    if s.duck is not None:
        s.duck.reset()
    await update.effective_message.reply_text("(history cleared)")


# ----------------------------------------------------------- text -> chat
async def on_text(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    text = (msg.text or "").strip()
    if not text:
        return
    s = _session(update)
    await _typing(update)
    try:
        # Image generation always uses a fresh dedicated client so the chat
        # client's history/effort is not disturbed.
        duck = s.client()
        reply = duck.ask(text, web_search=s.web_search)
    except Exception as exc:
        log.exception("chat failed")
        await msg.reply_text(f"[error] {exc}")
        return
    await _send_long(update, reply)


# --------------------------------------------------------- /image -> generate
async def cmd_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = " ".join(ctx.args).strip()
    if not prompt:
        await update.effective_message.reply_text(
            "usage: /image <prompt>"
        )
        return
    await update.effective_chat.send_action(ChatAction.UPLOAD_PHOTO)
    img_client = DuckChat(model=image_generation)
    try:
        data = img_client.generate_image(prompt)
    except Exception as exc:
        log.exception("image generation failed")
        await update.effective_message.reply_text(f"[error] {exc}")
        return
    finally:
        img_client.close()
    bio = io.BytesIO(data)
    bio.name = "duck.jpg"
    await update.effective_message.reply_photo(photo=bio, caption=prompt[:1000])


# ----------------------------------- photo with caption -> vision (ask_with_image)
async def on_photo(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg.photo:
        return
    caption = (msg.caption or "Describe this image.").strip()
    s = _session(update)
    await _typing(update)
    try:
        # Pick the largest available size.
        photo = msg.photo[-1]
        if photo.file_size and photo.file_size > MAX_IMAGE_BYTES:
            await msg.reply_text("image too large (max 8MB)")
            return
        f = await photo.get_file()
        data = bytes(await f.download_as_bytearray())
        duck = s.client()
        reply = duck.ask_with_image(
            caption,
            data,
            mime_type="image/jpeg",
            web_search=s.web_search,
        )
    except Exception as exc:
        log.exception("vision failed")
        await msg.reply_text(f"[error] {exc}")
        return
    await _send_long(update, reply)


# ---------------------------- /edit (must be a reply to a photo) -> image edit
async def cmd_edit(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    src = msg.reply_to_message
    if src is None or not src.photo:
        await msg.reply_text(
            "Reply to a photo with /edit <prompt> to edit it."
        )
        return
    prompt = " ".join(ctx.args).strip()
    if not prompt:
        await msg.reply_text("usage: /edit <prompt> (as a reply to a photo)")
        return
    await update.effective_chat.send_action(ChatAction.UPLOAD_PHOTO)
    photo = src.photo[-1]
    if photo.file_size and photo.file_size > MAX_IMAGE_BYTES:
        await msg.reply_text("source image too large (max 8MB)")
        return
    f = await photo.get_file()
    data = bytes(await f.download_as_bytearray())
    img_client = DuckChat(model=image_generation)
    try:
        part = ImagePart.from_bytes(data, mime_type="image/jpeg")
        out = img_client.edit_image(prompt, part)
    except Exception as exc:
        log.exception("image edit failed")
        await msg.reply_text(f"[error] {exc}")
        return
    finally:
        img_client.close()
    bio = io.BytesIO(out)
    bio.name = "duck-edit.jpg"
    await msg.reply_photo(photo=bio, caption=prompt[:1000])


# -------------------------------------------------------------------- main
def main() -> None:
    if BOT_TOKEN == "PUT_YOUR_BOT_TOKEN_HERE":
        raise SystemExit(
            "Edit bot.py and replace BOT_TOKEN with your real "
            "BotFather token before running."
        )
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("effort", cmd_effort))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("image", cmd_image))
    app.add_handler(CommandHandler("edit", cmd_edit))

    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_text)
    )

    log.info("p2d-duck telegram bot starting")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
