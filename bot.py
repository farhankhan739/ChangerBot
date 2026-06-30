"""
bot.py — Bulk Media Caption Replacer Bot
=========================================

A production-ready Telegram bot that:
  - Accepts media sent in bulk (photo/video/document/audio/animation/voice)
  - Replaces configured text inside captions while preserving formatting
    (bold, italic, code, spoiler, links, etc.)
  - Reposts media to a storage channel in strict input order (FIFO queue)
  - Handles Telegram FloodWait / network errors with automatic retry
  - Exposes /pause /resume /stop /status commands
  - Prints structured logs and running statistics

Stack: python-telegram-bot >= 22.0 (async), Python 3.11+
Designed to run on Railway using long polling.

Files expected alongside this script:
  config.json        -> replacement rules + non-secret settings
  Environment vars    -> BOT_TOKEN, STORAGE_CHANNEL_ID, ADMIN_IDS (Railway)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from telegram import Update, Message, MessageEntity
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut, NetworkError, BadRequest, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("caption_bot")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class ReplacementRule:
    old: str
    new: str


@dataclass
class Config:
    bot_token: str
    storage_channel_id: int
    admin_ids: set[int]
    rules: list[ReplacementRule]
    max_retries: int = 5
    retry_base_delay: float = 2.0

    @classmethod
    def load(cls, config_path: str = "config.json") -> "Config":
        # Secrets come from environment variables (Railway-friendly).
        bot_token = os.environ.get("BOT_TOKEN", "").strip()
        storage_channel_raw = os.environ.get("STORAGE_CHANNEL_ID", "").strip()
        admin_ids_raw = os.environ.get("ADMIN_IDS", "").strip()

        if not bot_token:
            raise RuntimeError("BOT_TOKEN environment variable is missing.")
        if not storage_channel_raw:
            raise RuntimeError("STORAGE_CHANNEL_ID environment variable is missing.")

        try:
            storage_channel_id = int(storage_channel_raw)
        except ValueError:
            raise RuntimeError("STORAGE_CHANNEL_ID must be an integer (e.g. -1004332279939).")

        admin_ids = set()
        if admin_ids_raw:
            for part in admin_ids_raw.split(","):
                part = part.strip()
                if part:
                    admin_ids.add(int(part))

        # Non-secret, easily editable settings live in config.json
        rules: list[ReplacementRule] = []
        max_retries = 5
        retry_base_delay = 2.0

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("replacement_rules", []):
                rules.append(ReplacementRule(old=r["old"], new=r["new"]))
            max_retries = data.get("max_retries", max_retries)
            retry_base_delay = data.get("retry_base_delay", retry_base_delay)
        else:
            log.warning("config.json not found — no replacement rules loaded.")

        if not rules:
            log.warning("No replacement rules configured. Captions will pass through unchanged.")

        return cls(
            bot_token=bot_token,
            storage_channel_id=storage_channel_id,
            admin_ids=admin_ids,
            rules=rules,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
        )


# --------------------------------------------------------------------------- #
# Caption + entity-preserving text replacement
# --------------------------------------------------------------------------- #
#
# Telegram entity offsets are measured in UTF-16 code units, not Python
# string indices. We operate on UTF-16 buffers so bold/italic/links/etc.
# still line up correctly after old text is replaced with new text of a
# different length.

def _to_utf16(text: str) -> bytes:
    return text.encode("utf-16-le")


def _utf16_len(text: str) -> int:
    return len(_to_utf16(text)) // 2


def replace_in_caption(
    caption: str,
    entities: Optional[list[MessageEntity]],
    rules: list[ReplacementRule],
) -> tuple[str, Optional[list[MessageEntity]], bool]:
    """
    Replace all configured old->new strings inside `caption`, shifting
    entity offsets/lengths so formatting stays aligned.

    Returns: (new_caption, new_entities, was_modified)
    """
    if not caption:
        return caption, entities, False

    units = list(_to_utf16(caption))  # raw byte list, 2 bytes per UTF-16 unit
    # Work at the unit (character-in-UTF16) level using a list of 2-byte chunks
    u16 = [caption.encode("utf-16-le")[i:i + 2] for i in range(0, len(caption.encode("utf-16-le")), 2)]

    modified = False
    # Track offset/length as plain mutable dicts since MessageEntity is immutable
    working_entities = (
        [{"offset": e.offset, "length": e.length, "src": e} for e in entities]
        if entities else []
    )

    for rule in rules:
        if not rule.old:
            continue
        old_u16_str = rule.old
        new_u16_str = rule.new
        old_len = _utf16_len(old_u16_str)
        new_len = _utf16_len(new_u16_str)

        search_pos = 0
        while True:
            # Reconstruct current text from u16 chunks to search
            current_text = b"".join(u16).decode("utf-16-le")
            idx = current_text.find(old_u16_str, search_pos)
            if idx == -1:
                break

            modified = True
            # Replace in the unit list
            new_chunk = [new_u16_str.encode("utf-16-le")[i:i + 2]
                         for i in range(0, len(new_u16_str.encode("utf-16-le")), 2)]
            u16[idx:idx + old_len] = new_chunk

            delta = new_len - old_len

            # Shift entity offsets/lengths that come after or contain the replacement point
            for ent in working_entities:
                if ent["offset"] >= idx + old_len:
                    ent["offset"] += delta
                elif ent["offset"] <= idx < ent["offset"] + ent["length"]:
                    # Replacement happens inside this entity's span — extend it
                    ent["length"] += delta

            search_pos = idx + new_len

    new_caption = b"".join(u16).decode("utf-16-le")

    new_entities = None
    if working_entities:
        new_entities = []
        for ent in working_entities:
            src = ent["src"]
            # Rebuild a fresh MessageEntity with updated offset/length,
            # carrying over any type-specific fields (url, user, language, etc.)
            kwargs = src.to_dict()
            kwargs["offset"] = ent["offset"]
            kwargs["length"] = ent["length"]
            new_entities.append(MessageEntity(**kwargs))

    return new_caption, new_entities, modified


# --------------------------------------------------------------------------- #
# Queue item
# --------------------------------------------------------------------------- #

@dataclass
class QueueItem:
    message: Message
    seq: int  # ensures ordering is explicit even if queue is ever inspected


@dataclass
class Stats:
    total_received: int = 0
    total_processed: int = 0
    modified: int = 0
    no_caption: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.time)

    def summary(self, queue_size: int) -> str:
        return (
            f"📊 *Stats*\n"
            f"Received: {self.total_received}\n"
            f"Processed: {self.total_processed}\n"
            f"Modified captions: {self.modified}\n"
            f"No caption (forwarded as-is): {self.no_caption}\n"
            f"Failed: {self.failed}\n"
            f"Pending in queue: {queue_size}"
        )


# --------------------------------------------------------------------------- #
# Bot state / worker
# --------------------------------------------------------------------------- #

class CaptionBot:
    def __init__(self, config: Config):
        self.config = config
        self.queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self.stats = Stats()
        self._seq_counter = 0
        self.paused = asyncio.Event()
        self.paused.set()  # set == "not paused" (worker runs)
        self.stop_requested = False
        self.worker_task: Optional[asyncio.Task] = None

    # ---- queueing ------------------------------------------------------ #

    async def enqueue(self, message: Message) -> int:
        self._seq_counter += 1
        item = QueueItem(message=message, seq=self._seq_counter)
        await self.queue.put(item)
        self.stats.total_received += 1
        log.info(f"Queued message {message.message_id} (queue position {self.queue.qsize()})")
        return self.queue.qsize()

    # ---- worker loop ----------------------------------------------------#

    async def start_worker(self, app: Application):
        self.worker_task = asyncio.create_task(self._worker(app))

    async def _worker(self, app: Application):
        log.info("Worker started — waiting for media...")
        while True:
            if self.stop_requested:
                log.info("Stop requested — worker exiting.")
                return

            item = await self.queue.get()
            # Honor pause: block here until resumed
            await self.paused.wait()

            if self.stop_requested:
                # Drop remaining item gracefully on stop
                self.queue.task_done()
                return

            try:
                await self._process_item(app, item)
            except Exception as e:  # noqa: BLE001 — top-level safety net
                self.stats.failed += 1
                log.error(f"Unhandled error processing message {item.message.message_id}: {e}")
            finally:
                self.queue.task_done()
                self.stats.total_processed += 1
                log.info(
                    f"Progress: {self.stats.total_processed}/{self.stats.total_received} processed"
                )

    # ---- per-message processing ----------------------------------------#

    async def _process_item(self, app: Application, item: QueueItem):
        msg = item.message
        log.info(f"Processing message {msg.message_id} (seq {item.seq})")

        media_kind, file_id = self._extract_media(msg)
        if media_kind is None:
            log.warning(f"Message {msg.message_id} has no supported media — skipping.")
            return

        caption = msg.caption or ""
        entities = msg.caption_entities or None

        if caption:
            log.info(f"Caption detected on message {msg.message_id}.")
            new_caption, new_entities, was_modified = replace_in_caption(
                caption, entities, self.config.rules
            )
            if was_modified:
                self.stats.modified += 1
                log.info(f"Replacement completed for message {msg.message_id}.")
            else:
                log.info(f"Caption present but no match found for message {msg.message_id}.")
        else:
            new_caption, new_entities = None, None
            self.stats.no_caption += 1
            log.info(f"No caption on message {msg.message_id} — forwarding unchanged.")

        await self._send_with_retry(
            app=app,
            media_kind=media_kind,
            file_id=file_id,
            caption=new_caption,
            caption_entities=new_entities,
            original_msg=msg,
        )

    @staticmethod
    def _extract_media(msg: Message) -> tuple[Optional[str], Optional[str]]:
        if msg.video:
            return "video", msg.video.file_id
        if msg.document:
            return "document", msg.document.file_id
        if msg.photo:
            return "photo", msg.photo[-1].file_id  # highest resolution
        if msg.audio:
            return "audio", msg.audio.file_id
        if msg.animation:
            return "animation", msg.animation.file_id
        if msg.voice:
            return "voice", msg.voice.file_id
        return None, None

    async def _send_with_retry(
        self,
        app: Application,
        media_kind: str,
        file_id: str,
        caption: Optional[str],
        caption_entities,
        original_msg: Message,
    ):
        attempt = 0
        while True:
            attempt += 1
            try:
                send_fn = {
                    "video": app.bot.send_video,
                    "document": app.bot.send_document,
                    "photo": app.bot.send_photo,
                    "audio": app.bot.send_audio,
                    "animation": app.bot.send_animation,
                    "voice": app.bot.send_voice,
                }[media_kind]

                kwargs = dict(
                    chat_id=self.config.storage_channel_id,
                    caption=caption,
                    caption_entities=caption_entities,
                )
                # Map the correct file kwarg name per media type
                file_kwarg = {
                    "video": "video",
                    "document": "document",
                    "photo": "photo",
                    "audio": "audio",
                    "animation": "animation",
                    "voice": "voice",
                }[media_kind]
                kwargs[file_kwarg] = file_id

                await send_fn(**kwargs)
                log.info(f"Upload successful for message {original_msg.message_id} -> storage channel.")
                return

            except RetryAfter as e:
                wait_s = float(e.retry_after) + 1
                log.warning(f"FloodWait detected — sleeping {wait_s:.1f}s before retry "
                            f"(message {original_msg.message_id}).")
                await asyncio.sleep(wait_s)
                log.info(f"Resuming after FloodWait for message {original_msg.message_id}.")
                continue  # never skip — retry same item

            except BadRequest as e:
                # Non-recoverable for this item (e.g. invalid file reference,
                # wrong/unauthorized chat ID, bot not admin in target chat).
                # NOTE: BadRequest must be caught BEFORE TimedOut/NetworkError
                # below, since BadRequest is a subclass of NetworkError in
                # python-telegram-bot — catching the parent first would
                # swallow BadRequest and trigger pointless retry/backoff.
                self.stats.failed += 1
                log.error(
                    f"BadRequest on message {original_msg.message_id}: {e}. "
                    f"Check STORAGE_CHANNEL_ID is correct and the bot is an admin "
                    f"with post permission in that channel."
                )
                return

            except (TimedOut, NetworkError) as e:
                if attempt > self.config.max_retries:
                    self.stats.failed += 1
                    log.error(f"Giving up on message {original_msg.message_id} after "
                              f"{attempt} attempts: {e}")
                    return
                delay = self.config.retry_base_delay * (2 ** (attempt - 1))
                log.warning(f"Network error on message {original_msg.message_id} "
                            f"(attempt {attempt}): {e}. Retrying in {delay:.1f}s.")
                await asyncio.sleep(delay)
                continue

            except TelegramError as e:
                if attempt > self.config.max_retries:
                    self.stats.failed += 1
                    log.error(f"Giving up on message {original_msg.message_id} after "
                              f"{attempt} attempts: {e}")
                    return
                delay = self.config.retry_base_delay * (2 ** (attempt - 1))
                log.warning(f"Telegram error on message {original_msg.message_id} "
                            f"(attempt {attempt}): {e}. Retrying in {delay:.1f}s.")
                await asyncio.sleep(delay)
                continue


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #

MEDIA_FILTER = (
    filters.VIDEO
    | filters.Document.ALL
    | filters.PHOTO
    | filters.AUDIO
    | filters.ANIMATION
    | filters.VOICE
)


def is_admin(config: Config, user_id: Optional[int]) -> bool:
    # If no admins configured, allow everyone (useful for personal bots).
    if not config.admin_ids:
        return True
    return user_id in config.admin_ids


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_state: CaptionBot = context.application.bot_data["caption_bot"]
    config: Config = context.application.bot_data["config"]

    if not is_admin(config, update.effective_user.id if update.effective_user else None):
        return

    msg = update.effective_message
    log.info(f"File received: message {msg.message_id} from chat {msg.chat_id}")
    position = await bot_state.enqueue(msg)
    await msg.reply_text(f"Queued (#{position} in line) ✅")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_state: CaptionBot = context.application.bot_data["caption_bot"]
    state = "PAUSED" if not bot_state.paused.is_set() else "RUNNING"
    text = f"State: *{state}*\n\n" + bot_state.stats.summary(bot_state.queue.qsize())
    await update.effective_message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_state: CaptionBot = context.application.bot_data["caption_bot"]
    config: Config = context.application.bot_data["config"]
    if not is_admin(config, update.effective_user.id if update.effective_user else None):
        return
    bot_state.paused.clear()
    log.info("Processing paused by admin command.")
    await update.effective_message.reply_text("⏸ Processing paused.")


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_state: CaptionBot = context.application.bot_data["caption_bot"]
    config: Config = context.application.bot_data["config"]
    if not is_admin(config, update.effective_user.id if update.effective_user else None):
        return
    bot_state.paused.set()
    log.info("Processing resumed by admin command.")
    await update.effective_message.reply_text("▶️ Processing resumed.")


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_state: CaptionBot = context.application.bot_data["caption_bot"]
    config: Config = context.application.bot_data["config"]
    if not is_admin(config, update.effective_user.id if update.effective_user else None):
        return
    bot_state.stop_requested = True
    bot_state.paused.set()  # make sure worker isn't blocked on pause, so it can see stop_requested
    log.info("Stop requested by admin command. Worker will exit after current item.")
    await update.effective_message.reply_text(
        "🛑 Stopping safely after the current file finishes. Remaining queued items will not be sent."
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "👋 Bulk Caption Replacer Bot is online.\n\n"
        "Send media (photo/video/document/audio/animation/voice) and I will queue, "
        "process captions, and repost to the storage channel in order.\n\n"
        "Commands: /status /pause /resume /stop"
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    config = Config.load()
    bot_state = CaptionBot(config)

    app = ApplicationBuilder().token(config.bot_token).build()
    app.bot_data["config"] = config
    app.bot_data["caption_bot"] = bot_state

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(MessageHandler(MEDIA_FILTER, handle_media))

    async def on_startup(app: Application):
        await bot_state.start_worker(app)
        log.info("Bot startup complete. Long polling...")

    app.post_init = on_startup

    log.info("Starting bot (long polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
