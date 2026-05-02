"""
AXIS Telegram Bot — voice and text interface to AXIS from any phone.

Forwards every message to the AXIS REST API (POST /task) and returns
the response. Voice notes are transcribed locally with faster-whisper
(configurable model size) then cleaned up with Claude before sending.

Environment variables:
  TELEGRAM_TOKEN       Bot token from @BotFather
  TELEGRAM_USER_ID     Authorized user's Telegram numeric ID
  ANTHROPIC_API_KEY    For Claude post-processing of Arabic transcription
  AXIS_API_URL         Full URL of the AXIS /task endpoint
  WHISPER_MODEL        faster-whisper model size (default: small)
                       tiny | base | small | medium | large-v3
"""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

import anthropic
import requests as http_lib
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _require(env: str, fallback_file: Path = None) -> str:
    val = os.environ.get(env, "")
    if val:
        return val
    if fallback_file and fallback_file.exists():
        return fallback_file.read_text().strip()
    raise RuntimeError(f"{env} env var not set" +
                       (f" and {fallback_file} not found" if fallback_file else "") + ".")


AXIS_API_URL  = os.environ.get("AXIS_API_URL", "https://axis-api.onrender.com/task")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

api_key        = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_USER_ID = os.environ.get("TELEGRAM_USER_ID", "")

client         = anthropic.Anthropic(api_key=api_key)
AUTHORIZED_UID = int(TELEGRAM_USER_ID) if TELEGRAM_USER_ID else 0


def _authorized(update: Update) -> bool:
    return update.effective_user is not None and update.effective_user.id == AUTHORIZED_UID


# ---------------------------------------------------------------------------
# AXIS API call
# ---------------------------------------------------------------------------

def _call_axis_api(text: str) -> str:
    """Blocking: POST task to AXIS REST API, return the answer string."""
    resp = http_lib.post(AXIS_API_URL, json={"task": text}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("answer") or data.get("message") or "(no response)"


async def _ask_axis(text: str) -> str:
    """Async wrapper: runs the blocking API call in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _call_axis_api, text)


# ---------------------------------------------------------------------------
# Voice transcription — faster-whisper + Claude post-processing
# ---------------------------------------------------------------------------

_whisper_model = None


def _find_ffmpeg() -> str:
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]:
        if Path(candidate).exists():
            return candidate
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


_FFMPEG: str = _find_ffmpeg()


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print(f"[AXIS Telegram] Whisper {WHISPER_MODEL} model loaded")
    return _whisper_model


def _normalize_audio(input_path: str) -> str:
    """Normalize volume and convert to 16 kHz mono WAV. Returns temp WAV path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    subprocess.run(
        [_FFMPEG, "-y", "-i", input_path,
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
         "-ar", "16000", "-ac", "1", wav_path],
        capture_output=True,
        check=True,
    )
    return wav_path


def _fix_transcription(raw: str) -> str:
    """Claude post-processing: fix Arabic transcription errors and names."""
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                "Fix any Arabic transcription errors in this text. "
                "Preserve names, technical terms, and numbers exactly as intended. "
                "Return ONLY the corrected text, nothing else.\n\n"
                f"{raw}"
            ),
        }],
    )
    fixed = resp.content[0].text.strip() if resp.content else raw
    return fixed or raw


def _transcribe(input_path: str) -> str:
    """Blocking: normalize → Whisper → Claude correction. Returns cleaned text."""
    wav_path = None
    try:
        try:
            wav_path = _normalize_audio(input_path)
            source   = wav_path
        except Exception:
            source = input_path  # ffmpeg failed — use original

        model = _get_whisper_model()
        segments, info = model.transcribe(
            source, language="ar", beam_size=5, vad_filter=True,
        )
        raw = " ".join(seg.text for seg in segments).strip()
        print(f"[AXIS Telegram] Whisper raw: {len(raw)} chars (lang={info.language})")
        if not raw:
            return ""
        fixed = _fix_transcription(raw)
        print(f"[AXIS Telegram] After correction: {len(fixed)} chars")
        return fixed
    finally:
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    await update.message.reply_text(
        "AXIS is ready.\n\nSend a text message or a voice note — I'll transcribe and respond."
    )


async def cmd_whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else "unknown"
    await update.message.reply_text(f"Your Telegram user ID: {uid}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    text = (update.message.text or "").strip()
    if not text:
        return

    thinking = await update.message.reply_text("⏳")
    try:
        response = await _ask_axis(text)
        await thinking.edit_text(response)
    except Exception as exc:
        await thinking.edit_text(f"⚠️ Error: {exc}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return

    status   = await update.message.reply_text("🎙 Transcribing...")
    ogg_path = None
    try:
        voice   = update.message.voice
        tg_file = await context.bot.get_file(voice.file_id)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            ogg_path = tmp.name
        await tg_file.download_to_drive(ogg_path)

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, _transcribe, ogg_path)

        if not text:
            await status.edit_text("⚠️ Could not transcribe audio. Please try again.")
            return

        await status.edit_text(f'🎙 "{text}"\n\n⏳ Asking AXIS...')
        response = await _ask_axis(text)
        await status.edit_text(f'🎙 "{text}"\n\n{response}')

    except Exception as exc:
        await status.edit_text(f"⚠️ Error: {exc}")
    finally:
        if ogg_path:
            Path(ogg_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"[AXIS Telegram] Starting — authorized UID: {AUTHORIZED_UID}")
    print(f"[AXIS Telegram] AXIS API: {AXIS_API_URL}")
    print(f"[AXIS Telegram] Whisper model: {WHISPER_MODEL}")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("whoami", cmd_whoami))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("[AXIS Telegram] Polling for messages...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
