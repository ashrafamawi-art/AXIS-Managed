"""
AXIS Telegram Bot — voice and text interface to AXIS from any phone.

Run with:
  ANTHROPIC_API_KEY=$(cat ~/.anthropic_key) python3 ~/AXIS-Managed/telegram_bot.py

Security:
  - Bot token loaded from ~/.telegram_token (never hardcoded)
  - Only messages from the authorized Telegram user ID (~/.telegram_user_id) are processed
  - Message content is never logged
"""

import asyncio
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import anthropic
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Config — mirrors session files used by chat.py so both share the same session
# ---------------------------------------------------------------------------

HERE            = Path(__file__).parent
AGENT_ID_FILE   = HERE / ".axis_agent_id"
ENV_ID_FILE     = HERE / ".axis_env_id"
SESSION_ID_FILE = HERE / ".axis_session_id"
BETA            = "managed-agents-2026-04-01"

# Secrets — env vars on cloud, local files for local dev.
def _read_secret(env_var: str, fallback_file: Path) -> str:
    val = os.environ.get(env_var, "")
    if val:
        return val
    if fallback_file.exists():
        return fallback_file.read_text().strip()
    raise RuntimeError(
        f"{env_var} env var not set and {fallback_file} not found."
    )

api_key = _read_secret("ANTHROPIC_API_KEY", Path.home() / ".anthropic_key")
client  = anthropic.Anthropic(api_key=api_key)

# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------

def _load_authorized_uid() -> int:
    return int(_read_secret("TELEGRAM_USER_ID", Path.home() / ".telegram_user_id"))


AUTHORIZED_UID: int = _load_authorized_uid()


def _authorized(update: Update) -> bool:
    return update.effective_user is not None and update.effective_user.id == AUTHORIZED_UID


# ---------------------------------------------------------------------------
# AXIS session management (same files as chat.py — shared session)
# ---------------------------------------------------------------------------

def _load_agent_id() -> str:
    return _read_secret("AXIS_AGENT_ID", AGENT_ID_FILE)


def _get_or_create_env() -> str:
    if ENV_ID_FILE.exists():
        env_id = ENV_ID_FILE.read_text().strip()
        try:
            env = client.beta.environments.retrieve(env_id, betas=[BETA])
            if getattr(env, "state", None) == "active":
                return env_id
        except Exception:
            pass
    env = client.beta.environments.create(name="axis-env", betas=[BETA])
    ENV_ID_FILE.write_text(env.id)
    return env.id


def _get_or_create_session(agent_id: str, env_id: str) -> str:
    if SESSION_ID_FILE.exists():
        sid = SESSION_ID_FILE.read_text().strip()
        try:
            s = client.beta.sessions.retrieve(sid, betas=[BETA])
            if getattr(s, "status", "") not in ("terminated", "expired", "error", "archived"):
                return s.id
        except Exception:
            pass
    s = client.beta.sessions.create(
        agent={"type": "agent", "id": agent_id},
        environment_id=env_id,
        betas=[BETA],
    )
    SESSION_ID_FILE.write_text(s.id)
    return s.id


def _query_axis_sync(session_id: str, text: str) -> str:
    """Blocking: send text to AXIS session, collect and return the full response."""
    chunks: list[str] = []

    with client.beta.sessions.events.stream(session_id=session_id, betas=[BETA]) as stream:
        def _send():
            client.beta.sessions.events.send(
                session_id=session_id,
                events=[{
                    "type":    "user.message",
                    "content": [{"type": "text", "text": text}],
                }],
                betas=[BETA],
            )

        sender = threading.Thread(target=_send, daemon=True)
        sender.start()

        for event in stream:
            etype = getattr(event, "type", None)
            if etype == "agent.message":
                for block in getattr(event, "content", []):
                    if getattr(block, "type", None) == "text":
                        chunks.append(block.text)
            elif etype == "session.status_idle":
                break
            elif etype == "session.status_terminated":
                SESSION_ID_FILE.unlink(missing_ok=True)
                break

        sender.join()

    return "".join(chunks).strip() or "(no response)"


async def _ask_axis(text: str) -> str:
    """Async wrapper: runs the blocking AXIS query in a thread pool."""
    agent_id   = _load_agent_id()
    env_id     = _get_or_create_env()
    session_id = _get_or_create_session(agent_id, env_id)
    loop       = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _query_axis_sync, session_id, text)


# ---------------------------------------------------------------------------
# Voice transcription — faster-whisper large-v3 + Claude post-processing
# ---------------------------------------------------------------------------

_whisper_model = None  # loaded once on first voice message

# ffmpeg — prefer system install, fall back to imageio-ffmpeg bundled binary.
def _find_ffmpeg() -> str:
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if Path(candidate).exists():
            return candidate
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # last resort: rely on PATH

_FFMPEG: str = _find_ffmpeg()


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # large-v3: best Arabic dialect + name recognition; int8 keeps CPU memory reasonable.
        _whisper_model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        print("[AXIS Telegram] Whisper large-v3 model loaded")
    return _whisper_model


def _normalize_audio(input_path: str) -> str:
    """
    Use ffmpeg to normalize volume (loudnorm) and convert to 16kHz mono WAV.
    Returns path to the normalized WAV file (caller must delete it).
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    subprocess.run(
        [
            _FFMPEG, "-y", "-i", input_path,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", "16000", "-ac", "1",
            wav_path,
        ],
        capture_output=True,
        check=True,
    )
    return wav_path


def _fix_transcription_sync(raw: str) -> str:
    """
    Send raw Whisper output to Claude to fix Arabic transcription errors,
    correct names, and preserve technical terms.
    """
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


def _transcribe_sync(input_path: str) -> str:
    """
    Blocking: normalize audio → transcribe with Whisper large-v3 → Claude correction.
    Returns final cleaned Arabic text.
    """
    wav_path = None
    try:
        try:
            wav_path = _normalize_audio(input_path)
            transcribe_from = wav_path
        except Exception:
            transcribe_from = input_path  # ffmpeg failed — use original file

        model = _get_whisper_model()
        segments, info = model.transcribe(
            transcribe_from,
            language="ar",
            beam_size=5,
            vad_filter=True,
        )
        raw = " ".join(seg.text for seg in segments).strip()
        print(f"[AXIS Telegram] whisper raw: {len(raw)} chars (lang={info.language})")

        if not raw:
            return ""

        fixed = _fix_transcription_sync(raw)
        print(f"[AXIS Telegram] after correction: {len(fixed)} chars")
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
    """Helper: replies with the sender's Telegram user ID."""
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

    status = await update.message.reply_text("🎙 Transcribing...")

    ogg_path = None
    try:
        # Download the OGG voice file Telegram sends
        voice   = update.message.voice
        tg_file = await context.bot.get_file(voice.file_id)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            ogg_path = tmp.name
        await tg_file.download_to_drive(ogg_path)

        # Transcribe in thread pool (CPU-bound)
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, _transcribe_sync, ogg_path)

        if not text:
            await status.edit_text("⚠️ Could not transcribe audio. Please try again.")
            return

        # Show what was heard, then query AXIS
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
    token = _read_secret("TELEGRAM_TOKEN", Path.home() / ".telegram_token")
    print(f"[AXIS Telegram] Starting bot — authorized UID: {AUTHORIZED_UID}")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("whoami", cmd_whoami))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("[AXIS Telegram] Polling for messages...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
