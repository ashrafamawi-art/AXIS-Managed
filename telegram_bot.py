"""
AXIS Telegram Bot — voice and text interface to AXIS from any phone.

Forwards every message to the AXIS REST API (POST /task) and returns
the response. Voice notes are transcribed with Deepgram Nova-3 (preferred)
or faster-whisper (fallback when DEEPGRAM_API_KEY is not set or Deepgram fails).
Claude post-processing is optional and off by default.

Timeouts (hard limits — user always sees a response):
  VOICE_HANDLER_TIMEOUT  60 s — total voice handler budget
  DEEPGRAM_TIMEOUT       20 s — Deepgram API call
  WHISPER_TIMEOUT        30 s — local Whisper transcription (inc. model load)
  AXIS_TIMEOUT           20 s — AXIS API call for voice

Environment variables:
  TELEGRAM_TOKEN              Bot token from @BotFather
  TELEGRAM_USER_ID            Authorized user's Telegram numeric ID
  ANTHROPIC_API_KEY           For optional Claude post-processing of transcription
  AXIS_API_URL                Full URL of the AXIS /task endpoint
                              Default: https://axis-api.onrender.com/task
  DEEPGRAM_API_KEY            Deepgram API key — enables Nova-3 Arabic transcription.
                              MUST be set in Render env vars; without it the bot falls
                              back to local Whisper which takes 40+ s to load cold.
  WHISPER_MODEL               faster-whisper model size (default: small)
                              tiny | base | small | medium | large-v3
  VOICE_POSTPROCESS_WITH_CLAUDE  Set to "true" to enable Claude Haiku post-correction
                              Default: false
"""

import asyncio
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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

AXIS_API_URL   = os.environ.get("AXIS_API_URL", "https://axis-api.onrender.com/task")
AXIS_BASE_URL  = AXIS_API_URL.replace("/task", "")
AXIS_HEALTH_URL = AXIS_API_URL.replace("/task", "/health")
WHISPER_MODEL  = os.environ.get("WHISPER_MODEL", "small")

api_key           = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_USER_ID  = os.environ.get("TELEGRAM_USER_ID", "")
DEEPGRAM_API_KEY  = os.environ.get("DEEPGRAM_API_KEY", "").strip()
VOICE_POSTPROCESS_WITH_CLAUDE = (
    os.environ.get("VOICE_POSTPROCESS_WITH_CLAUDE", "false").lower() == "true"
)

# Hard timeouts — never leave the user stuck
_VOICE_HANDLER_TIMEOUT: int = 60   # total budget for handle_voice
_DEEPGRAM_TIMEOUT:      int = 20   # Deepgram API call
_WHISPER_TIMEOUT:       int = 30   # Whisper model load + transcription
_AXIS_TIMEOUT:          int = 60   # AXIS API call from voice handler

client         = anthropic.Anthropic(api_key=api_key)
AUTHORIZED_UID = int(TELEGRAM_USER_ID) if TELEGRAM_USER_ID else 0

_FAILURE_MSG = "Voice transcription failed. Please resend a shorter voice message."


def _authorized(update: Update) -> bool:
    return update.effective_user is not None and update.effective_user.id == AUTHORIZED_UID


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# AXIS API call — no fallback, no local Claude response for text
# ---------------------------------------------------------------------------

def _call_axis_api(text: str, request_id: str = "") -> dict:
    """
    Blocking: POST task to AXIS REST API.
    request_id is derived from Telegram's update_id — stable across retries,
    so the server-side dedup cache can detect and skip duplicate deliveries.
    Returns the full response dict so callers can surface execution artifacts.
    Raises on HTTP error — caller shows the error directly.
    """
    payload: dict = {"task": text}
    if request_id:
        payload["request_id"] = request_id

    print(f"[{_ts()}] [AXIS Telegram] → POST {AXIS_API_URL}")
    print(f"[{_ts()}] [AXIS Telegram]   request_id={request_id or '(none)'}  "
          f"task={text[:100]!r}")

    resp = http_lib.post(AXIS_API_URL, json=payload, timeout=_AXIS_TIMEOUT)

    print(f"[{_ts()}] [AXIS Telegram] ← HTTP {resp.status_code} "
          f"({resp.elapsed.total_seconds():.1f}s)")
    resp.raise_for_status()

    data = resp.json()
    _log_response(data)
    return data


def _log_response(data: dict) -> None:
    intent  = data.get("routing", {}).get("intent", "?")
    agents  = data.get("routing", {}).get("agents", [])
    ms      = data.get("execution_ms", "?")
    risk    = data.get("security", {}).get("risk", "?")
    answer  = data.get("answer", "")
    arts    = data.get("artifacts", {})
    print(f"[{_ts()}] [AXIS Telegram]   intent={intent}  agents={agents}  "
          f"risk={risk}  execution_ms={ms}")
    print(f"[{_ts()}] [AXIS Telegram]   answer: {answer[:100]!r}...")
    if arts:
        print(f"[{_ts()}] [AXIS Telegram]   artifacts: {list(arts.keys())}")


def _format_response(data: dict) -> str:
    """
    Build the Telegram message from the AXIS API response.
    Shows the answer AND a brief execution summary from artifacts.
    """
    answer = data.get("answer") or data.get("message") or "(no response)"

    arts = data.get("artifacts", {})
    lines: list[str] = []

    # Calendar events created
    for item in arts.get("calendar", []):
        if isinstance(item, str):
            lines.append(f"📅 {item}")

    # Tasks saved
    for item in arts.get("tasks", []):
        if isinstance(item, str) and "PENDING CALENDAR" not in item:
            lines.append(f"✅ {item}")

    # Pending calendar items (calendar creds not set on Render)
    pending = [
        item for item in arts.get("tasks", [])
        if isinstance(item, str) and "PENDING CALENDAR" in item
    ]
    if pending:
        lines.append("⚠️ Calendar credentials not configured on Render — event saved as pending task.")

    # Notifications sent
    for item in arts.get("notifications", []):
        if isinstance(item, str):
            lines.append(f"🔔 {item}")

    suffix = ("\n\n" + "\n".join(lines)) if lines else ""
    return answer + suffix


async def _ask_axis(text: str, request_id: str = "") -> str:
    """Async wrapper: runs the blocking API call in a thread pool."""
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, _call_axis_api, text, request_id)
    return _format_response(data)


# ---------------------------------------------------------------------------
# Voice transcription — Deepgram Nova-3 (primary) + faster-whisper (fallback)
#
# IMPORTANT: WhisperModel("small") takes 40+ seconds to load cold on Render.
# Set DEEPGRAM_API_KEY in Render env vars to use Deepgram and avoid that delay.
# ---------------------------------------------------------------------------

_DEEPGRAM_TIMEOUT = 20   # seconds — asyncio.wait_for + SDK-level timeout
_WHISPER_TIMEOUT  = 30   # seconds — asyncio.wait_for

_whisper_model = None


def _vlog(event: str) -> None:
    """Structured single-line voice log. Never prints secret values."""
    print(f"[{_ts()}] [voice] {event}", flush=True)


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


def _ffmpeg_available() -> bool:
    try:
        subprocess.run([_FFMPEG, "-version"], capture_output=True, timeout=5)
        return True
    except Exception:
        return False


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _vlog(f"whisper_model_load_start model={WHISPER_MODEL}")
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        _vlog(f"whisper_model_load_done model={WHISPER_MODEL}")
    return _whisper_model


def _normalize_audio(input_path: str) -> str:
    """Convert to 16 kHz mono WAV with loudnorm. Returns temp WAV path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    subprocess.run(
        [_FFMPEG, "-y", "-i", input_path,
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
         "-ar", "16000", "-ac", "1", wav_path],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return wav_path


def _transcribe_deepgram(audio_path: str) -> str:
    """
    Blocking: Deepgram Nova-3 Arabic (SDK v6).
    SDK-level timeout = _DEEPGRAM_TIMEOUT seconds.
    Raises on any failure — caller handles.
    """
    from deepgram import DeepgramClient                    # lazy import
    from deepgram.core.request_options import RequestOptions
    dg = DeepgramClient(api_key=DEEPGRAM_API_KEY)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    response = dg.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-3",
        language="ar",
        smart_format=True,
        request_options=RequestOptions(timeout_in_seconds=_DEEPGRAM_TIMEOUT),
    )
    transcript = response.results.channels[0].alternatives[0].transcript
    return (transcript or "").strip()


def _transcribe_whisper(audio_path: str) -> str:
    """
    Blocking: local faster-whisper (includes model load on first call).
    WARNING: first call takes 40+ seconds on Render cold start.
    """
    _vlog("whisper_start")
    model = _get_whisper_model()
    segments, info = model.transcribe(
        audio_path, language="ar", beam_size=5, vad_filter=True,
    )
    raw = " ".join(seg.text for seg in segments).strip()
    _vlog(f"whisper_done chars={len(raw)} lang={info.language}")
    return raw


def _fix_transcription(raw: str) -> str:
    """Optional Claude Haiku post-processing: fix Arabic transcription errors."""
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


async def _run_transcription(source: str, loop: asyncio.AbstractEventLoop) -> str:
    """
    Async: attempt Deepgram (20 s) then Whisper fallback (30 s).
    Returns transcript string. Returns "" if both fail. Never raises.
    Emits _vlog() events at every stage.
    """
    if DEEPGRAM_API_KEY:
        _vlog("selected_backend=deepgram")
        _vlog(f"deepgram_key_present=true")
        _vlog("deepgram_request_start")
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, _transcribe_deepgram, source),
                timeout=_DEEPGRAM_TIMEOUT,
            )
            if raw:
                _vlog(f"deepgram_request_done chars={len(raw)}")
                return raw
            _vlog("deepgram_exception reason=empty_transcript")
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _vlog("deepgram_timeout")
        except Exception as exc:
            _vlog(f"deepgram_exception reason={type(exc).__name__}:{exc!s:.120}")

        # Whisper fallback
        _vlog("selected_backend=whisper (deepgram_failed)")
        _vlog("fallback_whisper_started")
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, _transcribe_whisper, source),
                timeout=_WHISPER_TIMEOUT,
            )
            _vlog(f"fallback_whisper_success chars={len(raw)}")
            return raw or ""
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _vlog(f"whisper_timeout after={_WHISPER_TIMEOUT}s")
            return ""
        except Exception as exc:
            _vlog(f"fallback_whisper_error reason={type(exc).__name__}:{exc!s:.120}")
            return ""

    else:
        _vlog("deepgram_key_present=false")
        _vlog("selected_backend=whisper (no_key)")
        _vlog("whisper_start")
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, _transcribe_whisper, source),
                timeout=_WHISPER_TIMEOUT,
            )
            _vlog(f"whisper_done chars={len(raw)}")
            return raw or ""
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _vlog(f"whisper_timeout after={_WHISPER_TIMEOUT}s")
            return ""
        except Exception as exc:
            _vlog(f"fallback_whisper_error reason={type(exc).__name__}:{exc!s:.120}")
            return ""


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all recent tasks."""
    if not _authorized(update):
        return
    msg = await update.message.reply_text("⏳ Fetching tasks...")
    try:
        r = http_lib.get(f"{AXIS_BASE_URL}/tasks?limit=20", timeout=15)
        r.raise_for_status()
        data    = r.json()
        count   = data.get("count", 0)
        records = data.get("tasks", [])
        if not records:
            await msg.edit_text("No tasks found.")
            return
        lines = [f"📋 *Tasks* ({count} total)\n"]
        STATUS_EMOJI = {
            "pending_confirmation": "⏳", "scheduled": "📅",
            "completed": "✅", "failed": "❌",
            "waiting_for_user": "🕐", "cancelled": "🚫",
        }
        for r_ in records[:20]:
            emoji = STATUS_EMOJI.get(r_["status"], "•")
            due   = f" · {r_['due_at'][:10]}" if r_.get("due_at") else ""
            lines.append(f"{emoji} `{r_['task_id'][:8]}` {r_['title']}{due}")
        if count > 20:
            lines.append(f"_…and {count - 20} more_")
        await msg.edit_text("\n".join(lines), parse_mode="Markdown")
    except Exception as exc:
        await msg.edit_text(f"⚠️ Error fetching tasks: {exc}")


async def cmd_pending(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List tasks pending confirmation."""
    if not _authorized(update):
        return
    msg = await update.message.reply_text("⏳ Checking pending tasks...")
    try:
        r = http_lib.get(f"{AXIS_BASE_URL}/tasks/pending", timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("count", 0) == 0:
            await msg.edit_text("No tasks pending confirmation.")
            return
        text = data.get("formatted") or f"⏳ {data['count']} pending task(s)."
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as exc:
        await msg.edit_text(f"⚠️ Error: {exc}")


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel a task: /cancel <task_id>"""
    if not _authorized(update):
        return
    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: /cancel <task_id>")
        return
    task_id = args[0]
    msg = await update.message.reply_text(f"⏳ Cancelling task `{task_id}`...", parse_mode="Markdown")
    try:
        r = http_lib.delete(f"{AXIS_BASE_URL}/tasks/{task_id}", timeout=15)
        if r.status_code == 404:
            await msg.edit_text(f"Task `{task_id}` not found.", parse_mode="Markdown")
        elif r.status_code == 200:
            await msg.edit_text(f"🚫 Task `{task_id}` cancelled.", parse_mode="Markdown")
        else:
            await msg.edit_text(f"⚠️ HTTP {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        await msg.edit_text(f"⚠️ Error: {exc}")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    await update.message.reply_text(
        "AXIS is ready.\n\nSend a text message or a voice note — I'll transcribe and respond.\n\n"
        "Commands: /tasks · /pending · /cancel <id> · /status"
    )


async def cmd_whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else "unknown"
    await update.message.reply_text(f"Your Telegram user ID: {uid}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check AXIS API connectivity."""
    if not _authorized(update):
        return
    msg = await update.message.reply_text("⏳ Checking AXIS API...")
    try:
        r = http_lib.get(AXIS_HEALTH_URL, timeout=15)
        if r.status_code == 200:
            data = r.json()
            svc  = data.get("service", "AXIS")
            await msg.edit_text(
                f"✅ AXIS API is online\n"
                f"URL: `{AXIS_API_URL}`\n"
                f"Service: {svc}",
                parse_mode="Markdown",
            )
        else:
            await msg.edit_text(f"⚠️ AXIS API returned HTTP {r.status_code}\nURL: {AXIS_API_URL}")
    except Exception as exc:
        await msg.edit_text(
            f"❌ AXIS API unreachable\n"
            f"URL: {AXIS_API_URL}\n"
            f"Error: {exc}"
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    text = (update.message.text or "").strip()
    if not text:
        return

    uid        = update.effective_user.id
    # update_id is unique per Telegram message and stable across retries
    request_id = f"tg-{update.update_id}"
    print(f"[{_ts()}] [AXIS Telegram] text uid={uid} update_id={update.update_id}: {text[:80]!r}")

    thinking = await update.message.reply_text("⏳")
    try:
        response = await _ask_axis(text, request_id=request_id)
        if len(response) > 4000:
            response = response[:4000] + "\n\n⚠️ [تم اختصار الرد — الجواب كان أطول]"
        await thinking.edit_text(response)
    except Exception as exc:
        print(f"[{_ts()}] [AXIS Telegram] ERROR calling AXIS API: {exc}")
        await thinking.edit_text(
            f"⚠️ AXIS API error: {exc}\n\n"
            f"API URL: {AXIS_API_URL}\n"
            "Use /status to check connectivity."
        )


async def _voice_inner(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    status,
    uid: int,
    request_id: str,
) -> None:
    """
    Core voice pipeline.  Called inside a 60-second asyncio.wait_for guard.
    Any exception propagates to handle_voice which always edits status.
    """
    loop     = asyncio.get_running_loop()
    ogg_path: Optional[str] = None
    wav_path: Optional[str] = None

    try:
        # ── Telegram download ───────────────────────────────────────────────
        voice   = update.message.voice
        _vlog(f"telegram_file_id_received file_id={voice.file_id}")
        _vlog("telegram_file_download_start")
        tg_file = await context.bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            ogg_path = tmp.name
        await tg_file.download_to_drive(ogg_path)
        _vlog(f"telegram_file_download_done path={ogg_path}")

        # ── ffmpeg normalize ────────────────────────────────────────────────
        _vlog("ffmpeg_start")
        try:
            wav_path = await loop.run_in_executor(None, _normalize_audio, ogg_path)
            source   = wav_path
            _vlog("ffmpeg_done")
        except Exception as exc:
            _vlog(f"ffmpeg_failed reason={exc!s:.80} using_original=true")
            source = ogg_path

        # ── Transcription with timeouts ─────────────────────────────────────
        text = await _run_transcription(source, loop)

        if not text:
            _vlog("transcription_empty — sending failure message")
            await status.edit_text(_FAILURE_MSG)
            return

        _vlog(f"transcription_complete chars={len(text)}")
        await status.edit_text(f'🎙 "{text}"\n\n⏳ Asking AXIS...')

        # ── AXIS API ────────────────────────────────────────────────────────
        _vlog("axis_request_start")
        response_data = await asyncio.wait_for(
            loop.run_in_executor(None, _call_axis_api, text, request_id),
            timeout=_AXIS_TIMEOUT,
        )
        reply = _format_response(response_data)
        _vlog("axis_request_done")

        if len(reply) > 4000:
            reply = reply[:4000] + "\n\n⚠️ [تم اختصار الرد — الجواب كان أطول]"
        await status.edit_text(f'🎙 "{text}"\n\n{reply}')
        _vlog("telegram_edit_done")

    finally:
        for path in (wav_path, ogg_path):
            if path:
                Path(path).unlink(missing_ok=True)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return

    uid        = update.effective_user.id
    request_id = f"tg-{update.update_id}"
    _vlog(f"voice_received uid={uid} update_id={update.update_id}")

    status = await update.message.reply_text("🎙 Transcribing...")

    try:
        await asyncio.wait_for(
            _voice_inner(update, context, status, uid, request_id),
            timeout=_VOICE_HANDLER_TIMEOUT,
        )
    except (asyncio.TimeoutError, asyncio.CancelledError):
        _vlog(f"voice_handler_timeout after={_VOICE_HANDLER_TIMEOUT}s")
        try:
            await status.edit_text(_FAILURE_MSG)
        except Exception:
            pass
    except Exception as exc:
        _vlog(f"voice_handler_error reason={type(exc).__name__}:{exc!s:.120}")
        try:
            await status.edit_text(_FAILURE_MSG)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _check_api_on_startup() -> None:
    """Log whether the AXIS API is reachable at startup. Non-fatal."""
    try:
        r = http_lib.get(AXIS_HEALTH_URL, timeout=10)
        if r.status_code == 200:
            print(f"[AXIS Telegram] ✅ AXIS API reachable: {AXIS_HEALTH_URL}")
        else:
            print(f"[AXIS Telegram] ⚠️ AXIS API returned HTTP {r.status_code}: {AXIS_HEALTH_URL}")
    except Exception as exc:
        print(f"[AXIS Telegram] ⚠️ AXIS API unreachable at startup: {exc}")
        print(f"[AXIS Telegram]    Messages will fail until the API is online.")


def main():
    print(f"[AXIS Telegram] Starting — authorized UID: {AUTHORIZED_UID}")
    print(f"[AXIS Telegram] AXIS API URL : {AXIS_API_URL}")
    print(f"[AXIS Telegram] Health URL   : {AXIS_HEALTH_URL}")
    dg_status = "set ✓" if DEEPGRAM_API_KEY else "NOT SET ← voice will use slow Whisper"
    print(f"[AXIS Telegram] DEEPGRAM_API_KEY : {dg_status}")
    if DEEPGRAM_API_KEY:
        print(f"[AXIS Telegram] STT backend  : Deepgram Nova-3 + Whisper fallback "
              f"(timeouts: dg={_DEEPGRAM_TIMEOUT}s whisper={_WHISPER_TIMEOUT}s)")
    else:
        print(f"[AXIS Telegram] STT backend  : faster-whisper ({WHISPER_MODEL}) "
              f"timeout={_WHISPER_TIMEOUT}s  [WARNING: cold load takes 40+ s]")
    print(f"[AXIS Telegram] Voice handler timeout : {_VOICE_HANDLER_TIMEOUT}s total")
    print(f"[AXIS Telegram] AXIS API timeout      : {_AXIS_TIMEOUT}s (voice)")
    print(f"[AXIS Telegram] Claude postprocess    : {'on' if VOICE_POSTPROCESS_WITH_CLAUDE else 'off'}")

    _check_api_on_startup()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("whoami",  cmd_whoami))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("tasks",   cmd_tasks))
    app.add_handler(CommandHandler("pending", cmd_pending))
    app.add_handler(CommandHandler("cancel",  cmd_cancel))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("[AXIS Telegram] Polling for messages...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
