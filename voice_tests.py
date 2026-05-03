"""
AXIS Voice Tests — routing, timeout, failure, and Telegram status-edit guarantees.

All tests use mocks; no real audio, Deepgram API, or Whisper models required.

Run with: pytest voice_tests.py -v
"""

import asyncio
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, call

import telegram_bot as bot

_FAILURE_MSG = bot._FAILURE_MSG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_loop():
    return asyncio.get_event_loop()


# ---------------------------------------------------------------------------
# 1. Backend selection
# ---------------------------------------------------------------------------

def test_deepgram_called_when_key_set():
    """_run_transcription() uses Deepgram when DEEPGRAM_API_KEY is present."""
    async def go():
        with (
            patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
            patch.object(bot, "_transcribe_deepgram", return_value="مرحبا") as mock_dg,
            patch.object(bot, "_transcribe_whisper") as mock_w,
        ):
            result = await bot._run_transcription("/tmp/test.wav", _make_loop())
        mock_dg.assert_called_once()
        mock_w.assert_not_called()
        assert result == "مرحبا"
    run(go())


def test_whisper_called_when_no_key():
    """_run_transcription() uses Whisper when DEEPGRAM_API_KEY is absent."""
    async def go():
        with (
            patch.object(bot, "DEEPGRAM_API_KEY", ""),
            patch.object(bot, "_transcribe_whisper", return_value="مرحبا") as mock_w,
            patch.object(bot, "_transcribe_deepgram") as mock_dg,
        ):
            result = await bot._run_transcription("/tmp/test.wav", _make_loop())
        mock_w.assert_called_once()
        mock_dg.assert_not_called()
        assert result == "مرحبا"
    run(go())


# ---------------------------------------------------------------------------
# 2. Deepgram failure → Whisper fallback
# ---------------------------------------------------------------------------

def test_whisper_fallback_on_deepgram_exception():
    """Any Deepgram exception triggers Whisper fallback."""
    async def go():
        with (
            patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
            patch.object(bot, "_transcribe_deepgram", side_effect=Exception("API error")),
            patch.object(bot, "_transcribe_whisper", return_value="أهلاً") as mock_w,
        ):
            result = await bot._run_transcription("/tmp/test.wav", _make_loop())
        mock_w.assert_called_once()
        assert result == "أهلاً"
    run(go())


def test_whisper_fallback_on_deepgram_empty():
    """Empty Deepgram transcript triggers Whisper fallback."""
    async def go():
        with (
            patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
            patch.object(bot, "_transcribe_deepgram", return_value=""),
            patch.object(bot, "_transcribe_whisper", return_value="نص من ويسبر") as mock_w,
        ):
            result = await bot._run_transcription("/tmp/test.wav", _make_loop())
        mock_w.assert_called_once()
        assert result == "نص من ويسبر"
    run(go())


# ---------------------------------------------------------------------------
# 3. Timeout behaviour
# ---------------------------------------------------------------------------

def test_deepgram_timeout_triggers_whisper_fallback():
    """Deepgram that exceeds _DEEPGRAM_TIMEOUT → Whisper fallback."""
    def slow_dg(_path):
        time.sleep(10)      # blocks thread; asyncio cancels the future
        return "never"

    async def go():
        with (
            patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
            patch.object(bot, "_DEEPGRAM_TIMEOUT", 0.05),
            patch.object(bot, "_WHISPER_TIMEOUT", 5),
            patch.object(bot, "_transcribe_deepgram", side_effect=slow_dg),
            patch.object(bot, "_transcribe_whisper", return_value="fallback") as mock_w,
        ):
            result = await bot._run_transcription("/tmp/test.wav", _make_loop())
        mock_w.assert_called_once()
        assert result == "fallback"
    run(go())


def test_whisper_timeout_returns_empty():
    """Whisper timeout (no Deepgram key) returns empty string, never raises."""
    def slow_w(_path):
        time.sleep(10)
        return "never"

    async def go():
        with (
            patch.object(bot, "DEEPGRAM_API_KEY", ""),
            patch.object(bot, "_WHISPER_TIMEOUT", 0.05),
            patch.object(bot, "_transcribe_whisper", side_effect=slow_w),
        ):
            result = await bot._run_transcription("/tmp/test.wav", _make_loop())
        assert result == ""
    run(go())


def test_both_backends_fail_returns_empty():
    """Both Deepgram and Whisper failing returns empty string, never raises."""
    async def go():
        with (
            patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
            patch.object(bot, "_transcribe_deepgram", side_effect=Exception("down")),
            patch.object(bot, "_transcribe_whisper", side_effect=Exception("crash")),
        ):
            result = await bot._run_transcription("/tmp/test.wav", _make_loop())
        assert result == ""
    run(go())


# ---------------------------------------------------------------------------
# 4. Telegram status always edited
# ---------------------------------------------------------------------------

def test_handle_voice_edits_status_on_empty_transcript():
    """handle_voice edits the status message when transcript is empty."""
    status_mock = AsyncMock()
    update_mock = MagicMock()
    update_mock.effective_user.id = 42
    update_mock.update_id = 1
    update_mock.message.voice.file_id = "fid"
    update_mock.message.reply_text = AsyncMock(return_value=status_mock)

    context_mock = MagicMock()
    tg_file = AsyncMock()
    tg_file.download_to_drive = AsyncMock()
    context_mock.bot.get_file = AsyncMock(return_value=tg_file)

    async def go():
        with (
            patch.object(bot, "AUTHORIZED_UID", 42),
            patch.object(bot, "_normalize_audio", return_value="/tmp/t.wav"),
            patch.object(bot, "_run_transcription", return_value=""),
            patch("pathlib.Path.unlink"),
            patch("tempfile.NamedTemporaryFile", MagicMock(
                return_value=MagicMock(
                    __enter__=MagicMock(return_value=MagicMock(name="/tmp/fake.ogg")),
                    __exit__=MagicMock(return_value=False),
                )
            )),
        ):
            await bot.handle_voice(update_mock, context_mock)

    run(go())
    status_mock.edit_text.assert_called_once()
    msg = status_mock.edit_text.call_args[0][0]
    assert "failed" in msg.lower() or "resend" in msg.lower()


def test_handle_voice_edits_status_on_exception():
    """handle_voice edits status when an unexpected exception is raised."""
    status_mock = AsyncMock()
    update_mock = MagicMock()
    update_mock.effective_user.id = 42
    update_mock.update_id = 2
    update_mock.message.voice.file_id = "fid2"
    update_mock.message.reply_text = AsyncMock(return_value=status_mock)

    context_mock = MagicMock()
    context_mock.bot.get_file = AsyncMock(side_effect=RuntimeError("Telegram API down"))

    async def go():
        with patch.object(bot, "AUTHORIZED_UID", 42):
            await bot.handle_voice(update_mock, context_mock)

    run(go())
    status_mock.edit_text.assert_called_once()
    msg = status_mock.edit_text.call_args[0][0]
    assert "failed" in msg.lower() or "resend" in msg.lower()


def test_handle_voice_edits_status_on_handler_timeout():
    """handle_voice edits status when the 60-second hard timeout fires."""
    status_mock = AsyncMock()
    update_mock = MagicMock()
    update_mock.effective_user.id = 42
    update_mock.update_id = 3
    update_mock.message.voice.file_id = "fid3"
    update_mock.message.reply_text = AsyncMock(return_value=status_mock)

    context_mock = MagicMock()

    async def slow_inner(*args, **kwargs):
        await asyncio.sleep(10)     # blocks longer than the 50 ms test timeout

    async def go():
        with (
            patch.object(bot, "AUTHORIZED_UID", 42),
            patch.object(bot, "_VOICE_HANDLER_TIMEOUT", 0.05),
            patch.object(bot, "_voice_inner", side_effect=slow_inner),
        ):
            await bot.handle_voice(update_mock, context_mock)

    run(go())
    status_mock.edit_text.assert_called_once()
    msg = status_mock.edit_text.call_args[0][0]
    assert "failed" in msg.lower() or "resend" in msg.lower()


# ---------------------------------------------------------------------------
# 5. Empty responses
# ---------------------------------------------------------------------------

def test_empty_deepgram_falls_back_to_whisper():
    async def go():
        with (
            patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
            patch.object(bot, "_transcribe_deepgram", return_value=""),
            patch.object(bot, "_transcribe_whisper", return_value="") as mock_w,
        ):
            result = await bot._run_transcription("/tmp/test.wav", _make_loop())
        mock_w.assert_called_once()
        assert result == ""
    run(go())


def test_empty_whisper_no_key_returns_empty():
    async def go():
        with (
            patch.object(bot, "DEEPGRAM_API_KEY", ""),
            patch.object(bot, "_transcribe_whisper", return_value=""),
        ):
            result = await bot._run_transcription("/tmp/test.wav", _make_loop())
        assert result == ""
    run(go())


# ---------------------------------------------------------------------------
# 6. Deepgram SDK call structure
# ---------------------------------------------------------------------------

def test_transcribe_deepgram_extracts_transcript():
    """_transcribe_deepgram() extracts transcript from Deepgram v6 response."""
    mock_response = MagicMock()
    mock_response.results.channels[0].alternatives[0].transcript = "اجتماع مع أحمد غداً"

    with patch("deepgram.DeepgramClient") as MockClient:
        instance = MockClient.return_value
        instance.listen.v1.media.transcribe_file.return_value = mock_response

        with (
            patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
            patch("builtins.open", MagicMock(return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"audio"))),
                __exit__=MagicMock(return_value=False),
            ))),
        ):
            result = bot._transcribe_deepgram("/tmp/audio.wav")

    assert result == "اجتماع مع أحمد غداً"


# ---------------------------------------------------------------------------
# 8. handle_voice sends failure message when transcription returns empty
# ---------------------------------------------------------------------------

def test_handle_voice_sends_failure_on_empty_transcript():
    """handle_voice edits the status message with a failure string when transcript is empty."""
    status_mock = AsyncMock()

    update_mock = MagicMock()
    update_mock.effective_user.id = 12345
    update_mock.update_id = 99
    update_mock.message.voice.file_id = "file123"
    update_mock.message.reply_text = AsyncMock(return_value=status_mock)

    context_mock = MagicMock()
    tg_file_mock = AsyncMock()
    tg_file_mock.download_to_drive = AsyncMock()
    context_mock.bot.get_file = AsyncMock(return_value=tg_file_mock)

    async def go():
        with (
            patch.object(bot, "AUTHORIZED_UID", 12345),
            patch.object(bot, "_normalize_audio", return_value="/tmp/test.wav"),
            patch.object(bot, "_transcribe_async", return_value=""),
            patch("pathlib.Path.unlink"),
            patch("tempfile.NamedTemporaryFile", MagicMock(
                return_value=MagicMock(
                    __enter__=MagicMock(return_value=MagicMock(name="/tmp/fake.ogg")),
                    __exit__=MagicMock(return_value=False),
                )
            )),
        ):
            await bot.handle_voice(update_mock, context_mock)

    run(go())

    # The status message must have been edited — user never left on "Transcribing..."
    status_mock.edit_text.assert_called_once()
    call_args = status_mock.edit_text.call_args[0][0]
    assert "transcription failed" in call_args.lower() or "resend" in call_args.lower()
