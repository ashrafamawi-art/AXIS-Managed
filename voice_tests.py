"""
AXIS Voice Transcription Tests — validates routing logic without real audio/API calls.

All tests use mocks so they run in CI without DEEPGRAM_API_KEY or Whisper models.

Run with: pytest voice_tests.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

import telegram_bot as bot


# ---------------------------------------------------------------------------
# 1. Backend selection: Deepgram when key set, Whisper when not
# ---------------------------------------------------------------------------

def test_deepgram_called_when_key_set():
    """_transcribe() routes to Deepgram when DEEPGRAM_API_KEY is present."""
    with (
        patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
        patch.object(bot, "VOICE_POSTPROCESS_WITH_CLAUDE", False),
        patch.object(bot, "_normalize_audio", return_value="/tmp/test.wav"),
        patch.object(bot, "_transcribe_deepgram", return_value="مرحبا") as mock_dg,
        patch.object(bot, "_transcribe_whisper") as mock_w,
        patch("pathlib.Path.unlink"),
    ):
        result = bot._transcribe("/tmp/fake.ogg")

    mock_dg.assert_called_once()
    mock_w.assert_not_called()
    assert result == "مرحبا"


def test_whisper_called_when_no_key():
    """_transcribe() routes to faster-whisper when DEEPGRAM_API_KEY is absent."""
    with (
        patch.object(bot, "DEEPGRAM_API_KEY", ""),
        patch.object(bot, "VOICE_POSTPROCESS_WITH_CLAUDE", False),
        patch.object(bot, "_normalize_audio", return_value="/tmp/test.wav"),
        patch.object(bot, "_transcribe_whisper", return_value="مرحبا") as mock_w,
        patch.object(bot, "_transcribe_deepgram") as mock_dg,
        patch("pathlib.Path.unlink"),
    ):
        result = bot._transcribe("/tmp/fake.ogg")

    mock_w.assert_called_once()
    mock_dg.assert_not_called()
    assert result == "مرحبا"


# ---------------------------------------------------------------------------
# 2. Deepgram failure falls back to Whisper
# ---------------------------------------------------------------------------

def test_whisper_fallback_on_deepgram_failure():
    """When Deepgram raises, _transcribe() falls back to faster-whisper silently."""
    with (
        patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
        patch.object(bot, "VOICE_POSTPROCESS_WITH_CLAUDE", False),
        patch.object(bot, "_normalize_audio", return_value="/tmp/test.wav"),
        patch.object(bot, "_transcribe_deepgram", side_effect=Exception("connection timeout")),
        patch.object(bot, "_transcribe_whisper", return_value="أهلاً") as mock_w,
        patch("pathlib.Path.unlink"),
    ):
        result = bot._transcribe("/tmp/fake.ogg")

    mock_w.assert_called_once()
    assert result == "أهلاً"


# ---------------------------------------------------------------------------
# 3. Claude post-processing flag
# ---------------------------------------------------------------------------

def test_fix_transcription_called_when_flag_on():
    """_fix_transcription() is called when VOICE_POSTPROCESS_WITH_CLAUDE=True."""
    with (
        patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
        patch.object(bot, "VOICE_POSTPROCESS_WITH_CLAUDE", True),
        patch.object(bot, "_normalize_audio", return_value="/tmp/test.wav"),
        patch.object(bot, "_transcribe_deepgram", return_value="مرحبا"),
        patch.object(bot, "_fix_transcription", return_value="مرحباً") as mock_fix,
        patch("pathlib.Path.unlink"),
    ):
        result = bot._transcribe("/tmp/fake.ogg")

    mock_fix.assert_called_once_with("مرحبا")
    assert result == "مرحباً"


def test_fix_transcription_skipped_when_flag_off():
    """_fix_transcription() is NOT called when VOICE_POSTPROCESS_WITH_CLAUDE=False."""
    with (
        patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
        patch.object(bot, "VOICE_POSTPROCESS_WITH_CLAUDE", False),
        patch.object(bot, "_normalize_audio", return_value="/tmp/test.wav"),
        patch.object(bot, "_transcribe_deepgram", return_value="مرحبا"),
        patch.object(bot, "_fix_transcription") as mock_fix,
        patch("pathlib.Path.unlink"),
    ):
        result = bot._transcribe("/tmp/fake.ogg")

    mock_fix.assert_not_called()
    assert result == "مرحبا"


# ---------------------------------------------------------------------------
# 4. Empty transcription propagates correctly
# ---------------------------------------------------------------------------

def test_empty_deepgram_response_returns_empty():
    """Empty Deepgram transcript is returned as empty string without crashing."""
    with (
        patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
        patch.object(bot, "VOICE_POSTPROCESS_WITH_CLAUDE", False),
        patch.object(bot, "_normalize_audio", return_value="/tmp/test.wav"),
        patch.object(bot, "_transcribe_deepgram", return_value=""),
        patch("pathlib.Path.unlink"),
    ):
        result = bot._transcribe("/tmp/fake.ogg")

    assert result == ""


def test_empty_whisper_response_returns_empty():
    """Empty Whisper transcript is returned as empty string without crashing."""
    with (
        patch.object(bot, "DEEPGRAM_API_KEY", ""),
        patch.object(bot, "VOICE_POSTPROCESS_WITH_CLAUDE", False),
        patch.object(bot, "_normalize_audio", return_value="/tmp/test.wav"),
        patch.object(bot, "_transcribe_whisper", return_value=""),
        patch("pathlib.Path.unlink"),
    ):
        result = bot._transcribe("/tmp/fake.ogg")

    assert result == ""


# ---------------------------------------------------------------------------
# 5. Normalize audio failure falls back to original path
# ---------------------------------------------------------------------------

def test_normalize_failure_uses_original_path():
    """If ffmpeg normalization fails, _transcribe() uses the original file path."""
    with (
        patch.object(bot, "DEEPGRAM_API_KEY", ""),
        patch.object(bot, "VOICE_POSTPROCESS_WITH_CLAUDE", False),
        patch.object(bot, "_normalize_audio", side_effect=Exception("ffmpeg not found")),
        patch.object(bot, "_transcribe_whisper", return_value="نص") as mock_w,
    ):
        result = bot._transcribe("/tmp/fake.ogg")

    mock_w.assert_called_once_with("/tmp/fake.ogg")
    assert result == "نص"


# ---------------------------------------------------------------------------
# 6. _transcribe_deepgram() extracts transcript from Deepgram response shape
# ---------------------------------------------------------------------------

def test_transcribe_deepgram_extracts_transcript():
    """_transcribe_deepgram() correctly extracts transcript from Deepgram v6 response."""
    mock_response = MagicMock()
    mock_response.results.channels[0].alternatives[0].transcript = "اجتماع مع أحمد غداً"

    import builtins
    real_open = builtins.open

    with patch("deepgram.DeepgramClient") as MockClient:
        instance = MockClient.return_value
        instance.listen.v1.media.transcribe_file.return_value = mock_response

        with (
            patch.object(bot, "DEEPGRAM_API_KEY", "fake-key"),
            patch("builtins.open", MagicMock(return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"fake-audio"))),
                __exit__=MagicMock(return_value=False),
            ))),
        ):
            result = bot._transcribe_deepgram("/tmp/audio.wav")

    assert result == "اجتماع مع أحمد غداً"
