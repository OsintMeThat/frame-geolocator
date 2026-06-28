"""Tests for the ``identify_spoken_language`` tool.

The ASR backend is stubbed so the orchestration is tested without downloading a Whisper
model. A real run is gated behind the ``RUN_WHISPER_TESTS`` env var (it downloads a
model and needs media), to keep the default suite fast and offline.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from frame_geolocator.tools.identify_spoken_language import (
    Segment,
    SpokenLanguageConfig,
    identify_spoken_language,
)

lang_mod = importlib.import_module("frame_geolocator.tools.identify_spoken_language")


@pytest.fixture()
def media(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not a real video, backend is stubbed")
    return path


def test_builds_result_from_backend(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_backend(source_path, config):
        return "ru", 0.98, [Segment(0.0, 1.5, "привет"), Segment(1.5, 3.0, "Москва")]

    monkeypatch.setattr(lang_mod, "_run_backend", fake_backend)
    result = identify_spoken_language(media)

    assert result.language == "ru"
    assert result.language_probability == 0.98
    assert result.has_speech is True
    assert "Москва" in result.transcription
    assert len(result.segments) == 2
    assert result.engine_used == "faster-whisper"


def test_no_speech_when_empty(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lang_mod, "_run_backend", lambda s, c: (None, 0.0, []))
    result = identify_spoken_language(media)
    assert result.has_speech is False
    assert result.transcription == ""


def test_missing_media_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        identify_spoken_language(tmp_path / "nope.mp4")


@pytest.mark.skipif(
    not os.environ.get("RUN_WHISPER_TESTS"), reason="set RUN_WHISPER_TESTS=1 to run real ASR"
)
def test_real_asr_detects_language() -> None:
    video = Path("data/videos/002.mp4")  # Putin / Russian
    if not video.exists():
        pytest.skip("real media not present")
    result = identify_spoken_language(video, SpokenLanguageConfig(model_size="base"))
    assert result.language is not None
