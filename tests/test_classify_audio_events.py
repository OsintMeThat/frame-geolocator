"""Tests for ``classify_audio_events``.

The PANNs backend is stubbed so the orchestration is tested without downloading the
~300 MB checkpoint. A real run is gated behind ``RUN_PANNS_TESTS``.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from frame_geolocator.tools.classify_audio_events import AudioEvent, classify_audio_events

events_mod = importlib.import_module("frame_geolocator.tools.classify_audio_events")


@pytest.fixture()
def media(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"stubbed backend, content unused")
    return path


def _stub_extract_audio(monkeypatch: pytest.MonkeyPatch, *, has_audio: bool) -> None:
    from frame_geolocator.tools.extract_audio import ExtractAudioResult

    def fake(source_path, out_path=None, config=None):
        out = Path(out_path) if out_path else Path(source_path).with_suffix(".wav")
        if has_audio:
            out.write_bytes(b"RIFF....WAVEfmt ")  # placeholder, backend is stubbed too
            return ExtractAudioResult(str(source_path), str(out), True, 16000, 1, 3.0)
        return ExtractAudioResult(str(source_path), None, False, 16000, 1, 0.0)

    monkeypatch.setattr(events_mod, "extract_audio", fake)


def test_returns_tags_and_events(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_extract_audio(monkeypatch, has_audio=True)

    def fake_backend(wav_path, config):
        clip = [AudioEvent("Gunshot, gunfire", 0.81, osint_highlight=True)]
        evts = [AudioEvent("Gunshot, gunfire", 0.81, 4.2, 6.8, osint_highlight=True)]
        return clip, evts

    monkeypatch.setattr(events_mod, "_run_backend", fake_backend)
    result = classify_audio_events(media)

    assert result.has_audio
    assert result.clip_tags[0].label == "Gunshot, gunfire"
    assert result.clip_tags[0].osint_highlight is True
    assert result.events[0].start_s == 4.2


def test_no_audio_short_circuits(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_extract_audio(monkeypatch, has_audio=False)
    result = classify_audio_events(media)
    assert result.has_audio is False
    assert result.clip_tags == []


def test_missing_media_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        classify_audio_events(tmp_path / "nope.mp4")


@pytest.mark.skipif(
    not os.environ.get("RUN_PANNS_TESTS"), reason="set RUN_PANNS_TESTS=1 to run real PANNs"
)
def test_real_panns_on_gunfire() -> None:
    video = Path("data/videos/003.mp4")  # Niamey, gunfire
    if not video.exists():
        pytest.skip("real media not present")
    result = classify_audio_events(video)
    assert result.has_audio
    assert result.clip_tags
