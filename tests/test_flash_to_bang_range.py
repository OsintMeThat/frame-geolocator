"""Tests for ``flash_to_bang_range``.

The signal-processing core is tested on synthetic signals, so no real explosion footage
is needed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from frame_geolocator.tools import FlashToBangConfig, flash_to_bang_range
from frame_geolocator.tools.flash_to_bang_range import (
    _detect_transients,
    _filter_transients_to_windows,
    _in_windows,
    _pair_flash_bang,
)


def test_detect_transients_finds_a_spike() -> None:
    times = np.linspace(0, 10, 1000)
    signal = np.zeros(1000)
    signal[500] = 50.0
    signal[501] = 25.0
    events = _detect_transients(times, signal, prominence_z=4.0, min_separation_s=0.3)
    assert events
    peak_time, _ = events[0]
    assert abs(peak_time - times[500]) < 0.1


def test_detect_transients_ignores_flat_signal() -> None:
    times = np.linspace(0, 5, 200)
    signal = np.ones(200) * 7.0
    assert _detect_transients(times, signal, prominence_z=4.0, min_separation_s=0.3) == []


def test_pair_computes_distance_from_delay() -> None:
    config = FlashToBangConfig(speed_of_sound_mps=343.0)
    events = _pair_flash_bang([(1.0, 6.0)], [(1.5, 6.0)], config)
    assert len(events) == 1
    assert events[0].delay_s == pytest.approx(0.5, abs=1e-6)
    assert events[0].distance_m == pytest.approx(171.5, abs=0.1)


def test_pair_skips_delay_beyond_max_distance() -> None:
    config = FlashToBangConfig(speed_of_sound_mps=343.0, max_distance_m=100.0)  # ~0.29s max
    assert _pair_flash_bang([(1.0, 6.0)], [(1.5, 6.0)], config) == []


def test_pair_picks_earliest_bang_after_flash() -> None:
    config = FlashToBangConfig()
    events = _pair_flash_bang([(1.0, 6.0)], [(3.0, 6.0), (1.4, 6.0)], config)
    assert events[0].bang_time_s == pytest.approx(1.4)


def test_in_windows_with_margin() -> None:
    windows = [(4.0, 6.0)]
    assert _in_windows(5.0, windows)
    assert _in_windows(3.7, windows)  # within 0.5 margin
    assert not _in_windows(8.0, windows)


def test_filter_transients_keeps_only_those_in_windows() -> None:
    transients = [(1.0, 5.0), (5.0, 6.0), (9.0, 7.0)]
    kept = _filter_transients_to_windows(transients, [(4.0, 6.0)])
    assert kept == [(5.0, 6.0)]


def _patch_classify(monkeypatch: pytest.MonkeyPatch, result) -> None:
    # Patch the module attribute directly: the package re-exports a function that shadows
    # the submodule, so a dotted-string monkeypatch target would resolve to the function.
    cae = importlib.import_module("frame_geolocator.tools.classify_audio_events")
    monkeypatch.setattr(cae, "classify_audio_events", lambda source_path, config=None: result)


def test_assess_audio_flags_music_as_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    f2b = importlib.import_module("frame_geolocator.tools.flash_to_bang_range")
    from frame_geolocator.tools.classify_audio_events import AudioEvent, ClassifyAudioResult

    _patch_classify(monkeypatch, ClassifyAudioResult(
        source_path="x.mp4", engine_used="panns", has_audio=True,
        clip_tags=[AudioEvent("Music", 0.82), AudioEvent("Speech", 0.4)],
        events=[AudioEvent("Music", 0.82, 0.0, 30.0)],
    ))
    windows, usable, note = f2b._assess_audio("x.mp4", FlashToBangConfig())
    assert windows == []
    assert usable is False
    assert "music" in note.lower()


def test_assess_audio_returns_impulse_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    f2b = importlib.import_module("frame_geolocator.tools.flash_to_bang_range")
    from frame_geolocator.tools.classify_audio_events import AudioEvent, ClassifyAudioResult

    _patch_classify(monkeypatch, ClassifyAudioResult(
        source_path="x.mp4", engine_used="panns", has_audio=True,
        clip_tags=[AudioEvent("Explosion", 0.6), AudioEvent("Music", 0.1)],
        events=[AudioEvent("Explosion", 0.6, 14.0, 15.0)],
    ))
    windows, usable, _ = f2b._assess_audio("x.mp4", FlashToBangConfig())
    assert windows == [(14.0, 15.0)]
    assert usable is True


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        flash_to_bang_range(tmp_path / "nope.mp4")
