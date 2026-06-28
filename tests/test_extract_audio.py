"""Tests for the ``extract_audio`` tool.

A tiny clip (with and without an audio track) is synthesized on the fly using the same
bundled ffmpeg the tool relies on, so the tests are self-contained.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from frame_geolocator.tools import ExtractAudioConfig, extract_audio

_HAS_FFMPEG = importlib.util.find_spec("imageio_ffmpeg") is not None
pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="imageio-ffmpeg not installed")


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _make_clip(path: Path, *, with_audio: bool) -> bool:
    """Synthesize a 2s test clip. Returns False if the codecs are unavailable."""
    cmd = [_ffmpeg(), "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and path.exists()


def test_extract_audio_from_clip_with_sound(tmp_path: Path) -> None:
    clip = tmp_path / "av.mp4"
    if not _make_clip(clip, with_audio=True):
        pytest.skip("bundled ffmpeg lacks required codecs")

    result = extract_audio(clip, tmp_path / "out.wav", ExtractAudioConfig(sample_rate=16000))
    assert result.has_audio
    assert result.audio_path and Path(result.audio_path).exists()
    assert result.sample_rate == 16000
    assert result.channels == 1
    assert result.duration_s > 1.0


def test_video_without_audio_reports_has_audio_false(tmp_path: Path) -> None:
    clip = tmp_path / "silent.mp4"
    if not _make_clip(clip, with_audio=False):
        pytest.skip("bundled ffmpeg lacks required codecs")

    result = extract_audio(clip, tmp_path / "out.wav")
    assert result.has_audio is False
    assert result.audio_path is None


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "nope.mp4")
