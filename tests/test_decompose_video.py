"""Tests for the ``decompose_video`` tool.

The tests synthesize a small clip on the fly (no external media needed), so they are
fully self-contained and run against the real OpenCV/ffmpeg backend.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from frame_geolocator.tools import DecomposeConfig, decompose_video


def _make_clip(path: Path, n_scenes: int = 4, frames_per_scene: int = 15, fps: int = 15) -> None:
    """Write a synthetic clip: several distinct, textured 'scenes'."""
    width, height = 320, 240
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    rng = np.random.default_rng(42)
    try:
        for scene in range(n_scenes):
            # A distinct base colour + sharp random texture per scene.
            base = np.full((height, width, 3), (scene * 50 % 256), dtype=np.uint8)
            texture = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            frame = cv2.addWeighted(base, 0.5, texture, 0.5, 0)
            cv2.rectangle(frame, (20 + scene * 10, 20), (120, 120), (255, 255, 255), 3)
            for _ in range(frames_per_scene):
                writer.write(frame)
    finally:
        writer.release()


@pytest.fixture()
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.avi"
    _make_clip(path)
    assert path.exists() and path.stat().st_size > 0
    return path


def test_decompose_produces_frames_and_sheet(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = decompose_video(clip, out, DecomposeConfig(target_frames=8, sample_fps=5))

    assert result.selected, "expected at least one selected frame"
    assert len(result.selected) <= 8
    for sel in result.selected:
        assert Path(sel.path).exists()
    assert result.contact_sheet_path and Path(result.contact_sheet_path).exists()
    assert (out / "result.json").exists()


def test_target_frames_is_respected(clip: Path, tmp_path: Path) -> None:
    result = decompose_video(
        clip, tmp_path / "out", DecomposeConfig(target_frames=3, sample_fps=5)
    )
    assert len(result.selected) <= 3


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        decompose_video(tmp_path / "nope.avi", tmp_path / "out")


def test_no_contact_sheet_option(clip: Path, tmp_path: Path) -> None:
    result = decompose_video(
        clip, tmp_path / "out", DecomposeConfig(contact_sheet=False, sample_fps=5)
    )
    assert result.contact_sheet_path is None
