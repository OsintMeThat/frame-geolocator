"""Tests for ``extract_skyline_profile``.

Synthetic images with a known sky/ground boundary verify the geometry deterministically
(no model, no real media), so correctness is asserted without overfitting to any frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from frame_geolocator.tools.extract_skyline_profile import (
    SkylineConfig,
    extract_skyline_profile,
    write_overlay,
)


def _sky_over_ground(path: Path, ridge_peak_col: int | None = None) -> None:
    """Blue sky on top, dark-green ground below, optional triangular peak."""
    h, w = 200, 300
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = (200, 150, 90)        # BGR blue-ish sky everywhere first
    ground_top = 120
    img[ground_top:] = (40, 90, 40)  # dark green ground
    if ridge_peak_col is not None:
        for x in range(w):
            # triangle peaking at ridge_peak_col, rising 60 px above the base
            rise = int(60 * max(0.0, 1 - abs(x - ridge_peak_col) / (w / 2)))
            img[ground_top - rise:ground_top, x] = (40, 90, 40)
    cv2.imwrite(str(path), img)


def test_flat_horizon_detected(tmp_path: Path) -> None:
    f = tmp_path / "flat.png"
    _sky_over_ground(f)
    prof = extract_skyline_profile(f)
    assert prof.usable
    assert prof.coverage > 0.9
    # Boundary near row 120/200 = 0.6, flat → low roughness.
    assert 0.55 < prof.mean_horizon_frac < 0.65
    assert prof.roughness < 0.03


def test_ridge_peak_is_higher_than_edges(tmp_path: Path) -> None:
    f = tmp_path / "ridge.png"
    _sky_over_ground(f, ridge_peak_col=150)
    prof = extract_skyline_profile(f)
    assert prof.usable
    # At the peak the boundary is higher up the image (smaller normalized value).
    peak = prof.horizon_norm[150]
    edge = prof.horizon_norm[10]
    assert peak >= 0 and edge >= 0
    assert peak < edge - 0.1
    assert prof.roughness > 0.03  # a ridge is more jagged than a flat horizon


def test_no_sky_is_flagged_unusable(tmp_path: Path) -> None:
    f = tmp_path / "ground.png"
    img = np.full((200, 300, 3), (40, 90, 40), dtype=np.uint8)  # all ground, no sky
    cv2.imwrite(str(f), img)
    prof = extract_skyline_profile(f, SkylineConfig(min_coverage=0.25))
    assert not prof.usable
    assert prof.coverage < 0.25


def test_fov_gives_elevation_degrees(tmp_path: Path) -> None:
    f = tmp_path / "flat.png"
    _sky_over_ground(f)
    prof = extract_skyline_profile(f, SkylineConfig(fov_v_deg=50.0))
    assert prof.horizon_deg is not None
    assert len(prof.horizon_deg) == prof.width


def test_overlay_written(tmp_path: Path) -> None:
    f = tmp_path / "ridge.png"
    _sky_over_ground(f, ridge_peak_col=150)
    prof = extract_skyline_profile(f)
    out = tmp_path / "overlay.png"
    write_overlay(f, prof, out)
    assert out.exists()


def test_missing_image_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_skyline_profile(tmp_path / "nope.png")
