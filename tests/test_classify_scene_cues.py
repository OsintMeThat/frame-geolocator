"""Tests for the ``classify_scene_cues`` tool.

Aggregation and the low-light gating are tested with a deterministic fake backend (no
torch / no model download). A real end-to-end CLIP test runs only when OpenCLIP is
installed *and* ``RUN_CLIP_TESTS=1`` is set, mirroring the audio tools' opt-in pattern.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from frame_geolocator.tools.classify_scene_cues import (
    SceneCuesConfig,
    classify_scene_cues,
)

scene_mod = importlib.import_module("frame_geolocator.tools.classify_scene_cues")


# --- Deterministic fake backend (no heavy deps) ----------------------------- #
def _hashvec(text: str, dim: int = 8) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    v = np.random.default_rng(seed).standard_normal(dim)
    return v / np.linalg.norm(v)


class _FakeBackend:
    name = "fake"

    def available(self) -> bool:
        return True

    def embed_image(self, image_rgb: np.ndarray) -> np.ndarray:
        # Deterministic but frame-dependent, so aggregation actually averages.
        return _hashvec(f"img{int(image_rgb.mean())}")

    def embed_texts(self, prompts: tuple[str, ...]) -> np.ndarray:
        return np.stack([_hashvec(p) for p in prompts])


class _Unavailable:
    name = "unavailable"

    def available(self) -> bool:
        return False

    def embed_image(self, image_rgb):  # pragma: no cover
        raise AssertionError("should not be called")

    def embed_texts(self, prompts):  # pragma: no cover
        raise AssertionError("should not be called")


@pytest.fixture()
def fake_registry(monkeypatch: pytest.MonkeyPatch):
    registry = {"fake": _FakeBackend(), "unavailable": _Unavailable()}
    monkeypatch.setattr(scene_mod, "_BACKENDS", registry)
    return registry


def _frame(path: Path, luminance: int) -> None:
    img = np.full((64, 64, 3), luminance, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _frames_dir(root: Path, luminances: list[int]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i, lum in enumerate(luminances):
        _frame(root / f"frame_{i:03d}.jpg", lum)
    return root


def _cfg(**kw) -> SceneCuesConfig:
    return SceneCuesConfig(backends=["fake"], **kw)


# --- Core behaviour --------------------------------------------------------- #
def test_runs_on_single_image(tmp_path: Path, fake_registry) -> None:
    f = tmp_path / "a.jpg"
    _frame(f, 130)
    res = classify_scene_cues(f, _cfg())
    assert res.backend_used == "fake"
    assert res.n_frames == 1
    groups = {g.group: g for g in res.groups}
    assert {"setting", "biome", "time_of_day"} <= set(groups)
    assert groups["setting"].top in scene_mod.CUE_GROUPS["setting"]["cues"]


def test_aggregates_over_directory(tmp_path: Path, fake_registry) -> None:
    d = _frames_dir(tmp_path / "frames", [120, 125, 130, 135])
    res = classify_scene_cues(d, _cfg())
    assert res.n_frames == 4
    # All bright → biome reliable and computed over every frame.
    biome = next(g for g in res.groups if g.group == "biome")
    assert biome.reliable is True
    assert biome.frames_used == 4


def test_low_light_marks_biome_unreliable(tmp_path: Path, fake_registry) -> None:
    d = _frames_dir(tmp_path / "dark", [20, 25, 30])  # all below default min_luminance
    res = classify_scene_cues(d, _cfg())
    biome = next(g for g in res.groups if g.group == "biome")
    assert biome.reliable is False
    assert biome.top is None
    assert biome.note and "low light" in biome.note
    # Scores are still reported for transparency/audit.
    assert biome.scores
    # A light-robust group stays reliable even in the dark.
    tod = next(g for g in res.groups if g.group == "time_of_day")
    assert tod.reliable is True


def test_biome_uses_only_bright_frames(tmp_path: Path, fake_registry) -> None:
    d = _frames_dir(tmp_path / "mixed", [20, 30, 120, 140])  # 2 dark, 2 bright
    res = classify_scene_cues(d, _cfg())
    biome = next(g for g in res.groups if g.group == "biome")
    assert biome.reliable is True
    assert biome.frames_used == 2  # only the two bright frames
    setting = next(g for g in res.groups if g.group == "setting")
    assert setting.frames_used == 4  # light-robust group uses all frames


def test_max_frames_subsamples(tmp_path: Path, fake_registry) -> None:
    d = _frames_dir(tmp_path / "many", [120] * 10)
    res = classify_scene_cues(d, _cfg(max_frames=3))
    assert res.n_frames == 3


def test_contact_sheet_is_skipped(tmp_path: Path, fake_registry) -> None:
    d = _frames_dir(tmp_path / "frames", [120, 130])
    _frame(d / "contact_sheet.jpg", 120)
    res = classify_scene_cues(d, _cfg())
    assert res.n_frames == 2


def test_groups_subset(tmp_path: Path, fake_registry) -> None:
    f = tmp_path / "a.jpg"
    _frame(f, 130)
    res = classify_scene_cues(f, _cfg(groups=["time_of_day"]))
    assert [g.group for g in res.groups] == ["time_of_day"]


def test_json_and_missing_source(tmp_path: Path, fake_registry) -> None:
    f = tmp_path / "a.jpg"
    _frame(f, 130)
    out = tmp_path / "scene.json"
    classify_scene_cues(f, _cfg()).to_json(out)
    assert out.exists() and "groups" in out.read_text()
    with pytest.raises(FileNotFoundError):
        classify_scene_cues(tmp_path / "nope.jpg", _cfg())


def test_raises_when_no_backend_usable(tmp_path: Path, fake_registry) -> None:
    f = tmp_path / "a.jpg"
    _frame(f, 130)
    with pytest.raises(RuntimeError):
        classify_scene_cues(f, SceneCuesConfig(backends=["unavailable"]))


# --- Real OpenCLIP (opt-in) ------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("RUN_CLIP_TESTS") != "1"
    or not scene_mod._BACKENDS["openclip"].available(),
    reason="set RUN_CLIP_TESTS=1 with the 'vision' extra installed",
)
def test_real_openclip_coastal(tmp_path: Path) -> None:
    # A blue lower half (sea) under a lighter sky — should not crash and should pick a
    # setting cue. We assert structure, not a specific label (model-dependent).
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:128] = (200, 200, 200)
    img[128:] = (150, 80, 20)  # BGR-ish blue water
    f = tmp_path / "sea.jpg"
    cv2.imwrite(str(f), img)
    res = classify_scene_cues(f)
    assert res.backend_used == "openclip"
    assert any(g.group == "setting" and g.top for g in res.groups)
