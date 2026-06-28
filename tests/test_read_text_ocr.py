"""Tests for the ``read_text_ocr`` tool.

The fallback orchestration is tested with stub backends (no heavy ML deps). A real
end-to-end OCR test runs only if a backend (easyocr/tesseract) is actually installed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from frame_geolocator.tools.read_text_ocr import OCRConfig, TextDetection, read_text_ocr

# The package re-exports a ``read_text_ocr`` function that shadows the submodule of the
# same name; import the module object explicitly to reach its internals (``_BACKENDS``).
ocr_mod = importlib.import_module("frame_geolocator.tools.read_text_ocr")


def _image_with_text(path: Path, text: str = "NIGER") -> None:
    img = np.full((200, 600, 3), 255, dtype=np.uint8)
    cv2.putText(img, text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


# --- Stub backends for deterministic fallback tests ------------------------- #
class _Working:
    name = "working"

    def available(self) -> bool:
        return True

    def run(self, image, languages, min_confidence):
        return [TextDetection(text="HELLO", confidence=0.9, bbox=[[0, 0], [1, 0], [1, 1], [0, 1]])]


class _Unavailable:
    name = "unavailable"

    def available(self) -> bool:
        return False

    def run(self, image, languages, min_confidence):  # pragma: no cover
        raise AssertionError("should not be called")


class _Crashing:
    name = "crashing"

    def available(self) -> bool:
        return True

    def run(self, image, languages, min_confidence):
        raise ValueError("boom")


@pytest.fixture()
def image(tmp_path: Path) -> Path:
    path = tmp_path / "sign.png"
    _image_with_text(path)
    return path


@pytest.fixture()
def stub_registry(monkeypatch: pytest.MonkeyPatch):
    registry = {"working": _Working(), "unavailable": _Unavailable(), "crashing": _Crashing()}
    monkeypatch.setattr(ocr_mod, "_BACKENDS", registry)
    return registry


def test_uses_first_working_backend(image: Path, stub_registry) -> None:
    result = read_text_ocr(image, OCRConfig(backends=["working"]))
    assert result.engine_used == "working"
    assert "HELLO" in result.full_text


def test_falls_back_past_unavailable_and_crashing(image: Path, stub_registry) -> None:
    result = read_text_ocr(image, OCRConfig(backends=["unavailable", "crashing", "working"]))
    assert result.engine_used == "working"
    assert result.detections


def test_raises_when_no_backend_usable(image: Path, stub_registry) -> None:
    with pytest.raises(RuntimeError):
        read_text_ocr(image, OCRConfig(backends=["unavailable", "crashing"]))


def test_unknown_backend_name_raises(image: Path, stub_registry) -> None:
    with pytest.raises(RuntimeError):
        read_text_ocr(image, OCRConfig(backends=["does_not_exist"]))


def test_missing_image_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_text_ocr(tmp_path / "nope.png")


# --- Real end-to-end OCR (only if a backend is installed) ------------------- #
def _has_real_backend() -> bool:
    return any(ocr_mod._BACKENDS[name].available() for name in ("easyocr", "tesseract"))


@pytest.mark.skipif(not _has_real_backend(), reason="no real OCR backend installed")
def test_real_ocr_reads_text(image: Path) -> None:
    result = read_text_ocr(image, OCRConfig(languages=["en"]))
    assert result.engine_used in ("easyocr", "tesseract")
    assert "NIGER" in result.full_text.upper()
