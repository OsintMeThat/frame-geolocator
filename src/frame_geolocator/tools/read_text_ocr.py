"""Tool B.6 — ``read_text_ocr``: multilingual OCR for OSINT media.

Text in a frame (street signs, shop fronts, building names, banners, plates) is one of
the highest-leverage geolocation cues: it can pin a country, a city, or even a specific
building in a single step. This tool reads it.

Design notes
------------
* **Local & free**, no LLM API (see project guardrails). Backends are pluggable and
  tried in priority order with **automatic fallback**: if the preferred engine is not
  installed or raises, the next one is used. The engine that actually produced the
  result is reported in :class:`OCRResult.engine_used`.
* Default chain: ``easyocr`` (good quality, CPU-friendly, multilingual) →
  ``tesseract`` (lightweight system fallback).
* Heavy backend libraries are imported lazily, so ``import frame_geolocator.tools`` and
  the rest of the pipeline stay light even without the OCR extra installed.
* Optional light preprocessing (upscale + CLAHE) helps on small / low-contrast text,
  which is typical of degraded OSINT media.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

DEFAULT_LANGUAGES = ["en", "fr"]
DEFAULT_BACKENDS = ["easyocr", "tesseract"]

# Generic ISO-639-1 codes -> Tesseract language data names.
_TESSERACT_LANG = {
    "en": "eng", "fr": "fra", "ru": "rus", "ar": "ara", "de": "deu",
    "es": "spa", "it": "ita", "pt": "por", "nl": "nld", "tr": "tur",
    "uk": "ukr", "pl": "pol", "fa": "fas", "zh": "chi_sim",
}


@dataclass
class OCRConfig:
    """Configuration for :func:`read_text_ocr`."""

    languages: list[str] = field(default_factory=lambda: list(DEFAULT_LANGUAGES))
    """Languages to look for (ISO-639-1). EasyOCR needs script-compatible sets."""

    backends: list[str] = field(default_factory=lambda: list(DEFAULT_BACKENDS))
    """Backend names in priority order; the first that works is used."""

    min_confidence: float = 0.3
    """Drop detections below this confidence (0..1)."""

    preprocess: bool = True
    """Upscale small images and CLAHE-normalize contrast before OCR."""

    upscale_min_width: int = 1000
    """If the image is narrower than this, upscale it to this width before OCR."""


@dataclass
class TextDetection:
    """A single piece of text found in the image."""

    text: str
    confidence: float
    bbox: list[list[int]]  # four [x, y] corner points, clockwise from top-left


@dataclass
class OCRResult:
    """Auditable result of :func:`read_text_ocr`."""

    image_path: str
    engine_used: str
    languages: list[str]
    detections: list[TextDetection] = field(default_factory=list)
    full_text: str = ""

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False),
                              encoding="utf-8")


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def _preprocess(bgr: np.ndarray, config: OCRConfig) -> np.ndarray:
    """Upscale small images and boost local contrast to help OCR on degraded text."""
    out = bgr
    h, w = out.shape[:2]
    if w < config.upscale_min_width:
        scale = config.upscale_min_width / w
        out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(lightness), a, b)), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class OCRBackend(Protocol):
    """Contract every OCR backend must satisfy."""

    name: str

    def available(self) -> bool:
        """Whether the backend can run (library + data present)."""
        ...

    def run(self, image_bgr: np.ndarray, languages: list[str], min_confidence: float
            ) -> list[TextDetection]:
        """Run OCR and return detections above ``min_confidence``."""
        ...


class EasyOCRBackend:
    """EasyOCR backend (PyTorch, multilingual, CPU-friendly)."""

    name = "easyocr"

    def __init__(self) -> None:
        self._readers: dict[tuple[str, ...], object] = {}

    def available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("easyocr") is not None

    def _reader(self, languages: list[str]):
        key = tuple(languages)
        if key not in self._readers:
            import easyocr

            # CPU only on this project's target hardware (no NVIDIA GPU).
            self._readers[key] = easyocr.Reader(list(languages), gpu=False, verbose=False)
        return self._readers[key]

    def run(self, image_bgr, languages, min_confidence):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections = []
        for bbox, text, conf in self._reader(languages).readtext(rgb):
            if conf < min_confidence or not text.strip():
                continue
            detections.append(
                TextDetection(
                    text=text.strip(),
                    confidence=round(float(conf), 3),
                    bbox=[[int(x), int(y)] for x, y in bbox],
                )
            )
        return detections


class TesseractBackend:
    """Tesseract backend (lightweight system OCR, used as fallback)."""

    name = "tesseract"

    def available(self) -> bool:
        import importlib.util

        if importlib.util.find_spec("pytesseract") is None:
            return False
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def run(self, image_bgr, languages, min_confidence):
        import pytesseract
        from pytesseract import Output

        lang = "+".join(_TESSERACT_LANG.get(code, code) for code in languages)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(gray, lang=lang, output_type=Output.DICT)
        detections = []
        for i, text in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if conf < 0 or not text.strip():
                continue
            conf /= 100.0
            if conf < min_confidence:
                continue
            x, y, w, h = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            detections.append(
                TextDetection(
                    text=text.strip(),
                    confidence=round(conf, 3),
                    bbox=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                )
            )
        return detections


# Backend registry. Tests may inject fakes here.
_BACKENDS: dict[str, OCRBackend] = {
    "easyocr": EasyOCRBackend(),
    "tesseract": TesseractBackend(),
}


def _order_top_to_bottom(detections: list[TextDetection]) -> list[TextDetection]:
    """Sort detections by vertical then horizontal position for readable full_text."""
    def key(d: TextDetection) -> tuple[int, int]:
        ys = [p[1] for p in d.bbox]
        xs = [p[0] for p in d.bbox]
        return (min(ys), min(xs))

    return sorted(detections, key=key)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def read_text_ocr(image_path: str | Path, config: OCRConfig | None = None) -> OCRResult:
    """Read text from an image, trying OCR backends in order with automatic fallback.

    Parameters
    ----------
    image_path:
        Path to the image (a frame extracted by ``decompose_video``, or any picture).
    config:
        Optional :class:`OCRConfig`.

    Returns
    -------
    OCRResult
        Detections, the concatenated text, and which engine produced them.

    Raises
    ------
    FileNotFoundError
        If the image does not exist.
    RuntimeError
        If none of the configured backends is available/usable.
    """
    config = config or OCRConfig()
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    if config.preprocess:
        image = _preprocess(image, config)

    errors: list[str] = []
    for name in config.backends:
        backend = _BACKENDS.get(name)
        if backend is None:
            errors.append(f"{name}: unknown backend")
            continue
        if not backend.available():
            errors.append(f"{name}: not installed/available")
            continue
        try:
            detections = backend.run(image, config.languages, config.min_confidence)
        except Exception as exc:  # noqa: BLE001 — fall back on any backend failure
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        ordered = _order_top_to_bottom(detections)
        return OCRResult(
            image_path=str(image_path),
            engine_used=backend.name,
            languages=list(config.languages),
            detections=ordered,
            full_text="\n".join(d.text for d in ordered),
        )

    raise RuntimeError(
        "No usable OCR backend. Install the 'ocr' extra (pip install -e '.[ocr]') and, "
        "for tesseract, the system package. Tried: " + "; ".join(errors)
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``read-text-ocr <image> [--lang en fr] [--backends easyocr tesseract]``."""
    parser = argparse.ArgumentParser(description="Multilingual OCR with fallback.")
    parser.add_argument("image", help="Path to the image / frame.")
    parser.add_argument("--lang", nargs="+", default=DEFAULT_LANGUAGES,
                        help="Languages (ISO-639-1), e.g. --lang en fr ru.")
    parser.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS,
                        help="Backend priority order.")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--json", help="Optional path to write the full result as JSON.")
    parser.add_argument("--case-file", metavar="PATH",
                        help="Shared case file to merge this result into.")
    args = parser.parse_args(argv)

    config = OCRConfig(
        languages=args.lang,
        backends=args.backends,
        min_confidence=args.min_confidence,
        preprocess=not args.no_preprocess,
    )
    result = read_text_ocr(args.image, config)
    print(f"[engine: {result.engine_used}] {len(result.detections)} detections")
    for det in result.detections:
        print(f"  ({det.confidence:.2f}) {det.text}")
    if args.json:
        result.to_json(args.json)
        print(f"Wrote {args.json}")
    if args.case_file:
        from frame_geolocator.case_file import merge
        merge(args.case_file, "read_text_ocr", result)
        print(f"Case file: {args.case_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
