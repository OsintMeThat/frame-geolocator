"""Tool G.28 — ``identify_spoken_language``: spoken language ID + transcription.

The spoken language (and accent) is a strong region prior, and the **transcription** can
surface place names said aloud — a street, a town — which are decisive cues. Runs a
local Whisper-family model; no LLM API.

Design notes
------------
* **Local & free**: faster-whisper (CTranslate2) with ``int8`` compute on CPU, matching
  this project's hardware (no NVIDIA GPU). The model is downloaded once and cached.
* faster-whisper decodes audio internally (via PyAV/ffmpeg), so it accepts a **video or
  audio path directly** — ``extract_audio`` is not required here (it stays the
  foundation for ``flash_to_bang_range``).
* The actual ASR call is isolated in ``_run_backend`` so the orchestration is unit-
  testable with a stub (no model download).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_MODEL_SIZE = "small"


@dataclass
class SpokenLanguageConfig:
    """Configuration for :func:`identify_spoken_language`."""

    model_size: str = DEFAULT_MODEL_SIZE
    """Whisper model size: tiny/base/small/medium/large-v3 (bigger = better, slower)."""

    compute_type: str = "int8"
    """CTranslate2 compute type; ``int8`` is the CPU-efficient default."""

    device: str = "cpu"
    """``cpu`` on this project's hardware."""

    transcribe: bool = True
    """Also return the transcription text, not just the detected language."""

    beam_size: int = 5
    """Decoding beam size when transcribing."""

    min_language_probability: float = 0.5
    """Below this language-detection probability, treat the audio as having no reliable
    speech. Whisper hallucinates plausible text on noise-dominated tracks (crowd,
    gunfire, music), so a low probability must not become a false region prior."""


@dataclass
class Segment:
    """A transcribed speech segment."""

    start: float
    end: float
    text: str


@dataclass
class SpokenLanguageResult:
    """Auditable result of :func:`identify_spoken_language`."""

    source_path: str
    engine_used: str
    language: str | None
    language_probability: float
    has_speech: bool
    transcription: str
    segments: list[Segment] = field(default_factory=list)
    model_size: str = DEFAULT_MODEL_SIZE

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False),
                              encoding="utf-8")


def _run_backend(
    source_path: str, config: SpokenLanguageConfig
) -> tuple[str | None, float, list[Segment]]:
    """Run faster-whisper. Returns (language, probability, segments).

    Isolated so tests can monkeypatch it without downloading a model.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "faster-whisper is not installed. Install the 'audio' extra: "
            "pip install -e '.[audio]'"
        ) from exc

    model = WhisperModel(config.model_size, device=config.device,
                         compute_type=config.compute_type)
    segments_iter, info = model.transcribe(source_path, beam_size=config.beam_size)
    segments: list[Segment] = []
    if config.transcribe:
        for seg in segments_iter:
            segments.append(Segment(start=round(seg.start, 2), end=round(seg.end, 2),
                                    text=seg.text.strip()))
    return info.language, float(info.language_probability), segments


def identify_spoken_language(
    source_path: str | Path, config: SpokenLanguageConfig | None = None
) -> SpokenLanguageResult:
    """Detect the spoken language of a media and optionally transcribe it.

    Parameters
    ----------
    source_path:
        A video or audio file (faster-whisper decodes both).
    config:
        Optional :class:`SpokenLanguageConfig`.

    Returns
    -------
    SpokenLanguageResult
    """
    config = config or SpokenLanguageConfig()
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Media not found: {source_path}")

    language, probability, segments = _run_backend(str(source_path), config)
    if probability >= config.min_language_probability:
        transcription = " ".join(s.text for s in segments).strip()
        has_speech = bool(transcription)
    else:
        # Low confidence → likely hallucination on noise-dominated audio. Report no
        # reliable speech rather than a misleading language/transcription.
        language = None
        segments = []
        transcription = ""
        has_speech = False

    return SpokenLanguageResult(
        source_path=str(source_path),
        engine_used="faster-whisper",
        language=language,
        language_probability=round(probability, 3),
        has_speech=has_speech,
        transcription=transcription,
        segments=segments,
        model_size=config.model_size,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``identify-spoken-language <media> [--model small] [--json out.json]``."""
    parser = argparse.ArgumentParser(description="Spoken language ID + transcription.")
    parser.add_argument("media", help="Path to a video or audio file.")
    parser.add_argument("--model", default=DEFAULT_MODEL_SIZE, help="Whisper model size.")
    parser.add_argument("--no-transcribe", action="store_true",
                        help="Only detect the language, do not transcribe.")
    parser.add_argument("--json", help="Optional path to write the full result as JSON.")
    args = parser.parse_args(argv)

    config = SpokenLanguageConfig(model_size=args.model, transcribe=not args.no_transcribe)
    result = identify_spoken_language(args.media, config)
    print(f"[engine: {result.engine_used}/{result.model_size}] "
          f"language={result.language} (p={result.language_probability:.2f})")
    if result.transcription:
        preview = result.transcription[:300]
        print(f"  transcription: {preview}{'…' if len(result.transcription) > 300 else ''}")
    if args.json:
        result.to_json(args.json)
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
