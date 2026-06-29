"""Tool G.30 — ``classify_audio_events``: detect sound events (gunshot, siren, …).

Beyond speech, a soundtrack contains acoustic events that are strong context cues:
gunfire, artillery, explosions, sirens (whose tones differ by country), aircraft,
helicopters, crowds, music, call to prayer. This tool tags them with timestamps.

Design notes
------------
* **Local & free**: PANNs (CNN14 trained on Google AudioSet, 527 classes) via the
  ``panns-inference`` package on PyTorch/CPU — no API. The checkpoint downloads once.
* Produces both **clip-level tags** (top-K for the whole media) and **framewise events**
  (label + time range + score), so the orchestrator gets "there is gunfire from 4.2s to
  6.8s", not just "gunfire somewhere".
* Speech is one of the AudioSet classes; for the spoken words/transcription use
  ``identify_spoken_language`` (Whisper), which is purpose-built for that.
* The ASR-style backend is isolated in ``_run_backend`` for stub-based unit testing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from frame_geolocator.tools.extract_audio import extract_audio

PANNS_SAMPLE_RATE = 32000  # PANNs models expect 32 kHz mono

# AudioSet label substrings that matter most for OSINT context (case-insensitive).
OSINT_HIGHLIGHTS = (
    "gunshot", "gunfire", "machine gun", "fusillade", "artillery", "cannon",
    "explosion", "boom", "eruption", "siren", "emergency vehicle", "police car",
    "ambulance", "fire engine", "aircraft", "helicopter", "drone",
)


@dataclass
class ClassifyAudioConfig:
    """Configuration for :func:`classify_audio_events`."""

    top_k: int = 10
    """How many clip-level tags to return."""

    event_threshold: float = 0.2
    """Minimum framewise score for a class to be reported as an event."""

    sample_rate: int = PANNS_SAMPLE_RATE
    """Resample rate for the model (PANNs requires 32 kHz)."""


@dataclass
class AudioEvent:
    """A detected sound. Clip-level tags have ``start_s``/``end_s`` = None."""

    label: str
    score: float
    start_s: float | None = None
    end_s: float | None = None
    osint_highlight: bool = False


@dataclass
class ClassifyAudioResult:
    """Auditable result of :func:`classify_audio_events`."""

    source_path: str
    engine_used: str
    has_audio: bool
    clip_tags: list[AudioEvent] = field(default_factory=list)
    events: list[AudioEvent] = field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False),
                              encoding="utf-8")


def _is_highlight(label: str) -> bool:
    low = label.lower()
    return any(key in low for key in OSINT_HIGHLIGHTS)


def _run_backend(
    wav_path: str, config: ClassifyAudioConfig
) -> tuple[list[AudioEvent], list[AudioEvent]]:
    """Run PANNs sound-event detection. Returns (clip_tags, events).

    Isolated so tests can monkeypatch it without downloading the checkpoint.
    """
    try:
        import librosa
        from panns_inference import SoundEventDetection, labels
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise RuntimeError(
            "panns-inference is not installed. Install the 'sound' extra: "
            "pip install -e '.[sound]'"
        ) from exc

    audio, _ = librosa.load(wav_path, sr=config.sample_rate, mono=True)
    if audio.size == 0:
        return [], []

    sed = SoundEventDetection(checkpoint_path=None, device="cpu")
    framewise = sed.inference(audio[None, :])[0]  # (frames, 527)
    n_frames = framewise.shape[0]
    duration = audio.size / config.sample_rate
    frame_dur = duration / n_frames if n_frames else 0.0

    clip_scores = framewise.max(axis=0)
    top = clip_scores.argsort()[::-1][: config.top_k]
    clip_tags = [
        AudioEvent(label=labels[i], score=round(float(clip_scores[i]), 3),
                   osint_highlight=_is_highlight(labels[i]))
        for i in top
    ]

    events: list[AudioEvent] = []
    for i in range(len(labels)):
        if clip_scores[i] < config.event_threshold:
            continue
        above = framewise[:, i] >= config.event_threshold
        idx = 0
        while idx < n_frames:
            if above[idx]:
                j = idx
                while j < n_frames and above[j]:
                    j += 1
                events.append(
                    AudioEvent(
                        label=labels[i],
                        score=round(float(framewise[idx:j, i].max()), 3),
                        start_s=round(idx * frame_dur, 2),
                        end_s=round(j * frame_dur, 2),
                        osint_highlight=_is_highlight(labels[i]),
                    )
                )
                idx = j
            else:
                idx += 1

    events.sort(key=lambda e: e.score, reverse=True)
    return clip_tags, events


def classify_audio_events(
    source_path: str | Path, config: ClassifyAudioConfig | None = None
) -> ClassifyAudioResult:
    """Detect and time-stamp sound events (gunfire, siren, explosion, …) in a media.

    Parameters
    ----------
    source_path:
        A video or audio file.
    config:
        Optional :class:`ClassifyAudioConfig`.

    Returns
    -------
    ClassifyAudioResult
        Clip-level tags and framewise events; ``has_audio=False`` if there is no sound.
    """
    config = config or ClassifyAudioConfig()
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Media not found: {source_path}")

    audio = extract_audio(source_path, source_path.with_suffix(".events.wav"),
                          None)
    if not audio.has_audio or audio.audio_path is None:
        return ClassifyAudioResult(source_path=str(source_path), engine_used="panns",
                                   has_audio=False)
    try:
        clip_tags, events = _run_backend(audio.audio_path, config)
    finally:
        Path(audio.audio_path).unlink(missing_ok=True)

    return ClassifyAudioResult(
        source_path=str(source_path),
        engine_used="panns",
        has_audio=True,
        clip_tags=clip_tags,
        events=events,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``classify-audio-events <media> [--threshold 0.2] [--json out.json]``."""
    parser = argparse.ArgumentParser(description="Detect sound events (gunshot, siren…).")
    parser.add_argument("media", help="Path to a video or audio file.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--json", help="Optional path to write the full result as JSON.")
    parser.add_argument("--case-file", metavar="PATH",
                        help="Shared case file to merge this result into.")
    args = parser.parse_args(argv)

    config = ClassifyAudioConfig(top_k=args.top_k, event_threshold=args.threshold)
    result = classify_audio_events(args.media, config)
    if not result.has_audio:
        print("No audio stream.")
        return 0
    print("Clip tags:")
    for tag in result.clip_tags:
        mark = " ★" if tag.osint_highlight else ""
        print(f"  {tag.score:.2f}  {tag.label}{mark}")
    if result.events:
        print("Events:")
        for ev in result.events[:20]:
            mark = " ★" if ev.osint_highlight else ""
            print(f"  [{ev.start_s:.1f}-{ev.end_s:.1f}s] {ev.score:.2f}  {ev.label}{mark}")
    if args.json:
        result.to_json(args.json)
        print(f"Wrote {args.json}")
    if args.case_file:
        from frame_geolocator.case_file import merge
        merge(args.case_file, "classify_audio_events", result)
        print(f"Case file: {args.case_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
