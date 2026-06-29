"""Tool G.29 — ``flash_to_bang_range``: acoustic ranging from flash-to-bang delay.

A classic OSINT/artillery technique: a visual event (muzzle flash, explosion) is seen
before its sound is heard. The delay times the speed of sound gives the **distance to
the source**:

    distance ≈ delay_seconds × speed_of_sound (~343 m/s at 20 °C)

This does **not** localize on its own — it produces a distance constraint that feeds
``rank_hypotheses`` / ``triangulate_from_landmarks``.

Design notes
------------
* **Local & free**, pure numeric (numpy + opencv + the audio extracted by
  ``extract_audio``).
* The signal-processing core (``_detect_transients``, ``_pair_flash_bang``) is separated
  from IO so it is unit-testable on synthetic signals — no real explosion footage needed.
* Flash detection uses a high brightness percentile per frame (a muzzle flash lights up
  a few very bright pixels even if small). Bang detection uses the audio energy
  envelope. Both reuse the same transient detector.
* Assumes audio and video share a common timeline (true for a single muxed clip).
"""

from __future__ import annotations

import argparse
import json
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from frame_geolocator.tools.extract_audio import extract_audio

SPEED_OF_SOUND_MPS = 343.0

# AudioSet label substrings for impulsive sounds a flash can be ranged against, and for
# detecting an unusable (music-dominated) soundtrack.
IMPULSE_KEYS = (
    "explosion", "boom", "gunshot", "gunfire", "machine gun", "artillery", "cannon",
    "eruption", "fireworks", "fusillade", "burst",
)
MUSIC_KEYS = ("music",)


@dataclass
class FlashToBangConfig:
    """Configuration for :func:`flash_to_bang_range`."""

    speed_of_sound_mps: float = SPEED_OF_SOUND_MPS
    """Speed of sound; ~343 at 20 °C, ~331 at 0 °C. Tune for known conditions."""

    min_delay_s: float = 0.05
    """Ignore flash/bang pairs closer than this (source essentially at the camera)."""

    max_distance_m: float = 5000.0
    """Plausibility cap; sets the max flash→bang delay considered (dist / speed)."""

    flash_prominence_z: float = 4.0
    """How many std above baseline a brightness transient must be to count as a flash."""

    audio_prominence_z: float = 4.0
    """How many std above baseline an energy transient must be to count as a bang."""

    min_separation_s: float = 0.3
    """Minimum spacing between detected events of the same kind."""

    audio_hop_s: float = 0.01
    """Audio energy-envelope hop (10 ms)."""

    baseline_window_s: float = 2.0
    """Trailing window for the onset baseline: a transient is a rise above the recent
    past, not above an absolute level. This keeps detection robust to sustained glow
    (an explosion's fireball) and to static overlays (text/logos are constant)."""

    auto_check_audio: bool = True
    """If True (and no ``impulse_windows`` are passed), classify the soundtrack with
    ``classify_audio_events`` and only range bangs that coincide with an impulsive sound
    (explosion/gunshot). A music-only or speech-only track is then reported unusable —
    audio is often replaced by music on social media. Needs the 'sound' extra; degrades
    gracefully (with a note) if it is missing."""

    impulse_threshold: float = 0.05
    """Min sound-event score to treat as an impulsive sound when auto-checking audio."""

    music_dominance: float = 0.5
    """Music clip-score at/above which a track with no impulses is called music-dominated."""


@dataclass
class RangedEvent:
    """A paired flash/bang and the implied distance to the source."""

    flash_time_s: float
    bang_time_s: float
    delay_s: float
    distance_m: float
    confidence: float


@dataclass
class FlashToBangResult:
    """Auditable result of :func:`flash_to_bang_range`."""

    video_path: str
    has_audio: bool
    audio_usable: bool = True
    audio_note: str | None = None
    flashes_s: list[float] = field(default_factory=list)
    bangs_s: list[float] = field(default_factory=list)
    events: list[RangedEvent] = field(default_factory=list)
    best: RangedEvent | None = None

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Signal-processing core (IO-free, unit-tested)
# --------------------------------------------------------------------------- #
def _causal_baseline(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average (looks only at the recent past).

    Unlike a centred average, the baseline at an onset reflects the pre-event level
    instead of being inflated by the event itself — so a sustained brightening (an
    explosion's fireball) still produces a clear rise at its onset.
    """
    n = x.size
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(n)
    lo = np.maximum(0, idx - window + 1)
    counts = (idx - lo + 1).astype(float)
    sums = cumsum[idx + 1] - cumsum[lo]
    return sums / counts


def _detect_transients(
    times: np.ndarray,
    signal: np.ndarray,
    prominence_z: float,
    min_separation_s: float,
    baseline_window_s: float = 2.0,
) -> list[tuple[float, float]]:
    """Return [(time, zscore)] of prominent positive onsets (local maxima).

    A transient is a point that rises ``prominence_z`` std above the *recent-past*
    baseline and is a local maximum; detections are thinned to respect
    ``min_separation_s``. The trailing baseline ignores static overlays (text/logos)
    because they are constant, and catches onsets of sustained events.
    """
    times = np.asarray(times, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if signal.size < 3:
        return []

    dt = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    win = max(3, int(round(baseline_window_s / dt))) if dt > 0 else 3
    baseline = _causal_baseline(signal, win)
    residual = signal - baseline
    std = float(residual.std()) or 1.0
    z = residual / std

    min_sep = max(1, int(round(min_separation_s / dt))) if dt > 0 else 1

    candidates = [
        i for i in range(1, signal.size - 1)
        if z[i] >= prominence_z and z[i] >= z[i - 1] and z[i] >= z[i + 1]
    ]
    candidates.sort(key=lambda i: z[i], reverse=True)
    chosen: list[int] = []
    for i in candidates:
        if all(abs(i - j) >= min_sep for j in chosen):
            chosen.append(i)
    chosen.sort()
    return [(float(times[i]), float(z[i])) for i in chosen]


def _is_impulse(label: str) -> bool:
    low = label.lower()
    return any(key in low for key in IMPULSE_KEYS)


def _in_windows(t: float, windows: list[tuple[float, float]]) -> bool:
    """Whether time ``t`` falls within any (start, end) window (small margin)."""
    return any(start - 0.5 <= t <= end + 0.5 for start, end in windows)


def _filter_transients_to_windows(
    transients: list[tuple[float, float]], windows: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    return [tr for tr in transients if _in_windows(tr[0], windows)]


def _assess_audio(
    source_path: str | Path, config: FlashToBangConfig
) -> tuple[list[tuple[float, float]] | None, bool, str | None]:
    """Classify the soundtrack to find impulsive-sound windows and rule out music.

    Returns ``(impulse_windows, usable, note)``. ``impulse_windows`` is ``None`` when the
    check could not run (sound extra missing) — meaning "do not filter bangs".
    """
    try:
        from frame_geolocator.tools.classify_audio_events import (
            ClassifyAudioConfig,
            classify_audio_events,
        )
    except ImportError:
        return None, True, "sound extra not installed: cannot rule out music/added audio"

    result = classify_audio_events(
        source_path, ClassifyAudioConfig(event_threshold=config.impulse_threshold)
    )
    if not result.has_audio:
        return [], False, "no audio"

    windows = [
        (e.start_s, e.end_s)
        for e in result.events
        if _is_impulse(e.label) and e.start_s is not None and e.end_s is not None
    ]
    music_score = max((t.score for t in result.clip_tags if "music" in t.label.lower()),
                      default=0.0)
    impulse_score = max((t.score for t in result.clip_tags if _is_impulse(t.label)),
                        default=0.0)

    if not windows:
        if music_score >= config.music_dominance:
            return [], False, (f"music-dominated audio ({music_score:.2f}) with no "
                               "impulsive sounds → ranging not applicable")
        return [], False, "no impulsive (explosion/gunshot) sound detected → ranging N/A"

    note = None
    if music_score > impulse_score:
        note = (f"music ({music_score:.2f}) louder than impulses ({impulse_score:.2f}); "
                "treat ranging with caution")
    return windows, True, note


def _pair_flash_bang(
    flashes: list[tuple[float, float]],
    bangs: list[tuple[float, float]],
    config: FlashToBangConfig,
) -> list[RangedEvent]:
    """Pair each flash with the earliest bang in the plausible delay window."""
    max_delay = config.max_distance_m / config.speed_of_sound_mps
    events: list[RangedEvent] = []
    for flash_time, flash_z in flashes:
        candidates = [
            (bt, bz) for bt, bz in bangs
            if config.min_delay_s <= bt - flash_time <= max_delay
        ]
        if not candidates:
            continue
        bang_time, bang_z = min(candidates, key=lambda x: x[0])
        delay = bang_time - flash_time
        events.append(
            RangedEvent(
                flash_time_s=round(flash_time, 3),
                bang_time_s=round(bang_time, 3),
                delay_s=round(delay, 3),
                distance_m=round(delay * config.speed_of_sound_mps, 1),
                confidence=round(min(1.0, (flash_z + bang_z) / 20.0), 3),
            )
        )
    return events


# --------------------------------------------------------------------------- #
# IO: build timelines from the media
# --------------------------------------------------------------------------- #
def _brightness_timeline(video_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame mean brightness over time.

    Mean (not a high percentile) because a flash/explosion lights up a large area of an
    otherwise dark frame, while a high percentile saturates on static white overlays
    (caption text, logos) and would miss the event.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    times: list[float] = []
    values: list[float] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            values.append(float(gray.mean()))
            times.append(idx / fps if fps else float(idx))
            idx += 1
    finally:
        cap.release()
    return np.asarray(times), np.asarray(values)


def _audio_envelope(wav_path: Path, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    with wave.open(str(wav_path), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if samples.size == 0:
        return np.asarray([]), np.asarray([])

    hop = max(1, int(sr * hop_s))
    n_frames = samples.size // hop
    if n_frames == 0:
        return np.asarray([]), np.asarray([])
    trimmed = samples[: n_frames * hop].reshape(n_frames, hop)
    energy = np.sqrt((trimmed ** 2).mean(axis=1))  # RMS per hop
    times = np.arange(n_frames) * hop_s
    return times, energy


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def flash_to_bang_range(
    video_path: str | Path,
    config: FlashToBangConfig | None = None,
    impulse_windows: list[tuple[float, float]] | None = None,
) -> FlashToBangResult:
    """Estimate distance(s) to sound sources from flash-to-bang delays in a video.

    Parameters
    ----------
    video_path:
        A video with both picture and sound.
    config:
        Optional :class:`FlashToBangConfig`.
    impulse_windows:
        Optional (start_s, end_s) windows where an impulsive sound (explosion/gunshot)
        occurs; bangs outside them are ignored. If omitted and ``config.auto_check_audio``
        is set, these are derived from ``classify_audio_events`` — which also flags a
        music-only/speech-only (unusable) soundtrack.

    Returns
    -------
    FlashToBangResult
        Detected flashes, bangs, paired ranged events, and the most confident estimate.
        ``has_audio=False`` if there is no audio; ``audio_usable=False`` (with a note) if
        the soundtrack has no impulsive sound to range against (e.g. replaced by music).
    """
    config = config or FlashToBangConfig()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    audio = extract_audio(video_path, video_path.with_suffix(".f2b.wav"))
    if not audio.has_audio or audio.audio_path is None:
        return FlashToBangResult(video_path=str(video_path), has_audio=False,
                                 audio_usable=False, audio_note="no audio stream")

    audio_usable, audio_note = True, None
    if impulse_windows is None and config.auto_check_audio:
        impulse_windows, audio_usable, audio_note = _assess_audio(video_path, config)

    try:
        flash_times, flash_vals = _brightness_timeline(video_path)
        bang_times, bang_energy = _audio_envelope(Path(audio.audio_path), config.audio_hop_s)
    finally:
        Path(audio.audio_path).unlink(missing_ok=True)

    flashes = _detect_transients(
        flash_times, flash_vals, config.flash_prominence_z,
        config.min_separation_s, config.baseline_window_s,
    )
    bangs = _detect_transients(
        bang_times, bang_energy, config.audio_prominence_z,
        config.min_separation_s, config.baseline_window_s,
    )
    # Restrict bangs to impulsive-sound windows when available (filters speech/crowd/music).
    if impulse_windows is not None:
        bangs = _filter_transients_to_windows(bangs, impulse_windows)

    events = _pair_flash_bang(flashes, bangs, config)
    best = max(events, key=lambda e: e.confidence) if events else None

    return FlashToBangResult(
        video_path=str(video_path),
        has_audio=True,
        audio_usable=audio_usable,
        audio_note=audio_note,
        flashes_s=[round(t, 3) for t, _ in flashes],
        bangs_s=[round(t, 3) for t, _ in bangs],
        events=events,
        best=best,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``flash-to-bang-range <video> [--max-distance 5000] [--json out.json]``."""
    parser = argparse.ArgumentParser(description="Acoustic ranging via flash-to-bang.")
    parser.add_argument("video", help="Path to a video with picture and sound.")
    parser.add_argument("--speed", type=float, default=SPEED_OF_SOUND_MPS,
                        help="Speed of sound (m/s).")
    parser.add_argument("--max-distance", type=float, default=5000.0)
    parser.add_argument("--json", help="Optional path to write the full result as JSON.")
    parser.add_argument("--case-file", metavar="PATH",
                        help="Shared case file to merge this result into.")
    args = parser.parse_args(argv)

    config = FlashToBangConfig(speed_of_sound_mps=args.speed, max_distance_m=args.max_distance)
    result = flash_to_bang_range(args.video, config)
    if not result.has_audio:
        print("No audio stream: cannot range.")
        return 0
    if result.audio_note:
        flag = "OK" if result.audio_usable else "UNUSABLE"
        print(f"[audio {flag}] {result.audio_note}")
    print(f"flashes={len(result.flashes_s)} bangs={len(result.bangs_s)} "
          f"events={len(result.events)}")
    for ev in result.events:
        print(f"  flash {ev.flash_time_s:.2f}s -> bang {ev.bang_time_s:.2f}s "
              f"| delay {ev.delay_s:.2f}s ≈ {ev.distance_m:.0f} m (conf {ev.confidence:.2f})")
    if result.best:
        print(f"best: ~{result.best.distance_m:.0f} m to source")
    if args.json:
        result.to_json(args.json)
        print(f"Wrote {args.json}")
    if args.case_file:
        from frame_geolocator.case_file import merge
        merge(args.case_file, "flash_to_bang_range", result)
        print(f"Case file: {args.case_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
