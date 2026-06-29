"""Tool G.27 — ``extract_audio``: demux a video's waveform.

Foundation of the audio branch (mirrors ``decompose_video`` for frames). A media's
soundtrack carries independent geolocation evidence: spoken language/place names
(``identify_spoken_language``) and visual-vs-acoustic timing (``flash_to_bang_range``).

Design notes
------------
* **Local & free, no system dependency**: uses the static ffmpeg binary bundled by the
  ``imageio-ffmpeg`` pip package (falls back to a system ``ffmpeg`` if present).
* Output defaults to **16 kHz mono WAV**, which is what Whisper-family ASR expects.
* Detects videos with no audio stream and reports ``has_audio=False`` instead of
  raising, so the orchestrator can simply skip the audio branch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_SAMPLE_RATE = 16000


@dataclass
class ExtractAudioConfig:
    """Configuration for :func:`extract_audio`."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1  # mono is best for ASR


@dataclass
class ExtractAudioResult:
    """Auditable result of :func:`extract_audio`."""

    video_path: str
    audio_path: str | None
    has_audio: bool
    sample_rate: int
    channels: int
    duration_s: float

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _ffmpeg_exe() -> str:
    """Locate an ffmpeg binary: bundled (imageio-ffmpeg) first, then system."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 — fall back to a system binary
        from shutil import which

        exe = which("ffmpeg")
        if exe:
            return exe
        raise RuntimeError(
            "No ffmpeg available. Install the 'audio' extra: pip install -e '.[audio]'"
        ) from None


def _wav_metadata(path: Path) -> tuple[int, int, float]:
    """Return (sample_rate, channels, duration_s) of a WAV file."""
    with wave.open(str(path), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        frames = wav.getnframes()
        duration = frames / sr if sr else 0.0
    return sr, channels, duration


def extract_audio(
    video_path: str | Path,
    out_path: str | Path | None = None,
    config: ExtractAudioConfig | None = None,
) -> ExtractAudioResult:
    """Extract the audio track of a video to a WAV file.

    Parameters
    ----------
    video_path:
        Source video.
    out_path:
        Destination ``.wav``. Defaults to the video path with a ``.wav`` suffix.
    config:
        Optional :class:`ExtractAudioConfig`.

    Returns
    -------
    ExtractAudioResult
        ``has_audio=False`` (and ``audio_path=None``) if the video has no audio stream.
    """
    config = config or ExtractAudioConfig()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_path = Path(out_path) if out_path else video_path.with_suffix(".wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _ffmpeg_exe(), "-y", "-i", str(video_path),
        "-vn", "-ac", str(config.channels), "-ar", str(config.sample_rate),
        "-f", "wav", str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603 — output suppressed

    has_audio = out_path.exists() and out_path.stat().st_size > 44  # 44 = WAV header
    if not has_audio:
        if out_path.exists():
            out_path.unlink()
        return ExtractAudioResult(
            video_path=str(video_path), audio_path=None, has_audio=False,
            sample_rate=config.sample_rate, channels=config.channels, duration_s=0.0,
        )

    sr, channels, duration = _wav_metadata(out_path)
    return ExtractAudioResult(
        video_path=str(video_path), audio_path=str(out_path), has_audio=True,
        sample_rate=sr, channels=channels, duration_s=round(duration, 3),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``extract-audio <video> [--out out.wav] [--sample-rate 16000]``."""
    parser = argparse.ArgumentParser(description="Extract a video's audio track to WAV.")
    parser.add_argument("video", help="Path to the source video.")
    parser.add_argument("--out", help="Output WAV path (default: alongside the video).")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--case-file", metavar="PATH",
                        help="Shared case file to merge this result into.")
    args = parser.parse_args(argv)

    result = extract_audio(
        args.video, args.out, ExtractAudioConfig(args.sample_rate, args.channels)
    )
    if result.has_audio:
        print(f"Audio: {result.audio_path} "
              f"({result.duration_s:.1f}s, {result.sample_rate}Hz, {result.channels}ch)")
    else:
        print("No audio stream in this video.")
    if args.case_file:
        from frame_geolocator.case_file import merge
        merge(args.case_file, "extract_audio", result)
        print(f"Case file: {args.case_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
