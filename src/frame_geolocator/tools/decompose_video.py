"""Tool A.1 — ``decompose_video``: intelligent video decomposition.

This is the entry point of the whole pipeline. It turns a (possibly degraded) video
into a small set of the most informative frames, optionally normalized for brightness,
plus a *contact sheet* (a stitched montage) so a downstream LLM can reason over many
frames at once.

Design notes
------------
* Pure numeric tool (OpenCV / ffmpeg backend). No LLM, no network. This makes it cheap,
  deterministic, and testable in isolation against a synthetic clip.
* Selection strategy (v1, deliberately simple and robust):
    1. Sample the video at a fixed stride derived from ``sample_fps``.
    2. For each sampled frame compute a sharpness score (variance of the Laplacian),
       a perceptual hash (dHash) and a colour histogram.
    3. Flag scene cuts via histogram correlation between consecutive samples (recorded
       as a signal; not yet used to gate selection).
    4. Drop frames below ``min_sharpness`` and near-duplicates (dHash Hamming distance).
    5. Spread the survivors across the timeline into ``target_frames`` temporal bins and
       keep the sharpest frame per bin.
    6. Optionally CLAHE-normalize brightness/contrast before writing.
* Everything the tool decides is reported in :class:`DecomposeResult` so the choice of
  frames is auditable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SAMPLE_FPS = 2.0


@dataclass
class DecomposeConfig:
    """Configuration for :func:`decompose_video`."""

    target_frames: int = 24
    """Maximum number of frames to keep (also the number of temporal bins)."""

    sample_fps: float = DEFAULT_SAMPLE_FPS
    """How many frames per second to inspect (not to keep). Lower = faster, coarser."""

    scene_threshold: float = 0.35
    """Histogram-correlation drop above which a scene cut is flagged (0..1)."""

    min_sharpness: float = 5.0
    """Discard frames whose Laplacian variance is below this (very blurry/flat)."""

    dedup_hamming: int = 6
    """dHash Hamming distance at/below which two frames are treated as duplicates."""

    normalize: bool = True
    """Apply CLAHE brightness/contrast normalization to written frames."""

    contact_sheet: bool = True
    """Also write a stitched montage of the selected frames."""

    contact_cols: int = 4
    """Number of columns in the contact sheet."""

    thumb_width: int = 320
    """Thumbnail width (px) used both for the contact sheet and dedup hashing scale."""

    jpeg_quality: int = 92
    """JPEG quality for written frames (1..100)."""


@dataclass
class SelectedFrame:
    """A frame kept by the tool."""

    index: int
    timestamp_s: float
    sharpness: float
    is_scene_cut: bool
    path: str


@dataclass
class DecomposeResult:
    """Auditable result of :func:`decompose_video`."""

    video_path: str
    out_dir: str
    fps: float
    frame_count: int
    duration_s: float
    sampled: int
    selected: list[SelectedFrame] = field(default_factory=list)
    contact_sheet_path: str | None = None

    def to_json(self, path: str | Path) -> None:
        """Write this result as JSON next to the extracted frames."""
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def _sharpness(gray: np.ndarray) -> float:
    """Focus measure: variance of the Laplacian. Higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    """Difference hash (perceptual). Robust to small changes, good for dedup."""
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for value in diff.flatten():
        bits = (bits << 1) | int(value)
    return bits


def _hamming(a: int, b: int) -> int:
    """Number of differing bits between two integer hashes."""
    return bin(a ^ b).count("1")


def _histogram(bgr: np.ndarray) -> np.ndarray:
    """Normalized HSV histogram used for scene-cut detection."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _normalize_brightness(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel to recover detail in dark or washed-out frames."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    merged = cv2.merge((clahe.apply(lightness), a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------- #
# Internal candidate type
# --------------------------------------------------------------------------- #
@dataclass
class _Candidate:
    index: int
    timestamp_s: float
    sharpness: float
    dhash: int
    is_scene_cut: bool
    frame: np.ndarray


def _collect_candidates(
    cap: cv2.VideoCapture, fps: float, stride: int, config: DecomposeConfig
) -> list[_Candidate]:
    """Walk the video at ``stride`` and gather per-sample measurements."""
    candidates: list[_Candidate] = []
    prev_hist: np.ndarray | None = None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = _histogram(frame)
            is_cut = False
            if prev_hist is not None:
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                is_cut = (1.0 - correlation) >= config.scene_threshold
            prev_hist = hist
            candidates.append(
                _Candidate(
                    index=index,
                    timestamp_s=index / fps if fps else float(index),
                    sharpness=_sharpness(gray),
                    dhash=_dhash(gray),
                    is_scene_cut=is_cut,
                    frame=frame,
                )
            )
        index += 1
    return candidates


def _select(candidates: list[_Candidate], config: DecomposeConfig) -> list[_Candidate]:
    """Filter blur/dups, then keep the sharpest frame per temporal bin."""
    usable = [c for c in candidates if c.sharpness >= config.min_sharpness]
    if not usable:
        return []

    # Deduplicate near-identical frames, keeping the sharper of each pair.
    kept: list[_Candidate] = []
    for cand in sorted(usable, key=lambda c: c.sharpness, reverse=True):
        if any(_hamming(cand.dhash, k.dhash) <= config.dedup_hamming for k in kept):
            continue
        kept.append(cand)

    if len(kept) <= config.target_frames:
        return sorted(kept, key=lambda c: c.index)

    # Spread across the timeline: bin by time, keep the sharpest per bin.
    times = [c.timestamp_s for c in kept]
    t_min, t_max = min(times), max(times)
    span = (t_max - t_min) or 1.0
    bins: dict[int, _Candidate] = {}
    for cand in kept:
        b = min(
            config.target_frames - 1,
            int((cand.timestamp_s - t_min) / span * config.target_frames),
        )
        if b not in bins or cand.sharpness > bins[b].sharpness:
            bins[b] = cand
    return sorted(bins.values(), key=lambda c: c.index)


def _build_contact_sheet(
    frames: list[tuple[np.ndarray, str]], cols: int, thumb_width: int
) -> np.ndarray:
    """Tile labelled thumbnails into a single montage image."""
    thumbs = []
    for img, label in frames:
        h, w = img.shape[:2]
        tw = thumb_width
        th = max(1, int(h * tw / w))
        thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        bar = np.zeros((20, tw, 3), dtype=np.uint8)
        cv2.putText(
            bar, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
        thumbs.append(np.vstack([bar, thumb]))

    cell_h = max(t.shape[0] for t in thumbs)
    cell_w = max(t.shape[1] for t in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        h, w = thumb.shape[:2]
        sheet[r * cell_h : r * cell_h + h, c * cell_w : c * cell_w + w] = thumb
    return sheet


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def decompose_video(
    video_path: str | Path,
    out_dir: str | Path,
    config: DecomposeConfig | None = None,
) -> DecomposeResult:
    """Decompose a video into a small set of informative, optionally enhanced frames.

    Parameters
    ----------
    video_path:
        Path to the source video (any format the OpenCV/ffmpeg backend can read).
    out_dir:
        Directory to write the selected frames, contact sheet and ``result.json`` into.
        Created if missing.
    config:
        Optional :class:`DecomposeConfig`; sensible defaults are used otherwise.

    Returns
    -------
    DecomposeResult
        An auditable record of what was sampled, selected and written.
    """
    config = config or DecomposeConfig()
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video (codec/backend issue?): {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = frame_count / fps if fps else 0.0
        stride = max(1, int(round(fps / config.sample_fps))) if fps else 1

        candidates = _collect_candidates(cap, fps, stride, config)
        chosen = _select(candidates, config)
    finally:
        cap.release()

    selected: list[SelectedFrame] = []
    sheet_inputs: list[tuple[np.ndarray, str]] = []
    for n, cand in enumerate(chosen):
        out_frame = _normalize_brightness(cand.frame) if config.normalize else cand.frame
        name = f"frame_{n:03d}_t{cand.timestamp_s:07.2f}.jpg"
        path = out_dir / name
        cv2.imwrite(
            str(path), out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality]
        )
        selected.append(
            SelectedFrame(
                index=cand.index,
                timestamp_s=round(cand.timestamp_s, 3),
                sharpness=round(cand.sharpness, 2),
                is_scene_cut=cand.is_scene_cut,
                path=str(path),
            )
        )
        sheet_inputs.append((out_frame, f"t={cand.timestamp_s:.1f}s sharp={cand.sharpness:.0f}"))

    contact_sheet_path: str | None = None
    if config.contact_sheet and sheet_inputs:
        sheet = _build_contact_sheet(sheet_inputs, config.contact_cols, config.thumb_width)
        contact_sheet_path = str(out_dir / "contact_sheet.jpg")
        cv2.imwrite(contact_sheet_path, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality])

    result = DecomposeResult(
        video_path=str(video_path),
        out_dir=str(out_dir),
        fps=round(fps, 3),
        frame_count=frame_count,
        duration_s=round(duration_s, 3),
        sampled=len(candidates),
        selected=selected,
        contact_sheet_path=contact_sheet_path,
    )
    result.to_json(out_dir / "result.json")
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI: ``decompose-video <video> --out <dir> [options]``."""
    parser = argparse.ArgumentParser(description="Intelligent video decomposition.")
    parser.add_argument("video", help="Path to the source video.")
    parser.add_argument("--out", required=True, help="Output directory for frames.")
    parser.add_argument("--frames", type=int, default=DecomposeConfig.target_frames,
                        help="Max frames to keep.")
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS,
                        help="Frames per second to inspect.")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable CLAHE brightness normalization.")
    parser.add_argument("--no-contact-sheet", action="store_true",
                        help="Do not write a contact sheet.")
    parser.add_argument("--case-file", metavar="PATH",
                        help="Shared case file to merge this result into.")
    args = parser.parse_args(argv)

    config = DecomposeConfig(
        target_frames=args.frames,
        sample_fps=args.sample_fps,
        normalize=not args.no_normalize,
        contact_sheet=not args.no_contact_sheet,
    )
    result = decompose_video(args.video, args.out, config)
    print(
        f"Sampled {result.sampled} frames, kept {len(result.selected)} "
        f"({result.duration_s:.1f}s @ {result.fps:.1f}fps) -> {result.out_dir}"
    )
    if result.contact_sheet_path:
        print(f"Contact sheet: {result.contact_sheet_path}")
    if args.case_file:
        from frame_geolocator.case_file import merge
        merge(args.case_file, "decompose_video", result)
        print(f"Case file: {args.case_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
