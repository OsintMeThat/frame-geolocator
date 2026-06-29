"""Tool D.15 — ``extract_skyline_profile``: the observed ridge silhouette as an angular profile.

When a frame shows mountains, the horizon/ridge silhouette against the sky is a strong,
camera-independent geolocation signature (the "two bumps and a tower" of a skyline). This
tool extracts it deterministically: it segments sky from ground and, for each image
column, records where the boundary sits — yielding a 1-D **skyline profile** that
``synthesize_skyline`` / ``match_skyline`` will later correlate against DEM-rendered
panoramas.

Design notes
------------
* **Local & deterministic** (`[NUM]`): OpenCV + numpy only, no model, no network. Sky is
  detected with an **adaptive** rule referenced to the top strip of the frame (usually
  sky), so it is not tuned to fixed colour constants.
* **Honest coverage.** Down-looking or sky-less frames (aerial nadir, indoor) yield few
  valid columns; the tool reports a ``coverage`` fraction and a ``usable`` flag rather
  than fabricating a horizon.
* **FOV-agnostic.** The profile is returned in normalized image height; if a vertical FOV
  is supplied it is also converted to elevation **degrees**. Absolute angles wait on
  ``estimate_fov`` — this tool stays usable without it.
* Optional **overlay** image (skyline drawn on the frame) for visual audit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SkylineConfig:
    """Configuration for :func:`extract_skyline_profile`."""

    top_ref_frac: float = 0.08
    """Fraction of the top of the frame sampled to learn the 'sky' reference."""

    smooth_window: int = 9
    """Median-smoothing window (odd, in columns) applied to the horizon line."""

    min_coverage: float = 0.25
    """Below this fraction of columns with a detected sky boundary, the profile is
    flagged ``usable=False`` (e.g. a down-looking aerial frame with no sky)."""

    fov_v_deg: float | None = None
    """Optional vertical field of view; if given, the profile is also in degrees."""


@dataclass
class SkylineProfile:
    """Auditable result of :func:`extract_skyline_profile`."""

    image_path: str
    width: int
    height: int
    coverage: float                 # fraction of columns with a sky/ground boundary
    usable: bool
    mean_horizon_frac: float        # average boundary height, 0 (top) .. 1 (bottom)
    roughness: float                # std of the normalized profile — flat vs jagged
    horizon_norm: list[float] = field(default_factory=list)  # per column, -1.0 if no sky
    horizon_deg: list[float] | None = None  # elevation angle per column, if fov given

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# Sky / ground segmentation
# --------------------------------------------------------------------------- #
def _sky_mask(bgr: np.ndarray, config: SkylineConfig) -> np.ndarray:
    """Boolean mask of sky-like pixels, adaptive to the frame's own top strip.

    Sky in OSINT mountain footage is either bright + low-saturation (overcast / haze /
    cloud) or blue. We learn brightness/saturation references from the top of the frame
    and accept pixels that are sky-like by either route.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    top_n = max(1, int(bgr.shape[0] * config.top_ref_frac))
    s_ref = float(np.median(s[:top_n]))
    v_ref = float(np.median(v[:top_n]))

    # Bright + relatively unsaturated (clouds, haze, pale sky).
    bright_unsat = (v >= max(110.0, v_ref - 60)) & (s <= max(70.0, s_ref + 55))
    # Blue sky (OpenCV hue 0..180; blue ≈ 100..140), reasonably bright.
    blue = (h >= 100) & (h <= 140) & (v >= 90)
    mask = bright_unsat | blue

    # Clean speckle so the column scan finds a stable boundary.
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def _column_horizon(mask: np.ndarray) -> np.ndarray:
    """Per-column boundary row: the first non-sky row counting from the top.

    A column whose top pixel is already non-sky (foreground occludes the sky) is marked
    invalid with -1, rather than guessing.
    """
    h, w = mask.shape
    horizon = np.full(w, -1, dtype=np.float64)
    for x in range(w):
        col = mask[:, x]
        if not col[0]:
            continue  # no sky at the top of this column → invalid
        # first False after the initial True run
        idx = np.argmax(~col)
        # argmax returns 0 if all True; handle the all-sky column explicitly
        horizon[x] = h - 1 if col.all() else idx
    return horizon


def _smooth(horizon: np.ndarray, window: int) -> np.ndarray:
    """Median-smooth valid entries, leaving -1 (invalid) columns untouched."""
    if window < 3:
        return horizon
    if window % 2 == 0:
        window += 1
    out = horizon.copy()
    half = window // 2
    valid = horizon >= 0
    for x in range(len(horizon)):
        if not valid[x]:
            continue
        lo, hi = max(0, x - half), min(len(horizon), x + half + 1)
        neighbourhood = horizon[lo:hi]
        good = neighbourhood[neighbourhood >= 0]
        if good.size:
            out[x] = float(np.median(good))
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_skyline_profile(
    image_path: str | Path, config: SkylineConfig | None = None
) -> SkylineProfile:
    """Extract the sky/ground boundary of an image as a normalized skyline profile.

    Parameters
    ----------
    image_path:
        Path to a frame (e.g. ``decompose_video`` output).
    config:
        Optional :class:`SkylineConfig`.

    Returns
    -------
    SkylineProfile
        Per-column boundary (normalized 0..1, -1 where no sky), coverage and a
        ``usable`` flag, plus elevation degrees if a vertical FOV was supplied.
    """
    config = config or SkylineConfig()
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = bgr.shape[:2]
    mask = _sky_mask(bgr, config)
    horizon = _smooth(_column_horizon(mask), config.smooth_window)

    valid = horizon >= 0
    coverage = float(valid.mean())
    norm = np.where(valid, horizon / max(h - 1, 1), -1.0)
    valid_norm = norm[valid]
    mean_frac = float(valid_norm.mean()) if valid_norm.size else 0.0
    roughness = float(valid_norm.std()) if valid_norm.size else 0.0

    horizon_deg = None
    if config.fov_v_deg is not None:
        # Elevation angle: 0 at image centre, positive upward.
        deg = (0.5 - norm) * config.fov_v_deg
        horizon_deg = [
            round(float(d), 3) if v_ else -999.0 for d, v_ in zip(deg, valid, strict=True)
        ]

    return SkylineProfile(
        image_path=str(image_path),
        width=w,
        height=h,
        coverage=round(coverage, 3),
        usable=coverage >= config.min_coverage,
        mean_horizon_frac=round(mean_frac, 4),
        roughness=round(roughness, 4),
        horizon_norm=[round(float(x), 4) for x in norm],
        horizon_deg=horizon_deg,
    )


def write_overlay(image_path: str | Path, profile: SkylineProfile, out_path: str | Path) -> None:
    """Draw the detected skyline on the frame and save it, for visual audit."""
    bgr = cv2.imread(str(image_path))
    pts = [
        (x, int(round(n * (profile.height - 1))))
        for x, n in enumerate(profile.horizon_norm)
        if n >= 0
    ]
    for x, y in pts:
        cv2.circle(bgr, (x, y), 1, (0, 0, 255), -1)
    cv2.imwrite(str(out_path), bgr)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """CLI: ``extract-skyline-profile <image> [--overlay out.jpg] [--json out.json]``."""
    parser = argparse.ArgumentParser(
        description="Extract the sky/ground horizon silhouette of a frame as a 1-D profile."
    )
    parser.add_argument("image", help="Path to the frame.")
    parser.add_argument("--fov-v-deg", type=float, default=None,
                        help="Vertical field of view (deg) to also output elevation angles.")
    parser.add_argument("--smooth", type=int, default=9, help="Median smoothing window.")
    parser.add_argument("--min-coverage", type=float, default=0.25,
                        help="Coverage below which the profile is flagged unusable.")
    parser.add_argument("--overlay", metavar="PATH", help="Write an overlay image for audit.")
    parser.add_argument("--json", metavar="PATH", help="Write the full result as JSON.")
    parser.add_argument("--case-file", metavar="PATH",
                        help="Shared case file to merge this result into.")
    args = parser.parse_args(argv)

    config = SkylineConfig(
        smooth_window=args.smooth, min_coverage=args.min_coverage, fov_v_deg=args.fov_v_deg
    )
    profile = extract_skyline_profile(args.image, config)

    flag = "usable" if profile.usable else "LOW COVERAGE"
    print(f"[{profile.width}x{profile.height}] coverage {profile.coverage:.0%} ({flag}), "
          f"mean horizon {profile.mean_horizon_frac:.2f}, roughness {profile.roughness:.3f}")

    if args.overlay:
        write_overlay(args.image, profile, args.overlay)
        print(f"Overlay: {args.overlay}")
    if args.json:
        profile.to_json(args.json)
        print(f"Wrote {args.json}")
    if args.case_file:
        from frame_geolocator.case_file import merge
        merge(args.case_file, "extract_skyline_profile", profile)
        print(f"Case file: {args.case_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
