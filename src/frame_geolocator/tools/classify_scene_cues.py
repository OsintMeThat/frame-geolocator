"""Tool B — ``classify_scene_cues``: media-only region priors from the pixels themselves.

The accompanying caption is often absent or useless (a propaganda watermark, a vague
"in California", a name that does not reveal the country). In those cases the *only*
geolocation signal is the **content of the frame**: is it coastal or mountainous, a
tropical or an arid biome, daytime or night. This tool reads those cues directly from
the image with a local zero-shot vision-language model (OpenCLIP).

Design notes
------------
* **Local & free**, no LLM API (see project guardrails). Zero-shot scoring against an
  **explicit, auditable cue taxonomy** (see :data:`CUE_GROUPS`) — never a single
  black-box "this is country X" guess. Each cue is a named, scored hypothesis the
  downstream ``restrict_candidate_region`` can fuse and the analyst can inspect.
* **Honest about the dark.** Empirically, CLIP's *biome/climate* judgement collapses on
  low-light footage (a dark night scene scores "snowy/cold" with high confidence even in
  the tropics). Light-sensitive cue groups are therefore **gated on frame luminance**:
  frames below :attr:`SceneCuesConfig.min_luminance` are excluded from those groups, and
  if none qualify the group is returned ``reliable=False`` rather than a confident wrong
  answer. Light-robust groups (setting, time of day) always run.
* **Multi-frame aggregation.** Given a directory of frames (e.g. the output of
  ``decompose_video``), scores are averaged across frames for a steadier prior than any
  single frame gives.
* The heavy backend is imported lazily and pluggable with automatic fallback, exactly
  like ``read_text_ocr``; tests inject a deterministic fake backend.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

DEFAULT_BACKENDS = ["openclip"]
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# CLIP softmax temperature. OpenCLIP's trained logit scale is ~100; we use a fixed value
# so scores are comparable across runs and the backend stays a thin embedder.
_LOGIT_SCALE = 100.0


# --------------------------------------------------------------------------- #
# Cue taxonomy — explicit and auditable. Edit here to extend; never a country.
# --------------------------------------------------------------------------- #
# Each group maps a short, stable cue *label* to the natural-language *prompt* shown to
# CLIP. ``light_sensitive`` groups are skipped on under-exposed frames (see module docs).
CUE_GROUPS: dict[str, dict] = {
    "setting": {
        "light_sensitive": False,
        "cues": {
            "coastal": "a coastal scene with the ocean, sea or cliffs",
            "urban": "a dense urban street with buildings",
            "aerial_city": "an aerial view looking down over a city",
            "mountain": "a mountainous landscape with peaks or ridges",
            "desert": "a barren sandy desert landscape",
            "rural": "open rural countryside or farmland",
            "forest": "dense forest or jungle",
        },
    },
    "biome": {
        "light_sensitive": True,
        "cues": {
            "tropical": "a hot humid tropical environment with lush green vegetation",
            "arid": "an arid desert climate, dry, sandy and barren",
            "temperate": "a temperate climate with green deciduous trees",
            "mediterranean": "a dry mediterranean climate with scrub and dry hills",
            "savanna": "a dry savanna with sparse trees and dry grass",
            "continental_cold": "a cold snowy climate with snow or bare winter trees",
        },
    },
    "time_of_day": {
        "light_sensitive": False,
        "cues": {
            "day": "an outdoor photo taken in daylight",
            "night": "an outdoor photo taken at night in the dark",
            "dusk_dawn": "an outdoor photo taken at dusk or dawn, in twilight",
        },
    },
}


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class SceneCuesConfig:
    """Configuration for :func:`classify_scene_cues`."""

    backends: list[str] = field(default_factory=lambda: list(DEFAULT_BACKENDS))
    """Backend names in priority order; the first available one is used."""

    model: str = DEFAULT_MODEL
    pretrained: str = DEFAULT_PRETRAINED

    groups: list[str] | None = None
    """Subset of :data:`CUE_GROUPS` to evaluate (default: all)."""

    max_frames: int = 12
    """When given a directory, evenly sample at most this many frames."""

    min_luminance: float = 70.0
    """Mean 0-255 luminance below which a frame is 'low light' and excluded from
    light-sensitive cue groups (biome)."""

    top_k: int = 3
    """How many scored cues to keep per group in the output."""


@dataclass
class CueScore:
    """A single scored cue hypothesis."""

    label: str
    score: float


@dataclass
class GroupCues:
    """Aggregated cues for one taxonomy group."""

    group: str
    reliable: bool
    top: str | None
    scores: list[CueScore]
    frames_used: int
    note: str | None = None


@dataclass
class SceneCuesResult:
    """Auditable result of :func:`classify_scene_cues`."""

    source: str
    backend_used: str
    model: str
    n_frames: int
    mean_luminance: float
    groups: list[GroupCues] = field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class SceneBackend(Protocol):
    """Contract every scene backend must satisfy: a thin image/text embedder."""

    name: str

    def available(self) -> bool: ...
    def embed_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a single L2-normalized image embedding."""
        ...

    def embed_texts(self, prompts: tuple[str, ...]) -> np.ndarray:
        """Return an (N, D) array of L2-normalized text embeddings."""
        ...


class OpenClipBackend:
    """OpenCLIP zero-shot backend (PyTorch, CPU-friendly, multilingual prompts)."""

    name = "openclip"

    def __init__(self, model: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED) -> None:
        self._model_name = model
        self._pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_cache: dict[tuple[str, ...], np.ndarray] = {}

    def available(self) -> bool:
        return (
            importlib.util.find_spec("open_clip") is not None
            and importlib.util.find_spec("torch") is not None
        )

    def _load(self):
        if self._model is not None:
            return
        import open_clip

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self._model_name)

    def embed_image(self, image_rgb: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        self._load()
        tensor = self._preprocess(Image.fromarray(image_rgb)).unsqueeze(0)
        with torch.no_grad():
            feats = self._model.encode_image(tensor)
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0]

    def embed_texts(self, prompts: tuple[str, ...]) -> np.ndarray:
        import torch

        if prompts in self._text_cache:
            return self._text_cache[prompts]
        self._load()
        with torch.no_grad():
            feats = self._model.encode_text(self._tokenizer(list(prompts)))
            feats /= feats.norm(dim=-1, keepdim=True)
        arr = feats.cpu().numpy()
        self._text_cache[prompts] = arr
        return arr


# Backend registry. Tests may inject fakes here.
_BACKENDS: dict[str, SceneBackend] = {
    "openclip": OpenClipBackend(),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_frames(path: Path, max_frames: int) -> list[Path]:
    """Return the image paths to score: the file itself, or an even sample of a dir."""
    if path.is_dir():
        frames = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS and not p.name.startswith("contact_sheet")
        )
        if not frames:
            raise FileNotFoundError(f"No images found in directory: {path}")
        if len(frames) > max_frames:
            idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
            frames = [frames[i] for i in sorted(set(idx))]
        return frames
    return [path]


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def classify_scene_cues(
    source: str | Path, config: SceneCuesConfig | None = None
) -> SceneCuesResult:
    """Score a frame (or a directory of frames) against the auditable cue taxonomy.

    Parameters
    ----------
    source:
        An image file, or a directory of frames (e.g. ``decompose_video`` output).
    config:
        Optional :class:`SceneCuesConfig`.

    Returns
    -------
    SceneCuesResult
        Per-group scored cues, with light-sensitive groups marked ``reliable=False``
        on under-exposed footage instead of returning a confident wrong answer.

    Notes
    -----
    The result is a **prior to verify**, not the final answer. Cues restrict *where to
    look*; the location must still be confirmed from the media's own content.
    """
    config = config or SceneCuesConfig()
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    frames = _resolve_frames(source, config.max_frames)

    # Load frames as RGB and measure per-frame luminance up front.
    images_rgb: list[np.ndarray] = []
    luminances: list[float] = []
    for fp in frames:
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        images_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        luminances.append(float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()))
    if not images_rgb:
        raise RuntimeError(f"Could not read any image from: {source}")

    # Pick the first available backend.
    backend: SceneBackend | None = None
    errors: list[str] = []
    for name in config.backends:
        cand = _BACKENDS.get(name)
        if cand is None:
            errors.append(f"{name}: unknown backend")
        elif not cand.available():
            errors.append(f"{name}: not installed/available")
        else:
            backend = cand
            break
    if backend is None:
        raise RuntimeError(
            "No usable scene backend. Install the 'vision' extra "
            "(pip install -e '.[vision]'). Tried: " + "; ".join(errors)
        )

    # Embed every frame once, reuse across all groups.
    image_embs = [backend.embed_image(img) for img in images_rgb]

    group_names = config.groups or list(CUE_GROUPS)
    out_groups: list[GroupCues] = []
    for gname in group_names:
        spec = CUE_GROUPS.get(gname)
        if spec is None:
            continue
        labels = list(spec["cues"].keys())
        prompts = tuple(spec["cues"].values())
        text_embs = backend.embed_texts(prompts)

        # Select usable frames: all, unless the group is light-sensitive.
        if spec["light_sensitive"]:
            usable = [i for i, lum in enumerate(luminances) if lum >= config.min_luminance]
        else:
            usable = list(range(len(image_embs)))

        reliable = bool(usable)
        used = usable if usable else list(range(len(image_embs)))  # fall back for transparency
        # Average the per-frame softmax distributions across used frames.
        dist = np.mean(
            [_softmax(_LOGIT_SCALE * (image_embs[i] @ text_embs.T)) for i in used], axis=0
        )
        order = np.argsort(dist)[::-1][: config.top_k]
        scores = [CueScore(label=labels[j], score=round(float(dist[j]), 3)) for j in order]
        note = None if reliable else "low light: biome unreliable, scores shown for audit only"
        out_groups.append(
            GroupCues(
                group=gname,
                reliable=reliable,
                top=labels[int(order[0])] if reliable else None,
                scores=scores,
                frames_used=len(used),
                note=note,
            )
        )

    return SceneCuesResult(
        source=str(source),
        backend_used=backend.name,
        model=config.model,
        n_frames=len(images_rgb),
        mean_luminance=round(float(np.mean(luminances)), 1),
        groups=out_groups,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """CLI: ``classify-scene-cues <image|frames_dir> [--json out.json]``."""
    parser = argparse.ArgumentParser(
        description="Media-only region priors (setting, biome, time-of-day) via zero-shot CLIP."
    )
    parser.add_argument("source", help="Image file or directory of frames.")
    parser.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS,
                        help="Backend priority order (default: openclip).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenCLIP model name.")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED, help="OpenCLIP weights tag.")
    parser.add_argument("--groups", nargs="+", default=None, metavar="GROUP",
                        help=f"Subset of cue groups (default: all). Choices: {list(CUE_GROUPS)}.")
    parser.add_argument("--max-frames", type=int, default=12,
                        help="Max frames to sample from a directory (default: 12).")
    parser.add_argument("--min-luminance", type=float, default=70.0,
                        help="Frames dimmer than this are excluded from biome cues (default: 70).")
    parser.add_argument("--json", metavar="PATH", help="Write the full result as JSON.")
    parser.add_argument("--case-file", metavar="PATH",
                        help="Shared case file to merge this result into.")
    args = parser.parse_args(argv)

    config = SceneCuesConfig(
        backends=args.backends,
        model=args.model,
        pretrained=args.pretrained,
        groups=args.groups,
        max_frames=args.max_frames,
        min_luminance=args.min_luminance,
    )
    result = classify_scene_cues(args.source, config)

    print(f"[backend: {result.backend_used}/{result.model}] "
          f"{result.n_frames} frame(s), mean luminance {result.mean_luminance}")
    for g in result.groups:
        flag = "" if g.reliable else "  (UNRELIABLE — low light)"
        top = g.top if g.top is not None else "—"
        print(f"  {g.group:12s}: {top}{flag}")
        for cue in g.scores:
            print(f"      {cue.score:.2f}  {cue.label}")

    if args.json:
        result.to_json(args.json)
        print(f"Wrote {args.json}")
    if args.case_file:
        from frame_geolocator.case_file import merge
        merge(args.case_file, "classify_scene_cues", result)
        print(f"Case file: {args.case_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
