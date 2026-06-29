"""Tool C.13 — ``restrict_candidate_region``: emit raw signals for region restriction.

This tool does **not** decide where to look. It emits **all available signals** —
toponyms, language, biome — with clean metadata. The orchestrator (Claude) reads these
signals and reasons over them:

* Which toponym to trust? Which is noise?
* If language and biome conflict, which wins?
* Should we expand the region or lower confidence?

Inputs are priors from the case file:
* toponyms + caption language + event  (``parse_context_note``),
* setting / biome cues + reliability   (``classify_scene_cues``).

The output is a **list of raw signals** (toponyms, language, biome) with their
geographic bounds and confidence metadata. Zero fusion logic. Claude decides.

Design discipline
------------------
* **Emit all signals, no filtering.** A mis-geocoded toponym, a noisy title-case extraction,
  an unreliable biome from a night frame — all are emitted with metadata so the orchestrator
  can decide whether to trust each.
* **Pure geometry, no judgment.** Bbox math, radius calculation, reference gazetteer —
  these are tested independently. No heuristic confidence, no signal ranking.
* **Metadata is explicit.** Each signal carries its source and any reliability flag:
  geocoding confidence, biome reliability, feature type, etc.
* **Local & free**: a deliberately coarse built-in gazetteer (no network, no key).
  Extended only as validation demands, not exhaustive.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Gazetteer — coarse named bounding boxes as (west, south, east, north) in degrees.
# Deliberately approximate; extend as validation requires. NOT exhaustive.
# --------------------------------------------------------------------------- #
BBox = tuple[float, float, float, float]

GAZETTEER: dict[str, BBox] = {
    "world": (-180.0, -90.0, 180.0, 90.0),
    # Europe
    "Europe": (-25.0, 34.0, 45.0, 72.0),
    "Eastern Europe": (15.0, 41.0, 60.0, 60.0),
    "Western Europe": (-10.0, 36.0, 20.0, 60.0),
    "France": (-5.0, 41.0, 10.0, 51.0),
    "Belgium": (2.0, 49.0, 7.0, 52.0),
    "Spain": (-10.0, 36.0, 4.0, 44.0),
    # Russia / post-Soviet
    "Russia": (27.0, 41.0, 180.0, 78.0),
    "Russia (European)": (27.0, 43.0, 60.0, 70.0),
    "Belarus": (23.0, 51.0, 33.0, 57.0),
    "Ukraine": (22.0, 44.0, 40.0, 53.0),
    "Kazakhstan": (46.0, 40.0, 87.0, 55.0),
    "Caucasus": (40.0, 38.0, 50.0, 45.0),
    # Francophone Africa
    "Niger": (0.0, 11.0, 16.0, 24.0),
    "Mali": (-12.0, 10.0, 5.0, 25.0),
    "Senegal": (-18.0, 12.0, -11.0, 17.0),
    "Côte d'Ivoire": (-9.0, 4.0, -2.0, 11.0),
    "DR Congo": (12.0, -14.0, 32.0, 6.0),
    # Latin America
    "Mexico": (-118.0, 14.0, -86.0, 33.0),
    "Venezuela": (-73.0, 0.0, -59.0, 13.0),
    "Colombia": (-79.0, -4.0, -67.0, 13.0),
    "Peru": (-82.0, -19.0, -68.0, 0.0),
    "Argentina": (-74.0, -55.0, -53.0, -21.0),
    "Chile (central)": (-74.0, -38.0, -70.0, -29.0),
    "Brazil": (-74.0, -34.0, -34.0, 6.0),
    # North America
    "USA": (-125.0, 24.0, -66.0, 49.0),
    "California": (-125.0, 32.0, -114.0, 42.0),
}

# Caption/OCR language (ISO 639-1) -> set of gazetteer regions where it is spoken.
# English is intentionally omitted: it is too widespread to restrict anything.
LANGUAGE_REGIONS: dict[str, list[str]] = {
    "ru": ["Russia", "Belarus", "Ukraine", "Kazakhstan", "Caucasus"],
    "fr": ["France", "Belgium", "Niger", "Mali", "Senegal", "Côte d'Ivoire", "DR Congo"],
    "es": ["Spain", "Mexico", "Venezuela", "Colombia", "Peru", "Argentina", "Chile (central)"],
    "pt": ["Brazil"],
}

# Biome cue (from classify_scene_cues) -> coarse climate-zone masks (one or more bboxes).
BIOME_MASKS: dict[str, list[BBox]] = {
    "tropical": [(-180.0, -23.5, 180.0, 23.5)],
    "savanna": [
        (-18.0, 5.0, 50.0, 18.0), (-65.0, -20.0, -40.0, -5.0), (120.0, -20.0, 150.0, -11.0),
    ],
    "mediterranean": [
        (-10.0, 30.0, 40.0, 46.0),    # Mediterranean basin
        (-124.0, 32.0, -117.0, 42.0),  # California
        (-74.0, -38.0, -70.0, -29.0),  # central Chile
        (16.0, -35.0, 26.0, -32.0),    # Cape
        (115.0, -36.0, 140.0, -30.0),  # SW Australia
    ],
    "arid": [(-18.0, 15.0, 30.0, 32.0), (35.0, 15.0, 60.0, 33.0), (-72.0, -30.0, -69.0, -16.0)],
    "temperate": [(-180.0, 30.0, 180.0, 60.0), (-180.0, -55.0, 180.0, -30.0)],
    "continental_cold": [(-180.0, 45.0, 180.0, 78.0)],
}

# Per-feature-type radius (km) for a geocoded point toponym.
_FEATURE_RADIUS_KM: dict[str, float] = {
    "city": 25.0, "town": 20.0, "village": 12.0, "hamlet": 8.0,
    "suburb": 8.0, "neighbourhood": 5.0, "quarter": 5.0,
    "peak": 30.0, "mountain": 30.0, "volcano": 30.0,
    "administrative": 120.0, "county": 120.0, "region": 250.0,
    "state": 300.0, "province": 250.0, "country": 600.0,
}
_DEFAULT_RADIUS_KM = 60.0


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class Priors:
    """The fused inputs, decoupled from the case-file format for easy testing."""

    toponyms: list[dict] = field(default_factory=list)  # {name, lat, lon, feature_type}
    language: str | None = None
    biome: str | None = None       # only set if classify_scene_cues marked it reliable
    setting: str | None = None     # coastal / mountain / … (cross-check only, v1)
    event: str | None = None


@dataclass
class RegionSignal:
    """A single signal for region restriction (not a final answer; raw input for orchestrator)."""
    source: str  # "toponym", "language", "biome"
    name: str  # "Paris (~25 km)", "Spanish-speaking", "tropical"
    boxes: list[BBox]  # geographic bounds
    area_km2: float  # total area of boxes
    metadata: dict = field(default_factory=dict)  # {
    #   "feature_type": "city", "geocoding_confidence": 0.95,
    #   "within_language_region": True, "within_biome": False,
    #   "reliable": True, …
    # }

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RestrictResult:
    """Raw signals emitted by the tool; zero fusion, zero final judgment."""
    signals: list[RegionSignal]
    gazetteer: dict[str, BBox] = field(default_factory=dict)  # ref regions available to orchestrator

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# Geometry helpers (pure python; bbox math only, no shapely dependency)
# --------------------------------------------------------------------------- #
def _box_area_km2(box: BBox) -> float:
    w, s, e, n = box
    mid = math.radians((n + s) / 2)
    height = (n - s) * 111.0
    width = (e - w) * 111.0 * max(math.cos(mid), 0.0)
    return max(height * width, 0.0)


def _boxes_area_km2(boxes: list[BBox]) -> float:
    # Approximate: sum without subtracting overlaps (a magnitude, not exact).
    return round(sum(_box_area_km2(b) for b in boxes), 1)


def _box_intersect(a: BBox, b: BBox) -> BBox | None:
    w, s, e, n = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (w, s, e, n) if w < e and s < n else None


def _intersect_lists(a: list[BBox], b: list[BBox]) -> list[BBox]:
    out = [r for x in a for y in b if (r := _box_intersect(x, y)) is not None]
    return out


def _point_box(lat: float, lon: float, radius_km: float) -> BBox:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _point_in_boxes(lat: float, lon: float, boxes: list[BBox]) -> bool:
    return any(w <= lon <= e and s <= lat <= n for (w, s, e, n) in boxes)


# --------------------------------------------------------------------------- #
# Signal emission (no fusion, no judgment — just facts)
# --------------------------------------------------------------------------- #
_GAZETTEER_LC = {k.lower(): (k, v) for k, v in GAZETTEER.items() if k != "world"}


def _gazetteer_lookup(name: str) -> tuple[str, BBox] | None:
    """Return (canonical_name, bbox) if *name* is a known coarse region, else None."""
    return _GAZETTEER_LC.get(name.strip().lower())


def _toponym_radius(t: dict) -> float:
    return _FEATURE_RADIUS_KM.get(t.get("feature_type") or "", _DEFAULT_RADIUS_KM)


def emit_signals(priors: Priors) -> RestrictResult:
    """Emit all available region-restriction signals without any judgment.

    The output is a flat list of signals: each toponym, the language region(s), the
    biome region(s). All metadata is attached (geocoding confidence, feature type,
    reliability flags, etc.). The orchestrator (Claude) reads these and reasons over
    which to trust, which conflict, and what confidence to assign.

    Zero fusion, zero confidence heuristics — just facts.
    """
    signals: list[RegionSignal] = []

    lang_regions = LANGUAGE_REGIONS.get(priors.language or "")
    lang_boxes = [GAZETTEER[r] for r in lang_regions] if lang_regions else None
    biome_boxes = BIOME_MASKS.get(priors.biome or "") if priors.biome else None

    # --- Emit all toponyms (geocoded or not) -------------------------------- #
    for topo in priors.toponyms:
        if topo.get("lat") is not None and topo.get("lon") is not None:
            # Geocoded toponym: emit as a point + radius
            radius = _toponym_radius(topo)
            box = _point_box(topo["lat"], topo["lon"], radius)
            label = f"{topo['name']} (~{radius:.0f} km)"

            # Check if it's a known admin region in the gazetteer
            gaz_hit = _gazetteer_lookup(topo["name"])
            if gaz_hit is not None:
                region_name, box = gaz_hit
                label = region_name

            metadata = {
                "geocoded": True,
                "geocode_source": topo.get("geocode_source"),
                "feature_type": topo.get("feature_type"),
                "lat": topo.get("lat"),
                "lon": topo.get("lon"),
                "radius_km": radius if gaz_hit is None else None,
            }
            # Cross-checks: is it inside language/biome regions?
            if lang_boxes is not None:
                metadata["within_language_region"] = _point_in_boxes(
                    topo["lat"], topo["lon"], lang_boxes
                )
            if biome_boxes is not None:
                metadata["within_biome_region"] = _point_in_boxes(
                    topo["lat"], topo["lon"], biome_boxes
                )

            signals.append(RegionSignal(
                source="toponym",
                name=label,
                boxes=[box],
                area_km2=_boxes_area_km2([box]),
                metadata=metadata,
            ))
        else:
            # Un-geocoded toponym: emit as-is, marked as un-geocoded
            signals.append(RegionSignal(
                source="toponym",
                name=topo.get("name", "?"),
                boxes=[],
                area_km2=0.0,
                metadata={"geocoded": False, "geocode_source": None},
            ))

    # --- Emit language signal ------------------------------------------------ #
    if lang_boxes is not None and priors.language is not None:
        signals.append(RegionSignal(
            source="language",
            name=f"{priors.language}-speaking region",
            boxes=lang_boxes,
            area_km2=_boxes_area_km2(lang_boxes),
            metadata={"language": priors.language},
        ))

    # --- Emit biome signal -------------------------------------------------- #
    if biome_boxes is not None and priors.biome is not None:
        signals.append(RegionSignal(
            source="biome",
            name=f"{priors.biome} climate zone",
            boxes=biome_boxes,
            area_km2=_boxes_area_km2(biome_boxes),
            metadata={
                "biome": priors.biome,
                "reliable": True,  # Only emitted if classify_scene_cues marked it reliable
            },
        ))

    return RestrictResult(signals=signals, gazetteer=GAZETTEER)


# --------------------------------------------------------------------------- #
# Case-file adapter
# --------------------------------------------------------------------------- #
def _spoken_toponyms(transcription: str) -> list[dict]:
    """Extract+geocode toponyms from a speech transcription, reusing parse_context_note.

    Spoken place names ("je suis à Vallon Combau… le Mont Aiguille") are a high-value cue,
    so the transcription is treated as just another text source — the same NER + geocoder
    the caption goes through. Degrades to no toponyms if parsing/geocoding is unavailable.
    """
    try:
        from frame_geolocator.tools.parse_context_note import parse_context_note
        result = parse_context_note(transcription)
    except Exception:
        return []
    return [
        {"name": t.name, "lat": t.lat, "lon": t.lon, "feature_type": t.feature_type}
        for t in result.toponyms
    ]


def priors_from_case(case: dict, parse_spoken: bool = True) -> Priors:
    """Extract :class:`Priors` from an accumulated case-file dict.

    Toponyms are fused from **both** text sources: the caption (``parse_context_note``)
    and the speech transcription (``identify_spoken_language``). The spoken language also
    backs up the caption language when the caption is absent.
    """
    p = Priors()
    ctx = case.get("parse_context_note") or {}
    p.toponyms = list(ctx.get("toponyms") or [])
    p.language = ctx.get("language")
    p.event = ctx.get("event_type")

    spoken = case.get("identify_spoken_language") or {}
    if not p.language and spoken.get("language"):
        p.language = spoken.get("language")
    transcription = spoken.get("transcription")
    if parse_spoken and transcription:
        p.toponyms.extend(_spoken_toponyms(transcription))

    scene = case.get("classify_scene_cues") or {}
    for group in scene.get("groups", []):
        if group.get("group") == "biome" and group.get("reliable"):
            p.biome = group.get("top")
        if group.get("group") == "setting":
            p.setting = group.get("top")
    return p


def restrict_candidate_region(
    case_file: str | Path | None = None, priors: Priors | None = None
) -> RestrictResult:
    """Emit region-restriction signals from priors (from a case file or given directly).

    Returns a list of raw signals (toponyms, language, biome) with metadata.
    The orchestrator (Claude) reasons over these signals to decide which to trust.
    """
    if priors is None:
        if case_file is None:
            raise ValueError("Provide either a case_file or explicit priors.")
        case = json.loads(Path(case_file).read_text(encoding="utf-8"))
        priors = priors_from_case(case)
    return emit_signals(priors)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """CLI: ``restrict-candidate-region --case-file case.json`` (+ optional overrides)."""
    parser = argparse.ArgumentParser(
        description="Emit region-restriction signals (toponyms, language, biome) with metadata."
    )
    parser.add_argument("--case-file", metavar="PATH",
                        help="Read priors from this accumulated case file.")
    parser.add_argument("--language", help="Override/supply caption language (ISO 639-1).")
    parser.add_argument("--biome", help="Override/supply a biome cue (e.g. tropical).")
    parser.add_argument("--toponym", nargs=3, metavar=("NAME", "LAT", "LON"),
                        action="append", help="Add a geocoded toponym (repeatable).")
    parser.add_argument("--json", metavar="PATH", help="Write the full result as JSON.")
    args = parser.parse_args(argv)

    priors: Priors | None = None
    if args.case_file:
        case = json.loads(Path(args.case_file).read_text(encoding="utf-8"))
        priors = priors_from_case(case)
    else:
        priors = Priors()
    if args.language:
        priors.language = args.language
    if args.biome:
        priors.biome = args.biome
    if args.toponym:
        for name, lat, lon in args.toponym:
            priors.toponyms.append({"name": name, "lat": float(lat), "lon": float(lon),
                                    "feature_type": "city"})

    result = emit_signals(priors)

    print(f"[{len(result.signals)} signal(s)]")
    for sig in result.signals:
        meta_str = ", ".join(f"{k}={v}" for k, v in sig.metadata.items() if v is not None)
        meta_part = f"  ({meta_str})" if meta_str else ""
        print(f"  · {sig.source:10s} {sig.name:30s} {sig.area_km2:>12,.0f} km²{meta_part}")

    if args.json:
        result.to_json(args.json)
        print(f"Wrote {args.json}")
    if args.case_file:
        from frame_geolocator.case_file import merge
        merge(args.case_file, "restrict_candidate_region", result)
        print(f"Case file: {args.case_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
