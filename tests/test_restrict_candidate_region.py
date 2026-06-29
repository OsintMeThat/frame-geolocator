"""Tests for ``restrict_candidate_region``.

Pure-stdlib tool, so everything runs without heavy deps. We test signal emission:
all signals are emitted with correct geometry and metadata, no fusion logic.
The orchestrator (Claude) decides which signals to trust.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import frame_geolocator.tools.restrict_candidate_region as rcr
from frame_geolocator.tools.restrict_candidate_region import (
    Priors,
    emit_signals,
    priors_from_case,
    restrict_candidate_region,
)


def _topo(name, lat, lon, ftype="city"):
    return {"name": name, "lat": lat, "lon": lon, "feature_type": ftype}


def _point_in_boxes(lat, lon, boxes) -> bool:
    return any(w <= lon <= e and s <= lat <= n for (w, s, e, n) in boxes)


def _find_signal(result, source, name_contains=None):
    """Find a signal by source and optional name substring."""
    for sig in result.signals:
        if sig.source == source:
            if name_contains is None or name_contains in sig.name:
                return sig
    return None


# --- Toponym emission -------------------------------------------------------- #
def test_geocoded_toponym_emits_as_signal() -> None:
    res = emit_signals(Priors(toponyms=[_topo("Niamey", 13.52, 2.11, "administrative")]))
    topo_sig = _find_signal(res, "toponym", "Niamey")
    assert topo_sig is not None
    assert topo_sig.metadata["geocoded"] is True
    assert topo_sig.metadata["feature_type"] == "administrative"
    assert _point_in_boxes(13.52, 2.11, topo_sig.boxes)
    # Administrative features are smaller regions (verified via geometry).
    assert topo_sig.area_km2 < 600_000


def test_all_toponyms_emitted_no_filtering() -> None:
    topos = [_topo("Paris", 48.85, 2.35), _topo("Lyon", 45.76, 4.84)]
    res = emit_signals(Priors(toponyms=topos))
    topo_sigs = [s for s in res.signals if s.source == "toponym"]
    assert len(topo_sigs) == 2
    names = [s.name for s in topo_sigs]
    assert any("Paris" in n for n in names)
    assert any("Lyon" in n for n in names)


def test_ungeocoded_toponym_emitted_with_metadata() -> None:
    res = emit_signals(Priors(toponyms=[{"name": "UnknownPlace"}]))
    ungeocoded = _find_signal(res, "toponym", "UnknownPlace")
    assert ungeocoded is not None
    assert ungeocoded.metadata["geocoded"] is False
    assert ungeocoded.boxes == []  # no geometry for ungeocoded


def test_toponym_metadata_shows_language_biome_overlap() -> None:
    # Paris is inside the French-speaking region and temperate zone.
    res = emit_signals(Priors(
        toponyms=[_topo("Paris", 48.85, 2.35)],
        language="fr",
        biome="temperate"
    ))
    topo_sig = _find_signal(res, "toponym", "Paris")
    assert topo_sig.metadata["within_language_region"] is True
    assert topo_sig.metadata["within_biome_region"] is True


def test_toponym_metadata_flags_conflict() -> None:
    # Paris in tropical zone is a conflict.
    res = emit_signals(Priors(
        toponyms=[_topo("Paris", 48.85, 2.35)],
        biome="tropical"
    ))
    topo_sig = _find_signal(res, "toponym", "Paris")
    assert topo_sig.metadata["within_biome_region"] is False  # conflict flag


def test_gazetteer_regions_use_true_extent() -> None:
    # "California" in the gazetteer should use its full extent, not point+radius.
    res = emit_signals(Priors(toponyms=[_topo("California", 36.70, -118.76)]))
    topo_sig = _find_signal(res, "toponym", "California")
    assert topo_sig.name == "California"  # Uses gazetteer name, not point+radius
    # Verify coverage: Dana Point (coast) and Lassen (mountains) both inside.
    assert _point_in_boxes(33.46, -117.71, topo_sig.boxes)  # Dana Point
    assert _point_in_boxes(40.49, -121.50, topo_sig.boxes)  # Lassen


# --- Language signal -------------------------------------------------------- #
def test_language_signal_emitted() -> None:
    res = emit_signals(Priors(language="ru"))
    lang_sig = _find_signal(res, "language")
    assert lang_sig is not None
    assert "ru" in lang_sig.name or "Russian" in lang_sig.name
    # Should cover Russia, Belarus, Ukraine, Caucasus, etc.
    assert lang_sig.area_km2 > 1_000_000
    assert _point_in_boxes(55.75, 37.62, lang_sig.boxes)  # Moscow


def test_english_does_not_emit_language_signal() -> None:
    res = emit_signals(Priors(language="en"))
    lang_sig = _find_signal(res, "language")
    assert lang_sig is None  # English is not restricted


# --- Biome signal ----------------------------------------------------------- #
def test_biome_signal_emitted() -> None:
    res = emit_signals(Priors(biome="tropical"))
    biome_sig = _find_signal(res, "biome")
    assert biome_sig is not None
    assert "tropical" in biome_sig.name
    assert biome_sig.metadata["biome"] == "tropical"
    assert biome_sig.metadata["reliable"] is True
    # Equatorial region should be covered.
    assert _point_in_boxes(0.0, 0.0, biome_sig.boxes)


# --- No prior: emit nothing -------------------------------------------------- #
def test_no_prior_emits_no_signals() -> None:
    res = emit_signals(Priors())
    assert res.signals == []


# --- Case-file adapter ------------------------------------------------------ #
def test_priors_from_case_reads_reliable_biome_only(tmp_path: Path) -> None:
    case = {
        "parse_context_note": {
            "language": "es",
            "event_type": "explosion",
            "toponyms": [_topo("Caracas", 10.5, -66.9, "administrative")],
        },
        "classify_scene_cues": {
            "groups": [
                {"group": "biome", "reliable": False, "top": "continental_cold"},
                {"group": "setting", "reliable": True, "top": "urban"},
            ]
        },
    }
    p = priors_from_case(case)
    assert p.language == "es"
    assert p.biome is None  # unreliable biome is not propagated
    assert p.setting == "urban"

    cf = tmp_path / "case.json"
    cf.write_text(json.dumps(case), encoding="utf-8")
    res = restrict_candidate_region(case_file=cf)
    # Should have toponym and language signals, no biome.
    assert len(res.signals) == 2
    assert _find_signal(res, "toponym") is not None
    assert _find_signal(res, "language") is not None
    assert _find_signal(res, "biome") is None


def test_requires_some_input() -> None:
    with pytest.raises(ValueError):
        restrict_candidate_region()


# --- Spoken transcription as a toponym source ------------------------------- #
def test_spoken_toponyms_added_to_signals(monkeypatch) -> None:
    monkeypatch.setattr(
        rcr, "_spoken_toponyms",
        lambda text: [_topo("Mont-Aiguille", 44.84, 5.55, "peak")],
    )
    case = {"identify_spoken_language": {"language": "fr",
                                         "transcription": "je suis au Mont Aiguille"}}
    p = priors_from_case(case)
    assert any(t["name"] == "Mont-Aiguille" for t in p.toponyms)
    res = emit_signals(p)
    # Should have both the Mont-Aiguille toponym and the fr language signal.
    assert _find_signal(res, "toponym", "Mont-Aiguille") is not None
    assert _find_signal(res, "language") is not None
    mont_sig = _find_signal(res, "toponym", "Mont-Aiguille")
    assert _point_in_boxes(44.84, 5.55, mont_sig.boxes)


def test_spoken_language_backs_up_absent_caption(monkeypatch) -> None:
    monkeypatch.setattr(rcr, "_spoken_toponyms", lambda text: [])
    p = priors_from_case({"identify_spoken_language": {"language": "es",
                                                       "transcription": "hola"}})
    assert p.language == "es"
    res = emit_signals(p)
    assert _find_signal(res, "language") is not None


def test_caption_language_takes_priority_over_spoken(monkeypatch) -> None:
    monkeypatch.setattr(rcr, "_spoken_toponyms", lambda text: [])
    case = {"parse_context_note": {"language": "fr", "toponyms": []},
            "identify_spoken_language": {"language": "es", "transcription": "hola"}}
    p = priors_from_case(case)
    assert p.language == "fr"
    res = emit_signals(p)
    lang_sig = _find_signal(res, "language")
    assert "fr" in lang_sig.name or "French" in lang_sig.name


def test_parse_spoken_false_skips_transcription(monkeypatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(rcr, "_spoken_toponyms",
                        lambda text: called.append(text) or [])
    case = {"identify_spoken_language": {"transcription": "je suis au Mont Aiguille"}}
    priors_from_case(case, parse_spoken=False)
    assert called == []
