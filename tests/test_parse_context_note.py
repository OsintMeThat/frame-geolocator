"""Tests for the ``parse_context_note`` tool.

NER backends and Nominatim geocoding are stubbed so these tests run fully
offline with no heavy ML dependencies.  Real integration tests are gated
behind ``pytest.mark.skipif`` and only run when the actual backends are
installed.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from frame_geolocator.tools.parse_context_note import (
    ContextNoteConfig,
    Toponym,
    parse_context_note,
)

# Reach into the module to swap registries / internal helpers in tests.
mod = importlib.import_module("frame_geolocator.tools.parse_context_note")


# --------------------------------------------------------------------------- #
# Stub NER backends
# --------------------------------------------------------------------------- #

class _WorkingNER:
    name = "working"

    def available(self) -> bool:
        return True

    def extract(self, text: str) -> list[str]:
        return ["Marseille", "Vieux-Port"]


class _UnavailableNER:
    name = "unavailable"

    def available(self) -> bool:
        return False

    def extract(self, text: str) -> list[str]:  # pragma: no cover
        raise AssertionError("should not be called")


class _CrashingNER:
    name = "crashing"

    def available(self) -> bool:
        return True

    def extract(self, text: str) -> list[str]:
        raise RuntimeError("boom")


@pytest.fixture()
def stub_backends(monkeypatch: pytest.MonkeyPatch):
    registry = {
        "working": _WorkingNER(),
        "unavailable": _UnavailableNER(),
        "crashing": _CrashingNER(),
    }
    monkeypatch.setattr(mod, "_BACKENDS", registry)
    return registry


# --------------------------------------------------------------------------- #
# Stub geocoder
# --------------------------------------------------------------------------- #

def _fake_geocode(name: str) -> Toponym:
    return Toponym(name=name, lat=43.3, lon=5.4, feature_type="city",
                   geocode_source="nominatim")


@pytest.fixture()
def stub_geocode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mod, "_geocode", _fake_geocode)


# --------------------------------------------------------------------------- #
# NER backend fallback orchestration
# --------------------------------------------------------------------------- #

def test_uses_first_working_ner_backend(stub_backends, stub_geocode) -> None:
    result = parse_context_note("test text", ContextNoteConfig(ner_backends=["working"],
                                                               geocode=False))
    assert result.toponyms
    assert result.toponyms[0].name == "Marseille"


def test_skips_unavailable_backend(stub_backends, stub_geocode) -> None:
    result = parse_context_note(
        "test text",
        ContextNoteConfig(ner_backends=["unavailable", "working"], geocode=False),
    )
    assert result.toponyms[0].name == "Marseille"


def test_falls_back_past_crashing_backend(stub_backends, stub_geocode) -> None:
    result = parse_context_note(
        "test text",
        ContextNoteConfig(ner_backends=["crashing", "working"], geocode=False),
    )
    assert result.toponyms[0].name == "Marseille"


def test_no_ner_backend_produces_empty_toponyms(stub_backends, stub_geocode) -> None:
    result = parse_context_note(
        "test text",
        ContextNoteConfig(ner_backends=["unavailable"], geocode=False),
    )
    assert result.toponyms == []


def test_unknown_backend_name_is_skipped(stub_backends, stub_geocode) -> None:
    result = parse_context_note(
        "test text",
        ContextNoteConfig(ner_backends=["does_not_exist", "working"], geocode=False),
    )
    assert result.toponyms


# --------------------------------------------------------------------------- #
# Geocoding behaviour
# --------------------------------------------------------------------------- #

def test_geocode_calls_up_to_limit(stub_backends, monkeypatch) -> None:
    calls: list[str] = []

    def _mock_geocode(name: str) -> Toponym:
        calls.append(name)
        return Toponym(name=name, lat=1.0, lon=2.0, geocode_source="nominatim")

    monkeypatch.setattr(mod, "_geocode", _mock_geocode)
    monkeypatch.setattr(mod, "_NOMINATIM_DELAY_S", 0)  # no sleep in tests

    parse_context_note("t", ContextNoteConfig(ner_backends=["working"],
                                              geocode=True, geocode_limit=1))
    assert len(calls) == 1  # limit respected; _WorkingNER returns 2 names


def test_no_geocode_when_disabled(stub_backends, monkeypatch) -> None:
    called = []
    monkeypatch.setattr(mod, "_geocode", lambda n: called.append(n) or Toponym(name=n))

    parse_context_note("t", ContextNoteConfig(ner_backends=["working"], geocode=False))
    assert called == []


def test_toponyms_without_coords_when_geocode_disabled(stub_backends) -> None:
    result = parse_context_note("t", ContextNoteConfig(ner_backends=["working"],
                                                       geocode=False))
    assert all(t.lat is None for t in result.toponyms)


def test_geocode_failure_returns_toponym_without_coords(stub_backends,
                                                         monkeypatch) -> None:
    monkeypatch.setattr(mod, "_geocode", lambda n: Toponym(name=n))
    monkeypatch.setattr(mod, "_NOMINATIM_DELAY_S", 0)

    result = parse_context_note("t", ContextNoteConfig(ner_backends=["working"],
                                                       geocode=True, geocode_limit=5))
    assert result.toponyms
    assert all(t.lat is None for t in result.toponyms)


# --------------------------------------------------------------------------- #
# Event classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    ("Fuertes explosiones en la ciudad", "explosion"),
    ("Protesters fill the streets in demonstration", "protest"),
    ("Houses could collapse after flooding", "flooding"),
    ("Army brigade spotted near the border", "military"),
    ("Wildfire spreads through the hills", "wildfire"),
    ("Earthquake hits the region at dawn", "earthquake"),
    ("Beautiful sunset over the lake", None),
])
def test_event_classification(text: str, expected: str | None,
                               stub_backends, stub_geocode) -> None:
    result = parse_context_note(text, ContextNoteConfig(ner_backends=["working"],
                                                        geocode=False))
    assert result.event_type == expected


# --------------------------------------------------------------------------- #
# Timestamp extraction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    ("Posted on 2024-01-15 at noon", "2024-01-15"),
    ("Filmed 15/01/2024 near the coast", "15/01/2024"),
    ("Event on 3 January 2024", "3 January 2024"),
    ("Explosion heard at 06:30 this morning", "06:30"),
    ("No date information here", None),
])
def test_timestamp_extraction(text: str, expected: str | None,
                               stub_backends, stub_geocode) -> None:
    result = parse_context_note(text, ContextNoteConfig(ner_backends=["unavailable"],
                                                        geocode=False))
    assert result.timestamp_hint == expected


# --------------------------------------------------------------------------- #
# Regex NER backend (no deps — always runs)
# --------------------------------------------------------------------------- #

def test_regex_backend_extracts_hashtag_toponyms() -> None:
    from frame_geolocator.tools.parse_context_note import RegexNERBackend
    backend = RegexNERBackend()
    names = backend.extract("Fuerte Tiuna… #caracas #venezuela #fuertetiuna")
    lower_names = [n.lower() for n in names]
    assert "caracas" in lower_names
    assert "venezuela" in lower_names


def test_regex_backend_extracts_title_case_toponyms() -> None:
    from frame_geolocator.tools.parse_context_note import RegexNERBackend
    backend = RegexNERBackend()
    names = backend.extract(
        "certains supporters ont plongé dans le Vieux-Port à Marseille"
    )
    assert any("Marseille" in n for n in names)


def test_regex_backend_always_available() -> None:
    from frame_geolocator.tools.parse_context_note import RegexNERBackend
    assert RegexNERBackend().available() is True


def test_high_precision_only_keeps_allcaps_drops_title_case() -> None:
    from frame_geolocator.tools.parse_context_note import RegexNERBackend
    backend = RegexNERBackend()
    text = "status from a friend in LA GUAIRA, Sometown nearby"
    full = backend.extract(text)
    hp = backend.extract(text, high_precision_only=True)
    assert "LA GUAIRA" in hp                 # ALL-CAPS kept in both modes
    assert any("Sometown" in n for n in full)  # title-case present in full mode
    assert all("Sometown" not in n for n in hp)  # …but dropped from the supplement


def test_supplement_recovers_allcaps_primary_missed(stub_backends, stub_geocode) -> None:
    # The stub primary backend never returns the ALL-CAPS place; the high-precision
    # supplement must still recover it (the real "spaCy misses LA GUAIRA" bug).
    result = parse_context_note(
        "noise in LA GUAIRA",
        ContextNoteConfig(ner_backends=["working"], geocode=False),
    )
    names = [t.name for t in result.toponyms]
    assert "LA GUAIRA" in names
    assert "Marseille" in names  # primary backend's output is still merged


# --------------------------------------------------------------------------- #
# JSON serialisation
# --------------------------------------------------------------------------- #

def test_to_json_roundtrip(tmp_path: Path, stub_backends, stub_geocode) -> None:
    result = parse_context_note(
        "Explosion near Marseille",
        ContextNoteConfig(ner_backends=["working"], geocode=True),
    )
    out = tmp_path / "result.json"
    result.to_json(out)
    data = json.loads(out.read_text())
    assert data["event_type"] == "explosion"
    assert data["language"] == result.language
    assert isinstance(data["toponyms"], list)


# --------------------------------------------------------------------------- #
# Real end-to-end (only if heavy backends are installed)
# --------------------------------------------------------------------------- #

def _has_langdetect() -> bool:
    import importlib.util
    return importlib.util.find_spec("langdetect") is not None


def _has_spacy_model() -> bool:
    try:
        import spacy
        spacy.load("xx_ent_wiki_sm")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_langdetect(), reason="langdetect not installed")
def test_real_language_detection() -> None:
    result = parse_context_note(
        "images des manifestants à Niamey fuyant les tirs de sommation",
        ContextNoteConfig(geocode=False, ner_backends=["regex"]),
    )
    assert result.language == "fr"
    assert result.language_confidence > 0.5


@pytest.mark.skipif(not _has_spacy_model(), reason="xx_ent_wiki_sm not installed")
def test_real_spacy_ner_finds_gpe() -> None:
    result = parse_context_note(
        "Protesters in Niamey flee warning shots from the presidential guard.",
        ContextNoteConfig(geocode=False, ner_backends=["spacy"]),
    )
    names = [t.name for t in result.toponyms]
    assert any("Niamey" in n for n in names)
