"""Tool C.12 — ``parse_context_note``: extract geolocation priors from an accompanying note.

The caption/description shipped with OSINT media is a primary input, not an afterthought.
It often names the city outright, specifies an event type that restricts the search region,
or carries a timestamp that constrains sun/shadow analysis.

Design notes
-----------
* **Local & free**: language detection via ``langdetect`` (offline, pure Python);
  NER via ``spaCy`` (optional heavy backend) with a regex heuristic fallback that
  requires no dependencies at all.  Backends are tried in priority order with
  automatic fallback, exactly like ``read_text_ocr``.
* Toponyms are geocoded via **Nominatim** (free OSM geocoder, no API key) to produce
  lat/lon priors for ``restrict_candidate_region``.
* **Discipline**: the caption restricts *where to look*; it is **never** accepted as
  the final answer.  A result the media's own content cannot confirm is reported as
  *unverified* by the pipeline (see §3 of architecture.md — caption leakage guard).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_DELAY_S = 1.1  # OSM fair-use: one request per second

# Keywords that signal a known event type.  Multi-language, lowercase.
_EVENT_KEYWORDS: dict[str, list[str]] = {
    "explosion": [
        "explosion", "blast", "detonation", "explosi", "boom",
        "взрыв", "взрыва",
        "explosiones", "explosión",
    ],
    "protest": [
        "protest", "demonstration", "rally",
        "manifestation", "manifestant", "manifestantes",
        "митинг", "протест",
    ],
    "flooding": [
        "flood", "flooding", "inondation",
        "наводнение", "inundaci",
    ],
    "military": [
        "military", "army", "brigade", "battalion", "bataillon", "soldier", "troops",
        "армия", "война", "батальон", "военн",
        "fuerte",  # Spanish: "Fuerte Tiuna" = military base
    ],
    "wildfire": [
        "wildfire", "fire", "incendie",
        "пожар", "fuego",
    ],
    "earthquake": [
        "earthquake", "seism", "tremblement",
        "землетрясение", "sismo",
    ],
}


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class Toponym:
    """A place name extracted from the note, optionally geocoded."""

    name: str
    lat: float | None = None
    lon: float | None = None
    feature_type: str | None = None   # "city", "country", "peak", …
    geocode_source: str | None = None  # "nominatim" or None


@dataclass
class ContextNoteConfig:
    """Configuration for :func:`parse_context_note`."""

    ner_backends: list[str] = field(default_factory=lambda: ["spacy", "regex"])
    """NER backend priority order.  First available backend is used."""

    geocode: bool = True
    """Call Nominatim to resolve lat/lon for each extracted toponym."""

    geocode_limit: int = 3
    """Maximum number of Nominatim calls per note (API fair-use)."""


@dataclass
class ContextNoteResult:
    """Auditable output of :func:`parse_context_note`."""

    text: str
    language: str            # ISO 639-1 code, e.g. "fr"; "und" if undetermined
    language_confidence: float
    toponyms: list[Toponym]
    event_type: str | None   # one of the keys in ``_EVENT_KEYWORDS``, or None
    timestamp_hint: str | None  # raw extracted string, not parsed to datetime

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #

def _detect_language(text: str) -> tuple[str, float]:
    """Return (iso_code, confidence).  Falls back to ("und", 0.0) if unavailable."""
    if importlib.util.find_spec("langdetect") is None:
        return "und", 0.0
    try:
        from langdetect import detect_langs
        results = detect_langs(text)
        if results:
            top = results[0]
            return top.lang, round(float(top.prob), 3)
    except Exception:
        pass
    return "und", 0.0


# --------------------------------------------------------------------------- #
# NER backends
# --------------------------------------------------------------------------- #

class NERBackend(Protocol):
    name: str

    def available(self) -> bool: ...
    def extract(self, text: str) -> list[str]:
        """Return candidate place-name strings found in ``text``."""
        ...


class SpacyNERBackend:
    """spaCy multilingual NER backend (requires ``spacy`` + a downloaded model)."""

    name = "spacy"

    def __init__(self) -> None:
        self._nlp = None

    def available(self) -> bool:
        return importlib.util.find_spec("spacy") is not None

    def _load(self):
        if self._nlp is not None:
            return self._nlp
        import spacy
        for model in ("xx_ent_wiki_sm", "en_core_web_sm"):
            try:
                self._nlp = spacy.load(model)
                return self._nlp
            except OSError:
                continue
        raise RuntimeError(
            "No spaCy model found. Install one: python -m spacy download xx_ent_wiki_sm"
        )

    def extract(self, text: str) -> list[str]:
        nlp = self._load()
        doc = nlp(text)
        geo_labels = {"GPE", "LOC", "FAC"}
        seen: set[str] = set()
        result: list[str] = []
        for ent in doc.ents:
            if ent.label_ in geo_labels:
                name = ent.text.strip()
                if name and name not in seen:
                    seen.add(name)
                    result.append(name)
        return result


class RegexNERBackend:
    """Heuristic toponym extractor — always available, no extra dependencies.

    Extracts:
    * Hashtag-encoded place names  (#paris → "paris", #Zaporizhzhia → "Zaporizhzhia")
    * Hyphenated proper nouns (Vieux-Port, Dana-Point)
    * Title-case word sequences of ≥2 chars.
    * ALL-CAPS sequences (LA GUAIRA, NIAMEY) of ≥3 chars.

    Negation filter: candidates preceded by "not in", "not at", "pas à",
    "pas en", "ni" (+ a space) are dropped — avoids "Moscow" from "not in Moscow".

    Over-generates by design: non-toponyms that pass negation filtering will
    fail to geocode and are retained without coordinates, which is harmless.
    """

    name = "regex"

    _HASHTAG = re.compile(r'#([A-Za-zÀ-ÿА-яёЁ][A-Za-zÀ-ÿА-яёЁ0-9_]+)')

    # Hyphenated proper nouns: Vieux-Port, Dana-Point (Title-Case with hyphens).
    _HYPHENATED = re.compile(
        r'(?<![A-Za-zÀ-ÿ])'
        r'([A-ZÁÀÂÄÉÈÊËÍÌÎÏÓÒÔÖÚÙÛÜÑ][a-záàâäéèêëíìîïóòôöúùûüñ]+'
        r'(?:-[A-ZÁÀÂÄÉÈÊËÍÌÎÏÓÒÔÖÚÙÛÜÑ][a-záàâäéèêëíìîïóòôöúùûüñ]+)+)'
    )

    # Title-case sequences (no hyphens — handled above).
    # Negative lookahead (?![A-Z]) rejects CamelCase mid-word matches
    # (e.g. "Whats" from "WhatsApp" is followed by "A" → skipped).
    _TITLE_CASE = re.compile(
        r'(?<![A-Za-zÀ-ÿА-яёЁ])'
        r'([A-ZÁÀÂÄÉÈÊËÍÌÎÏÓÒÔÖÚÙÛÜÑ][a-záàâäéèêëíìîïóòôöúùûüñ]{1,}'
        r'(?:\s+[A-ZÁÀÂÄÉÈÊËÍÌÎÏÓÒÔÖÚÙÛÜÑ][a-záàâäéèêëíìîïóòôöúùûüñ]{1,})*)'
        r'(?![A-ZÀ-Ÿ])'
    )

    # ALL-CAPS sequences: each word ≥2 chars, whole sequence ≥3 chars total
    # (catches "LA GUAIRA", "NIAMEY", "PARIS", but not lone "AM", "PM", "NY").
    _ALL_CAPS = re.compile(r'(?<![A-Z])([A-Z]{2,}(?:\s+[A-Z]{2,})*)')

    # Negation prefixes: if a candidate is immediately preceded by one of these,
    # it is dropped — avoids false positives like "Moscow" from "not in Moscow".
    _NEGATION = re.compile(
        r'(?:not\s+in|not\s+at|pas\s+[àa]|pas\s+en|ni)\s+$',
        re.IGNORECASE,
    )

    # Pure function words (articles, prepositions, conjunctions) in a handful
    # of OSINT-relevant languages.  Kept deliberately short: no proper nouns,
    # no domain-specific words — let Nominatim be the real toponym filter.
    _FUNCTION_WORDS = frozenset({
        # English
        "The", "This", "That", "And", "But", "Not", "For", "With", "From",
        "Into", "By", "Of", "In", "At", "On", "To", "Its",
        # French
        "Le", "La", "Les", "Un", "Une", "Des", "Du", "Au", "Aux",
        "Pour", "Dans", "Sur", "Avec", "Par", "Pas", "Qui", "Que",
        # Spanish
        "El", "Los", "Las", "Una", "Del", "Por", "Con", "Sin", "Pero",
        # Russian (romanised)
        "Ne",
    })

    def available(self) -> bool:
        return True

    def _negated(self, text: str, match_start: int) -> bool:
        """True if the 35 chars before match_start end with a negation prefix."""
        prefix = text[max(0, match_start - 35): match_start]
        return bool(self._NEGATION.search(prefix))

    @staticmethod
    def _complete_token(text: str, name: str) -> bool:
        """True if *name* appears as a complete token in *text* (not inside a longer word).

        Guards against regex backtracking artefacts such as extracting "What"
        from "WhatsApp" — "What" in "WhatsApp" is surrounded by letters, so it
        fails this check.
        """
        return bool(re.search(r'(?<![A-Za-zÀ-ÿ])' + re.escape(name) + r'(?![A-Za-zÀ-ÿ])', text))

    @staticmethod
    def _deduplicate(names: list[str]) -> list[str]:
        """Keep compound names; drop sub-words already covered by a compound."""
        out: list[str] = []
        for name in names:
            # Drop if this name is a bare word that appears inside a longer name.
            pattern = r'(?<![A-Za-z])' + re.escape(name) + r'(?![A-Za-z])'
            dominated = any(
                name != other and re.search(pattern, other)
                for other in names
            )
            if not dominated:
                out.append(name)
        return out

    def extract(self, text: str, high_precision_only: bool = False) -> list[str]:
        """Extract candidate toponyms.

        With ``high_precision_only`` only the case-robust, high-signal patterns are used
        (hashtags + ALL-CAPS): these are exactly the toponyms a model NER misses, so they
        make a safe **supplement** to spaCy without the bare title-case noise.
        """
        seen: set[str] = set()
        raw: list[str] = []

        def _add(name: str, start: int) -> None:
            name = name.strip()
            if (len(name) < 3
                    or name in self._FUNCTION_WORDS
                    or name in seen
                    or self._negated(text, start)
                    or not self._complete_token(text, name)):
                return
            seen.add(name)
            raw.append(name)

        # 1. Hashtags — highest signal, no negation risk.
        for m in self._HASHTAG.finditer(text):
            name = m.group(1)
            if len(name) >= 3 and name not in seen:
                seen.add(name)
                raw.append(name)

        # 2. ALL-CAPS sequences ≥2 chars per word (LA GUAIRA, NIAMEY, PARIS).
        for m in self._ALL_CAPS.finditer(text):
            _add(m.group(1), m.start(1))

        if not high_precision_only:
            # Hyphenated proper nouns (Vieux-Port, Dana-Point).
            for m in self._HYPHENATED.finditer(text):
                _add(m.group(1), m.start(1))
            # Title-case sequences — the lower-precision pattern (overlaps model NER).
            for m in self._TITLE_CASE.finditer(text):
                _add(m.group(1), m.start(1))

        return self._deduplicate(raw)


# Backend registry — tests may inject fakes here.
_BACKENDS: dict[str, NERBackend] = {
    "spacy": SpacyNERBackend(),
    "regex": RegexNERBackend(),
}

# Always-on, high-precision supplement (hashtags + ALL-CAPS) merged on top of whichever
# NER backend runs, to recover case-robust toponyms that model NER misses.
_HIGH_PRECISION = RegexNERBackend()


# --------------------------------------------------------------------------- #
# Geocoding (Nominatim)
# --------------------------------------------------------------------------- #

def _geocode(name: str) -> Toponym:
    """Look up ``name`` via Nominatim.  Returns a Toponym without coords on failure."""
    params = urllib.parse.urlencode({"q": name, "format": "jsonv2", "limit": 1})
    req = urllib.request.Request(
        f"{_NOMINATIM_URL}?{params}",
        headers={"User-Agent": "frame-geolocator/0.0.1 (OSINT research tool)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return Toponym(name=name)
    if not data:
        return Toponym(name=name)
    hit = data[0]
    return Toponym(
        name=name,
        lat=float(hit["lat"]),
        lon=float(hit["lon"]),
        feature_type=hit.get("type") or hit.get("class"),
        geocode_source="nominatim",
    )


# --------------------------------------------------------------------------- #
# Event classification
# --------------------------------------------------------------------------- #

def _classify_event(text: str) -> str | None:
    lower = text.lower()
    for event, keywords in _EVENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return event
    return None


# --------------------------------------------------------------------------- #
# Timestamp extraction
# --------------------------------------------------------------------------- #

_TS_PATTERNS = [
    re.compile(r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b'),
    re.compile(r'\b(\d{1,2}[./]\d{1,2}[./]\d{4})\b'),
    re.compile(
        r'\b(\d{1,2}\s+(?:'
        r'january|february|march|april|may|june|july|august|september|october|november|december|'
        r'janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre'
        r')[a-z]*\.?\s+\d{4})\b',
        re.IGNORECASE,
    ),
    re.compile(r'\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?)\b', re.IGNORECASE),
]


def _extract_timestamp(text: str) -> str | None:
    for pat in _TS_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def parse_context_note(
    text: str, config: ContextNoteConfig | None = None
) -> ContextNoteResult:
    """Parse an OSINT caption/note and extract geolocation priors.

    Parameters
    ----------
    text:
        Raw caption or description shipped with the media.
    config:
        Optional :class:`ContextNoteConfig`.

    Returns
    -------
    ContextNoteResult
        Language, toponyms (with optional lat/lon from Nominatim), event type,
        and raw timestamp hint.

    Notes
    -----
    The result is a **prior to verify**, not the final answer.  Downstream tools
    (``restrict_candidate_region``, then visual/terrain/audio analysis) must
    confirm any location suggested by the caption.
    """
    config = config or ContextNoteConfig()

    language, lang_conf = _detect_language(text)
    event_type = _classify_event(text)
    timestamp_hint = _extract_timestamp(text)

    # NER: first available backend that doesn't crash provides the primary toponyms…
    raw_names: list[str] = []
    seen_names: set[str] = set()

    def _merge(names: list[str]) -> None:
        for name in names:
            if name not in seen_names:
                seen_names.add(name)
                raw_names.append(name)

    for backend_name in config.ner_backends:
        backend = _BACKENDS.get(backend_name)
        if backend is None or not backend.available():
            continue
        try:
            _merge(backend.extract(text))
        except Exception:
            continue
        break

    # …then always supplement with high-precision, case-robust signals (hashtags and
    # ALL-CAPS place names). A model NER like spaCy systematically misses an uppercase
    # toponym such as "LA GUAIRA"; merging only these high-precision patterns recovers it
    # without importing the noisy bare title-case extraction (which would wrongly add
    # "Putin", "Post", "In California"…). General fix, not tuned to any one caption.
    _merge(_HIGH_PRECISION.extract(text, high_precision_only=True))

    # Geocode up to geocode_limit toponyms; sleep between calls for OSM fair-use.
    toponyms: list[Toponym] = []
    for i, name in enumerate(raw_names):
        if config.geocode and i < config.geocode_limit:
            if i > 0:
                time.sleep(_NOMINATIM_DELAY_S)
            toponyms.append(_geocode(name))
        else:
            toponyms.append(Toponym(name=name))

    return ContextNoteResult(
        text=text,
        language=language,
        language_confidence=lang_conf,
        toponyms=toponyms,
        event_type=event_type,
        timestamp_hint=timestamp_hint,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    """CLI: ``parse-context-note <text> [--no-geocode] [--json output.json]``."""
    parser = argparse.ArgumentParser(
        description="Extract geolocation priors (language, toponyms, event) from a caption."
    )
    parser.add_argument(
        "text",
        help="Caption / description text.  Pass - to read from stdin.",
    )
    parser.add_argument(
        "--ner-backends", nargs="+", default=["spacy", "regex"],
        metavar="BACKEND",
        help="NER backend priority order (default: spacy regex).",
    )
    parser.add_argument("--no-geocode", action="store_true",
                        help="Skip Nominatim geocoding (offline mode).")
    parser.add_argument("--geocode-limit", type=int, default=3, metavar="N",
                        help="Max Nominatim calls per run (default: 3).")
    parser.add_argument("--json", metavar="PATH",
                        help="Write full result as JSON to this path.")
    parser.add_argument("--case-file", metavar="PATH",
                        help="Shared case file to merge this result into.")
    args = parser.parse_args(argv)

    import sys
    text = sys.stdin.read() if args.text == "-" else args.text

    config = ContextNoteConfig(
        ner_backends=args.ner_backends,
        geocode=not args.no_geocode,
        geocode_limit=args.geocode_limit,
    )
    result = parse_context_note(text, config)

    print(f"[language: {result.language} ({result.language_confidence:.0%})]")
    if result.event_type:
        print(f"[event:    {result.event_type}]")
    if result.timestamp_hint:
        print(f"[time:     {result.timestamp_hint}]")
    print(f"[toponyms: {len(result.toponyms)}]")
    for t in result.toponyms:
        if t.lat is not None:
            print(f"  · {t.name}  →  {t.lat:.4f}, {t.lon:.4f}  ({t.feature_type})")
        else:
            print(f"  · {t.name}  →  (not geocoded)")

    if args.json:
        result.to_json(args.json)
        print(f"Wrote {args.json}")
    if args.case_file:
        from frame_geolocator.case_file import merge
        merge(args.case_file, "parse_context_note", result)
        print(f"Case file: {args.case_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
