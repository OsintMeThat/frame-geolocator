# FrameGeolocator

OSINT geolocation of images and videos through an **agentic system that reproduces a
human analyst's reasoning** — explicit, auditable, and generalizable — not a black-box
single-geography model.

The intelligence lives in a catalogue of small, independently-testable **tools**; an
orchestrating front-end (Claude via subscription, or an autonomous Python orchestrator)
only routes between them and interprets their output.  Every tool is **local & free**
(no paid LLM API).  See [docs/architecture.md](docs/architecture.md) for the full design.

> Language policy: the entire repository is in **English** (code, comments, docs).

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

Heavy ML backends are opt-in extras:

| Extra   | Needed by |
|---------|-----------|
| `ocr`    | `read-text-ocr` |
| `audio`  | `extract-audio`, `identify-spoken-language`, `flash-to-bang-range` |
| `sound`  | `classify-audio-events` |
| `nlp`    | `parse-context-note` (+ `python -m spacy download xx_ent_wiki_sm`) |
| `vision` | `classify-scene-cues` |

```bash
pip install -e '.[dev,ocr,audio,sound,nlp,vision]'   # everything
```

## Tests & lint

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

Tests synthesize their own media; real-model tests are opt-in:
`RUN_WHISPER_TESTS=1 pytest`, `RUN_PANNS_TESTS=1 pytest`.

---

## Tools (CLI)

Every tool is importable from `frame_geolocator.tools` and accepts `--json PATH` to
write a structured audit record.  Run any command with `-h` for the full flag list.

### `decompose-video`
Picks the sharpest, most informative frames from a video and produces a contact sheet.

```bash
decompose-video data/videos/001.mp4 --out data/frames/001 --frames 12
```

### `read-text-ocr`
Multilingual OCR of signs, plates, banners.  EasyOCR → Tesseract fallback.  Needs `ocr`.

```bash
read-text-ocr data/frames/003/frame_011_t0016.10.jpg --lang fr en --json ocr.json
```

### `extract-audio`
Demuxes a video to a 16 kHz mono WAV.  Foundation of the audio branch.  Needs `audio`.

```bash
extract-audio data/videos/002.mp4 --out data/frames/002/audio.wav
```

### `identify-spoken-language`
Local Whisper: detects the spoken language and transcribes speech.  Needs `audio`.

```bash
identify-spoken-language data/videos/006.mp4 --model base --json speech.json
```

### `classify-audio-events`
PANNs (AudioSet 527 classes): detects gunshots, explosions, sirens, crowd sounds…
OSINT-relevant labels are flagged `★`.  Needs `sound`.

```bash
classify-audio-events data/videos/003.mp4 --event-threshold 0.15 --json events.json
```

### `flash-to-bang-range`
Measures the delay between a visual flash and its sound → distance to source.
Needs `audio`.

```bash
flash-to-bang-range data/videos/clip.mp4 --json range.json
```

### `parse-context-note`
Extracts language, place names (geocoded via Nominatim), event type, and timestamp
from the caption/description shipped with the media.  Needs `nlp` for language
detection; the regex NER fallback works without any extra.

```bash
parse-context-note "images des manifestants à Niamey" --json priors.json
parse-context-note - < caption.txt --no-geocode
```

### `classify-scene-cues`
Reads region priors **from the pixels alone** — for the common case where the caption is
absent or useless (a watermark, a vague "in California"). Zero-shot OpenCLIP scores each
frame against an explicit, auditable cue taxonomy (setting, biome, time-of-day) — never a
black-box "country guess".  Biome cues are **gated on frame luminance**: on dark night
footage the group is returned `reliable: false` instead of a confident wrong answer (CLIP
otherwise calls a tropical night "snowy").  Point it at a single image or at a frames
directory (e.g. `decompose-video` output) to average cues over many frames.  Needs `vision`.

```bash
classify-scene-cues data/frames/004 --json scene.json
classify-scene-cues data/frames/004/frame_007_t0024.50.jpg --groups setting biome
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--groups` | all | Subset of cue groups: `setting biome time_of_day`. |
| `--max-frames` | `12` | Max frames evenly sampled from a directory. |
| `--min-luminance` | `70` | Frames dimmer than this are dropped from biome cues. |
| `--model` / `--pretrained` | `ViT-B-32` / `laion2b_s34b_b79k` | OpenCLIP backbone + weights. |
| `--json` / `--case-file` | — | Write a JSON audit record / merge into the shared case file. |

### `restrict-candidate-region`
Fuses the accumulated priors (toponyms + caption language + biome cues) into one or more
**named candidate search regions** with a confidence — the "where to look" reducer.  It
does *not* geolocate; it narrows the search.  Granularity follows the evidence: a city
radius when a toponym is geocoded, else a country, a language family
("Russian-speaking region"), a climate zone, or — when nothing is usable — the whole
world at confidence 0.  Cross-checks catch a mis-geocoded toponym (flagged as a
`conflict`, confidence lowered) instead of trusting it.  Pure-stdlib, no extra needed.

```bash
restrict-candidate-region --case-file data/frames/003/case.json --json region.json
restrict-candidate-region --language ru                 # → large "Russian-speaking" region
restrict-candidate-region --language es --biome tropical # → Spanish-speaking ∩ tropical
```

| Flag | Meaning |
|------|---------|
| `--case-file PATH` | Read priors from an accumulated case file (and merge the result back). |
| `--language CODE` | Supply/override the caption language (ISO 639-1). |
| `--biome NAME` | Supply/override a biome cue (e.g. `tropical`, `mediterranean`). |
| `--toponym NAME LAT LON` | Add a geocoded toponym (repeatable). |
| `--json PATH` | Write the full result (candidate regions, confidence, conflicts) as JSON. |

### `extract-skyline-profile`
Extracts the **sky/ground horizon silhouette** of a mountain frame as a 1-D profile — the
ridge signature that terrain matching will later correlate against DEM-rendered panoramas.
Deterministic (OpenCV, no model), with an adaptive sky rule and an honest `coverage` flag
(down-looking frames with no sky are marked unusable).  `--overlay` writes a JPG with the
detected skyline drawn on the frame so you can **reopen and visually check** it anytime.

```bash
extract-skyline-profile data/frames/008/frame_003.jpg --overlay skyline.jpg --json skyline.json
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--fov-v-deg` | — | Vertical FOV; if given, the profile is also output in elevation degrees. |
| `--smooth` | `9` | Median smoothing window (columns) for the horizon line. |
| `--min-coverage` | `0.25` | Coverage below which the profile is flagged unusable. |
| `--overlay PATH` | — | Write an overlay JPG (skyline drawn on the frame) for visual audit. |
| `--json` / `--case-file` | — | Write a JSON audit record / merge into the shared case file. |

---

## Project layout

```
frame-geolocator/
├── docs/architecture.md          # full design + tool catalogue
├── data/
│   ├── manifest.csv              # ground-truth dataset (media files gitignored)
│   └── frames/  videos/
├── src/frame_geolocator/tools/   # one module per tool
└── tests/
```

## Status

Early MVP — **10 tools** implemented, each with tests + CLI + JSON audit output.

The **region-restriction** path (caption → priors → candidate region) runs end to end:
`parse-context-note` + `classify-scene-cues` → `restrict-candidate-region`.  The
**terrain chain** has started with `extract-skyline-profile` (the observed ridge
silhouette).

Next on the terrain chain: `query_dem` → `synthesize_skyline` → `match_skyline` — turning
the observed skyline into a ranked viewpoint inside the candidate region.  See
[docs/architecture.md](docs/architecture.md).
