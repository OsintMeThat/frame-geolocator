# FrameGeolocator

OSINT geolocation of images and videos through an **agentic system that reproduces a
human analyst's reasoning** — explicit, auditable, and generalizable — rather than a
black-box single-geography model.

The intelligence lives in a catalogue of small, independently-testable **tools**; an
orchestrating front-end (Claude via subscription, or an autonomous Python orchestrator)
only routes between them and interprets their output. Every tool is **local & free** (no
paid LLM API). See [docs/architecture.md](docs/architecture.md) for the full design and
the tool catalogue.

> Language policy: the entire repository is in **English** (code, comments, docs).

## Requirements

- **Python ≥ 3.11**
- No system ffmpeg needed — the audio tools use a bundled ffmpeg via `imageio-ffmpeg`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[dev]'            # core + pytest/ruff
```

### Optional extras (installed per tool family)

Heavy ML deps are opt-in. Install what you need (or everything):

| Extra     | Pulls                                   | Needed by |
|-----------|-----------------------------------------|-----------|
| `ocr`     | easyocr (torch), pytesseract            | `read-text-ocr` |
| `audio`   | imageio-ffmpeg, faster-whisper          | `extract-audio`, `identify-spoken-language`, `flash-to-bang-range` |
| `sound`   | panns-inference (torch), librosa        | `classify-audio-events` |

```bash
pip install -e '.[dev,ocr,audio,sound]'   # everything
```

First run of an ML tool downloads its model once (EasyOCR ~100 MB, Whisper 75 MB–1.5 GB
by size, PANNs ~300 MB).

## Tests & lint

```bash
pytest -q            # tests synthesize their own media; real-model tests are skipped
ruff check src tests
```

Real-model tests are opt-in: `RUN_WHISPER_TESTS=1 pytest` and `RUN_PANNS_TESTS=1 pytest`.

---

## Tools (CLI commands)

Every tool is also importable from `frame_geolocator.tools` and writes a `result.json` /
`--json` audit record. Run any command with `-h` for the full option list.

### `decompose-video` — intelligent video decomposition
Turns a video into the most informative, brightness-normalized frames + a labelled
**contact sheet**. (core install)

```bash
decompose-video data/videos/001.mp4 --out data/frames/001 --frames 12 --sample-fps 2
```
| flag | meaning |
|------|---------|
| `--out DIR` | output dir (required) |
| `--frames N` | max frames to keep (default 24) |
| `--sample-fps F` | frames/sec to inspect (default 2) |
| `--no-normalize` | disable CLAHE brightness normalization |
| `--no-contact-sheet` | skip the montage |

Outputs: `frame_*.jpg`, `contact_sheet.jpg`, `result.json`.

### `read-text-ocr` — multilingual OCR
Reads signs, banners, plates. Tries EasyOCR then falls back to Tesseract automatically.
Needs the `ocr` extra. (Tesseract fallback also needs the system `tesseract-ocr` package.)

```bash
read-text-ocr data/frames/003/frame_011_t0016.10.jpg --lang fr en
read-text-ocr image.jpg --lang ru en --backends easyocr tesseract --json out.json
```
| flag | meaning |
|------|---------|
| `--lang …` | languages, ISO-639-1 (default `en fr`) |
| `--backends …` | engine priority order (default `easyocr tesseract`) |
| `--min-confidence F` | drop detections below (default 0.3) |
| `--no-preprocess` | disable upscale + contrast boost |
| `--json PATH` | write the full structured result |

### `extract-audio` — demux the waveform
Foundation of the audio branch. Outputs 16 kHz mono WAV. Needs the `audio` extra.

```bash
extract-audio data/videos/002.mp4 --out data/frames/002/audio.wav
```
| flag | meaning |
|------|---------|
| `--out PATH` | output WAV (default: alongside the video) |
| `--sample-rate N` | default 16000 |
| `--channels N` | default 1 (mono) |

Reports `No audio stream` when the video is silent.

### `identify-spoken-language` — spoken language ID + transcription
Local Whisper. Detects the spoken language (region prior) and **transcribes** speech to
horodated segments (place names said aloud are gold). Accepts a video or audio file.
Below a confidence threshold it honestly reports "no reliable speech". Needs `audio`.

```bash
identify-spoken-language data/videos/002.mp4 --model base --json words.json
```
| flag | meaning |
|------|---------|
| `--model SIZE` | tiny/base/small/medium/large-v3 (default small) |
| `--no-transcribe` | only detect the language |
| `--json PATH` | write language + transcription + segments |

### `classify-audio-events` — detect sound events
PANNs (AudioSet, 527 classes): gunshot, machine gun, artillery, explosion, siren,
aircraft, crowd, music… with timestamps. OSINT-relevant labels are marked `★`. Accepts
a video or audio file. Needs the `sound` extra.

```bash
classify-audio-events data/videos/003.mp4 --threshold 0.15 --json events.json
```
| flag | meaning |
|------|---------|
| `--top-k N` | clip-level tags to return (default 10) |
| `--threshold F` | min framewise score for an event (default 0.2) |
| `--json PATH` | write clip tags + framewise events |

### `flash-to-bang-range` — acoustic ranging
Measures the delay between a visual flash (muzzle flash, explosion) and its sound to
estimate **distance to the source** (`distance ≈ delay × 343 m/s`). Feeds geometric
fusion. Needs a clip with both a visible flash and its sound. Needs `audio`.

```bash
flash-to-bang-range data/videos/clip.mp4 --max-distance 5000 --json range.json
```
| flag | meaning |
|------|---------|
| `--speed F` | speed of sound m/s (default 343) |
| `--max-distance F` | plausibility cap in metres (default 5000) |
| `--json PATH` | write flashes, bangs, ranged events |

---

## Programmatic use

```python
from frame_geolocator.tools import decompose_video, read_text_ocr, identify_spoken_language

frames = decompose_video("data/videos/001.mp4", "data/frames/001")
text = read_text_ocr("data/frames/001/frame_000_t0002.00.jpg")
speech = identify_spoken_language("data/videos/002.mp4")
print(speech.language, speech.transcription)
```

## Project layout

```
frame-geolocator/
├── docs/architecture.md          # design + tool catalogue (~30 tools)
├── data/                         # validation dataset (media gitignored, manifest tracked)
│   ├── videos/  frames/  manifest.csv  README.md
├── src/frame_geolocator/
│   └── tools/                    # one module per tool
├── tests/
└── pyproject.toml
```

## Validation dataset

Ground-truth media (already geolocated and confirmed) live under [data/](data/) and are
used **only to validate** the pipeline — never as training data. Media files are not
committed; only `data/manifest.csv` is. See [data/README.md](data/README.md).

## Status

Early MVP. **6 tools implemented** (each with tests + CLI + JSON audit output):
`decompose_video`, `read_text_ocr`, `extract_audio`, `identify_spoken_language`,
`classify_audio_events`, `flash_to_bang_range`.

Next: landmark & architecture recognition (OpenCLIP/YOLO) and structure matching against
satellite/OSM — the bridge to actual coordinates. The hard terrain (skyline→DEM) matching
is deferred until a rural FPV case requires it. See [docs/architecture.md](docs/architecture.md).
