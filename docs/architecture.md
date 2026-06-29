# FrameGeolocator — Architecture

> OSINT geolocation of images and videos through an agentic system that reproduces a
> human analyst's reasoning. The system estimates where a media was captured, as
> precisely as the available evidence allows, with a calibrated confidence score.

---

## 1. Design philosophy

### The system is an agentic reasoning loop

The central bet of this project: **an LLM agent orchestrates and interprets tools; the
tools provide rich, structured inputs; together they reproduce a human analyst's reasoning.**

A multimodal LLM agent is excellent at:
- describing a scene, reading signs, recognizing architectural/biome cues,
- forming and prioritizing hypotheses from conflicting evidence,
- deciding *which tool to call next*, *which result to trust*, and *how to reconcile
  conflicts* — reasoning that a hand-coded rule engine cannot do honestly.
- understanding nuance: a large region with low confidence is more honest than a confident
  wrong point.

Tools are **not** trying to be smart. Each is deterministic, independently-testable, and
focuses on one numeric or extraction task:
- matching a ridge silhouette against the planet's terrain,
- computing elevation profiles, viewsheds, camera angles, distances,
- extracting text, language cues, and architectural signatures from media.

The LLM is the analyst's brain; the tools are the analyst's instruments (GIS, ruler,
satellite imagery, Street View, protractor). **The tools emit facts; the agent makes
sense of them.** We aim for **~20 small, robust tools** rather than monolithic hand-coded
logic, because the agent can reason over many structured signals in ways that no rule
engine can replicate.

### Search vs. verification: the agent's job

The hardest version of the problem — "locate this media anywhere on Earth" — is a
continuous planetary search and is not the MVP. The tractable core is:

> **Given a media + context, the agent uses tools to restrict the candidate region
> intelligently, then ranks hypotheses and returns the best position with a calibrated
> confidence score.**

The restriction step is **not** a mechanical filter; it is where the agent does its best
work. Tools provide rich inputs:
- caption toponyms + language (parsed by `parse_context_note`)
- visual cues: architecture, biome, script, sun angle (`classify_scene_cues`, etc.)
- terrain signatures: skyline, field morphology (`extract_skyline_profile`, etc.)

The agent reconciles these signals intelligently:
- *Do the caption and visual evidence agree?* If not, how do we resolve it?
- *Is the biome classification reliable* (not just a night-shot hallucination)?
- *What's the confidence* in each input?
- *What's the honest size of the region?* A large area with low confidence beats a
  confident wrong point.

Rather than a rigid rule, restriction is an **honest, auditable reasoning process** that
the agent traces for the user. This is where the system avoids black-box behavior.

---

## 2. System overview

```
                          ┌──────────────────────────┐
                          │      ORCHESTRATOR (LLM)   │
                          │  plans, calls tools,      │
                          │  interprets, scores,      │
                          │  keeps an audit log       │
                          └────────────┬─────────────┘
                                       │ function calling
      ┌────────────────┬───────────────┼────────────────┬─────────────────┐
      ▼                ▼               ▼                ▼                 ▼
 Media & Scene    Region          Terrain /        Geospatial        Geometric
 understanding    restriction     topographic      matching          fusion
 tools            tools           tools            tools             tools
```

Each tool has a **typed contract** (inputs/outputs), runs deterministically where
possible, and is **benchmarkable in isolation** against ground truth. The orchestrator
never does numeric geolocation itself — it only routes, interprets, and explains.

---

## 3. Pipeline (end to end)

The input is **not only the media**: it usually comes with a **context note** — the
social-media caption/description, source URL, and post timestamp (see `data/manifest.csv`).
This text is often the single highest-leverage cue (it may name the city outright). It is
parsed first and feeds region restriction; see "Context input" below.

```
 video/image  +  context note (caption, source, timestamp)
     │
     ▼
[0] Context parsing ── toponyms, language, event type, time from the caption
     │
     ▼
[1] Media preprocessing ── keyframes, deblur, quality, FOV estimate
     │
     ▼
[2] Scene understanding ── objects, OCR, language, architecture, biome, sun/shadow
     │
     ▼
[3] Region restriction ── language + biome + architecture → candidate bounding region
     │
     ▼
[4] Terrain analysis ── skyline profile / field morphology extracted from the media
     │
     ▼
[5] Geospatial matching ── DEM viewshed match + satellite/OSM/Street View verification
     │                       → N candidate positions
     ▼
[6] Geometric fusion ── camera pose, triangulation, scale → rank candidates
     │
     ▼
 coordinates + confidence + human-readable reasoning trace
```

In parallel, for videos with sound, an **audio branch** runs off `extract_audio`:
spoken-language ID + transcription feed **region restriction** (and may yield spoken
place names), while `flash_to_bang_range` feeds **geometric fusion** with a
distance-to-source constraint. See family G in the catalogue.

Every stage emits structured evidence into a shared **case file** (the audit log) so
the final answer is fully traceable: *why* these coordinates, *which* tools, *what*
confidence.

### Case file — the shared evidence accumulator

Each media gets a single `case.json` that every tool can write into via the
`--case-file PATH` flag.  The file is a flat JSON object keyed by tool name:

```json
{
  "decompose_video":        { "frame_count": 746, "selected": [...], … },
  "parse_context_note":     { "language": "fr", "toponyms": [...], … },
  "read_text_ocr":          { "engine_used": "easyocr", "detections": [...], … },
  "classify_scene_cues":    { "groups": [{"group": "biome", "reliable": false, …}], … },
  "identify_spoken_language": { "language": "es", "transcription": "…", … },
  "classify_audio_events":  { "clip_tags": [...], "events": [...], … },
  "restrict_candidate_region": { "candidates": [...], "confidence": 0.55, "conflicts": [] }
}
```

Rules:
- **Each tool merges its own key only** — it never touches other keys.
- Re-running a tool overwrites its own entry; earlier entries are untouched.
- The file is created on first write; the directory must already exist.

The utility lives in `frame_geolocator.case_file.merge(path, tool_name, result)`.
The orchestrator reads the accumulated case file to decide which tool to call next
and to build the final reasoning trace.

### Context input — the accompanying note (first-class)

The caption/description that ships with a media is a primary input, not an afterthought.
A tool **`parse_context_note`** extracts:
- **toponyms / place names** (NER + a gazetteer, e.g. GeoNames — local & free),
- **language** of the caption (region prior),
- **event type** (protest, explosion, flooding…) and any **timestamp**.

These become a strong **region-restriction prior** that the visual/audio/terrain evidence
must then **confirm**. Crucial discipline to stay honest (and analyst-like):

> The caption restricts *where to look*; it is **not** accepted as the answer. The system
> still has to verify the location from the media's own content. A result "confirmed only
> by the caption" is reported as **unverified**.

This separation is also how we avoid **caption leakage** in evaluation (see §7).

---

## 4. Tool catalogue (~20+ tools)

Tools are grouped by role. Each is a separate module with its own tests and its own
ground-truth benchmark. `[LLM]` = vision/LLM-backed, `[NUM]` = deterministic numeric,
`[API]` = external service.

### A. Media & frame tools
1. **`decompose_video`** `[NUM]` — intelligent video decomposition: scene-cut detection,
   sample & deduplicate frames, pick the sharpest/most informative ones, and emit
   **contact sheets** (stitched montages) so the LLM can reason over many frames at once.
   Optional brightness/contrast/gamma normalization to recover detail in dark or washed
   frames.
2. **`enhance_frame`** `[NUM]` — deblur / denoise / super-resolution / exposure recovery
   on a single frame to recover detail from degraded media.
3. **`assess_frame_quality`** `[NUM]` — score sharpness, exposure, usable content;
   gate the rest of the pipeline.
4. **`estimate_fov`** `[NUM/LLM]` — estimate camera field of view / focal proxy (drone
   FPV vs. phone vs. zoom), needed for any angular reasoning.

### B. Scene understanding tools
5. **`detect_landmarks`** `[LLM]` — locate and describe salient elements: buildings,
   towers, signs, vehicles, power lines, antennas.
6. **`read_text_ocr`** `[NUM/API]` — multilingual OCR of signs, plates, storefronts.
7. **`identify_language_script`** `[NUM/LLM]` — classify script/language from detected
   text → strong region prior.
8. **`classify_architecture`** `[LLM]` — regional building style, roof type, materials.
9. **`classify_biome_vegetation`** `[LLM/NUM]` — climate/vegetation/land-cover cues.
   *Implemented as `classify_scene_cues`* (OpenCLIP zero-shot), which also returns
   `setting` and `time_of_day` cues. Crucially it **gates biome on frame luminance**:
   on dark night footage the biome group is returned `reliable: false` instead of a
   confident wrong answer — the media-only prior for caption-less, text-less scenes.
10. **`estimate_sun_shadow`** `[NUM]` — sun azimuth/elevation from shadows → latitude
    band & time-of-day constraints.
11. **`estimate_scale_reference`** `[NUM/LLM]` — use known sizes (human ~1.7 m, car,
    door, lane width) to calibrate distances in-frame.

### C. Region restriction tools
12. **`parse_context_note`** `[NUM/LLM]` *(implemented)* — extract toponyms (NER +
    Nominatim geocoding), caption language, event type and timestamp from the
    **accompanying note**. Produces a strong region prior **to be verified, never taken as
    the answer** (see §7 leakage). The model NER is supplemented with high-precision,
    case-robust patterns (hashtags + ALL-CAPS) so an uppercase toponym such as "LA GUAIRA"
    that spaCy misses is still recovered. *Remaining limitation:* demonyms
    ("marseillaise" → Marseille) are not resolved; such a caption yields no or wrong toponym,
    which `restrict_candidate_region` flags as a conflict for the visual stage to settle.
13. **`restrict_candidate_region`** `[NUM]` *(implemented)* — emit **raw signals** for region
    restriction: all detected toponyms (geocoded or not), the language region(s), and the
    biome region(s). Zero fusion logic, zero judgment. Metadata on each signal:
    
    - Each toponym: lat/lon (if geocoded), feature type (city/country/peak), radius, and
      cross-checks: `within_language_region` (bool), `within_biome_region` (bool).
    - Language signal: the detected language and its geographic region(s).
    - Biome signal: the detected biome and its climate-zone region(s), plus a `reliable`
      flag (false if biome came from a night frame where it's unreliable).
    
    The agent (Claude) reads these signals and reasons: *Which toponym should I trust?
    Does the caption toponym conflict with visual evidence?* If the caption says Beirut
    but the biome is Arctic, the agent sees this conflict and can adjust. **Honest
    degradation** is the agent's job: it reports weak evidence as a large region with low
    confidence, never as a confident wrong point. Granularity: city radius → country →
    language family → climate zone. Architecture/sun-latitude cues are future signal sources.
14. **`enumerate_search_cells`** `[NUM]` — tile a candidate region into searchable
    cells with priority ordering.

### D. Terrain / topographic tools — the hard numeric core
15. **`extract_skyline_profile`** `[NUM]` *(implemented)* — segment horizon/ridge
    silhouette from the media → per-column boundary profile (normalized, or elevation
    degrees with a FOV). Adaptive sky/ground segmentation (no model); reports a `coverage`
    flag so sky-less/down-looking frames are marked unusable instead of fabricating a
    horizon. Optional overlay image for visual audit. The "two bumps on a ridge" signature.
16. **`query_dem`** `[NUM/API]` — fetch elevation (SRTM / Copernicus 30 m) for points
    or tiles.
17. **`synthesize_skyline`** `[NUM]` — from a DEM and a hypothesized viewpoint+heading,
    render the *expected* skyline (viewshed / panorama synthesis).
18. **`match_skyline`** `[NUM]` — correlate observed vs. synthesized skylines across
    candidate viewpoints → ranked viewpoint hypotheses with a similarity score.
19. **`extract_terrain_morphology`** `[NUM]` — field shapes, road/drainage patterns,
    slope, urban layout signature for matching.

### E. Geospatial matching & verification tools
20. **`fetch_satellite_tile`** `[API]` — Sentinel-2 (free, ~10 m) / Esri-Bing tiles for
    a cell.
21. **`query_osm_features`** `[API]` — Overpass: roads, building footprints, POIs,
    rivers, power lines for structural matching.
22. **`fetch_streetlevel`** `[API]` — **Mapillary** (free, crowdsourced) street-level
    imagery for *hypothesis verification at known coordinates* (not search — no
    reverse-image API exists). Replaces paid Google Street View.
23. **`match_structures`** `[NUM/LLM]` — compare in-media structures (footprint, road
    geometry, landmark layout) against satellite/OSM for a candidate cell.
24. **`reverse_geocode`** `[API]` — coordinates → admin context for sanity checks.

### F. Geometric fusion tools
25. **`estimate_camera_pose`** `[NUM]` — vanishing points / horizon → camera heading &
    tilt.
26. **`triangulate_from_landmarks`** `[NUM]` — resection from ≥2 identified landmarks at
    known coordinates → constrained position.
27. **`rank_hypotheses`** `[NUM]` — fuse all evidence (skyline score, structure match,
    geometry, priors) into a ranked list + **calibrated confidence**.

### G. Audio tools
A media's soundtrack carries independent geolocation evidence. All local & free.
28. **`extract_audio`** `[NUM]` — demux the waveform from a video (ffmpeg). Foundation
    for the audio branch, mirroring `decompose_video` for frames.
29. **`identify_spoken_language`** `[NUM]` — local ASR (Whisper / faster-whisper):
    detect the **spoken language/accent** (strong region prior) and **transcribe**
    speech — spoken place names (a street, a town) are high-value cues.
30. **`flash_to_bang_range`** `[NUM]` — acoustic ranging: measure the delay between a
    visual event (muzzle flash, explosion) and its sound onset; distance ≈ delay ×
    speed of sound (~343 m/s). Does not localize on its own — it feeds
    `rank_hypotheses`/`triangulate_from_landmarks` as a **distance-to-source
    constraint**. Restricts bangs to impulsive-sound windows from
    `classify_acoustic_scene` so a music/speech-only track is flagged unusable.
31. **`classify_acoustic_scene`** `[NUM/LLM]` — ambient & event sounds via PANNs/AudioSet
    (`classify_audio_events`): call to prayer (adhan), church bells, **country-specific
    emergency-siren tones**, gunfire, explosions, traffic, nature → region & time priors,
    and impulse windows for `flash_to_bang_range`.

> The catalogue intentionally exceeds 20. The MVP wires only a thin slice end to end
> (see §6); the rest are added **only when a validation failure demands them.**

---

## 5. Agents & orchestration

The system is driven by a **primary orchestrator** (Claude via MCP, or a local LLM
later) that reasons over the tool catalogue. For MVP, the orchestrator calls tools
from five roles:

| Role | Tools | Responsibility |
|------|-------|----------------|
| **Media extraction** | A, B | Decode the media into frames/audio; segment intelligible segments; assess quality. |
| **Scene understanding** | D (14–18), part of B | Extract skyline, terrain morphology, architectural/biome cues. |
| **Region restriction** | C (12–13) | Parse the caption for toponyms/language; emit candidate regions + conflict flags. |
| **Geospatial search & verify** | E, part of F | Fetch DEM/satellite/OSM; match terrain; verify hypotheses at candidate points. |
| **Geometric fusion & ranking** | F (25–27) | Apply camera geometry, triangulate, rank all hypotheses, output coordinates + confidence. |

The orchestrator (Claude) is the analyst. It:
- Decides which tool to call next based on accumulated evidence
- Interprets tool results, flags conflicts, and adjusts strategy
- Reasons over region-restriction inputs and chooses candidate regions intelligently
- Fuses evidence to rank hypotheses and output a calibrated confidence score
- Traces its reasoning for auditability

The tools emit facts and structured signals; the orchestrator makes sense of them.

---

## 6. MVP scope (hold strictly)

- **Input contract:** a media **plus a restricted candidate region** (manual hint or
  coarse prior at first). We do *not* attempt planetary search in the MVP.
- **First test material:** simple, clear videos/images (not blurry FPV). FPV-in-a-war-
  zone is the hardest case and comes later.
- **Thin end-to-end slice:** `extract_skyline_profile` → `query_dem` +
  `synthesize_skyline` → `match_skyline` → `rank_hypotheses`. This proves the
  tool-centric thesis on the single hardest tool chain.
- **Add nothing** until a measured validation failure justifies it.

---

## 7. Validation strategy

Ground truth = media **already geolocated and confirmed** from prior OSINT work. Used
for validation only — never as training data.

- **Per-tool benchmarks**, not just end-to-end. E.g. for `match_skyline`: does the true
  viewpoint appear in the top-K candidates?
- **Trivial baselines**: random position within the candidate region. We must beat it
  before claiming anything.
- **Leakage guard**: if ground-truth media were publicly published, ensure the LLM is
  *geolocating*, not *recognizing* a famous image. Prefer un-published or cropped
  variants for evaluation.
- **Caption leakage**: the context note often names the location. Score the pipeline
  **with and without** the caption. The caption may *restrict* the region, but a result
  the media's own content cannot confirm is reported **unverified** (see §3).
- **Metrics**: geodesic error (median, p90), top-K viewpoint recall, calibration of the
  confidence score (does 0.8 mean 80%?).
- Note: 10–15 medias is enough to *iterate*, not enough to *prove*. Track per-tool
  metrics to get signal from a small set.

---

## 8. Tech stack (local-first, no paid dependency)

Hard constraint: **prefer local libraries; only free or generously-free APIs; no paid
LLM API as a hard dependency.** This also reinforces the project's transparency and
reproducibility goals.

- **Python** as the foundation.
- **Numeric tools** `[NUM]`: numpy, opencv, scipy — fully local.
- **Vision / scene understanding** (replaces the "vision LLM" role, all local):
  - **OCR**: EasyOCR or PaddleOCR (multilingual, offline).
  - **Zero-shot classification** (architecture, biome, landmark type): OpenCLIP.
  - **Object / open-vocabulary detection**: YOLO (ultralytics), OWL-ViT / GroundingDINO.
  - **Scene description**: a local VLM via Ollama (Qwen2-VL, MiniCPM-V, Moondream).
- **Audio** (all local): ffmpeg (demux), librosa / numpy-scipy (signal processing,
  onset detection for flash-to-bang), faster-whisper (spoken language ID + transcription).
- **GIS / geometry**: rasterio, GDAL, numpy, shapely, pyproj; DEM from SRTM /
  Copernicus GLO-30 (free, downloadable, no key).
- **Geospatial APIs (free)**: OSM Overpass, Nominatim (reverse geocoding), **Mapillary**
  (street-level imagery, replaces paid Street View), Sentinel-2 via Copernicus Data
  Space (free account). Maxar / Google paid services are out of scope.
- **Orchestration — tools are orchestrator-agnostic.** No tool depends on any specific
  LLM. The same tools can be driven by interchangeable front-ends:
  - **Claude via the maintainer's subscription** (Claude Code / Desktop) — the best
    reasoning available, used for real investigations and development. Integrated via an
    **MCP server** exposing the tool catalogue (each tool already has a CLI, so Bash
    invocation works today; MCP is the structured path). This is *not* a paid-API hard
    dependency — it is an optional, swappable front-end.
  - **Built-in Python orchestrator** (rule-based now; local Ollama / free-tier later) —
    for autonomous, reproducible, shippable runs with no external dependency.
  The product must always run without Claude; Claude is an optional power front-end.
  Structured case-file logging throughout for auditability.

---

## 9. Known hard problems & open questions

- **Skyline → DEM matching** is the make-or-break tool. Continuous search space; needs
  good region restriction upstream and an efficient viewshed/correlation method.
- **No "find where this is" geo API exists.** Street View / satellite verify
  hypotheses; they don't search. Region restriction must come from cues + DEM matching.
- **War-zone coverage** (Ukraine) is poor in Street View and recent imagery — a reason
  to start elsewhere.
- **Camera calibration** from uncontrolled media is weak; geometric fusion should
  *rank* hypotheses, not invent absolute coordinates from nothing.
- **Confidence calibration** is a first-class deliverable, not an afterthought.

---

## 10. Guardrails

- No black-box single-geography model.
- No feature creep — every addition justified by a measured validation failure.
- Every agent and tool independently validatable against ground truth.
- Reasoning stays explicit and auditable (analyst logic, not an opaque oracle).
- **Local-first / no paid dependency**: prefer local libraries; only free or
  generously-free APIs; never a paid LLM API as a hard dependency.
