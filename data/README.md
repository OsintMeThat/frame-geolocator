# Validation dataset

Ground-truth media used **only to validate** the geolocation pipeline — never for
training. See [../docs/architecture.md](../docs/architecture.md) §7.

## Layout

```
data/
├── videos/      # raw example videos        (gitignored — heavy / sensitive)
├── frames/      # frames extracted by tools (gitignored — generated)
├── manifest.csv # ground-truth metadata     (COMMITTED)
└── README.md
```

Media files are **not committed** (size + sensitivity). Only `manifest.csv` is, so the
ground truth and difficulty labels are version-controlled and shareable.

## Adding an example

1. Drop the file in `videos/`.
2. Add a row to `manifest.csv`.

### `manifest.csv` columns

| column          | meaning                                                       |
|-----------------|---------------------------------------------------------------|
| `id`            | stable short id, e.g. `ex001`                                 |
| `filename`      | file in `videos/`                                            |
| `media_type`    | `video` or `image`                                          |
| `source`        | where it came from (URL, investigation ref)                  |
| `known_lat`     | confirmed latitude (decimal degrees)                         |
| `known_lon`     | confirmed longitude (decimal degrees)                        |
| `location_name` | human-readable place                                         |
| `difficulty`    | `easy` / `medium` / `hard` (FPV-blurry-warzone = `hard`)     |
| `confirmed`     | `yes` / `no` — is the ground truth verified?                |
| `notes`         | anything useful (cues present, occlusions, leakage risk)     |

## Start simple

For the first tools, prefer **easy** media: clear daylight footage with visible
horizon/skyline, signs, or distinctive architecture. Blurry FPV war-zone footage is the
hardest case and comes later.
