"""Shared case-file utility.

Every tool can merge its result into a per-media ``case.json`` so the orchestrator
and analyst have a single structured place to read all accumulated evidence.

Usage (from any tool's CLI)::

    decompose-video  001.mp4 --out frames/001 --case-file frames/001/case.json
    parse-context-note "caption…"  --case-file frames/001/case.json
    read-text-ocr   frame.jpg      --case-file frames/001/case.json

The file grows with each tool run; earlier entries are never overwritten by a
different tool.  Re-running the same tool updates its own entry only.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path


def merge(case_path: str | Path, tool_name: str, result: object) -> None:
    """Merge *result* into the shared case file at *case_path*.

    Parameters
    ----------
    case_path:
        Path to the shared ``case.json``.  Created if it does not exist.
    tool_name:
        Key under which the result is stored, e.g. ``"parse_context_note"``.
    result:
        A dataclass instance or plain dict.  Dataclasses are converted with
        :func:`dataclasses.asdict`; dicts are used as-is.
    """
    case_path = Path(case_path)
    data: dict = {}
    if case_path.exists():
        data = json.loads(case_path.read_text(encoding="utf-8"))
    data[tool_name] = asdict(result) if not isinstance(result, dict) else result  # type: ignore[arg-type]
    case_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
