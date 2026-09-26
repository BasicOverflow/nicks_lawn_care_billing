"""Minimal JSON helpers for OCR (no gold scoring)."""

from __future__ import annotations

import json
import re
from typing import Any


def try_parse_json(text: str) -> tuple[Any | None, str]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group(0)), m.group(0)
            except json.JSONDecodeError:
                pass
        repaired = _close_truncated_json(raw)
        if repaired is not None:
            return repaired, json.dumps(repaired, ensure_ascii=False)
    return None, raw


def _close_truncated_json(raw: str) -> Any | None:
    s = raw.strip()
    if not s.startswith("{"):
        return None
    for cut in range(len(s), max(len(s) - 800, 20), -1):
        chunk = s[:cut].rstrip()
        chunk = re.sub(r",\s*$", "", chunk)
        if chunk.count('"') % 2 == 1:
            chunk += '"'
        opens = chunk.count("{") - chunk.count("}")
        opens_a = chunk.count("[") - chunk.count("]")
        if opens < 0 or opens_a < 0:
            continue
        candidate = chunk + ("]" * opens_a) + ("}" * opens)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
