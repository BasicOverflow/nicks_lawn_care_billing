"""Convert freeform OCR/Markdown/HTML model output into sheet table JSON."""

from __future__ import annotations

import re
from html.parser import HTMLParser

from .jsonutil import try_parse_json


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[dict] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._cell = ""
        self._row: list[str] = []
        self._rows: list[list[str]] = []
        self._is_header_row = False
        self._columns: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        t = tag.lower()
        if t == "table":
            self._in_table = True
            self._rows = []
            self._columns = []
        elif self._in_table and t == "tr":
            self._in_row = True
            self._row = []
            self._is_header_row = False
        elif self._in_table and t in ("td", "th"):
            self._in_cell = True
            self._cell = ""
            if t == "th":
                self._is_header_row = True

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("td", "th") and self._in_cell:
            self._row.append(re.sub(r"\s+", " ", self._cell).strip())
            self._in_cell = False
        elif t == "tr" and self._in_row:
            if self._is_header_row and not self._columns and any(self._row):
                self._columns = list(self._row)
            elif any(c.strip() for c in self._row):
                self._rows.append(list(self._row))
            self._in_row = False
        elif t == "table" and self._in_table:
            cols = self._columns or [f"col_{i+1}" for i in range(max((len(r) for r in self._rows), default=0))]
            width = len(cols)
            norm_rows = []
            for r in self._rows:
                cells = list(r) + [""] * max(0, width - len(r))
                norm_rows.append(cells[:width])
            self.tables.append({"caption": None, "columns": cols, "rows": norm_rows})
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell += data


def _md_pipe_tables(text: str) -> list[dict]:
    tables: list[dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?\s*[-:| ]+\|", lines[i + 1]):
            header = [c.strip() for c in line.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i]:
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if any(row):
                    rows.append(row)
                i += 1
            width = len(header)
            norm = [r + [""] * max(0, width - len(r)) for r in rows]
            tables.append({"caption": None, "columns": header, "rows": [r[:width] for r in norm]})
            continue
        i += 1
    return tables


def freeform_to_extract(text: str) -> dict:
    """Best-effort: JSON → HTML tables → Markdown pipes → notes blob."""
    obj, _ = try_parse_json(text or "")
    if isinstance(obj, dict) and (obj.get("tables") or obj.get("rows")):
        obj.setdefault("title", None)
        obj.setdefault("tables", [])
        obj.setdefault("notes", [])
        obj.setdefault("complete", False)
        return obj

    parser = _TableParser()
    try:
        parser.feed(text or "")
    except Exception:
        parser.tables = []
    tables = parser.tables or _md_pipe_tables(text or "")
    notes: list[str] = []
    if not tables:
        # Keep raw text so scorers can still keyword-match
        blob = (text or "").strip()
        if blob:
            notes.append(blob[:12000])
    return {"title": None, "tables": tables, "notes": notes, "complete": False}
