"""Sheet JSON helpers shared by the work-completed OCR path."""

from __future__ import annotations

import json
import re

from .jsonutil import try_parse_json


def _as_obj(text: str) -> dict:
    obj, _ = try_parse_json(text or "")
    if not isinstance(obj, dict):
        return {"title": None, "tables": [], "notes": [], "complete": False}
    obj.setdefault("title", None)
    obj.setdefault("tables", [])
    obj.setdefault("notes", [])
    obj.setdefault("complete", False)
    if not isinstance(obj["tables"], list):
        obj["tables"] = []
    if not isinstance(obj["notes"], list):
        obj["notes"] = []
    return obj


def fill_stats(obj: dict) -> dict:
    """Non-empty cell density."""
    filled = total = rows = 0
    for t in obj.get("tables") or []:
        if not isinstance(t, dict):
            continue
        for row in t.get("rows") or []:
            if not isinstance(row, list) or not any(str(c).strip() for c in row):
                continue
            rows += 1
            for c in row:
                total += 1
                if str(c).strip():
                    filled += 1
    notes = sum(1 for n in (obj.get("notes") or []) if str(n).strip())
    return {
        "filled": filled + notes,
        "total": total + notes,
        "rows": rows,
        "fill_rate": (filled / total) if total else 0.0,
        "max_cols": max(
            (len(t.get("columns") or []) for t in (obj.get("tables") or []) if isinstance(t, dict)),
            default=0,
        ),
        "n_tables": len([t for t in (obj.get("tables") or []) if isinstance(t, dict)]),
    }


def _looks_like_address(cell: str) -> bool:
    s = str(cell).strip()
    if not s:
        return False
    return bool(re.search(r"\d+.+\b(rd|road|ln|lane|st|ave|dr|drive|way)\b", s, re.I)) or (
        bool(re.match(r"^\d+\s", s)) and len(s) > 5
    )


def _looks_like_name(cell: str) -> bool:
    s = str(cell).strip()
    if not s or _looks_like_address(s):
        return False
    if "," in s:
        return True
    letters = re.sub(r"[^A-Za-z]", "", s)
    return len(letters) >= 3 and not s.startswith("$")


def _row_key(row: list) -> str:
    return "|".join(str(c).strip().lower() for c in row)


def _first_cell_key(row: list) -> str:
    if not row:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(row[0]).strip().lower())


def dedupe_contact_rows(obj: dict) -> dict:
    """One row per contact key; keep the richest duplicate (most filled cells)."""
    out = json.loads(json.dumps(obj))
    for t in out.get("tables") or []:
        if not isinstance(t, dict):
            continue
        rows = [list(r) for r in (t.get("rows") or []) if isinstance(r, list)]
        if not rows:
            continue
        best: dict[str, list] = {}
        order: list[str] = []
        for row in rows:
            if not any(str(c).strip() for c in row):
                continue
            fk = _first_cell_key(row)
            if not fk:
                fk = "row:" + _row_key(row)
            if fk not in best:
                best[fk] = row
                order.append(fk)
                continue
            prev = best[fk]
            fill_new = sum(1 for c in row if str(c).strip())
            fill_old = sum(1 for c in prev if str(c).strip())
            if fill_new > fill_old or (
                fill_new == fill_old
                and len("|".join(str(c) for c in row)) > len("|".join(str(c) for c in prev))
            ):
                best[fk] = row
        t["rows"] = [best[k] for k in order]
    return out


def prefer_sheet_tables(obj: dict) -> dict:
    """Keep the real sheet grid; drop a narrow name-list scrap after joining names."""
    tables = [dict(t) for t in (obj.get("tables") or []) if isinstance(t, dict)]
    if not tables:
        return obj

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).upper()).strip()

    def _nrows(t: dict) -> int:
        return len([
            r for r in (t.get("rows") or [])
            if isinstance(r, list) and any(str(c).strip() for c in r)
        ])

    def _row_fill(t: dict) -> float:
        filled = total = 0
        for row in t.get("rows") or []:
            if not isinstance(row, list):
                continue
            for c in row:
                total += 1
                if str(c).strip():
                    filled += 1
        return (filled / total) if total else 0.0

    def _t_kind(t: dict) -> str:
        sample = [
            str(r[0])
            for r in (t.get("rows") or [])[:8]
            if isinstance(r, list) and r and str(r[0]).strip()
        ]
        if not sample:
            return "other"
        n_addr = sum(1 for s in sample if _looks_like_address(s))
        n_name = sum(1 for s in sample if _looks_like_name(s))
        if n_addr >= max(2, (len(sample) + 1) // 2):
            return "address"
        if n_name >= max(2, (len(sample) + 1) // 2):
            return "name"
        return "other"

    narrow = [
        t for t in tables
        if len(t.get("columns") or []) <= 2 and _t_kind(t) == "name"
    ]
    wide = [t for t in tables if len(t.get("columns") or []) >= 3]
    name_bank: list[str] = []
    for n in narrow:
        for r in n.get("rows") or []:
            if isinstance(r, list) and r and str(r[0]).strip():
                name_bank.append(str(r[0]).strip())

    for w in wide:
        wr = [list(r) for r in (w.get("rows") or []) if isinstance(r, list)]
        if not wr:
            continue
        for n in narrow:
            nr = [r for r in (n.get("rows") or []) if isinstance(r, list)]
            if len(nr) == len(wr):
                for i, (wrow, nrow) in enumerate(zip(wr, nr)):
                    wname = str(wrow[0]).strip() if wrow else ""
                    nname = str(nrow[0]).strip() if nrow else ""
                    if nname and (len(nname) > len(wname) or ("," in nname and "," not in wname)):
                        wr[i] = [nname] + list(wrow[1:])
        for i, wrow in enumerate(wr):
            wname = str(wrow[0]).strip() if wrow else ""
            if not wname or ("," in wname and len(wname) > 6):
                continue
            wn = _norm(wname)
            best = None
            for nname in name_bank:
                nn = _norm(nname)
                if wn == nn:
                    continue
                if wn in nn and ("," in nname or len(nname) > len(wname) + 2):
                    best = nname
                    break
            if best:
                wr[i] = [best] + list(wrow[1:])
        w["rows"] = wr
        cols = [str(c) for c in (w.get("columns") or [])]
        if cols and cols[0].lower() in ("address", "addr", "street"):
            cols[0] = "Contact"
            w["columns"] = cols

    wide_ok = [t for t in tables if len(t.get("columns") or []) >= 3 and _row_fill(t) >= 0.45]
    name_wide = [t for t in wide_ok if _t_kind(t) == "name"]
    if name_wide:
        wide_ok = name_wide
    if wide_ok:
        best = max(
            wide_ok,
            key=lambda t: (
                len(t.get("columns") or []),
                _nrows(t),
                _row_fill(t),
            ),
        )
        kept = [best]
    else:
        ranked = sorted(
            tables,
            key=lambda t: (
                1 if _t_kind(t) == "name" else 0,
                _row_fill(t),
                _nrows(t),
                len(t.get("columns") or []),
            ),
            reverse=True,
        )
        kept = [ranked[0]] if ranked else []

    out = dict(obj)
    out["tables"] = kept
    return dedupe_contact_rows(out)
