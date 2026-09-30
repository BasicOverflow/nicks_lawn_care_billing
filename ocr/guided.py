"""Knowledge-guided OCR: full-page image, parallel chunk prompts (no crops).

Instead of tiling/splitting the photo, send the same full image many times in
parallel. Each request asks the model to pull cell values only for a small
chunk of known Postgres entities (clients, parcels, etc.).
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .cancel import InferenceCancelled, raise_if_cancelled
from .chat import chat_with_image, novel_mode
from .jsonutil import try_parse_json
from .pipeline import fill_stats, merge_extracts, prefer_sheet_tables, _as_obj
from .work_marks import normalize_work_marks
from .prompts import (
    TABLE_JSON_SCHEMA,
    TRANSCRIBE,
    guided_chunk_prompt,
    guided_gap_prompt,
    guided_header_prompt,
    guided_work_prompt,
    guided_notes_prompt,
    guided_price_prompt,
    guided_unknown_prompt,
)

DEFAULT_CHUNK = 4
_HEADER_WORD = re.compile(
    r"\b(contact|client|name|address|price|hedge|lawn|notes?|billing|date|work|amount|completed)\b",
    re.I,
)
_VALUE_LIKE = re.compile(
    r"@|\$\s*\d|\bregular\s+mail\b|\d{1,5}\s+[A-Za-z]{3,}|,\s*[A-Za-z]",
    re.I,
)
_STOP_TOKENS = {"THE", "AND", "FOR", "MRS", "MR", "SEE"}


def _parallelism(cfg: dict) -> int:
    seqs = int((cfg.get("vllm_kwargs") or {}).get("max_num_seqs") or 4)
    return max(1, seqs)


def _max_out(cfg: dict) -> int:
    return int(cfg.get("max_output") or 4096)


def _chunks(items: list[str], size: int) -> list[list[str]]:
    size = max(1, size)
    return [items[i : i + size] for i in range(0, len(items), size)]


def _force_columns(obj: dict, columns: list[str]) -> None:
    """Keep the guided schema even if a chunk invents headers from cell values."""
    width = len(columns)
    for t in obj.get("tables") or []:
        if not isinstance(t, dict):
            continue
        t["columns"] = list(columns)
        rows = []
        for row in t.get("rows") or []:
            if not isinstance(row, list):
                continue
            cells = [str(c) for c in row]
            if len(cells) < width:
                cells = cells + [""] * (width - len(cells))
            rows.append(cells[:width])
        t["rows"] = rows


def _vision(
    model_id: str,
    cfg: dict,
    image: Path,
    prompt: str,
    *,
    reuse_prepared: bool = False,
) -> dict:
    text = chat_with_image(
        model_id,
        image,
        prompt,
        max_tokens=_max_out(cfg),
        temperature=0.0,
        guided_json=True,
        json_schema=TABLE_JSON_SCHEMA,
        reuse_prepared=reuse_prepared,
    )
    return prefer_sheet_tables(_as_obj(text))


def _work_columns(detected: list[str] | None) -> list[str]:
    """Printed work-log headers. Roster columns (address, price, billing) are not this sheet."""
    canon = ["CLIENT", "DATE & WORK COMPLETED"]
    if not detected or not headers_look_valid(detected):
        return canon
    roles = [_col_role(c) for c in detected]
    if "work" in roles and not ({"address", "price", "notes"} & set(roles)):
        return detected
    kept = [col for col, role in zip(detected, roles) if role in ("name", "work")]
    if len(kept) >= 2 and "work" in roles:
        return kept
    return canon


_BLEED = re.compile(
    r"@|\bregular mail\b|\b(?:rd|road|ln|lane|st|street|ave|avenue|dr|drive|way|blvd)\b",
    re.I,
)
_PHONE = re.compile(r"^\+?[\d\s().-]{7,}$")
_DAY_LIST = re.compile(
    r"^(?:[1-9]|[12]\d|3[01])[A-Za-z]*(?:\s+(?:[1-9]|[12]\d|3[01])[A-Za-z]*)*$"
)


def _off_sheet_value(text: str) -> bool:
    """Address, email, or phone copied from the client file rather than the work grid.

    A work cell of month days, such as "9 16 23", is not a phone number.
    """
    s = (text or "").strip()
    if not s or _DAY_LIST.fullmatch(s):
        return False
    digits = sum(ch.isdigit() for ch in s)
    if _PHONE.fullmatch(s) and digits >= 7:
        return True
    return bool(_BLEED.search(s))


def canonical_columns(sheet_kind: str) -> list[str]:
    kind = (sheet_kind or "mowing").lower()
    if kind == "hedges":
        return ["Contact", "Address", "Hedge", "Notes"]
    if kind in ("work", "work_completed"):
        return ["CLIENT", "DATE & WORK COMPLETED"]
    return ["Contact", "Address", "New Price", "Billing Address / Notes"]


def headers_look_valid(cols: list[str]) -> bool:
    """Reject columns that are actually cell values ($53, emails, street addresses)."""
    if not cols or len(cols) < 2 or len(cols) > 8:
        return False
    good = 0
    for raw in cols:
        s = str(raw).strip()
        if not s or len(s) > 48:
            return False
        if _VALUE_LIKE.search(s):
            return False
        if s.lower().startswith("col_"):
            return False
        if _HEADER_WORD.search(s):
            good += 1
    return good >= 2


def _columns_for_kind(sheet_kind: str, detected: list[str] | None) -> list[str]:
    canon = canonical_columns(sheet_kind)
    if detected and headers_look_valid(detected):
        kind = (sheet_kind or "").lower()
        if kind in ("mowing", "hedges"):
            roles = {_col_role(c) for c in detected}
            if "address" not in roles:
                return canon
        if kind in ("work", "work_completed"):
            return _work_columns(detected)
        return detected
    return canon


def _name_keys(name: str) -> list[str]:
    raw = re.sub(r"\([^)]*\)", " ", name)
    head = raw.split(",")[0] if "," in raw else raw
    keys: list[str] = []
    for part in re.findall(r"[A-Za-z0-9]+", head):
        key = part.upper()
        if len(key) >= 3 and key not in _STOP_TOKENS:
            keys.append(key)
    return keys


def _page_lines(boxes: list[tuple]) -> list[dict]:
    """Cluster RapidOCR boxes into horizontal lines, top to bottom."""
    if not boxes:
        return []
    heights = sorted(max(1.0, b[3] - b[1]) for b in boxes)
    med = heights[len(heights) // 2]
    thresh = max(8.0, med * 0.65)
    lines: list[dict] = []
    for b in sorted(boxes, key=lambda item: (item[1] + item[3]) / 2):
        cy = (b[1] + b[3]) / 2
        if lines and abs(cy - lines[-1]["y"]) <= thresh:
            ln = lines[-1]
            n = len(ln["boxes"])
            ln["y"] = (ln["y"] * n + cy) / (n + 1)
            ln["boxes"].append(b)
        else:
            lines.append({"y": cy, "boxes": [b]})
    for ln in lines:
        ordered = sorted(ln["boxes"], key=lambda item: item[0])
        ln["norm"] = re.sub(r"[^A-Z0-9]", "", "".join(str(b[4]) for b in ordered).upper())
    return lines


def _row_name(row: list, columns: list[str]) -> str:
    for i, col in enumerate(columns):
        c = str(col).lower()
        if any(word in c for word in ("contact", "name", "client")):
            return str(row[i]) if i < len(row) else ""
    return str(row[0]) if row else ""


def order_rows_like_photo(obj: dict, boxes: list[tuple]) -> dict:
    """Sort each table's rows top-to-bottom the way they sit on the photo.

    RapidOCR line positions locate the name. Rows that cannot be located keep
    their relative order after the ones that can.
    """
    lines = _page_lines(boxes)
    if len(lines) < 3 or not isinstance(obj, dict):
        return obj
    for table in obj.get("tables") or []:
        if not isinstance(table, dict):
            continue
        rows = [r for r in (table.get("rows") or []) if isinstance(r, list)]
        if len(rows) < 2:
            continue
        columns = [str(c) for c in (table.get("columns") or [])]
        claimed: set[int] = set()
        picks: list[tuple[int, int, int]] = []
        for ri, row in enumerate(rows):
            keys = [k for k in _name_keys(_row_name(row, columns)) if len(k) >= 4]
            if not keys:
                continue
            for li, ln in enumerate(lines):
                hit = sum(len(k) for k in keys if k in ln["norm"])
                if hit >= 4:
                    picks.append((hit, ri, li))
        picks.sort(key=lambda item: (-item[0], item[2]))
        assigned: dict[int, float] = {}
        for score, ri, li in picks:
            if ri in assigned or li in claimed:
                continue
            assigned[ri] = lines[li]["y"]
            claimed.add(li)
        if len(assigned) < max(3, int(len(rows) * 0.35)):
            continue
        placed: list[tuple[int, float, int, list]] = []
        for i, row in enumerate(rows):
            y = assigned.get(i)
            if y is None:
                placed.append((1, float(i), i, row))
            else:
                placed.append((0, y, i, row))
        placed.sort()
        table["rows"] = [row for *_, row in placed]
    return obj


READING_ORDER_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "properties": {"names": {"type": "array", "items": {"type": "string"}}},
    "required": ["names"],
}

READING_ORDER_PROMPT = """
Read the photographed sheet from top to bottom, the way the rows were written on the page.
List every customer or client name in that visual order. Do not alphabetize. Do not skip a row because a price, date, or note is blank.
Use the name as written, usually LAST, First. One entry per row. Do not include prices, dates, addresses, or phone numbers.
Return JSON {"names": ["name 1", "name 2"]}.
""".strip()


def _name_score(query: str, candidate: str) -> int:
    qk = [k for k in _name_keys(query) if len(k) >= 4]
    ck = [k for k in _name_keys(candidate) if len(k) >= 4]
    if not qk or not ck:
        return 0
    score = 0
    for q in qk:
        if q in ck:
            score += len(q) + 4
            continue
        for c in ck:
            if len(q) >= 4 and len(c) >= 4 and (q[:4] == c[:4]):
                score += 3
                break
    return score


def order_rows_by_reading_order(obj: dict, names: list[str]) -> dict:
    """Sort rows to follow a top-to-bottom name list from the same photo."""
    clean = [str(n).strip() for n in names if str(n).strip()]
    if len(clean) < 3 or not isinstance(obj, dict):
        return obj
    for table in obj.get("tables") or []:
        if not isinstance(table, dict):
            continue
        rows = [r for r in (table.get("rows") or []) if isinstance(r, list)]
        if len(rows) < 2:
            continue
        columns = [str(c) for c in (table.get("columns") or [])]
        unused = set(range(len(rows)))
        placed: dict[int, float] = {}
        for read_i, name in enumerate(clean):
            best_i = None
            best = 0
            for ri in unused:
                score = _name_score(name, _row_name(rows[ri], columns))
                if score > best:
                    best = score
                    best_i = ri
            if best_i is None or best < 4:
                continue
            placed[best_i] = float(read_i)
            unused.remove(best_i)
        if len(placed) < 3:
            continue
        matched = [rows[i] for i, _ in sorted(placed.items(), key=lambda item: (item[1], item[0]))]
        leftover = [rows[i] for i in range(len(rows)) if i not in placed]
        table["rows"] = matched + leftover
    return obj


def _snapshot_rows(obj: dict) -> list[list[str]]:
    out: list[list[str]] = []
    for table in obj.get("tables") or []:
        if not isinstance(table, dict):
            continue
        for row in table.get("rows") or []:
            if isinstance(row, list):
                out.append([str(c) for c in row])
    return out


def apply_photo_order(obj: dict, boxes: list[tuple], model_id: str, cfg: dict, page: Path, prefix: str) -> dict:
    """Put rows in photo order. Line boxes first; a reading pass if those miss."""
    before = _snapshot_rows(obj)
    order_rows_like_photo(obj, boxes)
    if _snapshot_rows(obj) != before:
        print(f"  {prefix} rows ordered from text positions on the photo", flush=True)
        return obj
    try:
        text = chat_with_image(
            model_id,
            page,
            READING_ORDER_PROMPT,
            max_tokens=min(_max_out(cfg), 2048),
            temperature=0.0,
            guided_json=True,
            json_schema=READING_ORDER_SCHEMA,
            schema_name="reading_order",
            reuse_prepared=True,
        )
        data = _as_obj(text)
        names = data.get("names") if isinstance(data, dict) else None
        if not isinstance(names, list):
            names = []
    except Exception as e:
        print(f"  {prefix} reading-order pass failed ({e})", flush=True)
        return obj
    print(f"  {prefix} reading-order names={len(names)}", flush=True)
    return order_rows_by_reading_order(obj, [str(n) for n in names])


def names_visible_on_page(image: Path, names: list[str], boxes: list[tuple] | None = None) -> list[str]:
    """Keep knowledge names whose tokens RapidOCR actually sees on the page."""
    if boxes is None:
        from .classical_ocr import ocr_page_boxes

        boxes = ocr_page_boxes(image)
    blob = re.sub(r"[^A-Z0-9]", "", " ".join(str(b[4]) for b in boxes).upper())
    if len(blob) < 40:
        return names
    kept: list[str] = []
    for name in names:
        if any(key in blob for key in _name_keys(name)):
            kept.append(name)
    # Classical OCR sometimes misses a sparse page — don't drop the roster then.
    if len(kept) < 3 and len(names) > 8:
        return names
    return kept or names


def _money(raw: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", str(raw).replace(",", ""))
    return float(m.group(1)) if m else None


_STREET = re.compile(
    r"\b(rd\.?|road|ln\.?|lane|st\.?|street|ave\.?|avenue|dr\.?|drive|way|blvd|court|ct\.?)\b",
    re.I,
)


def _is_dollar_amount(raw: str) -> bool:
    """True for $157 / 52.50. False for streets, emails, and notes."""
    s = str(raw).strip()
    if not s or _STREET.search(s) or "@" in s:
        return False
    if re.search(r"[A-Za-z]", s.replace("$", "")):
        return False
    amount = _money(s)
    return amount is not None and 0 < amount < 5000


def _address_index(columns: list[str]) -> int:
    for i, col in enumerate(columns):
        c = col.lower()
        if "address" in c and "billing" not in c:
            return i
    return 1 if len(columns) > 1 else -1


def _price_cell_bad(row: list, pi: int, addr_i: int) -> bool:
    raw = str(row[pi]).strip() if pi < len(row) else ""
    if not raw or not _is_dollar_amount(raw):
        return True
    if addr_i >= 0 and addr_i < len(row):
        addr = str(row[addr_i]).strip()
        if addr and raw.lower() == addr.lower():
            return True
    return False


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _price_index(columns: list[str], sheet_kind: str) -> int:
    kind = (sheet_kind or "").lower()
    for i, col in enumerate(columns):
        c = col.lower()
        if "address" in c or "note" in c or "billing" in c:
            continue
        if any(k in c for k in ("price", "hedge", "mow", "amount", "lawn")):
            return i
    if kind in ("work", "work_completed"):
        return -1
    return 2 if len(columns) > 2 else -1


def _primary_rows(obj: dict) -> tuple[dict | None, list[list]]:
    tables = [t for t in (obj.get("tables") or []) if isinstance(t, dict) and t.get("rows")]
    if not tables:
        return None, []
    table = max(tables, key=lambda t: len(t.get("rows") or []))
    rows = [list(r) for r in table.get("rows") or [] if isinstance(r, list)]
    return table, rows


def _records_for_names(records: list[dict], names: list[str]) -> list[dict]:
    by = {_name_key(r["name"]): r for r in records}
    out = []
    for name in names:
        rec = by.get(_name_key(name))
        out.append(rec or {"name": name})
    return out


def _price_reread_names(
    merged: dict,
    records: list[dict],
    columns: list[str],
    sheet_kind: str,
) -> list[dict]:
    """Blank prices, or a street written in the price column.

    Equal prices on neighboring rows are normal on these lists, so they are
    left alone. Re-asking them was overwriting a correct first read.
    """
    pi = _price_index(columns, sheet_kind)
    if pi < 0:
        return []
    addr_i = _address_index(columns)
    table, rows = _primary_rows(merged)
    targets: list[str] = []
    for row in rows:
        if not row or not str(row[0]).strip():
            continue
        if _price_cell_bad(row, pi, addr_i):
            if table is not None and pi < len(row):
                row[pi] = ""
            targets.append(str(row[0]))
    seen: set[str] = set()
    ordered = []
    for name in targets:
        k = _name_key(name)
        if k in seen:
            continue
        seen.add(k)
        ordered.append(name)
    return _records_for_names(records, ordered)


def _apply_prices(merged: dict, price_obj: dict, columns: list[str], sheet_kind: str) -> None:
    pi = _price_index(columns, sheet_kind)
    if pi < 0:
        return
    updates: dict[str, str] = {}
    for t in price_obj.get("tables") or []:
        if not isinstance(t, dict):
            continue
        for row in t.get("rows") or []:
            if not isinstance(row, list) or not row:
                continue
            price = str(row[1]).strip() if len(row) > 1 else ""
            if not _is_dollar_amount(price):
                continue
            updates[_name_key(str(row[0]))] = price
    table, _rows = _primary_rows(merged)
    if table is None:
        return
    for row in table.get("rows") or []:
        if not isinstance(row, list) or not row:
            continue
        price = updates.get(_name_key(str(row[0])))
        if not price:
            continue
        while len(row) <= pi:
            row.append("")
        row[pi] = price


def _value_index(columns: list[str], sheet_kind: str) -> int:
    kind = (sheet_kind or "").lower()
    if kind in ("work", "work_completed"):
        for i, col in enumerate(columns):
            if _col_role(col) == "work":
                return i
        return 1 if len(columns) > 1 else -1
    return _price_index(columns, sheet_kind)


def _gap_targets(
    merged: dict,
    records: list[dict],
    columns: list[str],
    sheet_kind: str,
) -> tuple[list[str], list[dict], list[str]]:
    """Names already extracted, known names with no row, and rows with an empty value cell."""
    _table, rows = _primary_rows(merged)
    extracted: list[str] = []
    have: set[str] = set()
    blank: list[str] = []
    vi = _value_index(columns, sheet_kind)
    price_sheet = (sheet_kind or "").lower() not in ("work", "work_completed")
    for row in rows:
        if not row or not str(row[0]).strip():
            continue
        name = str(row[0]).strip()
        key = _name_key(name)
        if key in have:
            continue
        have.add(key)
        extracted.append(name)
        raw = str(row[vi]).strip() if 0 <= vi < len(row) else ""
        if price_sheet:
            if not _is_dollar_amount(raw):
                blank.append(name)
        elif not raw:
            blank.append(name)
    missing = []
    for rec in records:
        key = _name_key(str(rec.get("name") or ""))
        if key and key not in have:
            missing.append(rec)
    return extracted, missing, blank


def _apply_gap_rows(merged: dict, extra: dict, columns: list[str], sheet_kind: str) -> tuple[int, int]:
    """Append names we did not have, and fill only empty value cells on names we did."""
    table, rows = _primary_rows(merged)
    if table is None:
        merged.setdefault("tables", []).append(
            {"caption": None, "columns": list(columns), "rows": []}
        )
        table = merged["tables"][-1]
        rows = table["rows"]
    dest_roles = [_col_role(c) for c in (table.get("columns") or columns)]
    by_key: dict[str, list] = {}
    for row in rows:
        if row and str(row[0]).strip():
            by_key.setdefault(_name_key(str(row[0])), row)
    vi = _value_index(columns, sheet_kind)
    price_sheet = (sheet_kind or "").lower() not in ("work", "work_completed")
    added = 0
    filled = 0
    width = len(dest_roles)
    for t in extra.get("tables") or []:
        if not isinstance(t, dict):
            continue
        src_roles = [_col_role(str(c)) for c in (t.get("columns") or columns)]
        for row in t.get("rows") or []:
            if not isinstance(row, list) or not row or not str(row[0]).strip():
                continue
            by_role: dict[str, str] = {}
            for i, cell in enumerate(row):
                role = src_roles[i] if i < len(src_roles) else ""
                if role and role not in by_role:
                    by_role[role] = str(cell)
            if "name" not in by_role:
                by_role["name"] = str(row[0])
            name = by_role["name"].strip()
            key = _name_key(name)
            if not key:
                continue
            existing = by_key.get(key)
            if existing is None:
                cells = []
                for role in dest_roles:
                    cells.append(name if role == "name" else by_role.get(role, ""))
                if len(cells) < width:
                    cells += [""] * (width - len(cells))
                cells = cells[:width]
                table.setdefault("rows", []).append(cells)
                by_key[key] = cells
                added += 1
                continue
            if vi < 0 or vi >= len(existing):
                continue
            current = str(existing[vi]).strip()
            incoming_role = "price" if price_sheet else "work"
            incoming = (by_role.get(incoming_role) or "").strip()
            if price_sheet and not _is_dollar_amount(incoming):
                continue
            if not incoming:
                continue
            if price_sheet and _is_dollar_amount(current):
                continue
            if not price_sheet and current:
                continue
            while len(existing) <= vi:
                existing.append("")
            existing[vi] = incoming
            filled += 1
    return added, filled


def _col_role(name: str) -> str:
    c = (name or "").lower()
    if any(k in c for k in ("contact", "client", "name")):
        return "name"
    if any(k in c for k in ("price", "hedge", "lawn", "amount")):
        return "price"
    if any(k in c for k in ("date", "work", "day")):
        return "work"
    if "note" in c or "billing" in c:
        return "notes"
    if "address" in c:
        return "address"
    return ""


def _append_by_role(merged: dict, extra: dict, columns: list[str]) -> int:
    """Add rows from a full-page read, lining cells up by column role."""
    table, rows = _primary_rows(merged)
    if table is None:
        merged.setdefault("tables", []).append(
            {"caption": None, "columns": list(columns), "rows": []}
        )
        table = merged["tables"][-1]
        rows = table["rows"]
    dest_roles = [_col_role(c) for c in (table.get("columns") or columns)]
    have = {_name_key(str(r[0])) for r in rows if r and str(r[0]).strip()}
    added = 0
    width = len(dest_roles)
    for t in extra.get("tables") or []:
        if not isinstance(t, dict):
            continue
        src_roles = [_col_role(str(c)) for c in (t.get("columns") or [])]
        for row in t.get("rows") or []:
            if not isinstance(row, list) or not row:
                continue
            by_role: dict[str, str] = {}
            for i, cell in enumerate(row):
                role = src_roles[i] if i < len(src_roles) else ""
                if role and role not in by_role:
                    by_role[role] = str(cell)
            if not by_role.get("name") and row:
                by_role["name"] = str(row[0])
            name = (by_role.get("name") or "").strip()
            key = _name_key(name)
            if not key or key in have:
                continue
            cells = []
            for role in dest_roles:
                if role == "name":
                    cells.append(name)
                else:
                    cells.append(by_role.get(role, ""))
            if len(cells) < width:
                cells += [""] * (width - len(cells))
            table.setdefault("rows", []).append(cells[:width])
            have.add(key)
            added += 1
    return added


def _append_unknown_rows(merged: dict, extra: dict, columns: list[str]) -> int:
    table, rows = _primary_rows(merged)
    if table is None:
        merged.setdefault("tables", []).append(
            {"caption": None, "columns": list(columns), "rows": []}
        )
        table = merged["tables"][-1]
        rows = table["rows"]
    have = {_name_key(str(r[0])) for r in rows if r and str(r[0]).strip()}
    added = 0
    width = len(table.get("columns") or columns)
    for t in extra.get("tables") or []:
        if not isinstance(t, dict):
            continue
        for row in t.get("rows") or []:
            if not isinstance(row, list) or not row or not str(row[0]).strip():
                continue
            key = _name_key(str(row[0]))
            if not key or key in have:
                continue
            cells = [str(c) for c in row]
            if len(cells) < width:
                cells += [""] * (width - len(cells))
            table.setdefault("rows", []).append(cells[:width])
            have.add(key)
            added += 1
    return added


def detect_headers(
    model_id: str,
    cfg: dict,
    image: Path,
    sheet_kind: str,
    *,
    reuse_prepared: bool = False,
) -> dict:
    """One full-page pass: title + column headers (+ any free notes)."""
    fallback = canonical_columns(sheet_kind)
    try:
        obj = _vision(
            model_id,
            cfg,
            image,
            guided_header_prompt(sheet_kind),
            reuse_prepared=reuse_prepared,
        )
    except Exception as e:
        print(f"  [guided] header fail: {e}", flush=True)
        return {"title": None, "columns": fallback, "notes": []}
    cols: list[str] = []
    for t in obj.get("tables") or []:
        if isinstance(t, dict) and t.get("columns"):
            cols = [str(c) for c in t["columns"]]
            break
    chosen = _columns_for_kind(sheet_kind, cols or None)
    if cols and chosen == fallback and cols != fallback:
        print(f"  [guided] rejected headers {cols!r} → {fallback}", flush=True)
    return {
        "title": obj.get("title"),
        "columns": chosen,
        "notes": [str(n) for n in (obj.get("notes") or []) if str(n).strip()],
    }


def extract_guided(
    model_id: str,
    cfg: dict,
    image_path: Path,
    *,
    knowledge_names: list[str] | None = None,
    knowledge_records: list[dict] | None = None,
    sheet_kind: str = "work",
    chunk_size: int = DEFAULT_CHUNK,
    wave_workers: int | None = None,
    fx_id: str | None = None,
) -> dict:
    """Run knowledge-guided parallel full-image OCR. Returns sheet JSON dict."""
    page = Path(image_path)
    if not page.is_file():
        raise FileNotFoundError(page)
    raise_if_cancelled()
    from .chat import prepare_image

    original = page
    page = prepare_image(original)
    try:
        return _guided_once(
            model_id,
            cfg,
            page,
            knowledge_names=knowledge_names,
            knowledge_records=knowledge_records,
            sheet_kind=sheet_kind,
            chunk_size=chunk_size,
            wave_workers=wave_workers,
            fx_id=fx_id or original.stem,
        )
    finally:
        if page != original:
            page.unlink(missing_ok=True)


def _sheet_names(model_id: str, cfg: dict, page: Path) -> list[str]:
    """Names written on the page, top to bottom. Not the client roster."""
    text = chat_with_image(
        model_id,
        page,
        READING_ORDER_PROMPT,
        max_tokens=min(_max_out(cfg), 2048),
        temperature=0.0,
        guided_json=True,
        json_schema=READING_ORDER_SCHEMA,
        schema_name="reading_order",
        reuse_prepared=True,
    )
    data = _as_obj(text)
    names = data.get("names") if isinstance(data, dict) else None
    if not isinstance(names, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        text_name = str(name).strip()
        key = _name_key(text_name)
        if not text_name or not key or key in seen:
            continue
        seen.add(key)
        out.append(text_name)
    return out


def _work_cells(obj: dict, columns: list[str]) -> list[list[str]]:
    """One name plus one work cell. Extra cells are more days or job notes, not new columns."""
    width = max(len(columns), 1)
    rows: list[list[str]] = []
    for table in obj.get("tables") or []:
        if not isinstance(table, dict):
            continue
        for row in table.get("rows") or []:
            if not isinstance(row, list) or not any(str(c).strip() for c in row):
                continue
            cells = [str(c).strip() for c in row]
            if width == 1:
                cells = cells[:1]
            elif len(cells) > width:
                head = cells[: width - 1]
                tail = " ".join(part for part in cells[width - 1 :] if part)
                cells = head + [tail]
            if len(cells) < width:
                cells += [""] * (width - len(cells))
            cells = cells[:width]
            for i in range(1, width):
                if _off_sheet_value(cells[i]):
                    cells[i] = ""
                else:
                    cells[i] = normalize_work_marks(cells[i])
            if cells[0]:
                rows.append(cells)
    return rows


def _take_work_row(name: str, pool: list[list[str]], columns: list[str]) -> list[str] | None:
    best_i = None
    best = 0
    for i, row in enumerate(pool):
        score = _name_score(name, _row_name(row, columns))
        if score > best:
            best = score
            best_i = i
    if best_i is None or best < 4:
        return None
    return pool.pop(best_i)


def _extract_work_sheet(
    model_id: str,
    cfg: dict,
    page: Path,
    *,
    sheet_kind: str,
    chunk_size: int,
    workers: int,
    prefix: str,
    t0: float,
) -> dict:
    """Columns and cells from the work-completed photo. The roster is not a column source."""
    header = detect_headers(model_id, cfg, page, sheet_kind, reuse_prepared=True)
    columns = header["columns"]
    print(f"  {prefix} work headers: {columns!r} title={header.get('title')!r}", flush=True)
    try:
        names = _sheet_names(model_id, cfg, page)
    except Exception as e:
        print(f"  {prefix} sheet names failed ({e})", flush=True)
        names = []
    print(f"  {prefix} names written on sheet: {len(names)}", flush=True)
    pool: list[list[str]] = []
    if names:
        groups = _chunks(names, chunk_size)

        def _one(group: list[str]) -> dict:
            raise_if_cancelled()
            return _vision(
                model_id,
                cfg,
                page,
                guided_work_prompt(group, columns, header.get("title")),
                reuse_prepared=True,
            )

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(groups)))) as pool_ex:
            futs = [pool_ex.submit(_one, group) for group in groups]
            for fut in as_completed(futs):
                try:
                    got = fut.result()
                except InferenceCancelled:
                    raise
                except Exception as e:
                    print(f"  {prefix} work chunk fail: {e}", flush=True)
                    continue
                pool.extend(_work_cells(got, columns))
        rows: list[list[str]] = []
        for name in names:
            hit = _take_work_row(name, pool, columns)
            if hit is None:
                rows.append([name] + [""] * (len(columns) - 1))
            else:
                rows.append(hit)
        rows.extend(pool)
    else:
        try:
            got = _vision(
                model_id,
                cfg,
                page,
                guided_work_prompt([], columns, header.get("title")),
                reuse_prepared=True,
            )
            rows = _work_cells(got, columns)
        except Exception as e:
            print(f"  {prefix} work transcribe fail: {e}", flush=True)
            rows = []
    merged = {
        "title": header.get("title"),
        "tables": [{"caption": None, "columns": columns, "rows": rows}],
        "notes": header.get("notes") or [],
        "complete": bool(rows),
    }
    st = fill_stats(merged)
    print(
        f"  {prefix} work sheet in {time.time() - t0:.0f}s — rows={st['rows']} cols={columns!r}",
        flush=True,
    )
    return merged


def _guided_once(
    model_id: str,
    cfg: dict,
    page: Path,
    *,
    knowledge_names: list[str] | None,
    knowledge_records: list[dict] | None,
    sheet_kind: str,
    chunk_size: int,
    wave_workers: int | None,
    fx_id: str | None,
) -> dict:
    prefix = f"[{fx_id or page.stem}]"
    workers = wave_workers if wave_workers is not None else _parallelism(cfg)
    if knowledge_records:
        all_records = [r for r in knowledge_records if str(r.get("name") or "").strip()]
    else:
        all_records = [{"name": n.strip()} for n in (knowledge_names or []) if str(n).strip()]
    all_names = [r["name"] for r in all_records]
    t0 = time.time()
    page_boxes: list[tuple] = []
    try:
        from .classical_ocr import ocr_page_boxes

        page_boxes = ocr_page_boxes(page)
    except Exception as e:
        print(f"  {prefix} page boxes failed ({e})", flush=True)
    if (sheet_kind or "").lower() in ("work", "work_completed"):
        return _extract_work_sheet(
            model_id,
            cfg,
            page,
            sheet_kind=sheet_kind,
            chunk_size=chunk_size,
            workers=workers,
            prefix=prefix,
            t0=t0,
        )
    visible = set(names_visible_on_page(page, all_names, page_boxes)) if all_names else set()
    records = [r for r in all_records if r["name"] in visible] if visible else []
    if all_names and not records:
        records = list(all_records)

    print(
        f"  {prefix} guided OCR: {len(records)}/{len(all_records)} names on page, "
        f"chunk={chunk_size}, workers={workers}, kind={sheet_kind}",
        flush=True,
    )

    # 1) Header / title (single full-page call)
    raise_if_cancelled()
    header = detect_headers(model_id, cfg, page, sheet_kind, reuse_prepared=True)
    columns = header["columns"]
    print(
        f"  {prefix} headers: {columns!r} title={header.get('title')!r}",
        flush=True,
    )

    if not records:
        # Fallback: one unguided full-page pull if knowledge is empty
        print(f"  {prefix} no knowledge — single full-page transcribe", flush=True)
        try:
            obj = _vision(model_id, cfg, page, TRANSCRIBE, reuse_prepared=True)
        except Exception as e:
            print(f"  {prefix} full-page fail: {e}", flush=True)
            obj = _as_obj("{}")
        obj["title"] = obj.get("title") or header.get("title")
        if header.get("notes"):
            obj["notes"] = list(dict.fromkeys([*(obj.get("notes") or []), *header["notes"]]))
        return apply_photo_order(prefer_sheet_tables(obj), page_boxes, model_id, cfg, page, prefix)

    groups = _chunks(records, chunk_size)
    hist: list[dict] = []
    parts: list[dict] = []

    def _one(gi: int, group: list) -> tuple[int, dict, dict]:
        raise_if_cancelled()
        prompt = guided_chunk_prompt(
            group,
            columns=columns,
            sheet_kind=sheet_kind,
            title=header.get("title"),
        )
        obj = _vision(model_id, cfg, page, prompt, reuse_prepared=True)
        _force_columns(obj, columns)
        st = fill_stats(obj)
        return gi, obj, st

    print(f"  {prefix} {len(groups)} parallel full-image chunks…", flush=True)
    raise_if_cancelled()
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(groups) or 1))) as pool:
        futs = {
            pool.submit(_one, i, g): i for i, g in enumerate(groups)
        }
        for fut in as_completed(futs):
            try:
                gi, obj, st = fut.result()
            except InferenceCancelled:
                raise
            except Exception as e:
                print(f"  {prefix} chunk fail: {e}", flush=True)
                continue
            parts.append(obj)
            hist.append(
                {
                    "pass": f"chunk-{gi}",
                    "cells": st["filled"],
                    "rows": st["rows"],
                    "fill_rate": round(st["fill_rate"], 3),
                    "names": [r["name"] if isinstance(r, dict) else r for r in groups[gi]],
                }
            )
            print(
                f"  {prefix} chunk-{gi}: rows={st['rows']} filled={st['filled']} "
                f"({', '.join(hist[-1]['names'][:3])}{'…' if len(groups[gi]) > 3 else ''})",
                flush=True,
            )

    # Optional margin/notes pass (still full image, once)
    raise_if_cancelled()
    try:
        notes_obj = _vision(
            model_id,
            cfg,
            page,
            guided_notes_prompt(sheet_kind, header.get("title")),
            reuse_prepared=True,
        )
        if notes_obj.get("notes"):
            parts.append({"title": None, "tables": [], "notes": notes_obj["notes"], "complete": False})
    except Exception as e:
        print(f"  {prefix} notes pass fail: {e}", flush=True)

    if not parts:
        merged = {
            "title": header.get("title"),
            "tables": [{"caption": None, "columns": columns, "rows": []}],
            "notes": header.get("notes") or [],
            "complete": False,
        }
    else:
        merged = prefer_sheet_tables(
            merge_extracts(*(json.dumps(p) for p in parts))
        )
        if header.get("title") and not merged.get("title"):
            merged["title"] = header["title"]
        # Prefer detected column headers when merge invented col_N
        for t in merged.get("tables") or []:
            if not isinstance(t, dict):
                continue
            cols = [str(c) for c in (t.get("columns") or [])]
            if not cols or all(c.lower().startswith("col_") for c in cols):
                t["columns"] = columns
            # Ensure column width matches
            width = len(t.get("columns") or columns)
            rows = []
            for row in t.get("rows") or []:
                if not isinstance(row, list):
                    continue
                cells = [str(c) for c in row]
                if len(cells) < width:
                    cells = cells + [""] * (width - len(cells))
                rows.append(cells[:width])
            t["rows"] = rows
        if header.get("notes"):
            seen = {str(n).lower() for n in (merged.get("notes") or [])}
            for n in header["notes"]:
                if n.lower() not in seen:
                    merged.setdefault("notes", []).append(n)

    # 2) Re-read prices that are blank or far from the filed price
    reread = _price_reread_names(merged, records, columns, sheet_kind)
    n_reread = 0
    if reread:
        print(f"  {prefix} price re-read {len(reread)} rows…", flush=True)
        price_groups = _chunks(reread, chunk_size)

        def _price_one(group: list) -> dict:
            raise_if_cancelled()
            return _vision(
                model_id,
                cfg,
                page,
                guided_price_prompt(group, sheet_kind=sheet_kind),
                reuse_prepared=True,
            )

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(price_groups)))) as pool:
            futs = [pool.submit(_price_one, g) for g in price_groups]
            for fut in as_completed(futs):
                try:
                    got = fut.result()
                except InferenceCancelled:
                    raise
                except Exception as e:
                    print(f"  {prefix} price re-read fail: {e}", flush=True)
                    continue
                _apply_prices(merged, got, columns, sheet_kind)
                n_reread += 1
        hist.append({"pass": "price-reread", "groups": len(price_groups), "rows": len(reread)})

    # Contacts on the sheet who are not in the knowledge roster.
    # A page that matches only a handful of known names is mostly people the
    # roster does not list. Read the whole table once; that is the same call
    # an upload can make.
    raise_if_cancelled()
    n_unknown = 0
    if len(records) < 8 and len(all_records) >= 15:
        print(
            f"  {prefix} few roster names on page — reading the whole table",
            flush=True,
        )
        try:
            full = _vision(model_id, cfg, page, TRANSCRIBE, reuse_prepared=True)
            n_full = _append_by_role(merged, full, columns)
            n_unknown += n_full
            print(f"  {prefix} whole-table rows added: {n_full}", flush=True)
            hist.append({"pass": "sparse-page", "rows": n_full})
        except Exception as e:
            print(f"  {prefix} whole-table read fail: {e}", flush=True)
    try:
        extra = _vision(
            model_id,
            cfg,
            page,
            guided_unknown_prompt(
                [r["name"] for r in records],
                columns=columns,
                sheet_kind=sheet_kind,
                title=header.get("title"),
            ),
            reuse_prepared=True,
        )
        _force_columns(extra, columns)
        n_unknown = _append_unknown_rows(merged, extra, columns)
        print(f"  {prefix} unknown contacts added: {n_unknown}", flush=True)
        hist.append({"pass": "unknown", "rows": n_unknown})
    except Exception as e:
        print(f"  {prefix} unknown pass fail: {e}", flush=True)

    gap_added = 0
    gap_filled = 0
    for gap_i in range(1, 3):
        raise_if_cancelled()
        extracted, missing, blank = _gap_targets(merged, records, columns, sheet_kind)
        if gap_i > 1 and not missing and not blank:
            break
        print(
            f"  {prefix} gap pass {gap_i}: extracted={len(extracted)} "
            f"missing={len(missing)} blank={len(blank)}",
            flush=True,
        )
        targets: list[tuple[list, list[str], bool]] = []
        for group in _chunks(missing, chunk_size):
            targets.append((group, [], False))
        for group in _chunks(blank, chunk_size):
            targets.append(([], group, False))
        if gap_i == 1:
            targets.append(([], [], True))

        def _gap_one(item: tuple[list, list[str], bool]) -> dict:
            raise_if_cancelled()
            miss, blanks, others = item
            return _vision(
                model_id,
                cfg,
                page,
                guided_gap_prompt(
                    miss,
                    blanks,
                    extracted,
                    columns=columns,
                    sheet_kind=sheet_kind,
                    title=header.get("title"),
                    ask_others=others,
                ),
                reuse_prepared=True,
            )

        round_added = 0
        round_filled = 0
        if targets:
            with ThreadPoolExecutor(max_workers=max(1, min(workers, len(targets)))) as pool:
                futs = [pool.submit(_gap_one, item) for item in targets]
                for fut in as_completed(futs):
                    try:
                        got = fut.result()
                    except InferenceCancelled:
                        raise
                    except Exception as e:
                        print(f"  {prefix} gap pass {gap_i} fail: {e}", flush=True)
                        continue
                    added, filled = _apply_gap_rows(merged, got, columns, sheet_kind)
                    round_added += added
                    round_filled += filled
        gap_added += round_added
        gap_filled += round_filled
        hist.append(
            {
                "pass": f"gap-{gap_i}",
                "missing": len(missing),
                "blank": len(blank),
                "added": round_added,
                "filled": round_filled,
            }
        )
        print(
            f"  {prefix} gap pass {gap_i}: added={round_added} filled={round_filled}",
            flush=True,
        )
        if round_added == 0 and round_filled == 0:
            break

    apply_photo_order(merged, page_boxes, model_id, cfg, page, prefix)
    st = fill_stats(merged)
    elapsed = round(time.time() - t0, 2)
    print(
        f"  {prefix} guided done in {elapsed}s — rows={st['rows']} "
        f"filled={st['filled']} fill={st['fill_rate']:.0%} chunks={len(groups)}",
        flush=True,
    )
    merged["_guided_meta"] = {
        "elapsed_s": elapsed,
        "chunks": len(groups),
        "knowledge": len(all_records),
        "names_on_page": len(records),
        "price_reread": len(reread),
        "unknown_added": n_unknown,
        "gap_added": gap_added,
        "gap_filled": gap_filled,
        "sheet_kind": sheet_kind,
        "passes": hist,
        "novel": novel_mode(),
    }
    return merged
