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
from .prompts import (
    TABLE_JSON_SCHEMA,
    TRANSCRIBE,
    guided_chunk_prompt,
    guided_gap_prompt,
    guided_header_prompt,
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


def _vision(model_id: str, cfg: dict, image: Path, prompt: str) -> dict:
    text = chat_with_image(
        model_id,
        image,
        prompt,
        max_tokens=_max_out(cfg),
        temperature=0.0,
        guided_json=True,
        json_schema=TABLE_JSON_SCHEMA,
    )
    return prefer_sheet_tables(_as_obj(text))


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


def names_visible_on_page(image: Path, names: list[str]) -> list[str]:
    """Keep knowledge names whose tokens RapidOCR actually sees on the page."""
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


def detect_headers(model_id: str, cfg: dict, image: Path, sheet_kind: str) -> dict:
    """One full-page pass: title + column headers (+ any free notes)."""
    fallback = canonical_columns(sheet_kind)
    try:
        obj = _vision(model_id, cfg, image, guided_header_prompt(sheet_kind))
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
    sheet_kind: str = "mowing",
    chunk_size: int = DEFAULT_CHUNK,
    wave_workers: int | None = None,
    fx_id: str | None = None,
) -> dict:
    """Run knowledge-guided parallel full-image OCR. Returns sheet JSON dict."""
    page = Path(image_path)
    if not page.is_file():
        raise FileNotFoundError(page)
    raise_if_cancelled()

    prefix = f"[{fx_id or page.stem}]"
    workers = wave_workers if wave_workers is not None else _parallelism(cfg)
    if knowledge_records:
        all_records = [r for r in knowledge_records if str(r.get("name") or "").strip()]
    else:
        all_records = [{"name": n.strip()} for n in (knowledge_names or []) if str(n).strip()]
    all_names = [r["name"] for r in all_records]
    t0 = time.time()
    visible = set(names_visible_on_page(page, all_names)) if all_names else set()
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
    header = detect_headers(model_id, cfg, page, sheet_kind)
    columns = header["columns"]
    print(
        f"  {prefix} headers: {columns!r} title={header.get('title')!r}",
        flush=True,
    )

    if not records:
        # Fallback: one unguided full-page pull if knowledge is empty
        print(f"  {prefix} no knowledge — single full-page transcribe", flush=True)
        try:
            obj = _vision(model_id, cfg, page, TRANSCRIBE)
        except Exception as e:
            print(f"  {prefix} full-page fail: {e}", flush=True)
            obj = _as_obj("{}")
        obj["title"] = obj.get("title") or header.get("title")
        if header.get("notes"):
            obj["notes"] = list(dict.fromkeys([*(obj.get("notes") or []), *header["notes"]]))
        return prefer_sheet_tables(obj)

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
        obj = _vision(model_id, cfg, page, prompt)
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
            full = _vision(model_id, cfg, page, TRANSCRIBE)
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
