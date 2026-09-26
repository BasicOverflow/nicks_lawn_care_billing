"""Gemma-oriented OCR: deterministic row-group crops → parallel VLM reads.

White paper + black grid → split into strips of 2–4 rows, upscale, feed Gemma
at max soft tokens so each call sees a manageable strip (avoids full-page laziness).
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

# Large phone sheets × scale=2.2 row crops can exceed Pillow's default pixel cap.
Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS or 0, 200_000_000)

from .chat import chat_with_image
from .jsonutil import try_parse_json
from .orient import ask_rotate_cw_deg, materialize_upright_page
from .pipeline import richness
from .prompts import (
    TABLE_JSON_SCHEMA,
    header_columns_prompt,
    price_contact_strip_prompt,
    row_group_prompt,
)
from .table_split import make_ruled_row_crops

DEFAULT_COLUMNS = ["Contact", "Address", "New Price", "Billing Address / Notes"]


def _as_obj(text: str) -> dict:
    obj, _ = try_parse_json(text or "")
    if not isinstance(obj, dict):
        return {"title": None, "tables": [], "notes": [], "complete": False}
    obj.setdefault("title", None)
    obj.setdefault("tables", [])
    obj.setdefault("notes", [])
    obj.setdefault("complete", False)
    return obj


def _has_price_col(cols: list[str]) -> bool:
    return any(re.search(r"price|mow|hedge|\$", c, re.I) for c in cols)


def _read_columns(model_id: str, header_path: Path | None, max_tokens: int) -> list[str]:
    """Prefer header OCR when it includes a price column; else sheet defaults."""
    if header_path is None or not header_path.is_file():
        return list(DEFAULT_COLUMNS)
    try:
        text = chat_with_image(
            model_id,
            header_path,
            header_columns_prompt(),
            max_tokens=min(512, max_tokens),
            temperature=0.0,
            guided_json=True,
            json_schema={
                "type": "object",
                "properties": {"columns": {"type": "array", "items": {"type": "string"}}},
                "required": ["columns"],
            },
            schema_name="header_columns",
            max_edge=2200,
            enhance=True,
        )
        obj, _ = try_parse_json(text or "")
        cols = (obj or {}).get("columns") if isinstance(obj, dict) else None
        if isinstance(cols, list) and 3 <= len(cols) <= 8:
            cleaned = [str(c).strip() for c in cols if str(c).strip()]
            if _has_price_col(cleaned):
                return cleaned
            print(
                f"  [gemma-rows] header missing price cols={cleaned}; using defaults",
                flush=True,
            )
    except Exception as e:
        print(f"  [gemma-rows] header columns fail: {e}", flush=True)
    return list(DEFAULT_COLUMNS)


def _pad_row(row: list, n: int) -> list[str]:
    cells = [str(c) for c in row]
    if len(cells) < n:
        cells = cells + [""] * (n - len(cells))
    return cells[:n]


def _name_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _trim_hallucinated(obj: dict, *, max_rows: int, columns: list[str]) -> dict:
    """Keep at most max_rows data rows; force shared column headers."""
    out = dict(obj)
    tables = []
    for t in obj.get("tables") or []:
        if not isinstance(t, dict):
            continue
        rows = []
        for row in t.get("rows") or []:
            if not isinstance(row, list):
                continue
            cells = _pad_row(row, len(columns))
            if not any(c.strip() for c in cells):
                continue
            if cells[0].strip().lower() in ("contact", "name", "client"):
                continue
            rows.append(cells)
            if len(rows) >= max_rows:
                break
        tables.append(
            {"caption": t.get("caption"), "columns": list(columns), "rows": rows}
        )
    out["tables"] = tables
    return out


def _ocr_crop(
    model_id: str,
    path: Path,
    prompt: str,
    max_tokens: int,
    *,
    max_rows: int,
    columns: list[str],
) -> dict:
    text = chat_with_image(
        model_id,
        path,
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        guided_json=True,
        json_schema=TABLE_JSON_SCHEMA,
        max_edge=2400,
        enhance=True,
    )
    return _trim_hallucinated(_as_obj(text), max_rows=max_rows, columns=columns)


def _concat_tables(columns: list[str], parts: list[dict]) -> dict:
    """Concatenate crop tables in order; dedupe identical first-cell."""
    rows: list[list[str]] = []
    seen: set[str] = set()
    for part in parts:
        for t in part.get("tables") or []:
            if not isinstance(t, dict):
                continue
            for row in t.get("rows") or []:
                if not isinstance(row, list):
                    continue
                cells = _pad_row(row, len(columns))
                if not any(c.strip() for c in cells):
                    continue
                key = _name_key(cells[0])
                if key and key in seen:
                    for i, prev in enumerate(rows):
                        if _name_key(prev[0]) == key:
                            if sum(1 for c in cells if c.strip()) > sum(
                                1 for c in prev if c.strip()
                            ):
                                rows[i] = cells
                            break
                    continue
                if key:
                    seen.add(key)
                rows.append(cells)
    return {
        "title": None,
        "tables": [{"caption": None, "columns": list(columns), "rows": rows}],
        "notes": [],
        "complete": True,
    }


def _price_col_index(columns: list[str]) -> int:
    for i, c in enumerate(columns):
        if re.search(r"price|mow|hedge", c, re.I):
            return i
    return min(2, len(columns) - 1)


def _merge_prices(merged: dict, price_obj: dict, columns: list[str]) -> dict:
    """Overlay Contact→Price pairs from the price-strip pass onto the main table."""
    price_i = _price_col_index(columns)
    pairs: dict[str, str] = {}
    for t in price_obj.get("tables") or []:
        if not isinstance(t, dict):
            continue
        for row in t.get("rows") or []:
            if not isinstance(row, list) or not row:
                continue
            name = _name_key(row[0])
            price = str(row[1] if len(row) > 1 else "").strip()
            if name and price:
                pairs[name] = price
    if not pairs:
        return merged
    tables = []
    for t in merged.get("tables") or []:
        if not isinstance(t, dict):
            continue
        rows = []
        for row in t.get("rows") or []:
            if not isinstance(row, list):
                continue
            cells = _pad_row(row, len(columns))
            key = _name_key(cells[0])
            if key in pairs:
                cur = cells[price_i].strip() if price_i < len(cells) else ""
                # Prefer strip price when missing/weak
                if not cur or not re.search(r"\d", cur):
                    cells[price_i] = pairs[key]
            rows.append(cells)
        # Also append names only found in price strip
        have = {_name_key(r[0]) for r in rows if r}
        for key, price in pairs.items():
            if key in have:
                continue
            # recover display name from price_obj
            disp = key
            for t2 in price_obj.get("tables") or []:
                for row in t2.get("rows") or []:
                    if isinstance(row, list) and _name_key(row[0]) == key:
                        disp = str(row[0]).strip()
                        break
            cells = [""] * len(columns)
            cells[0] = disp
            cells[price_i] = price
            rows.append(cells)
        tables.append({"caption": t.get("caption"), "columns": list(columns), "rows": rows})
    out = dict(merged)
    out["tables"] = tables
    return out


def extract_by_row_groups(
    model_id: str,
    image_path: Path | str,
    *,
    rows_per_crop: int = 3,
    max_seqs: int = 6,
    max_tokens: int = 2048,
    scale: float = 2.2,
    max_crops: int = 48,
) -> dict:
    """Split sheet into overlapping 2–4-row strips and OCR in parallel.

    Live path: VLM orientation → enhanced upright page → ruled crops (+ price strip).
    """
    page = Path(image_path)
    if not page.is_file():
        raise FileNotFoundError(page)
    rows_per_crop = max(2, min(4, int(rows_per_crop)))
    workers = max(1, int(max_seqs))
    max_rows_per_crop = rows_per_crop + 1

    rotate_cw, orient_reason = ask_rotate_cw_deg(model_id, page)
    upright = materialize_upright_page(page, rotate_cw_deg=rotate_cw, enhance=True)
    print(
        f"  [gemma-rows] orient rotate_cw={rotate_cw} ({orient_reason})",
        flush=True,
    )
    print(
        f"  [gemma-rows] split rows_per_crop={rows_per_crop} overlap=1 "
        f"workers={workers} scale={scale}",
        flush=True,
    )
    header, crops, price_strip = make_ruled_row_crops(
        upright,
        rows_per_crop=rows_per_crop,
        scale=scale,
        max_crops=max_crops,
        include_header=True,
        edge_pad_rows=0.5,
        x_pad=0.02,
        overlap=1,
        enhance_crops=False,  # upright page already enhance_ruled_sheet'd
    )
    print(
        f"  [gemma-rows] crops={len(crops)} header={'yes' if header else 'no'} "
        f"price_strip={'yes' if price_strip else 'no'}",
        flush=True,
    )

    columns = _read_columns(model_id, header, max_tokens)
    print(f"  [gemma-rows] columns={columns}", flush=True)

    prompt = row_group_prompt(columns, n_rows=rows_per_crop)
    ordered: list[tuple[int, str, dict]] = []
    price_obj: dict | None = None
    try:
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(crops) + 1))) as pool:
            futs = {
                pool.submit(
                    _ocr_crop,
                    model_id,
                    path,
                    prompt,
                    max_tokens,
                    max_rows=max_rows_per_crop,
                    columns=columns,
                ): ("row", idx, label, path)
                for label, path, idx in crops
            }
            if price_strip is not None:
                futs[
                    pool.submit(
                        _ocr_crop,
                        model_id,
                        price_strip,
                        price_contact_strip_prompt(),
                        max_tokens,
                        max_rows=80,
                        columns=["Contact", "Price"],
                    )
                ] = ("price", -1, "price_strip", price_strip)

            for fut in as_completed(futs):
                kind, idx, label, path = futs[fut]
                try:
                    obj = fut.result()
                    n = sum(
                        len(t.get("rows") or [])
                        for t in (obj.get("tables") or [])
                        if isinstance(t, dict)
                    )
                    print(f"  [gemma-rows] {label}: rows={n}", flush=True)
                    if kind == "price":
                        price_obj = obj
                    else:
                        ordered.append((idx, label, obj))
                except Exception as e:
                    print(f"  [gemma-rows] {label} fail: {e}", flush=True)
    finally:
        upright.unlink(missing_ok=True)
        if header is not None:
            header.unlink(missing_ok=True)
        if price_strip is not None:
            price_strip.unlink(missing_ok=True)
        for _label, path, _i in crops:
            path.unlink(missing_ok=True)

    if not ordered and not price_obj:
        return {"title": None, "tables": [], "notes": ["no row crops"], "complete": False}

    if ordered:
        ordered.sort(key=lambda x: x[0])
        merged = _concat_tables(columns, [o for _, _, o in ordered])
    else:
        merged = {
            "title": None,
            "tables": [{"caption": None, "columns": list(columns), "rows": []}],
            "notes": [],
            "complete": True,
        }
    if price_obj is not None:
        merged = _merge_prices(merged, price_obj, columns)
    print(f"  [gemma-rows] merged richness={richness(merged)}", flush=True)
    return merged
