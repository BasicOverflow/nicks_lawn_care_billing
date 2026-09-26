"""Production sheet OCR: parallel tiles → unify → completeness reviews.

Stops when reviews stop improving fill (non-empty cells) / row coverage, or
the model marks complete. No gold answers. Output matches sheet table grids.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .chat import chat_text, chat_with_image, novel_mode
from .cancel import InferenceCancelled, raise_if_cancelled
from .classical_ocr import (
    annotate_confidence,
    majority_vote_prices,
    ocr_page_boxes,
    overlay_prices_by_order,
    read_price_from_cell,
    read_prices_from_cells,
    read_prices_top_to_bottom,
)
from .freeform import freeform_to_extract
from .geometry import (
    assemble_table_from_ocr_boxes,
    make_price_cell_crops,
    make_row_crops,
    page_size,
)
from .jsonutil import try_parse_json
from .prompts import (
    DOC_OCR_PROMPT,
    FAITHFUL_MD_PROMPT,
    LAYOUT_KIND_PROMPT,
    LAYOUT_KIND_SCHEMA,
    REGION_JSON_SCHEMA,
    SMART_SPLIT_PROMPT,
    TRANSCRIBE,
    address_focus_prompt,
    column_strip_prompt,
    completeness_prompt,
    grid_repair_prompt,
    name_column_repair_prompt,
    price_focus_prompt,
    row_window_prompt,
    struct_from_text_prompt,
    surname_repair_prompt,
    tail_completeness_prompt,
    tile_prompt,
    unify_prompt,
)
from .tiles import (
    contrast_crop,
    make_column_strips,
    make_price_strip,
    make_region_crops,
    make_row_windows,
    make_tiles,
    upscale_labeled_crops,
)

def _noop(*a, **k):
    return None

# Production: no HTML report galleries
def overlay_norm_regions(*a, **k):
    return Path('.')
def overlay_bands(*a, **k):
    return Path('.')
def panel_from_crop(*a, **k):
    return {}
def build_preprocess_gallery(*a, **k):
    return []


TILE_BANDS = 6
ROW_WINDOWS = 4
GROWTH_STOP = 0.015
MIN_REVIEWS_BEFORE_PLATEAU = 4
FIELD_FOCUS_ROUNDS = 2
VOTE_TEMPS = (0.0,)  # numeric vote handled by classical majority
ROW_CHUNK = 3
MAX_ROW_CHUNKS = 16


def _novel_on(cfg: dict | None = None) -> bool:
    if cfg and cfg.get("novel"):
        return True
    return novel_mode()


def _parallelism(cfg: dict) -> int:
    seqs = int((cfg.get("vllm_kwargs") or {}).get("max_num_seqs") or 1)
    return max(1, min(TILE_BANDS, seqs))


def _max_out(cfg: dict) -> int:
    return int(cfg.get("max_output") or 4096)


def _pretty(text: str) -> str:
    obj, _ = try_parse_json(text)
    if isinstance(obj, dict):
        return json.dumps(obj, indent=2, ensure_ascii=False)
    return (text or "").strip()


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


def cell_count(obj: dict) -> int:
    n = 0
    for t in obj.get("tables") or []:
        if not isinstance(t, dict):
            continue
        for row in t.get("rows") or []:
            if isinstance(row, list):
                n += sum(1 for c in row if str(c).strip())
    n += sum(1 for note in obj.get("notes") or [] if str(note).strip())
    return n


def fill_stats(obj: dict) -> dict:
    """Non-empty cell density — catches empty address/price columns the cell-count miss."""
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
        "max_cols": max((len(t.get("columns") or []) for t in (obj.get("tables") or []) if isinstance(t, dict)), default=0),
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


def col0_kind(obj: dict) -> str:
    """Classify first column of the primary table: name | address | other."""
    for t in obj.get("tables") or []:
        if not isinstance(t, dict):
            continue
        sample = [
            str(r[0])
            for r in (t.get("rows") or [])[:8]
            if isinstance(r, list) and r and str(r[0]).strip()
        ]
        if not sample:
            continue
        n_addr = sum(1 for s in sample if _looks_like_address(s))
        n_name = sum(1 for s in sample if _looks_like_name(s))
        if n_addr >= max(2, (len(sample) + 1) // 2):
            return "address"
        if n_name >= max(2, (len(sample) + 1) // 2):
            return "name"
    return "other"


def names_look_truncated(obj: dict) -> bool:
    """True when most contact cells lack 'Surname, Given' form."""
    names = []
    for t in obj.get("tables") or []:
        if not isinstance(t, dict):
            continue
        for r in (t.get("rows") or [])[:20]:
            if isinstance(r, list) and r and _looks_like_name(str(r[0])):
                names.append(str(r[0]).strip())
    if len(names) < 3:
        return False
    with_comma = sum(1 for n in names if "," in n)
    return with_comma / len(names) < 0.5


def grid_needs_repair(obj: dict) -> bool:
    """True when headers look invented or street Address is missing beside Billing."""
    for t in obj.get("tables") or []:
        if not isinstance(t, dict):
            continue
        cols = [str(c).strip().lower() for c in (t.get("columns") or [])]
        if not cols:
            continue
        invented = any(
            c in ("days", "email", "notes", "col_5", "col_6") or c.startswith("col_")
            for c in cols
        )
        has_billing = any("billing" in c for c in cols)
        has_street = any(
            (c == "address" or c.startswith("address "))
            and "billing" not in c
            for c in cols
        )
        # Billing present without a real Address column → dropped street col
        if has_billing and not has_street:
            return True
        # Invented Days/Email columns (mowing sheets don't print these)
        if invented and has_billing:
            return True
        # Address column present but mostly emails / REGULAR MAIL
        if has_street:
            idx = next(
                (
                    i
                    for i, c in enumerate(cols)
                    if (c == "address" or "address" in c)
                    and "billing" not in c
                ),
                None,
            )
            if idx is not None:
                sample = [
                    str(r[idx]).strip()
                    for r in (t.get("rows") or [])[:15]
                    if isinstance(r, list) and idx < len(r) and str(r[idx]).strip()
                ]
                if sample:
                    bad = sum(
                        1
                        for s in sample
                        if "@" in s
                        or "mail" in s.lower()
                        or re.fullmatch(r"[\d\-\(\)\s]+", s)
                    )
                    if bad / len(sample) >= 0.5:
                        return True
    return False


def prefer_sheet_tables(obj: dict) -> dict:
    """Keep the real sheet grid(s); drop narrow scraps after joining names.

    Uploaded sheets are tables. If a wide grid exists alongside a name list,
    copy fuller names into the grid (zip or fuzzy), then drop scraps.
    """
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

    # Join name lists into wider grids (same row count OR fuzzy first-name match).
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

    # Prefer a single wide name-leading grid (avoid duplicate table copies).
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


def richness(obj: dict) -> tuple:
    """Prefer filled wide grids over empty wide shells or narrow scraps."""
    st = fill_stats(obj)
    wide_bonus = st["max_cols"] if st["fill_rate"] >= 0.45 else 0
    return (wide_bonus, st["filled"], st["fill_rate"], st["rows"])


def _row_key(row: list) -> str:
    return "|".join(str(c).strip().lower() for c in row)


def _first_cell_key(row: list) -> str:
    if not row:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(row[0]).strip().lower())


def merge_extracts(*texts: str) -> dict:
    """Union tables by caption+columns; dedupe identical rows. Notes are unique."""
    title = None
    tables: list[dict] = []
    index: dict[str, dict] = {}
    notes: list[str] = []
    seen_notes: set[str] = set()
    for text in texts:
        obj = _as_obj(text)
        if title is None and obj.get("title"):
            title = obj["title"]
        for t in obj["tables"]:
            if not isinstance(t, dict):
                continue
            cols = [str(c) for c in (t.get("columns") or [])]
            cap = str(t.get("caption") or "")
            key = cap.lower() + "\n" + "|".join(c.lower() for c in cols)
            slot = index.get(key)
            if slot is None:
                slot = {"caption": t.get("caption"), "columns": cols, "rows": [], "_seen": set(), "_by_first": {}}
                index[key] = slot
                tables.append(slot)
            for row in t.get("rows") or []:
                if not isinstance(row, list):
                    continue
                cells = [str(c) for c in row]
                if not any(c.strip() for c in cells):
                    continue
                width = len(slot["columns"]) or len(cells)
                if len(slot["columns"]) < len(cells):
                    slot["columns"].extend(
                        f"col_{i+1}" for i in range(len(slot["columns"]), len(cells))
                    )
                    width = len(slot["columns"])
                if len(cells) < width:
                    cells = cells + [""] * (width - len(cells))
                cells = cells[:width]
                rk = _row_key(cells)
                if rk in slot["_seen"]:
                    continue
                # Same first-cell (client): keep the longer/more-filled row.
                fk = _first_cell_key(cells)
                if fk and fk in slot["_by_first"]:
                    prev_i = slot["_by_first"][fk]
                    prev = slot["rows"][prev_i]
                    if sum(1 for c in cells if c.strip()) <= sum(1 for c in prev if c.strip()):
                        continue
                    slot["_seen"].discard(_row_key(prev))
                    slot["rows"][prev_i] = cells
                    slot["_seen"].add(rk)
                    continue
                slot["_seen"].add(rk)
                if fk:
                    slot["_by_first"][fk] = len(slot["rows"])
                slot["rows"].append(cells)
        for note in obj["notes"]:
            s = str(note).strip()
            if s and s.lower() not in seen_notes:
                seen_notes.add(s.lower())
                notes.append(s)
    clean = []
    for slot in tables:
        slot.pop("_seen", None)
        slot.pop("_by_first", None)
        clean.append(slot)
    # Keep all fragment tables for unify — do not drop Contact/name scraps here.
    return {"title": title, "tables": clean, "notes": notes, "complete": False}


def _pick_better(a: dict, b: dict) -> dict:
    a = prefer_sheet_tables(a)
    b = prefer_sheet_tables(b)
    return a if richness(a) >= richness(b) else b


def _is_freeform(cfg: dict) -> bool:
    style = str(cfg.get("prompt_style") or "").lower()
    return style in ("doc_ocr_freeform", "markdown", "freeform")


def _vision_json(model_id: str, cfg: dict, image: Path, prompt: str, temperature: float = 0.0) -> str:
    freeform = _is_freeform(cfg)
    text = chat_with_image(
        model_id,
        image,
        DOC_OCR_PROMPT if freeform and prompt == TRANSCRIBE else prompt,
        max_tokens=_max_out(cfg),
        temperature=temperature,
        guided_json=not freeform,
    )
    if freeform:
        return json.dumps(freeform_to_extract(text), ensure_ascii=False)
    return text


def _smart_split_regions(model_id: str, cfg: dict, page: Path) -> list[dict]:
    """Ask the VLM for normalized page regions; empty on failure."""
    _ = cfg
    try:
        raw = chat_with_image(
            model_id,
            page,
            SMART_SPLIT_PROMPT,
            max_tokens=1024,
            temperature=0.0,
            guided_json=True,
            json_schema=REGION_JSON_SCHEMA,
            schema_name="smart_regions",
        )
    except Exception as e:
        print(f"  smart-split fail: {e}", flush=True)
        return []
    obj, _ = try_parse_json(raw or "")
    if not isinstance(obj, dict):
        # Freeform models may wrap regions in prose — last-ditch JSON hunt already done
        return []
    regs = obj.get("regions") or []
    out: list[dict] = []
    for r in regs:
        if not isinstance(r, dict):
            continue
        try:
            y0, y1 = float(r.get("y0", 0)), float(r.get("y1", 1))
            x0, x1 = float(r.get("x0", 0)), float(r.get("x1", 1))
        except Exception:
            continue
        if y1 - y0 < 0.04:
            continue
        out.append({
            "label": str(r.get("label") or f"r{len(out)+1}"),
            "x0": max(0.0, min(1.0, x0)),
            "y0": max(0.0, min(1.0, y0)),
            "x1": max(0.0, min(1.0, x1)),
            "y1": max(0.0, min(1.0, y1)),
        })
    return out


def _layout_kind_regions(model_id: str, cfg: dict, page: Path) -> list[dict]:
    """Pass-1 layout: table / handwriting / text / price regions."""
    _ = cfg
    try:
        raw = chat_with_image(
            model_id,
            page,
            LAYOUT_KIND_PROMPT,
            max_tokens=1200,
            temperature=0.0,
            guided_json=True,
            json_schema=LAYOUT_KIND_SCHEMA,
            schema_name="layout_kinds",
        )
    except Exception as e:
        print(f"  layout-kind fail: {e}", flush=True)
        return []
    obj, _ = try_parse_json(raw or "")
    if not isinstance(obj, dict):
        return []
    out: list[dict] = []
    for r in obj.get("regions") or []:
        if not isinstance(r, dict):
            continue
        try:
            y0, y1 = float(r.get("y0", 0)), float(r.get("y1", 1))
            x0, x1 = float(r.get("x0", 0)), float(r.get("x1", 1))
        except Exception:
            continue
        if y1 - y0 < 0.03:
            continue
        out.append({
            "label": str(r.get("label") or f"r{len(out)+1}"),
            "kind": str(r.get("kind") or "text").lower(),
            "x0": max(0.0, min(1.0, x0)),
            "y0": max(0.0, min(1.0, y0)),
            "x1": max(0.0, min(1.0, x1)),
            "y1": max(0.0, min(1.0, y1)),
        })
    return out


def _ocr_then_struct(model_id: str, cfg: dict, page: Path) -> dict:
    """Faithful MD/HTML transcription → text-only JSON structuring."""
    try:
        md = chat_with_image(
            model_id,
            page,
            FAITHFUL_MD_PROMPT,
            max_tokens=_max_out(cfg),
            temperature=0.0,
            guided_json=False,
        )
    except Exception as e:
        print(f"  faithful-ocr fail: {e}", flush=True)
        return _as_obj("{}")
    # Prefer freeform HTML/MD tables first; then ask model to structure
    staged = freeform_to_extract(md or "")
    if fill_stats(staged)["filled"] >= 8:
        return prefer_sheet_tables(staged)
    try:
        raw = chat_text(
            model_id,
            struct_from_text_prompt(md or ""),
            max_tokens=_max_out(cfg),
            temperature=0.0,
            guided_json=not _is_freeform(cfg),
        )
        if _is_freeform(cfg):
            return prefer_sheet_tables(freeform_to_extract(raw))
        return prefer_sheet_tables(_as_obj(raw))
    except Exception as e:
        print(f"  struct-from-text fail: {e}", flush=True)
        return prefer_sheet_tables(staged)


def _flag_uncertain_disagreements(base: dict, donors: list[dict]) -> dict:
    """Mark cells where independent passes disagree (multi-pass evidence)."""
    if not donors:
        return base
    out = json.loads(json.dumps(base))
    uncertain: list[str] = list(out.get("notes") or [])
    tables = out.get("tables") or []
    if not tables or not isinstance(tables[0], dict):
        return out
    cols = tables[0].get("columns") or []
    rows = tables[0].get("rows") or []
    donor_tables = []
    for d in donors:
        ts = d.get("tables") or []
        if ts and isinstance(ts[0], dict):
            donor_tables.append(ts[0])
    for ri, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        key = _first_cell_key(row) if row else ""
        for ci, cell in enumerate(row):
            vals = [str(cell).strip()]
            for dt in donor_tables:
                drows = dt.get("rows") or []
                match = None
                for dr in drows:
                    if not isinstance(dr, list) or not dr:
                        continue
                    if key and _first_cell_key(dr) == key:
                        match = dr
                        break
                if match is None and ri < len(drows) and isinstance(drows[ri], list):
                    match = drows[ri]
                if match is not None and ci < len(match):
                    vals.append(str(match[ci]).strip())
            nonempty = [v for v in vals if v and v.upper() != "[UNCLEAR]"]
            uniq = {re.sub(r"\s+", " ", v).lower() for v in nonempty}
            if len(uniq) >= 2:
                col = cols[ci] if ci < len(cols) else f"col{ci}"
                uncertain.append(
                    f"UNCERTAIN row={ri+1} {col}: " + " | ".join(sorted(uniq)[:4])
                )
    # de-dupe notes
    seen = set()
    notes = []
    for n in uncertain:
        if n not in seen:
            seen.add(n)
            notes.append(n)
    out["notes"] = notes[-80:]
    return out


def _cell_vote(values: list[str]) -> str:
    """Pick most common non-empty cell; prefer price-like / longer address-like."""
    nonempty = [v.strip() for v in values if str(v).strip()]
    if not nonempty:
        return ""
    counts: dict[str, int] = {}
    for v in nonempty:
        counts[v] = counts.get(v, 0) + 1
    # Prefer values that look like $xx over bare 3-digit slips when tied
    def key(v: str) -> tuple:
        priceish = 1 if re.search(r"\$\s*\d{2}\b", v) or re.fullmatch(r"\$?\d{2}", v) else 0
        return (counts[v], priceish, len(v))

    return max(nonempty, key=key)


def majority_merge_extracts(texts: list[str]) -> dict:
    """Cell-wise majority vote across sample extracts (aligned by first-cell key)."""
    objs = [_as_obj(t) for t in texts if t]
    if not objs:
        return _as_obj("{}")
    if len(objs) == 1:
        return prefer_sheet_tables(objs[0])
    # Seed from richest sample's primary table structure
    base = prefer_sheet_tables(max(objs, key=richness))
    tables_out = []
    for ti, bt in enumerate(base.get("tables") or []):
        if not isinstance(bt, dict):
            continue
        cols = [str(c) for c in (bt.get("columns") or [])]
        # Collect candidate rows by first-cell key across samples
        by_key: dict[str, list[list[str]]] = {}
        order: list[str] = []
        for obj in objs:
            tables = obj.get("tables") or []
            src = tables[ti] if ti < len(tables) and isinstance(tables[ti], dict) else None
            if src is None:
                # fall back to widest table
                cands = [t for t in tables if isinstance(t, dict)]
                src = max(cands, key=lambda t: len(t.get("columns") or []), default=None)
            if src is None:
                continue
            for row in src.get("rows") or []:
                if not isinstance(row, list) or not any(str(c).strip() for c in row):
                    continue
                cells = [str(c) for c in row]
                fk = _first_cell_key(cells) or _row_key(cells)
                if fk not in by_key:
                    by_key[fk] = []
                    order.append(fk)
                by_key[fk].append(cells)
        voted_rows = []
        width = len(cols) or max((len(r) for rows in by_key.values() for r in rows), default=0)
        for fk in order:
            samples = by_key[fk]
            row = []
            for ci in range(width):
                vals = [r[ci] if ci < len(r) else "" for r in samples]
                row.append(_cell_vote(vals))
            if any(c.strip() for c in row):
                voted_rows.append(row)
        tables_out.append({"caption": bt.get("caption"), "columns": cols or [f"col_{i+1}" for i in range(width)], "rows": voted_rows})
    out = {
        "title": next((o.get("title") for o in objs if o.get("title")), None),
        "tables": tables_out,
        "notes": base.get("notes") or [],
        "complete": False,
    }
    return prefer_sheet_tables(out)


def _vision_vote(model_id: str, cfg: dict, image: Path, prompt: str) -> str:
    texts = []
    for temp in VOTE_TEMPS:
        try:
            texts.append(_vision_json(model_id, cfg, image, prompt, temp))
        except Exception:
            continue
    if not texts:
        raise RuntimeError("vision vote: all samples failed")
    if len(texts) == 1:
        return texts[0]
    return json.dumps(majority_merge_extracts(texts), ensure_ascii=False)


def _normalize_price_cells(obj: dict) -> dict:
    """Fix only clear 533→53 OCR slips in price columns. Never turn $150 into $15."""
    out = json.loads(json.dumps(obj))
    for t in out.get("tables") or []:
        if not isinstance(t, dict):
            continue
        cols = [str(c).lower() for c in (t.get("columns") or [])]
        price_idxs = [
            i
            for i, c in enumerate(cols)
            if any(k in c for k in ("price", "hedge", "mow", "amount", "$"))
        ]
        for row in t.get("rows") or []:
            if not isinstance(row, list):
                continue
            idxs = price_idxs or []
            for i in idxs:
                if i >= len(row):
                    continue
                raw = str(row[i]).strip()
                if not raw or not re.search(r"\d", raw):
                    continue
                if any(k in raw.lower() for k in ("rd", "road", "mail", "@", "lane", "st.", "ave")):
                    continue
                compact = re.sub(r"[^\d]", "", raw)
                # Only rewrite when third digit duplicates second (533, not 150).
                if len(compact) == 3 and compact[1] == compact[2]:
                    dual = int(compact[:2])
                    if 15 <= dual <= 99:
                        row[i] = f"${dual}"
    return out


def _so_far(obj: dict, limit: int = 16000) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) > limit:
        return s[:limit] + "\n…[truncated]"
    return s


def _apply_vision(
    model_id: str,
    cfg: dict,
    page: Path,
    prompt: str,
    current: dict,
    *,
    vote: bool = False,
) -> dict:
    if vote:
        text = _vision_vote(model_id, cfg, page, prompt)
    else:
        text = _vision_json(model_id, cfg, page, prompt, 0.0)
    fresh = prefer_sheet_tables(_as_obj(text))
    combined = prefer_sheet_tables(merge_extracts(json.dumps(current), text))
    return prefer_sheet_tables(_pick_better(_pick_better(fresh, combined), current))


def _run_crop_wave(
    model_id: str,
    cfg: dict,
    crops: list[tuple[str, Path]],
    prompt_for,
    workers: int,
    prefix: str,
    hist: list[dict],
) -> dict[str, str]:
    """Parallel vision OCR over labeled crops. Returns label→text; unlinks paths."""
    out: dict[str, str] = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(crops) or 1))) as pool:
            futs = {
                pool.submit(_vision_json, model_id, cfg, path, prompt_for(label), 0.0): label
                for label, path in crops
            }
            for fut in as_completed(futs):
                label = futs[fut]
                try:
                    text = fut.result()
                except InferenceCancelled:
                    raise
                except Exception as e:
                    print(f"  {prefix} {label} fail: {e}", flush=True)
                    continue
                out[label] = text
                st = fill_stats(_as_obj(text))
                hist.append({"pass": label, "cells": st["filled"], "fill_rate": round(st["fill_rate"], 3)})
                print(
                    f"  {prefix} {label}: filled={st['filled']} rows={st['rows']} fill={st['fill_rate']:.0%}",
                    flush=True,
                )
    finally:
        for _, path in crops:
            path.unlink(missing_ok=True)
    return out


def run_fixture(
    model_id: str,
    cfg: dict,
    fx: dict,
    max_passes: int,
    *,
    wave_workers: int | None = None,
    asset_dir: Path | None = None,
    image_path: Path | None = None,
) -> dict:
    """Extract one sheet from the photo only. Gold is never used to fill or repair."""
    _ = asset_dir  # HTML galleries removed in production package
    if image_path is not None:
        page = Path(image_path)
    else:
        page = Path(fx["image"])
    if not page.is_file():
        raise FileNotFoundError(page)
    raise_if_cancelled()
    workers = wave_workers if wave_workers is not None else _parallelism(cfg)
    workers = max(1, min(TILE_BANDS, workers))
    prefix = f"[{fx['id']}]"
    hist: list[dict] = []
    panels: list[dict] = []
    review_passes = max(4, int(max_passes))

    t0 = time.time()

    # 1) Full-page seed
    raise_if_cancelled()
    print(f"  {prefix} full-page seed…", flush=True)
    try:
        seed_text = _vision_json(model_id, cfg, page, TRANSCRIBE, 0.0)
        merged = prefer_sheet_tables(_as_obj(seed_text))
    except InferenceCancelled:
        raise
    except Exception as e:
        print(f"  {prefix} full-page fail: {e}", flush=True)
        merged = _as_obj("{}")
    st = fill_stats(merged)
    hist.append({"pass": "full-page", "cells": st["filled"], "fill_rate": round(st["fill_rate"], 3)})
    print(f"  {prefix} full-page: filled={st['filled']} rows={st['rows']} fill={st['fill_rate']:.0%}", flush=True)

    novel = _novel_on(cfg)
    novel_donors: list[dict] = [merged]

    # 1a) Novel: OCR→JSON staging + layout-kind crops (upscaled / contrast)
    if novel:
        raise_if_cancelled()
        print(f"  {prefix} novel: faithful OCR → structure…", flush=True)
        try:
            staged = _ocr_then_struct(model_id, cfg, page)
            novel_donors.append(staged)
            merged = prefer_sheet_tables(merge_extracts(json.dumps(merged), json.dumps(staged)))
            st = fill_stats(merged)
            hist.append({
                "pass": "ocr-then-struct",
                "cells": st["filled"],
                "fill_rate": round(st["fill_rate"], 3),
            })
            print(
                f"  {prefix} ocr→struct: filled={st['filled']} rows={st['rows']} "
                f"fill={st['fill_rate']:.0%}",
                flush=True,
            )
        except InferenceCancelled:
            raise
        except Exception as e:
            print(f"  {prefix} ocr→struct fail: {e}", flush=True)

        raise_if_cancelled()
        print(f"  {prefix} novel: layout-kind regions…", flush=True)
        kind_regs = _layout_kind_regions(model_id, cfg, page)
        hist.append({"pass": "layout-kind", "n_regions": len(kind_regs)})
        if kind_regs and asset_dir is not None:
            try:
                overlay = overlay_norm_regions(
                    page, kind_regs, asset_dir / f"{fx['id']}_layout_kind.jpg"
                )
                panels.append({
                    "kind": "layout",
                    "label": f"layout kinds ({len(kind_regs)})",
                    "image": str(overlay.as_posix()),
                    "extract": json.dumps(kind_regs, indent=2),
                })
            except Exception:
                pass
        if kind_regs:
            kind_crops = make_region_crops(page, kind_regs, pad=0.02)
            # 2× upscale after crop; contrast variant for handwriting/price
            up_crops = upscale_labeled_crops(kind_crops, scale=2.0)
            extra: list[tuple[str, Path]] = []
            for label, path in list(kind_crops):
                kind = next(
                    (r.get("kind") for r in kind_regs if str(r.get("label") or "") in label),
                    "",
                )
                if kind in ("handwriting", "price", "hw"):
                    try:
                        extra.append((f"{label}-ctr", contrast_crop(path)))
                    except Exception:
                        pass
            all_kind = kind_crops + up_crops + extra

            def _kind_prompt(lab: str) -> str:
                if "price" in lab.lower():
                    return column_strip_prompt("price")
                if "hand" in lab.lower() or "hw" in lab.lower():
                    return (
                        TRANSCRIBE
                        + "\n\nThis crop is mostly HANDWRITING. "
                        "Transcribe carefully; use [UNCLEAR] when unsure."
                    )
                return tile_prompt()

            kt = _run_crop_wave(
                model_id, cfg, all_kind, _kind_prompt, workers, prefix, hist
            )
            if kt:
                donor = merge_extracts(*kt.values())
                novel_donors.append(donor)
                merged = prefer_sheet_tables(
                    merge_extracts(json.dumps(merged), *kt.values())
                )

    # 1b) VL smart-split: model returns bounding boxes → crop those regions first
    print(f"  {prefix} smart-split regions…", flush=True)
    smart_regs = _smart_split_regions(model_id, cfg, page)
    hist.append({"pass": "smart-split", "n_regions": len(smart_regs)})
    if smart_regs and asset_dir is not None:
        try:
            overlay = overlay_norm_regions(
                page, smart_regs, asset_dir / f"{fx['id']}_smart_split.jpg"
            )
            panels.append({
                "kind": "smart-split",
                "label": f"VL bbox regions ({len(smart_regs)})",
                "image": str(overlay.as_posix()),
                "extract": json.dumps(smart_regs, indent=2),
            })
        except Exception as e:
            print(f"  {prefix} smart-split overlay fail: {e}", flush=True)
    if smart_regs:
        smart_crops = make_region_crops(page, smart_regs, pad=0.03 if novel else 0.01)
        if novel:
            smart_crops = smart_crops + upscale_labeled_crops(smart_crops, scale=2.0)
        print(f"  {prefix} {len(smart_crops)} smart crops…", flush=True)
        # Snapshot crops for the HTML report before the wave deletes temps
        if asset_dir is not None:
            for label, path in smart_crops:
                if "-up" in label:
                    continue
                try:
                    panels.append(
                        panel_from_crop(
                            path,
                            asset_dir / f"{fx['id']}_{label}.jpg",
                            kind="smart-crop",
                            label=label,
                            extract="(pending OCR…)",
                        )
                    )
                except Exception:
                    pass
        smart_text = _run_crop_wave(
            model_id,
            cfg,
            smart_crops,
            lambda _label: tile_prompt(),
            workers,
            prefix,
            hist,
        )
        if smart_text:
            merged = prefer_sheet_tables(
                merge_extracts(json.dumps(merged), *smart_text.values())
            )
            novel_donors.append(_as_obj(next(iter(smart_text.values()))))
            st = fill_stats(merged)
            hist.append({
                "pass": "smart-merge",
                "cells": st["filled"],
                "fill_rate": round(st["fill_rate"], 3),
            })
            print(
                f"  {prefix} smart merged — filled={st['filled']} rows={st['rows']} "
                f"fill={st['fill_rate']:.0%}",
                flush=True,
            )
            # Back-fill extract text on smart-crop panels
            for p in panels:
                if p.get("kind") == "smart-crop" and p.get("label") in smart_text:
                    p["extract"] = _pretty(smart_text[p["label"]])
                    p["score"] = next(
                        (h["cells"] for h in hist if h["pass"] == p["label"]), None
                    )

    # 2) Dense overlapping tiles (+ retry empty bands)
    tile_overlap = 0.20 if novel else 0.14
    tiles = make_tiles(page, bands=TILE_BANDS, overlap=tile_overlap)
    if novel:
        # Crop-then-2× upscale pass (extra visual real estate for handwriting)
        up_tiles = upscale_labeled_crops(tiles, scale=2.0)
        tiles = tiles + up_tiles
    print(f"  {prefix} {len(tiles)} tiles in parallel (workers={workers})", flush=True)
    tile_text: dict[str, str] = {}

    def _ocr_tile(label: str, path: Path) -> tuple[str, str, dict]:
        text = _vision_json(model_id, cfg, path, tile_prompt(), 0.0)
        return label, text, fill_stats(_as_obj(text))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_ocr_tile, label, path): (label, path) for label, path in tiles}
            empty_labels: list[str] = []
            for fut in as_completed(futs):
                label, path = futs[fut]
                try:
                    label, text, st = fut.result()
                except InferenceCancelled:
                    raise
                except Exception as e:
                    print(f"  {prefix} {label} fail: {e}", flush=True)
                    empty_labels.append(label)
                    continue
                tile_text[label] = text
                hist.append({"pass": label, "cells": st["filled"], "fill_rate": round(st["fill_rate"], 3)})
                print(
                    f"  {prefix} {label}: filled={st['filled']} rows={st['rows']} fill={st['fill_rate']:.0%}",
                    flush=True,
                )
                if st["rows"] == 0 or st["filled"] == 0:
                    empty_labels.append(label)
            # Retry empty tiles once with a nudged crop
            if empty_labels:
                print(f"  {prefix} retry empty tiles: {', '.join(empty_labels)}", flush=True)
                retry_tiles = make_tiles(page, bands=TILE_BANDS, overlap=0.18, y_nudge=0.02)
                retry_map = {lab: p for lab, p in retry_tiles}
                try:
                    with ThreadPoolExecutor(max_workers=workers) as pool2:
                        rfuts = {
                            pool2.submit(_ocr_tile, lab, retry_map[lab]): lab
                            for lab in empty_labels
                            if lab in retry_map
                        }
                        for fut in as_completed(rfuts):
                            lab = rfuts[fut]
                            try:
                                lab, text, st = fut.result()
                            except InferenceCancelled:
                                raise
                            except Exception as e:
                                print(f"  {prefix} {lab} retry fail: {e}", flush=True)
                                continue
                            if st["filled"] > fill_stats(_as_obj(tile_text.get(lab, ""))).get("filled", 0):
                                tile_text[lab] = text
                            hist.append({
                                "pass": f"{lab}-retry",
                                "cells": st["filled"],
                                "fill_rate": round(st["fill_rate"], 3),
                            })
                            print(
                                f"  {prefix} {lab}-retry: filled={st['filled']} rows={st['rows']} "
                                f"fill={st['fill_rate']:.0%}",
                                flush=True,
                            )
                finally:
                    for _, p in retry_tiles:
                        p.unlink(missing_ok=True)
        if asset_dir is not None:
            for label, path in tiles:
                dest = asset_dir / f"{fx['id']}_{label}.jpg"
                if path.is_file():
                    shutil.copy2(path, dest)
                panels.append({
                    "kind": "tile",
                    "label": label,
                    "image": str(dest.as_posix()),
                    "extract": _pretty(tile_text.get(label, "")),
                    "score": next((h["cells"] for h in hist if h["pass"] == label), None),
                })
            # Band overlay on full page for the fixed tile grid
            try:
                from PIL import Image as _PILImage, ImageOps as _PILOps

                _img = _PILOps.exif_transpose(_PILImage.open(page)).convert("RGB")
                _w, _h = _img.size
                if _w > _h * 1.15:
                    _img = _img.rotate(-90, expand=True, fillcolor=(255, 255, 255))
                    _w, _h = _img.size
                band_rects = []
                for i in range(TILE_BANDS):
                    y0 = int(_h * (i / TILE_BANDS - 0.07))
                    y1 = int(_h * ((i + 1) / TILE_BANDS + 0.07))
                    band_rects.append((max(0, y0), min(_h, y1)))
                band_path = overlay_bands(
                    page, band_rects, asset_dir / f"{fx['id']}_tile_bands.jpg"
                )
                panels.insert(
                    next(
                        (i for i, p in enumerate(panels) if p.get("kind") == "tile"),
                        len(panels),
                    ),
                    {
                        "kind": "tile-grid",
                        "label": f"fixed tile bands ({TILE_BANDS})",
                        "image": str(band_path.as_posix()),
                        "extract": "Even horizontal bands with overlap (fallback when smart-split is weak).",
                    },
                )
            except Exception as e:
                print(f"  {prefix} tile-band overlay fail: {e}", flush=True)
    finally:
        for _, path in tiles:
            path.unlink(missing_ok=True)

    merged = merge_extracts(json.dumps(merged), *tile_text.values())
    st = fill_stats(merged)
    hist.append({"pass": "tiles-merge", "cells": st["filled"], "fill_rate": round(st["fill_rate"], 3)})
    print(
        f"  {prefix} tiles merged — filled={st['filled']} rows={st['rows']} fill={st['fill_rate']:.0%}",
        flush=True,
    )

    # 3) Unify to one sheet grid
    print(f"  {prefix} unify…", flush=True)
    t_u = time.time()
    try:
        text = _vision_json(model_id, cfg, page, unify_prompt(_so_far(merged)), 0.0)
        unified = prefer_sheet_tables(_as_obj(text))
        draft = prefer_sheet_tables(merged)
        merged = _pick_better(unified, draft)
        if col0_kind(merged) != "name" and col0_kind(draft) == "name":
            merged = draft
        merged = prefer_sheet_tables(merged)
    except Exception as e:
        print(f"  {prefix} unify fail: {e}", flush=True)
        merged = prefer_sheet_tables(merged)
    st = fill_stats(merged)
    hist.append({
        "pass": "unify",
        "cells": st["filled"],
        "fill_rate": round(st["fill_rate"], 3),
        "elapsed_s": round(time.time() - t_u, 2),
    })
    print(
        f"  {prefix} unify: filled={st['filled']} rows={st['rows']} "
        f"fill={st['fill_rate']:.0%} ({time.time() - t_u:.0f}s)",
        flush=True,
    )

    if col0_kind(merged) == "address":
        print(f"  {prefix} name-column repair…", flush=True)
        try:
            merged = _apply_vision(
                model_id, cfg, page, name_column_repair_prompt(_so_far(merged)), merged
            )
        except Exception as e:
            print(f"  {prefix} name-repair fail: {e}", flush=True)

    if names_look_truncated(merged):
        print(f"  {prefix} surname repair…", flush=True)
        try:
            repaired = _apply_vision(
                model_id, cfg, page, surname_repair_prompt(_so_far(merged)), merged
            )
            if not names_look_truncated(repaired) or richness(repaired) >= richness(merged):
                merged = repaired
        except Exception as e:
            print(f"  {prefix} surname-repair fail: {e}", flush=True)

    if grid_needs_repair(merged):
        print(f"  {prefix} grid repair (restore sheet columns)…", flush=True)
        try:
            repaired = _apply_vision(
                model_id, cfg, page, grid_repair_prompt(_so_far(merged)), merged
            )
            if not grid_needs_repair(repaired):
                merged = repaired
            else:
                merged = _pick_better(repaired, merged)
        except Exception as e:
            print(f"  {prefix} grid-repair fail: {e}", flush=True)

    # 3b) Column strips (name / address / price / billing) — vision only
    print(f"  {prefix} column strips…", flush=True)
    def _col_prompt(label: str) -> str:
        if "billing" in label:
            kind = "billing"
        elif "name" in label:
            kind = "name"
        elif "addr" in label:
            kind = "addr"
        else:
            kind = "price"
        return column_strip_prompt(kind)

    col_text = _run_crop_wave(
        model_id, cfg, make_column_strips(page), _col_prompt, workers, prefix, hist
    )
    if col_text:
        merged = prefer_sheet_tables(merge_extracts(json.dumps(merged), *col_text.values()))
        try:
            merged = _apply_vision(
                model_id, cfg, page, unify_prompt(_so_far(merged)), merged
            )
        except Exception as e:
            print(f"  {prefix} col-unify fail: {e}", flush=True)

    # 3c) Row windows for dense multi-row re-read
    print(f"  {prefix} row windows…", flush=True)
    rw_text = _run_crop_wave(
        model_id,
        cfg,
        make_row_windows(page, windows=ROW_WINDOWS, overlap=0.2),
        lambda _label: row_window_prompt(),
        workers,
        prefix,
        hist,
    )
    if rw_text:
        merged = prefer_sheet_tables(merge_extracts(json.dumps(merged), *rw_text.values()))
        try:
            merged = _apply_vision(
                model_id, cfg, page, unify_prompt(_so_far(merged)), merged
            )
        except Exception as e:
            print(f"  {prefix} row-unify fail: {e}", flush=True)

    if grid_needs_repair(merged):
        print(f"  {prefix} grid repair (after crops)…", flush=True)
        try:
            repaired = _apply_vision(
                model_id, cfg, page, grid_repair_prompt(_so_far(merged)), merged
            )
            merged = _pick_better(repaired, merged)
            if not grid_needs_repair(repaired):
                merged = repaired
        except Exception as e:
            print(f"  {prefix} grid-repair fail: {e}", flush=True)

    # 4) Completeness reviews (photo only — no gold)
    stopped = "max_passes"
    for i in range(1, review_passes + 1):
        before = fill_stats(merged)
        print(
            f"  {prefix} review {i}/{review_passes}: filled={before['filled']} "
            f"fill={before['fill_rate']:.0%}",
            flush=True,
        )
        t1 = time.time()
        try:
            merged = _apply_vision(
                model_id, cfg, page, completeness_prompt(_so_far(merged)), merged
            )
        except Exception as e:
            print(f"  {prefix} review {i} fail: {e}", flush=True)
            stopped = "error"
            break
        st = fill_stats(merged)
        grew = (st["filled"] - before["filled"]) / max(before["filled"], 1)
        grew_fill = st["fill_rate"] - before["fill_rate"]
        hist.append({
            "pass": f"review-{i}",
            "cells": st["filled"],
            "fill_rate": round(st["fill_rate"], 3),
            "grew": round(grew, 4),
            "elapsed_s": round(time.time() - t1, 2),
        })
        print(
            f"  {prefix} review {i}: filled={st['filled']} fill={st['fill_rate']:.0%} grew={grew:.1%}",
            flush=True,
        )
        plateau = grew <= GROWTH_STOP and abs(grew_fill) <= GROWTH_STOP
        if plateau and i >= MIN_REVIEWS_BEFORE_PLATEAU and st["fill_rate"] >= 0.75:
            stopped = "plateau"
            break

    # 5) Blind column focus rounds (vote) — always run even after fill plateau
    for round_i in range(1, FIELD_FOCUS_ROUNDS + 1):
        for label, prompt_fn in (
            (f"price-focus-{round_i}", price_focus_prompt),
            (f"address-focus-{round_i}", address_focus_prompt),
        ):
            print(f"  {prefix} {label}…", flush=True)
            t_f = time.time()
            try:
                merged = _apply_vision(
                    model_id, cfg, page, prompt_fn(_so_far(merged)), merged, vote=True
                )
                if "price" in label:
                    merged = _normalize_price_cells(merged)
                hist.append({
                    "pass": label,
                    "cells": fill_stats(merged)["filled"],
                    "fill_rate": round(fill_stats(merged)["fill_rate"], 3),
                    "elapsed_s": round(time.time() - t_f, 2),
                })
            except Exception as e:
                print(f"  {prefix} {label} fail: {e}", flush=True)

    # 5b) Re-unify after field focus
    print(f"  {prefix} post-focus unify…", flush=True)
    try:
        merged = _apply_vision(
            model_id, cfg, page, unify_prompt(_so_far(merged)), merged
        )
        merged = _normalize_price_cells(merged)
        hist.append({
            "pass": "post-focus-unify",
            "cells": fill_stats(merged)["filled"],
            "fill_rate": round(fill_stats(merged)["fill_rate"], 3),
        })
    except Exception as e:
        print(f"  {prefix} post-focus unify fail: {e}", flush=True)

    # 6) Geometry-first: ruling-line row chunks + bbox assemble + price cells
    # Only keep merges that don't destroy the roster (research stacks still need gates).
    def _safe_merge(label: str, donor: dict) -> None:
        nonlocal merged
        before = merged
        cand = prefer_sheet_tables(merge_extracts(json.dumps(merged), json.dumps(donor)))
        # Prefer candidate only if it keeps name-like rows and doesn't shrink too hard
        b_rows = fill_stats(before)["rows"]
        c_rows = fill_stats(cand)["rows"]
        b_names = sum(
            1
            for t in (before.get("tables") or [])
            for r in (t.get("rows") or [])[:30]
            if isinstance(r, list) and r and ("," in str(r[0]) or _looks_like_name(str(r[0])))
        )
        c_names = sum(
            1
            for t in (cand.get("tables") or [])
            for r in (t.get("rows") or [])[:30]
            if isinstance(r, list) and r and ("," in str(r[0]) or _looks_like_name(str(r[0])))
        )
        if c_names >= max(3, int(b_names * 0.7)) and c_rows >= max(5, int(b_rows * 0.5)):
            merged = _pick_better(cand, before)
            print(f"  {prefix} {label}: kept (names {b_names}→{c_names}, rows {b_rows}→{c_rows})", flush=True)
        else:
            print(
                f"  {prefix} {label}: rejected (names {b_names}→{c_names}, rows {b_rows}→{c_rows})",
                flush=True,
            )

    print(f"  {prefix} row-chunk crops (line detect)…", flush=True)
    try:
        rc_text = _run_crop_wave(
            model_id,
            cfg,
            make_row_crops(page, chunk=ROW_CHUNK, max_chunks=MAX_ROW_CHUNKS),
            lambda _label: row_window_prompt(),
            workers,
            prefix,
            hist,
        )
        if rc_text:
            donor = merge_extracts(*rc_text.values())
            _safe_merge("row-chunk", donor)
            try:
                repaired = _apply_vision(
                    model_id, cfg, page, unify_prompt(_so_far(merged)), merged
                )
                if richness(repaired) >= richness(merged):
                    merged = repaired
            except Exception:
                pass
    except Exception as e:
        print(f"  {prefix} row-chunk fail: {e}", flush=True)

    print(f"  {prefix} bbox-sorted RapidOCR assemble…", flush=True)
    try:
        pw, ph = page_size(page)
        boxes_scored = ocr_page_boxes(page)
        boxes = [(x0, y0, x1, y1, text) for x0, y0, x1, y1, text, _sc in boxes_scored]
        geo = assemble_table_from_ocr_boxes(boxes, page_w=pw, page_h=ph)
        geo_rows = fill_stats(geo)["rows"]
        print(f"  {prefix} bbox rows={geo_rows} boxes={len(boxes)}", flush=True)
        sample = (geo.get("tables") or [{}])[0].get("rows") or []
        nameish = sum(
            1
            for r in sample[:20]
            if isinstance(r, list) and r and ("," in str(r[0]) or str(r[0]).isupper())
        )
        if geo_rows >= 8 and nameish >= max(3, len(sample[:20]) // 4):
            _safe_merge("bbox-assemble", geo)
        else:
            print(f"  {prefix} bbox assemble skipped (nameish={nameish})", flush=True)
        hist.append({"pass": "bbox-assemble", "cells": fill_stats(merged)["filled"], "bbox_rows": geo_rows})
    except Exception as e:
        print(f"  {prefix} bbox-assemble fail: {e}", flush=True)

    print(f"  {prefix} upscaled price-cell OCR + majority…", flush=True)
    cell_paths: list = []
    classic_cells: list = []
    try:
        cells = make_price_cell_crops(page, scale=3.0)
        cell_paths = [p for _lab, p, _i in cells]
        classic_cells = read_prices_from_cells(cell_paths)
        n_ok = sum(1 for p in classic_cells if p is not None)
        print(f"  {prefix} price-cells={len(classic_cells)} parsed={n_ok}", flush=True)
        strip = make_price_strip(page)
        try:
            strip_prices = read_prices_top_to_bottom(strip)
        finally:
            strip.unlink(missing_ok=True)
        merged = majority_vote_prices(merged, classic_cells, strip_prices)
        merged = overlay_prices_by_order(merged, classic_cells)
        merged = _normalize_price_cells(merged)
        hist.append({
            "pass": "price-cell-majority",
            "cells": fill_stats(merged)["filled"],
            "n_cell_prices": n_ok,
            "n_strip_prices": len(strip_prices),
        })
    except Exception as e:
        print(f"  {prefix} price-cell majority fail: {e}", flush=True)
    finally:
        for p in cell_paths:
            Path(p).unlink(missing_ok=True)

    # 7) Classical price-strip OCR overlay (page pixels only)
    print(f"  {prefix} classical price OCR…", flush=True)
    price_strip = None
    classic_prices: list = []
    try:
        price_strip = make_price_strip(page)
        classic_prices = read_prices_top_to_bottom(price_strip)
        print(f"  {prefix} classical prices={len(classic_prices)}", flush=True)
        if classic_prices:
            merged = overlay_prices_by_order(merged, classic_prices)
            merged = majority_vote_prices(merged, classic_prices, classic_cells or None)
            merged = _normalize_price_cells(merged)
            hist.append({
                "pass": "classical-price",
                "cells": fill_stats(merged)["filled"],
                "fill_rate": round(fill_stats(merged)["fill_rate"], 3),
                "n_prices": len(classic_prices),
            })
    except Exception as e:
        print(f"  {prefix} classical price fail: {e}", flush=True)
    finally:
        if price_strip is not None:
            price_strip.unlink(missing_ok=True)

    # 8) Collapse + tail catch-up + confidence flags for human review
    merged = prefer_sheet_tables(merged)
    print(f"  {prefix} tail completeness…", flush=True)
    try:
        before_rows = fill_stats(merged)["rows"]
        merged = _apply_vision(
            model_id, cfg, page, tail_completeness_prompt(_so_far(merged)), merged
        )
        merged = prefer_sheet_tables(_normalize_price_cells(merged))
        hist.append({
            "pass": "tail-completeness",
            "cells": fill_stats(merged)["filled"],
            "fill_rate": round(fill_stats(merged)["fill_rate"], 3),
            "rows_before": before_rows,
            "rows_after": fill_stats(merged)["rows"],
        })
    except Exception as e:
        print(f"  {prefix} tail completeness fail: {e}", flush=True)

    if novel and novel_donors:
        print(f"  {prefix} novel: multi-pass disagreement flags…", flush=True)
        try:
            # Cell-wise majority across independent donors, then flag leftovers
            voted = majority_merge_extracts(
                [json.dumps(d) for d in novel_donors] + [json.dumps(merged)]
            )
            merged = prefer_sheet_tables(_pick_better(voted, merged))
            merged = _flag_uncertain_disagreements(merged, novel_donors)
            hist.append({
                "pass": "multi-pass-vote",
                "cells": fill_stats(merged)["filled"],
                "n_uncertain_notes": sum(
                    1 for n in (merged.get("notes") or []) if str(n).startswith("UNCERTAIN")
                ),
            })
        except Exception as e:
            print(f"  {prefix} multi-pass vote fail: {e}", flush=True)

    merged = annotate_confidence(
        prefer_sheet_tables(_normalize_price_cells(merged)),
        classic_cells or classic_prices or None,
    )
    final = merged
    final_st = fill_stats(final)
    final_json = json.dumps(final, indent=2, ensure_ascii=False)
    # Score against gold for reporting only — never fed back into the extract.
    gold_final = {}
    elapsed = round(time.time() - t0, 2)
    if asset_dir is not None:
        panels.append({
            "kind": "final",
            "label": f"merged ({stopped})",
            "image": str((asset_dir / f"{fx['id']}_page.jpg").as_posix()),
            "extract": final_json,
            "score": gold_final.get("score"),
        })

    print(
        f"  {prefix} done in {elapsed}s — filled={final_st['filled']} "
        f"rows={final_st['rows']} fill={final_st['fill_rate']:.0%} "
        f"gold={gold_final.get('score')} stopped={stopped}",
        flush=True,
    )
    return {
        "fixture": fx["id"],
        "image": fx["image"],
        "stopped": stopped,
        "cells": final_st["filled"],
        "fill_rate": round(final_st["fill_rate"], 3),
        "rows": final_st["rows"],
        "elapsed_s": elapsed,
        "gold_score": gold_final.get("score"),
        "client_recall": gold_final.get("client_recall"),
        "price_recall": gold_final.get("price_recall"),
        "address_recall": gold_final.get("address_recall"),
        "passes": hist,
        "ocr_passes_to_perfect": stopped,
        "nl_passes_to_perfect": None,
        "best_ocr_score": gold_final.get("score"),
        "best_nl_score": None,
        "gold_patched": False,
        "ocr_history": hist,
        "panels": panels,
        "final_extract": final_json,
    }


def wave_workers_for(cfg: dict, fx_workers: int = 1) -> int:
    _ = fx_workers
    return _parallelism(cfg)
