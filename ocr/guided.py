"""Knowledge-guided OCR for a photographed work-completed sheet.

The full page is sent several times in parallel. Each request asks for the
work cell beside a small group of names written on that page.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .cancel import InferenceCancelled, raise_if_cancelled
from .chat import chat_with_image
from .prompts import TABLE_JSON_SCHEMA, guided_header_prompt, guided_work_prompt
from .tables import _as_obj, fill_stats, prefer_sheet_tables
from .work_marks import normalize_work_marks

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


def _row_name(row: list, columns: list[str]) -> str:
    for i, col in enumerate(columns):
        c = str(col).lower()
        if any(word in c for word in ("contact", "name", "client")):
            return str(row[i]) if i < len(row) else ""
    return str(row[0]) if row else ""


READING_ORDER_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "properties": {"names": {"type": "array", "items": {"type": "string"}}},
    "required": ["names"],
}

WORK_NAME_PROMPT = """
Read this photographed work-completed sheet from top to bottom, the way the rows were written.
List a client name only when that row has work written in the work cell: a day, a hedge mark, or a job note.
Skip the name when the work cell is empty. A printed name with nothing beside it is not work completed.
Do not alphabetize. Use the name as written, usually LAST, First. One entry per row that has work.
Do not include days, prices, addresses, or phone numbers.
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


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


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
    """Run full-image OCR on a work-completed photo. Returns sheet JSON dict."""
    _ = (knowledge_names, knowledge_records)
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
            sheet_kind=sheet_kind,
            chunk_size=chunk_size,
            wave_workers=wave_workers,
            fx_id=fx_id or original.stem,
        )
    finally:
        if page != original:
            page.unlink(missing_ok=True)


def _sheet_names(model_id: str, cfg: dict, page: Path) -> list[str]:
    """Names on the work sheet that have work written beside them. Not the client roster."""
    text = chat_with_image(
        model_id,
        page,
        WORK_NAME_PROMPT,
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
            if cells[0] and any(part.strip() for part in cells[1:]):
                rows.append(cells)
    return rows


def _work_text_len(row: list) -> int:
    return sum(len(str(cell).strip()) for cell in list(row)[1:])


def _collapse_work_rows(rows: list[list[str]], columns: list[str]) -> list[list[str]]:
    """One review line per name.

    Chunks read the whole photo, so the same line often comes back more than once.
    A later copy replaces the earlier one only when it has more work text.
    """
    out: list[list[str]] = []
    at: dict[str, int] = {}
    for row in rows:
        cells = [str(c) for c in row]
        key = _name_key(_row_name(cells, columns))
        if not key:
            out.append(cells)
            continue
        prev = at.get(key)
        if prev is None:
            at[key] = len(out)
            out.append(cells)
            continue
        if _work_text_len(cells) > _work_text_len(out[prev]):
            out[prev] = cells
    return out


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


def _assemble_work_rows(
    names: list[str], pool: list[list[str]], columns: list[str]
) -> list[list[str]]:
    """Place each sheet name once. A name with no work written is left out."""
    pool = _collapse_work_rows([row for row in pool if _work_text_len(row) > 0], columns)
    rows: list[list[str]] = []
    for name in names:
        hit = _take_work_row(name, pool, columns)
        if hit is None:
            key = _name_key(name)
            for i, row in enumerate(pool):
                if key and _name_key(_row_name(row, columns)) == key:
                    hit = pool.pop(i)
                    break
        if hit is not None and _work_text_len(hit) > 0:
            rows.append(hit)
    rows.extend(row for row in pool if _work_text_len(row) > 0)
    return _collapse_work_rows(rows, columns)


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
        rows = _assemble_work_rows(names, pool, columns)
    else:
        try:
            got = _vision(
                model_id,
                cfg,
                page,
                guided_work_prompt([], columns, header.get("title")),
                reuse_prepared=True,
            )
            rows = _collapse_work_rows(_work_cells(got, columns), columns)
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
    sheet_kind: str,
    chunk_size: int,
    wave_workers: int | None,
    fx_id: str | None,
) -> dict:
    prefix = f"[{fx_id or page.stem}]"
    workers = wave_workers if wave_workers is not None else _parallelism(cfg)
    t0 = time.time()
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
