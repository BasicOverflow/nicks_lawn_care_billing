"""Classical (non-VLM) OCR helpers — price digits, page boxes, confidence."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _parse_price_token(text: str, *, require_dollar: bool = False) -> float | None:
    s = str(text).strip()
    if not s:
        return None
    has_dollar = "$" in s
    if require_dollar and not has_dollar:
        return None
    m = re.search(r"\$?\s*(\d{2,3})(?:\.\d{2})?\b", s)
    if not m:
        return None
    val = float(m.group(1))
    if val > 999:
        return None
    compact = re.sub(r"[^\d]", "", s)
    # Only 533-style slips (third digit duplicates second). Keep $150/$190.
    if len(compact) == 3 and compact[1] == compact[2]:
        dual = float(compact[:2])
        if 15 <= dual <= 99:
            return dual
    # Street numbers (200–399) without $ are usually addresses, not prices.
    if not has_dollar and 200 <= val <= 399:
        return None
    # Tiny noise
    if val < 10 and not has_dollar:
        return None
    return val


def _looks_like_price_text(text: str) -> bool:
    s = str(text).strip()
    if "$" in s:
        return _parse_price_token(s) is not None
    # bare 2-digit amounts common on sheets
    m = re.fullmatch(r"\$?\s*(\d{2})(?:\.\d{2})?", s)
    return bool(m) and 10 <= int(m.group(1)) <= 99


def read_prices_top_to_bottom(image_path: Path) -> list[float]:
    """OCR the price-column strip; return prices ordered top→bottom."""
    try:
        img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        w, h = img.size
        if w < 400:
            scale = 400 / max(w, 1)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        result, _ = _engine()(img)
    except Exception:
        return []
    if not result:
        return []
    lines: list[tuple[float, float]] = []
    for item in result:
        if not item or len(item) < 2:
            continue
        box, text = item[0], item[1]
        # Prefer $-tagged; fall back to careful bare parse
        price = _parse_price_token(str(text), require_dollar=True)
        if price is None:
            price = _parse_price_token(str(text), require_dollar=False)
        if price is None:
            continue
        try:
            ys = [p[1] for p in box]
            y_mid = sum(ys) / len(ys)
        except Exception:
            y_mid = float(len(lines))
        lines.append((y_mid, price))
    lines.sort(key=lambda t: t[0])
    out: list[float] = []
    last_y = -1e9
    for y, p in lines:
        if out and abs(y - last_y) < 8 and abs(out[-1] - p) < 0.51:
            continue
        out.append(p)
        last_y = y
    return out


def read_price_from_cell(image_path: Path) -> float | None:
    """OCR a single upscaled price cell; return best price or None."""
    try:
        img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        w, h = img.size
        target = 120
        if min(w, h) < target:
            scale = target / max(1, min(w, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        result, _ = _engine()(img)
    except Exception:
        return None
    if not result:
        return None
    best: tuple[float, float] | None = None  # (score, price)
    for item in result:
        if not item or len(item) < 2:
            continue
        text = str(item[1])
        score = float(item[2]) if len(item) > 2 else 0.5
        # Prefer explicit dollar amounts
        price = _parse_price_token(text, require_dollar=True)
        bonus = 0.2 if price is not None else 0.0
        if price is None:
            price = _parse_price_token(text, require_dollar=False)
        if price is None:
            continue
        sc = score + bonus
        if best is None or sc > best[0]:
            best = (sc, price)
    return None if best is None else best[1]


def _fmt_price(p: float) -> str:
    return f"${int(p)}" if p == int(p) else f"${p}"


def read_prices_from_cells(cell_paths: list[Path]) -> list[float | None]:
    return [read_price_from_cell(p) for p in cell_paths]


def ocr_page_boxes(image_path: Path) -> list[tuple[float, float, float, float, str, float]]:
    """Full-page RapidOCR → (x0,y0,x1,y1,text,score) in image pixels."""
    try:
        img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        w, h = img.size
        # Mild upscale for small handwriting
        if max(w, h) < 2200:
            scale = 2200 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            sx = sy = scale
        else:
            sx = sy = 1.0
        result, _ = _engine()(img)
    except Exception:
        return []
    if not result:
        return []
    out: list[tuple[float, float, float, float, str, float]] = []
    for item in result:
        if not item or len(item) < 2:
            continue
        box, text = item[0], str(item[1]).strip()
        if not text:
            continue
        score = float(item[2]) if len(item) > 2 else 0.5
        try:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0, x1 = min(xs) / sx, max(xs) / sx
            y0, y1 = min(ys) / sy, max(ys) / sy
        except Exception:
            continue
        out.append((x0, y0, x1, y1, text, score))
    return out


def overlay_prices_by_order(obj: dict, prices: list[float | None]) -> dict:
    """Fill/correct price column using ordered classical prices (skip None)."""
    if not prices:
        return obj
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
        rows = [r for r in (t.get("rows") or []) if isinstance(r, list)]
        if not rows:
            continue
        if not price_idxs:
            price_idxs = [len(rows[0]) - 1] if rows[0] else []
        if not price_idxs:
            continue
        pi = price_idxs[0]
        for i, row in enumerate(rows):
            if i >= len(prices) or prices[i] is None:
                continue
            while len(row) <= pi:
                row.append("")
            raw = str(row[pi]).strip()
            classic = float(prices[i])
            classic_s = _fmt_price(classic)
            if not raw:
                row[pi] = classic_s
                continue
            digits = re.sub(r"[^\d]", "", raw)
            vl = _parse_price_token(raw)
            # Majority / repair rules (research: vote numeric fields)
            if vl is None:
                row[pi] = classic_s
            elif abs(vl - classic) < 0.51:
                row[pi] = classic_s  # normalize formatting
            elif len(digits) <= 2 and classic >= 100 and classic <= 250:
                # VL dropped a digit ($15 vs $150) — prefer classical 3-digit
                row[pi] = classic_s
            elif len(digits) == 1 and 10 <= classic <= 250:
                row[pi] = classic_s
            elif (
                len(digits) == 3
                and digits[1] == digits[2]
                and classic < 100
                and int(digits[:2]) == int(classic)
            ):
                row[pi] = classic_s
            # else: keep VL — don't overwrite with street-number false positives
        t["rows"] = rows
    return out


def majority_vote_prices(obj: dict, *price_lists: list[float | None]) -> dict:
    """Per-row majority among VL cell + classical lists; write back to price col."""
    lists = [pl for pl in price_lists if pl]
    if not lists:
        return obj
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
        rows = [r for r in (t.get("rows") or []) if isinstance(r, list)]
        if not rows or not price_idxs:
            continue
        pi = price_idxs[0]
        n = len(rows)
        for i in range(n):
            votes: list[float] = []
            raw = ""
            if pi < len(rows[i]):
                raw = str(rows[i][pi]).strip()
                vl = _parse_price_token(raw)
                if vl is not None:
                    votes.append(vl)
            for pl in lists:
                if i < len(pl) and pl[i] is not None:
                    votes.append(float(pl[i]))
            if not votes:
                continue
            # If VL looks truncated (1–2 digits) and a classic vote is 3-digit, prefer classic
            digits = re.sub(r"[^\d]", "", raw)
            classic_long = [v for v in votes if v >= 100]
            if len(digits) <= 2 and classic_long:
                winner = classic_long[0]
            else:
                buckets: dict[int, list[float]] = {}
                for v in votes:
                    buckets.setdefault(int(round(v)), []).append(v)
                winner_key = max(
                    buckets.keys(),
                    key=lambda k: (len(buckets[k]), len(str(k)), k),
                )
                winner = buckets[winner_key][0]
            while len(rows[i]) <= pi:
                rows[i].append("")
            rows[i][pi] = _fmt_price(winner)
        t["rows"] = rows
    return out


def annotate_confidence(obj: dict, classic_prices: list[float | None] | None = None) -> dict:
    """Add uncertain_cells notes for human review (no gold used)."""
    out = json.loads(json.dumps(obj))
    notes = list(out.get("notes") or [])
    uncertain: list[str] = []
    for t in out.get("tables") or []:
        if not isinstance(t, dict):
            continue
        cols = [str(c).lower() for c in (t.get("columns") or [])]
        idx_contact = next(
            (i for i, c in enumerate(cols) if any(k in c for k in ("contact", "name", "client"))),
            0,
        )
        idx_addr = next(
            (
                i
                for i, c in enumerate(cols)
                if "address" in c and "billing" not in c and "note" not in c
            ),
            None,
        )
        idx_price = next(
            (i for i, c in enumerate(cols) if any(k in c for k in ("price", "hedge", "mow"))),
            None,
        )
        for ri, row in enumerate(t.get("rows") or []):
            if not isinstance(row, list):
                continue
            name = str(row[idx_contact]).strip() if idx_contact < len(row) else f"row{ri}"
            issues: list[str] = []
            if idx_addr is not None:
                addr = str(row[idx_addr]).strip() if idx_addr < len(row) else ""
                if not addr:
                    issues.append("empty_address")
                elif "@" in addr or "mail" in addr.lower():
                    issues.append("address_looks_like_billing")
            if idx_price is not None and idx_price < len(row):
                raw = str(row[idx_price]).strip()
                digits = re.sub(r"[^\d]", "", raw)
                vl = _parse_price_token(raw)
                if not raw or vl is None:
                    issues.append("missing_price")
                elif len(digits) == 1:
                    issues.append("price_single_digit")
                if classic_prices and ri < len(classic_prices) and classic_prices[ri] is not None:
                    if vl is not None and abs(vl - float(classic_prices[ri])) >= 0.51:
                        issues.append(
                            f"price_disagree_vl={vl}_ocr={classic_prices[ri]}"
                        )
            if issues:
                uncertain.append(f"{name}: {', '.join(issues)}")
    if uncertain:
        notes.append("UNCERTAIN_CELLS (human review):")
        notes.extend(uncertain[:80])
        if len(uncertain) > 80:
            notes.append(f"…and {len(uncertain) - 80} more")
    out["notes"] = notes
    out["uncertain_count"] = len(uncertain)
    return out
