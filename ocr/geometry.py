"""Geometry helpers: ruling-line row detection, cell crops, bbox row assembly."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def _load_upright(image_path: Path) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    w, h = img.size
    if w > h * 1.15:
        img = img.rotate(-90, expand=True, fillcolor=(255, 255, 255))
    return img


def _save_temp(crop: Image.Image, stem: str, tag: str, *, scale: float = 1.0) -> Path:
    if scale != 1.0:
        w, h = crop.size
        crop = crop.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
    path = Path(tempfile.gettempdir()) / (
        f"nini_geo_{os.getpid()}_{stem}_{tag}_{os.urandom(3).hex()}.jpg"
    )
    crop.save(path, format="JPEG", quality=95, optimize=True)
    return path


def detect_row_bands(
    image_path: Path,
    *,
    min_rows: int = 8,
    max_rows: int = 60,
    pad: float = 0.002,
) -> list[tuple[int, int]]:
    """Detect horizontal row bands via ruling-line morphology.

    Returns list of (y0, y1) in upright image coords, top→bottom, excluding
    a likely header band at the top.
    """
    import cv2

    img = _load_upright(image_path)
    w, h = img.size
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Emphasize horizontal rules
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    k = max(40, w // 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    rules = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    # Project horizontal ink to find rule y positions
    proj = rules.sum(axis=1).astype(np.float32)
    # Smooth
    win = max(3, h // 400)
    if win % 2 == 0:
        win += 1
    proj_s = cv2.GaussianBlur(proj.reshape(-1, 1), (1, win), 0).ravel()
    thr = float(np.percentile(proj_s, 85))
    peaks: list[int] = []
    i = 0
    while i < h:
        if proj_s[i] >= thr:
            j = i
            while j < h and proj_s[j] >= thr * 0.7:
                j += 1
            peaks.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    # Fallback: equal splits if too few rules
    if len(peaks) < min_rows:
        n = min(max_rows, max(min_rows, 40))
        step = h / (n + 1)
        peaks = [int(step * (k + 1)) for k in range(n)]
    # Dedup peaks that are too close
    min_gap = max(8, h // 120)
    cleaned: list[int] = []
    for y in peaks:
        if not cleaned or y - cleaned[-1] >= min_gap:
            cleaned.append(y)
    peaks = cleaned
    if len(peaks) < 2:
        return [(0, h)]
    # Row band between consecutive rules; skip very top header-ish band
    bands: list[tuple[int, int]] = []
    pad_px = int(h * pad)
    for a, b in zip(peaks, peaks[1:]):
        y0 = max(0, a - pad_px)
        y1 = min(h, b + pad_px)
        if y1 - y0 < max(10, h // 200):
            continue
        # Skip first band if it's tiny (title) — keep if tall enough for a row
        bands.append((y0, y1))
    # Drop first band when it looks like a header (short + near top)
    if bands and bands[0][0] < h * 0.08 and (bands[0][1] - bands[0][0]) < h * 0.035:
        bands = bands[1:]
    if len(bands) > max_rows:
        # Keep evenly spaced subset
        idx = np.linspace(0, len(bands) - 1, max_rows).astype(int)
        bands = [bands[i] for i in idx]
    return bands or [(0, h)]


def make_row_crops(
    image_path: Path,
    *,
    chunk: int = 3,
    max_chunks: int = 20,
) -> list[tuple[str, Path]]:
    """Crop consecutive row chunks (geometry-first multipass)."""
    img = _load_upright(image_path)
    w, h = img.size
    bands = detect_row_bands(image_path)
    out: list[tuple[str, Path]] = []
    for i in range(0, len(bands), chunk):
        if len(out) >= max_chunks:
            break
        group = bands[i : i + chunk]
        y0 = group[0][0]
        y1 = group[-1][1]
        # Slight vertical pad into neighbors
        y0 = max(0, y0 - int(h * 0.005))
        y1 = min(h, y1 + int(h * 0.005))
        crop = img.crop((0, y0, w, y1))
        out.append((f"rowchunk{len(out)+1}", _save_temp(crop, image_path.stem, f"rc{i}")))
    return out


def make_price_cell_crops(
    image_path: Path,
    *,
    x0f: float | None = None,
    x1f: float | None = None,
    scale: float = 3.0,
    max_cells: int = 55,
) -> list[tuple[str, Path, int]]:
    """Upscaled price-column cell crops aligned to detected rows.

    Defaults cover both mowing (New Price ~mid) and hedges (Hedge ~mid-right).
    Returns (label, path, row_index).
    """
    img = _load_upright(image_path)
    w, h = img.size
    bands = detect_row_bands(image_path)
    # Dual band: try mid price then hedge-ish if needed — crop a wider mid strip
    if x0f is None or x1f is None:
        x0f, x1f = 0.40, 0.72
    x0 = int(w * x0f)
    x1 = int(w * x1f)
    out: list[tuple[str, Path, int]] = []
    for i, (y0, y1) in enumerate(bands[:max_cells]):
        yy0 = max(0, y0 - 2)
        yy1 = min(h, y1 + 2)
        cell = img.crop((x0, yy0, x1, yy1))
        path = _save_temp(cell, image_path.stem, f"pc{i}", scale=scale)
        out.append((f"price-cell{i+1}", path, i))
    return out


def assemble_table_from_ocr_boxes(
    boxes: list[tuple[float, float, float, float, str]],
    *,
    page_w: int,
    page_h: int,
) -> dict:
    """Cluster OCR boxes into rows by y, columns by x → sheet-like JSON table.

    Each box: (x0, y0, x1, y1, text). Coordinates in page pixels.
    """
    if not boxes:
        return {"title": None, "tables": [], "notes": [], "complete": False}

    # Sort by y mid
    items = []
    for x0, y0, x1, y1, text in boxes:
        text = str(text).strip()
        if not text:
            continue
        items.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "xm": (x0 + x1) / 2, "ym": (y0 + y1) / 2,
            "text": text,
        })
    items.sort(key=lambda t: t["ym"])

    # Cluster into rows
    row_tol = max(12.0, page_h * 0.012)
    rows_boxes: list[list[dict]] = []
    for it in items:
        if not rows_boxes or abs(it["ym"] - rows_boxes[-1][0]["ym"]) > row_tol:
            rows_boxes.append([it])
        else:
            rows_boxes[-1].append(it)
            # update row ym reference toward mean
            rows_boxes[-1][0]["ym"] = sum(b["ym"] for b in rows_boxes[-1]) / len(rows_boxes[-1])

    # Column boundaries from x distribution (4-col roster bias)
    xs = sorted(it["xm"] for it in items)
    # Use quartiles / gaps
    col_edges = [0.0, page_w * 0.28, page_w * 0.48, page_w * 0.60, float(page_w)]
    headers = ["Contact", "Address", "New Price", "Billing Address / Notes"]

    def col_of(xm: float) -> int:
        for i in range(len(col_edges) - 1):
            if col_edges[i] <= xm < col_edges[i + 1]:
                return min(i, len(headers) - 1)
        return len(headers) - 1

    rows: list[list[str]] = []
    for rb in rows_boxes:
        cells = [""] * len(headers)
        # Sort left→right within row; concatenate texts in same col
        rb_sorted = sorted(rb, key=lambda b: b["xm"])
        buckets: list[list[str]] = [[] for _ in headers]
        for b in rb_sorted:
            buckets[col_of(b["xm"])].append(b["text"])
        for i, parts in enumerate(buckets):
            cells[i] = " ".join(parts).strip()
        if any(cells):
            rows.append(cells)

    # Drop likely header row if first looks like headers
    if rows:
        joined = " ".join(rows[0]).lower()
        if "contact" in joined or ("address" in joined and "price" in joined):
            rows = rows[1:]

    return {
        "title": None,
        "tables": [{"caption": None, "columns": headers, "rows": rows}],
        "notes": [],
        "complete": False,
    }


def page_size(image_path: Path) -> tuple[int, int]:
    img = _load_upright(image_path)
    return img.size
