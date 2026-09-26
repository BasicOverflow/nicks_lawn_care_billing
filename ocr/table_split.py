"""Deterministic table split for white paper + black grid lines.

Isolates the table frame and row bands so a VLM sees small groups (2–4 rows)
per image instead of a full dense page — avoids omission/"laziness".
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary ink mask for black-on-white sheets (lines + handwriting)."""
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    paper = float(np.percentile(gray, 90))
    thr = max(40.0, min(200.0, paper * 0.72))
    _, hard = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 8
    )
    ink = cv2.bitwise_or(otsu, hard)
    return cv2.bitwise_or(ink, adapt)


def _horizontal_rule_mask(gray: np.ndarray) -> np.ndarray:
    """Emphasize thin dark horizontal rules (blackhat + short open)."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kw = max(35, min(100, w // 50))
    blackhat = cv2.morphologyEx(
        blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    )
    _, bh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Short open only — long kernels wipe phone-photo grid lines
    open_w = max(18, min(48, w // 100))
    opened = cv2.morphologyEx(
        bh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (open_w, 1))
    )
    return cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)


def _vertical_rule_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kh = max(35, min(100, h // 50))
    blackhat = cv2.morphologyEx(
        blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    )
    _, bh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    open_h = max(18, min(48, h // 100))
    opened = cv2.morphologyEx(
        bh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, open_h))
    )
    return cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)


def _projection_peaks(
    mask: np.ndarray,
    *,
    axis: int,
    min_gap: int,
    min_frac: float = 0.04,
    percentile: float = 50.0,
) -> list[int]:
    """Peak y (axis=1) or x (axis=0) positions from a binary rule mask."""
    proj = mask.sum(axis=axis).astype(np.float64)
    if proj.max() < 1:
        return []
    span = mask.shape[1 - axis]
    pos = proj[proj > 0]
    thr = max(
        float(np.percentile(pos, percentile)) if pos.size else 0.0,
        span * min_frac * 255 * 0.2,
    )
    peaks: list[int] = []
    i = 0
    n = len(proj)
    while i < n:
        if proj[i] >= thr:
            j = i
            best = i
            best_v = proj[i]
            while j < n and proj[j] >= thr * 0.3:
                if proj[j] > best_v:
                    best_v = proj[j]
                    best = j
                j += 1
            peaks.append(best)
            i = j + 1
        else:
            i += 1
    cleaned: list[int] = []
    for y in peaks:
        if not cleaned or y - cleaned[-1] >= min_gap:
            cleaned.append(y)
        elif proj[y] > proj[cleaned[-1]]:
            cleaned[-1] = y
    return cleaned


def _contour_table_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = gray.shape
    ink = _ink_mask(gray)
    k = max(5, min(w, h) // 200)
    closed = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=2
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    page_area = w * h
    for c in contours:
        area = cv2.contourArea(c)
        if area < page_area * 0.08:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > w * 0.96 and bh > h * 0.96:
            continue
        if bw < w * 0.35 or bh < h * 0.2:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, x + bw, y + bh)
    return best


def _load_upright(image_path: Path, *, rotate_cw_deg: int = 0) -> Image.Image:
    """EXIF-correct, then apply optional clockwise rotation (from VLM / caller)."""
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    rot = int(rotate_cw_deg) % 360
    if rot:
        img = img.rotate(-rot, expand=True, fillcolor=(255, 255, 255))
    return img


def _save_temp(crop: Image.Image, stem: str, tag: str, *, scale: float = 1.0) -> Path:
    if scale != 1.0:
        w, h = crop.size
        crop = crop.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
    path = Path(tempfile.gettempdir()) / (
        f"nini_tsplit_{os.getpid()}_{stem}_{tag}_{os.urandom(3).hex()}.jpg"
    )
    crop.save(path, format="JPEG", quality=95, optimize=True)
    return path


def detect_horizontal_rules(
    image_path: Path,
    *,
    min_rules: int = 6,
    max_rules: int = 80,
    y_range: tuple[int, int] | None = None,
) -> list[int]:
    """Return y pixel positions of horizontal black rules (top→bottom)."""
    img = _load_upright(image_path)
    w, h = img.size
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    mask = _horizontal_rule_mask(gray)
    if y_range is None:
        box = _contour_table_bbox(gray)
        if box is not None:
            y_range = (box[1], box[3])
    if y_range is not None:
        y0, y1 = max(0, y_range[0]), min(h, y_range[1])
        mask = mask.copy()
        mask[:y0, :] = 0
        mask[y1:, :] = 0
        min_gap = max(8, (y1 - y0) // 80)
    else:
        min_gap = max(10, h // 200)
    peaks = _projection_peaks(
        mask, axis=1, min_gap=min_gap, min_frac=0.035, percentile=50.0
    )
    peaks = _merge_rules(peaks, min_gap=min_gap)
    if len(peaks) > max_rules:
        idx = np.linspace(0, len(peaks) - 1, max_rules).astype(int)
        peaks = [peaks[i] for i in idx]
    return peaks if len(peaks) >= min_rules else peaks


def detect_table_bbox(image_path: Path) -> tuple[int, int, int, int] | None:
    """Largest dark rectangular frame on a white page, or None."""
    img = _load_upright(image_path)
    w, h = img.size
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    best = _contour_table_bbox(gray)
    page_area = w * h

    rules = detect_horizontal_rules(image_path, min_rules=4, max_rules=80)
    if len(rules) >= 4:
        vm = _vertical_rule_mask(gray)
        # Limit vertical peaks to rule y-span
        y0r, y1r = rules[0], rules[-1]
        vm2 = vm.copy()
        vm2[: max(0, y0r - 20), :] = 0
        vm2[min(h, y1r + 20) :, :] = 0
        xs = _projection_peaks(
            vm2, axis=0, min_gap=max(10, w // 200), min_frac=0.04, percentile=50.0
        )
        y0 = max(0, rules[0] - int(h * 0.01))
        y1 = min(h, rules[-1] + int(h * 0.01))
        if len(xs) >= 2:
            x0 = max(0, xs[0] - int(w * 0.01))
            x1 = min(w, xs[-1] + int(w * 0.01))
        else:
            x0, x1 = int(w * 0.05), int(w * 0.95)
        rule_box = (x0, y0, x1, y1)
        if best is None:
            return rule_box
        bx0, by0, bx1, by1 = best
        if (bx1 - bx0) * (by1 - by0) > page_area * 0.85:
            return rule_box
        return (
            max(0, min(bx0, x0)),
            max(0, min(by0, y0)),
            min(w, max(bx1, x1)),
            min(h, max(by1, y1)),
        )
    return best


def _merge_rules(rules: list[int], *, min_gap: int) -> list[int]:
    """Collapse double-line / near-duplicate horizontal rules."""
    if not rules:
        return []
    gaps = [b - a for a, b in zip(rules, rules[1:]) if b > a]
    med = float(np.median(gaps)) if gaps else float(min_gap)
    merge_gap = max(min_gap, int(med * 0.42))
    out: list[int] = [rules[0]]
    for y in rules[1:]:
        if y - out[-1] < merge_gap:
            out[-1] = (out[-1] + y) // 2
        else:
            out.append(y)
    return out


def _band_has_ink(img: Image.Image, y0: int, y1: int, x0: int, x1: int) -> bool:
    crop = img.crop((x0, y0, x1, max(y1, y0 + 2))).convert("L")
    arr = np.asarray(crop, dtype=np.float32)
    if arr.size < 20:
        return False
    # Ink on white: lower mean and/or higher std than blank paper
    return float(arr.mean()) < 242 or float(arr.std()) > 12


def detect_row_bands_ruled(
    image_path: Path,
    *,
    pad: float = 0.003,
    min_rows: int = 4,
    max_rows: int = 70,
) -> list[tuple[int, int]]:
    """Row bands between consecutive horizontal black rules.

    Falls back to equal splits inside the table bbox if rule detection is weak.
    """
    img = _load_upright(image_path)
    w, h = img.size
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    rough = _contour_table_bbox(gray)
    y_range = (rough[1], rough[3]) if rough is not None else None
    rules = detect_horizontal_rules(
        image_path, min_rules=3, max_rules=max_rows + 8, y_range=y_range
    )
    bbox = detect_table_bbox(image_path)
    if bbox is not None:
        _, y0, _, y1 = bbox
        rules = [y for y in rules if y0 - h * 0.02 <= y <= y1 + h * 0.02]
        span = max(1, y1 - y0)
        rules = _merge_rules(rules, min_gap=max(8, span // 90))
        if len(rules) < 2:
            rules = [y0, y1]
        else:
            if rules[0] > y0 + 8:
                rules = [y0] + rules
            if rules[-1] < y1 - 8:
                rules = rules + [y1]

    bands: list[tuple[int, int]] = []
    pad_px = int(h * pad)
    if len(rules) >= 2:
        gaps = [b - a for a, b in zip(rules, rules[1:]) if b > a]
        med_gap = float(np.median(gaps)) if gaps else h * 0.05
        max_row_h = max(med_gap * 2.0, h * 0.035)
        min_row_h = max(8, int(med_gap * 0.35))
        for a, b in zip(rules, rules[1:]):
            ya = max(0, a - pad_px)
            yb = min(h, b + pad_px)
            ht = yb - ya
            if ht < min_row_h or ht > max_row_h:
                continue
            bands.append((ya, yb))
        # Drop tiny title strip above first real row
        if bands and bands[0][0] < h * 0.08 and (bands[0][1] - bands[0][0]) < med_gap * 0.55:
            bands = bands[1:]

    if len(bands) < min_rows:
        if bbox is not None:
            _, y0, _, y1 = bbox
        elif rough is not None:
            _, y0, _, y1 = rough
        else:
            y0, y1 = int(h * 0.08), int(h * 0.92)
        n = min(max_rows, max(min_rows, 22))
        step = (y1 - y0) / n
        bands = []
        for i in range(n):
            ya = int(y0 + i * step)
            yb = int(y0 + (i + 1) * step)
            bands.append((max(0, ya - pad_px), min(h, yb + pad_px)))

    if len(bands) > max_rows:
        idx = np.linspace(0, len(bands) - 1, max_rows).astype(int)
        bands = [bands[i] for i in idx]
    return bands


def _find_header_band_idx(
    img: Image.Image,
    bands: list[tuple[int, int]],
    x0: int,
    x1: int,
) -> int:
    """Index of the column-header row (skip title / blank strips)."""
    for i, (y0, y1) in enumerate(bands[:6]):
        if not _band_has_ink(img, y0, y1, x0, x1):
            continue
        crop = img.crop((x0, y0, x1, y1)).convert("L")
        # Header rows are usually short and near the top of the table
        if (y1 - y0) > img.size[1] * 0.08:
            continue
        return i
    return 0 if bands else -1


def make_ruled_row_crops(
    image_path: Path,
    *,
    rows_per_crop: int = 3,
    scale: float = 2.2,
    max_crops: int = 48,
    include_header: bool = True,
    x_pad: float = 0.02,
    edge_pad_rows: float = 0.5,
    overlap: int = 1,
    enhance_crops: bool = True,
) -> tuple[Path | None, list[tuple[str, Path, int]], Path | None]:
    """Crop overlapping groups of consecutive table rows, upscaled for VLM.

    Returns (header_crop, [(label, path, start_i)], price_strip_or_None).
    """
    from .preprocess import enhance_ruled_sheet

    rows_per_crop = max(2, min(4, int(rows_per_crop)))
    overlap = max(0, min(rows_per_crop - 1, int(overlap)))
    step = max(1, rows_per_crop - overlap)
    img = _load_upright(image_path)
    w, h = img.size
    bands = detect_row_bands_ruled(image_path)
    bbox = detect_table_bbox(image_path)
    if bbox is not None:
        tx0, _, tx1, _ = bbox
        x0 = max(0, tx0 - int(w * x_pad))
        x1 = min(w, tx1 + int(w * x_pad))
    else:
        x0, x1 = int(w * x_pad), int(w * (1 - x_pad))

    # Drop trailing blank bands only (keep interior empties for alignment)
    while bands and not _band_has_ink(img, bands[-1][0], bands[-1][1], x0, x1):
        bands = bands[:-1]
    # Drop leading blank / title-only if no ink
    while len(bands) > 2 and not _band_has_ink(img, bands[0][0], bands[0][1], x0, x1):
        bands = bands[1:]

    header_path: Path | None = None
    data_bands = bands
    hdr_i = _find_header_band_idx(img, bands, x0, x1) if bands else -1
    if include_header and hdr_i >= 0:
        y0, y1 = bands[hdr_i]
        mean_h = max(8, y1 - y0)
        hy0 = max(0, y0 - int(mean_h * 0.3))
        hy1 = min(h, y1 + int(mean_h * 0.35))
        header = img.crop((x0, hy0, x1, hy1))
        if enhance_crops:
            try:
                header = enhance_ruled_sheet(header)
            except Exception:
                pass
        header_path = _save_temp(header, image_path.stem, "hdr", scale=max(2.0, scale))
        data_bands = bands[hdr_i + 1 :]

    out: list[tuple[str, Path, int]] = []
    for i in range(0, len(data_bands), step):
        if len(out) >= max_crops:
            break
        group = data_bands[i : i + rows_per_crop]
        if not group:
            break
        if len(group) < max(1, rows_per_crop - 1) and i > 0:
            # trailing stub already covered by prior overlap
            break
        mean_band = max(8.0, float(np.mean([b[1] - b[0] for b in group])))
        pad_y = max(4, int(mean_band * edge_pad_rows))
        y0 = max(0, group[0][0] - pad_y)
        y1 = min(h, group[-1][1] + pad_y)
        crop = img.crop((x0, y0, x1, y1))
        arr = np.asarray(crop.convert("L"))
        if float(arr.mean()) > 245 and float(arr.std()) < 8:
            continue
        if enhance_crops:
            try:
                crop = enhance_ruled_sheet(crop)
            except Exception:
                pass
        path = _save_temp(crop, image_path.stem, f"rg{i}", scale=scale)
        out.append((f"rows{i+1}-{i+len(group)}", path, i))

    price_path = make_price_column_crop(
        image_path, scale=max(2.0, scale), enhance=enhance_crops
    )
    return header_path, out, price_path


def make_price_column_crop(
    image_path: Path,
    *,
    scale: float = 2.2,
    enhance: bool = True,
    x_pad: float = 0.01,
) -> Path | None:
    """Crop Contact + Price columns (for a focused price second pass)."""
    from .preprocess import enhance_ruled_sheet

    img = _load_upright(image_path)
    w, h = img.size
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    bbox = detect_table_bbox(image_path)
    if bbox is None:
        return None
    tx0, ty0, tx1, ty1 = bbox
    vm = _vertical_rule_mask(gray)
    vm2 = vm.copy()
    vm2[: max(0, ty0 - 10), :] = 0
    vm2[min(h, ty1 + 10) :, :] = 0
    xs = _projection_peaks(
        vm2, axis=0, min_gap=max(12, w // 120), min_frac=0.04, percentile=50.0
    )
    xs = [x for x in xs if tx0 - 20 <= x <= tx1 + 20]
    if len(xs) < 3:
        # Fallback: middle-right third of table ≈ price col with contact on left
        x0 = max(0, tx0 - int(w * x_pad))
        x1 = min(w, tx0 + int((tx1 - tx0) * 0.72))
    else:
        # Contact col = between xs[0]..xs[1], price often xs[2]..xs[3]
        x0 = max(0, xs[0] - int(w * x_pad))
        # Include through price column (3rd gap); stop before wide notes col
        if len(xs) >= 4:
            x1 = min(w, xs[3] + int(w * x_pad))
        else:
            x1 = min(w, xs[-1] + int((tx1 - xs[-1]) * 0.35))
    crop = img.crop((x0, max(0, ty0 - 4), x1, min(h, ty1 + 4)))
    if enhance:
        try:
            crop = enhance_ruled_sheet(crop)
        except Exception:
            pass
    return _save_temp(crop, image_path.stem, "pricecols", scale=scale)
