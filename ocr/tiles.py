"""Page tiling helpers for dense roster OCR."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageOps


def _load_upright(image_path: Path) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    w, h = img.size
    # Mild upright for landscape binder shots of portrait sheets
    if w > h * 1.15:
        img = img.rotate(-90, expand=True, fillcolor=(255, 255, 255))
    return img


def _save_temp(crop: Image.Image, stem: str, tag: str) -> Path:
    path = Path(tempfile.gettempdir()) / (
        f"nini_tile_{os.getpid()}_{stem}_{tag}_{os.urandom(3).hex()}.jpg"
    )
    crop.save(path, format="JPEG", quality=95, optimize=True)
    return path


def make_tiles(
    image_path: Path,
    bands: int = 3,
    overlap: float = 0.08,
    *,
    y_nudge: float = 0.0,
) -> list[tuple[str, Path]]:
    """Return [(label, path), ...] horizontal bands covering the page.

    Overlap reduces dropped rows at band boundaries. ``y_nudge`` shifts all
    bands by a fraction of page height (used when retrying empty tiles).
    Temp JPEGs; caller should unlink.
    """
    img = _load_upright(image_path)
    w, h = img.size
    out: list[tuple[str, Path]] = []
    step = 1.0 / bands
    nudge_px = int(h * y_nudge)
    for i in range(bands):
        y0 = max(0, int(h * (i * step - overlap)) + nudge_px)
        y1 = min(h, int(h * ((i + 1) * step + overlap)) + nudge_px)
        if y1 <= y0:
            continue
        crop = img.crop((0, y0, w, y1))
        out.append((f"tile{i+1}", _save_temp(crop, image_path.stem, f"b{i}")))
    return out


def make_column_strips(image_path: Path) -> list[tuple[str, Path]]:
    """Vertical strips aligned to typical 4-col roster layout.

    Contact | Address | New Price | Billing Address / Notes
    """
    img = _load_upright(image_path)
    w, h = img.size
    specs = [
        ("col-name", 0.00, 0.30),
        ("col-addr", 0.22, 0.50),
        ("col-price", 0.42, 0.62),
        ("col-billing", 0.55, 1.00),
    ]
    out: list[tuple[str, Path]] = []
    for label, x0f, x1f in specs:
        x0 = int(w * x0f)
        x1 = min(w, int(w * x1f))
        crop = img.crop((x0, 0, x1, h))
        out.append((label, _save_temp(crop, image_path.stem, label)))
    return out


def make_row_windows(
    image_path: Path,
    windows: int = 4,
    overlap: float = 0.18,
) -> list[tuple[str, Path]]:
    """Fewer, taller horizontal windows for dense multi-row re-reads."""
    img = _load_upright(image_path)
    w, h = img.size
    out: list[tuple[str, Path]] = []
    step = 1.0 / windows
    for i in range(windows):
        y0 = max(0, int(h * (i * step - overlap)))
        y1 = min(h, int(h * ((i + 1) * step + overlap)))
        crop = img.crop((0, y0, w, y1))
        out.append((f"rowwin{i+1}", _save_temp(crop, image_path.stem, f"rw{i}")))
    return out


def make_price_strip(image_path: Path) -> Path:
    """Right-side crop for classical digit OCR (prices)."""
    img = _load_upright(image_path)
    w, h = img.size
    x0 = int(w * 0.58)
    return _save_temp(img.crop((x0, 0, w, h)), image_path.stem, "price_strip")


def make_region_crops(
    image_path: Path,
    regions: list[dict],
    *,
    pad: float = 0.01,
) -> list[tuple[str, Path]]:
    """Crop normalized regions from smart-split (x0/y0/x1/y1 in 0..1)."""
    img = _load_upright(image_path)
    w, h = img.size
    out: list[tuple[str, Path]] = []
    for i, reg in enumerate(regions):
        try:
            x0 = max(0, int((float(reg.get("x0", 0)) - pad) * w))
            y0 = max(0, int((float(reg.get("y0", 0)) - pad) * h))
            x1 = min(w, int((float(reg.get("x1", 1)) + pad) * w))
            y1 = min(h, int((float(reg.get("y1", 1)) + pad) * h))
        except Exception:
            continue
        if x1 - x0 < 20 or y1 - y0 < 20:
            continue
        label = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(reg.get("label") or f"region{i+1}"))[:40]
        crop = img.crop((x0, y0, x1, y1))
        out.append((f"smart-{label or i+1}", _save_temp(crop, image_path.stem, f"sm{i}")))
    return out


def upscale_crop(path: Path, scale: float = 2.0, *, max_edge: int = 2800) -> Path:
    """2× LANCZOS upscale after crop (not whole-page). Caps long edge."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    nw, nh = int(w * scale), int(h * scale)
    long = max(nw, nh)
    if long > max_edge:
        f = max_edge / long
        nw, nh = int(nw * f), int(nh * f)
    if nw <= w and nh <= h:
        return path
    out = img.resize((nw, nh), Image.Resampling.LANCZOS)
    return _save_temp(out, path.stem, f"up{int(scale*10)}")


def contrast_crop(path: Path) -> Path:
    """CLAHE/unsharp RGB variant of a crop (handwriting-friendly)."""
    from .preprocess import contrast_variant

    img = contrast_variant(Image.open(path).convert("RGB"))
    return _save_temp(img, path.stem, "ctr")


def upscale_labeled_crops(
    crops: list[tuple[str, Path]],
    *,
    scale: float = 2.0,
) -> list[tuple[str, Path]]:
    """Return new labeled paths with upscaled images; originals unchanged."""
    out: list[tuple[str, Path]] = []
    for label, path in crops:
        try:
            up = upscale_crop(path, scale=scale)
            out.append((f"{label}-up", up))
        except Exception:
            continue
    return out

