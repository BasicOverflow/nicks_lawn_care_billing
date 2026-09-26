"""Document photo preprocessing for OCR (contrast, illumination, mild deskew)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


PreprocessLevel = Literal["off", "light", "doc"]


def preprocess_level() -> PreprocessLevel:
    raw = (os.environ.get("NINI_OCR_PREPROCESS") or "doc").strip().lower()
    if raw in ("0", "off", "none", "false", "no"):
        return "off"
    if raw in ("1", "light", "soft"):
        return "light"
    return "doc"


def enhance_for_ocr(img: Image.Image, level: PreprocessLevel | None = None) -> Image.Image:
    """Return an RGB image tuned for handwriting / phone-doc OCR.

    Goals: flatten uneven lighting (phone glare / warp shadows), raise local
    contrast, mild sharpen — without aggressive binarization that hurts VLMs.
    """
    lvl = preprocess_level() if level is None else level
    if lvl == "off":
        return img.convert("RGB")

    img = img.convert("RGB")
    try:
        return _enhance_cv2(img, lvl)
    except Exception:
        return _enhance_pil(img, lvl)


def _enhance_pil(img: Image.Image, level: PreprocessLevel) -> Image.Image:
    # Autocontrast + gentle boosts; no OpenCV needed.
    out = ImageOps.autocontrast(img, cutoff=1)
    if level == "doc":
        out = ImageEnhance.Contrast(out).enhance(1.4)
        out = ImageEnhance.Sharpness(out).enhance(1.4)
        out = ImageEnhance.Brightness(out).enhance(1.05)
    else:
        out = ImageEnhance.Contrast(out).enhance(1.15)
        out = ImageEnhance.Sharpness(out).enhance(1.1)
    return out


def _enhance_cv2(img: Image.Image, level: PreprocessLevel) -> Image.Image:
    import cv2

    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    bgr = _deskew_small(bgr) if level == "doc" else bgr
    bgr = _flatten_illumination(bgr)
    bgr = _clahe_lab(bgr, clip=3.5 if level == "doc" else 2.0)
    if level == "doc":
        bgr = _unsharp(bgr, amount=0.85, sigma=0.9)
    # Mild denoise that keeps ink edges
    bgr = cv2.bilateralFilter(bgr, d=5, sigmaColor=40, sigmaSpace=40)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _flatten_illumination(bgr: np.ndarray) -> np.ndarray:
    """Divide by large-kernel blur to cancel soft lighting / page curl shadows."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Kernel ~1/30 of min side — large enough for lighting, small enough to keep ink
    k = max(31, (min(v.shape) // 30) | 1)
    bg = cv2.GaussianBlur(v, (k, k), 0)
    bg = np.maximum(bg, 1)
    # Normalize toward mid-gray paper
    flat = cv2.divide(v.astype(np.float32), bg.astype(np.float32), scale=200.0)
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, flat]), cv2.COLOR_HSV2BGR)


def _clahe_lab(bgr: np.ndarray, clip: float = 3.0) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def _unsharp(bgr: np.ndarray, amount: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    import cv2

    blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def _deskew_small(bgr: np.ndarray, max_deg: float = 4.0) -> np.ndarray:
    """Correct small page tilt only; skip large angles (avoid wrecking layouts)."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size < 500:
        return bgr
    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV angle is [-90, 0); normalize to small skew
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.3 or abs(angle) > max_deg:
        return bgr
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        bgr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def perspective_rectify(img: Image.Image) -> Image.Image:
    """Four-point page warp when a clear paper quad is found; else return img.

    Safe for phone photos of binder sheets: only warps when the detected
    contour covers a large fraction of the frame and has 4 corners.
    """
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    # Work on a downscaled copy for contour speed
    scale = 1000 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = bgr
        scale = 1.0
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    page = None
    page_area = 0.0
    frame = float(sh * sw)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < frame * 0.25:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if area > page_area:
            page_area = area
            page = approx
    if page is None or page_area < frame * 0.35:
        return img.convert("RGB")

    pts = (page.reshape(4, 2).astype(np.float32) / scale)
    # Order: tl, tr, br, bl
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = ordered
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_w = int(max(width_a, width_b))
    max_h = int(max(height_a, height_b))
    if max_w < 200 or max_h < 200:
        return img.convert("RGB")
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(bgr, m, (max_w, max_h), flags=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def contrast_variant(img: Image.Image) -> Image.Image:
    """Extra local-contrast RGB for faint handwriting — not binarized."""
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _clahe_lab(bgr, clip=4.0)
    bgr = _unsharp(bgr, amount=1.0, sigma=0.8)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def enhance_ruled_sheet(img: Image.Image) -> Image.Image:
    """Whiten paper + punch black ink/grid for ruled table photos.

    Milder than a hard binarize: flatten glare, raise local contrast, nudge
    paper toward white and ink slightly darker so rows read clearer to a VLM.
    """
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _deskew_small(bgr, max_deg=3.0)
    bgr = _flatten_illumination(bgr)
    bgr = _clahe_lab(bgr, clip=3.2)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    paper = float(np.percentile(l, 92))
    ink_thr = max(50.0, paper * 0.82)
    lf = l.astype(np.float32)
    hi = np.clip((lf - ink_thr) / max(1.0, 255.0 - ink_thr), 0, 1)
    lf = lf + hi * (255.0 - lf) * 0.28
    lo = np.clip((ink_thr - lf) / max(1.0, ink_thr), 0, 1)
    lf = lf * (1.0 - 0.18 * lo)
    l2 = np.clip(lf, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    bgr = _unsharp(bgr, amount=0.7, sigma=0.9)
    bgr = cv2.bilateralFilter(bgr, d=5, sigmaColor=45, sigmaSpace=45)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def prepare_page(image_path: Path, *, novel: bool = False) -> Image.Image:
    """Load EXIF-upright page; optionally perspective-rectify then enhance."""
    path = Path(image_path)
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    if w > h * 1.15:
        img = img.rotate(-90, expand=True, fillcolor=(255, 255, 255))
    if novel:
        try:
            img = perspective_rectify(img)
        except Exception:
            pass
    return enhance_for_ocr(img)

