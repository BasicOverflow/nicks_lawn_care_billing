"""Probe orientation + export debug crops until upright is correct."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def ink(gray: np.ndarray) -> np.ndarray:
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    paper = float(np.percentile(gray, 90))
    thr = max(40.0, min(200.0, paper * 0.72))
    _, hard = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 8
    )
    return cv2.bitwise_or(cv2.bitwise_or(otsu, hard), adapt)


def count_rules(gray: np.ndarray, horizontal: bool = True) -> tuple[int, float]:
    h, w = gray.shape
    m = ink(gray)
    if horizontal:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(80, w // 10), 1))
        rules = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        rules = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        proj = rules.sum(axis=1).astype(float)
        span = w
    else:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(80, h // 10)))
        rules = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        rules = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
        proj = rules.sum(axis=0).astype(float)
        span = h
    if proj.max() < 1:
        return 0, 0.0
    thr = max(float(np.percentile(proj[proj > 0], 55)), span * 0.2 * 255 * 0.12)
    peaks: list[int] = []
    i = 0
    L = len(proj)
    while i < L:
        if proj[i] >= thr:
            j = i
            best = i
            bv = proj[i]
            while j < L and proj[j] >= thr * 0.45:
                if proj[j] > bv:
                    bv = proj[j]
                    best = j
                j += 1
            peaks.append(best)
            i = j + 1
        else:
            i += 1
    gap = max(10, L // 180)
    cleaned: list[int] = []
    for y in peaks:
        if not cleaned or y - cleaned[-1] >= gap:
            cleaned.append(y)
        elif proj[y] > proj[cleaned[-1]]:
            cleaned[-1] = y
    if len(cleaned) < 3:
        return len(cleaned), float(len(cleaned))
    gaps = np.diff(cleaned)
    med = float(np.median(gaps))
    consistency = float(np.mean(np.abs(gaps - med) < med * 0.45)) if med > 0 else 0.0
    score = len(cleaned) * (0.5 + consistency)
    return len(cleaned), score


def main() -> None:
    out = ROOT / "bench_results" / "orient_probe"
    out.mkdir(parents=True, exist_ok=True)
    idx = json.loads((ROOT / "ground_truth" / "index.json").read_text(encoding="utf-8"))
    for fx in idx["fixtures"]:
        im = ImageOps.exif_transpose(Image.open(fx["image"])).convert("RGB")
        best = None
        for rot in (0, -90, 90, 180):
            x = im if rot == 0 else im.rotate(rot, expand=True, fillcolor=(255, 255, 255))
            g = cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2GRAY)
            nh, sh = count_rules(g, True)
            nv, sv = count_rules(g, False)
            key = (sh - 0.35 * sv, nh, -abs(rot) * 0.001)
            print(
                f"{fx['id']} rot={rot:4d} H={nh:2d}/{sh:.1f} V={nv:2d}/{sv:.1f} key={key[0]:.1f}"
            )
            if best is None or key > best[0]:
                best = (key, rot, x)
        assert best is not None
        thumb = best[2].copy()
        thumb.thumbnail((900, 900))
        thumb.save(out / f"{fx['id']}_best{best[1]}.jpg", quality=85)
        print(f"  -> best rot={best[1]} key={best[0]}")


if __name__ == "__main__":
    main()
