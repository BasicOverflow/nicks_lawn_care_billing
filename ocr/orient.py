"""VLM page orientation for live OCR (phone binder shots)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .chat import chat_with_image
from .jsonutil import try_parse_json
from .prompts import PAGE_ORIENT_PROMPT, PAGE_ORIENT_VERIFY_PROMPT, PAGE_ORIENT_VERIFY_SCHEMA


def _exif_rgb(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _rotate_cw(img: Image.Image, rotate_cw_deg: int) -> Image.Image:
    rot = int(rotate_cw_deg) % 360
    if not rot:
        return img
    return img.rotate(-rot, expand=True, fillcolor=(255, 255, 255))


def _candidate_rotations(base: Image.Image) -> list[tuple[int, str]]:
    """Portrait phone shots of landscape tables → only offer 90° and 270° CW."""
    w, h = base.size
    if h > w * 1.05:
        return [(90, "B"), (270, "D")]
    return [(0, "A"), (90, "B"), (180, "C"), (270, "D")]


def _make_orient_montage(image_path: Path, *, cell: int = 640) -> tuple[Path, dict[str, int]]:
    """Montage of candidate rotations; returns (path, label→rotate_cw_deg)."""
    base = _exif_rgb(image_path)
    cands = _candidate_rotations(base)
    label_to_deg = {lab: deg for deg, lab in cands}
    tiles: list[Image.Image] = []
    for deg, lab in cands:
        tile = _rotate_cw(base, deg)
        tile.thumbnail((cell, cell))
        canvas = Image.new("RGB", (cell, cell), (245, 245, 245))
        ox = (cell - tile.size[0]) // 2
        oy = (cell - tile.size[1]) // 2
        canvas.paste(tile, (ox, oy))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 52, 40), fill=(20, 20, 20))
        draw.text((14, 8), lab, fill=(255, 255, 255))
        tiles.append(canvas)

    if len(tiles) == 2:
        grid = Image.new("RGB", (cell * 2 + 12, cell + 8), (30, 30, 30))
        grid.paste(tiles[0], (4, 4))
        grid.paste(tiles[1], (cell + 8, 4))
    else:
        grid = Image.new("RGB", (cell * 2 + 8, cell * 2 + 8), (30, 30, 30))
        grid.paste(tiles[0], (4, 4))
        grid.paste(tiles[1], (cell + 4, 4))
        grid.paste(tiles[2], (4, cell + 4))
        grid.paste(tiles[3], (cell + 4, cell + 4))

    out = Path(tempfile.gettempdir()) / (
        f"nini_orient_montage_{os.getpid()}_{image_path.stem}_{os.urandom(3).hex()}.jpg"
    )
    grid.save(out, format="JPEG", quality=92, optimize=True)
    return out, label_to_deg


def _verify_upright(model_id: str, upright_path: Path) -> tuple[bool, int, str]:
    """Ask if page is upright; if not, return extra CW degrees to apply."""
    try:
        text = chat_with_image(
            model_id,
            upright_path,
            PAGE_ORIENT_VERIFY_PROMPT,
            max_tokens=96,
            temperature=0.0,
            guided_json=True,
            json_schema=PAGE_ORIENT_VERIFY_SCHEMA,
            schema_name="page_orient_verify",
            max_edge=1400,
            enhance=False,
        )
        obj, _ = try_parse_json(text or "")
        if not isinstance(obj, dict):
            return True, 0, "verify parse fail"
        ok = bool(obj.get("ok", True))
        extra = int(obj.get("fix_rotate_cw_deg", 0)) % 360
        if extra not in (0, 90, 180, 270):
            extra = 0
        reason = str(obj.get("reason") or "").strip() or ("ok" if ok else "needs fix")
        if ok:
            return True, 0, reason
        return False, extra, reason
    except Exception as e:
        return True, 0, f"verify fail: {e}"


def _rule_score(base: Image.Image, rotate_cw_deg: int) -> tuple[int, float]:
    """Count detected horizontal rules after rotation; also return width/height."""
    from .table_split import detect_horizontal_rules

    img = _rotate_cw(base, rotate_cw_deg)
    w, h = img.size
    aspect = w / max(1, h)
    max_edge = 1800
    scale = max_edge / max(w, h)
    if scale < 1.0:
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.BILINEAR,
        )
    tmp = Path(tempfile.gettempdir()) / (
        f"nini_orient_score_{os.getpid()}_{int(rotate_cw_deg) % 360}_{os.urandom(3).hex()}.jpg"
    )
    try:
        img.save(tmp, format="JPEG", quality=85)
        n = len(detect_horizontal_rules(tmp))
    except Exception:
        n = 0
    finally:
        tmp.unlink(missing_ok=True)
    return n, aspect


def ask_rotate_cw_deg(
    model_id: str,
    image_path: Path | str,
    *,
    max_tokens: int = 128,
) -> tuple[int, str]:
    """Ask the VLM which rotation is upright for table OCR.

    Portrait binder shots only offer landscape candidates (90° / 270° CW).
    Verify may propose a further spin; accept it only when horizontal-rule
    geometry improves (stops verify from flipping already-upright landscape
    binder-flat shots into sideways portrait).
    """
    page = Path(image_path)
    montage, label_to_deg = _make_orient_montage(page)
    labels = sorted(label_to_deg.keys())
    try:
        prompt = PAGE_ORIENT_PROMPT.format(labels=", ".join(labels))
        # Dynamic schema enum for available labels
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "choice": {"type": "string", "enum": labels},
                "rotate_cw_deg": {
                    "type": "integer",
                    "enum": sorted(set(label_to_deg.values())),
                },
                "reason": {"type": "string"},
            },
            "required": ["choice", "rotate_cw_deg", "reason"],
        }
        text = chat_with_image(
            model_id,
            montage,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            guided_json=True,
            json_schema=schema,
            schema_name="page_orient",
            max_edge=1500,
            enhance=False,
        )
        obj, _ = try_parse_json(text or "")
        if not isinstance(obj, dict):
            deg = next(iter(label_to_deg.values()))
            reason = "orient parse failed; defaulted"
        else:
            choice = str(obj.get("choice") or "").strip().upper()
            if choice in label_to_deg:
                deg = label_to_deg[choice]
            else:
                deg = int(obj.get("rotate_cw_deg", 90)) % 360
                if deg not in label_to_deg.values():
                    deg = next(iter(label_to_deg.values()))
            reason = str(obj.get("reason") or "").strip() or f"choice={choice}"

        upright = materialize_upright_page(page, rotate_cw_deg=deg, enhance=False)
        try:
            ok, extra, vreason = _verify_upright(model_id, upright)
            if not ok and extra:
                cand = (deg + extra) % 360
                base = _exif_rgb(page)
                n0, ar0 = _rule_score(base, deg)
                n1, ar1 = _rule_score(base, cand)
                trial = _rotate_cw(base, cand)
                src_landscape = base.size[0] >= base.size[1]
                src_portrait = base.size[1] > base.size[0] * 1.05
                trial_portrait = trial.size[1] > trial.size[0] * 1.05
                # Flat landscape binder shots are already upright in-camera.
                # Verify often proposes +90 into portrait; that makes column
                # dividers look like "horizontal rules" and falsely scores higher.
                if src_landscape and trial_portrait:
                    reason = (
                        f"{reason} | verify+{extra} ignored "
                        f"(landscape capture→portrait; rules {n0}/{n1}): {vreason}"
                    )
                # Portrait phone shots of landscape tables must stay landscape.
                elif src_portrait and trial_portrait:
                    reason = (
                        f"{reason} | verify+{extra} ignored "
                        f"(would stay/portrait; rules {n0}/{n1}): {vreason}"
                    )
                elif n1 >= n0 + 4 or (n1 > n0 and ar1 >= 1.0 and ar0 < 0.95):
                    deg = cand
                    reason = (
                        f"{reason} | verify+{extra}: {vreason} "
                        f"(rules {n0}->{n1})"
                    )
                else:
                    reason = (
                        f"{reason} | verify+{extra} ignored "
                        f"(rules {n0}>={n1}, ar {ar0:.2f}/{ar1:.2f}): {vreason}"
                    )
            else:
                reason = f"{reason} | verify: {vreason}"
        finally:
            upright.unlink(missing_ok=True)
        return deg, reason
    except Exception as e:
        # Safe default for portrait binder sheets: 90° CW
        base = _exif_rgb(page)
        fallback = 90 if base.size[1] > base.size[0] else 0
        return fallback, f"orient fail: {e}"
    finally:
        montage.unlink(missing_ok=True)


def materialize_upright_page(
    image_path: Path | str,
    *,
    rotate_cw_deg: int = 0,
    enhance: bool = True,
) -> Path:
    """EXIF-transpose + clockwise rotate → deskew/whitened JPEG for cropping."""
    from .preprocess import enhance_ruled_sheet, perspective_rectify

    src = Path(image_path)
    img = _rotate_cw(_exif_rgb(src), rotate_cw_deg)
    if enhance:
        try:
            img = perspective_rectify(img)
        except Exception:
            pass
        try:
            img = enhance_ruled_sheet(img)
        except Exception:
            pass
    out = Path(tempfile.gettempdir()) / (
        f"nini_upright_{os.getpid()}_{src.stem}_{int(rotate_cw_deg) % 360}_{os.urandom(3).hex()}.jpg"
    )
    img.save(out, format="JPEG", quality=95, optimize=True)
    return out
