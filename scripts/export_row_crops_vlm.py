"""Deploy-aware: VLM-orient each gold fixture, export upright + row-group crops."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
os.environ["NINI_NOVEL"] = "1"

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bench_gemma4_vs_qwen", ROOT / "scripts" / "bench_gemma4_vs_qwen.py"
)
_bench = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_bench)

OUT = ROOT / "bench_results" / "row_group_crops"
GEMMA_ID = _bench.GEMMA_ID


def main() -> None:
    if not _bench.model_up(GEMMA_ID):
        print("=== Deploy Gemma ===", flush=True)
        print("deploy_s=", round(_bench.deploy_gemma(), 2), flush=True)
    else:
        print("Gemma already up", flush=True)

    from ocr.orient import ask_rotate_cw_deg, materialize_upright_page
    from ocr.table_split import (
        detect_horizontal_rules,
        detect_row_bands_ruled,
        detect_table_bbox,
        make_ruled_row_crops,
    )

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    fixtures = _bench.load_fixtures()
    summary = []
    for fx in fixtures:
        print(f"\n=== {fx['id']} ===", flush=True)
        dest = OUT / fx["id"]
        dest.mkdir()
        t0 = time.time()
        rot, reason = ask_rotate_cw_deg(GEMMA_ID, fx["image"])
        upright = materialize_upright_page(fx["image"], rotate_cw_deg=rot)
        print(f"  orient rotate_cw={rot} ({reason}) {time.time()-t0:.1f}s", flush=True)

        # Save upright thumb + full
        from PIL import Image

        uimg = Image.open(upright)
        uimg.save(dest / "_upright_full.jpg", quality=90)
        thumb = uimg.copy()
        thumb.thumbnail((1000, 1000))
        thumb.save(dest / "_upright.jpg", quality=85)

        rules = detect_horizontal_rules(upright)
        bbox = detect_table_bbox(upright)
        bands = detect_row_bands_ruled(upright)
        heights = [b[1] - b[0] for b in bands]
        meta = {
            "rotate_cw": rot,
            "reason": reason,
            "upright_size": list(uimg.size),
            "rules": len(rules),
            "bbox": bbox,
            "bands": len(bands),
            "band_h_mean": round(sum(heights) / max(1, len(heights)), 1),
            "band_h_max": max(heights) if heights else 0,
            "equal_fallback": len(rules) < 4,
        }
        print(f"  meta={meta}", flush=True)

        hdr, crops, price_strip = make_ruled_row_crops(
            upright, rows_per_crop=3, scale=1.0, edge_pad_rows=0.5, x_pad=0.02, overlap=1
        )
        if hdr and hdr.is_file():
            shutil.copy(hdr, dest / "00_header.jpg")
            hdr.unlink(missing_ok=True)
        if price_strip and price_strip.is_file():
            shutil.copy(price_strip, dest / "00_price_strip.jpg")
            price_strip.unlink(missing_ok=True)
        for i, (label, path, _i) in enumerate(crops):
            shutil.copy(path, dest / f"{i+1:02d}_{label}.jpg")
            path.unlink(missing_ok=True)
        upright.unlink(missing_ok=True)
        meta["crops"] = len(crops)
        meta["price_strip"] = True
        (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        summary.append({"id": fx["id"], **meta})
        print(f"  crops={len(crops)}", flush=True)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nWrote", OUT, flush=True)


if __name__ == "__main__":
    main()
