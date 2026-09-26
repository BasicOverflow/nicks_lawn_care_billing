"""Export row-group crops for visual QA."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ocr.table_split import (
    _load_upright,
    detect_horizontal_rules,
    detect_row_bands_ruled,
    detect_table_bbox,
    make_ruled_row_crops,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench_results" / "row_group_crops"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    idx = json.loads((ROOT / "ground_truth" / "index.json").read_text(encoding="utf-8"))
    for fx in idx["fixtures"]:
        if "work" not in str(fx.get("kind") or "").lower():
            continue
        img = ROOT / fx["image"]
        dest = OUT / fx["id"]
        dest.mkdir()
        upright = _load_upright(img)
        upright.thumbnail((1000, 1000))
        upright.save(dest / "_upright.jpg", quality=85)
        rules = detect_horizontal_rules(img)
        bbox = detect_table_bbox(img)
        bands = detect_row_bands_ruled(img)
        heights = [b[1] - b[0] for b in bands]
        meta = {
            "upright_size": list(_load_upright(img).size),
            "rules": len(rules),
            "bbox": bbox,
            "bands": len(bands),
            "band_h_mean": round(sum(heights) / len(heights), 1) if heights else 0,
            "band_h_max": max(heights) if heights else 0,
        }
        print(fx["id"], meta)
        hdr, crops = make_ruled_row_crops(
            img, rows_per_crop=3, scale=1.0, edge_pad_rows=0.45, x_pad=0.02
        )
        if hdr and hdr.is_file():
            shutil.copy(hdr, dest / "00_header.jpg")
            hdr.unlink(missing_ok=True)
        for i, (label, path, _idx) in enumerate(crops):
            shutil.copy(path, dest / f"{i+1:02d}_{label}.jpg")
            path.unlink(missing_ok=True)
        (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  crops={len(crops)}")


if __name__ == "__main__":
    main()
