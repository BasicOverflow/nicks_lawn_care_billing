#!/usr/bin/env python3
"""Timed Gemma 4 row-group OCR (2–4 rows/crop) vs prior single-pass + Qwen numbers.

Usage:
  py -3 scripts/bench_gemma_rows.py
  set GEMMA_ROWS_PER_CROP=3
  set GEMMA_MAX_SEQS=4
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
os.environ["NINI_NOVEL"] = "1"

# Import sibling bench helpers without requiring scripts as a package
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bench_gemma4_vs_qwen", ROOT / "scripts" / "bench_gemma4_vs_qwen.py"
)
_bench = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_bench)

GEMMA_HF = _bench.GEMMA_HF
GEMMA_ID = _bench.GEMMA_ID
OUT = _bench.OUT
SOFT_TOKENS = _bench.SOFT_TOKENS
deploy_gemma = _bench.deploy_gemma
load_fixtures = _bench.load_fixtures
mean = _bench.mean
model_up = _bench.model_up
score_extract = _bench.score_extract
shutdown_model = _bench.shutdown_model


OUT_ROWS = ROOT / "bench_results" / "gemma4_row_groups.json"
ROWS_PER = max(2, min(4, int(os.environ.get("GEMMA_ROWS_PER_CROP", "3"))))
MAX_SEQS = int(os.environ.get("GEMMA_MAX_SEQS", "6"))


def main() -> None:
    # Bump concurrency for parallel row-group crops
    os.environ["GEMMA_MAX_SEQS"] = str(MAX_SEQS)

    fixtures = load_fixtures()
    prior = {}
    if OUT.is_file():
        try:
            prior = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            prior = {}

    print(
        f"fixtures={len(fixtures)} soft_tokens={SOFT_TOKENS} "
        f"rows_per_crop={ROWS_PER} seqs={MAX_SEQS}",
        flush=True,
    )
    results: dict = {
        "mode": "ruled_row_groups",
        "soft_tokens": SOFT_TOKENS,
        "rows_per_crop": ROWS_PER,
        "max_seqs": MAX_SEQS,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gemma": {"model": GEMMA_HF, "id": GEMMA_ID, "per_image": []},
        "baselines": {
            "singlepass": prior.get("gemma_singlepass_baseline") or (
                prior.get("gemma") if prior.get("gemma", {}).get("mode") != "novel_multipass" else None
            ),
            "multipass": prior.get("gemma") if prior.get("gemma", {}).get("mode") == "novel_multipass" else None,
            "qwen": prior.get("qwen"),
        },
    }

    t0 = time.time()
    if not model_up(GEMMA_ID):
        print("=== Deploy Gemma 4 E2B @ 1120 soft tokens ===", flush=True)
        results["gemma"]["deploy_s"] = round(deploy_gemma(), 2)
    else:
        results["gemma"]["deploy_s"] = 0.0
        print("Gemma already up", flush=True)

    from ocr.gemma_rows import extract_by_row_groups

    for fx in fixtures:
        print(f"  gemma row-groups {fx['id']}…", flush=True)
        t1 = time.time()
        try:
            extract = extract_by_row_groups(
                GEMMA_ID,
                fx["image"],
                rows_per_crop=ROWS_PER,
                max_seqs=MAX_SEQS,
                max_tokens=2048,
                scale=2.2,
            )
            err = None
        except Exception as e:
            extract = {"title": None, "tables": [], "notes": [], "complete": False}
            err = str(e)
        elapsed = time.time() - t1
        sc = score_extract(extract, fx["gold"])
        row = {"id": fx["id"], "seconds": round(elapsed, 2), "error": err, **sc}
        results["gemma"]["per_image"].append(row)
        print(
            f"    {elapsed:.1f}s client={sc['client_recall']} price={sc['price_recall']} err={err}",
            flush=True,
        )

    times = [r["seconds"] for r in results["gemma"]["per_image"]]
    results["gemma"]["total_infer_s"] = round(sum(times), 2)
    results["gemma"]["mean_infer_s"] = mean(times)
    results["gemma"]["mean_client_recall"] = mean(
        [r["client_recall"] for r in results["gemma"]["per_image"]]
    )
    results["gemma"]["mean_price_recall"] = mean(
        [r["price_recall"] for r in results["gemma"]["per_image"]]
    )
    results["gemma"]["shutdown_s"] = round(shutdown_model(GEMMA_ID), 2)

    sp = (results["baselines"].get("singlepass") or {}).get("mean_client_recall")
    qw = (results["baselines"].get("qwen") or {}).get("mean_client_recall")
    results["summary"] = {
        "rows_per_crop": ROWS_PER,
        "mean_infer_s": results["gemma"]["mean_infer_s"],
        "mean_client_recall": results["gemma"]["mean_client_recall"],
        "mean_price_recall": results["gemma"]["mean_price_recall"],
        "vs_singlepass_client": sp,
        "vs_qwen_client": qw,
    }
    results["total_wall_s"] = round(time.time() - t0, 2)
    results["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    OUT_ROWS.parent.mkdir(parents=True, exist_ok=True)
    OUT_ROWS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2), flush=True)
    print(f"Wrote {OUT_ROWS} wall={results['total_wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
