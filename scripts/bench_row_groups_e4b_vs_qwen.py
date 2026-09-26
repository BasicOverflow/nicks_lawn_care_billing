#!/usr/bin/env python3
"""Row-group OCR bench: bigger Gemma 4 (E4B) vs Qwen2.5-VL-3B on the same pipeline.

Usage:
  py -3 scripts/bench_row_groups_e4b_vs_qwen.py
  set GEMMA_HF=google/gemma-4-E4B-it
  set GEMMA_MAX_SOFT_TOKENS=896
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

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bench_gemma4_vs_qwen", ROOT / "scripts" / "bench_gemma4_vs_qwen.py"
)
_bench = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_bench)

OUT = ROOT / "bench_results" / "row_groups_e4b_vs_qwen.json"
ROWS_PER = max(2, min(4, int(os.environ.get("GEMMA_ROWS_PER_CROP", "3"))))
MAX_SEQS = int(os.environ.get("GEMMA_MAX_SEQS", "4"))
# E4B is larger than E2B; default a bit under max soft tokens for 24GB headroom
SOFT = int(os.environ.get("GEMMA_MAX_SOFT_TOKENS", "560"))

GEMMA_HF = os.environ.get("GEMMA_HF", "google/gemma-4-E4B-it")
GEMMA_ID = os.environ.get("GEMMA_ID", "gemma4-e4b-ocr")
QWEN_ID = _bench.QWEN_ID
QWEN_HF = "Qwen/Qwen2.5-VL-3B-Instruct"

load_fixtures = _bench.load_fixtures
mean = _bench.mean
model_up = _bench.model_up
score_extract = _bench.score_extract
shutdown_model = _bench.shutdown_model
submit_job = _bench.submit_job
write_gemma_job_scripts = _bench.write_gemma_job_scripts
RAY_HIVE = _bench.RAY_HIVE


def deploy_gemma(hf: str, model_id: str, soft: int, seqs: int) -> float:
    write_gemma_job_scripts()
    (RAY_HIVE / ".runtime_env_bust").write_text(f"gemma-e4b-{time.time():.0f}\n", encoding="utf-8")
    # Free sibling OCR models (E2B leftover, Qwen, etc.)
    for mid in (model_id, "gemma4-e2b-ocr", "gemma4-e4b-ocr", QWEN_ID, "qwen25-vl-3b-novel"):
        try:
            submit_job(
                "python job_shutdown_model.py",
                {"NINI_MODEL_ID": mid, "RAY_HIVE_NAMESPACE": "ray_hive"},
                f"pre-shutdown-{mid}",
                120,
            )
        except Exception:
            pass
    env = {
        "NINI_MODEL_ID": model_id,
        "NINI_MODEL_NAME": hf,
        "NINI_MAX_IN": "auto",
        "NINI_MAX_OUT": "4096",
        "NINI_MAX_SOFT_TOKENS": str(soft),
        "NINI_MAX_NUM_SEQS": str(seqs),
        "NINI_GPU": os.environ.get("NINI_GPU", "ergos-06-nv:gpu0"),
        "NINI_SHUTDOWN_ALL": "0",
        "RAY_HIVE_NAMESPACE": "ray_hive",
        "PYTHONUNBUFFERED": "1",
    }
    _, elapsed = submit_job("python -u job_deploy_gemma4_ocr.py", env, "gemma-e4b-deploy", 2400)
    for _ in range(90):
        if model_up(model_id):
            return elapsed
        time.sleep(2)
    raise RuntimeError(f"{model_id} deploy ok but /v1/models never came up")


def ensure_qwen() -> float:
    import ocr

    t0 = time.time()
    if not model_up(QWEN_ID):
        print("loading qwen…", flush=True)
        ocr.load_model()
    return time.time() - t0


def run_row_groups(model_id: str, fixtures: list[dict], label: str) -> dict:
    from ocr.gemma_rows import extract_by_row_groups

    per = []
    for fx in fixtures:
        print(f"  {label} row-groups {fx['id']}…", flush=True)
        t1 = time.time()
        try:
            extract = extract_by_row_groups(
                model_id,
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
        per.append(row)
        print(
            f"    {elapsed:.1f}s client={sc['client_recall']} price={sc['price_recall']} err={err}",
            flush=True,
        )
    times = [r["seconds"] for r in per]
    return {
        "per_image": per,
        "total_infer_s": round(sum(times), 2),
        "mean_infer_s": mean(times),
        "mean_client_recall": mean([r["client_recall"] for r in per]),
        "mean_price_recall": mean([r["price_recall"] for r in per]),
    }


def main() -> None:
    os.environ["GEMMA_MAX_SEQS"] = str(MAX_SEQS)
    fixtures = load_fixtures()
    prior_e2b = {}
    e2b_path = ROOT / "bench_results" / "gemma4_row_groups.json"
    if e2b_path.is_file():
        try:
            prior_e2b = json.loads(e2b_path.read_text(encoding="utf-8"))
        except Exception:
            prior_e2b = {}

    print(
        f"fixtures={len(fixtures)} soft_tokens={SOFT} rows_per_crop={ROWS_PER} seqs={MAX_SEQS}",
        flush=True,
    )
    print(f"Gemma HF={GEMMA_HF} id={GEMMA_ID}", flush=True)
    print(f"Qwen HF={QWEN_HF} id={QWEN_ID}", flush=True)

    results: dict = {
        "mode": "ruled_row_groups",
        "rows_per_crop": ROWS_PER,
        "max_seqs": MAX_SEQS,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gemma_e4b": {
            "model": GEMMA_HF,
            "id": GEMMA_ID,
            "soft_tokens": SOFT,
            "per_image": [],
        },
        "qwen_rows": {"model": QWEN_HF, "id": QWEN_ID, "per_image": []},
        "baseline_e2b_rows": {
            "model": (prior_e2b.get("gemma") or {}).get("model"),
            "mean_infer_s": (prior_e2b.get("gemma") or {}).get("mean_infer_s"),
            "mean_client_recall": (prior_e2b.get("gemma") or {}).get("mean_client_recall"),
            "mean_price_recall": (prior_e2b.get("gemma") or {}).get("mean_price_recall"),
        },
    }

    wall0 = time.time()

    # --- Gemma E4B ---
    print(f"=== Deploy {GEMMA_HF} @ {SOFT} soft tokens ===", flush=True)
    results["gemma_e4b"]["deploy_s"] = round(deploy_gemma(GEMMA_HF, GEMMA_ID, SOFT, MAX_SEQS), 2)
    g = run_row_groups(GEMMA_ID, fixtures, "gemma-e4b")
    results["gemma_e4b"].update(g)
    results["gemma_e4b"]["shutdown_s"] = round(shutdown_model(GEMMA_ID), 2)

    # --- Qwen same pipeline ---
    print("=== Ensure Qwen for same row-group pipeline ===", flush=True)
    results["qwen_rows"]["deploy_s"] = round(ensure_qwen(), 2)
    q = run_row_groups(QWEN_ID, fixtures, "qwen")
    results["qwen_rows"].update(q)
    results["qwen_rows"]["shutdown_s"] = round(shutdown_model(QWEN_ID), 2)

    ge = results["gemma_e4b"]
    qw = results["qwen_rows"]
    e2b = results["baseline_e2b_rows"]
    results["summary"] = {
        "gemma_e4b_mean_infer_s": ge["mean_infer_s"],
        "gemma_e4b_client": ge["mean_client_recall"],
        "gemma_e4b_price": ge["mean_price_recall"],
        "qwen_rows_mean_infer_s": qw["mean_infer_s"],
        "qwen_rows_client": qw["mean_client_recall"],
        "qwen_rows_price": qw["mean_price_recall"],
        "e2b_rows_client": e2b.get("mean_client_recall"),
        "e2b_rows_infer_s": e2b.get("mean_infer_s"),
        "speedup_e4b_vs_qwen_rows": round(qw["mean_infer_s"] / ge["mean_infer_s"], 2)
        if ge["mean_infer_s"]
        else None,
        "winner_client": (
            "gemma_e4b"
            if ge["mean_client_recall"] > qw["mean_client_recall"]
            else "qwen_rows"
            if qw["mean_client_recall"] > ge["mean_client_recall"]
            else "tie"
        ),
    }
    results["total_wall_s"] = round(time.time() - wall0, 2)
    results["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2), flush=True)
    print(f"Wrote {OUT} wall={results['total_wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
