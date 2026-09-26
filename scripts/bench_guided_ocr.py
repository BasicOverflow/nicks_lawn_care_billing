#!/usr/bin/env python3
"""Bench knowledge-guided parallel full-image OCR on gold fixtures.

Usage:
  py -3 scripts/bench_guided_ocr.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
os.environ.setdefault("NINI_NOVEL", "1")

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bench_gemma4_vs_qwen", ROOT / "scripts" / "bench_gemma4_vs_qwen.py"
)
_bench = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_bench)

OUT = ROOT / "bench_results" / "guided_ocr.json"
CHUNK = int(os.environ.get("NINI_OCR_CHUNK", "4"))
QWEN_ID = _bench.QWEN_ID

load_fixtures = _bench.load_fixtures
mean = _bench.mean
model_up = _bench.model_up
score_extract = _bench.score_extract


def sheet_kind_for(fx: dict) -> str:
    kind = str(fx.get("kind") or "").lower()
    if "hedge" in kind:
        return "hedges"
    return "mowing"


def ensure_qwen() -> float:
    import ocr

    t0 = time.time()
    if model_up(QWEN_ID):
        print(f"{QWEN_ID} already up", flush=True)
        return 0.0
    print(f"loading {QWEN_ID}…", flush=True)
    ocr.load_model()
    for _ in range(90):
        if model_up(QWEN_ID):
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"{QWEN_ID} did not come up")
    return time.time() - t0


def prior_qwen_novel() -> dict:
    path = ROOT / "bench_results" / "gemma4_vs_qwen.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("qwen") or {}
    except Exception:
        return {}


def prior_row_groups_qwen() -> dict:
    path = ROOT / "bench_results" / "row_groups_e4b_vs_qwen.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("qwen") or {}
    except Exception:
        return {}


def main() -> None:
    prior_path = ROOT / "bench_results" / "guided_ocr.json"
    prior_guided = {}
    if prior_path.is_file():
        try:
            prior_guided = json.loads(prior_path.read_text(encoding="utf-8"))
        except Exception:
            prior_guided = {}
    import ocr
    from app import db

    fixtures = load_fixtures()
    if not fixtures:
        raise SystemExit("no fixtures")

    db.init_db()
    with db.connect() as conn:
        kb_mow = db.knowledge_names_for_sheet(conn, "mowing")
        kb_hedge = db.knowledge_names_for_sheet(conn, "hedges")

    models = {
        p.strip().lower()
        for p in os.environ.get("NINI_BENCH_MODELS", "qwen,gemma").split(",")
        if p.strip()
    }
    qwen = prior_guided.get("qwen") if isinstance(prior_guided.get("qwen"), dict) else None
    deploy_qwen_s = 0.0
    if "qwen" in models:
        deploy_qwen_s = ensure_qwen()
        print(
            f"guided OCR bench: {len(fixtures)} fixtures, chunk={CHUNK}, "
            f"kb mow={len(kb_mow)} hedges={len(kb_hedge)}",
            flush=True,
        )
        qwen = run_model(QWEN_ID, None, "qwen")
        qwen["deploy_s"] = round(deploy_qwen_s, 2)
        per = []
        cache: dict[str, list] = {}
        for fx in fixtures:
            sk = sheet_kind_for(fx)
            if sk not in cache:
                with db.connect() as conn:
                    cache[sk] = db.knowledge_records_for_sheet(conn, sk)
            records = cache[sk]
            print(
                f"  [{label}] {fx['id']} kind={sk} records={len(records)}…",
                flush=True,
            )
            t1 = time.time()
            try:
                extract = ocr.extract_sheet(
                    fx["image"],
                    sheet_kind=sk,
                    knowledge_records=records,
                    chunk_size=CHUNK,
                    model_id=model_id,
                    model_cfg=model_cfg,
                )
                err = None
            except Exception as e:
                extract = {"title": None, "tables": [], "notes": [], "complete": False}
                err = str(e)
            elapsed = time.time() - t1
            sc = score_extract(extract, fx["gold"])
            n_rows = 0
            for t in extract.get("tables") or []:
                if isinstance(t, dict):
                    n_rows += len(t.get("rows") or [])
            row = {
                "id": fx["id"],
                "sheet_kind": sk,
                "knowledge_names": len(records),
                "seconds": round(elapsed, 2),
                "error": err,
                "n_pred_rows": n_rows,
                "title": extract.get("title"),
                **sc,
                "extract_preview": {
                    "columns": (extract.get("tables") or [{}])[0].get("columns")
                    if extract.get("tables")
                    else [],
                    "n_notes": len(extract.get("notes") or []),
                    "sample_rows": ((extract.get("tables") or [{}])[0].get("rows") or [])[:3]
                    if extract.get("tables")
                    else [],
                },
            }
            per.append(row)
            print(
                f"    {elapsed:.1f}s client={sc['client_recall']} "
                f"price={sc['price_recall']} rows={n_rows} err={err}",
                flush=True,
            )
        times = [r["seconds"] for r in per]
        return {
            "id": model_id,
            "label": label,
            "per_image": per,
            "total_infer_s": round(sum(times), 2),
            "mean_infer_s": mean(times),
            "mean_client_recall": mean([r["client_recall"] for r in per]),
            "mean_price_recall": mean([r["price_recall"] for r in per]),
        }

    qwen = run_model(QWEN_ID, None, "qwen")
    qwen["deploy_s"] = round(deploy_qwen_s, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"partial": True, "qwen": qwen}, indent=2), encoding="utf-8")

    print("deploying gemma for the same guided method…", flush=True)
    t_g = time.time()
    if not model_up(_bench.GEMMA_ID):
        _bench.deploy_gemma()
    gemma_deploy_s = round(time.time() - t_g, 2)
    gemma_cfg = {
        "id": _bench.GEMMA_ID,
        "max_output": 4096,
        "vllm_kwargs": {"max_num_seqs": _bench.GEMMA_MAX_SEQS},
    }
    gemma = run_model(_bench.GEMMA_ID, gemma_cfg, "gemma")
    gemma["model"] = _bench.GEMMA_HF
    gemma["deploy_s"] = gemma_deploy_s

    result = {
        "mode": "knowledge_guided_locators_price_reread_unknown",
        "chunk_size": CHUNK,
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "knowledge": {"mowing": len(kb_mow), "hedges": len(kb_hedge)},
        "qwen": qwen,
        "gemma": gemma,
        "mean_client_recall": qwen["mean_client_recall"],
        "mean_price_recall": qwen["mean_price_recall"],
        "mean_infer_s": qwen["mean_infer_s"],
        "per_image": qwen["per_image"],
        "prior_guided": {
            "mean_client_recall": prior_guided.get("mean_client_recall"),
            "mean_price_recall": prior_guided.get("mean_price_recall"),
            "mean_infer_s": prior_guided.get("mean_infer_s"),
            "per_image": [
                {
                    "id": r.get("id"),
                    "client_recall": r.get("client_recall"),
                    "price_recall": r.get("price_recall"),
                    "seconds": r.get("seconds"),
                }
                for r in prior_guided.get("per_image") or []
            ],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "qwen_client": qwen["mean_client_recall"],
        "qwen_price": qwen["mean_price_recall"],
        "qwen_s": qwen["mean_infer_s"],
        "gemma_client": gemma["mean_client_recall"],
        "gemma_price": gemma["mean_price_recall"],
        "gemma_s": gemma["mean_infer_s"],
        "prior_client": prior_guided.get("mean_client_recall"),
        "prior_price": prior_guided.get("mean_price_recall"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
