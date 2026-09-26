#!/usr/bin/env python3
"""Timed OCR bench: Gemma 4 (max soft tokens 1120) vs production qwen25-vl-3b novel.

Usage:
  py -3 scripts/bench_gemma4_vs_qwen.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

RAY_HIVE = Path(os.environ.get("RAY_HIVE", Path.home() / "Desktop" / "ray-hive"))
DASHBOARD = os.environ.get("RAY_DASHBOARD", "http://10.0.1.52:8265")
SERVE = os.environ.get("RAY_SERVE", "http://10.0.1.52:8000")
OUT = ROOT / "bench_results" / "gemma4_vs_qwen.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

GEMMA_ID = "gemma4-e2b-ocr"
GEMMA_HF = "google/gemma-4-E2B-it"
SOFT_TOKENS = 1120  # max budget for OCR resolution
QWEN_ID = "qwen25-vl-3b"
# Gemma is ~40× faster than Qwen novel — spend that budget on more passes.
GEMMA_MAX_PASSES = int(os.environ.get("GEMMA_MAX_PASSES", "8"))
GEMMA_MAX_SEQS = int(os.environ.get("GEMMA_MAX_SEQS", "4"))
# Reuse prior Qwen novel numbers unless REBENCH_QWEN=1
REBENCH_QWEN = (os.environ.get("REBENCH_QWEN") or "").strip().lower() in ("1", "true", "yes")
OUT_MULTI = ROOT / "bench_results" / "gemma4_multipass_vs_qwen.json"


def _norm_name(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _prices_from_gold(gold: dict) -> dict[str, float]:
    out = {}
    for row in gold.get("rows") or []:
        name = _norm_name(row.get("client") or "")
        if not name:
            continue
        price = row.get("mow_price")
        if price is None:
            price = row.get("hedge_price")
        if price is not None:
            out[name] = float(price)
    return out


def _prices_from_extract(extract: dict) -> dict[str, float]:
    """Best-effort: first col ~ name, any numeric col ~ price."""
    out: dict[str, float] = {}
    for t in extract.get("tables") or []:
        cols = [str(c).lower() for c in (t.get("columns") or [])]
        name_i = next(
            (i for i, c in enumerate(cols) if "contact" in c or "name" in c or "client" in c),
            0,
        )
        price_i = next(
            (i for i, c in enumerate(cols) if "price" in c or "mow" in c or "hedge" in c),
            -1,
        )
        for row in t.get("rows") or []:
            if not isinstance(row, list) or not row:
                continue
            name = _norm_name(str(row[name_i] if name_i < len(row) else ""))
            if not name or name in ("CONTACT", "NAME", "UNCLEAR"):
                continue
            price = None
            if price_i >= 0 and price_i < len(row):
                m = re.search(r"(\d+(?:\.\d+)?)", str(row[price_i]).replace(",", ""))
                if m:
                    price = float(m.group(1))
            if price is None:
                for cell in row:
                    m = re.search(r"\$?\s*(\d+(?:\.\d+)?)", str(cell))
                    if m and float(m.group(1)) >= 10:
                        price = float(m.group(1))
                        break
            if price is not None:
                out[name] = price
    return out


def score_extract(extract: dict, gold: dict) -> dict:
    g = _prices_from_gold(gold)
    p = _prices_from_extract(extract)
    if not g:
        return {"client_recall": 0.0, "price_recall": 0.0, "n_gold": 0, "n_pred": len(p)}
    hit = 0
    price_hit = 0
    for name, gp in g.items():
        # fuzzy: any pred name containing / contained
        match = None
        for pn, pp in p.items():
            if name == pn or name in pn or pn in name:
                match = pp
                break
        if match is not None:
            hit += 1
            if abs(match - gp) <= 1.01:
                price_hit += 1
    n = len(g)
    return {
        "client_recall": round(hit / n, 4),
        "price_recall": round(price_hit / n, 4),
        "n_gold": n,
        "n_pred": len(p),
        "matched_clients": hit,
        "matched_prices": price_hit,
    }


def load_fixtures() -> list[dict]:
    idx = json.loads((ROOT / "ground_truth" / "index.json").read_text(encoding="utf-8"))
    out = []
    for fx in idx["fixtures"]:
        img = ROOT / fx["image"]
        gold_path = ROOT / fx["gold_file"]
        if not img.is_file() or not gold_path.is_file():
            continue
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        out.append({"id": fx["id"], "image": img, "gold": gold, "kind": fx.get("kind")})
    return out


def wait_job(client, jid: str, label: str, timeout: int = 900) -> str:
    t0 = time.time()
    last = ""
    while True:
        st = str(client.get_job_status(jid))
        if st != last:
            print(f"  [{label}] {st} t={time.time()-t0:.0f}s", flush=True)
            last = st
        if st in ("SUCCEEDED", "FAILED", "STOPPED"):
            return st
        if time.time() - t0 > timeout:
            try:
                client.stop_job(jid)
            except Exception:
                pass
            return "TIMEOUT"
        time.sleep(4)


def submit_job(entrypoint: str, env: dict, label: str, timeout: int = 900) -> tuple[str, float]:
    from ray.job_submission import JobSubmissionClient

    saved = os.environ.pop("RAY_ADDRESS", None)
    try:
        client = JobSubmissionClient(DASHBOARD)
    finally:
        if saved is not None:
            os.environ["RAY_ADDRESS"] = saved
    t0 = time.time()
    jid = client.submit_job(
        entrypoint=entrypoint,
        runtime_env={"working_dir": str(RAY_HIVE), "env_vars": env},
    )
    st = wait_job(client, jid, label, timeout)
    elapsed = time.time() - t0
    if st != "SUCCEEDED":
        try:
            logs = client.get_job_logs(jid) or ""
            print(logs[-6000:], flush=True)
        except Exception as e:
            print(f"(could not fetch job logs: {e})", flush=True)
        raise RuntimeError(f"{label} job {st} after {elapsed:.1f}s")
    return jid, elapsed


def write_gemma_job_scripts() -> None:
    deploy = '''"""Deploy Gemma 4 E2B for OCR bench — max soft tokens."""
import json
import os
import ray
from ray_hive import RayHive
from ray_hive.core.model_specs import MultimodalAttentionSpecs

MODEL_ID = os.environ["NINI_MODEL_ID"]
MODEL_NAME = os.environ["NINI_MODEL_NAME"]
# "auto" fills leftover VRAM after weights+soft-token image budget.
MAX_IN = os.environ.get("NINI_MAX_IN", "auto")
MAX_OUT = int(os.environ.get("NINI_MAX_OUT", "4096"))
SOFT = int(os.environ.get("NINI_MAX_SOFT_TOKENS", "1120"))
# Only 24GB card in cluster can hold E2B @ 1120 soft tokens.
GPU = os.environ.get("NINI_GPU", "ergos-06-nv:gpu0")
MAX_SEQS = int(os.environ.get("NINI_MAX_NUM_SEQS", "2"))


class Gemma4MultimodalAttentionSpecs(MultimodalAttentionSpecs):
    def _layer_types(self):
        types = self.hf_params.get("layer_types")
        if types is not None:
            return list(types)
        return ["full_attention"] * self.num_layers

    def _kv_producer_layer_types(self):
        types = self._layer_types()
        shared = int(self.hf_params.get("num_kv_shared_layers") or 0)
        if shared <= 0:
            return types
        return types[: max(0, len(types) - shared)]

    def _layer_head_dim(self, layer_type: str) -> int:
        if layer_type == "full_attention":
            return int(self.hf_params.get("global_head_dim") or self.head_dim)
        return self.head_dim

    def _kv_tensors(self, layer_type: str) -> int:
        if layer_type == "full_attention" and self.hf_params.get("attention_k_eq_v"):
            return 1
        return 2

    @property
    def kv_layers(self) -> int:
        return len(self._kv_producer_layer_types())

    def kv_bytes_per_token(self) -> float:
        total = 0.0
        for layer_type in self._kv_producer_layer_types():
            total += (
                self._kv_tensors(layer_type)
                * self.kv_bytes_per_element
                * self.kv_heads
                * self._layer_head_dim(layer_type)
            )
        return total / self.tp_size

    def kv_bytes_per_sequence(self, max_model_len: int) -> float:
        window = self.hf_params.get("sliding_window")
        total = 0.0
        for layer_type in self._kv_producer_layer_types():
            bytes_per_token = (
                self._kv_tensors(layer_type)
                * self.kv_bytes_per_element
                * self.kv_heads
                * self._layer_head_dim(layer_type)
            )
            if layer_type == "sliding_attention" and window is not None:
                seq_tokens = min(max_model_len, int(window))
            else:
                seq_tokens = max_model_len
            total += bytes_per_token * seq_tokens
        return total / self.tp_size


def main():
    if not ray.is_initialized():
        ray.init(address="auto", namespace=os.environ.get("RAY_HIVE_NAMESPACE", "ray_hive"))
    hive = RayHive(address="auto", suppress_logging=False, show_banner=False)
    try:
        hive.shutdown(MODEL_ID)
    except Exception:
        pass
    # Free GPU from other OCR models
    for mid in ("qwen25-vl-3b", "qwen25-vl-3b-novel"):
        try:
            hive.shutdown(mid)
        except Exception:
            pass
    max_in = MAX_IN if MAX_IN == "auto" else int(MAX_IN)
    kwargs = dict(
        trust_remote_code=True,
        reasoning_parser="gemma4",
        default_chat_template_kwargs={"enable_thinking": False},
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_soft_tokens": SOFT},
        max_num_seqs=MAX_SEQS,
        mm_processor_cache_gb=0,
        enable_prefix_caching=False,
    )
    print(
        f"deploy {MODEL_ID} soft_tokens={SOFT} max_in={max_in} max_out={MAX_OUT} gpu={GPU} kwargs={kwargs}",
        flush=True,
    )
    # sleep_timeout>0 doubles fixed VRAM (weights held while asleep) and
    # pushes E2B over the 24GB card; bench keeps the model awake.
    status = hive.deploy_model(
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        max_input_prompt_length=max_in,
        max_output_prompt_length=MAX_OUT,
        replicas=1,
        gpu=GPU,
        attention_cls=Gemma4MultimodalAttentionSpecs,
        sleep_timeout=-1,
        idle_timeout=-1,
        vllm_kwargs=kwargs,
    )
    print(status, flush=True)
    print("DEPLOY_OK", flush=True)


if __name__ == "__main__":
    main()
'''
    (RAY_HIVE / "job_deploy_gemma4_ocr.py").write_text(deploy, encoding="utf-8")
    # reuse shutdown
    src = ROOT / "ocr" / "job_shutdown_model.py"
    if src.is_file():
        (RAY_HIVE / "job_shutdown_model.py").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def model_up(model_id: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{SERVE}/{model_id}/v1/models", timeout=8) as r:
            return r.status == 200
    except Exception:
        return False


def deploy_gemma() -> float:
    write_gemma_job_scripts()
    # Bust Ray working_dir cache so updated job_deploy_gemma4_ocr.py is uploaded.
    (RAY_HIVE / ".runtime_env_bust").write_text(f"gemma-bench-{time.time():.0f}\n", encoding="utf-8")
    env = {
        "NINI_MODEL_ID": GEMMA_ID,
        "NINI_MODEL_NAME": GEMMA_HF,
        "NINI_MAX_IN": "auto",  # maximize text window after soft-token image budget
        "NINI_MAX_OUT": "4096",
        "NINI_MAX_SOFT_TOKENS": str(SOFT_TOKENS),
        "NINI_MAX_NUM_SEQS": str(GEMMA_MAX_SEQS),
        "NINI_GPU": "ergos-06-nv:gpu0",
        "NINI_SHUTDOWN_ALL": "0",
        "RAY_HIVE_NAMESPACE": "ray_hive",
        "PYTHONUNBUFFERED": "1",
    }
    # shutdown target first
    submit_job(
        "python job_shutdown_model.py",
        {"NINI_MODEL_ID": GEMMA_ID, "RAY_HIVE_NAMESPACE": "ray_hive"},
        "gemma-shutdown",
        180,
    )
    _, elapsed = submit_job("python -u job_deploy_gemma4_ocr.py", env, "gemma-deploy", 1800)
    for _ in range(60):
        if model_up(GEMMA_ID):
            return elapsed
        time.sleep(2)
    raise RuntimeError("Gemma deploy succeeded but /v1/models never came up")


def shutdown_model(model_id: str) -> float:
    t0 = time.time()
    write_gemma_job_scripts()
    try:
        submit_job(
            "python job_shutdown_model.py",
            {
                "NINI_MODEL_ID": model_id,
                "NINI_SHUTDOWN_ALL": "0",
                "RAY_HIVE_NAMESPACE": "ray_hive",
            },
            f"shutdown-{model_id}",
            180,
        )
    except Exception as e:
        print(f"shutdown {model_id}: {e}", flush=True)
    return time.time() - t0


def gemma_extract(image: Path) -> dict:
    """Novel multipass extract on Gemma — more passes funded by fast inference."""
    os.environ["NINI_NOVEL"] = "1"
    os.environ.setdefault("NINI_IMAGE_EDGE", "2800")
    os.environ.setdefault("NINI_NOVEL_IMAGE_EDGE", "2800")

    from ocr.pipeline import run_fixture

    cfg = {
        "id": GEMMA_ID,
        "hf_name": GEMMA_HF,
        "max_input": 3072,
        "max_output": 4096,
        "novel": True,
        "vllm_kwargs": {
            "trust_remote_code": True,
            "limit_mm_per_prompt": {"image": 1},
            "mm_processor_cache_gb": 0,
            "max_num_seqs": GEMMA_MAX_SEQS,
            "mm_processor_kwargs": {"max_soft_tokens": SOFT_TOKENS},
        },
        "prompt_style": "chat_ocr_md",
    }
    fx = {"id": image.stem, "image": str(image)}
    row = run_fixture(
        GEMMA_ID,
        cfg,
        fx,
        GEMMA_MAX_PASSES,
        wave_workers=GEMMA_MAX_SEQS,
        asset_dir=None,
        image_path=image,
    )
    raw = row.get("final_extract") or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"title": None, "tables": [], "notes": [raw[:4000]], "complete": False}


def qwen_extract(image: Path) -> dict:
    import ocr

    return ocr.extract_sheet(image)


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def main() -> None:
    bench_t0 = time.time()
    fixtures = load_fixtures()
    print(
        f"fixtures={len(fixtures)} soft_tokens={SOFT_TOKENS} "
        f"gemma_max_passes={GEMMA_MAX_PASSES} seqs={GEMMA_MAX_SEQS}",
        flush=True,
    )
    prior = {}
    if OUT.is_file():
        try:
            prior = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            prior = {}

    prior_gemma = prior.get("gemma") or {}
    if prior_gemma.get("mode") == "novel_multipass":
        singlepass_baseline = prior.get("gemma_singlepass_baseline")
    else:
        singlepass_baseline = prior_gemma if prior_gemma.get("per_image") else None

    results: dict = {
        "soft_tokens": SOFT_TOKENS,
        "gemma_mode": "novel_multipass",
        "gemma_max_passes": GEMMA_MAX_PASSES,
        "gemma_max_seqs": GEMMA_MAX_SEQS,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gemma": {
            "model": GEMMA_HF,
            "id": GEMMA_ID,
            "mode": "novel_multipass",
            "max_passes": GEMMA_MAX_PASSES,
            "per_image": [],
        },
        "qwen": {"model": "Qwen/Qwen2.5-VL-3B-Instruct", "id": QWEN_ID, "per_image": []},
        "gemma_singlepass_baseline": singlepass_baseline,
    }

    # --- Gemma multipass ---
    print(
        f"=== Deploy Gemma 4 E2B @ {SOFT_TOKENS} soft tokens, novel multipass "
        f"(max_passes={GEMMA_MAX_PASSES}) ===",
        flush=True,
    )
    t_deploy_g = deploy_gemma()
    results["gemma"]["deploy_s"] = round(t_deploy_g, 2)
    print(f"Gemma deploy wall={t_deploy_g:.1f}s", flush=True)

    for fx in fixtures:
        print(f"  gemma multipass OCR {fx['id']}…", flush=True)
        t0 = time.time()
        try:
            extract = gemma_extract(fx["image"])
            err = None
        except Exception as e:
            extract = {"title": None, "tables": [], "notes": [], "complete": False}
            err = str(e)
        elapsed = time.time() - t0
        sc = score_extract(extract, fx["gold"])
        row = {
            "id": fx["id"],
            "seconds": round(elapsed, 2),
            "error": err,
            **sc,
            "n_tables": len(extract.get("tables") or []),
        }
        results["gemma"]["per_image"].append(row)
        print(f"    {elapsed:.1f}s client={sc['client_recall']} price={sc['price_recall']} err={err}", flush=True)

    g_times = [r["seconds"] for r in results["gemma"]["per_image"]]
    results["gemma"]["total_infer_s"] = round(sum(g_times), 2)
    results["gemma"]["mean_infer_s"] = mean(g_times)
    results["gemma"]["mean_client_recall"] = mean([r["client_recall"] for r in results["gemma"]["per_image"]])
    results["gemma"]["mean_price_recall"] = mean([r["price_recall"] for r in results["gemma"]["per_image"]])

    print("=== Shutdown Gemma ===", flush=True)
    results["gemma"]["shutdown_s"] = round(shutdown_model(GEMMA_ID), 2)

    # --- Qwen: reuse prior novel timings unless REBENCH_QWEN ---
    prior_qwen = prior.get("qwen") or {}
    if not REBENCH_QWEN and prior_qwen.get("per_image"):
        print("=== Reusing prior Qwen novel results (set REBENCH_QWEN=1 to re-run) ===", flush=True)
        results["qwen"] = dict(prior_qwen)
        results["qwen"]["reused"] = True
    else:
        print("=== Deploy/ensure Qwen ===", flush=True)
        import ocr

        t0 = time.time()
        if not model_up(QWEN_ID):
            print("loading qwen…", flush=True)
            ocr.load_model()
        results["qwen"]["deploy_s"] = round(time.time() - t0, 2)
        print(f"Qwen ready in {results['qwen']['deploy_s']:.1f}s", flush=True)

        for fx in fixtures:
            print(f"  qwen novel OCR {fx['id']}…", flush=True)
            t0 = time.time()
            try:
                extract = qwen_extract(fx["image"])
                err = None
            except Exception as e:
                extract = {"title": None, "tables": [], "notes": [], "complete": False}
                err = str(e)
            elapsed = time.time() - t0
            sc = score_extract(extract, fx["gold"])
            row = {
                "id": fx["id"],
                "seconds": round(elapsed, 2),
                "error": err,
                **sc,
                "n_tables": len(extract.get("tables") or []),
            }
            results["qwen"]["per_image"].append(row)
            print(f"    {elapsed:.1f}s client={sc['client_recall']} price={sc['price_recall']} err={err}", flush=True)

        q_times = [r["seconds"] for r in results["qwen"]["per_image"]]
        results["qwen"]["total_infer_s"] = round(sum(q_times), 2)
        results["qwen"]["mean_infer_s"] = mean(q_times)
        results["qwen"]["mean_client_recall"] = mean([r["client_recall"] for r in results["qwen"]["per_image"]])
        results["qwen"]["mean_price_recall"] = mean([r["price_recall"] for r in results["qwen"]["per_image"]])
        results["qwen"]["reused"] = False

    baseline = results.get("gemma_singlepass_baseline") or {}
    results["summary"] = {
        "winner_speed": "gemma" if results["gemma"]["mean_infer_s"] < results["qwen"]["mean_infer_s"] else "qwen",
        "winner_client_recall": (
            "gemma"
            if results["gemma"]["mean_client_recall"] > results["qwen"]["mean_client_recall"]
            else "qwen"
            if results["qwen"]["mean_client_recall"] > results["gemma"]["mean_client_recall"]
            else "tie"
        ),
        "speedup_vs_qwen": round(
            results["qwen"]["mean_infer_s"] / results["gemma"]["mean_infer_s"], 2
        )
        if results["gemma"]["mean_infer_s"]
        else None,
        "gemma_vs_singlepass_client": {
            "singlepass": baseline.get("mean_client_recall"),
            "multipass": results["gemma"]["mean_client_recall"],
        },
        "gemma_vs_singlepass_price": {
            "singlepass": baseline.get("mean_price_recall"),
            "multipass": results["gemma"]["mean_price_recall"],
        },
    }

    results["total_wall_s"] = round(time.time() - bench_t0, 2)
    results["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    OUT_MULTI.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Also refresh primary OUT so consumers see latest gemma numbers
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2), flush=True)
    print(f"total_wall={results['total_wall_s']}s Wrote {OUT_MULTI}", flush=True)


if __name__ == "__main__":
    main()
