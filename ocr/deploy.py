"""Deploy / shutdown the fixed OCR model on ray-hive via Ray Jobs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

JOB_DIR = Path(__file__).resolve().parent
DASHBOARD = os.environ.get("RAY_DASHBOARD", "http://10.0.1.52:8265")
SERVE = os.environ.get("RAY_SERVE", "http://10.0.1.52:8000")
HIVE_PIP = os.environ.get("RAY_HIVE_PIP", "ray-hive==0.2.1")
HIVE_PIP_INDEX = os.environ.get(
    "HIVE_PIP_EXTRA_INDEX",
    "http://10.0.0.52:3000/api/packages/admin/pypi/simple/",
)


def _client():
    from ray.job_submission import JobSubmissionClient

    saved_ray = os.environ.pop("RAY_ADDRESS", None)
    try:
        return JobSubmissionClient(DASHBOARD)
    finally:
        if saved_ray is not None:
            os.environ["RAY_ADDRESS"] = saved_ray


def _runtime(env: dict) -> dict:
    # Only the ocr/ job scripts — never upload /opt/ray-hive (stale gpu_registry API).
    # Pip ray-hive on the job worker matches cluster 0.2.x images (see ray_hive/cluster/actor_env.py).
    merged = dict(env)
    merged.setdefault("PIP_EXTRA_INDEX_URL", HIVE_PIP_INDEX)
    merged.setdefault("PIP_INDEX_URL", "https://pypi.org/simple")
    merged.setdefault("RAY_HIVE_NAMESPACE", "ray_hive")
    return {
        "working_dir": str(JOB_DIR),
        "excludes": ["__pycache__", "*.pyc"],
        "pip": [HIVE_PIP],
        "env_vars": merged,
    }


def _tail_logs(client, jid: str) -> str:
    try:
        logs = client.get_job_logs(jid) or ""
    except Exception:
        return ""
    lines = [ln.strip() for ln in logs.strip().splitlines() if ln.strip()]
    return " | ".join(lines[-12:])[:800]


def _wait(client, jid: str, label: str, timeout: int) -> str:
    print(f"[{label}] {jid}", flush=True)
    t0 = time.time()
    last = ""
    while True:
        st = str(client.get_job_status(jid))
        if st != last:
            print(f"[{label}] {st} t={time.time() - t0:.0f}s", flush=True)
            last = st
        if st in ("SUCCEEDED", "FAILED", "STOPPED"):
            return st
        if time.time() - t0 > timeout:
            try:
                client.stop_job(jid)
            except Exception:
                pass
            return "TIMEOUT"
        time.sleep(5)


def _fail(client, jid: str, st: str) -> None:
    tail = _tail_logs(client, jid)
    if tail:
        raise RuntimeError(f"deploy failed: {st}: {tail}")
    raise RuntimeError(f"deploy failed: {st}")


def deploy_model(model_id: str, cfg: dict) -> None:
    client = _client()
    vkw = dict(cfg.get("vllm_kwargs") or {})
    extra = {
        k: v
        for k, v in vkw.items()
        if k not in ("max_num_seqs", "mm_processor_cache_gb", "trust_remote_code", "limit_mm_per_prompt")
    }

    jid = client.submit_job(
        entrypoint="python job_shutdown_model.py",
        runtime_env=_runtime(
            {
                "NINI_MODEL_ID": model_id,
                "NINI_SHUTDOWN_ALL": "0",
                "RAY_HIVE_NAMESPACE": "ray_hive",
            }
        ),
    )
    _wait(client, jid, "shutdown", 180)

    env = {
        "NINI_MODEL_ID": model_id,
        "NINI_MODEL_NAME": cfg["hf_name"],
        "NINI_MAX_IN": str(cfg["max_input"]),
        "NINI_MAX_OUT": str(cfg["max_output"]),
        "NINI_MAX_NUM_SEQS": str(int(vkw.get("max_num_seqs", 3))),
        "NINI_ALLOCATOR": "conserve",
        "NINI_MM_CACHE_GB": str(vkw.get("mm_processor_cache_gb", 0)),
        "NINI_SLEEP_TIMEOUT": os.environ.get("NINI_SLEEP_TIMEOUT", "-1"),
        "NINI_IDLE_TIMEOUT": os.environ.get("NINI_IDLE_TIMEOUT", "-1"),
        "RAY_HIVE_NAMESPACE": "ray_hive",
        "PYTHONUNBUFFERED": "1",
    }
    if extra:
        env["NINI_VLLM_EXTRA_JSON"] = json.dumps(extra)
    jid = client.submit_job(
        entrypoint="python -u job_deploy_model.py",
        runtime_env=_runtime(env),
    )
    st = _wait(client, jid, "deploy", 720)
    if st != "SUCCEEDED":
        _fail(client, jid, st)
    from . import model_status

    for _ in range(40):
        if model_status().get("up"):
            print(f"model ready: {SERVE}/{model_id}", flush=True)
            return
        time.sleep(3)
    raise RuntimeError("deploy succeeded but the model did not answer a readiness ping")


def shutdown_model(model_id: str) -> None:
    client = _client()
    jid = client.submit_job(
        entrypoint="python job_shutdown_model.py",
        runtime_env=_runtime(
            {
                "NINI_MODEL_ID": model_id,
                "NINI_SHUTDOWN_ALL": "0",
                "RAY_HIVE_NAMESPACE": "ray_hive",
            }
        ),
    )
    t0 = time.time()
    while time.time() - t0 < 120:
        st = str(client.get_job_status(jid))
        if st in ("SUCCEEDED", "FAILED", "STOPPED"):
            return
        time.sleep(3)
