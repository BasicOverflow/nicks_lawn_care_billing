"""Deploy / shutdown the fixed OCR model on ray-hive via Ray Jobs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAY_HIVE = Path(os.environ.get("RAY_HIVE", Path.home() / "Desktop" / "ray-hive"))
DASHBOARD = os.environ.get("RAY_DASHBOARD", "http://10.0.1.52:8265")
SERVE = os.environ.get("RAY_SERVE", "http://10.0.1.52:8000")


def _model_up(model_id: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{SERVE}/{model_id}/v1/models", timeout=8) as r:
            return r.status == 200
    except Exception:
        return False


def deploy_model(model_id: str, cfg: dict) -> None:
    from ray.job_submission import JobSubmissionClient

    for name in ("job_deploy_model.py", "job_shutdown_model.py"):
        src = Path(__file__).parent / name
        if src.is_file() and RAY_HIVE.is_dir():
            (RAY_HIVE / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    saved_ray = os.environ.pop("RAY_ADDRESS", None)
    try:
        client = JobSubmissionClient(DASHBOARD)
    finally:
        if saved_ray is not None:
            os.environ["RAY_ADDRESS"] = saved_ray

    vkw = dict(cfg.get("vllm_kwargs") or {})
    extra = {
        k: v
        for k, v in vkw.items()
        if k not in ("max_num_seqs", "mm_processor_cache_gb", "trust_remote_code", "limit_mm_per_prompt")
    }

    def wait(jid: str, label: str, timeout: int = 720) -> str:
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

    jid = client.submit_job(
        entrypoint="python job_shutdown_model.py",
        runtime_env={
            "working_dir": str(RAY_HIVE),
            "env_vars": {
                "NINI_MODEL_ID": model_id,
                "NINI_SHUTDOWN_ALL": "1",
                "RAY_HIVE_NAMESPACE": "ray_hive",
            },
        },
    )
    wait(jid, "shutdown", 180)

    env = {
        "NINI_MODEL_ID": model_id,
        "NINI_MODEL_NAME": cfg["hf_name"],
        "NINI_MAX_IN": str(cfg["max_input"]),
        "NINI_MAX_OUT": str(cfg["max_output"]),
        "NINI_MAX_NUM_SEQS": str(int(vkw.get("max_num_seqs", 3))),
        "NINI_ALLOCATOR": "conserve",
        "NINI_MM_CACHE_GB": str(vkw.get("mm_processor_cache_gb", 0)),
        # Sleep after 10m idle; fully offload (destroy) after 15m.
        "NINI_SLEEP_TIMEOUT": os.environ.get("NINI_SLEEP_TIMEOUT", "600"),
        "NINI_IDLE_TIMEOUT": os.environ.get("NINI_IDLE_TIMEOUT", "900"),
        "RAY_HIVE_NAMESPACE": "ray_hive",
        "PYTHONUNBUFFERED": "1",
    }
    if extra:
        env["NINI_VLLM_EXTRA_JSON"] = json.dumps(extra)
    (RAY_HIVE / ".runtime_env_bust").write_text(f"nini-{time.time():.0f}\n", encoding="utf-8")
    jid = client.submit_job(
        entrypoint="python -u job_deploy_model.py",
        runtime_env={"working_dir": str(RAY_HIVE), "env_vars": env},
    )
    st = wait(jid, "deploy", 720)
    if st != "SUCCEEDED":
        raise RuntimeError(f"deploy failed: {st}")
    for _ in range(30):
        if _model_up(model_id):
            print(f"model ready: {SERVE}/{model_id}", flush=True)
            return
        time.sleep(2)
    raise RuntimeError("deploy succeeded but /v1/models never came up")


def shutdown_model(model_id: str) -> None:
    from ray.job_submission import JobSubmissionClient

    src = Path(__file__).parent / "job_shutdown_model.py"
    if src.is_file() and RAY_HIVE.is_dir():
        (RAY_HIVE / "job_shutdown_model.py").write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8"
        )
    saved_ray = os.environ.pop("RAY_ADDRESS", None)
    try:
        client = JobSubmissionClient(DASHBOARD)
    finally:
        if saved_ray is not None:
            os.environ["RAY_ADDRESS"] = saved_ray
    jid = client.submit_job(
        entrypoint="python job_shutdown_model.py",
        runtime_env={
            "working_dir": str(RAY_HIVE),
            "env_vars": {
                "NINI_MODEL_ID": model_id,
                "NINI_SHUTDOWN_ALL": "1",
                "RAY_HIVE_NAMESPACE": "ray_hive",
            },
        },
    )
    t0 = time.time()
    while time.time() - t0 < 120:
        st = str(client.get_job_status(jid))
        if st in ("SUCCEEDED", "FAILED", "STOPPED"):
            return
        time.sleep(3)
