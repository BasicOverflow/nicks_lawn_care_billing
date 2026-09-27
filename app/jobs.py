"""In-memory jobs. Each one keeps its own progress so they can run together."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Progress:
    job_id: str = ""
    kind: str = ""  # upload | ocr | generate | email | model | chat
    status: str = "idle"  # idle | running | done | error | cancelled
    message: str = ""
    percent: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    updated_at: float = 0.0


_lock = threading.Lock()
_jobs: dict[str, Progress] = {}
_local = threading.local()
# Keep a finished bar long enough for the next poll to paint it.
_KEEP_FINISHED = 25.0


def use(job_id: str) -> None:
    """Bind this thread's later progress updates to one job."""
    _local.job_id = job_id


def _bound(job_id: str | None) -> str | None:
    return job_id or getattr(_local, "job_id", None)


def _public(job: Progress) -> dict:
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "status": job.status,
        "message": job.message,
        "percent": job.percent,
        "detail": dict(job.detail),
    }


def _prune(now: float) -> None:
    stale = [
        jid
        for jid, job in _jobs.items()
        if job.status in ("done", "error", "cancelled") and now - job.updated_at > _KEEP_FINISHED
    ]
    for jid in stale:
        _jobs.pop(jid, None)


def get() -> dict:
    from . import ocr_queue

    q = ocr_queue.snapshot()
    now = time.time()
    with _lock:
        _prune(now)
        ordered = sorted(_jobs.values(), key=lambda job: job.updated_at)
        latest = ordered[-1] if ordered else Progress()
        payload = _public(latest)
        payload["jobs"] = [_public(job) for job in ordered if job.status != "idle"]
        payload["queue_depth"] = q["queue_depth"]
        payload["queued"] = q["queued"]
        payload["ocr_busy"] = q["busy"]
        return payload


def set_progress(
    *,
    kind: str | None = None,
    status: str | None = None,
    message: str | None = None,
    percent: int | None = None,
    job_id: str | None = None,
    detail: dict | None = None,
) -> None:
    jid = _bound(job_id)
    if not jid:
        return
    with _lock:
        job = _jobs.get(jid)
        if job is None:
            job = Progress(job_id=jid)
            _jobs[jid] = job
        if kind is not None:
            job.kind = kind
        if status is not None:
            job.status = status
        if message is not None:
            job.message = message
        if percent is not None:
            job.percent = max(0, min(100, int(percent)))
        if detail is not None:
            job.detail = detail
        job.updated_at = time.time()


def new_job(kind: str, message: str = "") -> str:
    jid = uuid.uuid4().hex[:12]
    set_progress(job_id=jid, kind=kind, status="running", message=message, percent=0, detail={})
    use(jid)
    return jid


def new_job_replace(job_id: str, kind: str, message: str = "") -> str:
    """Reuse an existing job id (queued OCR becoming active)."""
    set_progress(job_id=job_id, kind=kind, status="running", message=message, percent=0, detail={})
    use(job_id)
    return job_id


def running_id(kind: str | None = None) -> str | None:
    with _lock:
        for job in _jobs.values():
            if job.status == "running" and (kind is None or job.kind == kind):
                return job.job_id
    return None


def done(message: str = "Done", detail: dict | None = None, *, job_id: str | None = None) -> None:
    set_progress(job_id=job_id, status="done", message=message, percent=100, detail=detail)


def fail(message: str, *, job_id: str | None = None) -> None:
    set_progress(job_id=job_id, status="error", message=message, percent=100)


def cancel(message: str = "Cancelled", *, job_id: str | None = None) -> None:
    set_progress(job_id=job_id, status="cancelled", message=message, percent=0, detail={})
