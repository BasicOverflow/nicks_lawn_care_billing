"""Global in-memory job / progress (single-user)."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Progress:
    job_id: str = ""
    kind: str = ""  # upload | ocr | generate | email | model
    status: str = "idle"  # idle | running | done | error | cancelled
    message: str = ""
    percent: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


_lock = threading.Lock()
_state = Progress()


def get() -> dict:
    from . import ocr_queue

    q = ocr_queue.snapshot()
    with _lock:
        return {
            "job_id": _state.job_id,
            "kind": _state.kind,
            "status": _state.status,
            "message": _state.message,
            "percent": _state.percent,
            "detail": dict(_state.detail),
            "queue_depth": q["queue_depth"],
            "queued": q["queued"],
            "ocr_busy": q["busy"],
        }


def set_progress(
    *,
    kind: str | None = None,
    status: str | None = None,
    message: str | None = None,
    percent: int | None = None,
    job_id: str | None = None,
    detail: dict | None = None,
) -> None:
    with _lock:
        if job_id is not None:
            _state.job_id = job_id
        if kind is not None:
            _state.kind = kind
        if status is not None:
            _state.status = status
        if message is not None:
            _state.message = message
        if percent is not None:
            _state.percent = max(0, min(100, int(percent)))
        if detail is not None:
            _state.detail = detail


def new_job(kind: str, message: str = "") -> str:
    jid = uuid.uuid4().hex[:12]
    set_progress(job_id=jid, kind=kind, status="running", message=message, percent=0, detail={})
    return jid


def new_job_replace(job_id: str, kind: str, message: str = "") -> str:
    """Reuse an existing job id (queued OCR becoming active)."""
    set_progress(job_id=job_id, kind=kind, status="running", message=message, percent=0, detail={})
    return job_id


def done(message: str = "Done", detail: dict | None = None) -> None:
    set_progress(status="done", message=message, percent=100, detail=detail)


def fail(message: str) -> None:
    set_progress(status="error", message=message, percent=100)


def cancel(message: str = "Cancelled") -> None:
    set_progress(status="cancelled", message=message, percent=0, detail={})
