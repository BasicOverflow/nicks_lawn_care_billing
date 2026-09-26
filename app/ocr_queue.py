"""Sequential OCR upload queue — accept new batches while one runs."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class QueuedOcr:
    job_id: str
    paths: list[Path]
    month: str
    sheet_kind: str
    status: str = "queued"  # queued | running | done | error | cancelled


_lock = threading.Lock()
_queue: list[QueuedOcr] = []
_active: QueuedOcr | None = None
_worker_busy = False


def snapshot() -> dict[str, Any]:
    with _lock:
        return {
            "active_job_id": _active.job_id if _active else None,
            "queue_depth": len(_queue),
            "queued": [
                {
                    "job_id": q.job_id,
                    "month": q.month,
                    "sheet_kind": q.sheet_kind,
                    "n_files": len(q.paths),
                    "status": q.status,
                }
                for q in _queue
            ],
            "busy": _worker_busy or _active is not None,
        }


def enqueue(
    *,
    paths: list[Path],
    month: str,
    sheet_kind: str,
    job_id: str | None = None,
) -> tuple[str, str]:
    """Add OCR work. Returns (job_id, 'running'|'queued')."""
    global _worker_busy, _active
    jid = job_id or uuid.uuid4().hex[:12]
    item = QueuedOcr(job_id=jid, paths=list(paths), month=month, sheet_kind=sheet_kind)
    start_now = False
    with _lock:
        if _active is None and not _worker_busy and not _queue:
            item.status = "running"
            _active = item
            _worker_busy = True
            start_now = True
        else:
            item.status = "queued"
            _queue.append(item)
    if start_now:
        return jid, "running"
    return jid, "queued"


def set_worker(fn: Callable[[QueuedOcr], None]) -> None:
    global _worker_fn
    _worker_fn = fn


_worker_fn: Callable[[QueuedOcr], None] | None = None


def kick(worker: Callable[[QueuedOcr], None] | None = None) -> None:
    """Ensure a worker is running for the active job (if any)."""
    global _worker_fn
    if worker is not None:
        _worker_fn = worker
    with _lock:
        item = _active
        fn = _worker_fn
    if item is not None and fn is not None:
        threading.Thread(target=_run_safe, args=(item, fn), daemon=True).start()


def _run_safe(item: QueuedOcr, worker: Callable[[QueuedOcr], None]) -> None:
    global _active, _worker_busy, _worker_fn
    _worker_fn = worker
    try:
        worker(item)
    finally:
        next_item = None
        fn = worker
        with _lock:
            if _active and _active.job_id == item.job_id:
                _active = None
            _worker_busy = False
            if _queue:
                next_item = _queue.pop(0)
                next_item.status = "running"
                _active = next_item
                _worker_busy = True
            fn = _worker_fn or worker
        if next_item is not None and fn is not None:
            threading.Thread(target=_run_safe, args=(next_item, fn), daemon=True).start()


def cancel_active(job_id: str | None = None) -> dict[str, Any]:
    """Signal cancel for the running OCR job (queue kept)."""
    from ocr.cancel import abort_inflight, request_cancel

    with _lock:
        active_id = _active.job_id if _active else None
    target = job_id or active_id
    if not target:
        n = abort_inflight()
        return {"signaled": False, "aborted_http": n, "cleared_queue": 0}
    signaled = request_cancel(target)
    n = abort_inflight()
    return {"signaled": signaled, "job_id": target, "aborted_http": n, "cleared_queue": 0}


def cancel_all() -> dict[str, Any]:
    """Cancel active OCR and drop the queue."""
    from ocr.cancel import abort_inflight, request_cancel

    cleared = 0
    with _lock:
        cleared = len(_queue)
        for q in _queue:
            q.status = "cancelled"
        _queue.clear()
        active_id = _active.job_id if _active else None
    signaled = False
    if active_id:
        signaled = request_cancel(active_id)
    n = abort_inflight()
    return {
        "signaled": signaled,
        "job_id": active_id,
        "aborted_http": n,
        "cleared_queue": cleared,
    }


def remove_queued(job_id: str) -> bool:
    with _lock:
        for i, q in enumerate(_queue):
            if q.job_id == job_id:
                _queue.pop(i)
                return True
    return False
