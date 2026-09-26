"""Cancel flags + abortable inference HTTP for OCR / chat."""

from __future__ import annotations

import json
import threading
import uuid
from typing import Any


class InferenceCancelled(Exception):
    """Raised when the user cancels an in-flight OCR / inference job."""


_lock = threading.Lock()
_cancel_job_id: str | None = None
_cancel_requested = False
# Active HTTP responses that should be closed on cancel (stream readers).
_active_responses: list[Any] = []


def begin_job(job_id: str) -> None:
    """Bind cancel scope to this job id (clears prior cancel)."""
    global _cancel_job_id, _cancel_requested
    with _lock:
        _cancel_job_id = job_id
        _cancel_requested = False


def clear_job(job_id: str | None = None) -> None:
    global _cancel_job_id, _cancel_requested
    with _lock:
        if job_id is None or _cancel_job_id == job_id:
            _cancel_job_id = None
            _cancel_requested = False


def request_cancel(job_id: str | None = None) -> bool:
    """Mark cancel + close in-flight HTTP so vLLM aborts on disconnect.

    Returns True if a matching running job was signaled.
    """
    global _cancel_requested
    with _lock:
        if job_id is not None and _cancel_job_id and job_id != _cancel_job_id:
            return False
        if not _cancel_job_id and job_id is None:
            # Still abort any orphan HTTP
            _cancel_requested = True
            _close_active_locked()
            return True
        if not _cancel_job_id:
            return False
        _cancel_requested = True
        _close_active_locked()
        return True


def is_cancelled() -> bool:
    with _lock:
        return _cancel_requested


def raise_if_cancelled() -> None:
    if is_cancelled():
        raise InferenceCancelled("OCR cancelled by user")


def register_response(resp: Any) -> None:
    with _lock:
        _active_responses.append(resp)


def unregister_response(resp: Any) -> None:
    with _lock:
        try:
            _active_responses.remove(resp)
        except ValueError:
            pass


def _close_active_locked() -> None:
    for resp in list(_active_responses):
        try:
            resp.close()
        except Exception:
            pass
    _active_responses.clear()


def abort_inflight() -> int:
    """Force-close active inference HTTP connections. Returns count closed."""
    with _lock:
        n = len(_active_responses)
        _close_active_locked()
        return n


def new_request_id() -> str:
    return uuid.uuid4().hex
