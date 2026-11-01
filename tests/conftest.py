"""Offline fixtures: fake store, fake S3, no live Postgres/SMTP/GPU."""

from __future__ import annotations

import pytest

from app import billing, chatdata, db, emailer, jobs, ocr_queue, storage
from ocr import cancel as ocr_cancel
from tests.fakes import DB_FUNCS, FakeStorage, FakeStore


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch):
    store = FakeStore()
    blobs = FakeStorage()

    monkeypatch.setattr(db, "init_db", store.init_db)
    monkeypatch.setattr(db, "connect", store.connect)
    for name in DB_FUNCS:
        if name == "connect":
            continue
        monkeypatch.setattr(db, name, getattr(store, name))

    monkeypatch.setattr(storage, "put_bytes", blobs.put_bytes)
    monkeypatch.setattr(storage, "get_bytes", blobs.get_bytes)
    monkeypatch.setattr(storage, "delete_key", blobs.delete_key)
    monkeypatch.setattr(storage, "ensure_bucket", blobs.ensure_bucket)
    monkeypatch.setattr(billing, "storage", storage)
    monkeypatch.setattr(emailer, "storage", storage)
    monkeypatch.setattr(chatdata, "storage", storage)

    monkeypatch.setattr("app.config.SMTP_HOST", "")
    monkeypatch.setattr("app.config.SMTP_FROM", "")
    monkeypatch.setattr("app.config.SMTP_USER", "")
    monkeypatch.setattr("app.config.SMTP_PASSWORD", "")

    def _offline_extract(*_a, **_k):
        return {
            "title": None,
            "tables": [{
                "columns": ["CLIENT", "DATE & WORK COMPLETED"],
                "rows": [["SMITH, Mary", "8"]],
            }],
            "notes": [],
            "complete": True,
        }

    monkeypatch.setattr("ocr.ensure_model", lambda: None)
    monkeypatch.setattr("ocr.extract_sheet", _offline_extract)
    monkeypatch.setattr("ocr.unload_model", lambda: None)
    monkeypatch.setattr("ocr.model_status", lambda live=True: {
        "id": "qwen25-vl-3b", "up": True, "base_url": "http://fake",
    })
    monkeypatch.setattr("ocr.chat.chat_text", lambda *_a, **_k: '{"reply": "ok", "mutations": []}')

    with jobs._lock:
        jobs._jobs.clear()
    jobs._local.job_id = None
    with ocr_queue._lock:
        ocr_queue._queue.clear()
        ocr_queue._active = None
        ocr_queue._worker_busy = False
    ocr_queue._worker_fn = None
    ocr_cancel.clear_job()

    yield store, blobs

    with jobs._lock:
        jobs._jobs.clear()
    with ocr_queue._lock:
        ocr_queue._queue.clear()
        ocr_queue._active = None
        ocr_queue._worker_busy = False
    ocr_cancel.clear_job()


@pytest.fixture
def store(isolate_runtime):
    return isolate_runtime[0]


@pytest.fixture
def blobs(isolate_runtime):
    return isolate_runtime[1]


@pytest.fixture
def today_nov():
    from datetime import date
    return date(2026, 11, 1)
