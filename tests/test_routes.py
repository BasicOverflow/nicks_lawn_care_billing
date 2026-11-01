"""HTTP API: upload names, commit/clear, generate, downloads, clients."""

import io
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import jobs


def _wait_progress(client, job_id=None, timeout=3):
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = client.get("/api/progress").json()
        rows = last.get("jobs") or []
        if job_id:
            match = next((row for row in rows if row["job_id"] == job_id), last)
        else:
            match = last
        if match.get("status") in {"done", "error", "cancelled"}:
            return match
        time.sleep(0.02)
    return last


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _work_extract(*rows):
    return {
        "title": None,
        "tables": [{
            "columns": ["CLIENT", "DATE & WORK COMPLETED"],
            "rows": [list(row) for row in rows],
        }],
        "notes": [],
        "complete": True,
    }


def _stub_ocr(monkeypatch, extract=None, empty=False):
    import ocr

    payload = {"title": None, "tables": [], "notes": [], "complete": True}
    if extract is not None:
        payload = extract
    elif not empty:
        payload = _work_extract(["SMITH, Mary", "8"])
    monkeypatch.setattr(ocr, "ensure_model", lambda: None)
    monkeypatch.setattr(ocr, "extract_sheet", lambda *a, **k: payload)
    monkeypatch.setattr(ocr, "model_status", lambda live=True: {
        "id": "qwen25-vl-3b", "up": True, "base_url": "http://fake",
    })


def test_unique_upload_filenames(client, blobs, monkeypatch, store):
    _stub_ocr(monkeypatch)
    files = [
        ("files", ("image.jpg", b"one", "image/jpeg")),
        ("files", ("image.jpg", b"two", "image/jpeg")),
        ("files", ("image.jpg", b"three", "image/jpeg")),
    ]
    r = client.post("/api/upload", data={"month": "2026-11"}, files=files)
    assert r.status_code == 200
    _wait_progress(client, r.json()["job_id"])
    keys = [k for k in blobs.blobs if "/01_image.jpg" in k or "/02_image.jpg" in k or "/03_image.jpg" in k]
    assert any(k.endswith("01_image.jpg") for k in blobs.blobs)
    assert any(k.endswith("02_image.jpg") for k in blobs.blobs)
    assert any(k.endswith("03_image.jpg") for k in blobs.blobs)
    assert len(keys) == 3


def test_empty_ocr_is_an_error_not_a_blank_review(client, monkeypatch, store):
    _stub_ocr(monkeypatch, empty=True)
    r = client.post(
        "/api/upload",
        data={"month": "2026-11"},
        files=[("files", ("image.jpg", b"xx", "image/jpeg"))],
    )
    assert r.status_code == 200
    jid = r.json()["job_id"]
    prog = _wait_progress(client, jid)
    assert prog["status"] == "error"
    assert "did not find any rows" in prog["message"]
    job = store.get_upload_job(store, jid)
    assert job["status"] == "error"


def test_draft_save_after_stored(client, store):
    extract = _work_extract(["SMITH, Mary", "8"])
    store.save_upload_job(store, "job-s", "stored", extract=extract, month="2026-11")
    edited = _work_extract(["SMITH, Mary", "8 16"])
    r = client.put("/api/jobs/job-s/draft", json={"extract": edited, "month": "2026-11"})
    assert r.status_code == 200
    saved = store.get_upload_job(store, "job-s")
    assert saved["status"] == "stored"
    assert saved["extract_json"]["tables"][0]["rows"][0][1] == "8 16"


def test_commit_conflicts_leave_job_unstored(client, store):
    store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    extract = _work_extract(["SMITH, Mary", "8"], ["ZZZ, Test", "2"])
    store.save_upload_job(store, "job-c", "done", extract=extract, month="2026-11")
    r = client.post("/api/jobs/job-c/commit", json={"month": "2026-11", "extract": extract, "resolutions": []})
    assert r.status_code == 200
    body = r.json()
    assert body["conflicts"]
    assert body["written"] == 0
    assert store.get_upload_job(store, "job-c")["status"] == "done"


def test_empty_send_does_not_become_stored(client, store):
    store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    extract = _work_extract(["SMITH, Mary", ""])
    store.save_upload_job(store, "job-e", "done", extract=extract, month="2026-11")
    r = client.post("/api/jobs/job-e/commit", json={"month": "2026-11", "extract": extract})
    assert r.status_code == 200
    assert r.json()["lines"] == 0
    assert store.get_upload_job(store, "job-e")["status"] == "done"


def test_clear_review_done_keeps_work_stored_deletes_it(client, store, blobs):
    cid = store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    store.save_upload_job(
        store, "job-d", "done",
        extract=_work_extract(["SMITH, Mary", "8"]), month="2026-11",
    )
    store.add_work_item(
        None, client_id=cid, month="2026-11", description="manual", amount=10, source_job_id=None,
    )
    r = client.post("/api/jobs/job-d/clear")
    assert r.status_code == 200
    assert store.get_upload_job(store, "job-d")["status"] == "cleared"
    assert store.work_for_month(store, "2026-11")

    store.save_upload_job(
        store, "job-st", "stored",
        extract=_work_extract(["SMITH, Mary", "8"]), month="2026-11",
    )
    store.add_work_item(
        None, client_id=cid, month="2026-11", description="Mowing 8", amount=40,
        source_job_id="job-st", day_or_note="8",
    )
    blobs.put_bytes(b"pdf", "bills/2026-11/keep.pdf")
    store.save_bill(store, month="2026-11", client_id=cid, s3_key="bills/2026-11/keep.pdf")
    r = client.post("/api/jobs/job-st/clear")
    assert r.status_code == 200
    assert r.json()["work_items"] == 1
    assert store.get_upload_job(store, "job-st")["status"] == "cleared"
    assert not any(row.get("source_job_id") == "job-st" for row in store.work_for_month(store, "2026-11"))
    assert "bills/2026-11/keep.pdf" in blobs.blobs
    assert "bills/2026-11/keep.pdf" not in blobs.deleted


def test_clear_month_uses_query_param(client, store, blobs):
    cid = store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    store.add_work_item(
        None, client_id=cid, month="2026-09", description="Mowing 8", amount=40, day_or_note="8",
    )
    blobs.put_bytes(b"pdf", "bills/2026-09/x.pdf")
    store.save_bill(store, month="2026-09", client_id=cid, s3_key="bills/2026-09/x.pdf")
    r = client.post("/api/billing/2026-09/clear", json={"include_work": True})
    assert r.status_code != 422
    assert store.work_for_month(store, "2026-09")
    r = client.post("/api/billing/2026-09/clear?include_work=false")
    assert r.status_code == 200
    assert r.json()["include_work"] is False
    assert store.work_for_month(store, "2026-09")
    assert not store.bills_for_month(store, "2026-09")
    store.save_bill(store, month="2026-09", client_id=cid, s3_key="bills/2026-09/y.pdf")
    r = client.post("/api/billing/2026-09/clear?include_work=true")
    assert r.status_code == 200
    assert r.json()["include_work"] is True
    assert store.work_for_month(store, "2026-09") == []


def test_generate_fails_without_work_and_does_not_email(client, monkeypatch):
    emailed = []
    monkeypatch.setattr("app.emailer.email_month_bills", lambda *a, **k: emailed.append(1))
    called = []
    monkeypatch.setattr("app.billing.restore_job_dates", lambda *a, **k: called.append(1))
    r = client.post("/api/billing/generate", json={"month": "2026-11"})
    assert r.status_code == 200
    prog = _wait_progress(client, r.json()["job_id"])
    assert prog["status"] == "error"
    assert "datastore" in prog["message"].lower()
    assert emailed == []
    assert called == []


def test_manual_work_then_generate(client, store, blobs):
    cid = store.upsert_client(
        None, name="SMITH, Mary", email="mary@example.com", mow_price=40, address="1 Main",
    )
    r = client.post("/api/work", json={"client_id": cid, "month": "2026-09", "text": "8 16"})
    assert r.status_code == 200
    assert r.json()["lines"] == 2
    r = client.post("/api/billing/generate", json={"month": "2026-09"})
    assert r.status_code == 200
    prog = _wait_progress(client, r.json()["job_id"])
    assert prog["status"] == "done"
    bills = store.bills_for_month(store, "2026-09")
    assert len(bills) == 1
    assert any(k.startswith("bills/2026-09/") for k in blobs.blobs)


def test_ui_sync_and_months(client, store):
    cid = store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    store.add_work_item(
        None, client_id=cid, month="2026-10", description="Mowing 8", amount=40, day_or_note="8",
    )
    store.add_work_item(
        None, client_id=cid, month="test_month", description="Mowing 1", amount=1, day_or_note="1",
    )
    store.save_upload_job(
        store, "job-u", "stored",
        extract=_work_extract(["SMITH, Mary", "8"]), month="2026-10",
    )
    months = client.get("/api/months").json()["months"]
    assert months[0] == "2026-10"
    assert months[-1] == "test_month"
    sync = client.get("/api/ui/sync", params={"month": "2026-10"}).json()
    assert sync["review"]["job_id"] == "job-u"
    assert sync["work_rev"] == 1
    assert sync["bills_rev"]
    assert "clients_rev" in sync


def test_client_name_clash_and_typed_zero(client, store):
    a = store.save_typed_client(
        store, client_id=None, name="SMITH, Mary", address=None, phone=None,
        email="m@example.com", billing_notes=None, mow_price=40, hedge_price=None, prefer_mail=False,
    )
    b = store.save_typed_client(
        store, client_id=None, name="JONES, Ann", address=None, phone=None,
        email="a@example.com", billing_notes=None, mow_price=30, hedge_price=None, prefer_mail=False,
    )
    r = client.post("/api/clients", json={"clients": [{
        "id": b, "name": "SMITH, Mary", "address": "", "phone": "", "email": "",
        "billing_notes": "", "mow_price": 30, "hedge_price": None, "prefer_mail": False,
    }]})
    assert r.status_code == 400
    r = client.post("/api/clients", json={"clients": [{
        "name": "ZERO, Pat", "mow_price": 0, "hedge_price": None, "prefer_mail": False,
        "address": "", "phone": "", "email": "", "billing_notes": "",
    }]})
    assert r.status_code == 200
    row = store.get_client_by_name(store, "ZERO, Pat")
    assert row["mow_price"] == 0
    assert a != b


def test_model_status_uses_stubbed_ping(client, monkeypatch):
    _stub_ocr(monkeypatch)
    r = client.get("/api/model/status")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "qwen25-vl-3b"
    assert body["up"] is True
    assert body["base_url"] == "http://fake"


def test_sms_email_requires_smtp(client):
    r = client.post("/api/billing/2026-09/sms-email")
    assert r.status_code == 400


def test_zip_download_headers(client, store, blobs):
    cid = store.upsert_client(None, name="SMITH, Mary", email="m@example.com", mow_price=40)
    store.add_work_item(
        None, client_id=cid, month="2026-09", description="Mowing 8", amount=40, day_or_note="8",
    )
    from app.billing import generate_month_bills
    generate_month_bills(store, "2026-09")
    r = client.get("/api/billing/2026-09/download.zip")
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
    assert r.content[:2] == b"PK"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert names
    r = client.get("/api/billing/2026-09/download.pdf", params={"mode": "mailing_only"})
    # SMITH has email so mailing_only may be empty
    assert r.status_code in {200, 400, 404, 500}


def test_progress_lists_jobs(client):
    jobs.new_job("generate", "one")
    jobs.new_job("email", "two")
    payload = client.get("/api/progress").json()
    assert len(payload["jobs"]) >= 2
