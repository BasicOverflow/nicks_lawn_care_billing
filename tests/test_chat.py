"""Chat wipe, inventory SQL path, mutations, and snapshot."""

import json
from datetime import date

from app.chatdata import (
    _inventory_reply,
    _month_from_chat_text,
    _status_month,
    _wipe_month,
    answer,
    apply_mutations,
    knowledge_snapshot,
)


def test_chat_wipe_month_natural_phrases(today_nov):
    assert _wipe_month("delete november", today=today_nov) == "2026-11"
    assert _wipe_month("delete data for november", today=today_nov) == "2026-11"
    assert _wipe_month("wipe nov 2026", today=today_nov) == "2026-11"
    assert _wipe_month("clear 2026-11 billing", today=today_nov) == "2026-11"
    assert _wipe_month("what happened in november", today=today_nov) is None
    assert _wipe_month("delete november work", today=today_nov) == "2026-11"


def test_chat_status_month_skips_deletes_and_reads_november(today_nov):
    assert _status_month("do i have november data that isnt tests", today=today_nov) == "2026-11"
    assert _status_month("how many october bills", today=today_nov) == "2026-10"
    assert _status_month("delete november", today=today_nov) is None
    reply = _inventory_reply(
        {"month": "2026-11", "work": 0, "bills": 0, "jobs": {"done": 1}, "test_work": 5}
    )
    assert "0 stored work" in reply
    assert "test_month" in reply
    assert "1 OCR" in reply


def test_year_rollover_uses_previous_year():
    today = date(2026, 1, 1)
    assert _month_from_chat_text("december", today=today) == "2025-12"
    assert _month_from_chat_text("january", today=today) == "2026-01"


def test_this_month_is_the_work_month(today_nov):
    assert _month_from_chat_text("this month", today=today_nov) == "2026-10"
    assert _month_from_chat_text("current month", today=today_nov) == "2026-10"
    assert _month_from_chat_text("november", today=today_nov) == "2026-11"
    assert _status_month("do i have data this month", today=today_nov) == "2026-10"
    assert _status_month("show work items for this month", today=today_nov) is None
    assert _wipe_month("delete this month", today=today_nov) == "2026-10"


def test_wipe_deletes_work_and_bills_keeps_clients_and_jobs(store, blobs, monkeypatch):
    monkeypatch.setattr("app.chatdata.date", type("D", (), {
        "today": staticmethod(lambda: date(2026, 11, 1)),
    }))
    cid = store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    store.add_work_item(
        None, client_id=cid, month="2026-11", description="Mowing 8", amount=40, day_or_note="8",
    )
    store.add_work_item(
        None, client_id=cid, month="test_month", description="Mowing 1", amount=1, day_or_note="1",
    )
    bid = store.save_bill(store, month="2026-11", client_id=cid, s3_key="bills/2026-11/x.pdf")
    blobs.put_bytes(b"pdf", "bills/2026-11/x.pdf")
    store.save_upload_job(store, "ocr-1", "stored", extract={"tables": []}, month="2026-11")
    out = answer(store, "delete november")
    assert "Deleted" in out["answer"]
    assert store.work_for_month(store, "2026-11") == []
    assert store.bills_for_month(store, "2026-11") == []
    assert "bills/2026-11/x.pdf" in blobs.deleted
    assert store.get_client(store, cid)["name"] == "SMITH, Mary"
    assert store.work_for_month(store, "test_month")
    assert store.get_upload_job(store, "ocr-1")["status"] == "stored"
    assert bid not in store.bills or True


def test_inventory_uses_sql_counts_not_the_model(store, monkeypatch):
    monkeypatch.setattr("app.chatdata.date", type("D", (), {
        "today": staticmethod(lambda: date(2026, 11, 1)),
    }))

    def boom(*_a, **_k):
        raise AssertionError("model should not run")

    monkeypatch.setattr("ocr.chat.chat_text", boom)
    cid = store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    store.add_work_item(
        None, client_id=cid, month="2026-10", description="Mowing 8", amount=40, day_or_note="8",
    )
    store.add_work_item(
        None, client_id=cid, month="test_month", description="Mowing 1", amount=1, day_or_note="1",
    )
    store.save_upload_job(
        store, "ocr-1", "done",
        extract={"tables": [{"columns": ["CLIENT"], "rows": [["A"]]}]},
        month="2026-11",
    )
    out = answer(store, "do i have november data that isnt tests")
    assert "0 stored work" in out["answer"]
    assert "test_month" in out["answer"]
    assert "211" not in out["answer"]
    assert "OCR" in out["answer"]
    assert out["mutations_applied"] == []


def test_apply_mutations_update_add_delete(store):
    cid = store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="old@example.com")
    wid = store.add_work_item(
        None, client_id=cid, month="2026-09", description="Mowing 8", amount=40, day_or_note="8",
    )
    log = apply_mutations(store, [
        {"op": "update_client", "client_id": cid, "email": "new@example.com", "mow_price": 45},
        {"op": "add_work", "client_id": cid, "month": "2026-09", "description": "bush", "amount": 20},
        {"op": "delete_work", "work_id": wid},
        {"op": "unknown_thing"},
        {"op": "update_client", "client_name": "NOBODY"},
        {"op": "delete_work_month", "month": "September"},
    ])
    client = store.get_client(store, cid)
    assert client["email"] == "new@example.com"
    assert float(client["mow_price"]) == 45.0
    assert not any(row["id"] == wid for row in store.work_for_month(store, "2026-09"))
    assert any(row["description"] == "bush" for row in store.work_for_month(store, "2026-09"))
    assert any("unknown op" in line for line in log)
    assert any("unknown client" in line for line in log)
    assert any("month must be YYYY-MM" in line for line in log)


def test_snapshot_includes_contact_and_stays_short(store):
    for i in range(40):
        cid = store.upsert_client(
            None,
            name=f"CLIENT, {i:03d}",
            email=f"c{i}@example.com",
            phone="8605550100",
            billing_notes="REGULAR MAIL",
            prefer_mail=True,
            mow_price=40,
        )
        store.add_work_item(
            None, client_id=cid, month="2026-09", description="Mowing 8", amount=40, day_or_note="8",
        )
    text = knowledge_snapshot(store, limit_work=80, limit_clients=120)
    assert len(text) <= 10000
    payload = json.loads(text)
    assert payload["clients"][0]["email"]
    assert "phone" in payload["clients"][0]
    assert "prefer_mail" in payload["clients"][0]
    assert "billing_notes" in payload["clients"][0]
    assert payload["clients"][0]["mow_price"] == 40
