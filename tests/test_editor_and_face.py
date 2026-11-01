"""Bill editor dates, hand-edited prices, and month-only face overrides."""

from app.billing import (
    _editable_bill_lines,
    _reprice_visits,
    _work_items_from_editor_line,
    apply_bill_face,
    compile_invoice_lines,
    delivery_channel,
    face_bill,
    generate_month_bills,
    stored_letter,
    default_greeting,
)


def test_reprice_visits_keeps_a_hand_edited_mow_amount():
    lines = [
        {
            "description": "Mowing 8",
            "amount": 28,
            "day_or_note": "8",
            "mow_price": 32,
        },
    ]
    out = _reprice_visits(None, lines)
    assert float(out[0]["amount"]) == 28.0


def test_reprice_fills_only_remaining_zero_amounts(store):
    cid = store.upsert_client(None, name="SMITH, Mary", mow_price=32, email="m@example.com")
    zero = store.add_work_item(
        None, client_id=cid, month="2026-09", description="Mowing 8", amount=0, day_or_note="8",
    )
    edited = store.add_work_item(
        None, client_id=cid, month="2026-09", description="Mowing 15", amount=28, day_or_note="15",
    )
    lines = store.work_for_month(store, "2026-09")
    out = _reprice_visits(store, lines)
    by_id = {int(row["id"]): row for row in out}
    assert float(by_id[zero]["amount"]) == 32.0
    assert float(by_id[edited]["amount"]) == 28.0


def test_editor_accepts_multiple_dates_on_one_line():
    stored = [
        {"description": "Mowing 8", "amount": 50, "day_or_note": "8"},
        {"description": "Mowing 15", "amount": 50, "day_or_note": "15"},
        {"description": "Mowing 22", "amount": 50, "day_or_note": "22"},
    ]
    editor = _editable_bill_lines(stored, "2026-09")
    assert len(editor) == 1
    assert editor[0]["description"] == "Mowing"
    assert editor[0]["date"] == "9/8, 9/15, 9/22"

    items = _work_items_from_editor_line({
        "description": "Mowing",
        "amount": 150,
        "date": "9/4, 9/12, 9/21",
        "kind": "visit",
    }, "2026-09")
    assert len(items) == 3
    assert [item["day_or_note"] for item in items] == ["4", "12", "21"]
    rows = compile_invoice_lines(items, "2026-09")
    mowing = next(row for row in rows if row["description"] == "Mowing")
    assert mowing["date"] == "9/4, 9/12, 9/21"
    assert float(mowing["amount"]) == 150.0

    past = _work_items_from_editor_line({
        "description": "Mowing",
        "amount": 120,
        "date": "8/4, 8/12, 8/21",
        "kind": "visit",
    }, "2026-09")
    assert len(past) == 3
    assert all(item["day_or_note"].startswith("prior:2026-08:") for item in past)
    past_rows = compile_invoice_lines(past, "2026-09")
    past_mow = next(row for row in past_rows if row["description"] == "Mowing")
    assert past_mow["date"] == "8/4, 8/12, 8/21"

    job = _work_items_from_editor_line({
        "description": "Bush trimming",
        "amount": 50,
        "date": "8/12",
        "kind": "visit",
    }, "2026-09")
    assert job[0]["day_or_note"] == "prior:2026-08:12"
    job_row = next(
        row for row in compile_invoice_lines(job, "2026-09")
        if row["description"] == "Bush trimming"
    )
    assert job_row["date"] == "8/12"


def test_bill_face_uses_this_months_name_email_and_delivery():
    row = face_bill({
        "client_name": "SMITH, Mary",
        "email": "mary@example.com",
        "delivery": "email",
        "display_name": "SMITH, Marie",
        "bill_email": "",
        "bill_delivery": "mail",
        "prefer_mail": False,
    })
    assert row["client_name"] == "SMITH, Marie"
    assert row["email"] == ""
    assert delivery_channel(row) == "mail"
    plain = face_bill({
        "client_name": "SMITH, Mary",
        "email": "mary@example.com",
        "phone": "",
        "prefer_mail": False,
        "bill_email": None,
        "bill_delivery": None,
    })
    assert plain["client_name"] == "SMITH, Mary"
    assert delivery_channel(plain) == "email"


def test_apply_bill_face_month_only_vs_roster(store, blobs):
    cid = store.upsert_client(
        None, name="SMITH, Mary", email="mary@example.com", mow_price=40, address="1 Main",
    )
    store.add_work_item(
        None, client_id=cid, month="2026-09", description="Mowing 8", amount=40, day_or_note="8",
    )
    generate_month_bills(store, "2026-09")
    bill = store.bill_for_client_month(store, "2026-09", cid)
    store.bills[bill["id"]]["emailed_at"] = store.now()
    apply_bill_face(
        store, "2026-09", cid,
        name="SMITH, Marie", email="", delivery="mail", save_to_client=False,
    )
    client = store.get_client(store, cid)
    assert client["name"] == "SMITH, Mary"
    assert client["email"] == "mary@example.com"
    faced = store.bills_for_month(store, "2026-09")[0]
    assert faced["client_name"] == "SMITH, Marie"
    assert delivery_channel(faced) == "mail"
    assert faced["emailed_at"] is None
    apply_bill_face(
        store, "2026-09", cid,
        name="SMITH, Marie", email="marie@example.com", delivery="email", save_to_client=True,
    )
    client = store.get_client(store, cid)
    assert client["name"] == "SMITH, Marie"
    assert client["email"] == "marie@example.com"


def test_stored_letter_equal_to_default_is_none():
    assert stored_letter("Dear Mary,", default_greeting("SMITH, Mary")) is None
