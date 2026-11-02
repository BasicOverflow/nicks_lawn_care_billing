"""confirm_extract replace-on-resend, conflicts, and store_work_text."""

from app.knowledge import confirm_extract, pick_editable_job, rebuild_month_from_review, store_work_text


def _sheet(*rows, columns=None):
    return {
        "tables": [{
            "columns": columns or ["CLIENT", "DATE & WORK COMPLETED"],
            "rows": [list(row) for row in rows],
        }],
    }


def test_work_column_only_uses_roster_prices_not_sheet_price(store):
    cid = store.upsert_client(
        None, name="BOB TEST", mow_price=40, hedge_price=25, email="bob@example.com",
    )
    extract = _sheet(
        ["BOB TEST", "1 Main St", "$75", "8 16h | bush trim 50"],
        columns=["CLIENT", "Address", "Price", "DATE & WORK COMPLETED"],
    )
    result = confirm_extract(store, extract, month="2026-09", source_job_id="job-a")
    assert result["conflicts"] == []
    assert result["lines"] == 3
    work = store.work_for_month(store, "2026-09")
    kinds = {(row["description"], float(row["amount"] or 0), row["day_or_note"]) for row in work}
    assert ("Mowing 8", 40.0, "8") in kinds
    assert ("Hedging 16", 25.0, "16h") in kinds
    assert ("bush trim", 50.0, None) in kinds
    assert all(row["amount"] != 75 for row in work)
    assert cid == work[0]["client_id"]


def test_blank_work_writes_no_lines(store):
    store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    result = confirm_extract(
        store,
        _sheet(["SMITH, Mary", ""]),
        month="2026-09",
        source_job_id="job-a",
    )
    assert result["lines"] == 0
    assert store.work_for_month(store, "2026-09") == []


def test_conflicts_do_not_delete_existing_work(store):
    cid = store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    store.add_work_item(
        None, client_id=cid, month="2026-09", description="Mowing 8", amount=40,
        source_job_id="job-a", day_or_note="8",
    )
    result = confirm_extract(
        store,
        _sheet(["SMITH, Mary", "9"], ["ZZZ, Test", "2"]),
        month="2026-09",
        source_job_id="job-a",
    )
    assert result["written"] == 0
    assert result["conflicts"]
    assert [row["description"] for row in store.work_for_month(store, "2026-09")] == ["Mowing 8"]


def test_resend_same_job_replaces_other_job_adds(store):
    store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    confirm_extract(
        store, _sheet(["SMITH, Mary", "8 16"]), month="2026-09", source_job_id="job-a",
    )
    confirm_extract(
        store, _sheet(["SMITH, Mary", "9"]), month="2026-09", source_job_id="job-a",
    )
    days = sorted(row["day_or_note"] for row in store.work_for_month(store, "2026-09"))
    assert days == ["9"]
    confirm_extract(
        store, _sheet(["SMITH, Mary", "22"]), month="2026-09", source_job_id="job-b",
    )
    days = sorted(row["day_or_note"] for row in store.work_for_month(store, "2026-09"))
    assert days == ["22"]


def test_manual_work_survives_sheet_resend(store):
    cid = store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    confirm_extract(
        store, _sheet(["SMITH, Mary", "8"]), month="2026-09", source_job_id="job-a",
    )
    store_work_text(store, client_id=cid, month="2026-09", text="bush 50")
    confirm_extract(
        store, _sheet(["SMITH, Mary", "16"]), month="2026-09", source_job_id="job-a",
    )
    descs = {row["description"] for row in store.work_for_month(store, "2026-09")}
    assert "bush" in descs
    assert "Mowing 16" in descs
    assert "Mowing 8" not in descs


def test_resend_different_month_leaves_old_month(store):
    store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    confirm_extract(
        store, _sheet(["SMITH, Mary", "8"]), month="2026-09", source_job_id="job-a",
    )
    confirm_extract(
        store, _sheet(["SMITH, Mary", "2"]), month="2026-10", source_job_id="job-a",
    )
    assert [row["day_or_note"] for row in store.work_for_month(store, "2026-09")] == ["8"]
    assert [row["day_or_note"] for row in store.work_for_month(store, "2026-10")] == ["2"]


def test_rebuild_includes_one_off_client_not_a_similar_roster_name(store):
    store.upsert_client(
        None, name="BRADY, Jib & Susan", mow_price=23, email="jib@example.com",
    )
    sheet = _sheet(["BRADY, Jim", "5 13 20 28"], ["FITZNER, Bill", "15h"])
    first = confirm_extract(
        store,
        sheet,
        month="2026-10",
        source_job_id="job-a",
        resolutions=[
            {"name": "BRADY, Jim", "add_permanently": False},
            {"name": "FITZNER, Bill", "add_permanently": False},
        ],
    )
    assert first["conflicts"] == []
    store.save_upload_job(
        store, "job-a", "stored", "Stored", extract=sheet, month="2026-10",
    )
    rebuilt = rebuild_month_from_review(store, "2026-10")
    assert rebuilt["conflicts"] == []
    by_name = {}
    for row in store.work_for_month(store, "2026-10"):
        by_name.setdefault(row["client_name"], []).append(row["day_or_note"])
    assert sorted(by_name["BRADY, Jim"]) == ["13", "20", "28", "5"]
    assert "BRADY, Jib & Susan" not in by_name
    assert by_name["FITZNER, Bill"] == ["15h"]


def test_one_off_client_is_not_on_roster(store):
    result = confirm_extract(
        store,
        _sheet(["NEW, Pat", "8"]),
        month="2026-09",
        source_job_id="job-a",
        resolutions=[{"name": "NEW, Pat", "add_permanently": False, "mow_price": 30}],
    )
    assert result["conflicts"] == []
    clients = store.list_clients(store)
    assert clients[0]["on_roster"] is False
    assert clients[0]["mow_price"] == 30


def test_match_client_id_files_under_existing(store):
    cid = store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    result = confirm_extract(
        store,
        _sheet(["SMYTH, Mary", "8"]),
        month="2026-09",
        source_job_id="job-a",
        resolutions=[{"name": "SMYTH, Mary", "match_client_id": cid}],
    )
    assert result["conflicts"] == []
    work = store.work_for_month(store, "2026-09")
    assert work[0]["client_id"] == cid
    assert len(store.list_clients(store)) == 1


def test_notes_are_not_stored(store):
    cid = store.upsert_client(None, name="SMITH, Mary", mow_price=40, email="m@example.com")
    written = store_work_text(store, client_id=cid, month="2026-09", text="see bill")
    assert written == 0
    assert store.work_for_month(store, "2026-09") == []


def test_pick_editable_job_prefers_newer_stored_over_old_unsent():
    sheet = {"tables": [{"columns": ["CLIENT", "DATE & WORK COMPLETED"], "rows": [["A", "8"]]}]}
    smoke = {"tables": [{"columns": ["CLIENT", "DATE & WORK COMPLETED"], "rows": [["SMOKE", "1"]]}]}
    jobs = [
        {"id": "old-done", "status": "done", "month": "2026-11", "updated_at": "2026-11-01T10:00:00", "extract_json": smoke},
        {"id": "sent", "status": "stored", "month": "2026-11", "updated_at": "2026-11-01T18:00:00", "extract_json": sheet},
        {"id": "other", "status": "stored", "month": "2026-10", "updated_at": "2026-11-01T19:00:00", "extract_json": sheet},
    ]
    picked = pick_editable_job(jobs, "2026-11")
    assert picked["id"] == "sent"


def test_rebuild_from_saved_review_replaces_stale_job_lines(store):
    store.upsert_client(None, name="ARBORIO, Peter", mow_price=45, email="a@example.com")
    confirm_extract(
        store,
        _sheet(["ARBORIO, Peter", "2 9 15 19 26"]),
        month="2026-11",
        source_job_id="old-job",
    )
    store.save_upload_job(
        store, "new-job", "stored", "Stored",
        extract=_sheet(["ARBORIO, Peter", "2 Branches Dump Run $15 | 9 19 26"]),
        month="2026-11",
    )
    result = rebuild_month_from_review(store, "2026-11")
    assert result["conflicts"] == []
    rows = store.work_for_month(store, "2026-11")
    kinds = {(row["description"], float(row["amount"] or 0), row["day_or_note"]) for row in rows}
    assert ("Branches Dump Run", 15.0, "2") in kinds
    assert ("Mowing 9", 45.0, "9") in kinds
    assert ("Mowing 19", 45.0, "19") in kinds
    assert ("Mowing 26", 45.0, "26") in kinds
    assert ("Mowing 15", 45.0, "15") not in kinds
    assert ("Mowing 2", 45.0, "2") not in kinds
