"""Roster filing, conflicts, and work-column extract."""

from app.knowledge import extract_rows, filing_client, open_conflicts
from ocr.work_marks import parse_work_marks


def test_unknown_name_is_a_conflict_until_answered():
    extract = {
        "tables": [{
            "columns": ["CLIENT", "DATE & WORK COMPLETED"],
            "rows": [
                ["SMITH, Mary", "9 16"],
                ["ZZZ, Test", "15h"],
                ["ZZZ, Test", "bush 40"],
            ],
        }],
    }
    conflicts = open_conflicts(extract, ["SMITH Mary"], [])
    assert [row["name"] for row in conflicts] == ["ZZZ, Test"]
    assert "15h" in conflicts[0]["work"]
    assert "bush 40" in conflicts[0]["work"]
    closed = open_conflicts(extract, ["SMITH, Mary"], [{"name": "ZZZ, Test", "add_permanently": True}])
    assert closed == []


def test_near_spelling_suggests_the_roster_client():
    extract = {
        "tables": [{
            "columns": ["CLIENT", "DATE & WORK COMPLETED"],
            "rows": [["SMIHT, Mary", "9 16"]],
        }],
    }
    roster = [
        {"id": 4, "name": "SMITH, Mary"},
        {"id": 5, "name": "JONES, Ann"},
    ]
    conflicts = open_conflicts(extract, roster, [])
    assert conflicts[0]["suggestion"]["id"] == 4
    assert conflicts[0]["suggestion"]["name"] == "SMITH, Mary"
    close = {
        "tables": [{
            "columns": ["CLIENT", "DATE & WORK COMPLETED"],
            "rows": [["SMITH, Marie", "4"]],
        }],
    }
    assert open_conflicts(close, roster, [])[0]["suggestion"]["name"] == "SMITH, Mary"
    other = {
        "tables": [{
            "columns": ["CLIENT", "DATE & WORK COMPLETED"],
            "rows": [["SMITH, John", "2"]],
        }],
    }
    assert open_conflicts(other, roster, [])[0]["suggestion"] is None


def test_set_up_client_is_used_for_a_near_spelling():
    roster = [
        {"id": 4, "name": "SMITH, Mary", "address": "123 Main St.", "email": "mary@example.com", "mow_price": 40},
        {"id": 8, "name": "SMIHT, Mary"},
        {"id": 9, "name": "MMM, Test", "address": "9 Oak", "phone": "8605550100"},
    ]
    assert filing_client("SMIHT, Mary", roster)["id"] == 4
    assert filing_client("Test MMM", roster)["id"] == 9
    assert filing_client("SMITH, John", roster) is None
    doubled = [
        {"id": 45, "name": "KEELER, Chris", "address": "43 Brockett Rd.", "email": "cj@example.com", "mow_price": 50},
        {"id": 115, "name": "KELLEER, Chris"},
    ]
    assert filing_client("KELLEER, Chris", doubled)["id"] == 45
    informal = [
        {
            "id": 109,
            "name": "SMALL, Kelly & Ryan",
            "address": "299 Giants Neck Rd.",
            "email": "kelly@example.com",
            "mow_price": 40,
        },
        {"id": 151, "name": "Kelly Small", "email": "kellyhill1@gmail.com"},
        {
            "id": 107,
            "name": "RASKUMAR, Priya",
            "address": "170 Giants Neck Rd.",
            "email": "priya@example.com",
            "mow_price": 40,
        },
    ]
    assert filing_client("Kelly Small", informal)["id"] == 109
    assert filing_client("Priya", informal)["id"] == 107
    assert filing_client("PRYIA", informal)["id"] == 107
    church = [
        {
            "id": 104,
            "name": "SAINT SOPHIA CHURCH",
            "address": "200 Hempstead Street",
            "email": "office@saintsophianl.org",
            "mow_price": 115,
        },
        {"id": 148, "name": "ST Sophia", "email": "office@saintsophianl.org"},
    ]
    assert filing_client("ST Sophia", church)["id"] == 104
    near = {
        "tables": [{
            "columns": ["CLIENT", "DATE & WORK COMPLETED"],
            "rows": [["SMIHT, Mary", "9"]],
        }],
    }
    assert open_conflicts(near, roster, []) == []
    other = {
        "tables": [{
            "columns": ["CLIENT", "DATE & WORK COMPLETED"],
            "rows": [["SMITH, John", "2"]],
        }],
    }
    assert open_conflicts(other, roster, [])[0]["name"] == "SMITH, John"


def test_two_set_up_similar_clients_stay_unresolved():
    roster = [
        {"id": 1, "name": "ANDERSON, Pat", "email": "a@example.com", "mow_price": 40},
        {"id": 2, "name": "ANDERSEN, Pat", "email": "b@example.com", "mow_price": 42},
    ]
    assert filing_client("ANDERSN, Pat", roster) is None


def test_empty_name_does_not_file():
    roster = [{"id": 1, "name": "SMITH, Mary", "mow_price": 40, "email": "a@b.com"}]
    assert filing_client("", roster) is None
    assert filing_client("   ", roster) is None


def test_extract_rows_work_column_not_name_or_price():
    extract = {
        "tables": [{
            "columns": ["CLIENT", "Address", "Price", "DATE & WORK COMPLETED"],
            "rows": [["BOB TEST", "1 Main St", "$75", "8 16h | bush trim 50"]],
        }],
    }
    rows = extract_rows(extract)
    assert len(rows) == 1
    assert rows[0]["work"] == "8 16h | bush trim 50"
    kinds = [m["kind"] for m in parse_work_marks(rows[0]["work"])]
    assert "mow" in kinds
    assert "hedge" in kinds
    assert "custom" in kinds
    assert "$75" not in rows[0]["work"]


def test_extract_rows_when_work_column_is_first():
    extract = {
        "tables": [{
            "columns": ["DATE & WORK COMPLETED", "CLIENT"],
            "rows": [["15h 8 16", "SMITH, Jane"]],
        }],
    }
    rows = extract_rows(extract)
    assert rows[0]["name"] == "SMITH, Jane"
    assert rows[0]["work"] == "15h 8 16"
    assert "mow" in [m["kind"] for m in parse_work_marks(rows[0]["work"])]
