"""OCR helpers that do not call the vision model."""

from ocr.cancel import InferenceCancelled, begin_job, raise_if_cancelled, request_cancel
from ocr.guided import (
    _assemble_work_rows,
    _chunks,
    _off_sheet_value,
    canonical_columns,
    headers_look_valid,
)
from ocr.jsonutil import try_parse_json
from ocr.prompts import guided_work_prompt


def test_repeated_work_chunk_is_one_review_row():
    columns = ["CLIENT", "DATE & WORK COMPLETED"]
    names = ["SMITH, Mary", "JONES, Ann", "LEE, Pat", "DIAZ, Omar", "NGUYEN, Kim"]
    once = [
        ["SMITH, Mary", "9 16"],
        ["JONES, Ann", "15h"],
        ["LEE, Pat", "5 bush trimming 50"],
        ["DIAZ, Omar", "12"],
        ["NGUYEN, Kim", "3 10"],
    ]
    rows = _assemble_work_rows(names, once + once, columns)
    assert [row[0] for row in rows] == names
    assert [row[1] for row in rows] == [row[1] for row in once]


def test_richer_copy_wins_and_unmatched_name_stays_once():
    columns = ["CLIENT", "DATE & WORK COMPLETED"]
    names = ["SMITH, Mary"]
    pool = [
        ["SMITH, Mary", "9"],
        ["SMITH, Mary", "9 16 23"],
        ["ADDED, Bob", "2"],
        ["ADDED, Bob", "2"],
    ]
    rows = _assemble_work_rows(names, pool, columns)
    assert rows == [
        ["SMITH, Mary", "9 16 23"],
        ["ADDED, Bob", "2"],
    ]


def test_empty_work_rows_are_left_out():
    columns = ["CLIENT", "DATE & WORK COMPLETED"]
    names = ["SMITH, Mary", "BLANK, Pat", "JONES, Ann"]
    pool = [
        ["SMITH, Mary", "9 16"],
        ["BLANK, Pat", ""],
        ["JONES, Ann", "15h"],
    ]
    rows = _assemble_work_rows(names, pool, columns)
    assert [row[0] for row in rows] == ["SMITH, Mary", "JONES, Ann"]
    prompt = guided_work_prompt(["BLANK, Pat"], columns)
    assert "Skip a row when the work cell is blank" in prompt
    assert "If the work cell is blank on the paper, use" not in prompt


def test_headers_look_valid_rejects_cell_values():
    assert headers_look_valid(["CLIENT", "DATE & WORK COMPLETED"])
    assert not headers_look_valid(["$53", "mary@example.com"])
    assert not headers_look_valid(["12 Main Street", "Price"])
    assert not headers_look_valid(["CLIENT"])


def test_off_sheet_value_clears_bleed_not_day_lists():
    assert _off_sheet_value("mary@example.com")
    assert _off_sheet_value("860-555-0100")
    assert _off_sheet_value("12 Main Street")
    assert not _off_sheet_value("9 16 23")
    assert not _off_sheet_value("15h")


def test_json_repair_accepts_fences_and_truncated_objects():
    obj, _ = try_parse_json('```json\n{"title": "sheet", "tables": [], "notes": [], "complete": true}\n```')
    assert obj["title"] == "sheet"
    obj, _ = try_parse_json('{"title": "x", "tables": [{"columns": ["CLIENT"], "rows": [["A"')
    assert isinstance(obj, dict)
    assert obj.get("title") == "x"


def test_canonical_work_columns():
    assert canonical_columns("work") == ["CLIENT", "DATE & WORK COMPLETED"]
    assert canonical_columns("work_completed") == ["CLIENT", "DATE & WORK COMPLETED"]


def test_chunks_split_names_by_four():
    names = [f"N{i}" for i in range(9)]
    parts = _chunks(names, 4)
    assert parts == [["N0", "N1", "N2", "N3"], ["N4", "N5", "N6", "N7"], ["N8"]]


def test_raise_if_cancelled_after_request():
    begin_job("job-a")
    raise_if_cancelled()
    assert request_cancel("job-a")
    try:
        raise_if_cancelled()
        raise AssertionError("expected cancel")
    except InferenceCancelled:
        pass
