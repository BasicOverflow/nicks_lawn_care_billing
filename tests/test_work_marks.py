"""Work-cell parser: days, hedges, custom jobs, and known traps."""

from ocr.work_marks import normalize_work_marks, parse_work_marks, review_boxes


def _kinds(text):
    return [(m["kind"], m.get("day"), m.get("name"), m.get("amount")) for m in parse_work_marks(text)]


def test_custom_amount_without_dollar():
    marks = _kinds("bush trimming 50")
    assert ("custom", None, "bush trimming", 50.0) in marks


def test_address_after_price_is_not_a_visit():
    marks = _kinds("trimming $50 12 Main St")
    kinds = [m[0] for m in marks]
    assert kinds == ["custom"]
    assert marks[0][3] == 50.0
    marks = _kinds("50 12 Main St")
    assert [m[0] for m in marks] == ["custom"]
    assert marks[0][3] == 50.0
    marks = _kinds("$50 12 Main St")
    assert [m[0] for m in marks] == ["custom"]


def test_adjacent_numbers_stay_apart():
    marks = _kinds("trimming $50 12")
    assert ("custom", None, "trimming", 50.0) in marks
    assert all(m[3] != 5012 for m in marks)


def test_priced_job_parsing_from_pipe_segments():
    marks = _kinds("16 | Mowing for 262 Giants Neck Rd 32")
    assert ("mow", 16, "", None) in marks
    assert ("custom", None, "Mowing for 262 Giants Neck Rd", 32.0) in marks
    marks = _kinds("8 16 25 | 8 Hedge Trimming $100")
    assert ("mow", 16, "", None) in marks
    assert ("mow", 25, "", None) in marks
    assert ("mow", 8, "", None) not in marks
    assert ("custom", 8, "Hedge Trimming", 100.0) in marks
    marks = _kinds("8 24 S. Lee Mowing 53")
    assert ("mow", 8, "", None) in marks
    assert ("custom", 24, "S. Lee Mowing", 53.0) in marks


def test_day_written_with_a_job_is_the_job_date():
    marks = _kinds("5 bush trimming $50")
    assert ("custom", 5, "bush trimming", 50.0) in marks
    assert ("mow", 5, "", None) not in marks
    marks = _kinds("9 16 5 bush trimming 50")
    assert ("mow", 9, "", None) in marks
    assert ("mow", 16, "", None) in marks
    assert ("mow", 5, "", None) not in marks
    assert ("custom", 5, "bush trimming", 50.0) in marks
    marks = _kinds("5 14 21 21 | Hedge Trimming 87 | 29")
    mow_days = [m[1] for m in marks if m[0] == "mow"]
    assert mow_days == [5, 14, 21, 29]
    assert ("custom", 21, "Hedge Trimming", 87.0) in marks
    marks = _kinds("8 | bush trimming $50")
    assert ("custom", None, "bush trimming", 50.0) in marks
    assert ("mow", 8, "", None) in marks


def test_hedge_marks():
    assert _kinds("15h")[0][0] == "hedge"
    assert _kinds("15H")[0][0] == "hedge"
    assert _kinds("15 h")[0][0] == "hedge"
    assert _kinds("15 hedge")[0][0] == "hedge"
    marks = _kinds("9h bush $40")
    assert ("hedge", 9, "", None) in marks
    assert ("custom", None, "bush", 40.0) in marks or ("custom", 9, "bush", 40.0) in marks


def test_bare_days_are_month_days_not_a_slash_date():
    marks = _kinds("9 16 23")
    assert [m for m in marks if m[0] == "mow"] == [
        ("mow", 9, "", None),
        ("mow", 16, "", None),
        ("mow", 23, "", None),
    ]


def test_slash_and_comma_days_split_apart():
    assert normalize_work_marks("9/16/23") == "9 16 23"
    assert normalize_work_marks("9, 16, 23") == "9 16 23"
    assert [m[1] for m in _kinds("9/16/2023") if m[0] == "mow"] == [9, 16]
    assert [m[1] for m in _kinds("9, 16, 23") if m[0] == "mow"] == [9, 16, 23]


def test_seagrass_trimming_is_one_custom_job():
    marks = _kinds("17 Seagrass Trimming 225")
    assert ("custom", 17, "Seagrass Trimming", 225.0) in marks
    assert ("mow", 17, "", None) not in marks
    spaced = _kinds("17  Seagrass   Trimming  225")
    assert ("custom", 17, "Seagrass Trimming", 225.0) in spaced


def test_seagrass_with_mows_before_or_piped():
    marks = _kinds("9 16 17 Seagrass Trimming 225")
    assert ("mow", 9, "", None) in marks
    assert ("mow", 16, "", None) in marks
    assert ("custom", 17, "Seagrass Trimming", 225.0) in marks
    piped = _kinds("17 Seagrass Trimming 225 | 9 16")
    assert ("custom", 17, "Seagrass Trimming", 225.0) in piped
    assert ("mow", 9, "", None) in piped
    assert ("mow", 16, "", None) in piped


def test_days_after_a_real_price_stay_mow_days():
    marks = _kinds("17 Seagrass Trimming 225 9 16")
    assert ("custom", 17, "Seagrass Trimming", 225.0) in marks
    assert ("mow", 9, "", None) in marks
    assert ("mow", 16, "", None) in marks
    assert ("mow", 17, "", None) not in marks


def test_trailing_dollar_amount_is_the_price():
    intended = _kinds("22 Hedged Trimming 65")
    assert ("custom", 22, "Hedged Trimming", 65.0) in intended
    assert ("mow", 22, "", None) not in intended
    marks = _kinds("22 Hedged Trimming 65 $26")
    assert ("custom", 22, "Hedged Trimming", 65.0) in marks
    assert ("mow", 26, "", None) not in marks
    dump = _kinds("2 Branches Dump Run 15 | 2 9 19 26")
    assert ("custom", 2, "Branches Dump Run", 15.0) in dump
    assert ("mow", 9, "", None) in dump
    assert ("mow", 19, "", None) in dump
    assert ("mow", 26, "", None) in dump
    assert ("mow", 2, "", None) not in dump
    assert ("mow", 15, "", None) not in dump


def test_dump_run_dollar_amount_is_not_a_mow_day():
    marks = _kinds("dump run $15")
    assert ("custom", None, "dump run", 15.0) in marks
    assert ("mow", 15, "", None) not in marks


def test_zero_custom_is_ignored_and_cents_are_kept():
    assert not any(m[0] == "custom" and m[3] == 0 for m in _kinds("bush 0"))
    assert ("custom", None, "bush trimming", 50.5) in _kinds("bush trimming $50.50")


def test_empty_and_note_only_cells():
    assert parse_work_marks("") == []
    assert parse_work_marks("   ") == []
    notes = [m for m in parse_work_marks("see bill") if m["kind"] == "note"]
    assert notes
    assert not any(m["kind"] in {"mow", "hedge", "custom"} for m in parse_work_marks("see bill"))


def test_invalid_days_are_not_visits():
    assert not any(m["kind"] == "mow" for m in parse_work_marks("0"))
    assert not any(m["kind"] == "mow" for m in parse_work_marks("32"))


def test_hedge_word_after_a_mow_becomes_hedge():
    marks = _kinds("9 16 15 hedge")
    assert ("hedge", 15, "", None) in marks or any(
        m[0] == "hedge" and m[1] == 15 for m in marks
    )
    assert ("mow", 9, "", None) in marks
    assert ("mow", 16, "", None) in marks


def test_review_boxes_label_each_mark():
    boxes = review_boxes("8 15h | bush trimming 50")
    assert "8" in boxes
    assert "15h" in boxes
    assert any("bush trimming" in box and "$50" in box for box in boxes)
