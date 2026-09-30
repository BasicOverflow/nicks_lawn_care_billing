"""Money, name, and delivery rules. Uses placeholder clients only."""

from app.billing import _first_name, compile_invoice_lines, contact_email, delivery_channel
from ocr.work_marks import parse_work_marks


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
    assert ("custom",  None, "trimming", 50.0) in [(a, b, c, d) for a, b, c, d in marks] or (
        "custom", None, "trimming", 50.0
    ) in marks
    # 12 follows the price with no street, so it can still be a mow. It must not become 5012.
    assert all(m[3] != 5012 for m in marks)


def test_day_plus_custom_stays_both():
    marks = _kinds("5 bush trimming $50")
    assert ("mow", 5, "", None) in marks
    assert ("custom", 5, "bush trimming", 50.0) in marks


def test_hedge_marks():
    assert _kinds("15h")[0][0] == "hedge"
    assert _kinds("15H")[0][0] == "hedge"
    assert _kinds("15 h")[0][0] == "hedge"
    assert _kinds("15 hedge")[0][0] == "hedge"
    marks = _kinds("9h bush $40")
    assert ("hedge", 9, "", None) in marks
    assert ("custom", None, "bush", 40.0) in marks or ("custom", 9, "bush", 40.0) in marks


def test_tax_includes_pretax_prior_and_not_an_already_taxed_total():
    base = [{"description": "Mowing", "amount": 100, "day_or_note": "5"}]
    rows = compile_invoice_lines(base + [{
        "description": "Previous bill (2026-08)",
        "amount": 50,
        "day_or_note": "prior:2026-08",
    }], "2026-09")
    by_name = {row["description"]: row for row in rows}
    assert by_name["Sales tax"]["amount"] == __import__("decimal").Decimal("9.53")
    assert by_name["Total"]["amount"] == __import__("decimal").Decimal("159.53")
    assert by_name["Previous bill (2026-08)"]["date"] == ""

    rows = compile_invoice_lines(base + [{
        "description": "Previous bill (2026-08)",
        "amount": 50,
        "day_or_note": "prior-taxed:2026-08",
    }], "2026-09")
    by_name = {row["description"]: row for row in rows}
    assert by_name["Sales tax"]["amount"] == __import__("decimal").Decimal("6.35")
    assert by_name["Total"]["amount"] == __import__("decimal").Decimal("156.35")


def test_discount_reduces_tax_and_has_no_date():
    rows = compile_invoice_lines([
        {"description": "Mowing", "amount": 100, "day_or_note": "5"},
        {"description": "Discount", "amount": 10, "day_or_note": "discount"},
    ], "2026-09")
    by_name = {row["description"]: row for row in rows}
    assert by_name["Discount"]["date"] == ""
    assert by_name["Discount"]["amount"] == __import__("decimal").Decimal("-10.00")
    assert by_name["Sales tax"]["amount"] == __import__("decimal").Decimal("5.72")


def test_dear_is_the_given_name():
    assert _first_name("SMITH, Mary") == "Mary"
    assert _first_name("AGOSTINI, Lillian") == "Lillian"
    assert _first_name("SMITH, MARY") == "Mary"


def test_delivery():
    assert delivery_channel({"email": "a@b.com", "phone": "860"}) == "email"
    assert delivery_channel({"email": "", "billing_notes": "REGULAR MAIL. a@b.com", "phone": "860"}) == "email"
    assert contact_email({"email": "", "billing_notes": "see a@b.com"}) == "a@b.com"
    assert delivery_channel({"email": "", "phone": "860", "billing_notes": ""}) == "sms"
    assert delivery_channel({"email": "", "phone": "", "billing_notes": ""}) == "mail"


def test_repeated_work_chunk_is_one_review_row():
    from ocr.guided import _assemble_work_rows

    columns = ["CLIENT", "DATE & WORK COMPLETED"]
    names = ["SMITH, Mary", "JONES, Ann", "LEE, Pat", "DIAZ, Omar", "NGUYEN, Kim"]
    once = [
        ["SMITH, Mary", "9 16"],
        ["JONES, Ann", "15h"],
        ["LEE, Pat", "5 bush trimming 50"],
        ["DIAZ, Omar", "12"],
        ["NGUYEN, Kim", "3 10"],
    ]
    # Two chunks each returned the whole sheet.
    rows = _assemble_work_rows(names, once + once, columns)
    assert [row[0] for row in rows] == names
    assert [row[1] for row in rows] == [row[1] for row in once]


def test_richer_copy_wins_and_unmatched_name_stays_once():
    from ocr.guided import _assemble_work_rows

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
    from ocr.guided import _assemble_work_rows
    from ocr.prompts import guided_work_prompt

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

