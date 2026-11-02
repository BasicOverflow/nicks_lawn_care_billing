"""Invoice compile, CT tax, letter date, and tax workbook."""

from datetime import date
from decimal import Decimal
from io import BytesIO

from openpyxl import load_workbook
from pypdf import PdfReader

from app.billing import (
    _first_name,
    _invoice_when,
    _split_amount,
    build_pdf_bytes,
    compile_invoice_lines,
    default_closing,
    default_greeting,
    default_signoff,
    invoice_sentence,
    letter_or_default,
    stored_letter,
    tax_client_figures,
    tax_workbook_bytes,
    work_month_ym,
)


def test_prior_month_job_shows_previous_date():
    rows = compile_invoice_lines([
        {"description": "Mowing", "amount": 40, "day_or_note": "5"},
        {
            "description": "Bush trimming",
            "amount": 50,
            "day_or_note": "prior:2026-08:12",
        },
    ], "2026-09")
    by_name = {row["description"]: row for row in rows}
    assert by_name["Bush trimming"]["date"] == "8/12"
    assert by_name["Total"]["amount"] == Decimal("95.72")


def test_mowing_days_group_and_zero_fills_from_roster():
    rows = compile_invoice_lines([
        {"description": "Bush trimming", "amount": 50, "day_or_note": "5"},
        {"description": "Mowing 8", "amount": 0, "day_or_note": "8", "mow_price": 42},
        {"description": "Mowing 15", "amount": 0, "day_or_note": "15", "mow_price": 42},
    ], "2026-09")
    by_name = {row["description"]: row for row in rows}
    assert by_name["Bush trimming"]["date"] == "9/5"
    assert by_name["Mowing"]["amount"] == Decimal("84.00")
    assert by_name["Mowing"]["date"] == "9/8, 9/15"


def test_hedging_days_group_on_one_row():
    rows = compile_invoice_lines([
        {"description": "Hedging 15", "amount": 0, "day_or_note": "15h", "hedge_price": 40},
        {"description": "Hedging 22", "amount": 0, "day_or_note": "22h", "hedge_price": 40},
    ], "2026-09")
    hedge = next(row for row in rows if row["description"] == "Hedging")
    assert hedge["date"] == "9/15, 9/22"
    assert hedge["amount"] == Decimal("80.00")


def test_zero_visit_with_no_roster_price_still_prints():
    rows = compile_invoice_lines([
        {"description": "Mowing 8", "amount": 0, "day_or_note": "8"},
    ], "2026-09")
    mowing = next(row for row in rows if row["description"] == "Mowing")
    assert mowing["amount"] == Decimal("0.00")
    assert mowing["date"] == "9/8"
    tax = next(row for row in rows if row["description"] == "Sales tax")
    assert tax["amount"] == Decimal("0.00")


def test_tax_rate_is_ct_half_up():
    rows = compile_invoice_lines([
        {"description": "Mowing", "amount": 100, "day_or_note": "5"},
    ], "2026-09")
    by_name = {row["description"]: row for row in rows}
    assert by_name["Sales tax"]["amount"] == Decimal("6.35")
    rows = compile_invoice_lines([
        {"description": "Mowing", "amount": 111, "day_or_note": "5"},
    ], "2026-09")
    by_name = {row["description"]: row for row in rows}
    assert by_name["Sales tax"]["amount"] == Decimal("7.05")


def test_tax_includes_pretax_prior_and_not_an_already_taxed_total():
    base = [{"description": "Mowing", "amount": 100, "day_or_note": "5"}]
    rows = compile_invoice_lines(base + [{
        "description": "Previous bill (2026-08)",
        "amount": 50,
        "day_or_note": "prior:2026-08",
    }], "2026-09")
    by_name = {row["description"]: row for row in rows}
    assert by_name["Sales tax"]["amount"] == Decimal("9.53")
    assert by_name["Total"]["amount"] == Decimal("159.53")
    assert by_name["Previous bill (2026-08)"]["date"] == ""

    rows = compile_invoice_lines(base + [{
        "description": "Previous bill (2026-08)",
        "amount": 50,
        "day_or_note": "prior-taxed:2026-08",
    }], "2026-09")
    by_name = {row["description"]: row for row in rows}
    assert by_name["Sales tax"]["amount"] == Decimal("6.35")
    assert by_name["Total"]["amount"] == Decimal("156.35")


def test_discount_reduces_tax_and_has_no_date():
    rows = compile_invoice_lines([
        {"description": "Mowing", "amount": 100, "day_or_note": "5"},
        {"description": "Discount", "amount": 10, "day_or_note": "discount"},
    ], "2026-09")
    by_name = {row["description"]: row for row in rows}
    assert by_name["Discount"]["date"] == ""
    assert by_name["Discount"]["amount"] == Decimal("-10.00")
    assert by_name["Sales tax"]["amount"] == Decimal("5.72")


def test_sales_tax_and_total_input_rows_are_dropped():
    rows = compile_invoice_lines([
        {"description": "Mowing", "amount": 100, "day_or_note": "5"},
        {"description": "Sales tax", "amount": 99, "day_or_note": "5"},
        {"description": "Total", "amount": 999, "day_or_note": "5"},
    ], "2026-09")
    tax_rows = [row for row in rows if row["description"] == "Sales tax"]
    assert len(tax_rows) == 1
    assert tax_rows[0]["amount"] == Decimal("6.35")


def test_tax_workbook_separates_revenue_tax_and_previous_unpaid():
    taxed_now = tax_client_figures([
        {"description": "Mowing", "amount": 100, "day_or_note": "5"},
        {"description": "Previous bill (2026-08)", "amount": 50, "day_or_note": "prior:2026-08"},
    ])
    assert taxed_now["revenue"] == Decimal("100.00")
    assert taxed_now["prior"] == Decimal("50.00")
    assert taxed_now["tax"] == Decimal("9.53")
    already = tax_client_figures([
        {"description": "Mowing", "amount": 40, "day_or_note": "2"},
        {"description": "Previous bill (2026-08)", "amount": 20, "day_or_note": "prior-taxed:2026-08"},
    ])
    assert already["revenue"] == Decimal("40.00")
    assert already["tax"] == Decimal("2.54")
    assert already["prior_taxed"] == Decimal("20.00")
    plain = tax_client_figures([{"description": "Mowing", "amount": 10, "day_or_note": "1"}])
    data = tax_workbook_bytes("2026-09", [
        ("SMITH, Mary", taxed_now),
        ("JONES, Ann", already),
        ("LEE, Pat", plain),
    ])
    book = load_workbook(BytesIO(data))
    sheet = book.active
    assert sheet["A1"].value.startswith("Tax table for September 2026")
    assert sheet["B3"].value == "This month's revenue"
    assert sheet["D3"].value == "Sales tax"
    assert "Previous unpaid" in sheet["C3"].value
    assert sheet["B4"].value == 100
    assert sheet["C4"].value == 50
    assert sheet["D4"].value == 9.53
    assert "August 2026" in sheet["G4"].value
    assert sheet["E5"].value == 20
    assert sheet["C6"].value == 0
    labels = [sheet.cell(row, 1).value for row in range(8, 14)]
    assert "This month's revenue" in labels
    assert "Sales tax" in labels
    assert any(str(label).startswith("Previous unpaid") for label in labels)


def test_dear_is_the_given_name():
    assert _first_name("SMITH, Mary") == "Mary"
    assert _first_name("AGOSTINI, Lillian") == "Lillian"
    assert _first_name("SMITH, MARY") == "Mary"
    assert _first_name("SMITH Mary") == "Mary"
    assert _first_name("Kelly Small") == "Kelly"


def test_invoice_sentence_can_be_replaced():
    assert invoice_sentence("2026-09") == "Below is the invoice for any work done in September."
    custom = "Below is the invoice for work done in September, plus the August balance."
    assert invoice_sentence("2026-09", custom) == custom
    assert invoice_sentence("2026-09", "  ") == "Below is the invoice for any work done in September."


def test_letter_wording_restores_when_cleared():
    assert default_greeting("SMITH, Mary") == "Dear Mary,"
    assert "checks payable" in default_closing()
    assert default_signoff().startswith("With regards,")
    assert letter_or_default("  ", default_greeting("SMITH, Mary")) == "Dear Mary,"
    assert letter_or_default("Hello Mary,", default_greeting("SMITH, Mary")) == "Hello Mary,"
    assert stored_letter("Dear Mary,", default_greeting("SMITH, Mary")) is None
    assert stored_letter("Hello Mary,", default_greeting("SMITH, Mary")) == "Hello Mary,"


def test_work_month_is_previous_calendar_month():
    assert work_month_ym(date(2026, 11, 1)) == "2026-10"
    assert work_month_ym(date(2026, 1, 15)) == "2025-12"
    assert work_month_ym(date(2026, 10, 31)) == "2026-09"


def test_letter_date_is_the_first_of_the_next_month():
    letter, work, month_n = _invoice_when("2026-09")
    assert letter == "October 1st, 2026"
    assert work == "September"
    assert month_n == 9
    letter, work, month_n = _invoice_when("2026-12")
    assert letter == "January 1st, 2027"
    assert work == "December"


def test_test_month_letter_uses_today(monkeypatch):
    monkeypatch.setattr("app.billing.date", type("D", (), {
        "today": staticmethod(lambda: date(2026, 11, 1)),
    }))
    letter, work, month_n = _invoice_when("test_month")
    assert "November" in letter
    assert "2026" in letter
    assert work == "test month"
    assert month_n == 0


def test_split_amount_puts_remainder_on_last_piece():
    assert _split_amount(100, 3) == [33.33, 33.33, 33.34]
    assert _split_amount(50, 1) == [50.0]
    assert _split_amount(10, 0) == []


def test_pdf_contains_next_month_date_and_dear_first_name():
    pdf = build_pdf_bytes(
        company="Test Co",
        client_name="SMITH, Mary",
        address="1 Main St",
        month="2026-09",
        lines=[{"description": "Mowing 8", "amount": 40, "day_or_note": "8"}],
    )
    text = "".join((page.extract_text() or "") for page in PdfReader(BytesIO(pdf)).pages)
    assert "October 1st, 2026" in text
    assert "Dear Mary" in text
