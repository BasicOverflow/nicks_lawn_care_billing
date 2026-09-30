"""PDF bills + tax table."""

from __future__ import annotations

import io
import re
from collections import defaultdict
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen import canvas

from . import config, storage

_MONEY = Decimal("0.01")
_TAX_RATE = Decimal(str(config.CT_SALES_TAX_RATE))
_THANKS = (
    "Thank you for your business. I look forward to continuing my work with you. "
    "Let me know if you have any questions or anything else I can help you with. "
    "Please make the checks payable to Nick's Lawn Care LLC."
)
_MOW = re.compile(r"(?i)^mow(?:ing)?(?:\s+(\d{1,2}))?$")
_HEDGE = re.compile(r"(?i)^hedg(?:e|ing)(?:\s+(\d{1,2}))?$")
_NOTE_DAY = re.compile(r"(?i)^(\d{1,2})h?$")
_PRIOR_NOTE = re.compile(r"(?i)^prior(?P<taxed>-taxed)?:(?P<month>.*)$")
_EMAIL = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)


def _money(amount) -> Decimal:
    return Decimal(str(amount or 0)).quantize(_MONEY, rounding=ROUND_HALF_UP)


def _money_label(amount: Decimal) -> str:
    if amount == amount.to_integral():
        return f"${int(amount)}"
    return f"${amount:.2f}"


def _ordinal(day: int) -> str:
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def _first_name(name: str) -> str:
    """Given name only. Roster rows are stored LAST, First."""
    raw = " ".join(str(name or "").split())
    if not raw:
        return "Customer"
    if "," in raw:
        given = raw.split(",", 1)[1].strip()
    else:
        given = raw
    parts = given.split()
    word = parts[0] if parts else ""
    if not word:
        return "Customer"
    if word.isupper():
        word = word.title()
    return word


def _invoice_when(month: str) -> tuple[str, str, int]:
    """Letter date, work-month name, and work-month number.

    A YYYY-MM month is billed on the first of the next month. Any other label,
    such as a test month, uses today's date and that label as the period.
    """
    raw = str(month or "").strip()
    parts = raw.split("-", 1)
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        year, mon = int(parts[0]), int(parts[1])
        if 1 <= mon <= 12:
            work = date(year, mon, 1)
            if mon == 12:
                issued = date(year + 1, 1, 1)
            else:
                issued = date(year, mon + 1, 1)
            letter = f"{issued.strftime('%B')} {_ordinal(issued.day)}, {issued.year}"
            return letter, work.strftime("%B"), mon
    today = date.today()
    letter = f"{today.strftime('%B')} {_ordinal(today.day)}, {today.year}"
    label = raw.replace("_", " ") or "this period"
    return letter, label, 0


def _service(line: dict) -> tuple[str, int | None]:
    """Job name and day-of-month for one stored work row."""
    desc = str(line.get("description") or "").strip()
    note = str(line.get("day_or_note") or "").strip()
    day = None
    noted = _NOTE_DAY.fullmatch(note)
    if noted:
        day = int(noted.group(1))
    mow = _MOW.fullmatch(desc)
    if mow:
        return "Mowing", day if day is not None else (int(mow.group(1)) if mow.group(1) else None)
    hedge = _HEDGE.fullmatch(desc)
    if hedge:
        return "Hedging", day if day is not None else (int(hedge.group(1)) if hedge.group(1) else None)
    return desc, day


def line_role(line: dict) -> tuple[str, str]:
    """visit, discount, prior, or prior_taxed, plus the prior month label."""
    note = str(line.get("day_or_note") or "").strip()
    desc = str(line.get("description") or "").strip()
    if note.lower() == "discount" or desc.lower().startswith("discount"):
        return "discount", ""
    prior = _PRIOR_NOTE.match(note)
    if prior:
        role = "prior_taxed" if prior.group("taxed") else "prior"
        return role, (prior.group("month") or "").strip()
    kind = str(line.get("kind") or "").strip()
    if kind in {"discount", "prior", "prior_taxed"}:
        return kind, str(line.get("prior_month") or "").strip()
    return "visit", ""


def contact_email(row: dict) -> str:
    """Addresses to email. Notes count only when the email field is empty."""
    email = (row.get("email") or "").strip()
    if "@" in email:
        return email
    return ", ".join(_EMAIL.findall(row.get("billing_notes") or ""))


def delivery_channel(row: dict) -> str:
    """email, then sms when there is a phone, otherwise paper mail."""
    if "@" in contact_email(row):
        return "email"
    if (row.get("phone") or "").strip():
        return "sms"
    return "mail"


def compile_invoice_lines(lines: list[dict], month: str) -> list[dict]:
    """Group visits into Date / Description / Price rows, then sales tax and total.

    Every mowing day in the month shares one row. Every hedge day shares one row.
    A written job keeps its name. A discount reduces the taxable subtotal and has
    no date. A previous-month total is taxed with this month unless it is marked
    as already including sales tax, in which case it is added after the tax.
    """
    _letter, _work_name, month_num = _invoice_when(month)
    groups: list[dict] = []
    index: dict[str, dict] = {}
    for line in lines:
        role, _prior_month = line_role(line)
        label, day = _service(line)
        if not label:
            continue
        if label.lower() in {"sales tax", "total"}:
            continue
        key = f"{role}:{label.casefold()}"
        row = index.get(key)
        if row is None:
            row = {"description": label, "days": [], "amount": Decimal("0"), "role": role}
            index[key] = row
            groups.append(row)
        if role == "visit" and day is not None and 1 <= day <= 31 and day not in row["days"]:
            row["days"].append(day)
        row["amount"] += _money(line.get("amount"))
    table = []
    taxable = Decimal("0")
    after_tax = Decimal("0")
    taxed_rows = []
    for row in groups:
        amount = row["amount"].quantize(_MONEY, rounding=ROUND_HALF_UP)
        if row["role"] == "discount":
            amount = -abs(amount)
        days = sorted(row["days"])
        if row["role"] == "visit" and days:
            if month_num:
                dates = ", ".join(f"{month_num}/{day}" for day in days)
            else:
                dates = ", ".join(str(day) for day in days)
        else:
            dates = ""
        printed = {"date": dates, "description": row["description"], "amount": amount}
        if row["role"] == "prior_taxed":
            taxed_rows.append(printed)
            after_tax += amount
        else:
            table.append(printed)
            taxable += amount
    tax = (taxable * _TAX_RATE).quantize(_MONEY, rounding=ROUND_HALF_UP)
    total = (taxable + tax + after_tax).quantize(_MONEY, rounding=ROUND_HALF_UP)
    table.append({"date": "", "description": "Sales tax", "amount": tax})
    table.extend(taxed_rows)
    table.append({"date": "", "description": "Total", "amount": total})
    return table


def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if stringWidth(trial, font, size) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def build_pdf_bytes(*, company: str, client_name: str, address: str, month: str,
                    lines: list[dict], email: str = "") -> bytes:
    del company, address, email
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    page_w, page_h = letter
    left = 0.75 * inch
    right = page_w - 0.75 * inch
    width = right - left
    rows = compile_invoice_lines(lines, month)
    letter_date, work_month, _month_num = _invoice_when(month)

    cell = ParagraphStyle(
        "invCell", fontName="Times-Roman", fontSize=11, leading=14,
        alignment=TA_LEFT, textColor=colors.black,
    )
    cell_center = ParagraphStyle("invCenter", parent=cell, alignment=TA_CENTER)
    cell_right = ParagraphStyle("invRight", parent=cell, alignment=TA_RIGHT)
    data = [[
        Paragraph("Date", cell_center),
        Paragraph("Description", cell_center),
        Paragraph("Price", cell_center),
    ]]
    for row in rows:
        data.append([
            Paragraph(escape(row["date"]), cell),
            Paragraph(escape(row["description"]), cell),
            Paragraph(escape(_money_label(row["amount"])), cell_right),
        ])
    table = Table(data, colWidths=[width * 0.34, width * 0.46, width * 0.20])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
    ]))
    _tw, table_h = table.wrap(width, page_h)

    def footer() -> None:
        c.setFillColor(colors.black)
        c.setFont("Times-Italic", 11)
        y = 0.95 * inch
        for line in (config.SERVICE_LINE, config.TAX_REGISTRATION):
            c.drawCentredString(page_w / 2, y, line)
            y -= 14
        y -= 8
        c.drawCentredString(page_w / 2, y, config.INSURED_LINE)

    def letterhead(y: float) -> float:
        c.setFillColor(colors.black)
        c.setFont("Times-Roman", 12)
        for line in (
            config.OWNER_NAME,
            config.COMPANY_NAME,
            config.COMPANY_ADDRESS,
            f"{config.COMPANY_PHONE}  {config.COMPANY_EMAIL}",
        ):
            c.drawCentredString(page_w / 2, y, line)
            y -= 15
        return y - 22

    y = letterhead(page_h - 0.7 * inch)
    c.setFont("Times-Roman", 12)
    c.drawString(left, y, letter_date)
    y -= 22
    c.drawString(left, y, f"Dear {_first_name(client_name)},")
    y -= 20
    c.drawString(left, y, f"Below is the invoice for any work done in {work_month}.")
    y -= 18
    table.drawOn(c, left, y - table_h)
    y = y - table_h - 28
    thanks_width = width * 0.92
    for line in _wrap(_THANKS, "Times-Roman", 12, thanks_width):
        c.setFont("Times-Roman", 12)
        c.drawCentredString(page_w / 2, y, line)
        y -= 15
    y -= 22
    c.setFont("Times-Roman", 12)
    c.drawString(left, y, "With regards,")
    y -= 28
    c.drawString(left, y, config.INVOICE_SIGNOFF)
    footer()
    c.showPage()
    c.save()
    return buf.getvalue()


def _reprice_visits(conn, lines: list[dict]) -> list[dict]:
    """Mowing and hedging use the client's current prices. Other lines stay."""
    from . import db

    for line in lines:
        role, _prior = line_role(line)
        if role != "visit":
            continue
        label, _day = _service(line)
        price = None
        if label == "Mowing":
            price = line.get("mow_price")
        elif label == "Hedging":
            price = line.get("hedge_price")
        if price is None:
            continue
        amount = float(price)
        if float(line.get("amount") or 0) != amount:
            db.update_work_item(conn, int(line["id"]), description=line.get("description") or label, amount=amount)
            line["amount"] = amount
    return lines


def generate_month_bills(conn, month: str) -> list[dict]:
    """Create PDFs for each client with work in month; upload to S3; save bill rows."""
    from . import db

    rows = db.work_for_month(conn, month)
    by_client: dict[int, list] = defaultdict(list)
    meta: dict[int, dict] = {}
    for r in rows:
        cid = int(r["client_id"])
        by_client[cid].append(dict(r))
        meta[cid] = r
    out = []
    for cid, lines in by_client.items():
        m = meta[cid]
        _reprice_visits(conn, lines)
        pdf = build_pdf_bytes(
            company=config.COMPANY_NAME,
            client_name=m["client_name"],
            address=m.get("address") or "",
            month=month,
            lines=lines,
            email=m.get("email") or "",
        )
        key = f"bills/{month}/{cid}_{m['client_name'].replace(' ', '_')[:40]}.pdf"
        storage.put_bytes(pdf, key, content_type="application/pdf")
        bid = db.upsert_bill(conn, month=month, client_id=cid, s3_key=key)
        out.append({"bill_id": bid, "client_id": cid, "client_name": m["client_name"],
                    "email": m.get("email"), "s3_key": key})
    return out


def get_editable_bill(conn, month: str, client_id: int) -> dict | None:
    from . import db

    client = db.get_client(conn, client_id)
    if not client:
        return None
    lines = db.work_for_client_month(conn, month, client_id)
    bill = conn.execute(
        """
        SELECT * FROM bills WHERE month = %s AND client_id = %s
        ORDER BY id DESC LIMIT 1
        """,
        (month, client_id),
    ).fetchone()
    return {
        "month": month,
        "client_id": client_id,
        "client_name": client["name"],
        "email": client.get("email") or "",
        "address": client.get("address") or "",
        "s3_key": bill["s3_key"] if bill else None,
        "bill_id": bill["id"] if bill else None,
        "lines": [_editor_line(ln) for ln in lines],
        "preview": [
            {"date": row["date"], "description": row["description"], "amount": float(row["amount"])}
            for row in compile_invoice_lines(lines, month)
        ],
    }


def _editor_line(ln: dict) -> dict:
    role, prior_month = line_role(ln)
    amount = float(ln["amount"]) if ln.get("amount") is not None else 0.0
    return {
        "id": int(ln["id"]),
        "description": ln.get("description") or "",
        "amount": amount,
        "kind": role,
        "prior_month": prior_month,
    }


def _note_for_line(ln: dict, existing: dict | None) -> str | None:
    role = str(ln.get("kind") or "").strip() or line_role(ln)[0]
    prior_month = str(ln.get("prior_month") or "").strip()
    desc = str(ln.get("description") or "").strip()
    if role == "discount" or desc.lower().startswith("discount"):
        return "discount"
    if role == "prior_taxed":
        return f"prior-taxed:{prior_month}"
    if role == "prior":
        return f"prior:{prior_month}"
    if existing and existing.get("day_or_note"):
        return existing.get("day_or_note")
    return None


def save_editable_bill(conn, month: str, client_id: int, *, email: str, address: str,
                       lines: list[dict]) -> dict:
    """Update client + work lines, regenerate PDF, return updated edit payload."""
    from . import db

    client = db.get_client(conn, client_id)
    if not client:
        raise ValueError("client not found")
    db.update_client_contact(conn, client_id, email=email or None, address=address or None)

    stored = {int(r["id"]): r for r in db.work_for_client_month(conn, month, client_id)}
    existing_ids = set(stored)
    keep_ids = set()
    for ln in lines:
        desc = str(ln.get("description") or "").strip()
        raw_amt = ln.get("amount")
        try:
            amt = float(raw_amt) if raw_amt is not None and str(raw_amt).strip() != "" else 0.0
        except (TypeError, ValueError):
            amt = 0.0
        wid = ln.get("id")
        existing = stored.get(int(wid)) if wid and int(wid) in existing_ids else None
        note = _note_for_line(ln, existing)
        if existing:
            db.update_work_item(
                conn, int(wid), description=desc, amount=amt, day_or_note=note,
            )
            keep_ids.add(int(wid))
        elif desc or amt:
            new_id = db.add_work_item(
                conn,
                client_id=client_id,
                month=month,
                day_or_note=note,
                description=desc,
                amount=amt,
            )
            keep_ids.add(new_id)
    for wid in existing_ids - keep_ids:
        db.delete_work_item(conn, wid)

    rows = [dict(r) for r in db.work_for_client_month(conn, month, client_id)]
    client = db.get_client(conn, client_id)
    for row in rows:
        row["mow_price"] = client.get("mow_price")
        row["hedge_price"] = client.get("hedge_price")
    _reprice_visits(conn, rows)
    pdf = build_pdf_bytes(
        company=config.COMPANY_NAME,
        client_name=client["name"],
        address=client.get("address") or "",
        month=month,
        lines=rows,
        email=client.get("email") or "",
    )
    key = f"bills/{month}/{client_id}_{client['name'].replace(' ', '_')[:40]}.pdf"
    storage.put_bytes(pdf, key, content_type="application/pdf")
    bid = db.upsert_bill(conn, month=month, client_id=client_id, s3_key=key)
    return {
        "bill_id": bid,
        "client_id": client_id,
        "client_name": client["name"],
        "email": client.get("email"),
        "s3_key": key,
        "month": month,
    }


def tax_table_tsv(conn, month: str) -> str:
    from . import db

    rows = db.work_for_month(conn, month)
    lines = ["client\tdescription\tamount\tmonth"]
    for r in rows:
        amount = r.get("amount")
        shown = "" if amount is None else amount
        lines.append(
            f"{r['client_name']}\t{r.get('description') or ''}\t{shown}\t{month}"
        )
    return "\n".join(lines) + "\n"


def tax_table_xlsx(conn, month: str) -> bytes:
    """One sheet of invoice lines, including sales tax and the total."""
    from openpyxl import Workbook
    from . import db

    rows = db.work_for_month(conn, month)
    by_client: dict[int, list] = defaultdict(list)
    order: list[int] = []
    names: dict[int, str] = {}
    for row in rows:
        cid = int(row["client_id"])
        if cid not in by_client:
            order.append(cid)
            names[cid] = row["client_name"]
        by_client[cid].append(row)
    book = Workbook()
    sheet = book.active
    sheet.title = "Tax"
    sheet.append(["Client", "Date", "Description", "Amount"])
    for cid in order:
        for line in compile_invoice_lines(by_client[cid], month):
            sheet.append([
                names[cid],
                line["date"],
                line["description"],
                float(line["amount"]),
            ])
    buf = io.BytesIO()
    book.save(buf)
    return buf.getvalue()


def _has_email(bill: dict) -> bool:
    return delivery_channel(bill) == "email"


def zip_bills(conn, month: str, *, mode: str = "all") -> bytes:
    """Zip PDFs. mode: all | mailing_only (paper) | sms_only | with_email."""
    import zipfile
    from . import db

    bills = db.bills_for_month(conn, month)
    if mode == "mailing_only":
        bills = [b for b in bills if delivery_channel(b) == "mail"]
    elif mode == "sms_only":
        bills = [b for b in bills if delivery_channel(b) == "sms"]
    elif mode == "with_email":
        bills = [b for b in bills if delivery_channel(b) == "email"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for b in bills:
            data = storage.get_bytes(b["s3_key"])
            name = f"{b['client_name'].replace(' ', '_')}.pdf"
            zf.writestr(name, data)
    return buf.getvalue()
