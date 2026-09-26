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
    parts = str(name or "").split()
    return parts[0] if parts else "Customer"


def _invoice_when(month: str) -> tuple[str, str, int]:
    """Letter date, work-month name, and work-month number."""
    year_s, month_s = str(month).split("-", 1)
    year, mon = int(year_s), int(month_s)
    work = date(year, mon, 1)
    if mon == 12:
        issued = date(year + 1, 1, 1)
    else:
        issued = date(year, mon + 1, 1)
    letter = f"{issued.strftime('%B')} {_ordinal(issued.day)}, {issued.year}"
    return letter, work.strftime("%B"), mon


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


def compile_invoice_lines(lines: list[dict], month: str) -> list[dict]:
    """Group visits into Date / Description / Price rows, then sales tax and total.

    Every mowing day in the month shares one row. Every hedge day shares one row.
    A written job keeps its name. The price is the sum of the visits on that row.
    """
    _letter, _work_name, month_num = _invoice_when(month)
    groups: list[dict] = []
    index: dict[str, dict] = {}
    for line in lines:
        label, day = _service(line)
        if not label:
            continue
        if label.lower() in {"sales tax", "total"}:
            continue
        key = label.casefold()
        row = index.get(key)
        if row is None:
            row = {"description": label, "days": [], "amount": Decimal("0")}
            index[key] = row
            groups.append(row)
        if day is not None and 1 <= day <= 31 and day not in row["days"]:
            row["days"].append(day)
        row["amount"] += _money(line.get("amount"))
    table = []
    subtotal = Decimal("0")
    for row in groups:
        amount = row["amount"].quantize(_MONEY, rounding=ROUND_HALF_UP)
        subtotal += amount
        dates = ", ".join(f"{month_num}/{day}" for day in sorted(row["days"]))
        table.append({
            "date": dates,
            "description": row["description"],
            "amount": amount,
        })
    tax = (subtotal * _TAX_RATE).quantize(_MONEY, rounding=ROUND_HALF_UP)
    total = (subtotal + tax).quantize(_MONEY, rounding=ROUND_HALF_UP)
    table.append({"date": "", "description": "Sales tax", "amount": tax})
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


def generate_month_bills(conn, month: str) -> list[dict]:
    """Create PDFs for each client with work in month; upload to S3; save bill rows."""
    from . import db

    rows = db.work_for_month(conn, month)
    by_client: dict[int, list] = defaultdict(list)
    meta: dict[int, dict] = {}
    for r in rows:
        cid = int(r["client_id"])
        by_client[cid].append(r)
        meta[cid] = r
    out = []
    for cid, lines in by_client.items():
        m = meta[cid]
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
        "lines": [
            {
                "id": int(ln["id"]),
                "description": ln.get("description") or "",
                "amount": float(ln["amount"]) if ln.get("amount") is not None else 0.0,
            }
            for ln in lines
        ],
    }


def save_editable_bill(conn, month: str, client_id: int, *, email: str, address: str,
                       lines: list[dict]) -> dict:
    """Update client + work lines, regenerate PDF, return updated edit payload."""
    from . import db

    client = db.get_client(conn, client_id)
    if not client:
        raise ValueError("client not found")
    db.update_client_contact(conn, client_id, email=email or None, address=address or None)

    existing_ids = {int(r["id"]) for r in db.work_for_client_month(conn, month, client_id)}
    keep_ids = set()
    for ln in lines:
        desc = str(ln.get("description") or "").strip()
        raw_amt = ln.get("amount")
        try:
            amt = float(raw_amt) if raw_amt is not None and str(raw_amt).strip() != "" else 0.0
        except (TypeError, ValueError):
            amt = 0.0
        wid = ln.get("id")
        if wid and int(wid) in existing_ids:
            db.update_work_item(conn, int(wid), description=desc, amount=amt)
            keep_ids.add(int(wid))
        elif desc or amt:
            new_id = db.add_work_item(
                conn, client_id=client_id, month=month, description=desc, amount=amt,
            )
            keep_ids.add(new_id)
    for wid in existing_ids - keep_ids:
        db.delete_work_item(conn, wid)

    rows = db.work_for_client_month(conn, month, client_id)
    client = db.get_client(conn, client_id)
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
        lines.append(
            f"{r['client_name']}\t{r.get('description') or ''}\t{r.get('amount') or ''}\t{month}"
        )
    return "\n".join(lines) + "\n"


def _has_email(bill: dict) -> bool:
    email = (bill.get("email") or "").strip()
    return bool(email and "@" in email)


def zip_bills(conn, month: str, *, mode: str = "all") -> bytes:
    """Zip PDFs. mode: all | mailing_only (no email — print/mail) | with_email."""
    import zipfile
    from . import db

    bills = db.bills_for_month(conn, month)
    if mode == "mailing_only":
        bills = [b for b in bills if not _has_email(b)]
    elif mode == "with_email":
        bills = [b for b in bills if _has_email(b)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for b in bills:
            data = storage.get_bytes(b["s3_key"])
            name = f"{b['client_name'].replace(' ', '_')}.pdf"
            zf.writestr(name, data)
    return buf.getvalue()
