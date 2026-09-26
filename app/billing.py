"""PDF bills + tax table."""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from . import config, storage


def build_pdf_bytes(*, company: str, client_name: str, address: str, month: str,
                    lines: list[dict], email: str = "") -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    y = h - inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(inch, y, company)
    y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(inch, y, f"Invoice — {month}")
    y -= 28
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, y, client_name)
    y -= 16
    c.setFont("Helvetica", 10)
    if address:
        c.drawString(inch, y, address)
        y -= 14
    if email:
        c.drawString(inch, y, email)
        y -= 14
    y -= 10
    c.line(inch, y, w - inch, y)
    y -= 20
    total = 0.0
    c.setFont("Helvetica-Bold", 10)
    c.drawString(inch, y, "Description")
    c.drawRightString(w - inch, y, "Amount")
    y -= 16
    c.setFont("Helvetica", 10)
    for line in lines:
        desc = str(line.get("description") or "")[:80]
        amt = float(line.get("amount") or 0)
        total += amt
        if y < inch:
            c.showPage()
            y = h - inch
        c.drawString(inch, y, desc)
        c.drawRightString(w - inch, y, f"${amt:.2f}")
        y -= 14
    y -= 10
    c.line(inch, y, w - inch, y)
    y -= 18
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, y, "Total")
    c.drawRightString(w - inch, y, f"${total:.2f}")
    y -= 28
    c.setFont("Helvetica", 8)
    c.drawString(inch, y, config.CT_TAX_NOTE)
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
