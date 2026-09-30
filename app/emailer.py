"""SMTP bill emailer (Gmail or any authenticated SMTP)."""

from __future__ import annotations

import re
import smtplib
from email.message import EmailMessage

from . import config, db, storage

_ADDRESS = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)


def smtp_configured() -> bool:
    return bool(config.SMTP_HOST and config.SMTP_FROM)


def person_name(raw: str) -> str:
    """First name, then last name. The roster stores most names as LAST, First."""
    text = " ".join(str(raw or "").split())
    if not text:
        return ""
    if "," not in text:
        return text
    last, _, first = text.partition(",")
    first = " ".join(first.split())
    last = " ".join(last.split())
    if last.isupper():
        last = last.title()
    if first and last:
        return f"{first} {last}"
    return first or last


def recipient_addresses(raw: str) -> list[str]:
    """Every address in a client email field, in order, without duplicates."""
    found: list[str] = []
    seen: set[str] = set()
    for match in _ADDRESS.findall(raw or ""):
        key = match.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(match)
    return found


_ATTACH_LIMIT = 20 * 1024 * 1024


def _send(msg: EmailMessage) -> None:
    if not smtp_configured():
        raise RuntimeError("SMTP not configured (set SMTP_HOST and SMTP_FROM)")
    msg["From"] = config.SMTP_FROM
    if config.SMTP_TLS:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=60) as s:
            s.starttls()
            if config.SMTP_USER:
                s.login(config.SMTP_USER, config.SMTP_PASSWORD)
            s.send_message(msg)
    else:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=60) as s:
            if config.SMTP_USER:
                s.login(config.SMTP_USER, config.SMTP_PASSWORD)
            s.send_message(msg)


def send_pdf(*, to_addr: str, subject: str, body: str, pdf_bytes: bytes, filename: str) -> None:
    send_pdfs(to_addr=to_addr, subject=subject, body=body, attachments=[(filename, pdf_bytes)])


def send_pdfs(*, to_addr: str, subject: str, body: str, attachments: list[tuple[str, bytes]]) -> None:
    msg = EmailMessage()
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    for filename, pdf_bytes in attachments:
        msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)
    _send(msg)


def email_month_bills(conn, month: str, on_progress=None) -> dict:
    """Email each client bill that has an address and has not already been sent."""
    from .billing import contact_email, delivery_channel

    bills = db.bills_for_month(conn, month)
    sent, skipped, errors = [], [], []
    total = len(bills) or 1
    for i, b in enumerate(bills):
        if on_progress:
            on_progress(int(100 * i / total), f"Emailing {b['client_name']}…")
        if delivery_channel(b) != "email":
            skipped.append(b["client_name"])
            continue
        if b.get("emailed_at"):
            skipped.append(b["client_name"])
            continue
        addresses = recipient_addresses(contact_email(b))
        if not addresses:
            skipped.append(b["client_name"])
            continue
        try:
            pdf = storage.get_bytes(b["s3_key"])
            who = person_name(b.get("client_name") or "") or b["client_name"]
            send_pdf(
                to_addr=", ".join(addresses),
                subject=f"{who} — {config.COMPANY_NAME} invoice — {month}",
                body=f"Hi {who},\n\nPlease find your invoice for {month} attached.\n\nThanks,\n{config.COMPANY_NAME}\n",
                pdf_bytes=pdf,
                filename=f"invoice_{month}.pdf",
            )
            db.mark_bill_emailed(conn, int(b["id"]))
            sent.append(b["client_name"])
        except Exception as e:
            errors.append({"client": b["client_name"], "error": str(e)})
    return {"sent": sent, "skipped": skipped, "errors": errors, "smtp_ok": smtp_configured()}


def email_sms_pack(conn, month: str, on_progress=None) -> dict:
    """Email the SMS bills to Nick so he can text them. Not sent to the clients."""
    from .billing import delivery_channel

    bills = [b for b in db.bills_for_month(conn, month) if delivery_channel(b) == "sms"]
    if not bills:
        return {"sent": 0, "clients": [], "messages": 0}
    batches: list[list[tuple[str, bytes]]] = []
    current: list[tuple[str, bytes]] = []
    size = 0
    names: list[str] = []
    for b in bills:
        pdf = storage.get_bytes(b["s3_key"])
        filename = f"{b['client_name'].replace(' ', '_')}.pdf"
        if current and size + len(pdf) > _ATTACH_LIMIT:
            batches.append(current)
            current = []
            size = 0
        current.append((filename, pdf))
        size += len(pdf)
        names.append(b["client_name"])
    if current:
        batches.append(current)
    nick = config.COMPANY_EMAIL
    total = len(batches) or 1
    for i, batch in enumerate(batches):
        if on_progress:
            on_progress(int(100 * i / total), f"Emailing SMS bills to Nick ({i + 1}/{total})…")
        more = f" ({i + 1} of {len(batches)})" if len(batches) > 1 else ""
        send_pdfs(
            to_addr=nick,
            subject=f"SMS bills — {month}{more}",
            body=(
                f"These {month} bills need a text. "
                f"{len(batch)} PDF{'s' if len(batch) != 1 else ''} attached.\n"
            ),
            attachments=batch,
        )
    return {"sent": len(names), "clients": names, "messages": len(batches)}
