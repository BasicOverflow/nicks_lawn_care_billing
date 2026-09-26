"""SMTP bill emailer (docker-mailserver / any SMTP)."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from . import config, db, storage


def smtp_configured() -> bool:
    return bool(config.SMTP_HOST and config.SMTP_FROM)


def send_pdf(*, to_addr: str, subject: str, body: str, pdf_bytes: bytes, filename: str) -> None:
    if not smtp_configured():
        raise RuntimeError("SMTP not configured (set SMTP_HOST and SMTP_FROM)")
    msg = EmailMessage()
    msg["From"] = config.SMTP_FROM
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)
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


def email_month_bills(conn, month: str, on_progress=None) -> dict:
    """Email each bill that has a client email. Returns counts."""
    bills = db.bills_for_month(conn, month)
    sent, skipped, errors = [], [], []
    total = len(bills) or 1
    for i, b in enumerate(bills):
        if on_progress:
            on_progress(int(100 * i / total), f"Emailing {b['client_name']}…")
        email = (b.get("email") or "").strip()
        if not email or "@" not in email:
            skipped.append(b["client_name"])
            continue
        try:
            pdf = storage.get_bytes(b["s3_key"])
            send_pdf(
                to_addr=email,
                subject=f"{config.COMPANY_NAME} invoice — {month}",
                body=f"Hi {b['client_name']},\n\nPlease find your invoice for {month} attached.\n\nThanks,\n{config.COMPANY_NAME}\n",
                pdf_bytes=pdf,
                filename=f"invoice_{month}.pdf",
            )
            db.mark_bill_emailed(conn, int(b["id"]))
            sent.append(b["client_name"])
        except Exception as e:
            errors.append({"client": b["client_name"], "error": str(e)})
    return {"sent": sent, "skipped": skipped, "errors": errors, "smtp_ok": smtp_configured()}
