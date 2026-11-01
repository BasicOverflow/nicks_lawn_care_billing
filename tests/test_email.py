"""Delivery channel, recipient lists, email-all skip rules, SMS pack to Nick."""

from app.billing import contact_email, delivery_channel
from app.emailer import (
    email_month_bills,
    email_sms_pack,
    person_name,
    recipient_addresses,
    smtp_configured,
)


def test_delivery():
    assert delivery_channel({"email": "a@b.com", "phone": "860"}) == "email"
    assert delivery_channel({"email": "", "billing_notes": "REGULAR MAIL. a@b.com", "phone": "860"}) == "email"
    assert contact_email({"email": "", "billing_notes": "see a@b.com"}) == "a@b.com"
    assert delivery_channel({"email": "", "phone": "860", "billing_notes": ""}) == "sms"
    assert delivery_channel({"email": "", "phone": "", "billing_notes": ""}) == "mail"
    assert delivery_channel({"email": "a@b.com", "phone": "860", "prefer_mail": True}) == "mail"
    assert delivery_channel({"email": "", "phone": "860", "prefer_mail": True}) == "mail"
    assert delivery_channel({"delivery": "sms", "email": "a@b.com", "phone": "860"}) == "sms"
    assert delivery_channel({"delivery": "mail", "email": "a@b.com"}) == "mail"
    assert delivery_channel({"delivery": "email", "phone": "860"}) == "email"


def test_recipient_addresses_dedupes_and_keeps_order():
    assert recipient_addresses("Ann <a@example.com>, b@example.com; A@example.com") == [
        "a@example.com",
        "b@example.com",
    ]


def test_person_name_title_cases_last():
    assert person_name("SMITH, Mary") == "Mary Smith"
    assert person_name("Kelly Small") == "Kelly Small"
    assert person_name("NOWICK, Pat") == "Pat Nowick"


def test_smtp_configured_false_without_host():
    assert smtp_configured() is False


def test_email_month_skips_mail_sms_and_already_sent(store, blobs, monkeypatch):
    monkeypatch.setattr("app.config.SMTP_HOST", "mail.example.com")
    monkeypatch.setattr("app.config.SMTP_FROM", "billing@example.com")
    sent = []

    def capture(msg):
        sent.append(msg)

    monkeypatch.setattr("app.emailer._send", capture)
    mail = store.upsert_client(
        None, name="MAIL, Pat", email="pat@example.com", prefer_mail=True, mow_price=40,
    )
    sms = store.upsert_client(None, name="SMS, Ann", phone="8605550100", mow_price=40)
    email_c = store.upsert_client(
        None, name="SMITH, Mary", email="mary@example.com, other@example.com", mow_price=40,
    )
    already = store.upsert_client(
        None, name="JONES, Ann", email="ann@example.com", mow_price=40,
    )
    for cid in (mail, sms, email_c, already):
        store.add_work_item(
            None, client_id=cid, month="2026-09", description="Mowing 8", amount=40, day_or_note="8",
        )
        key = f"bills/2026-09/{cid}.pdf"
        blobs.put_bytes(b"%PDF-1.4", key)
        bid = store.save_bill(store, month="2026-09", client_id=cid, s3_key=key)
        if cid == already:
            store.mark_bill_emailed(store, bid)
    result = email_month_bills(store, "2026-09")
    assert result["sent"] == ["SMITH, Mary"]
    assert "MAIL, Pat" in result["skipped"]
    assert "SMS, Ann" in result["skipped"]
    assert "JONES, Ann" in result["skipped"]
    assert len(sent) == 1
    assert sent[0]["To"] == "mary@example.com, other@example.com"
    assert "Mary Smith" in sent[0]["Subject"]
    assert "2026-09" in sent[0]["Subject"]


def test_rebuild_clears_emailed_at(store, blobs):
    from app.billing import generate_month_bills

    cid = store.upsert_client(None, name="SMITH, Mary", email="mary@example.com", mow_price=40)
    store.add_work_item(
        None, client_id=cid, month="2026-09", description="Mowing 8", amount=40, day_or_note="8",
    )
    generate_month_bills(store, "2026-09")
    bid = store.bill_for_client_month(store, "2026-09", cid)["id"]
    store.mark_bill_emailed(store, bid)
    generate_month_bills(store, "2026-09")
    assert store.bill_for_client_month(store, "2026-09", cid)["emailed_at"] is None


def test_sms_pack_goes_to_nick_and_splits_large_attachments(store, blobs, monkeypatch):
    monkeypatch.setattr("app.config.SMTP_HOST", "mail.example.com")
    monkeypatch.setattr("app.config.SMTP_FROM", "billing@example.com")
    monkeypatch.setattr("app.config.COMPANY_EMAIL", "nick@example.com")
    monkeypatch.setattr("app.emailer._ATTACH_LIMIT", 20)
    sent = []

    def capture(msg):
        sent.append(msg)

    monkeypatch.setattr("app.emailer._send", capture)
    a = store.upsert_client(None, name="SMS, Ann", phone="8605550100", mow_price=40)
    b = store.upsert_client(None, name="SMS, Bob", phone="8605550101", mow_price=40)
    for cid in (a, b):
        store.add_work_item(
            None, client_id=cid, month="2026-09", description="Mowing 8", amount=40, day_or_note="8",
        )
        key = f"bills/2026-09/{cid}.pdf"
        blobs.put_bytes(b"x" * 15, key)
        store.save_bill(store, month="2026-09", client_id=cid, s3_key=key)
    result = email_sms_pack(store, "2026-09")
    assert result["sent"] == 2
    assert result["messages"] == 2
    assert all(msg["To"] == "nick@example.com" for msg in sent)
    assert not any("Ann" in (msg["To"] or "") for msg in sent)


def test_sms_pack_empty(store):
    assert email_sms_pack(store, "2026-09") == {"sent": 0, "clients": [], "messages": 0}
