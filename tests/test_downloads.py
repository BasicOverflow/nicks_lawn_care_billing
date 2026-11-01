"""Merged mail PDF, zip uniqueness, and download filters."""

import io
import zipfile

import pytest
from pypdf import PdfReader

from app.billing import _bills_for_download, build_pdf_bytes, merge_bills_pdf, zip_bills


def _pdf(name="Client"):
    return build_pdf_bytes(
        company="Test Co",
        client_name=name,
        address="",
        month="2026-09",
        lines=[{"description": "Mowing 8", "amount": 40, "day_or_note": "8"}],
    )


def test_merge_mail_bills_pdf_combines_pages():
    import app.billing as billing_mod

    one = _pdf("Alpha")
    two = _pdf("Beta")

    class FakeStorage:
        def __init__(self, blobs):
            self.blobs = blobs

        def get_bytes(self, key):
            return self.blobs[key]

    blobs = {"a.pdf": one, "b.pdf": two}
    bills = [
        {"client_name": "Alpha", "s3_key": "a.pdf", "delivery": "mail", "prefer_mail": True},
        {"client_name": "Beta", "s3_key": "b.pdf", "delivery": "mail", "prefer_mail": True},
    ]
    old_storage = billing_mod.storage
    billing_mod.storage = FakeStorage(blobs)
    try:
        old_filter = billing_mod._bills_for_download
        billing_mod._bills_for_download = lambda _conn, _month, mode="all": bills
        merged = merge_bills_pdf(None, "2026-09", mode="mailing_only")
    finally:
        billing_mod._bills_for_download = old_filter
        billing_mod.storage = old_storage
    reader = PdfReader(io.BytesIO(merged))
    assert len(reader.pages) == 2


def test_download_modes_filter_delivery(store, blobs):
    mail = store.upsert_client(None, name="MAIL, Pat", prefer_mail=True, mow_price=40)
    sms = store.upsert_client(None, name="SMS, Ann", phone="8605550100", mow_price=40)
    email_c = store.upsert_client(None, name="SMITH, Mary", email="mary@example.com", mow_price=40)
    for cid, key in (
        (mail, "mail.pdf"),
        (sms, "sms.pdf"),
        (email_c, "email.pdf"),
    ):
        blobs.put_bytes(_pdf(), key)
        store.save_bill(store, month="2026-09", client_id=cid, s3_key=key)
    names = lambda mode: [b["client_name"] for b in _bills_for_download(store, "2026-09", mode=mode)]
    assert names("mailing_only") == ["MAIL, Pat"]
    assert names("sms_only") == ["SMS, Ann"]
    assert names("with_email") == ["SMITH, Mary"]
    assert set(names("all")) == {"MAIL, Pat", "SMS, Ann", "SMITH, Mary"}


def test_empty_download_raises():
    with pytest.raises(ValueError, match="No bills match"):
        merge_bills_pdf(None, "2026-09", mode="all")


def test_zip_names_include_client_id(store, blobs):
    a = store.upsert_client(None, name="Same Name", mow_price=40, prefer_mail=True)
    b = store.upsert_client(None, name="Same  Name", mow_price=40, prefer_mail=True)
    blobs.put_bytes(_pdf("A"), f"{a}.pdf")
    blobs.put_bytes(_pdf("B"), f"{b}.pdf")
    store.save_bill(store, month="2026-09", client_id=a, s3_key=f"{a}.pdf")
    store.save_bill(store, month="2026-09", client_id=b, s3_key=f"{b}.pdf")
    raw = zip_bills(store, "2026-09", mode="all")
    names = zipfile.ZipFile(io.BytesIO(raw)).namelist()
    assert len(names) == 2
    assert len(set(names)) == 2
    assert any(name.startswith(f"{a}_") for name in names)
    assert any(name.startswith(f"{b}_") for name in names)
