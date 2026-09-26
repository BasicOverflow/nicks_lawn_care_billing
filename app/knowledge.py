"""Confirm extracts into knowledge; simple conflict detection."""

from __future__ import annotations

import re
from typing import Any

from . import db


def _cell(row: list, idx: int) -> str:
    if idx < 0 or idx >= len(row):
        return ""
    return str(row[idx] or "").strip()


def _price(raw: str) -> float | None:
    if not raw:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", raw.replace(",", ""))
    return float(m.group(1)) if m else None


def extract_rows(extract: dict) -> list[dict]:
    """Flatten OCR tables into {name, address, price, notes, cells}."""
    out: list[dict] = []
    for t in extract.get("tables") or []:
        if not isinstance(t, dict):
            continue
        cols = [str(c).lower() for c in (t.get("columns") or [])]
        name_i = next((i for i, c in enumerate(cols) if "contact" in c or "name" in c or i == 0), 0)
        addr_i = next((i for i, c in enumerate(cols) if "address" in c and "billing" not in c), -1)
        price_i = next((i for i, c in enumerate(cols) if "price" in c or "hedge" in c or "mow" in c), -1)
        bill_i = next((i for i, c in enumerate(cols) if "billing" in c or "note" in c or "email" in c), -1)
        for row in t.get("rows") or []:
            if not isinstance(row, list) or not any(str(c).strip() for c in row):
                continue
            name = _cell(row, name_i)
            if not name or name.upper() in ("[UNCLEAR]", "CONTACT", "NAME"):
                continue
            out.append({
                "name": name,
                "address": _cell(row, addr_i) if addr_i >= 0 else "",
                "price": _price(_cell(row, price_i)) if price_i >= 0 else None,
                "notes": _cell(row, bill_i) if bill_i >= 0 else "",
                "cells": [str(c) for c in row],
            })
    return out


def find_conflicts(conn, rows: list[dict], month: str) -> list[dict]:
    """Compare incoming rows to existing clients / month work."""
    conflicts = []
    for r in rows:
        existing = db.get_client_by_name(conn, r["name"])
        if not existing:
            continue
        issues = []
        if r.get("price") is not None:
            for field, key in (("mow_price", "mow"), ("hedge_price", "hedge")):
                old = existing.get(field)
                if old is not None and abs(float(old) - float(r["price"])) > 0.01:
                    issues.append({"field": field, "knowledge": float(old), "incoming": float(r["price"])})
        if r.get("address") and existing.get("address") and r["address"] != existing["address"]:
            issues.append({"field": "address", "knowledge": existing["address"], "incoming": r["address"]})
        if issues:
            conflicts.append({"name": r["name"], "client_id": existing["id"], "issues": issues, "incoming": r})
    return conflicts


def confirm_extract(
    conn,
    extract: dict,
    *,
    month: str,
    source_job_id: str | None = None,
    resolutions: dict | None = None,
    sheet_kind: str = "mowing",
) -> dict:
    """Write rows into clients + work_items. resolutions: {name: 'a'|'b'|'merge'}."""
    resolutions = resolutions or {}
    rows = extract_rows(extract)
    written = 0
    for r in rows:
        name = r["name"]
        res = resolutions.get(name, "b")  # default prefer incoming
        existing = db.get_client_by_name(conn, name)
        email = phone = None
        notes = r.get("notes") or ""
        if "@" in notes:
            email = notes.split()[0] if notes.split() else notes
        price = r.get("price")
        mow = price if sheet_kind == "mowing" else None
        hedge = price if sheet_kind == "hedges" else None
        if existing and res == "a":
            cid = existing["id"]
        else:
            cid = db.upsert_client(
                conn,
                name=name,
                email=email,
                address=r.get("address") or None,
                billing_notes=notes or None,
                mow_price=mow,
                hedge_price=hedge,
            )
        # Work completed style: put day/note in description from all cells
        desc = " | ".join(c for c in (r.get("cells") or [])[1:] if c.strip()) or notes or "service"
        db.add_work_item(
            conn,
            client_id=cid,
            month=month,
            day_or_note=None,
            description=desc[:500],
            amount=price,
            source_job_id=source_job_id,
        )
        written += 1
    return {"written": written, "month": month}
