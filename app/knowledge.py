"""Turn a reviewed sheet table into clients and work items."""

from __future__ import annotations

import re
from typing import Any

from . import db

from ocr.work_marks import parse_work_marks


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


def _stored_amount(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def confirm_extract(
    conn,
    extract: dict,
    *,
    month: str,
    source_job_id: str | None = None,
    sheet_kind: str = "work",
) -> dict:
    """Store a reviewed work-completed sheet as one bill line per job.

    A plain day is a mowing visit at that client's stored mowing price. A day
    with h is a hedge visit at the stored hedge price. A written job name and
    dollar amount is its own line, using that written price. Contact fields and
    the stored prices themselves are left as they are.
    """
    del sheet_kind
    if source_job_id:
        db.delete_work_from_job(conn, month, source_job_id)
    rows = extract_rows(extract)
    clients = 0
    lines = 0
    for r in rows:
        name = r["name"]
        cid = db.upsert_client(conn, name=name)
        lines += store_work_text(
            conn,
            client_id=cid,
            month=month,
            text=_work_text(r),
            source_job_id=source_job_id,
        )
        clients += 1
    return {"written": clients, "lines": lines, "month": month}


def _work_text(row: dict) -> str:
    """Days and jobs from the sheet. The address cell is not a visit."""
    cells = [str(cell).strip() for cell in (row.get("cells") or [])]
    address = (row.get("address") or "").strip()
    parts = []
    for index, cell in enumerate(cells):
        if index == 0 or not cell:
            continue
        if address and cell == address:
            continue
        parts.append(cell)
    return " ".join(parts)


def store_work_text(conn, *, client_id: int, month: str, text: str, source_job_id: str | None = None) -> int:
    """Store mow, hedge, and custom lines parsed from one work cell."""
    client = db.get_client(conn, client_id) or {}
    mow = _stored_amount(client.get("mow_price"))
    hedge = _stored_amount(client.get("hedge_price"))
    jobs = [mark for mark in parse_work_marks(text) if mark["kind"] != "note"]
    written = 0
    for mark in jobs:
        kind = mark["kind"]
        if kind == "mow":
            desc = f"Mowing {mark['day']}"
            amount = 0.0 if mow is None else mow
            day = str(mark["day"])
        elif kind == "hedge":
            desc = f"Hedging {mark['day']}"
            amount = 0.0 if hedge is None else hedge
            day = f"{mark['day']}h"
        else:
            desc = str(mark["name"])
            amount = float(mark["amount"])
            day = str(mark["day"]) if mark.get("day") else None
        db.add_work_item(
            conn,
            client_id=client_id,
            month=month,
            day_or_note=day,
            description=desc[:500],
            amount=amount,
            source_job_id=source_job_id,
        )
        written += 1
    return written
