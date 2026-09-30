"""Chat with stored clients / work / bills via the OCR model (text-only).

Supports read answers and write mutations (update client/work, add/delete work).
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from . import db

CHAT_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "mutations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "update_client",
                            "update_work",
                            "add_work",
                            "delete_work",
                            "delete_work_month",
                        ],
                    },
                    "client_id": {"type": ["integer", "null"]},
                    "client_name": {"type": ["string", "null"]},
                    "work_id": {"type": ["integer", "null"]},
                    "name": {"type": ["string", "null"]},
                    "email": {"type": ["string", "null"]},
                    "phone": {"type": ["string", "null"]},
                    "address": {"type": ["string", "null"]},
                    "billing_notes": {"type": ["string", "null"]},
                    "mow_price": {"type": ["number", "null"]},
                    "hedge_price": {"type": ["number", "null"]},
                    "month": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "amount": {"type": ["number", "null"]},
                },
                "required": ["op"],
            },
        },
    },
    "required": ["reply", "mutations"],
}


def knowledge_snapshot(conn, *, limit_work: int = 80, limit_clients: int = 120) -> str:
    """Client and work list small enough for the vision model's context.

    Prompts past roughly 14,000 characters make the server return an empty
    reply, which the chat showed as "No response from model."
    """
    clients = [row for row in db.list_clients(conn) if row.get("on_roster") is not False]
    work = list(
        conn.execute(
            """
            SELECT w.id, w.client_id, w.month, w.description, w.amount,
                   c.name AS client_name
            FROM work_items w
            JOIN clients c ON c.id = w.client_id
            ORDER BY w.month DESC, c.name, w.id
            LIMIT %s
            """,
            (max(limit_work, 200),),
        ).fetchall()
    )
    bills = list(
        conn.execute(
            """
            SELECT b.id, b.month, b.client_id, c.name AS client_name,
                   b.emailed_at IS NOT NULL AS emailed
            FROM bills b JOIN clients c ON c.id = b.client_id
            ORDER BY b.month DESC, c.name
            LIMIT 80
            """
        ).fetchall()
    )
    month_counts: dict[str, int] = {}
    for row in conn.execute(
        "SELECT month, count(*) AS n FROM work_items GROUP BY month"
    ).fetchall():
        month_counts[str(row["month"])] = int(row["n"])

    def pack(n_clients: int, n_work: int, n_bills: int) -> str:
        payload = {
            "client_count": len(clients),
            "work_counts_by_month": month_counts,
            "clients_truncated": n_clients < len(clients),
            "clients": [
                {
                    "id": int(c["id"]),
                    "name": c["name"],
                    "email": c.get("email") or "",
                    "mow_price": float(c["mow_price"]) if c.get("mow_price") is not None else None,
                    "hedge_price": float(c["hedge_price"]) if c.get("hedge_price") is not None else None,
                }
                for c in clients[:n_clients]
            ],
            "work_items": [
                {
                    "id": int(w["id"]),
                    "client_id": int(w["client_id"]),
                    "client": w["client_name"],
                    "month": w["month"],
                    "description": w.get("description"),
                    "amount": float(w["amount"]) if w.get("amount") is not None else None,
                }
                for w in work[:n_work]
            ],
            "bills": [
                {
                    "id": int(b["id"]),
                    "client_id": int(b["client_id"]),
                    "client": b["client_name"],
                    "month": b["month"],
                    "emailed": bool(b.get("emailed")),
                }
                for b in bills[:n_bills]
            ],
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    n_clients, n_work, n_bills = min(limit_clients, len(clients)), min(limit_work, len(work)), min(40, len(bills))
    text = pack(n_clients, n_work, n_bills)
    while len(text) > 10000 and (n_clients > 15 or n_work > 10 or n_bills > 0):
        if n_work > 10:
            n_work = max(10, n_work // 2)
        elif n_bills > 0:
            n_bills = n_bills // 2
        else:
            n_clients = max(15, n_clients // 2)
        text = pack(n_clients, n_work, n_bills)
    return text


def _resolve_client_id(conn, mut: dict) -> int | None:
    if mut.get("client_id") is not None:
        try:
            return int(mut["client_id"])
        except (TypeError, ValueError):
            pass
    name = (mut.get("client_name") or mut.get("name") or "").strip()
    if name:
        row = db.get_client_by_name(conn, name)
        if row:
            return int(row["id"])
    return None


def apply_mutations(conn, mutations: list[dict]) -> list[str]:
    """Apply mutation list; return human-readable log lines."""
    applied: list[str] = []
    for raw in mutations or []:
        if not isinstance(raw, dict):
            continue
        op = (raw.get("op") or "").strip().lower()
        try:
            if op == "update_client":
                cid = _resolve_client_id(conn, raw)
                if not cid:
                    applied.append("skipped update_client (unknown client)")
                    continue
                before = db.get_client(conn, cid)
                db.patch_client(
                    conn,
                    cid,
                    name=raw.get("name"),
                    email=raw.get("email"),
                    phone=raw.get("phone"),
                    address=raw.get("address"),
                    billing_notes=raw.get("billing_notes"),
                    mow_price=raw.get("mow_price"),
                    hedge_price=raw.get("hedge_price"),
                )
                after = db.get_client(conn, cid)
                applied.append(
                    f"updated client #{cid} {before['name'] if before else ''} → "
                    f"email={after.get('email')!s}, address={after.get('address')!s}, "
                    f"mow={after.get('mow_price')}, hedge={after.get('hedge_price')}"
                )
            elif op == "update_work":
                wid = raw.get("work_id")
                if wid is None:
                    applied.append("skipped update_work (no work_id)")
                    continue
                wid = int(wid)
                db.patch_work_item(
                    conn,
                    wid,
                    description=raw.get("description"),
                    amount=raw.get("amount"),
                    month=raw.get("month"),
                )
                applied.append(f"updated work_item #{wid}")
            elif op == "add_work":
                cid = _resolve_client_id(conn, raw)
                month = (raw.get("month") or "").strip()
                if not cid or not month:
                    applied.append("skipped add_work (need client + month)")
                    continue
                desc = str(raw.get("description") or "service")
                amt = raw.get("amount")
                try:
                    amt_f = float(amt) if amt is not None and str(amt).strip() != "" else None
                except (TypeError, ValueError):
                    amt_f = None
                new_id = db.add_work_item(
                    conn, client_id=cid, month=month, description=desc, amount=amt_f,
                )
                applied.append(f"added work_item #{new_id} for client #{cid} ({month})")
            elif op == "delete_work":
                wid = raw.get("work_id")
                if wid is None:
                    applied.append("skipped delete_work (no work_id)")
                    continue
                db.delete_work_item(conn, int(wid))
                applied.append(f"deleted work_item #{wid}")
            elif op == "delete_work_month":
                month = (raw.get("month") or "").strip()
                if not re.fullmatch(r"20\d{2}-\d{2}", month):
                    applied.append("skipped delete_work_month (month must be YYYY-MM)")
                    continue
                cleared = db.delete_month_work(conn, month)
                applied.append(
                    f"deleted {cleared['work_items']} work lines and {cleared['bills']} bills for {month}"
                )
            else:
                applied.append(f"skipped unknown op {op!r}")
        except Exception as e:
            applied.append(f"failed {op}: {e}")
    return applied


_MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _wipe_month(question: str, *, today: date | None = None) -> str | None:
    """YYYY-MM when the user asked to delete a whole month of work. Otherwise None.

    This does not call the model. A month-wide delete is one statement, and
    asking the model to list every work id was returning an empty reply.
    """
    text = (question or "").strip().lower()
    if not re.search(r"\b(delete|remove|wipe|clear|erase|drop)\b", text):
        return None
    if not re.search(r"\b(work[-\s]?completed|work|jobs?|visits?|mows?|mowing|bills?|invoices?)\b", text):
        return None
    today = today or date.today()
    explicit = re.search(r"\b(20\d{2})-(\d{2})\b", text)
    if explicit:
        month_n = int(explicit.group(2))
        if 1 <= month_n <= 12:
            return f"{explicit.group(1)}-{month_n:02d}"
        return None
    names = "|".join(sorted(_MONTH_NAMES, key=len, reverse=True))
    named = re.search(rf"\b({names})\b(?:\s+(20\d{{2}}))?", text)
    if not named:
        return None
    month_n = _MONTH_NAMES[named.group(1)]
    if named.group(2):
        year = int(named.group(2))
    elif month_n <= today.month:
        year = today.year
    else:
        year = today.year - 1
    return f"{year}-{month_n:02d}"


def answer(conn, question: str, history: list[dict[str, str]] | None = None) -> dict:
    """Chat: answer and optionally mutate datastore. Returns {answer, mutations_applied}."""
    direct = _wipe_month(question)
    if direct:
        cleared = db.delete_month_work(conn, direct)
        work_n = cleared["work_items"]
        bill_n = cleared["bills"]
        if work_n or bill_n:
            reply = (
                f"Deleted {work_n} work line{'' if work_n == 1 else 's'}"
                f" and {bill_n} bill{'' if bill_n == 1 else 's'} for {direct}. "
                "Clients and their prices were kept."
            )
        else:
            reply = f"There was no work or bill stored for {direct}."
        applied = [
            f"deleted {work_n} work lines and {bill_n} bills for {direct}"
        ]
        return {"answer": reply, "mutations_applied": applied, "mutations": [
            {"op": "delete_work_month", "month": direct}
        ]}

    import ocr
    from ocr.chat import chat_text
    from ocr.jsonutil import try_parse_json

    snap = knowledge_snapshot(conn)
    hist = history or []
    hist_txt = ""
    for turn in hist[-8:]:
        role = turn.get("role") or "user"
        hist_txt += f"{role}: {turn.get('content') or ''}\n"

    prompt = (
        "You are the assistant for Nick's Lawn Care billing datastore.\n"
        "You can ANSWER questions and/or CHANGE stored data when the user asks.\n\n"
        "Return JSON only with:\n"
        '  "reply": string (what to tell the user),\n'
        '  "mutations": array of change ops (empty if read-only).\n\n'
        "Mutation ops:\n"
        '- update_client: set client_id (or client_name) and any of name,email,phone,address,'
        "billing_notes,mow_price,hedge_price. Omit fields you are not changing; "
        "use empty string to clear text fields.\n"
        "- update_work: set work_id and any of description,amount,month.\n"
        "- add_work: set client_id or client_name, month (YYYY-MM), description, amount.\n"
        "- delete_work: set work_id.\n"
        "- delete_work_month: set month YYYY-MM to delete every work line and bill for that month. "
        "Use this for 'delete all September work'. Do not emit one delete_work per row.\n"
        "Use ids from the snapshot. Never invent clients. Do not delete clients.\n"
        "If the user is only asking a question, mutations=[].\n"
        "If they ask to change data, include mutations and summarize them in reply.\n\n"
        f"DATASTORE JSON:\n{snap}\n\n"
        f"Recent chat:\n{hist_txt}\n"
        f"user: {question.strip()}\n"
    )
    text = chat_text(
        ocr.MODEL_ID,
        prompt,
        max_tokens=2048,
        temperature=0.1,
        guided_json=True,
        json_schema=CHAT_REPLY_SCHEMA,
        schema_name="chat_reply",
    )
    obj, _ = try_parse_json(text or "")
    reply = ""
    mutations: list[dict] = []
    if isinstance(obj, dict):
        reply = str(obj.get("reply") or "").strip()
        raw_m = obj.get("mutations") or []
        if isinstance(raw_m, list):
            mutations = [m for m in raw_m if isinstance(m, dict)]
    if not reply:
        reply = (text or "").strip()
    if not reply:
        raise RuntimeError("The model returned no text.")

    applied = apply_mutations(conn, mutations) if mutations else []
    if applied:
        reply = reply.rstrip() + "\n\nChanges applied:\n- " + "\n- ".join(applied)
    return {"answer": reply, "mutations_applied": applied, "mutations": mutations}

