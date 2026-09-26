"""Chat with stored clients / work / bills via the OCR model (text-only).

Supports read answers and write mutations (update client/work, add/delete work).
"""

from __future__ import annotations

import json
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


def knowledge_snapshot(conn, *, limit_work: int = 400, limit_clients: int = 200) -> str:
    clients = db.list_clients(conn)[:limit_clients]
    work = list(
        conn.execute(
            """
            SELECT w.id, w.client_id, w.month, w.description, w.amount, w.day_or_note,
                   c.name AS client_name
            FROM work_items w
            JOIN clients c ON c.id = w.client_id
            ORDER BY w.month DESC, c.name, w.id
            LIMIT %s
            """,
            (limit_work,),
        ).fetchall()
    )
    bills = list(
        conn.execute(
            """
            SELECT b.id, b.month, b.client_id, c.name AS client_name, c.email,
                   b.emailed_at IS NOT NULL AS emailed
            FROM bills b JOIN clients c ON c.id = b.client_id
            ORDER BY b.month DESC, c.name
            LIMIT 200
            """
        ).fetchall()
    )
    payload = {
        "clients": [
            {
                "id": int(c["id"]),
                "name": c["name"],
                "email": c.get("email"),
                "phone": c.get("phone"),
                "address": c.get("address"),
                "billing_notes": c.get("billing_notes"),
                "mow_price": float(c["mow_price"]) if c.get("mow_price") is not None else None,
                "hedge_price": float(c["hedge_price"]) if c.get("hedge_price") is not None else None,
            }
            for c in clients
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
            for w in work
        ],
        "bills": [
            {
                "id": int(b["id"]),
                "client_id": int(b["client_id"]),
                "client": b["client_name"],
                "month": b["month"],
                "email": b.get("email"),
                "emailed": bool(b.get("emailed")),
            }
            for b in bills
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > 60000:
        text = text[:60000] + "…"
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
            else:
                applied.append(f"skipped unknown op {op!r}")
        except Exception as e:
            applied.append(f"failed {op}: {e}")
    return applied


def answer(conn, question: str, history: list[dict[str, str]] | None = None) -> dict:
    """Chat: answer and optionally mutate datastore. Returns {answer, mutations_applied}."""
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
        # Fallback: treat raw model text as reply (no mutations)
        reply = (text or "").strip() or "No response from model."
        # strip accidental JSON fence
        if reply.startswith("{"):
            reply = "Done." if mutations else reply[:2000]

    applied = apply_mutations(conn, mutations) if mutations else []
    if applied:
        reply = reply.rstrip() + "\n\nChanges applied:\n- " + "\n- ".join(applied)
    return {"answer": reply, "mutations_applied": applied, "mutations": mutations}


def resolve_conflicts_nl(
    conflicts: list[dict[str, Any]],
    instruction: str,
) -> dict[str, str]:
    """Map natural-language conflict instructions → {client_name: 'a'|'b'|'merge'}."""
    if not conflicts:
        return {}
    if not (instruction or "").strip():
        return {c["name"]: "b" for c in conflicts}

    import ocr
    from ocr.chat import chat_text
    from ocr.jsonutil import try_parse_json

    schema = {
        "type": "object",
        "properties": {
            "resolutions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "choice": {"type": "string", "enum": ["a", "b", "merge"]},
                        "note": {"type": "string"},
                    },
                    "required": ["name", "choice"],
                },
            }
        },
        "required": ["resolutions"],
    }
    compact = []
    for c in conflicts:
        compact.append(
            {
                "name": c["name"],
                "issues": c.get("issues"),
                "incoming": {
                    "address": (c.get("incoming") or {}).get("address"),
                    "price": (c.get("incoming") or {}).get("price"),
                    "notes": (c.get("incoming") or {}).get("notes"),
                },
            }
        )
    prompt = (
        "Resolve datastore conflicts for a lawn-care billing app.\n"
        "For each conflict, choose:\n"
        "  a = keep existing knowledge (ignore incoming for that client)\n"
        "  b = prefer incoming sheet values (overwrite)\n"
        "  merge = apply incoming updates (same as b for this app)\n\n"
        f"Conflicts JSON:\n{json.dumps(compact, ensure_ascii=False)[:12000]}\n\n"
        f"User instructions:\n{instruction.strip()}\n\n"
        "Return JSON only with resolutions array."
    )
    text = chat_text(
        ocr.MODEL_ID,
        prompt,
        max_tokens=1024,
        temperature=0.0,
        guided_json=True,
        json_schema=schema,
        schema_name="conflict_resolutions",
    )
    obj, _ = try_parse_json(text or "")
    out: dict[str, str] = {c["name"]: "b" for c in conflicts}
    if isinstance(obj, dict):
        for item in obj.get("resolutions") or []:
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or "").strip()
            choice = (item.get("choice") or "b").strip().lower()
            if name and choice in ("a", "b", "merge"):
                out[name] = "b" if choice == "merge" else choice
    return out
