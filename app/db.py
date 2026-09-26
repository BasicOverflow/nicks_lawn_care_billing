"""Postgres access — thin helpers, no ORM sprawl."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT,
  phone TEXT,
  address TEXT,
  billing_address TEXT,
  billing_notes TEXT,
  mow_price NUMERIC,
  hedge_price NUMERIC,
  hedge_roster BOOLEAN NOT NULL DEFAULT FALSE,
  prefer_mail BOOLEAN NOT NULL DEFAULT FALSE,
  entity_kind TEXT NOT NULL DEFAULT 'client',
  sort_order INTEGER,
  mowing_group TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS clients_name_norm ON clients (lower(trim(name)));

CREATE TABLE IF NOT EXISTS work_items (
  id SERIAL PRIMARY KEY,
  client_id INTEGER REFERENCES clients(id) ON DELETE CASCADE,
  month TEXT NOT NULL,
  day_or_note TEXT,
  description TEXT,
  amount NUMERIC,
  source_job_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS work_items_month ON work_items (month);

CREATE TABLE IF NOT EXISTS upload_jobs (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  progress_msg TEXT,
  extract_json JSONB,
  s3_prefix TEXT,
  month TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bills (
  id SERIAL PRIMARY KEY,
  month TEXT NOT NULL,
  client_id INTEGER REFERENCES clients(id) ON DELETE CASCADE,
  s3_key TEXT NOT NULL,
  emailed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS bills_month ON bills (month);
"""

# Additive migrations for DBs created before billing_address / guided-OCR fields.
_MIGRATIONS = (
    "ALTER TABLE clients ADD COLUMN IF NOT EXISTS billing_address TEXT",
    "ALTER TABLE clients ADD COLUMN IF NOT EXISTS prefer_mail BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE clients ADD COLUMN IF NOT EXISTS entity_kind TEXT NOT NULL DEFAULT 'client'",
    "ALTER TABLE clients ADD COLUMN IF NOT EXISTS sort_order INTEGER",
    "ALTER TABLE clients ADD COLUMN IF NOT EXISTS mowing_group TEXT",
    "ALTER TABLE clients ADD COLUMN IF NOT EXISTS hedge_roster BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS sheet_kind TEXT",
    "ALTER TABLE upload_jobs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
)


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)
        for stmt in _MIGRATIONS:
            conn.execute(stmt)


def now() -> datetime:
    return datetime.now(timezone.utc)


def upsert_client(
    conn,
    *,
    name: str,
    email=None,
    phone=None,
    address=None,
    billing_address=None,
    billing_notes=None,
    mow_price=None,
    hedge_price=None,
    hedge_roster=None,
    prefer_mail=None,
    entity_kind=None,
    sort_order=None,
    mowing_group=None,
) -> int:
    existing = get_client_by_name(conn, name)
    if existing:
        conn.execute(
            """
            UPDATE clients SET
              email = COALESCE(%s, email),
              phone = COALESCE(%s, phone),
              address = COALESCE(%s, address),
              billing_address = COALESCE(%s, billing_address),
              billing_notes = COALESCE(%s, billing_notes),
              mow_price = COALESCE(%s, mow_price),
              hedge_price = COALESCE(%s, hedge_price),
              hedge_roster = COALESCE(%s, hedge_roster),
              prefer_mail = COALESCE(%s, prefer_mail),
              entity_kind = COALESCE(%s, entity_kind),
              sort_order = COALESCE(%s, sort_order),
              mowing_group = COALESCE(%s, mowing_group),
              updated_at = NOW()
            WHERE id = %s
            """,
            (
                email,
                phone,
                address,
                billing_address,
                billing_notes,
                mow_price,
                hedge_price,
                hedge_roster,
                prefer_mail,
                entity_kind,
                sort_order,
                mowing_group,
                existing["id"],
            ),
        )
        return int(existing["id"])
    row = conn.execute(
        """
        INSERT INTO clients (
          name, email, phone, address, billing_address, billing_notes,
          mow_price, hedge_price, hedge_roster, prefer_mail, entity_kind, sort_order, mowing_group
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
        """,
        (
            name,
            email,
            phone,
            address,
            billing_address,
            billing_notes,
            mow_price,
            hedge_price,
            bool(hedge_roster) if hedge_roster is not None else False,
            bool(prefer_mail) if prefer_mail is not None else False,
            entity_kind or "client",
            sort_order,
            mowing_group,
        ),
    ).fetchone()
    return int(row["id"])


def get_client_by_name(conn, name: str) -> dict | None:
    return conn.execute(
        "SELECT * FROM clients WHERE lower(trim(name)) = lower(trim(%s))",
        (name,),
    ).fetchone()


def save_typed_client(
    conn,
    *,
    client_id: int | None,
    name: str,
    address: str | None,
    phone: str | None,
    email: str | None,
    billing_notes: str | None,
    mow_price: float | None,
    hedge_price: float | None,
    prefer_mail: bool,
) -> int:
    """Write a hand-entered roster row. Blank prices clear the stored price."""
    name = name.strip()
    if not name:
        raise ValueError("name required")
    other = get_client_by_name(conn, name)
    hedge_roster = hedge_price is not None
    if client_id:
        if other and int(other["id"]) != int(client_id):
            raise ValueError(f"{name} is already on file")
        if not get_client(conn, int(client_id)):
            raise ValueError(f"Client {client_id} was not found")
        conn.execute(
            """
            UPDATE clients SET
              name = %s,
              address = %s,
              phone = %s,
              email = %s,
              billing_notes = %s,
              mow_price = %s,
              hedge_price = %s,
              hedge_roster = %s,
              prefer_mail = %s,
              updated_at = NOW()
            WHERE id = %s
            """,
            (
                name, address, phone, email, billing_notes,
                mow_price, hedge_price, hedge_roster, bool(prefer_mail),
                int(client_id),
            ),
        )
        return int(client_id)
    if other:
        return save_typed_client(
            conn,
            client_id=int(other["id"]),
            name=name,
            address=address,
            phone=phone,
            email=email,
            billing_notes=billing_notes,
            mow_price=mow_price,
            hedge_price=hedge_price,
            prefer_mail=prefer_mail,
        )
    return upsert_client(
        conn,
        name=name,
        email=email,
        phone=phone,
        address=address,
        billing_notes=billing_notes,
        mow_price=mow_price,
        hedge_price=hedge_price,
        hedge_roster=hedge_roster,
        prefer_mail=prefer_mail,
    )


def write_office_knowledge(
    conn,
    client_id: int,
    *,
    name: str | None = None,
    billing_notes: str | None = None,
    hedge_price: float | None = None,
    hedge_roster: bool = False,
) -> None:
    """Overwrite fields the office import owns. NULL clears a previous value.

    upsert_client uses COALESCE, which would leave a ground-truth hedge price
    in place. This path is the one the seeder uses for those columns.
    """
    conn.execute(
        """
        UPDATE clients SET
          name = COALESCE(%s, name),
          billing_notes = %s,
          hedge_price = %s,
          hedge_roster = %s,
          updated_at = NOW()
        WHERE id = %s
        """,
        (name, billing_notes, hedge_price, bool(hedge_roster), client_id),
    )


def list_clients(conn) -> list[dict]:
    return list(
        conn.execute(
            """
            SELECT * FROM clients
            ORDER BY sort_order NULLS LAST, name
            """
        ).fetchall()
    )


_KNOWLEDGE_SELECT = """
SELECT name, address, billing_address, billing_notes, phone, email,
       mow_price, hedge_price, prefer_mail, entity_kind
FROM clients
"""


def _plain_record(row: dict) -> dict:
    rec = {
        "name": str(row["name"]),
        "address": row.get("address") or "",
        "billing_address": row.get("billing_address") or "",
        "billing_notes": row.get("billing_notes") or "",
        "phone": row.get("phone") or "",
        "email": row.get("email") or "",
        "prefer_mail": bool(row.get("prefer_mail")),
        "entity_kind": row.get("entity_kind") or "client",
    }
    for key in ("mow_price", "hedge_price"):
        val = row.get(key)
        rec[key] = float(val) if val is not None else None
    return rec


def knowledge_records_for_sheet(conn, sheet_kind: str = "mowing") -> list[dict]:
    """Live client rows used to guide OCR. No ground-truth labels."""
    kind = (sheet_kind or "mowing").strip().lower()
    order = " ORDER BY sort_order NULLS LAST, name"
    if kind == "hedges":
        where = """
            WHERE name NOT ILIKE '%%smoke%%'
              AND (hedge_roster OR hedge_price IS NOT NULL)
        """
    elif kind in ("work", "work_completed", "other"):
        where = " WHERE name NOT ILIKE '%%smoke%%' "
    else:
        where = """
            WHERE name NOT ILIKE '%%smoke%%'
              AND (
                mow_price IS NOT NULL
                OR entity_kind IN ('parcel', 'association')
              )
        """
    rows = conn.execute(_KNOWLEDGE_SELECT + where + order).fetchall()
    if not rows and kind not in ("hedges", "work", "work_completed", "other"):
        rows = conn.execute(
            _KNOWLEDGE_SELECT + " WHERE name NOT ILIKE '%%smoke%%'" + order
        ).fetchall()
    return [_plain_record(r) for r in rows if r.get("name")]


def knowledge_names_for_sheet(conn, sheet_kind: str = "mowing") -> list[str]:
    """Names to guide OCR chunk prompts, filtered by sheet kind."""
    return [r["name"] for r in knowledge_records_for_sheet(conn, sheet_kind)]


def add_work_item(conn, *, client_id: int, month: str, day_or_note=None,
                  description=None, amount=None, source_job_id=None) -> int:
    row = conn.execute(
        """
        INSERT INTO work_items (client_id, month, day_or_note, description, amount, source_job_id)
        VALUES (%s,%s,%s,%s,%s,%s) RETURNING id
        """,
        (client_id, month, day_or_note, description, amount, source_job_id),
    ).fetchone()
    return int(row["id"])


def work_for_month(conn, month: str) -> list[dict]:
    return list(
        conn.execute(
            """
            SELECT w.*, c.name AS client_name, c.email, c.address, c.phone, c.billing_notes
            FROM work_items w
            JOIN clients c ON c.id = w.client_id
            WHERE w.month = %s
            ORDER BY c.name, w.id
            """,
            (month,),
        ).fetchall()
    )


def months_with_work(conn) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT month FROM work_items ORDER BY month DESC"
    ).fetchall()
    return [r["month"] for r in rows]


def save_upload_job(conn, job_id: str, status: str, progress_msg: str = "",
                    extract: Any = None, s3_prefix: str | None = None, month: str | None = None,
                    sheet_kind: str | None = None) -> None:
    conn.execute(
        """
        INSERT INTO upload_jobs (id, status, progress_msg, extract_json, s3_prefix, month, sheet_kind, updated_at)
        VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s, NOW())
        ON CONFLICT (id) DO UPDATE SET
          status = EXCLUDED.status,
          progress_msg = EXCLUDED.progress_msg,
          extract_json = COALESCE(EXCLUDED.extract_json, upload_jobs.extract_json),
          s3_prefix = COALESCE(EXCLUDED.s3_prefix, upload_jobs.s3_prefix),
          month = COALESCE(EXCLUDED.month, upload_jobs.month),
          sheet_kind = COALESCE(EXCLUDED.sheet_kind, upload_jobs.sheet_kind),
          updated_at = NOW()
        """,
        (
            job_id,
            status,
            progress_msg,
            json.dumps(extract) if extract is not None else None,
            s3_prefix,
            month,
            sheet_kind,
        ),
    )


def save_review_draft(conn, job_id: str, extract: dict, month: str | None, sheet_kind: str | None) -> dict | None:
    """Write in-progress table edits. Leaves a stored job alone."""
    row = conn.execute(
        """
        UPDATE upload_jobs SET
          extract_json = %s::jsonb,
          month = COALESCE(%s, month),
          sheet_kind = COALESCE(%s, sheet_kind),
          updated_at = NOW()
        WHERE id = %s AND status = 'done'
        RETURNING id, month, sheet_kind, updated_at
        """,
        (json.dumps(extract), month or None, sheet_kind or None, job_id),
    ).fetchone()
    return row


def clear_review(conn, job_id: str) -> dict | None:
    """Drop an open review without writing clients or work."""
    return conn.execute(
        """
        UPDATE upload_jobs SET
          status = 'cleared',
          progress_msg = 'Review cleared',
          updated_at = NOW()
        WHERE id = %s AND status = 'done'
        RETURNING id, status, updated_at
        """,
        (job_id,),
    ).fetchone()


def get_upload_job(conn, job_id: str) -> dict | None:
    return conn.execute("SELECT * FROM upload_jobs WHERE id = %s", (job_id,)).fetchone()


def save_bill(conn, *, month: str, client_id: int, s3_key: str) -> int:
    row = conn.execute(
        """
        INSERT INTO bills (month, client_id, s3_key) VALUES (%s,%s,%s) RETURNING id
        """,
        (month, client_id, s3_key),
    ).fetchone()
    return int(row["id"])


def bills_for_month(conn, month: str) -> list[dict]:
    return list(
        conn.execute(
            """
            SELECT b.*, c.name AS client_name, c.email
            FROM bills b JOIN clients c ON c.id = b.client_id
            WHERE b.month = %s ORDER BY c.name
            """,
            (month,),
        ).fetchall()
    )


def work_for_client_month(conn, month: str, client_id: int) -> list[dict]:
    return list(
        conn.execute(
            """
            SELECT w.*, c.name AS client_name, c.email, c.address, c.phone, c.billing_notes
            FROM work_items w
            JOIN clients c ON c.id = w.client_id
            WHERE w.month = %s AND w.client_id = %s
            ORDER BY w.id
            """,
            (month, client_id),
        ).fetchall()
    )


def get_client(conn, client_id: int) -> dict | None:
    return conn.execute("SELECT * FROM clients WHERE id = %s", (client_id,)).fetchone()


def update_client_contact(conn, client_id: int, *, email=None, address=None, name=None) -> None:
    conn.execute(
        """
        UPDATE clients SET
          email = %s,
          address = %s,
          name = COALESCE(%s, name),
          updated_at = NOW()
        WHERE id = %s
        """,
        (email or None, address or None, name, client_id),
    )


def patch_client(
    conn,
    client_id: int,
    *,
    name=None,
    email=None,
    phone=None,
    address=None,
    billing_notes=None,
    mow_price=None,
    hedge_price=None,
) -> None:
    """Update only fields present in kwargs (use "" to clear text; None = leave unchanged).

    Callers should pass keys they want to change; missing keys are left alone.
    We detect 'provided' via a sentinel by checking which args were passed — use
    explicit _unset for clarity.
    """
    row = get_client(conn, client_id)
    if not row:
        raise ValueError("client not found")

    def _text(new, old):
        if new is None:
            return old
        s = str(new).strip()
        return s or None

    def _num(new, old):
        if new is None:
            return old
        if str(new).strip() == "":
            return None
        return float(new)

    # Convention: only update fields that appear in the mutation (not None).
    # For chat mutations, omitted fields are None → keep old.
    conn.execute(
        """
        UPDATE clients SET
          name = %s,
          email = %s,
          phone = %s,
          address = %s,
          billing_notes = %s,
          mow_price = %s,
          hedge_price = %s,
          updated_at = NOW()
        WHERE id = %s
        """,
        (
            _text(name, row["name"]) if name is not None else row["name"],
            _text(email, row["email"]) if email is not None else row["email"],
            _text(phone, row["phone"]) if phone is not None else row["phone"],
            _text(address, row["address"]) if address is not None else row["address"],
            _text(billing_notes, row["billing_notes"])
            if billing_notes is not None
            else row["billing_notes"],
            _num(mow_price, row["mow_price"]) if mow_price is not None else row["mow_price"],
            _num(hedge_price, row["hedge_price"]) if hedge_price is not None else row["hedge_price"],
            client_id,
        ),
    )


def update_work_item(conn, work_id: int, *, description: str, amount) -> None:
    conn.execute(
        "UPDATE work_items SET description = %s, amount = %s WHERE id = %s",
        (description, amount, work_id),
    )


def patch_work_item(conn, work_id: int, *, description=None, amount=None, month=None) -> None:
    row = conn.execute("SELECT * FROM work_items WHERE id = %s", (work_id,)).fetchone()
    if not row:
        raise ValueError("work item not found")
    new_desc = row["description"] if description is None else str(description)
    if amount is None:
        new_amt = row["amount"]
    elif str(amount).strip() == "":
        new_amt = None
    else:
        new_amt = float(amount)
    new_month = row["month"] if month is None else str(month).strip()
    conn.execute(
        "UPDATE work_items SET description = %s, amount = %s, month = %s WHERE id = %s",
        (new_desc, new_amt, new_month, work_id),
    )


def delete_work_item(conn, work_id: int) -> None:
    conn.execute("DELETE FROM work_items WHERE id = %s", (work_id,))


def delete_month_work(conn, month: str) -> dict:
    """Remove stored work and generated bills for one YYYY-MM. Clients stay."""
    month = (month or "").strip()
    work = conn.execute(
        "DELETE FROM work_items WHERE month = %s RETURNING id",
        (month,),
    ).fetchall()
    bills = conn.execute(
        "DELETE FROM bills WHERE month = %s RETURNING id",
        (month,),
    ).fetchall()
    return {"month": month, "work_items": len(work), "bills": len(bills)}


def upsert_bill(conn, *, month: str, client_id: int, s3_key: str) -> int:
    existing = conn.execute(
        """
        SELECT id FROM bills WHERE month = %s AND client_id = %s
        ORDER BY id DESC LIMIT 1
        """,
        (month, client_id),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE bills SET s3_key = %s, emailed_at = NULL WHERE id = %s",
            (s3_key, existing["id"]),
        )
        return int(existing["id"])
    return save_bill(conn, month=month, client_id=client_id, s3_key=s3_key)


def mark_bill_emailed(conn, bill_id: int) -> None:
    conn.execute("UPDATE bills SET emailed_at = NOW() WHERE id = %s", (bill_id,))
