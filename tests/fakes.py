"""In-memory stand-ins for Postgres and MinIO. No live services."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock


def _now():
    return datetime.now(timezone.utc)


def _copy(row: dict | None) -> dict | None:
    return None if row is None else deepcopy(row)


class FakeStorage:
    def __init__(self):
        self.blobs: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def put_bytes(self, data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
        self.blobs[key] = data
        return key

    def get_bytes(self, key: str) -> bytes:
        return self.blobs[key]

    def delete_key(self, key: str) -> None:
        self.deleted.append(key)
        self.blobs.pop(key, None)

    def ensure_bucket(self) -> None:
        return None


class FakeStore:
    """Dict-backed clients, work, upload jobs, and bills."""

    def __init__(self):
        self.clients: dict[int, dict] = {}
        self.work_items: dict[int, dict] = {}
        self.upload_jobs: dict[str, dict] = {}
        self.bills: dict[int, dict] = {}
        self._client_id = 1
        self._work_id = 1
        self._bill_id = 1
        self._lock = Lock()

    @contextmanager
    def connect(self):
        yield self

    def init_db(self) -> None:
        return None

    def now(self):
        return _now()

    def _name_key(self, name: str) -> str:
        return str(name or "").strip().lower()

    def get_client(self, conn, client_id: int) -> dict | None:
        return _copy(self.clients.get(int(client_id)))

    def get_client_by_name(self, conn, name: str) -> dict | None:
        key = self._name_key(name)
        for row in self.clients.values():
            if self._name_key(row["name"]) == key:
                return _copy(row)
        return None

    def list_clients(self, conn) -> list[dict]:
        rows = list(self.clients.values())
        rows.sort(key=lambda row: (row.get("sort_order") is None, row.get("sort_order") or 0, row["name"]))
        return [_copy(row) for row in rows]

    def upsert_client(self, conn, *, name: str, email=None, phone=None, address=None,
                      billing_address=None, billing_notes=None, mow_price=None,
                      hedge_price=None, hedge_roster=None, prefer_mail=None,
                      on_roster=None, entity_kind=None, sort_order=None,
                      mowing_group=None) -> int:
        existing = self.get_client_by_name(conn, name)
        if existing:
            row = self.clients[int(existing["id"])]
            for key, value in {
                "email": email, "phone": phone, "address": address,
                "billing_address": billing_address, "billing_notes": billing_notes,
                "mow_price": mow_price, "hedge_price": hedge_price,
                "hedge_roster": hedge_roster, "prefer_mail": prefer_mail,
                "on_roster": on_roster, "entity_kind": entity_kind,
                "sort_order": sort_order, "mowing_group": mowing_group,
            }.items():
                if value is not None:
                    row[key] = value
            row["updated_at"] = _now()
            return int(row["id"])
        cid = self._client_id
        self._client_id += 1
        self.clients[cid] = {
            "id": cid,
            "name": name,
            "email": email,
            "phone": phone,
            "address": address,
            "billing_address": billing_address,
            "billing_notes": billing_notes,
            "mow_price": mow_price,
            "hedge_price": hedge_price,
            "hedge_roster": bool(hedge_roster) if hedge_roster is not None else False,
            "prefer_mail": bool(prefer_mail) if prefer_mail is not None else False,
            "on_roster": True if on_roster is None else bool(on_roster),
            "entity_kind": entity_kind or "client",
            "sort_order": sort_order,
            "mowing_group": mowing_group,
            "delivery": "mail" if prefer_mail else None,
            "updated_at": _now(),
        }
        return cid

    def _sync_mail_choice(self, conn, client_id: int, prefer_mail: bool) -> None:
        row = self.clients.get(int(client_id))
        if not row:
            return
        if prefer_mail:
            row["delivery"] = "mail"
            return
        if row.get("delivery") == "mail":
            row["delivery"] = None

    def save_typed_client(
        self, conn, *, client_id: int | None, name: str, address: str | None,
        phone: str | None, email: str | None, billing_notes: str | None,
        mow_price: float | None, hedge_price: float | None, prefer_mail: bool,
        on_roster: bool = True,
    ) -> int:
        name = name.strip()
        if not name:
            raise ValueError("name required")
        other = self.get_client_by_name(conn, name)
        hedge_roster = hedge_price is not None
        if client_id:
            if other and int(other["id"]) != int(client_id):
                raise ValueError(f"{name} is already on file")
            if not self.get_client(conn, int(client_id)):
                raise ValueError(f"Client {client_id} was not found")
            row = self.clients[int(client_id)]
            row.update({
                "name": name, "address": address, "phone": phone, "email": email,
                "billing_notes": billing_notes, "mow_price": mow_price,
                "hedge_price": hedge_price, "hedge_roster": hedge_roster,
                "prefer_mail": bool(prefer_mail), "on_roster": bool(on_roster),
                "updated_at": _now(),
            })
            self._sync_mail_choice(conn, int(client_id), prefer_mail)
            return int(client_id)
        if other:
            return self.save_typed_client(
                conn, client_id=int(other["id"]), name=name, address=address,
                phone=phone, email=email, billing_notes=billing_notes,
                mow_price=mow_price, hedge_price=hedge_price,
                prefer_mail=prefer_mail, on_roster=on_roster,
            )
        new_id = self.upsert_client(
            conn, name=name, email=email, phone=phone, address=address,
            billing_notes=billing_notes, mow_price=mow_price, hedge_price=hedge_price,
            hedge_roster=hedge_roster, prefer_mail=prefer_mail, on_roster=on_roster,
        )
        self._sync_mail_choice(conn, new_id, prefer_mail)
        return new_id

    def knowledge_records_for_sheet(self, conn, sheet_kind: str = "mowing") -> list[dict]:
        rows = []
        for row in self.list_clients(conn):
            if row.get("on_roster") is False:
                continue
            if "smoke" in str(row.get("name") or "").lower():
                continue
            rows.append({
                "name": row["name"],
                "address": row.get("address") or "",
                "billing_address": row.get("billing_address") or "",
                "billing_notes": row.get("billing_notes") or "",
                "phone": row.get("phone") or "",
                "email": row.get("email") or "",
                "prefer_mail": bool(row.get("prefer_mail")),
                "entity_kind": row.get("entity_kind") or "client",
                "mow_price": float(row["mow_price"]) if row.get("mow_price") is not None else None,
                "hedge_price": float(row["hedge_price"]) if row.get("hedge_price") is not None else None,
            })
        return rows

    def knowledge_names_for_sheet(self, conn, sheet_kind: str = "mowing") -> list[str]:
        return [row["name"] for row in self.knowledge_records_for_sheet(conn, sheet_kind)]

    def add_work_item(self, conn, *, client_id: int, month: str, day_or_note=None,
                      description=None, amount=None, source_job_id=None) -> int:
        wid = self._work_id
        self._work_id += 1
        self.work_items[wid] = {
            "id": wid,
            "client_id": int(client_id),
            "month": month,
            "day_or_note": day_or_note,
            "description": description,
            "amount": amount,
            "source_job_id": source_job_id,
            "created_at": _now(),
        }
        return wid

    def _work_with_client(self, work: dict) -> dict:
        client = self.clients.get(int(work["client_id"])) or {}
        row = dict(work)
        row["client_name"] = client.get("name")
        row["email"] = client.get("email")
        row["address"] = client.get("address")
        row["phone"] = client.get("phone")
        row["billing_notes"] = client.get("billing_notes")
        row["mow_price"] = client.get("mow_price")
        row["hedge_price"] = client.get("hedge_price")
        return row

    def work_for_month(self, conn, month: str) -> list[dict]:
        rows = [w for w in self.work_items.values() if w["month"] == month]
        rows.sort(key=lambda row: row["id"])
        return [_copy(self._work_with_client(row)) for row in rows]

    def work_for_client_month(self, conn, month: str, client_id: int) -> list[dict]:
        rows = [
            w for w in self.work_items.values()
            if w["month"] == month and int(w["client_id"]) == int(client_id)
        ]
        rows.sort(key=lambda row: row["id"])
        return [_copy(self._work_with_client(row)) for row in rows]

    def months_with_work(self, conn) -> list[str]:
        months = sorted({w["month"] for w in self.work_items.values()})

        def calendar(month: str) -> bool:
            parts = str(month).split("-")
            return len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 4 and parts[1].isdigit()

        dated = sorted((m for m in months if calendar(m)), reverse=True)
        other = sorted(m for m in months if not calendar(m))
        return dated + other

    def save_upload_job(self, conn, job_id: str, status: str, progress_msg: str = "",
                        extract=None, s3_prefix: str | None = None, month: str | None = None,
                        sheet_kind: str | None = None) -> None:
        row = self.upload_jobs.get(job_id) or {
            "id": job_id,
            "created_at": _now(),
            "extract_json": None,
            "s3_prefix": None,
            "month": None,
            "sheet_kind": None,
        }
        row["status"] = status
        row["progress_msg"] = progress_msg
        if extract is not None:
            row["extract_json"] = deepcopy(extract)
        if s3_prefix is not None:
            row["s3_prefix"] = s3_prefix
        if month is not None:
            row["month"] = month
        if sheet_kind is not None:
            row["sheet_kind"] = sheet_kind
        row["updated_at"] = _now()
        self.upload_jobs[job_id] = row

    def save_review_draft(self, conn, job_id: str, extract: dict, month: str | None,
                          sheet_kind: str | None) -> dict | None:
        row = self.upload_jobs.get(job_id)
        if not row or row.get("status") not in ("done", "stored"):
            return None
        row["extract_json"] = deepcopy(extract)
        if month:
            row["month"] = month
        if sheet_kind:
            row["sheet_kind"] = sheet_kind
        row["updated_at"] = _now()
        return {"id": row["id"], "month": row.get("month"), "sheet_kind": row.get("sheet_kind"),
                "updated_at": row["updated_at"]}

    def clear_review(self, conn, job_id: str) -> dict | None:
        row = self.upload_jobs.get(job_id)
        if not row or row.get("status") not in ("done", "stored"):
            return None
        work_items = 0
        if row.get("status") == "stored":
            month = (row.get("month") or "").strip()
            if month:
                work_items = self.delete_work_from_job(conn, month, job_id)
        row["status"] = "cleared"
        row["progress_msg"] = "Review cleared"
        row["updated_at"] = _now()
        return {"id": row["id"], "status": row["status"], "updated_at": row["updated_at"],
                "work_items": work_items}

    def get_upload_job(self, conn, job_id: str) -> dict | None:
        return _copy(self.upload_jobs.get(job_id))

    def list_editable_jobs(self, conn) -> list[dict]:
        rows = [
            row for row in self.upload_jobs.values()
            if row.get("extract_json") is not None and row.get("status") in ("done", "stored")
        ]
        rows.sort(key=lambda row: (row.get("updated_at") or row["created_at"]), reverse=True)
        return [_copy(row) for row in rows[:24]]

    def latest_upload_status(self, conn, month: str | None = None) -> str | None:
        rows = list(self.upload_jobs.values())
        if month:
            rows = [row for row in rows if row.get("month") == month]
        if not rows:
            return None
        rows.sort(key=lambda row: (row.get("updated_at") or row["created_at"]), reverse=True)
        return rows[0].get("status")

    def latest_calendar_upload_month(self, conn) -> str | None:
        rows = [
            row for row in self.upload_jobs.values()
            if row.get("month") and len(str(row["month"]).split("-")) == 2
            and str(row["month"]).split("-")[0].isdigit()
        ]
        if not rows:
            return None
        rows.sort(key=lambda row: (row.get("updated_at") or row["created_at"]), reverse=True)
        return rows[0].get("month")

    def clients_updated_at(self, conn):
        if not self.clients:
            return None
        return max(row["updated_at"] for row in self.clients.values())

    def month_bills_rev(self, conn, month: str) -> str:
        bills = [b for b in self.bills.values() if b["month"] == month]
        n = len(bills)
        max_id = max((b["id"] for b in bills), default=0)
        emailed = [b.get("emailed_at") for b in bills if b.get("emailed_at")]
        last = max(emailed) if emailed else None
        return f"{n}:{max_id}:{last}"

    def month_work_count(self, conn, month: str) -> int:
        return sum(1 for w in self.work_items.values() if w["month"] == month)

    def job_updated_at(self, conn, job_id: str):
        row = self.upload_jobs.get(job_id)
        return None if not row else row.get("updated_at")

    def set_bill_letter(self, conn, bill_id: int, *, note: str | None, greeting: str | None,
                        closing: str | None, signoff: str | None) -> None:
        row = self.bills.get(int(bill_id))
        if not row:
            return
        row["cover_note"] = note or None
        row["greeting"] = greeting or None
        row["closing"] = closing or None
        row["signoff"] = signoff or None

    def bill_letter(self, conn, month: str, client_id: int) -> dict:
        bill = self.bill_for_client_month(conn, month, client_id)
        if not bill:
            return {}
        return {
            "cover_note": bill.get("cover_note"),
            "greeting": bill.get("greeting"),
            "closing": bill.get("closing"),
            "signoff": bill.get("signoff"),
            "display_name": bill.get("display_name"),
        }

    def set_bill_face(self, conn, bill_id: int, *, display_name: str | None,
                      bill_email: str | None, bill_delivery: str | None) -> None:
        row = self.bills.get(int(bill_id))
        if not row:
            return
        row["display_name"] = display_name
        row["bill_email"] = bill_email
        row["bill_delivery"] = bill_delivery

    def save_bill(self, conn, *, month: str, client_id: int, s3_key: str) -> int:
        bid = self._bill_id
        self._bill_id += 1
        self.bills[bid] = {
            "id": bid, "month": month, "client_id": int(client_id), "s3_key": s3_key,
            "cover_note": None, "greeting": None, "closing": None, "signoff": None,
            "display_name": None, "bill_email": None, "bill_delivery": None,
            "emailed_at": None, "created_at": _now(),
        }
        return bid

    def bill_for_client_month(self, conn, month: str, client_id: int) -> dict | None:
        rows = [
            b for b in self.bills.values()
            if b["month"] == month and int(b["client_id"]) == int(client_id)
        ]
        if not rows:
            return None
        rows.sort(key=lambda row: row["id"], reverse=True)
        return _copy(rows[0])

    def bills_for_month(self, conn, month: str) -> list[dict]:
        from app.billing import _retarget_month_clients, delivery_channel, face_bill

        _retarget_month_clients(conn, month)
        rows = [b for b in self.bills.values() if b["month"] == month]

        def work_order(bill: dict) -> tuple:
            ids = [
                w["id"] for w in self.work_items.values()
                if int(w["client_id"]) == int(bill["client_id"]) and w["month"] == bill["month"]
            ]
            return (min(ids) if ids else 10**9, bill["id"])

        rows.sort(key=work_order)
        faced = []
        for bill in rows:
            client = self.clients.get(int(bill["client_id"])) or {}
            item = dict(bill)
            item["client_name"] = client.get("name")
            item["email"] = client.get("email")
            item["phone"] = client.get("phone")
            item["billing_notes"] = client.get("billing_notes")
            item["prefer_mail"] = client.get("prefer_mail")
            item["delivery"] = client.get("delivery")
            item["address"] = client.get("address")
            item["mow_price"] = client.get("mow_price")
            item["hedge_price"] = client.get("hedge_price")
            roster_name = item.get("client_name") or ""
            roster_email = item.get("email") or ""
            roster_delivery = delivery_channel(item)
            shown = face_bill(item)
            shown["roster_name"] = roster_name
            shown["roster_email"] = roster_email
            shown["roster_delivery"] = roster_delivery
            faced.append(shown)
        return faced

    def update_client_contact(self, conn, client_id: int, *, email=None, address=None,
                              name=None, phone=None, delivery=None, prefer_mail=False) -> None:
        row = self.clients.get(int(client_id))
        if not row:
            return
        row["email"] = email or None
        row["address"] = address or None
        row["phone"] = phone or None
        row["delivery"] = delivery or None
        row["prefer_mail"] = bool(prefer_mail)
        if name is not None:
            row["name"] = name
        row["updated_at"] = _now()

    def patch_client(self, conn, client_id: int, *, name=None, email=None, phone=None,
                     address=None, billing_notes=None, mow_price=None, hedge_price=None) -> None:
        row = self.clients.get(int(client_id))
        if not row:
            raise ValueError("client not found")

        def text(new, old):
            if new is None:
                return old
            s = str(new).strip()
            return s or None

        def num(new, old):
            if new is None:
                return old
            if str(new).strip() == "":
                return None
            return float(new)

        if name is not None:
            row["name"] = text(name, row["name"])
        if email is not None:
            row["email"] = text(email, row["email"])
        if phone is not None:
            row["phone"] = text(phone, row["phone"])
        if address is not None:
            row["address"] = text(address, row["address"])
        if billing_notes is not None:
            row["billing_notes"] = text(billing_notes, row["billing_notes"])
        if mow_price is not None:
            row["mow_price"] = num(mow_price, row["mow_price"])
        if hedge_price is not None:
            row["hedge_price"] = num(hedge_price, row["hedge_price"])
        row["updated_at"] = _now()

    def update_work_item(self, conn, work_id: int, *, description: str, amount, day_or_note=None) -> None:
        row = self.work_items.get(int(work_id))
        if not row:
            return
        row["description"] = description
        row["amount"] = amount
        if day_or_note is not None:
            row["day_or_note"] = day_or_note

    def patch_work_item(self, conn, work_id: int, *, description=None, amount=None, month=None) -> None:
        row = self.work_items.get(int(work_id))
        if not row:
            raise ValueError("work item not found")
        if description is not None:
            row["description"] = str(description)
        if amount is not None:
            row["amount"] = None if str(amount).strip() == "" else float(amount)
        if month is not None:
            row["month"] = str(month).strip()

    def delete_work_item(self, conn, work_id: int) -> None:
        self.work_items.pop(int(work_id), None)

    def delete_work_from_job(self, conn, month: str, source_job_id: str) -> int:
        drop = [
            wid for wid, row in self.work_items.items()
            if row["month"] == month and row.get("source_job_id") == source_job_id
        ]
        for wid in drop:
            self.work_items.pop(wid, None)
        return len(drop)

    def delete_sourced_month_work(self, conn, month: str) -> int:
        drop = [
            wid for wid, row in self.work_items.items()
            if row["month"] == month and row.get("source_job_id")
        ]
        for wid in drop:
            self.work_items.pop(wid, None)
        return len(drop)

    def delete_client_month(self, conn, month: str, client_id: int) -> list[str]:
        keys = []
        drop_bills = [
            bid for bid, row in self.bills.items()
            if row["month"] == month and int(row["client_id"]) == int(client_id)
        ]
        for bid in drop_bills:
            key = self.bills.pop(bid).get("s3_key")
            if key:
                keys.append(key)
        drop_work = [
            wid for wid, row in self.work_items.items()
            if row["month"] == month and int(row["client_id"]) == int(client_id)
        ]
        for wid in drop_work:
            self.work_items.pop(wid, None)
        return keys

    def delete_client(self, conn, client_id: int) -> list[str]:
        keys = [
            row["s3_key"] for row in self.bills.values()
            if int(row["client_id"]) == int(client_id) and row.get("s3_key")
        ]
        for bid in [bid for bid, row in self.bills.items() if int(row["client_id"]) == int(client_id)]:
            self.bills.pop(bid, None)
        for wid in [wid for wid, row in self.work_items.items() if int(row["client_id"]) == int(client_id)]:
            self.work_items.pop(wid, None)
        self.clients.pop(int(client_id), None)
        return keys

    def set_client_email(self, conn, client_id: int, email: str | None) -> None:
        row = self.clients.get(int(client_id))
        if row:
            row["email"] = email or None
            row["updated_at"] = _now()

    def delete_month_bills(self, conn, month: str) -> list[str]:
        month = (month or "").strip()
        keys = []
        for bid in [bid for bid, row in self.bills.items() if row["month"] == month]:
            key = self.bills.pop(bid).get("s3_key")
            if key:
                keys.append(key)
        return keys

    def delete_month_work(self, conn, month: str) -> dict:
        month = (month or "").strip()
        keys = self.delete_month_bills(conn, month)
        drop = [wid for wid, row in self.work_items.items() if row["month"] == month]
        for wid in drop:
            self.work_items.pop(wid, None)
        return {"month": month, "work_items": len(drop), "bills": len(keys), "s3_keys": keys}

    def reassign_month_client(self, conn, month: str, source_id: int, target_id: int) -> None:
        if int(source_id) == int(target_id):
            return
        for row in self.work_items.values():
            if row["month"] == month and int(row["client_id"]) == int(source_id):
                row["client_id"] = int(target_id)
        target_bill = self.bill_for_client_month(conn, month, target_id)
        if target_bill:
            for bid in [
                bid for bid, row in self.bills.items()
                if row["month"] == month and int(row["client_id"]) == int(source_id)
            ]:
                self.bills.pop(bid, None)
            return
        for row in self.bills.values():
            if row["month"] == month and int(row["client_id"]) == int(source_id):
                row["client_id"] = int(target_id)

    def upsert_bill(self, conn, *, month: str, client_id: int, s3_key: str) -> int:
        existing = self.bill_for_client_month(conn, month, client_id)
        if existing:
            row = self.bills[int(existing["id"])]
            row["s3_key"] = s3_key
            row["emailed_at"] = None
            return int(existing["id"])
        return self.save_bill(conn, month=month, client_id=client_id, s3_key=s3_key)

    def mark_bill_emailed(self, conn, bill_id: int) -> None:
        row = self.bills.get(int(bill_id))
        if row:
            row["emailed_at"] = _now()

    def month_inventory(self, conn, month: str) -> dict:
        jobs: dict[str, int] = {}
        for row in self.upload_jobs.values():
            if row.get("month") != month:
                continue
            status = str(row.get("status") or "")
            jobs[status] = jobs.get(status, 0) + 1
        return {
            "month": month,
            "work": self.month_work_count(conn, month),
            "bills": sum(1 for b in self.bills.values() if b["month"] == month),
            "jobs": jobs,
            "test_work": self.month_work_count(conn, "test_month"),
        }

    def chat_snapshot_rows(self, conn, *, work_limit: int = 200, bill_limit: int = 80) -> dict:
        work = [_copy(self._work_with_client(w)) for w in self.work_items.values()]
        work.sort(key=lambda row: (row["month"], row.get("client_name") or "", row["id"]), reverse=True)
        bills = []
        for bill in self.bills.values():
            client = self.clients.get(int(bill["client_id"])) or {}
            bills.append({
                "id": bill["id"],
                "month": bill["month"],
                "client_id": bill["client_id"],
                "client_name": client.get("name"),
                "emailed": bill.get("emailed_at") is not None,
            })
        bills.sort(key=lambda row: (row["month"], row.get("client_name") or ""), reverse=True)
        month_counts: dict[str, int] = {}
        for row in self.work_items.values():
            month_counts[str(row["month"])] = month_counts.get(str(row["month"]), 0) + 1
        return {
            "work": work[:work_limit],
            "bills": bills[:bill_limit],
            "month_counts": month_counts,
        }

    def client_has_work(self, conn, client_id: int) -> bool:
        return any(int(row["client_id"]) == int(client_id) for row in self.work_items.values())


DB_FUNCS = (
    "connect",
    "upsert_client",
    "get_client_by_name",
    "save_typed_client",
    "list_clients",
    "knowledge_records_for_sheet",
    "knowledge_names_for_sheet",
    "add_work_item",
    "work_for_month",
    "months_with_work",
    "save_upload_job",
    "save_review_draft",
    "clear_review",
    "get_upload_job",
    "set_bill_letter",
    "bill_letter",
    "set_bill_face",
    "save_bill",
    "bills_for_month",
    "work_for_client_month",
    "get_client",
    "update_client_contact",
    "patch_client",
    "update_work_item",
    "patch_work_item",
    "delete_work_item",
    "delete_work_from_job",
    "delete_sourced_month_work",
    "delete_client_month",
    "delete_client",
    "set_client_email",
    "delete_month_bills",
    "delete_month_work",
    "reassign_month_client",
    "upsert_bill",
    "mark_bill_emailed",
    "month_inventory",
    "chat_snapshot_rows",
    "list_editable_jobs",
    "latest_upload_status",
    "latest_calendar_upload_month",
    "clients_updated_at",
    "month_bills_rev",
    "month_work_count",
    "bill_for_client_month",
    "client_has_work",
    "job_updated_at",
)

