#!/usr/bin/env python3
"""Seed Postgres clients from the paper-office files in .tmp.

Sources (all things Nick already has before a photo is taken):
  - Client List_CY2026_Price Increase.xlsx  (mow prices + email/mail notes)
  - hedgesclientlist.doc                    (who is on the hedges roster)
  - GNBAClientPhoneList.doc                 (phones)
  - Clients Mailing Labels.docx             (billing / winter addresses)
  - RevisedWorkCompleted.docx               (canonical row order)
  - Lawngroups.doc                          (mowing route groups)

Ground-truth JSON under ground_truth/ is for scoring tests only. This script
does not read it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from docx import Document
from openpyxl import load_workbook

from app import db

TMP = ROOT / ".tmp"
HEDGES_TEXT = TMP / "_extracted" / "hedgesclientlist.clean.txt"

PARCEL_PREFIXES = (
    "R.O.W.",
    "WEST BENCHES",
    "EAST BENCHES",
    "CLUB HOUSE",
    "FIELD",
    "FLAG",
    "COTTAGE",
    "LOWER POND",
    "UPPER POND",
    "CAUSEWAY",
    "GNBA",
)


def _norm_key(name: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    return s


def _entity_kind(name: str) -> str:
    u = (name or "").upper()
    if "GIANTS NECK BEACH" in u or u.startswith("GNBA"):
        return "association"
    if any(u.startswith(p) for p in PARCEL_PREFIXES):
        return "parcel"
    return "client"


def _parse_billing_notes(raw: str) -> tuple[str | None, str | None, bool, str]:
    """Return email, phone-ish, prefer_mail, cleaned notes."""
    notes = (raw or "").strip()
    if not notes:
        return None, None, False, ""
    prefer_mail = "REGULAR MAIL" in notes.upper()
    email = None
    phone = None
    m = re.search(r"[\w.+-]+@[\w.-]+\.\w+", notes)
    if m:
        email = m.group(0)
    # bare phone in notes (e.g. Gleason 860-961-5457)
    pm = re.search(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)", notes)
    if pm and "@" not in notes[max(0, pm.start() - 2) : pm.end() + 2]:
        phone = re.sub(r"[^\d+]", "", pm.group(0))
        if phone.startswith("1") and len(phone) == 11:
            phone = phone[1:]
        if len(phone) == 10:
            phone = f"{phone[:3]}-{phone[3:6]}-{phone[6:]}"
        else:
            phone = pm.group(0).strip()
    return email, phone, prefer_mail, notes


def load_cy2026() -> list[dict]:
    path = TMP / "Client List_CY2026_Price Increase.xlsx"
    wb = load_workbook(str(path), data_only=True)
    ws = wb["CY2026 Client List Data"]
    rows = list(ws.iter_rows(values_only=True))
    out: list[dict] = []
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        name = str(row[0]).strip()
        if name.upper().startswith("REGULAR MAIL"):
            continue
        address = str(row[1] or "").strip() or None
        new_price = row[4]
        try:
            mow = float(new_price) if new_price not in (None, "") else None
        except (TypeError, ValueError):
            mow = None
        if mow == 0:
            mow = None
        email, phone, prefer_mail, notes = _parse_billing_notes(str(row[5] or ""))
        out.append(
            {
                "name": name,
                "address": address,
                "mow_price": mow,
                "email": email,
                "phone": phone,
                "prefer_mail": prefer_mail,
                "billing_notes": notes or None,
                "entity_kind": _entity_kind(name),
            }
        )
    return out


def load_hedges_office() -> list[dict]:
    """Hedges roster from the Word template extract, not from a scored photo."""
    if not HEDGES_TEXT.is_file():
        return []
    lines = [
        ln.strip()
        for ln in HEDGES_TEXT.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    start = 0
    for i, ln in enumerate(lines):
        if ln.lower() == "notes" and i >= 1:
            start = i + 1
            break
    name_re = re.compile(r"^[A-Z][A-Z' .\-]+,\s+\S")
    price_re = re.compile(r"^\$?\d+(?:\.\d+)?(?:\s*/\s*\$?\d+(?:\.\d+)?)?$")
    addr_re = re.compile(
        r"\d|Rd\.?|Road|Ln\.?|Lane|St\.?|Street|Ave|Court|Ct\.?",
        re.I,
    )

    def is_name(s: str) -> bool:
        if s.upper().startswith("HEDGES CLIENT") or s.startswith("Julie Cameron"):
            return False
        return bool(name_re.match(s)) and not price_re.match(s)

    out: list[dict] = []
    i = start
    while i < len(lines):
        if lines[i].startswith("Julie Cameron") or lines[i].upper().startswith(
            "HEDGES CLIENT"
        ):
            break
        if not is_name(lines[i]):
            i += 1
            continue
        name = lines[i]
        i += 1
        address = None
        prices: list[str] = []
        notes: list[str] = []
        while i < len(lines) and not is_name(lines[i]):
            if lines[i].startswith("Julie Cameron") or lines[i].upper().startswith(
                "HEDGES CLIENT"
            ):
                break
            ln = lines[i]
            if price_re.match(ln):
                prices.append(ln)
            elif address is None and addr_re.search(ln) and len(ln) < 80:
                address = ln
            else:
                notes.append(ln)
            i += 1
        hedge_price = None
        if len(prices) == 1 and "/" not in prices[0]:
            m = re.search(r"\d+(?:\.\d+)?", prices[0])
            if m:
                hedge_price = float(m.group(0))
        elif prices:
            notes.append("hedge cell lists " + " and ".join(prices))
        out.append(
            {
                "name": name,
                "address": address,
                "hedge_price": hedge_price,
                "billing_notes": " ".join(notes) or None,
                "hedge_roster": True,
                "entity_kind": "client",
            }
        )
    return out


def load_work_order() -> dict[str, int]:
    path = TMP / "RevisedWorkCompleted.docx"
    d = Document(str(path))
    order: dict[str, int] = {}
    if not d.tables:
        return order
    for i, row in enumerate(d.tables[0].rows[1:]):
        name = row.cells[0].text.strip()
        if not name:
            continue
        order[_norm_key(name)] = i + 1
        order[_norm_key(name.split("(")[0])] = i + 1
    return order


def load_mailing_labels() -> dict[str, str]:
    """Map rough name key → multi-line billing address from Avery labels."""
    path = TMP / "Clients Mailing Labels.docx"
    d = Document(str(path))
    by_key: dict[str, str] = {}
    if not d.tables:
        return by_key
    for row in d.tables[0].rows:
        for cell in row.cells:
            text = cell.text.strip()
            if not text:
                continue
            # labels often use " | " between lines in our extract; real docx uses newlines
            lines = [ln.strip() for ln in re.split(r"\n+| \| ", text) if ln.strip()]
            if len(lines) < 2:
                continue
            name = lines[0]
            addr = "\n".join(lines[1:])
            by_key[_norm_key(name)] = addr
            # also surname-only
            if "," in name:
                sk = _norm_key(name.split(",")[0])
                by_key.setdefault(sk, addr)
            else:
                parts = name.split()
                if parts:
                    by_key[_norm_key(parts[-1])] = addr
    return by_key


def load_phones_from_ole() -> dict[str, str]:
    """Best-effort phone name→number from GNBAClientPhoneList.doc via ole strings."""
    path = TMP / "GNBAClientPhoneList.doc"
    if not path.is_file():
        return {}
    try:
        import olefile
    except ImportError:
        return {}
    ole = olefile.OleFileIO(str(path))
    data = b""
    for stream in ("WordDocument", "1Table", "0Table"):
        if ole.exists(stream):
            data += ole.openstream(stream).read()
    ole.close()
    chunks: list[str] = []
    for m in re.finditer(rb"(?:[\x20-\x7e]\x00){3,}", data):
        try:
            chunks.append(m.group().decode("utf-16-le"))
        except Exception:
            pass
    # Pair Name\nPhone patterns from cleaned extract we already know
    text_path = TMP / "_extracted" / "GNBAClientPhoneList.clean.txt"
    if text_path.is_file():
        lines = [
            ln.strip()
            for ln in text_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    else:
        lines = [re.sub(r"\s+", " ", c).strip() for c in chunks if len(c.strip()) > 2]

    phones: dict[str, str] = {}
    phone_re = re.compile(r"^\d{3}[-.\s]?\d{3}[-.\s]?\d{4}$")
    i = 0
    while i < len(lines) - 1:
        a, b = lines[i], lines[i + 1]
        if phone_re.match(b) and not phone_re.match(a) and len(a) < 60:
            if a.lower() not in ("steve", "steve colonis", "may 3, 2010"):
                phones[_norm_key(a)] = b
                # also last-token key
                tok = a.split()[-1]
                phones.setdefault(_norm_key(tok), b)
            i += 2
            continue
        i += 1
    return phones


def load_lawngroups() -> dict[str, str]:
    """Map name fragment → group label from Lawngroups.doc clean extract."""
    text_path = TMP / "_extracted" / "Lawngroups.clean.txt"
    if not text_path.is_file():
        # regenerate quickly via ole if needed
        return {}
    lines = [
        ln.strip()
        for ln in text_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    # After author metadata, names until cadence notes / MOWING GROUPS
    start = 0
    for i, ln in enumerate(lines):
        if ln.lower() in ("bernardi", "cottage") or (
            ln[0].isalpha() and "steve" not in ln.lower() and i > 2
        ):
            if ln.lower() not in ("steve", "steve colonis", "may 3, 2010"):
                start = i
                break
    names: list[str] = []
    for ln in lines[start:]:
        if ln.upper().startswith("MOWING GROUPS"):
            break
        if ":" in ln and any(
            k in ln.lower() for k in ("every", "son mows", "days")
        ):
            continue
        if len(ln) > 40:
            continue
        if ln.lower().startswith(("steve", "may ")):
            continue
        names.append(ln)
    # Document doesn't label group numbers clearly in OLE extract — store
    # sequential group buckets of ~10 as soft hints.
    out: dict[str, str] = {}
    bucket = 10
    for i, name in enumerate(names):
        g = f"group-{(i // bucket) + 1}"
        out[_norm_key(name)] = g
    return out


def _lookup(mapping: dict[str, str], name: str) -> str | None:
    k = _norm_key(name)
    if k in mapping:
        return mapping[k]
    # try surname before comma (stored only when that surname was unused)
    if "," in name:
        k2 = _norm_key(name.split(",")[0])
        if k2 in mapping:
            return mapping[k2]
    return None


def merge_records() -> list[dict]:
    by_key: dict[str, dict] = {}

    def upsert(rec: dict) -> None:
        name = rec["name"].strip()
        k = _norm_key(name)
        if not k:
            return
        if k not in by_key:
            by_key[k] = {"name": name}
        cur = by_key[k]
        for field, val in rec.items():
            if field == "name":
                # Prefer longer / comma form
                if val and (len(str(val)) > len(str(cur.get("name") or "")) or (
                    "," in str(val) and "," not in str(cur.get("name") or "")
                )):
                    cur["name"] = val
                continue
            if val is None or val == "":
                continue
            if field == "billing_notes" and cur.get("billing_notes") and val != cur["billing_notes"]:
                if str(val) not in str(cur["billing_notes"]):
                    cur["billing_notes"] = f"{cur['billing_notes']} | {val}"
                continue
            if field not in cur or cur[field] in (None, ""):
                cur[field] = val

    for r in load_cy2026():
        upsert(r)
    for r in load_hedges_office():
        upsert(r)

    order = load_work_order()
    phones = load_phones_from_ole()
    mail = load_mailing_labels()
    groups = load_lawngroups()

    for k, rec in by_key.items():
        name = rec["name"]
        so = order.get(_norm_key(name)) or order.get(_norm_key(name.split("(")[0]))
        if so:
            rec["sort_order"] = so
        ph = _lookup(phones, name)
        if ph:
            rec["phone"] = ph
        ba = _lookup(mail, name)
        if ba:
            rec["billing_address"] = ba
        g = _lookup(groups, name)
        if g:
            rec["mowing_group"] = g
        rec.setdefault("entity_kind", _entity_kind(name))
        rec.setdefault("hedge_roster", False)

    # Add any work-order-only names not in CY2026 (e.g. Heenahan, Shoemaker)
    path = TMP / "RevisedWorkCompleted.docx"
    d = Document(str(path))
    for i, row in enumerate(d.tables[0].rows[1:]):
        name = row.cells[0].text.strip()
        if not name:
            continue
        k = _norm_key(name)
        if k not in by_key:
            upsert(
                {
                    "name": name,
                    "sort_order": i + 1,
                    "entity_kind": _entity_kind(name),
                }
            )
        elif "sort_order" not in by_key[k]:
            by_key[k]["sort_order"] = i + 1

    return sorted(
        by_key.values(),
        key=lambda r: (r.get("sort_order") is None, r.get("sort_order") or 0, r["name"]),
    )


def _same_person(a: str, b: str) -> bool:
    ka, kb = _norm_key(a), _norm_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    short, long = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
    return len(short) >= 8 and long.startswith(short)


def main() -> None:
    db.init_db()
    records = merge_records()
    print(f"parsed {len(records)} knowledge rows from office files")
    hedges = [r for r in records if r.get("hedge_roster")]
    print(f"hedges roster from template: {len(hedges)}")
    for r in hedges:
        print(
            f"  hedge {r['name']}: price={r.get('hedge_price')} "
            f"notes={r.get('billing_notes')!r}"
        )
    with db.connect() as conn:
        existing = db.list_clients(conn)
        used: set[int] = set()

        def find_client(name: str) -> dict | None:
            exact = [
                row
                for row in existing
                if int(row["id"]) not in used
                and _norm_key(row["name"]) == _norm_key(name)
            ]
            if exact:
                return exact[0]
            pref = [
                row
                for row in existing
                if int(row["id"]) not in used and _same_person(row["name"], name)
            ]
            return pref[0] if pref else None

        for r in records:
            match = find_client(r["name"])
            if match is None:
                cid = db.upsert_client(
                    conn,
                    name=r["name"],
                    email=r.get("email"),
                    phone=r.get("phone"),
                    address=r.get("address"),
                    billing_address=r.get("billing_address"),
                    billing_notes=r.get("billing_notes"),
                    mow_price=r.get("mow_price"),
                    hedge_price=r.get("hedge_price"),
                    hedge_roster=bool(r.get("hedge_roster")),
                    prefer_mail=r.get("prefer_mail"),
                    entity_kind=r.get("entity_kind"),
                    sort_order=r.get("sort_order"),
                    mowing_group=r.get("mowing_group"),
                )
                used.add(cid)
                continue
            used.add(int(match["id"]))
            db.upsert_client(
                conn,
                name=match["name"],
                email=r.get("email"),
                phone=r.get("phone"),
                address=r.get("address"),
                billing_address=r.get("billing_address"),
                mow_price=r.get("mow_price"),
                prefer_mail=r.get("prefer_mail"),
                entity_kind=r.get("entity_kind"),
                sort_order=r.get("sort_order"),
                mowing_group=r.get("mowing_group"),
            )
            office_name = None
            if _norm_key(r["name"]) != _norm_key(match["name"]):
                taken = any(
                    _norm_key(row["name"]) == _norm_key(r["name"]) for row in existing
                )
                if not taken:
                    office_name = r["name"]
            db.write_office_knowledge(
                conn,
                int(match["id"]),
                name=office_name,
                billing_notes=r.get("billing_notes"),
                hedge_price=r.get("hedge_price") if r.get("hedge_roster") else None,
                hedge_roster=bool(r.get("hedge_roster")),
            )
        for row in existing:
            if int(row["id"]) in used:
                continue
            notes = row.get("billing_notes") or ""
            low = notes.lower()
            if any(p in low for p in ("handwritten row", "crossed out", "printed $")):
                notes = None
            db.write_office_knowledge(
                conn,
                int(row["id"]),
                billing_notes=notes or None,
                hedge_price=None,
                hedge_roster=False,
            )
            print(f"left in place (not in office files): {row['name']}")
        total = conn.execute("SELECT count(*) AS n FROM clients").fetchone()["n"]
        with_mow = conn.execute(
            "SELECT count(*) AS n FROM clients WHERE mow_price IS NOT NULL"
        ).fetchone()["n"]
        with_hedge = conn.execute(
            "SELECT count(*) AS n FROM clients WHERE hedge_price IS NOT NULL"
        ).fetchone()["n"]
        on_roster = conn.execute(
            "SELECT count(*) AS n FROM clients WHERE hedge_roster"
        ).fetchone()["n"]
        with_phone = conn.execute(
            "SELECT count(*) AS n FROM clients WHERE phone IS NOT NULL"
        ).fetchone()["n"]
        with_bill = conn.execute(
            "SELECT count(*) AS n FROM clients WHERE billing_address IS NOT NULL"
        ).fetchone()["n"]
    print(
        f"upserted office rows={len(records)}; total={total} mow={with_mow} "
        f"hedge_price={with_hedge} hedge_roster={on_roster} "
        f"phone={with_phone} billing_addr={with_bill}"
    )


if __name__ == "__main__":
    main()
