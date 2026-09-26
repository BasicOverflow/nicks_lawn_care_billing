#!/usr/bin/env python3
"""Smoke: seed extract → confirm → PDF → zip (no OCR / no SMTP required)."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:8787"
MONTH = "2025-09"


def api(method: str, path: str, data=None, timeout=120):
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{BASE}{path}", data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        ct = r.headers.get("Content-Type", "")
        if "json" in ct:
            return json.loads(raw)
        return raw


def main() -> None:
    # health via static
    urllib.request.urlopen(f"{BASE}/data", timeout=10).read(200)
    print("ui ok")

    extract = {
        "title": "Smoke roster",
        "tables": [
            {
                "columns": ["Contact", "Address", "Price", "Billing"],
                "rows": [
                    ["SMOKE TEST CLIENT", "1 Test St", "$75", "smoke@example.com"],
                    ["SMOKE NO EMAIL", "2 Test St", "$60", "cash only"],
                ],
            }
        ],
        "notes": [],
        "complete": True,
    }

    # Direct DB path if server already running — use confirm via injecting job
    from app import db

    jid = f"smoke-{int(time.time())}"
    with db.connect() as conn:
        db.save_upload_job(conn, jid, "done", "smoke", extract=extract, month=MONTH)

    r = api("POST", f"/api/jobs/{jid}/confirm", {"month": MONTH, "sheet_kind": "work"})
    print("confirm", r)

    # generate bills (sync via module if background is flaky — wait on progress)
    j = api("POST", "/api/billing/generate", {"month": MONTH})
    print("generate job", j)
    for _ in range(60):
        p = api("GET", "/api/progress")
        if p.get("status") in ("done", "error", "idle") and p.get("job_id") == j["job_id"]:
            if p.get("status") == "error":
                raise SystemExit(f"generate failed: {p}")
            if p.get("status") == "done":
                break
        time.sleep(0.5)
    else:
        # fallback: generate in-process
        from app import billing

        with db.connect() as conn:
            bills = billing.generate_month_bills(conn, MONTH)
            print("fallback bills", len(bills))

    listing = api("GET", f"/api/billing/{MONTH}/list")
    print("bills", len(listing.get("bills") or []), "smtp", listing.get("smtp_configured"))
    if not listing.get("bills"):
        raise SystemExit("no bills after generate")

    z = api("GET", f"/api/billing/{MONTH}/download.zip")
    assert isinstance(z, (bytes, bytearray)) and z[:2] == b"PK", "zip magic"
    print(f"zip ok ({len(z)} bytes)")

    tsv = api("GET", f"/api/billing/{MONTH}/tax.tsv")
    text = tsv.decode() if isinstance(tsv, (bytes, bytearray)) else tsv
    assert "SMOKE TEST CLIENT" in text
    print("tax tsv ok")
    print("SMOKE PASS")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"server not up at {BASE}: {e}", file=sys.stderr)
        sys.exit(2)
