#!/usr/bin/env python3
"""One-shot: ensure DB exists, init schema, optional S3 ping."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import psycopg

ADMIN = os.environ.get(
    "DATABASE_ADMIN_URL",
    "postgresql://postgres@10.0.220.145:5433/postgres",
)
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres@10.0.220.145:5433/nicks_billing",
)


def main() -> None:
    dbname = "nicks_billing"
    with psycopg.connect(ADMIN, autocommit=True) as conn:
        row = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
        ).fetchone()
        if not row:
            conn.execute(f'CREATE DATABASE "{dbname}"')
            print(f"created database {dbname}")
        else:
            print(f"database {dbname} exists")

    from app import db, storage, config

    db.init_db()
    print("schema ok")
    if config.S3_ACCESS_KEY:
        try:
            storage.ensure_bucket()
            storage.put_bytes(b"ok", "_smoke/ping.txt", content_type="text/plain")
            assert storage.get_bytes("_smoke/ping.txt") == b"ok"
            print(f"s3 ok bucket={config.S3_BUCKET}")
        except Exception as e:
            print(f"s3 FAIL: {e}")
            sys.exit(1)
    else:
        print("s3 skipped (no S3_ACCESS_KEY)")


if __name__ == "__main__":
    main()
