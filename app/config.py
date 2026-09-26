"""App config from environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres@10.0.220.145:5433/nicks_billing",
)
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://10.0.220.145:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
S3_BUCKET = os.environ.get("S3_BUCKET", "nicks-lawn-billing")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")

SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "")
SMTP_TLS = os.environ.get("SMTP_TLS", "1").strip() in ("1", "true", "yes")

TMP_DIR = ROOT / ".tmp"
TMP_DIR.mkdir(exist_ok=True)

COMPANY_NAME = os.environ.get("COMPANY_NAME", "Nick's Lawn Care LLC")
HOUR_RATE = float(os.environ.get("HOUR_RATE", "100"))
CT_TAX_NOTE = "CT sales tax ~6.35% (apply as needed for taxable items)"
