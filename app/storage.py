"""MinIO / S3 helpers (path-style)."""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.client import Config

from . import config


def client():
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        aws_access_key_id=config.S3_ACCESS_KEY or None,
        aws_secret_access_key=config.S3_SECRET_KEY or None,
        region_name=config.S3_REGION,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def ensure_bucket() -> None:
    c = client()
    try:
        c.head_bucket(Bucket=config.S3_BUCKET)
    except Exception:
        c.create_bucket(Bucket=config.S3_BUCKET)


def put_file(local_path: Path, key: str, content_type: str | None = None) -> str:
    kwargs = {}
    if content_type:
        kwargs["ExtraArgs"] = {"ContentType": content_type}
    client().upload_file(str(local_path), config.S3_BUCKET, key, **kwargs)
    return key


def put_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    client().put_object(Bucket=config.S3_BUCKET, Key=key, Body=data, ContentType=content_type)
    return key


def download_to(key: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    client().download_file(config.S3_BUCKET, key, str(dest))
    return dest


def delete_key(key: str) -> None:
    try:
        client().delete_object(Bucket=config.S3_BUCKET, Key=key)
    except Exception:
        return


def get_bytes(key: str) -> bytes:
    obj = client().get_object(Bucket=config.S3_BUCKET, Key=key)
    return obj["Body"].read()
