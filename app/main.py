"""FastAPI entry: static UI + API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import config, db, storage
from .routes import router

STATIC = Path(__file__).resolve().parents[1] / "static"

app = FastAPI(title="Nick's Lawn Care Billing", version="1.0.0")
app.include_router(router)


@app.on_event("startup")
def startup():
    try:
        db.init_db()
    except Exception as e:
        print(f"WARNING: db init failed: {e}", flush=True)
    try:
        if config.S3_ACCESS_KEY:
            storage.ensure_bucket()
    except Exception as e:
        print(f"WARNING: S3 bucket ensure failed: {e}", flush=True)


@app.get("/")
def root():
    return RedirectResponse("/data")


@app.get("/data")
def data_page():
    return FileResponse(STATIC / "data.html")


@app.get("/billing")
def billing_page():
    return FileResponse(STATIC / "billing.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
