"""HTTP app: static billing pages and the JSON API under /api.

OpenAPI at /docs describes that API. Sheet OCR uses Qwen2.5-VL-3B on ray-hive
(model id qwen25-vl-3b). This process only stores data, builds PDFs, and
submits the model job.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import config, db, storage
from .routes import router

STATIC = Path(__file__).resolve().parents[1] / "static"

app = FastAPI(
    title="Nick's Lawn Care Billing",
    version="1.0.0",
    description=(
        "Photograph work-completed sheets, review the OCR table, and store that "
        "month's work. Type client names, mowing prices, and hedge prices by hand. "
        "Generate PDF bills, a tax spreadsheet, and optional email.\n\n"
        "The vision model is **Qwen2.5-VL-3B-Instruct**, served on ray-hive as "
        "`qwen25-vl-3b`. Load it with `POST /api/model/load` before OCR or chat. "
        "Long work returns a `job_id`; poll `GET /api/progress` until `status` is "
        "`done`, `error`, or `cancelled`."
    ),
    openapi_tags=[
        {"name": "status", "description": "Live progress for OCR, chat, billing, email, and model jobs."},
        {"name": "model", "description": "Load and unload the Qwen vision model on ray-hive."},
        {"name": "uploads", "description": "Photo batches, the OCR queue, and saving a reviewed table."},
        {"name": "knowledge", "description": "Clients, months, and the data chat."},
        {"name": "billing", "description": "PDF bills, edits, downloads, tax export, and email."},
    ],
)
app.include_router(router)


@app.middleware("http")
async def full_page_bodies(request: Request, call_next):
    """Ignore Range on HTML and static files.

    After a dropped connection the browser retries the document with a byte
    range. FileResponse answers 206, and a slice of the page never renders.
    """
    path = request.url.path
    if path in {"/", "/data", "/billing", "/howto", "/clients"} or path.startswith("/static/"):
        request.scope["headers"] = [
            (key, value) for key, value in request.scope["headers"] if key.lower() != b"range"
        ]
    return await call_next(request)


@app.on_event("startup")
def startup() -> None:
    """Create database tables and the photo bucket if credentials are set."""
    try:
        db.init_db()
    except Exception as e:
        print(f"WARNING: db init failed: {e}", flush=True)
    try:
        if config.S3_ACCESS_KEY:
            storage.ensure_bucket()
    except Exception as e:
        print(f"WARNING: S3 bucket ensure failed: {e}", flush=True)


@app.get("/", include_in_schema=False)
def root():
    """Send the browser to the work workflow."""
    return RedirectResponse("/billing")


@app.get("/clients", include_in_schema=False)
def clients_redirect():
    """Client list lives on the data page."""
    return RedirectResponse("/data")


@app.get("/data", include_in_schema=False)
def data_page():
    """Editable client list: contact, mowing price, and hedge price."""
    return FileResponse(STATIC / "data.html")


@app.get("/billing", include_in_schema=False)
def billing_page():
    """Upload, review, then generate bills, one step under the next."""
    return FileResponse(STATIC / "billing.html")


@app.get("/howto", include_in_schema=False)
def howto_page():
    """Short instructions for generating and reviewing bills."""
    return FileResponse(STATIC / "howto.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
