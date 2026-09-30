"""JSON API for model control, sheet OCR, stored clients, and monthly bills.

Every path is under `/api`. Responses that take a while return `job_id` and
finish on `GET /api/progress`.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi import Path as ApiPath
from fastapi.responses import Response

from . import billing, chatdata, config, db, emailer, jobs, knowledge, ocr_queue, storage
from .schemas import (
    AcceptedJob,
    ChatRequest,
    CommitJob,
    GenerateBills,
    ClientEmail,
    ClientRoster,
    BillFace,
    ManualWork,
    ReviewDraft,
    ModelStatus,
    ProgressView,
    SaveBill,
    UploadAccepted,
    UploadCancel,
)

router = APIRouter(prefix="/api")


@router.get("/progress", tags=["status"], response_model=ProgressView)
def progress():
    """Current background job, plus how many OCR batches are waiting.

    The UI polls this while a model deploy, OCR pass, chat reply,
    bill generation, or email send is running. `jobs` lists every one still
    running or just finished, and the header draws a bar for each.
    The top-level fields repeat the newest job. `status` is `idle`, `running`,
    `done`, `error`, or `cancelled`. `detail` holds the result when the job finishes.
    """
    return jobs.get()


@router.get("/model/status", tags=["model"], response_model=ModelStatus)
def model_status():
    """Whether Qwen2.5-VL-3B is answering on ray-hive.

    `id` is always `qwen25-vl-3b`. `up` is true when that Serve route returns
    a model list. `base_url` is the OpenAI-style chat endpoint for that model.
    """
    import ocr

    return ocr.model_status()


@router.post("/model/load", tags=["model"], response_model=AcceptedJob)
def model_load():
    """Deploy Qwen2.5-VL-3B-Instruct on the Ray cluster.

    Blocks the worker until Serve reports the model ready, then records the
    result on `GET /api/progress`. Safe to call when the model is already up;
    the deploy job replaces the previous copy. The container itself does not
    load model weights.
    """
    jid = jobs.new_job("model", "Deploying OCR model…")

    def run():
        jobs.use(jid)
        try:
            import ocr

            jobs.set_progress(percent=20, message="Submitting deploy job…")
            ocr.load_model()
            jobs.done("Model ready", detail=ocr.model_status())
        except Exception as e:
            jobs.fail(str(e))

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}


@router.post("/model/unload", tags=["model"])
def model_unload():
    """Shut down the Qwen vision model on ray-hive.

    OCR and chat fail until `POST /api/model/load` brings it back. Stored
    clients, work, and PDFs are left as they are.
    """
    try:
        import ocr

        ocr.unload_model()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@router.post("/upload", tags=["uploads"], response_model=UploadAccepted)
async def upload(
    files: list[UploadFile] = File(
        ...,
        description="One or more sheet photos. JPEG or PNG. Each file is one page.",
    ),
    month: str = Form(
        ...,
        description="Billing month these sheets belong to, YYYY-MM.",
    ),
    sheet_kind: str = Form(
        "work",
        description="Ignored. Uploaded photos are always a work-completed log.",
    ),
):
    """Queue work-completed photos for knowledge-guided OCR.

    Mowing prices and hedge prices are typed on the Data page, not photographed.
    Files are saved under `uploads/{job_id}/`. Photos in this batch are read
    together. If another batch is already running, this one waits. Poll
    `GET /api/progress` for the merged tables. Confirm with
    `POST /api/jobs/{job_id}/commit`.
    """
    del sheet_kind
    sheet_kind = "work"
    if not files:
        raise HTTPException(400, "No files")

    jid = uuid.uuid4().hex[:12]
    local_dir = config.TMP_DIR / jid
    local_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, f in enumerate(files):
        raw_name = Path(f.filename or f"img_{i}.jpg").name
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name).strip("._") or f"img_{i}.jpg"
        dest = local_dir / f"{i + 1:02d}_{safe}"
        data = await f.read()
        dest.write_bytes(data)
        paths.append(dest)

    for p in paths:
        key = f"uploads/{jid}/{p.name}"
        try:
            storage.put_bytes(p.read_bytes(), key, content_type="image/jpeg")
        except Exception as e:
            raise HTTPException(500, f"S3 upload failed: {e}") from e

    state_jid, state = ocr_queue.enqueue(
        paths=paths, month=month, sheet_kind=sheet_kind, job_id=jid,
    )
    jid = state_jid

    with db.connect() as conn:
        db.save_upload_job(
            conn,
            jid,
            "queued" if state == "queued" else "running",
            "Queued…" if state == "queued" else "OCR starting…",
            s3_prefix=f"uploads/{jid}/",
            month=month,
            sheet_kind=sheet_kind,
        )

    def worker(item):
        from ocr.cancel import InferenceCancelled, begin_job, clear_job, raise_if_cancelled

        begin_job(item.job_id)
        n = len(item.paths)
        jobs.new_job_replace(
            item.job_id,
            "ocr",
            f"OCR {item.month} ({n} photo{'s' if n != 1 else ''})…",
        )
        try:
            import ocr

            jobs.set_progress(
                percent=10,
                message=f"Reading {n} photo{'s' if n != 1 else ''} at the same time…",
            )
            with db.connect() as conn:
                knowledge_records = db.knowledge_records_for_sheet(conn, item.sheet_kind)
            merged = {"title": None, "tables": [], "notes": [], "complete": False}
            seqs = int((ocr.MODEL_CFG.get("vllm_kwargs") or {}).get("max_num_seqs") or 4)
            slots = max(1, min(n, seqs))
            per_photo = max(1, seqs // slots)

            def _one(index: int, path: Path):
                raise_if_cancelled()
                return index, ocr.extract_sheet(
                    path,
                    sheet_kind=item.sheet_kind,
                    knowledge_records=knowledge_records,
                    wave_workers=per_photo,
                )

            parts: list[dict | None] = [None] * n
            with ThreadPoolExecutor(max_workers=slots) as pool:
                futures = [pool.submit(_one, i, p) for i, p in enumerate(item.paths)]
                finished = 0
                try:
                    for fut in as_completed(futures):
                        index, part = fut.result()
                        parts[index] = part
                        finished += 1
                        jobs.set_progress(
                            percent=10 + int(80 * finished / max(n, 1)),
                            message=f"OCR {finished}/{n} photos finished…",
                        )
                except Exception:
                    for fut in futures:
                        fut.cancel()
                    raise
            for part in parts:
                if not part:
                    continue
                merged["tables"].extend(part.get("tables") or [])
                merged["notes"].extend(part.get("notes") or [])
                if part.get("title") and not merged.get("title"):
                    merged["title"] = part["title"]
            raise_if_cancelled()
            with db.connect() as conn:
                db.save_upload_job(
                    conn, item.job_id, "done", "OCR complete",
                    extract=merged, month=item.month, sheet_kind=item.sheet_kind,
                )
            qsnap = ocr_queue.snapshot()
            jobs.done(
                "OCR complete — review the table",
                detail={
                    "job_id": item.job_id,
                    "month": item.month,
                    "sheet_kind": item.sheet_kind,
                    "extract": merged,
                    "queue_depth": qsnap["queue_depth"],
                },
            )
        except InferenceCancelled:
            with db.connect() as conn:
                db.save_upload_job(conn, item.job_id, "cancelled", "Cancelled by user")
            jobs.cancel("OCR cancelled — inference aborted")
        except Exception as e:
            with db.connect() as conn:
                db.save_upload_job(conn, item.job_id, "error", str(e))
            jobs.fail(str(e))
        finally:
            clear_job(item.job_id)

    ocr_queue.set_worker(worker)
    if state == "running":
        ocr_queue.kick(worker)

    q = ocr_queue.snapshot()
    return {
        "job_id": jid,
        "month": month,
        "state": state,
        "queue_depth": q["queue_depth"],
        "n_files": len(paths),
    }


@router.post("/upload/cancel", tags=["uploads"])
def upload_cancel(body: UploadCancel | None = None):
    """Stop the running OCR pass, or remove one batch from the queue.

    In-flight vision calls abort. Batches still waiting stay unless `all` is
    true or `job_id` names a queued batch. The response returns as soon as the
    cancel is signaled; the worker updates `GET /api/progress` when it stops.
    """
    body = body or UploadCancel()
    if body.all:
        result = ocr_queue.cancel_all()
        ocr_id = jobs.running_id("ocr")
        if ocr_id:
            jobs.set_progress(job_id=ocr_id, message="Cancelling OCR and clearing queue…")
        return {"ok": True, **result, "message": "Cancel all signaled"}

    job_id = (body.job_id or "").strip() or None
    if job_id and ocr_queue.remove_queued(job_id):
        return {"ok": True, "cancelled": True, "job_id": job_id, "message": "Removed from queue"}

    result = ocr_queue.cancel_active(job_id)
    ocr_id = job_id or jobs.running_id("ocr")
    if ocr_id:
        jobs.set_progress(job_id=ocr_id, message="Cancelling OCR — aborting inference…")
    return {
        "ok": True,
        "cancelled": result.get("signaled"),
        **result,
        "message": "Cancel signaled; waiting for worker to stop",
    }


@router.get("/jobs/{job_id}", tags=["uploads"])
def get_job(
    job_id: str = ApiPath(..., description="Upload id returned by POST /api/upload."),
):
    """Saved state of one upload: status, month, and the OCR tables if finished.

    `extract` is the JSON the model returned (title, tables, notes). It is empty
    until OCR succeeds. This is the stored job, not the live progress bar.
    """
    with db.connect() as conn:
        row = db.get_upload_job(conn, job_id)
    if not row:
        raise HTTPException(404, "job not found")
    return {
        "id": row["id"],
        "status": row["status"],
        "progress_msg": row["progress_msg"],
        "month": row["month"],
        "extract": row["extract_json"],
    }


def _review_payload(row: dict | None, closed_status: str | None = None) -> dict:
    if not row or not row.get("extract_json"):
        return {
            "job_id": None,
            "month": None,
            "sheet_kind": None,
            "updated_at": None,
            "extract": None,
            "closed_status": closed_status,
        }
    extract = row["extract_json"]
    if isinstance(extract, str):
        extract = json.loads(extract)
    stamp = row.get("updated_at")
    return {
        "job_id": row["id"],
        "month": row.get("month"),
        "sheet_kind": row.get("sheet_kind"),
        "updated_at": stamp.isoformat() if hasattr(stamp, "isoformat") else stamp,
        "extract": extract,
    }


@router.get("/review/open", tags=["uploads"])
def review_open():
    """The newest OCR result that has not been stored yet.

    Every browser loads this, so a review started on a phone is the same
    table on a computer.
    """
    with db.connect() as conn:
        row = conn.execute(
            """
            SELECT id, month, sheet_kind, updated_at, extract_json
            FROM upload_jobs
            WHERE status = 'done' AND extract_json IS NOT NULL
            ORDER BY updated_at DESC NULLS LAST, created_at DESC
            LIMIT 1
            """
        ).fetchone()
        closed = None
        if not row:
            latest = conn.execute(
                """
                SELECT status FROM upload_jobs
                ORDER BY updated_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """
            ).fetchone()
            if latest and latest.get("status") in ("cleared", "stored"):
                closed = latest["status"]
    return _review_payload(row, closed)


@router.put("/jobs/{job_id}/draft", tags=["uploads"])
def save_draft(
    job_id: str = ApiPath(..., description="Upload id of the open review."),
    body: ReviewDraft = ...,
):
    """Save table edits without storing them as clients or work.

    Other open browsers pick this up from `GET /api/review/open`.
    """
    if not body.extract:
        raise HTTPException(400, "extract required")
    with db.connect() as conn:
        row = db.save_review_draft(
            conn, job_id, body.extract, body.month, "work",
        )
    if not row:
        raise HTTPException(404, "open review not found")
    stamp = row.get("updated_at")
    return {
        "job_id": row["id"],
        "month": row.get("month"),
        "sheet_kind": row.get("sheet_kind"),
        "updated_at": stamp.isoformat() if hasattr(stamp, "isoformat") else stamp,
    }


@router.post("/jobs/{job_id}/clear", tags=["uploads"])
def clear_job(job_id: str = ApiPath(..., description="Upload id of the open review to drop.")):
    """Remove the open review tables without saving them as clients or work.

    Other open browsers drop the same review on their next refresh.
    """
    with db.connect() as conn:
        row = db.clear_review(conn, job_id)
    if not row:
        raise HTTPException(404, "open review not found")
    return {"ok": True, "job_id": row["id"]}


@router.post("/jobs/{job_id}/commit", tags=["uploads"])
async def commit_job(
    job_id: str = ApiPath(..., description="Upload id whose reviewed extract should be stored."),
    body: CommitJob = ...,
):
    """Store the reviewed work-completed table as work lines for the month.

    A name that is not on the client list is returned in `conflicts` and nothing
    is stored until the request includes details for that person, plus whether
    they should stay on the client list. Known clients are stored as they are.
    Each plain day becomes a mowing line at that client's mowing price. A day
    marked with h becomes a hedge line at the hedge price. A job name with a
    dollar amount written on the sheet becomes its own line.
    """
    month = body.month
    if not month:
        raise HTTPException(400, "month required (YYYY-MM)")
    sheet_kind = "work"
    with db.connect() as conn:
        row = db.get_upload_job(conn, job_id)
        if body.extract:
            extract = body.extract
        else:
            if not row or not row.get("extract_json"):
                raise HTTPException(404)
            extract = row["extract_json"]
            if isinstance(extract, str):
                extract = json.loads(extract)
        if not extract:
            raise HTTPException(404, "extract missing")
        result = knowledge.confirm_extract(
            conn,
            extract,
            month=month,
            source_job_id=job_id,
            sheet_kind=sheet_kind,
            resolutions=[item.model_dump() for item in body.resolutions],
        )
        if not result.get("conflicts"):
            db.save_upload_job(
                conn, job_id, "stored", "Stored",
                extract=extract, month=month,
            )
    return result


@router.post("/jobs/{job_id}/confirm", tags=["uploads"], include_in_schema=False)
async def confirm_job(
    job_id: str = ApiPath(..., description="Upload id. Same meaning as commit."),
    body: CommitJob = ...,
):
    """Same as commit. Kept so older clients that post to /confirm still work."""
    return await commit_job(job_id, body)


@router.post("/chat", tags=["knowledge"], response_model=AcceptedJob)
async def chat(body: ChatRequest):
    """Ask about stored clients, work, and bills, or change them in plain language.

    Reads Postgres and answers with the loaded Qwen model. The model may also
    update a client, add or delete a work line, or change a price. Those edits
    are applied before the reply is returned. `detail.answer` is the text.
    `detail.mutations_applied` lists each change. Send prior turns in `history`
    without the new message.
    """
    question = (body.message or body.question or "").strip()
    if not question:
        raise HTTPException(400, "message required")
    history = [turn.model_dump() for turn in body.history]
    jid = jobs.new_job("chat", "Thinking…")

    def run():
        jobs.use(jid)
        try:
            with db.connect() as conn:
                result = chatdata.answer(conn, question, history=history)
            jobs.done(
                "Chat reply",
                detail={
                    "answer": result["answer"],
                    "question": question,
                    "mutations_applied": result.get("mutations_applied") or [],
                },
            )
        except Exception as e:
            jobs.fail(str(e))

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}


@router.get("/months", tags=["knowledge"])
def months():
    """Months that already have confirmed work, newest first.

    Use these values as the `month` field when generating bills.
    """
    with db.connect() as conn:
        return {"months": db.months_with_work(conn)}


def _public_client(row: dict) -> dict:
    def num(value):
        return float(value) if value is not None else None

    return {
        "id": int(row["id"]),
        "name": row.get("name") or "",
        "address": row.get("address") or "",
        "phone": row.get("phone") or "",
        "email": row.get("email") or "",
        "billing_notes": row.get("billing_notes") or "",
        "mow_price": num(row.get("mow_price")),
        "hedge_price": num(row.get("hedge_price")),
        "prefer_mail": bool(row.get("prefer_mail")),
    }


@router.get("/clients", tags=["knowledge"])
def clients():
    """Every client kept on the list: name, contact, mow price, and hedge price.

    Someone billed for one month and not added to the list is left out.
    These rows are typed in. Prices are dollars. Missing prices are null.
    """
    with db.connect() as conn:
        rows = [row for row in db.list_clients(conn) if row.get("on_roster") is not False]
        return {"clients": [_public_client(row) for row in rows]}


@router.post("/clients", tags=["knowledge"])
def save_clients(body: ClientRoster):
    """Save the typed roster: names, contact, mowing prices, and hedge prices.

    A blank price clears that price. A row with no name is skipped. Matching
    an existing name updates that client.
    """
    saved = 0
    with db.connect() as conn:
        for item in body.clients:
            name = (item.name or "").strip()
            if not name:
                continue

            def blank(value: str) -> str | None:
                text = (value or "").strip()
                return text or None

            try:
                db.save_typed_client(
                    conn,
                    client_id=item.id,
                    name=name,
                    address=blank(item.address),
                    phone=blank(item.phone),
                    email=blank(item.email),
                    billing_notes=blank(item.billing_notes),
                    mow_price=item.mow_price,
                    hedge_price=item.hedge_price,
                    prefer_mail=item.prefer_mail,
                )
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            saved += 1
        rows = [
            _public_client(row)
            for row in db.list_clients(conn)
            if row.get("on_roster") is not False
        ]
    return {"saved": saved, "clients": rows}


@router.post("/clients/{client_id}/email", tags=["knowledge"])
def set_client_email(
    client_id: int = ApiPath(..., description="Numeric client id."),
    body: ClientEmail = ...,
):
    """Change one client's email. The delivery label follows this value."""
    email = (body.email or "").strip() or None
    with db.connect() as conn:
        if not db.get_client(conn, client_id):
            raise HTTPException(404, "client not found")
        db.set_client_email(conn, client_id, email)
        row = db.get_client(conn, client_id)
    return {"client": _public_client(row), "delivery": billing.delivery_channel(row)}


@router.delete("/clients/{client_id}", tags=["knowledge"])
def remove_client(client_id: int = ApiPath(..., description="Numeric client id.")):
    """Delete a client, their work, their bills, and the stored PDFs."""
    with db.connect() as conn:
        if not db.get_client(conn, client_id):
            raise HTTPException(404, "client not found")
        keys = db.delete_client(conn, client_id)
    for key in keys:
        storage.delete_key(key)
    return {"ok": True, "deleted_pdfs": len(keys)}


@router.post("/work", tags=["knowledge"])
def add_manual_work(body: ManualWork):
    """Store one typed work row for a month, using the same day and job rules as a sheet."""
    month = (body.month or "").strip()
    text = (body.text or "").strip()
    if not month or not text:
        raise HTTPException(400, "month and text required")
    with db.connect() as conn:
        if not db.get_client(conn, body.client_id):
            raise HTTPException(404, "client not found")
        written = knowledge.store_work_text(
            conn, client_id=body.client_id, month=month, text=text,
        )
    return {"ok": True, "lines": written, "month": month, "client_id": body.client_id}


@router.post("/billing/generate", tags=["billing"], response_model=AcceptedJob)
def billing_generate(body: GenerateBills):
    """Build one PDF bill per client who has work in the month.

    Fails the progress job if that month has no confirmed work. Finished PDFs
    are stored in the `nicks-lawn-billing` bucket. `detail.bills` lists each
    client and the object key. Poll `GET /api/progress` for completion.
    """
    month = body.month
    if not month:
        raise HTTPException(400, "month required")
    jid = jobs.new_job("generate", f"Generating bills for {month}…")

    def run():
        jobs.use(jid)
        try:
            with db.connect() as conn:
                if not db.work_for_month(conn, month):
                    jobs.fail(f"No confirmed work for {month}")
                    return
                jobs.set_progress(percent=40, message="Building PDFs…")
                bills = billing.generate_month_bills(conn, month)
            jobs.done(f"Generated {len(bills)} bills", detail={"month": month, "bills": bills})
        except Exception as e:
            jobs.fail(str(e))

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}


@router.get("/billing/{month}/list", tags=["billing"])
def billing_list(
    month: str = ApiPath(..., description="Billing month, YYYY-MM."),
):
    """Bills already generated for one month, with contact info and line items.

    `month` is YYYY-MM. `has_email` is true when the client address contains
    `@`, which is who `POST /billing/{month}/email` will try to send to.
    `smtp_configured` says whether the server has SMTP settings at all.
    `s3_key` is passed to `GET /api/billing/pdf`.
    """
    with db.connect() as conn:
        bills = db.bills_for_month(conn, month)
        out = []
        for b in bills:
            email = (b.get("email") or "").strip()
            cid = int(b["client_id"])
            detail = billing.get_editable_bill(conn, month, cid) or {}
            channel = billing.delivery_channel(b)
            out.append({
                "id": b["id"],
                "month": b["month"],
                "client_id": cid,
                "client_name": b["client_name"],
                "email": detail.get("email") or email or "",
                "phone": b.get("phone") or "",
                "address": detail.get("address") or "",
                "s3_key": b["s3_key"],
                "emailed_at": b["emailed_at"].isoformat() if b.get("emailed_at") else None,
                "has_email": channel == "email",
                "delivery": channel,
                "roster_name": b.get("roster_name") or b["client_name"],
                "roster_email": b.get("roster_email") or "",
                "roster_delivery": b.get("roster_delivery") or channel,
                "lines": detail.get("lines") or [],
            })
        return {"bills": out, "smtp_configured": emailer.smtp_configured()}


@router.post("/billing/{month}/face/{client_id}", tags=["billing"])
def billing_face(
    month: str = ApiPath(..., description="Billing month, YYYY-MM."),
    client_id: int = ApiPath(..., description="Numeric client id."),
    body: BillFace = ...,
):
    """Update this month's bill name, email, and delivery, then rebuild the PDF.

    The client list changes only when `save_to_client` is true.
    """
    try:
        with db.connect() as conn:
            saved = billing.apply_bill_face(
                conn,
                month,
                client_id,
                name=body.name,
                email=body.email,
                delivery=body.delivery,
                save_to_client=body.save_to_client,
            )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "bill": saved}


@router.get("/billing/{month}/tax.tsv", tags=["billing"])
def billing_tax(
    month: str = ApiPath(..., description="Billing month, YYYY-MM."),
):
    """Tab-separated client, description, amount, and month for a tax spreadsheet.

    One row per work line in that month. The file is an attachment named
    `tax_{month}.tsv`.
    """
    with db.connect() as conn:
        tsv = billing.tax_table_tsv(conn, month)
    return Response(
        tsv,
        media_type="text/tab-separated-values",
        headers={"Content-Disposition": f'attachment; filename="tax_{month}.tsv"'},
    )


@router.get("/billing/{month}/tax.xlsx", tags=["billing"])
def billing_tax_xlsx(
    month: str = ApiPath(..., description="Billing month, YYYY-MM."),
):
    """Excel workbook of this month's revenue, sales tax, and any earlier unpaid balance."""
    with db.connect() as conn:
        data = billing.tax_table_xlsx(conn, month)
    return Response(
        data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="tax_{month}.xlsx"'},
    )


@router.get("/billing/{month}/download.zip", tags=["billing"])
def billing_zip(
    month: str = ApiPath(..., description="Billing month, YYYY-MM."),
    mode: str = Query(
        "all",
        description="all, mailing_only (paper), sms_only, or with_email.",
    ),
):
    """Zip of PDF bills for the month, filtered by delivery.

    `mode` must be `all`, `mailing_only`, `sms_only`, or `with_email`.
    """
    if mode not in ("all", "mailing_only", "sms_only", "with_email"):
        raise HTTPException(400, "mode must be all|mailing_only|sms_only|with_email")
    with db.connect() as conn:
        data = billing.zip_bills(conn, month, mode=mode)
    suffix = {
        "all": "all",
        "mailing_only": "mail",
        "sms_only": "sms",
        "with_email": "with_email",
    }[mode]
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="bills_{month}_{suffix}.zip"'},
    )


@router.get("/billing/{month}/edit/{client_id}", tags=["billing"])
def billing_edit_get(
    month: str = ApiPath(..., description="Billing month, YYYY-MM."),
    client_id: int = ApiPath(..., description="Numeric client id from GET /api/clients."),
):
    """Current bill lines and contact fields for one client in one month.

    `client_id` is the numeric id from `GET /api/clients` or the bill list.
    `lines` are the work items that will print. 404 if that client does not exist.
    """
    with db.connect() as conn:
        data = billing.get_editable_bill(conn, month, client_id)
    if not data:
        raise HTTPException(404, "client/bill not found")
    return data


@router.post("/billing/{month}/edit/{client_id}", tags=["billing"])
def billing_edit_save(
    month: str = ApiPath(..., description="Billing month, YYYY-MM."),
    client_id: int = ApiPath(..., description="Numeric client id from GET /api/clients."),
    body: SaveBill = ...,
):
    """Replace this client's email, address, and work lines, then rebuild the PDF.

    Send the full line list. A line with an existing `id` is updated. A line
    without `id` is added. Stored lines omitted from `lines` are deleted.
    `bill.s3_key` is the new PDF. `detail` is the same shape as the edit GET.
    """
    lines = [ln.model_dump() for ln in body.lines]
    if not isinstance(lines, list):
        raise HTTPException(400, "lines array required")
    try:
        with db.connect() as conn:
            saved = billing.save_editable_bill(
                conn,
                month,
                client_id,
                email=body.email,
                address=body.address,
                lines=lines,
                intro=body.intro,
                greeting=body.greeting,
                closing=body.closing,
                signoff=body.signoff,
                phone=body.phone,
                delivery=body.delivery,
                save_to_client=body.save_to_client,
            )
            detail = billing.get_editable_bill(conn, month, client_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    return {"ok": True, "bill": saved, "detail": detail}


@router.delete("/billing/{month}/{client_id}", tags=["billing"])
def billing_delete(
    month: str = ApiPath(..., description="Billing month, YYYY-MM."),
    client_id: int = ApiPath(..., description="Numeric client id."),
):
    """Delete this client's bill and work for the month, including the PDF."""
    with db.connect() as conn:
        if not db.get_client(conn, client_id):
            raise HTTPException(404, "client not found")
        keys = db.delete_client_month(conn, month, client_id)
    for key in keys:
        storage.delete_key(key)
    return {"ok": True}


@router.post("/billing/{month}/sms-email", tags=["billing"], response_model=AcceptedJob)
def billing_sms_email(
    month: str = ApiPath(..., description="Billing month, YYYY-MM."),
):
    """Email the SMS bills to Nick so he can text them. Clients are not emailed."""
    if not emailer.smtp_configured():
        raise HTTPException(400, "SMTP not configured")
    jid = jobs.new_job("email", f"Emailing SMS bills for {month}…")

    def run():
        jobs.use(jid)
        try:
            def prog(pct, msg):
                jobs.set_progress(percent=pct, message=msg)

            with db.connect() as conn:
                result = emailer.email_sms_pack(conn, month, on_progress=prog)
            jobs.done(
                f"Sent {result['messages']} message(s) covering {result['sent']} SMS bills",
                detail=result,
            )
        except Exception as e:
            jobs.fail(str(e))

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}


@router.get("/billing/pdf", tags=["billing"])
def billing_pdf(
    key: str = Query(..., description="Object key from a bill's s3_key, such as bills/2026-09/smith.pdf."),
):
    """Stream one stored PDF for inline view or download.

    404 when the key is missing from the bucket.
    """
    try:
        data = storage.get_bytes(key)
    except Exception as e:
        raise HTTPException(404, str(e)) from e
    name = key.rsplit("/", 1)[-1] or "bill.pdf"
    return Response(
        data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{name}"'},
    )


@router.post("/billing/{month}/email", tags=["billing"], response_model=AcceptedJob)
def billing_email(
    month: str = ApiPath(..., description="Billing month, YYYY-MM. PDFs must already be generated."),
):
    """Email generated PDFs to every client in the month who has an email address.

    Requires SMTP settings on the server. Clients without an `@` address are
    skipped, not failed. `detail.sent` and `detail.skipped` list the outcome.
    Returns a progress `job_id`.
    """
    if not emailer.smtp_configured():
        raise HTTPException(400, "SMTP not configured")
    jid = jobs.new_job("email", f"Emailing bills for {month}…")

    def run():
        jobs.use(jid)
        try:
            def prog(pct, msg):
                jobs.set_progress(percent=pct, message=msg)

            with db.connect() as conn:
                result = emailer.email_month_bills(conn, month, on_progress=prog)
            jobs.done(
                f"Sent {len(result['sent'])}, skipped {len(result['skipped'])}",
                detail=result,
            )
        except Exception as e:
            jobs.fail(str(e))

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}


def _summaries_from_docstrings() -> None:
    """Use the first docstring line as the Swagger title, and the rest as the body."""
    for route in router.routes:
        doc = inspect.getdoc(route.endpoint) or ""
        if not doc:
            continue
        title, _, body = doc.partition("\n")
        route.summary = title.strip()
        route.description = body.strip() or title.strip()


_summaries_from_docstrings()
