"""JSON API for model control, sheet OCR, stored clients, and monthly bills.

Every path is under `/api`. Responses that take a while return `job_id` and
finish on `GET /api/progress`.
"""

from __future__ import annotations

import inspect
import json
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi import Path as ApiPath
from fastapi.responses import Response

from . import billing, chatdata, config, db, emailer, jobs, knowledge, ocr_queue, storage
from .schemas import (
    AcceptedJob,
    ChatRequest,
    CommitJob,
    CorrectJob,
    GenerateBills,
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

    The UI polls this while a model deploy, OCR pass, correction, chat reply,
    bill generation, or email send is running. `status` is `idle`, `running`,
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
        "mowing",
        description="mowing reads the lawn price column. hedges reads the hedge price column.",
    ),
):
    """Queue sheet photos for knowledge-guided OCR.

    Files are saved to object storage under `uploads/{job_id}/`, then read by
    Qwen. If another batch is already running, this one waits. Poll
    `GET /api/progress` for the merged tables and any conflicts with stored
    client rows. Confirm with `POST /api/jobs/{job_id}/commit`.
    """
    if not files:
        raise HTTPException(400, "No files")

    jid = uuid.uuid4().hex[:12]
    local_dir = config.TMP_DIR / jid
    local_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, f in enumerate(files):
        dest = local_dir / (f.filename or f"img_{i}.jpg")
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
                message="Running OCR (knowledge-guided parallel)…",
            )
            with db.connect() as conn:
                knowledge_records = db.knowledge_records_for_sheet(conn, item.sheet_kind)
            merged = {"title": None, "tables": [], "notes": [], "complete": False}
            for i, p in enumerate(item.paths):
                raise_if_cancelled()
                jobs.set_progress(
                    percent=10 + int(80 * i / max(n, 1)),
                    message=f"OCR {p.name} ({i+1}/{n}, {len(knowledge_records)} kb)…",
                )
                part = ocr.extract_sheet(
                    p,
                    sheet_kind=item.sheet_kind,
                    knowledge_records=knowledge_records,
                )
                merged["tables"].extend(part.get("tables") or [])
                merged["notes"].extend(part.get("notes") or [])
                if part.get("title") and not merged.get("title"):
                    merged["title"] = part["title"]
            raise_if_cancelled()
            with db.connect() as conn:
                db.save_upload_job(
                    conn, item.job_id, "done", "OCR complete",
                    extract=merged, month=item.month,
                )
                rows = knowledge.extract_rows(merged)
                conflicts = knowledge.find_conflicts(conn, rows, item.month)
            qsnap = ocr_queue.snapshot()
            jobs.done(
                "OCR complete — review the table",
                detail={
                    "job_id": item.job_id,
                    "month": item.month,
                    "sheet_kind": item.sheet_kind,
                    "extract": merged,
                    "conflicts": conflicts,
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
        jobs.set_progress(message="Cancelling OCR and clearing queue…")
        return {"ok": True, **result, "message": "Cancel all signaled"}

    job_id = (body.job_id or "").strip() or None
    if job_id and ocr_queue.remove_queued(job_id):
        return {"ok": True, "cancelled": True, "job_id": job_id, "message": "Removed from queue"}

    result = ocr_queue.cancel_active(job_id)
    jobs.set_progress(message="Cancelling OCR — aborting inference…")
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


@router.post("/jobs/{job_id}/correct", tags=["uploads"], response_model=AcceptedJob)
async def correct_job(
    job_id: str = ApiPath(..., description="Upload id whose OCR JSON should be rewritten."),
    body: CorrectJob = ...,
):
    """Re-read the sheet with a written instruction and replace the OCR JSON.

    Needs the model loaded. The photo from the original upload is sent again
    when it is still on disk. Returns a new progress `job_id`. The upload id in
    the path stays the record that gets overwritten. When the job finishes,
    `detail.extract` is the replacement and `detail.conflicts` lists clashes
    with data already stored for that month.
    """
    instruction = body.instruction.strip()
    if not instruction:
        raise HTTPException(400, "instruction required")
    with db.connect() as conn:
        row = db.get_upload_job(conn, job_id)
    if not row or not row.get("extract_json"):
        raise HTTPException(404, "job/extract missing")
    extract = row["extract_json"]
    if isinstance(extract, str):
        extract = json.loads(extract)
    month = body.month or row.get("month") or ""
    jid = jobs.new_job("ocr", "Applying correction…")

    def run():
        try:
            import ocr

            local = config.TMP_DIR / job_id
            img = next(local.glob("*"), None) if local.is_dir() else None
            fixed = ocr.apply_correction(img, extract, instruction)
            with db.connect() as conn:
                db.save_upload_job(conn, job_id, "done", "Corrected", extract=fixed)
                rows = knowledge.extract_rows(fixed)
                conflicts = knowledge.find_conflicts(conn, rows, month) if month else []
            jobs.done(
                "Correction applied",
                detail={"job_id": job_id, "extract": fixed, "conflicts": conflicts},
            )
        except Exception as e:
            jobs.fail(str(e))

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid}


@router.get("/jobs/{job_id}/conflicts", tags=["uploads"])
def job_conflicts(
    job_id: str = ApiPath(..., description="Upload id to compare with stored clients."),
    month: str = Query(..., description="Month to compare against stored work, YYYY-MM."),
):
    """Rows in this OCR result that disagree with clients or prices already saved.

    Each conflict names the client and which field differs. `rows` is the full
    parsed table the check used. Nothing is written.
    """
    with db.connect() as conn:
        row = db.get_upload_job(conn, job_id)
        if not row or not row.get("extract_json"):
            raise HTTPException(404)
        extract = row["extract_json"]
        if isinstance(extract, str):
            extract = json.loads(extract)
        rows = knowledge.extract_rows(extract)
        return {"conflicts": knowledge.find_conflicts(conn, rows, month), "rows": rows}


@router.post("/jobs/{job_id}/commit", tags=["uploads"])
async def commit_job(
    job_id: str = ApiPath(..., description="Upload id whose reviewed extract should be stored."),
    body: CommitJob = ...,
):
    """Store the reviewed OCR table as clients and work items for the month.

    New clients are inserted. Matching clients are updated. Work lines are added
    for the month. If the same client already has different prices or contact
    info, `conflict_nl` says which side wins. The response counts what was
    written and includes the resolutions the model chose.
    """
    month = body.month
    if not month:
        raise HTTPException(400, "month required (YYYY-MM)")
    sheet_kind = body.sheet_kind or "mowing"
    conflict_nl = body.conflict_nl.strip()
    with db.connect() as conn:
        row = db.get_upload_job(conn, job_id)
        if not row or not row.get("extract_json"):
            raise HTTPException(404)
        extract = row["extract_json"]
        if isinstance(extract, str):
            extract = json.loads(extract)
        rows = knowledge.extract_rows(extract)
        conflicts = knowledge.find_conflicts(conn, rows, month)
        resolutions = chatdata.resolve_conflicts_nl(conflicts, conflict_nl) if conflicts else {}
        result = knowledge.confirm_extract(
            conn,
            extract,
            month=month,
            source_job_id=job_id,
            resolutions=resolutions,
            sheet_kind=sheet_kind,
        )
        result["conflicts_resolved"] = len(conflicts)
        result["resolutions"] = resolutions
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


@router.get("/clients", tags=["knowledge"])
def clients():
    """Every client on file: name, contact, mow price, and hedge price.

    This is the knowledge base OCR uses when it reads a new sheet. Prices are
    dollars. Missing prices are null.
    """
    with db.connect() as conn:
        return {"clients": db.list_clients(conn)}


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
            out.append({
                "id": b["id"],
                "month": b["month"],
                "client_id": cid,
                "client_name": b["client_name"],
                "email": detail.get("email") or email or "",
                "address": detail.get("address") or "",
                "s3_key": b["s3_key"],
                "emailed_at": b["emailed_at"].isoformat() if b.get("emailed_at") else None,
                "has_email": bool(email and "@" in email),
                "lines": detail.get("lines") or [],
            })
        return {"bills": out, "smtp_configured": emailer.smtp_configured()}


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


@router.get("/billing/{month}/download.zip", tags=["billing"])
def billing_zip(
    month: str = ApiPath(..., description="Billing month, YYYY-MM."),
    mode: str = Query(
        "all",
        description="all: every bill. mailing_only: clients with no email. with_email: clients who have an email.",
    ),
):
    """Zip of PDF bills for the month, filtered by who gets paper mail.

    `mode` must be `all`, `mailing_only`, or `with_email`. The zip filename
    includes the month and that mode.
    """
    if mode not in ("all", "mailing_only", "with_email"):
        raise HTTPException(400, "mode must be all|mailing_only|with_email")
    with db.connect() as conn:
        data = billing.zip_bills(conn, month, mode=mode)
    suffix = {"all": "all", "mailing_only": "mailing_only", "with_email": "with_email"}[mode]
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
            )
            detail = billing.get_editable_bill(conn, month, client_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    return {"ok": True, "bill": saved, "detail": detail}


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
