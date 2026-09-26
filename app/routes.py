"""API routes: model, data upload/review/confirm, billing, chat."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from . import billing, chatdata, config, db, emailer, jobs, knowledge, ocr_queue, storage

router = APIRouter(prefix="/api")


@router.get("/progress")
def progress():
    return jobs.get()


@router.get("/model/status")
def model_status():
    import ocr

    return ocr.model_status()


@router.post("/model/load")
def model_load():
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


@router.post("/model/unload")
def model_unload():
    try:
        import ocr

        ocr.unload_model()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@router.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    month: str = Form(...),
    sheet_kind: str = Form("mowing"),
):
    """Accept one or many images. Queues if another OCR batch is already running."""
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


@router.post("/upload/cancel")
def upload_cancel(body: dict | None = None):
    """Cancel active OCR (keeps queue) or pass all=true to clear the queue too."""
    body = body or {}
    if body.get("all"):
        result = ocr_queue.cancel_all()
        jobs.set_progress(message="Cancelling OCR and clearing queue…")
        return {"ok": True, **result, "message": "Cancel all signaled"}

    job_id = (body.get("job_id") or "").strip() or None
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


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
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


@router.post("/jobs/{job_id}/correct")
async def correct_job(job_id: str, body: dict):
    instruction = (body.get("instruction") or "").strip()
    if not instruction:
        raise HTTPException(400, "instruction required")
    with db.connect() as conn:
        row = db.get_upload_job(conn, job_id)
    if not row or not row.get("extract_json"):
        raise HTTPException(404, "job/extract missing")
    extract = row["extract_json"]
    if isinstance(extract, str):
        extract = json.loads(extract)
    month = body.get("month") or row.get("month") or ""
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


@router.get("/jobs/{job_id}/conflicts")
def job_conflicts(job_id: str, month: str):
    with db.connect() as conn:
        row = db.get_upload_job(conn, job_id)
        if not row or not row.get("extract_json"):
            raise HTTPException(404)
        extract = row["extract_json"]
        if isinstance(extract, str):
            extract = json.loads(extract)
        rows = knowledge.extract_rows(extract)
        return {"conflicts": knowledge.find_conflicts(conn, rows, month), "rows": rows}


@router.post("/jobs/{job_id}/commit")
async def commit_job(job_id: str, body: dict):
    """Send OCR extract into datastore. Optional NL conflict instructions."""
    month = body.get("month")
    if not month:
        raise HTTPException(400, "month required (YYYY-MM)")
    sheet_kind = body.get("sheet_kind") or "mowing"
    conflict_nl = (body.get("conflict_nl") or "").strip()
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


@router.post("/jobs/{job_id}/confirm")
async def confirm_job(job_id: str, body: dict):
    """Alias of commit for older clients."""
    return await commit_job(job_id, body)


@router.post("/chat")
async def chat(body: dict):
    question = (body.get("message") or body.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "message required")
    history = body.get("history") or []
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


@router.get("/months")
def months():
    with db.connect() as conn:
        return {"months": db.months_with_work(conn)}


@router.get("/clients")
def clients():
    with db.connect() as conn:
        return {"clients": db.list_clients(conn)}


@router.post("/billing/generate")
def billing_generate(body: dict):
    month = body.get("month")
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


@router.get("/billing/{month}/list")
def billing_list(month: str):
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


@router.get("/billing/{month}/tax.tsv")
def billing_tax(month: str):
    with db.connect() as conn:
        tsv = billing.tax_table_tsv(conn, month)
    return Response(
        tsv,
        media_type="text/tab-separated-values",
        headers={"Content-Disposition": f'attachment; filename="tax_{month}.tsv"'},
    )


@router.get("/billing/{month}/download.zip")
def billing_zip(month: str, mode: str = "all"):
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


@router.get("/billing/{month}/edit/{client_id}")
def billing_edit_get(month: str, client_id: int):
    with db.connect() as conn:
        data = billing.get_editable_bill(conn, month, client_id)
    if not data:
        raise HTTPException(404, "client/bill not found")
    return data


@router.post("/billing/{month}/edit/{client_id}")
def billing_edit_save(month: str, client_id: int, body: dict):
    lines = body.get("lines")
    if not isinstance(lines, list):
        raise HTTPException(400, "lines array required")
    try:
        with db.connect() as conn:
            saved = billing.save_editable_bill(
                conn,
                month,
                client_id,
                email=str(body.get("email") or ""),
                address=str(body.get("address") or ""),
                lines=lines,
            )
            detail = billing.get_editable_bill(conn, month, client_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    return {"ok": True, "bill": saved, "detail": detail}


@router.get("/billing/pdf")
def billing_pdf(key: str):
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


@router.post("/billing/{month}/email")
def billing_email(month: str):
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
