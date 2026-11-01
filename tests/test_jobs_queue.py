"""In-memory progress bars and the sequential OCR queue."""

import time

from app import jobs, ocr_queue


def test_two_running_jobs_both_appear():
    a = jobs.new_job("generate", "one")
    b = jobs.new_job("email", "two")
    payload = jobs.get()
    ids = {row["job_id"] for row in payload["jobs"]}
    assert a in ids
    assert b in ids
    assert len(payload["jobs"]) >= 2


def test_percent_is_clamped():
    jid = jobs.new_job("ocr", "go")
    jobs.set_progress(job_id=jid, percent=-10)
    assert jobs.get()["percent"] == 0
    jobs.use(jid)
    jobs.set_progress(percent=150)
    latest = next(row for row in jobs.get()["jobs"] if row["job_id"] == jid)
    assert latest["percent"] == 100


def test_finished_jobs_prune_after_keep_window(monkeypatch):
    jid = jobs.new_job("chat", "ask")
    jobs.done("ok", job_id=jid)
    now = time.time()
    monkeypatch.setattr(jobs.time, "time", lambda: now + 30)
    payload = jobs.get()
    assert all(row["job_id"] != jid for row in payload["jobs"])


def test_second_ocr_batch_queues_then_kick_runs_it():
    ran = []

    def worker(item):
        ran.append(item.job_id)

    first, state = ocr_queue.enqueue(paths=[], month="2026-11", sheet_kind="work", job_id="a")
    assert state == "running"
    second, state = ocr_queue.enqueue(paths=[], month="2026-11", sheet_kind="work", job_id="b")
    assert state == "queued"
    snap = ocr_queue.snapshot()
    assert snap["queue_depth"] == 1
    assert snap["queued"][0]["job_id"] == second
    ocr_queue.kick(worker)
    deadline = time.time() + 2
    while ran != [first, second] and time.time() < deadline:
        time.sleep(0.01)
    assert ran == [first, second]


def test_remove_queued_and_cancel_all():
    ocr_queue.enqueue(paths=[], month="2026-11", sheet_kind="work", job_id="active")
    ocr_queue.enqueue(paths=[], month="2026-11", sheet_kind="work", job_id="queued")
    assert ocr_queue.remove_queued("queued")
    assert ocr_queue.snapshot()["queue_depth"] == 0
    ocr_queue.enqueue(paths=[], month="2026-11", sheet_kind="work", job_id="later")
    result = ocr_queue.cancel_all()
    assert result["cleared_queue"] == 1
    assert ocr_queue.snapshot()["queue_depth"] == 0
