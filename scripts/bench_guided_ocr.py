#!/usr/bin/env python3
"""Bench guided OCR using the same Postgres knowledge an upload uses.

Gold JSON is loaded only after extract_sheet returns, and only to score.
Only work-completed fixtures are scored. The run refuses to start if a
prompt locator contains a dollar amount.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
os.environ.setdefault("NINI_NOVEL", "1")

OUT = ROOT / "bench_results" / "guided_ocr.json"
CHUNK = int(os.environ.get("NINI_OCR_CHUNK", "4"))
QWEN_ID = "qwen25-vl-3b"
SERVE = os.environ.get("RAY_SERVE", "http://10.0.1.52:8000")


def _norm_name(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_fixtures() -> list[dict]:
    """Work-completed photos only. Rosters and hedge lists are typed, not OCR'd."""
    idx = json.loads((ROOT / "ground_truth" / "index.json").read_text(encoding="utf-8"))
    out = []
    for fx in idx["fixtures"]:
        if "work" not in str(fx.get("kind") or "").lower():
            continue
        img = ROOT / fx["image"]
        gold_path = ROOT / fx["gold_file"]
        if not img.is_file() or not gold_path.is_file():
            continue
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        out.append({"id": fx["id"], "image": img, "gold": gold, "kind": fx.get("kind")})
    return out


def model_up(model_id: str) -> bool:
    try:
        with urllib.request.urlopen(f"{SERVE}/{model_id}/v1/models", timeout=8) as r:
            return r.status == 200
    except Exception:
        return False


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def sheet_kind_for(fx: dict) -> str:
    return "work"


def _tokens(name: str) -> list[str]:
    return [t for t in _norm_name(name).split() if len(t) >= 4]


def _names_match(gold: str, pred: str) -> bool:
    """Same person. A shared short token such as POND is not enough."""
    if gold == pred:
        return True
    gt, pt = _tokens(gold), _tokens(pred)
    if not gt or not pt or gt[0] != pt[0]:
        return False
    if len(gt) == 1 or len(pt) == 1:
        return len(gt[0]) >= 5
    return len(set(gt) & set(pt)) >= 2


def _day_set(values) -> set[str]:
    out: set[str] = set()
    for value in values:
        for num in re.findall(r"\d+", str(value)):
            out.add(str(int(num)))
    return out


def score_live(extract: dict, gold: dict) -> dict:
    """Score after the fact. Gold is not an input to OCR.

    A price counts only from the price column. Names need the same surname
    plus another shared token when both sides have a given name.
    """
    gold_rows = []
    for row in gold.get("rows") or []:
        name = _norm_name(row.get("client") or "")
        if not name:
            continue
        price = row.get("mow_price")
        if price is None:
            price = row.get("hedge_price")
        gold_rows.append(
            {
                "name": name,
                "price": None if price is None else float(price),
                "days": _day_set(row.get("days") or []),
            }
        )
    pred_rows = []
    for table in extract.get("tables") or []:
        if not isinstance(table, dict):
            continue
        cols = [str(c).lower() for c in (table.get("columns") or [])]
        name_i = next(
            (i for i, c in enumerate(cols) if "contact" in c or "name" in c or "client" in c),
            0,
        )
        price_i = next(
            (i for i, c in enumerate(cols) if "price" in c or "mow" in c or c == "hedge" or "amount" in c),
            -1,
        )
        work_i = next(
            (i for i, c in enumerate(cols) if "date" in c or "work" in c or "day" in c),
            -1,
        )
        for row in table.get("rows") or []:
            if not isinstance(row, list) or not row:
                continue
            name = _norm_name(str(row[name_i] if name_i < len(row) else ""))
            if not name or name in ("CONTACT", "NAME", "UNCLEAR"):
                continue
            price = None
            if 0 <= price_i < len(row):
                m = re.search(r"(\d+(?:\.\d+)?)", str(row[price_i]).replace(",", ""))
                if m:
                    price = float(m.group(1))
            cell = str(row[work_i]) if 0 <= work_i < len(row) else ""
            pred_rows.append(
                {"name": name, "price": price, "days": _day_set([cell])}
            )
    used: set[int] = set()
    hit = price_hit = price_n = day_hit = day_n = 0
    for grow in gold_rows:
        match = None
        for i, prow in enumerate(pred_rows):
            if i in used:
                continue
            if _names_match(grow["name"], prow["name"]):
                match = prow
                used.add(i)
                break
        if grow["price"] is not None:
            price_n += 1
        if grow["days"]:
            day_n += 1
        if match is None:
            continue
        hit += 1
        if (
            grow["price"] is not None
            and match["price"] is not None
            and abs(match["price"] - grow["price"]) <= 1.01
        ):
            price_hit += 1
        if grow["days"] and grow["days"] <= match["days"]:
            day_hit += 1
    n = len(gold_rows)
    return {
        "client_recall": round(hit / n, 4) if n else 0.0,
        "price_recall": round(price_hit / price_n, 4) if price_n else None,
        "day_recall": round(day_hit / day_n, 4) if day_n else None,
        "n_gold": n,
        "n_price": price_n,
        "n_days": day_n,
        "n_pred": len(pred_rows),
        "matched_clients": hit,
        "matched_prices": price_hit,
        "matched_days": day_hit,
    }


def _weighted(per: list[dict], num_key: str, den_key: str) -> float | None:
    num = sum(int(r.get(num_key) or 0) for r in per)
    den = sum(int(r.get(den_key) or 0) for r in per)
    return round(num / den, 4) if den else None


def assert_live_knowledge() -> dict[str, int]:
    """Work-completed OCR hints must not contain prices or gold-file phrases."""
    from app import db
    from ocr.prompts import _locator_line

    banned = ("handwritten row", "crossed out", "printed $")
    counts: dict[str, int] = {}
    db.init_db()
    with db.connect() as conn:
        rows = db.knowledge_records_for_sheet(conn, "work")
        counts["work"] = len(rows)
        for rec in rows:
            notes = (rec.get("billing_notes") or "").lower()
            for phrase in banned:
                if phrase in notes:
                    raise SystemExit(
                        f"{rec['name']} note contains {phrase!r}. "
                        "That phrase came from a scored gold file, not an office template."
                    )
            line = _locator_line(rec, "work")
            if re.search(r"\$\s*\d", line):
                raise SystemExit(f"OCR locator includes a dollar amount: {line}")
    return counts


def ensure_qwen() -> float:
    import ocr

    t0 = time.time()
    if model_up(QWEN_ID):
        print(f"{QWEN_ID} already up", flush=True)
        return 0.0
    print(f"loading {QWEN_ID}…", flush=True)
    ocr.load_model()
    for _ in range(90):
        if model_up(QWEN_ID):
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"{QWEN_ID} did not come up")
    return time.time() - t0


def main() -> None:
    import ocr
    from app import db

    fixtures = load_fixtures()
    if not fixtures:
        raise SystemExit("no fixtures")

    counts = assert_live_knowledge()
    print(f"live knowledge counts: {counts}", flush=True)

    def run_model(model_id: str, model_cfg: dict | None, label: str) -> dict:
        per = []
        cache: dict[str, list] = {}
        for fx in fixtures:
            sk = sheet_kind_for(fx)
            if sk not in cache:
                with db.connect() as conn:
                    cache[sk] = db.knowledge_records_for_sheet(conn, sk)
            records = cache[sk]
            print(
                f"  [{label}] {fx['id']} kind={sk} records={len(records)}…",
                flush=True,
            )
            t1 = time.time()
            try:
                extract = ocr.extract_sheet(
                    fx["image"],
                    sheet_kind=sk,
                    knowledge_records=records,
                    chunk_size=CHUNK,
                    model_id=model_id,
                    model_cfg=model_cfg,
                )
                err = None
            except Exception as e:
                extract = {"title": None, "tables": [], "notes": [], "complete": False}
                err = str(e)
            elapsed = time.time() - t1
            sc = score_live(extract, fx["gold"])
            n_rows = sum(
                len(t.get("rows") or [])
                for t in (extract.get("tables") or [])
                if isinstance(t, dict)
            )
            row = {
                "id": fx["id"],
                "sheet_kind": sk,
                "knowledge_names": len(records),
                "seconds": round(elapsed, 2),
                "error": err,
                "n_pred_rows": n_rows,
                "title": extract.get("title"),
                **sc,
                "extract_preview": {
                    "columns": (extract.get("tables") or [{}])[0].get("columns")
                    if extract.get("tables")
                    else [],
                    "sample_rows": ((extract.get("tables") or [{}])[0].get("rows") or [])[:3]
                    if extract.get("tables")
                    else [],
                },
            }
            per.append(row)
            print(
                f"    {elapsed:.1f}s client={sc['client_recall']} "
                f"price={sc['price_recall']} days={sc['day_recall']} "
                f"rows={n_rows} err={err}",
                flush=True,
            )
        times = [r["seconds"] for r in per]
        priced = [r["price_recall"] for r in per if r["price_recall"] is not None]
        return {
            "id": model_id,
            "label": label,
            "per_image": per,
            "total_infer_s": round(sum(times), 2),
            "mean_infer_s": mean(times),
            "row_client_recall": _weighted(per, "matched_clients", "n_gold"),
            "row_price_recall": _weighted(per, "matched_prices", "n_price"),
            "row_day_recall": _weighted(per, "matched_days", "n_days"),
            "page_mean_client_recall": mean([r["client_recall"] for r in per]),
            "page_mean_price_recall": mean(priced) if priced else None,
            "mean_client_recall": _weighted(per, "matched_clients", "n_gold"),
            "mean_price_recall": _weighted(per, "matched_prices", "n_price"),
            "mean_infer_s": mean(times),
        }

    print(
        f"guided OCR bench: {len(fixtures)} fixtures, chunk={CHUNK}",
        flush=True,
    )
    deploy_qwen_s = ensure_qwen()
    qwen = run_model(QWEN_ID, None, "qwen")
    qwen["deploy_s"] = round(deploy_qwen_s, 2)

    result = {
        "mode": "live_office_knowledge",
        "chunk_size": CHUNK,
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "knowledge": counts,
        "qwen": qwen,
        "mean_client_recall": qwen.get("mean_client_recall"),
        "mean_price_recall": qwen.get("mean_price_recall"),
        "mean_infer_s": qwen.get("mean_infer_s"),
        "per_image": qwen.get("per_image") or [],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "qwen_client": qwen.get("mean_client_recall"),
        "qwen_price": qwen.get("mean_price_recall"),
        "qwen_days": qwen.get("row_day_recall"),
        "qwen_s": qwen.get("mean_infer_s"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
