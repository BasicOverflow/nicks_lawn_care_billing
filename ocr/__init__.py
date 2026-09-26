"""Isolated OCR package — knowledge-guided parallel full-image OCR via ray-hive."""

from __future__ import annotations

import json
import os
from pathlib import Path

MODEL_ID = "qwen25-vl-3b"
MODEL_CFG = {
    "id": MODEL_ID,
    "hf_name": "Qwen/Qwen2.5-VL-3B-Instruct",
    "max_input": 3072,
    "max_output": 4096,
    "novel": True,
    "vllm_kwargs": {
        "trust_remote_code": True,
        "limit_mm_per_prompt": {"image": 1},
        "mm_processor_cache_gb": 0,
        "max_num_seqs": 4,
        "mm_processor_kwargs": {"max_pixels": 3211264, "min_pixels": 401408},
    },
    "prompt_style": "chat_ocr_md",
}

# How many knowledge names per parallel full-image request
GUIDED_CHUNK_SIZE = int(os.environ.get("NINI_OCR_CHUNK", "4"))


def _ensure_env() -> None:
    os.environ.setdefault("NINI_NOVEL", "1")
    os.environ.setdefault("NINI_IMAGE_EDGE", "2800")
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")


def _load_knowledge_records(sheet_kind: str = "mowing") -> list[dict]:
    """Pull guiding client rows from Postgres (live knowledge, not gold)."""
    try:
        from app import db

        with db.connect() as conn:
            return db.knowledge_records_for_sheet(conn, sheet_kind)
    except Exception as e:
        print(f"  knowledge load failed ({e}); guided OCR will fall back", flush=True)
        return []


def extract_sheet(
    image_path: Path | str,
    *,
    max_passes: int = 4,
    sheet_kind: str = "mowing",
    knowledge_names: list[str] | None = None,
    knowledge_records: list[dict] | None = None,
    chunk_size: int | None = None,
    model_id: str | None = None,
    model_cfg: dict | None = None,
) -> dict:
    """Knowledge-guided OCR: parallel full-image chunks (no tile/crop split).

    ``max_passes`` is accepted for API compatibility; guided mode does not use it.
    """
    _ = max_passes
    _ensure_env()
    from .guided import extract_guided

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    records = knowledge_records
    if records is None:
        records = _load_knowledge_records(sheet_kind)
        if knowledge_names is not None:
            wanted = {n.strip().lower() for n in knowledge_names}
            filtered = [r for r in records if str(r.get("name") or "").strip().lower() in wanted]
            records = filtered or [{"name": n} for n in knowledge_names if str(n).strip()]

    cfg = dict(MODEL_CFG)
    if model_cfg:
        cfg.update(model_cfg)
    obj = extract_guided(
        model_id or MODEL_ID,
        cfg,
        path,
        knowledge_records=records,
        sheet_kind=sheet_kind,
        chunk_size=chunk_size or GUIDED_CHUNK_SIZE,
        fx_id=path.stem,
    )
    obj.pop("_guided_meta", None)
    return obj


def extract_sheet_legacy_multipass(
    image_path: Path | str,
    *,
    max_passes: int = 4,
) -> dict:
    """Previous novel multipass tile/crop pipeline (benches / fallback)."""
    _ensure_env()
    from .pipeline import run_fixture

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    fx = {"id": path.stem, "image": str(path)}
    row = run_fixture(
        MODEL_ID,
        dict(MODEL_CFG),
        fx,
        max_passes,
        wave_workers=None,
        asset_dir=None,
        image_path=path,
    )
    raw = row.get("final_extract") or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"title": None, "tables": [], "notes": [raw[:4000]], "complete": False}


def extract_sheet_gemma_rows(
    image_path: Path | str,
    *,
    model_id: str = "gemma4-e2b-ocr",
    rows_per_crop: int = 3,
    max_seqs: int = 4,
) -> dict:
    """Gemma path: black-grid row groups (2–4 rows/crop) in parallel."""
    _ensure_env()
    from .gemma_rows import extract_by_row_groups

    return extract_by_row_groups(
        model_id,
        Path(image_path),
        rows_per_crop=rows_per_crop,
        max_seqs=max_seqs,
    )


def apply_correction(
    image_path: Path | str | None,
    table_json: dict,
    instruction: str,
) -> dict:
    """Apply a plain-English correction to an extract (optional image re-look)."""
    _ensure_env()
    from .chat import chat_text, chat_with_image
    from .jsonutil import try_parse_json

    prompt = (
        "You are correcting a JSON extract of a paper sheet.\n\n"
        f"Current JSON:\n{json.dumps(table_json, indent=2, ensure_ascii=False)[:12000]}\n\n"
        f"User correction:\n{instruction.strip()}\n\n"
        "Return the FULL corrected JSON only (same schema: title, tables, notes, complete). "
        "Do not invent rows. Apply the correction faithfully."
    )
    if image_path and Path(image_path).is_file():
        text = chat_with_image(
            MODEL_ID,
            Path(image_path),
            prompt,
            max_tokens=int(MODEL_CFG["max_output"]),
            temperature=0.0,
            guided_json=True,
        )
    else:
        text = chat_text(
            MODEL_ID,
            prompt,
            max_tokens=int(MODEL_CFG["max_output"]),
            temperature=0.0,
            guided_json=True,
        )
    obj, _ = try_parse_json(text or "")
    if isinstance(obj, dict):
        obj.setdefault("title", None)
        obj.setdefault("tables", [])
        obj.setdefault("notes", [])
        obj.setdefault("complete", False)
        return obj
    return table_json


def model_status() -> dict:
    """Return {id, up, base_url}."""
    _ensure_env()
    import urllib.request

    from ray_hive.core.ray_utils import serve_base_url

    base = serve_base_url()
    up = False
    try:
        with urllib.request.urlopen(f"{base}/{MODEL_ID}/v1/models", timeout=5) as r:
            up = r.status == 200
    except Exception:
        up = False
    return {"id": MODEL_ID, "up": up, "base_url": f"{base}/{MODEL_ID}"}


def load_model() -> None:
    """Deploy qwen25-vl-3b on ray-hive (blocks until ready)."""
    from .deploy import deploy_model

    _ensure_env()
    deploy_model(MODEL_ID, MODEL_CFG)


def unload_model() -> None:
    from .deploy import shutdown_model

    _ensure_env()
    shutdown_model(MODEL_ID)
