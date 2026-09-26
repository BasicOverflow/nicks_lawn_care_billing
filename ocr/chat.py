"""Chat completions against a deployed ray-hive model (OpenAI route)."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .prompts import TABLE_JSON_SCHEMA


def serve_base() -> str:
    """Ray Serve HTTP root. Does not import ray (that pull needs jinja2 on the app)."""
    explicit = (os.environ.get("RAY_SERVE_URL") or os.environ.get("RAY_SERVE") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    addr = os.environ.get("RAY_ADDRESS", "")
    if addr.startswith("ray://"):
        host = addr.removeprefix("ray://").split(":")[0]
        return f"http://{host}:8000"
    return "http://10.0.1.52:8000"


def file_to_data_url(path: str | Path, mime: str | None = None) -> str:
    path = Path(path)
    if mime is None:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


# Phone photos are ~5k px; shrink so vLLM VL encoders don't OOM / 500.
# DeepSeek-OCR* crop tiling blows up past ~900px on the long edge.
# Novel mode biases toward ~2–3MP (long edge ~2500–2800).
MAX_IMAGE_EDGE = int(os.environ.get("NINI_IMAGE_EDGE", "2000"))
NOVEL_IMAGE_EDGE = int(os.environ.get("NINI_NOVEL_IMAGE_EDGE", "2800"))
DEEPSEEK_MAX_IMAGE_EDGE = int(os.environ.get("NINI_DEEPSEEK_IMAGE_EDGE", "900"))


def novel_mode() -> bool:
    return (os.environ.get("NINI_NOVEL") or "").strip().lower() in ("1", "true", "yes", "on")


def prepare_image(
    path: Path,
    max_edge: int | None = None,
    *,
    enhance: bool = True,
    contrast: bool = False,
) -> Path:
    """Return path to a JPEG no larger than max_edge on the long side (may be temp).

    Applies document OCR preprocessing (CLAHE / illumination flatten / mild
    deskew) unless ``NINI_OCR_PREPROCESS=off`` or enhance=False.
    """
    from PIL import Image, ImageOps

    from .preprocess import contrast_variant, enhance_for_ocr, perspective_rectify

    if max_edge is None:
        max_edge = NOVEL_IMAGE_EDGE if novel_mode() else MAX_IMAGE_EDGE

    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    # Shrink before deskew / contrast. Phone photos are large enough that
    # doing this on the full frame, four times at once, gets the app killed.
    w, h = img.size
    scale = max_edge / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    if novel_mode():
        try:
            img = perspective_rectify(img)
        except Exception:
            pass
    if enhance:
        img = enhance_for_ocr(img)
    if contrast:
        try:
            img = contrast_variant(img)
        except Exception:
            pass
    w, h = img.size
    scale = max_edge / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    out = Path(tempfile.gettempdir()) / (
        f"nini_ocr_{os.getpid()}_{path.stem}_{os.urandom(4).hex()}.jpg"
    )
    img.save(out, format="JPEG", quality=95, optimize=True)
    return out


def chat_with_image(
    model_id: str,
    image_path: Path,
    text: str,
    *,
    max_tokens: int = 2048,
    temperature: float | None = None,
    timeout: int = 600,
    guided_json: bool = True,
    json_schema: dict | None = None,
    schema_name: str = "sheet_tables",
    enhance: bool = True,
    contrast: bool = False,
    max_edge: int | None = None,
    reuse_prepared: bool = False,
) -> str:
    is_deepseek = "deepseek" in model_id.lower() and "ocr" in model_id.lower()
    if temperature is None:
        temperature = 0.0  # deterministic OCR
    edge = DEEPSEEK_MAX_IMAGE_EDGE if is_deepseek else (
        max_edge if max_edge is not None else (NOVEL_IMAGE_EDGE if novel_mode() else MAX_IMAGE_EDGE)
    )
    prepared = image_path if reuse_prepared else prepare_image(
        image_path, max_edge=edge, enhance=enhance, contrast=contrast
    )
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": file_to_data_url(prepared)}},
                    {"type": "text", "text": text},
                ],
            }
        ]
        return _chat_completions(
            model_id,
            messages,
            max_tokens,
            temperature,
            timeout,
            is_deepseek,
            guided_json=guided_json and not is_deepseek,
            json_schema=json_schema,
            schema_name=schema_name,
        )
    finally:
        if not reuse_prepared:
            prepared.unlink(missing_ok=True)


def chat_text(
    model_id: str,
    text: str,
    *,
    max_tokens: int = 2048,
    temperature: float | None = None,
    timeout: int = 600,
    guided_json: bool = True,
    json_schema: dict | None = None,
    schema_name: str = "sheet_tables",
) -> str:
    """Text-only chat (no image) — used for OCR→JSON staging."""
    is_deepseek = "deepseek" in model_id.lower() and "ocr" in model_id.lower()
    if temperature is None:
        temperature = 0.0
    messages = [{"role": "user", "content": text}]
    return _chat_completions(
        model_id,
        messages,
        max_tokens,
        temperature,
        timeout,
        is_deepseek,
        guided_json=guided_json and not is_deepseek,
        json_schema=json_schema,
        schema_name=schema_name,
    )


def _chat_completions(
    model_id: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    timeout: int,
    is_deepseek: bool,
    *,
    guided_json: bool = False,
    json_schema: dict | None = None,
    schema_name: str = "sheet_tables",
) -> str:
    from .cancel import (
        InferenceCancelled,
        is_cancelled,
        new_request_id,
        raise_if_cancelled,
        register_response,
        unregister_response,
    )

    raise_if_cancelled()
    request_id = new_request_id()
    body: dict = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Stream so closing the HTTP connection aborts vLLM generation (VRAM free).
        "stream": True,
    }
    if novel_mode() or temperature == 0.0:
        body["top_p"] = 1.0
        body["repetition_penalty"] = 1.08
        body["top_k"] = -1
    if is_deepseek:
        body["skip_special_tokens"] = False
        body["vllm_xargs"] = {
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": [128821, 128822],
        }
    elif guided_json:
        schema = json_schema or TABLE_JSON_SCHEMA
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema},
        }

    url = f"{serve_base()}/{model_id}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
            "Accept": "text/event-stream",
        },
    )
    resp = None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        register_response(resp)
        chunks: list[str] = []
        while True:
            if is_cancelled():
                try:
                    resp.close()
                except Exception:
                    pass
                _try_server_abort(model_id, request_id)
                raise InferenceCancelled("OCR cancelled by user")
            line = resp.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text or text.startswith(":"):
                continue
            if text.startswith("data:"):
                payload = text[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    frame = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (
                    ((frame.get("choices") or [{}])[0].get("delta") or {}).get("content")
                )
                if delta:
                    chunks.append(str(delta))
                # Some servers send full message on non-delta stream frames
                msg = ((frame.get("choices") or [{}])[0].get("message") or {}).get("content")
                if msg and not delta:
                    chunks.append(str(msg))
                err = frame.get("error")
                if err:
                    message = err.get("message") if isinstance(err, dict) else str(err)
                    raise RuntimeError(message[:500])
        if not chunks:
            raise RuntimeError("The model returned no text.")
        return "".join(chunks)
    except InferenceCancelled:
        raise
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        # Fallback: non-stream if server rejects stream
        if e.code in (400, 422) and "stream" in detail.lower():
            body.pop("stream", None)
            return _chat_completions_nonstream(
                model_id, body, timeout, request_id=request_id
            )
        raise RuntimeError(f"HTTP {e.code}: {detail[:2000]}") from e
    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        if is_cancelled():
            raise InferenceCancelled("OCR cancelled by user") from e
        raise
    finally:
        if resp is not None:
            unregister_response(resp)
            try:
                resp.close()
            except Exception:
                pass


def _chat_completions_nonstream(
    model_id: str, body: dict, timeout: int, *, request_id: str
) -> str:
    """Non-stream fallback; still closes connection on cancel mid-read."""
    from .cancel import InferenceCancelled, is_cancelled, raise_if_cancelled, register_response, unregister_response

    raise_if_cancelled()
    req = urllib.request.Request(
        f"{serve_base()}/{model_id}/v1/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
    )
    resp = None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        register_response(resp)
        raw = resp.read()
        if is_cancelled():
            raise InferenceCancelled("OCR cancelled by user")
        data = json.loads(raw.decode())
        content = data["choices"][0]["message"].get("content")
        return content if content is not None else ""
    except InferenceCancelled:
        raise
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail[:2000]}") from e
    finally:
        if resp is not None:
            unregister_response(resp)
            try:
                resp.close()
            except Exception:
                pass


def _try_server_abort(model_id: str, request_id: str) -> None:
    """Best-effort vLLM abort endpoint (version-dependent)."""
    for path in (
        f"/{model_id}/abort_requests",
        f"/{model_id}/v1/abort_requests",
        "/abort_requests",
    ):
        try:
            req = urllib.request.Request(
                f"{serve_base()}{path}",
                data=json.dumps({"request_ids": [request_id]}).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as _:
                return
        except Exception:
            continue
