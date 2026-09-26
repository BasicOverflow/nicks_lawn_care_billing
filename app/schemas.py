"""Request bodies for the billing API.

Field text is what Swagger shows next to each parameter.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UploadCancel(BaseModel):
    """Stop OCR that is running, or drop a batch that has not started."""

    all: bool = Field(
        False,
        description="Cancel the running batch and drop every upload still waiting.",
    )
    job_id: str | None = Field(
        None,
        description="Upload id to cancel. Omit to stop whichever batch is running.",
    )


class CorrectJob(BaseModel):
    """Plain-language fix applied to one OCR result."""

    instruction: str = Field(
        ...,
        description="What to change, in everyday language. Example: set the hedge price for SMITH to 80.",
    )
    month: str | None = Field(
        None,
        description="Billing month YYYY-MM. Used when checking the corrected rows for conflicts.",
    )


class CommitJob(BaseModel):
    """Write a reviewed OCR result into clients and work for one month."""

    month: str = Field(..., description="Billing month, YYYY-MM.")
    sheet_kind: str = Field(
        "mowing",
        description="Which price column the sheet is: mowing or hedges.",
    )
    conflict_nl: str = Field(
        "",
        description="How to resolve clashes with data already stored. Required when conflicts exist.",
    )


class ChatTurn(BaseModel):
    """One earlier message in the data chat."""

    role: str = Field(..., description="user or assistant.")
    content: str = Field("", description="Message text.")


class ChatRequest(BaseModel):
    """Question or edit sent to the data chat."""

    message: str = Field("", description="What to ask or change.")
    question: str | None = Field(
        None,
        description="Same as message. Older clients send this name instead.",
    )
    history: list[ChatTurn] = Field(
        default_factory=list,
        description="Earlier turns. The new message is not included.",
    )


class GenerateBills(BaseModel):
    """Build PDFs for every client who has work in a month."""

    month: str = Field(..., description="Billing month, YYYY-MM.")


class BillLineIn(BaseModel):
    """One line on an editable bill."""

    id: int | None = Field(
        None,
        description="Existing work-item id. Omit or null to add a new line.",
    )
    description: str = Field("", description="Line text printed on the bill.")
    amount: float | str | None = Field(
        None,
        description="Dollar amount. Blank becomes 0.",
    )


class SaveBill(BaseModel):
    """Replacement contact info and line items for one client's bill."""

    email: str = Field("", description="Client email. Empty means no email on file.")
    address: str = Field("", description="Mailing address printed on the bill.")
    lines: list[BillLineIn] = Field(
        ...,
        description="Full set of lines for this client and month. Lines left out are removed.",
    )


class QueuedUpload(BaseModel):
    """One sheet batch waiting behind the current OCR pass."""

    job_id: str = Field(..., description="Upload id. Cancel this id to drop it before it starts.")
    month: str = Field("", description="Billing month YYYY-MM.")
    sheet_kind: str = Field("", description="mowing, hedges, or work.")
    n_files: int = Field(0, description="How many photos are in the batch.")
    status: str = Field("", description="queued until the worker picks it up.")


class ProgressView(BaseModel):
    """Live status the UI polls while a background job runs."""

    job_id: str = Field("", description="Id of the job in progress. Empty when nothing has run.")
    kind: str = Field(
        "",
        description="upload, ocr, generate, email, model, or chat. Empty while idle.",
    )
    status: str = Field(
        "idle",
        description="idle, running, done, error, or cancelled.",
    )
    message: str = Field("", description="Short status line shown in the UI.")
    percent: int = Field(0, description="0–100.")
    detail: dict = Field(
        default_factory=dict,
        description="Job result. Shape depends on kind: chat puts answer here, OCR puts extract and conflicts.",
    )
    queue_depth: int = Field(0, description="Upload batches still waiting.")
    queued: list[QueuedUpload] = Field(default_factory=list, description="Those waiting batches.")
    ocr_busy: bool = Field(False, description="True while a batch is running or one is queued.")


class ModelStatus(BaseModel):
    """Whether the vision model route is up on ray-hive."""

    id: str = Field("qwen25-vl-3b", description="ray-hive model id. Always Qwen2.5-VL-3B.")
    up: bool = Field(..., description="True when GET {base_url}/v1/models succeeds.")
    base_url: str = Field("", description="OpenAI-compatible base for that model.")


class AcceptedJob(BaseModel):
    """Handle for work that finishes later on GET /api/progress."""

    job_id: str = Field(..., description="Poll GET /api/progress until status is done, error, or cancelled.")


class UploadAccepted(BaseModel):
    """A photo batch that was stored and placed on the OCR queue."""

    job_id: str = Field(..., description="Upload id. Use it to commit, correct, or cancel this batch.")
    month: str = Field(..., description="Billing month the batch was filed under, YYYY-MM.")
    state: str = Field(..., description="queued if OCR is busy, otherwise running.")
    queue_depth: int = Field(..., description="Batches waiting, including this one when it was queued.")
    n_files: int = Field(..., description="Photos accepted.")
