"""Prompts for sheet-agnostic table OCR (any work log, roster, or list)."""

from __future__ import annotations

import json
import re

# Cells are strings so later pipelines can interpret prices, dates, names.
# additionalProperties allowed so guided decoding never drops unexpected fields.
TABLE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "title": {"type": ["string", "null"]},
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "caption": {"type": ["string", "null"]},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "rows": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "required": ["caption", "columns", "rows"],
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
        "complete": {"type": "boolean"},
    },
    "required": ["title", "tables", "notes", "complete"],
}

TRANSCRIBE = """
Transcribe the sheet into JSON tables that MATCH the grids on the page EXACTLY.

Goal: reconstruct EVERYTHING visible — every column header, every cell, every row —
as written. Do not normalize, summarize, or reinterpret.

Rules:
- columns = headers left-to-right EXACTLY as printed on the sheet.
  Example mowing sheets often have: Contact | Address | New Price | Billing Address / Notes
  Those are FOUR different columns. Do not merge or drop any of them.
- "Address" (street number + road) is NOT the same as "Billing Address / Notes"
  (email, phone, REGULAR MAIL, service notes). Keep BOTH when both appear.
- NEVER invent columns that are not on the sheet (do not add Days, Email, Notes, etc.
  unless those headers are actually printed).
- Each row = ONE horizontal line. Keep every cell on that line in order, including blanks as "".
- NEVER invent values. If unreadable, use "" or "[UNCLEAR]".
- NEVER split one visual grid into multiple tables.
- Copy FULL names as written (e.g. "SMITH, Jane").
- Every cell is a string. Copy handwriting and strikeouts' replacement text as written.
- Do not drop rows. Margin text that is not part of a grid goes in "notes".
- Set "complete" true only if every visible row, cell, and note is in the JSON.

Return ONLY JSON:
{
  "title": string | null,
  "tables": [{"caption": string | null, "columns": [string], "rows": [[string]]}],
  "notes": [string],
  "complete": boolean
}
""".strip()


def tile_prompt() -> str:
    return (
        TRANSCRIBE
        + "\n\nThis image is one HORIZONTAL BAND of a larger page. "
        "Extract every row visible in this band only. "
        "Keep the same left-to-right columns as the full sheet "
        "(include street Address AND Billing Address / Notes when both exist)."
    )


def unify_prompt(so_far_json: str) -> str:
    """Full-page rewrite: force one grid-faithful table layout using tile hints."""
    return f"""
You are correcting a partial extract of this sheet. Draft so far:

{so_far_json}

Look at the FULL page photo. Rebuild the JSON so it MATCHES the table grid(s) EXACTLY:
- columns = every printed header left-to-right (copy header text as written).
- If the sheet has both "Address" and "Billing Address / Notes", output BOTH columns.
  Street addresses go under Address; emails / REGULAR MAIL / phones go under Billing.
- Do NOT invent columns (no Days/Email/Notes unless printed on the sheet).
- Each row = one horizontal line; FULL contact name as written.
- Fill every visible cell. Do not invent clients or prices.
- If the draft merged or dropped columns, restore the real sheet layout.

Return the FULL corrected JSON only. Set "complete" false unless nothing is missing.
""".strip()


def completeness_prompt(so_far_json: str) -> str:
    return f"""
This is everything extracted from this sheet so far:

{so_far_json}

Look at the photo again. Is that everything?
- Headers must match the sheet. Restore any missing column (especially street Address
  when Billing Address / Notes is present).
- Every visible cell on every row must be filled when present on the page.
- Do not invent columns or values. Do not drop columns.
- Keep ONE table per visual grid.
- If anything is missing or wrong, return the FULL updated JSON with the same sheet layout.
- If nothing visible is missing, return the same content and set "complete" to true.

Return ONLY the JSON object.
""".strip()


def name_column_repair_prompt(so_far_json: str) -> str:
    """Triggered when extract starts with addresses and drops the contact column."""
    return f"""
The draft JSON is missing the left-hand contact/name column of the sheet:

{so_far_json}

Look at the photo. Rebuild the FULL JSON with columns matching the sheet headers exactly
(left-to-right). Each row must start with the full name as written, then every other cell.
Do not invent data. Do not invent columns. Return ONLY the corrected JSON.
""".strip()


def surname_repair_prompt(so_far_json: str) -> str:
    """Triggered when contact cells look truncated (first name only)."""
    return f"""
The draft has truncated contact names (missing surnames):

{so_far_json}

Look at the photo. Rewrite EVERY contact/name cell as the FULL name written on the sheet
(e.g. "SMITH, Jane" not "Jane"). Keep every other column as-is.
Do not invent rows. Return ONLY the corrected JSON.
""".strip()


def grid_repair_prompt(so_far_json: str) -> str:
    """Triggered when headers look invented or street Address column is missing."""
    return f"""
The draft does not match the sheet's real column layout:

{so_far_json}

Look at the photo's header row carefully. Rebuild the FULL JSON so that:
- columns[] are EXACTLY the printed headers left-to-right (typically Contact, Address,
  New Price, Billing Address / Notes on mowing sheets — use whatever is actually printed).
- Address = street number + road name for each contact.
- Billing Address / Notes = email, phone, REGULAR MAIL, or service notes — NOT the street.
- Remove invented columns that are not printed (e.g. Days, Email, Notes) unless those
  headers appear on the sheet.
- Keep every row; copy every cell from the matching horizontal line.

Return ONLY the corrected JSON.
""".strip()


def price_focus_prompt(so_far_json: str) -> str:
    return f"""
Draft extract:

{so_far_json}

Look at the photo's price column only (e.g. "New Price" / "Hedge"). Rewrite every price
cell to the exact amount written (usually like $32, $53). Common mistake: $53 → 533.
Keep ALL columns and rows unchanged otherwise. Return ONLY the full updated JSON.
""".strip()


def address_focus_prompt(so_far_json: str) -> str:
    return f"""
Draft extract:

{so_far_json}

Look at the photo's STREET Address column (NOT "Billing Address / Notes").
For EVERY row, fill or correct the Address cell with the street number + road name on that line
(e.g. "123 Main St.").
Do NOT put emails, phones, or REGULAR MAIL in Address — those belong in Billing Address / Notes.
Do not invent addresses. Keep Contact, Price, and Billing cells. Return ONLY the full updated JSON.
""".strip()


def tail_completeness_prompt(so_far_json: str) -> str:
    """Catch missing rows at the bottom / short pages without gold hints."""
    return f"""
Draft extract so far:

{so_far_json}

Look especially at the BOTTOM third of the photo and any rows after the last contact
already in the JSON. Add any missing full rows that are visible (Contact, Address,
Price, Billing Address / Notes — matching the sheet headers).
Do not invent rows. Do not drop existing correct rows. Keep exact column headers.
Return ONLY the full updated JSON.
""".strip()


def column_strip_prompt(kind: str) -> str:
    """kind: name | addr | price | billing — vertical band of the sheet."""
    focus = {
        "name": "Contact/name column (FULL names as written)",
        "addr": "street Address column only (number + road, e.g. 123 Main St.) — NOT billing/email",
        "price": "price column (amounts like $32, $53 — not 533)",
        "billing": "Billing Address / Notes column (emails, phones, REGULAR MAIL, service notes)",
    }.get(kind, "visible cells")
    return f"""
This image is a VERTICAL STRIP of a larger sheet (one column region).

Extract every visible row in this strip into JSON tables.
Focus on the {focus}.
Use a matching column header. Keep row order top-to-bottom.
Copy handwriting exactly. Do not invent values. Use "" for blank cells.

Return ONLY JSON with title/tables/notes/complete.
""".strip()


def row_group_prompt(columns: list[str] | None = None, n_rows: int = 3) -> str:
    """Prompt for a crop with 2–4 consecutive table rows (not the full page)."""
    cols = columns or ["Contact", "Address", "New Price", "Billing Address / Notes"]
    col_line = " | ".join(cols)
    return f"""
This image is a HORIZONTAL STRIP of a printed table on white paper (black grid lines).
It is a padded crop of about {n_rows} consecutive data rows — NOT the whole page.
Extra context may appear at the TOP and BOTTOM edges (neighboring rows partially visible).

Columns left-to-right (keep ALL of them, including blanks as ""):
{col_line}

Rules:
- Extract ONLY rows that are FULLY visible — not cut off on ANY border (top, bottom, left, or right).
- If a row is clipped at the edge (partial letters, cut-off cells, missing tops/bottoms of handwriting), SKIP it entirely. Do not guess clipped text.
- Same for cells: if a cell is cut off at the left/right edge, leave that cell as "" (or skip the whole row if Contact is clipped).
- Of the fully-inside rows, extract EVERY one top → bottom. One JSON row per complete horizontal line.
- Do not invent rows. Prefer fewer complete rows over inventing from partial edge content.
- Copy handwriting exactly (e.g. "SMITH, Jane"). Unreadable → "" or "[UNCLEAR]".
- Prices like $32 / $53 — do not invent digits.
- Ignore anything outside the table cells.

Return ONLY JSON:
{{
  "title": null,
  "tables": [{{"caption": null, "columns": {json.dumps(cols)}, "rows": [["...", ...], ...]}}],
  "notes": [],
  "complete": true
}}
""".strip()


def price_contact_strip_prompt() -> str:
    """Focused Contact + Price columns for a second pass."""
    return """
This image is a VERTICAL STRIP of a ruled table showing mainly the Contact (name)
column and the Price column (New Price / Hedge amounts like $32, $53, $95).

Extract EVERY fully visible data row top → bottom as:
["Contact name", "price as written"]

Rules:
- Skip header rows (Contact / New Price / Hedge).
- Skip rows cut off at the top or bottom edge.
- Copy names exactly. Prices keep the $ if printed. Crossed-out → use the handwritten replacement.
- Do not invent names or prices. Blank price → "".

Return ONLY JSON:
{
  "title": null,
  "tables": [{"caption": null, "columns": ["Contact", "Price"], "rows": [["SMITH, Jane", "$95"], ...]}],
  "notes": [],
  "complete": true
}
""".strip()


def row_window_prompt() -> str:
    """Legacy multipass window prompt (several consecutive rows, full columns)."""
    return (
        TRANSCRIBE
        + "\n\nThis image is a HORIZONTAL WINDOW covering several consecutive rows. "
        "Extract every full row visible here with ALL sheet columns "
        "(Contact, Address, Price, Billing Address / Notes when present)."
    )


def header_columns_prompt() -> str:
    return """
This crop is the HEADER of a printed table (column titles on white paper).
Read the column headers left-to-right EXACTLY as printed.
Return ONLY JSON:
{"columns": ["Contact", "Address", ...]}
Do not invent columns that are not printed.
""".strip()



REGION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "x0": {"type": "number"},
                    "y0": {"type": "number"},
                    "x1": {"type": "number"},
                    "y1": {"type": "number"},
                },
                "required": ["label", "x0", "y0", "x1", "y1"],
            },
        }
    },
    "required": ["regions"],
}

SMART_SPLIT_PROMPT = """
Look at this phone photo of a paper sheet with a table.

Propose 4–10 non-overlapping (or slightly overlapping) horizontal REGIONS that
smartly cover the readable table content from top to bottom. Prefer splitting
BETWEEN rows (along ruling lines), not through handwriting.

Coordinates are normalized fractions of the image: 0 = left/top, 1 = right/bottom.
Each region should be wide enough to include all columns (x0 near 0, x1 near 1)
unless a region is a single column strip.

Return ONLY JSON:
{"regions":[{"label":"rows 1-8","x0":0,"y0":0.08,"x1":1,"y1":0.35}, ...]}
""".strip()


DOC_OCR_PROMPT = """
Extract everything visible on this sheet of paper.

Requirements:
- Preserve table structure (headers and every row).
- Transcribe typed text exactly.
- Transcribe handwritten text as accurately as possible.
- Use "" or [unclear] for unreadable cells — do not invent text.
- Prefer Markdown with an HTML <table> if helpful, or plain Markdown tables.

Return structured content only (no preamble).
""".strip()


FAITHFUL_MD_PROMPT = """
Transcribe this document EXACTLY.

Rules:
- Preserve the reading order.
- Preserve table structure (Markdown or HTML <table>).
- Preserve line breaks where meaningful.
- Transcribe handwritten and typed text.
- Do not infer or correct spelling.
- Do not paraphrase.
- If a character or word cannot be confidently read, output [UNCLEAR].
- Do not invent missing text.
- For tables, preserve rows and columns.
- Distinguish handwritten text from printed text when possible
  (e.g. mark cells with (hw) suffix only when clearly handwritten).

Return transcription only — no preamble.
""".strip()


def struct_from_text_prompt(transcription: str) -> str:
    return f"""
Given the following faithful transcription of a paper sheet, convert it into
JSON tables that MATCH the sheet grids EXACTLY.

Transcription:
{transcription[:14000]}

Rules:
- columns = headers left-to-right as written.
- Each row = one horizontal line; keep blanks as "".
- NEVER invent values. Use "" for [UNCLEAR] or missing cells.
- Keep Address and Billing Address / Notes as separate columns when both appear.
- notes = margin text not in the grid.
- complete = false unless the transcription clearly covers every row.

Return ONLY JSON:
{{
  "title": string | null,
  "tables": [{{"caption": string | null, "columns": [string], "rows": [[string]]}}],
  "notes": [string],
  "complete": boolean
}}
""".strip()


LAYOUT_KIND_PROMPT = """
Look at this page photo. Propose regions by KIND for a document OCR pipeline.

Return 3–12 regions covering the readable content. Prefer:
- kind "table" for the main grid(s)
- kind "handwriting" for handwritten notes / annotations
- kind "text" for printed headers / titles
- kind "price" for a dense price column strip if useful

Coordinates are normalized 0..1 (x0,y0,x1,y1). Slight overlap (5–15%) is OK.

Return ONLY JSON:
{"regions":[{"label":"main table","kind":"table","x0":0,"y0":0.1,"x1":1,"y1":0.9}, ...]}
""".strip()


LAYOUT_KIND_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "kind": {"type": "string"},
                    "x0": {"type": "number"},
                    "y0": {"type": "number"},
                    "x1": {"type": "number"},
                    "y1": {"type": "number"},
                },
                "required": ["label", "kind", "x0", "y0", "x1", "y1"],
            },
        }
    },
    "required": ["regions"],
}


PAGE_ORIENT_PROMPT = """
This image is a montage of the SAME phone photo of a printed client/table sheet
(after EXIF), shown in different rotations. Panel labels: {labels}.

Pick the ONE panel where:
1. Text reads normally left→right, top→bottom (NOT sideways, NOT upside-down).
2. Table rows run HORIZONTALLY.
3. Column headers (Contact / Address / Price / Notes) sit across the TOP.

Return ONLY JSON:
{{"choice": one of [{labels}], "rotate_cw_deg": matching degrees, "reason": "short why"}}
Mapping: A=0, B=90, C=180, D=270 (only labels shown are valid).
""".strip()


PAGE_ORIENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "choice": {"type": "string"},
        "rotate_cw_deg": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["choice", "rotate_cw_deg", "reason"],
}


PAGE_ORIENT_VERIFY_PROMPT = """
Look at this single page photo (already rotated once).

Is the printed table upright for reading?
- Text left→right, top→bottom
- Row lines horizontal
- Headers at the TOP

If YES: ok=true and fix_rotate_cw_deg=0.
If NO: ok=false and set fix_rotate_cw_deg to how many MORE degrees clockwise
to fix it (90, 180, or 270).

Return ONLY JSON:
{"ok": true|false, "fix_rotate_cw_deg": 0|90|180|270, "reason": "short"}
""".strip()


PAGE_ORIENT_VERIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ok": {"type": "boolean"},
        "fix_rotate_cw_deg": {"type": "integer", "enum": [0, 90, 180, 270]},
        "reason": {"type": "string"},
    },
    "required": ["ok", "fix_rotate_cw_deg", "reason"],
}


def guided_work_prompt(names: list[str], columns: list[str], title: str | None = None) -> str:
    """Visible work-log rows only. No roster address, phone, email, or price."""
    col_line = " | ".join(columns)
    title_bit = f'Title: "{title}".\n' if title else ""
    if names:
        who = (
            "Look for these names. Return a row only when that name is written on the page "
            "and the work cell on that line is not empty:\n"
            + "\n".join(f"- {n}" for n in names)
        )
    else:
        who = (
            "Return every row that has work written in the work cell. "
            "Skip a name when nothing is written beside it."
        )
    return f"""
This is a photo of a work-completed sheet, not a price list.
{title_bit}Transcribe ONLY the table printed on the paper.

Columns left to right, exactly as printed:
{col_line}

{who}

Rules:
- One row is one horizontal line on the paper.
- Copy the name and everything written in the work cell on that line. Nothing else.
- The numbers in that cell are days of this month. Each number is its own day. They are not a month/day date.
- A cell that shows 9, then 16, then 23 means day 9, day 16, and day 23. Write every number, separated by spaces.
- Do not join days with a slash or a hyphen. Do not drop a day to form a pair.
- WRONG: "9/16" or "9/16/23" or "9/23"
- RIGHT: "9 16 23"
- A letter written on a day stays with that day. 19h is one mark. RIGHT: "9 16 19h 23"
- Words written in the cell are details about a job. Copy them too, in the same cell, after the days they belong to.
- RIGHT: "5 11 22 Bush trimming $50"
- RIGHT: "14 paid"
- Do not add address, phone, email, mowing price, hedge price, or billing notes. Those are not columns on this sheet and must not be copied from memory.
- Skip a row when the work cell is blank. Do not return that name, and do not use "" to stand for missing work.
- A printed name with no days, hedge mark, or job note is an empty row. Leave it out.
- Omit anyone who is not written on this page.
- Do not add columns. Put every day and every job note for that person in the one work cell.

Return ONLY JSON:
{{
  "title": {json.dumps(title)},
  "tables": [{{"caption": null, "columns": {json.dumps(columns)}, "rows": [["...", "..."]]}}],
  "notes": [],
  "complete": true
}}
""".strip()


def guided_header_prompt(sheet_kind: str) -> str:
    kind = (sheet_kind or "mowing").lower()
    hint = {
        "hedges": "Often titled HEDGES CLIENT LIST with Contact | Address | Hedge | Notes.",
        "mowing": "Often Contact | Address | New Price | Billing Address / Notes.",
        "work": "Often CLIENT | DATE & WORK COMPLETED (monthly work log).",
        "work_completed": "Often CLIENT | DATE & WORK COMPLETED (monthly work log).",
    }.get(kind, "Read whatever headers are printed.")
    return f"""
Look at this full photo of a paper sheet. Do NOT extract every data row.

Return JSON with:
- title: the printed title if any
- tables: ONE table with columns[] = the printed headers left-to-right EXACTLY,
  and rows: [] (empty — headers only)
- notes: any margin / footer notes that are not part of the grid
- complete: false

Hint for this sheet kind ({kind}): {hint}
Return ONLY JSON.
""".strip()


def _locator_line(rec: dict, sheet_kind: str) -> str:
    """Known Postgres fields that help find the row. The sheet still wins.

    Dollar amounts on file stay out of this line. Filed prices are for billing,
    and the photographed cell is often a handwritten correction of that price.
    """
    if isinstance(rec, str):
        return rec
    bits = [str(rec.get("name") or "").strip()]
    if rec.get("address"):
        bits.append(f"service address: {rec['address']}")
    if rec.get("phone"):
        bits.append(f"phone: {rec['phone']}")
    if rec.get("email"):
        bits.append(f"email: {rec['email']}")
    elif rec.get("prefer_mail"):
        bits.append("billed by regular mail")
    kind = (sheet_kind or "mowing").lower()
    if kind in ("work", "work_completed"):
        bits.append("work dates are not on file — read them from the sheet")
    notes = _notes_without_money(str(rec.get("billing_notes") or ""))
    if notes:
        bits.append(f"notes: {notes[:90]}")
    return " — ".join(bits)


_MONEY_TEXT = re.compile(r"\$\s*\d+(?:\.\d+)?|\b\d+\.\d{2}\b")


def _notes_without_money(notes: str) -> str:
    cleaned = _MONEY_TEXT.sub("", notes or "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;|/])", r"\1", cleaned)
    return cleaned.strip(" |;-")


def guided_chunk_prompt(
    names: list,
    *,
    columns: list[str],
    sheet_kind: str = "mowing",
    title: str | None = None,
) -> str:
    """Ask the model to pull only the listed knowledge rows from the full page."""
    col_line = " | ".join(columns)
    bullets = "\n".join(f"- {_locator_line(n, sheet_kind)}" for n in names)
    title_bit = f'Title (if helpful): "{title}".\n' if title else ""
    kind = (sheet_kind or "mowing").lower()
    value_hint = {
        "hedges": "Copy Address, Hedge price, and Notes cells for each listed contact that appears.",
        "work": "Copy every day number in DATE & WORK COMPLETED as its own day of the month, separated by spaces, plus any job words on that line. Never join them into a date like 9/16.",
        "work_completed": "Copy every day number in DATE & WORK COMPLETED as its own day of the month, separated by spaces, plus any job words on that line. Never join them into a date like 9/16.",
        "mowing": "Copy Address, New Price (or Lawn), and Billing Address / Notes for each listed contact that appears.",
    }.get(kind, "Copy every cell on that contact's row.")
    return f"""
This is a FULL photo of a paper sheet (not a crop).
{title_bit}From the image, pull the table values ONLY for these known clients.
Each line starts with who they are. Address, phone, email, and notes are already on file.
Use them only to find the correct horizontal row. Transcribe the cells written on that row.

{bullets}

Columns left-to-right (use these headers):
{col_line}

Rules:
- {value_hint}
- The Contact/CLIENT cell is the person's name, never the street address.
- The price/Hedge cell is the dollar amount on that same line. Do not repeat the street there.
- If the price cell is blank, use "". Handwriting that replaces a crossed-out number wins.
- Do not copy a neighboring row's price.
- One JSON row per listed name that is ACTUALLY visible. Omit names that are not on this sheet.
- Unreadable cells are "" or "[UNCLEAR]". Do not invent days, addresses, or prices.

Return ONLY JSON:
{{
  "title": {json.dumps(title)},
  "tables": [{{"caption": null, "columns": {json.dumps(columns)}, "rows": [["...", ...], ...]}}],
  "notes": [],
  "complete": true
}}
""".strip()


def guided_price_prompt(records: list, *, sheet_kind: str = "mowing") -> str:
    """Second look at price cells that were blank, copied, or filled with a street."""
    names = []
    for rec in records:
        names.append(str(rec.get("name") if isinstance(rec, dict) else rec).strip())
    bullets = "\n".join(f"- {n}" for n in names if n)
    kind = "hedge" if (sheet_kind or "").lower() == "hedges" else "lawn / new price"
    return f"""
This is a FULL photo of a paper sheet.
For each name, read ONLY the {kind} dollar amount on that same horizontal row.

The price is a dollar amount written in the price column.
It is NOT the street address. Do not return a road name.
WRONG: ["SMITH, Jane", "123 Main St."]
RIGHT: ["SMITH, Jane", "$7"]

If the price cell is blank or crossed out with no replacement, return "".
Do not reuse the price from the row above or below.

{bullets}

Return ONLY JSON:
{{
  "title": null,
  "tables": [{{"caption": null, "columns": ["Contact", "Price"], "rows": [["SMITH, Jane", "$7"]]}}],
  "notes": [],
  "complete": true
}}
""".strip()


def guided_unknown_prompt(
    known_names: list[str],
    *,
    columns: list[str],
    sheet_kind: str,
    title: str | None = None,
) -> str:
    """Contacts visible on the sheet who are not in the knowledge list."""
    shown = known_names[:80]
    extra = ""
    if len(known_names) > len(shown):
        extra = f"\n({len(known_names) - len(shown)} more known names omitted.)"
    bullets = "\n".join(f"- {n}" for n in shown)
    col_line = " | ".join(columns)
    title_bit = f'Title hint: "{title}".\n' if title else ""
    return f"""
This is a FULL photo of a paper sheet ({sheet_kind}).
{title_bit}These clients are already known. Do not return them:
{bullets}{extra}

Return every OTHER contact or parcel row that is visible and is not in that list.
Columns:
{col_line}

Rules:
- Contact/CLIENT is the name as written, not the street.
- Copy the price and notes on that same line. Blank → "".
- If every visible row is already in the known list, return rows: [].

Return ONLY JSON with those new rows only.
""".strip()


def guided_gap_prompt(
    missing: list,
    blank_names: list[str],
    extracted_names: list[str],
    *,
    columns: list[str],
    sheet_kind: str,
    title: str | None = None,
    ask_others: bool = False,
) -> str:
    """Second look that sees what this page already produced.

    Missing names and blank cells come from the extract plus the Postgres
    roster. Filed dollar amounts are not included.
    """
    col_line = " | ".join(columns)
    title_bit = f'Title hint: "{title}".\n' if title else ""
    have = "\n".join(f"- {n}" for n in extracted_names[:80]) or "- (none yet)"
    missing_lines = "\n".join(
        f"- {_locator_line(rec, sheet_kind)}" for rec in missing if rec
    )
    blank_lines = "\n".join(f"- {n}" for n in blank_names if n)
    sections = [
        "These contacts are already in the extract. Do not return them again unless one is listed as blank:",
        have,
    ]
    if missing_lines:
        sections.append(
            "These known clients have no row yet. If a name is on the sheet, return that full row. If it is not on the sheet, omit it:\n"
            + missing_lines
        )
    if blank_lines:
        kind = (sheet_kind or "").lower()
        cell = "date / work-completed cell" if kind in ("work", "work_completed") else "price/Hedge cell"
        sections.append(
            f"These rows exist but the {cell} is blank or not a dollar amount. Return the name and the cells on that same line:\n"
            + blank_lines
        )
    if ask_others:
        sections.append(
            "Also return any other visible table row whose contact is not in the already-extracted list."
        )
    body = "\n\n".join(sections)
    return f"""
This is a FULL photo of a paper sheet ({sheet_kind}).
{title_bit}A first pass already extracted part of the table. Use that list only to see what is still missing. Read every filled cell from the photo, not from the list.

{body}

Columns left-to-right:
{col_line}

Rules:
- Return ONLY the missing or blank rows. If nothing is missing, return rows: [].
- Contact/CLIENT is the name as written, never the street.
- The price/Hedge cell is the dollar amount on that line. Do not repeat the street there.
- Handwriting that replaces a crossed-out number wins.
- Do not copy a neighbor's price. Do not invent rows.

Return ONLY JSON:
{{
  "title": null,
  "tables": [{{"caption": null, "columns": {json.dumps(columns)}, "rows": [["...", ...]]}}],
  "notes": [],
  "complete": true
}}
""".strip()


def guided_notes_prompt(sheet_kind: str, title: str | None = None) -> str:
    title_bit = f'Title hint: "{title}".\n' if title else ""
    return f"""
This is a FULL photo of a paper sheet.
{title_bit}Extract ONLY margin / footer / handwritten notes that are NOT inside the main table grid
(a footer about someone who is not a row in the table, a season note, an association note).

Return JSON with empty tables and notes filled:
{{
  "title": null,
  "tables": [],
  "notes": ["...", ...],
  "complete": true
}}
If there are no such notes, return notes: [].
Sheet kind: {sheet_kind}.
""".strip()


