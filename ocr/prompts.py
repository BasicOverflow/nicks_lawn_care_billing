"""Prompts for work-completed sheet OCR."""

from __future__ import annotations

import json
import re

# Cells are strings so later steps can interpret days, marks, and names.
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
    kind = (sheet_kind or "work").lower()
    hint = {
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
    kind = (sheet_kind or "work").lower()
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
