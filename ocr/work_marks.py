"""Work-log cells are a list of days, not a month/day date."""

from __future__ import annotations

import re

# A day of the month, optionally with a handwritten mark such as 19h.
_DAY = r"(?:[1-9]|[12]\d|3[01])[A-Za-z]*"
_YEAR = r"(?:19|20)\d{2}"
# 9/16/23, 9-16-23, 9 / 16, including a trailing year the model invented.
_JOINED_DAYS = re.compile(
    rf"(?<![\d.]){_DAY}(?:\s*[/-]\s*{_DAY})+(?:\s*[/-]\s*{_YEAR})?(?![\d.])"
)
_COMMA_DAYS = re.compile(
    rf"(?<![\d.]){_DAY}(?:\s*,\s*{_DAY})+(?![\d.])"
)
_SPACES = re.compile(r"[ \t]{2,}")


def _split_joined(match: re.Match[str]) -> str:
    parts = re.split(r"\s*[/-]\s*", match.group(0))
    days = [part for part in parts if not re.fullmatch(_YEAR, part)]
    return " ".join(days)


def _split_commas(match: re.Match[str]) -> str:
    parts = re.split(r"\s*,\s*", match.group(0))
    return " ".join(parts)


def normalize_work_marks(text: str) -> str:
    """Keep each month-day as its own token, and keep any job wording.

    Handwritten rows look like "9    16    23" or "5 bush trimming $50".
    A vision model often collapses the numbers into a date such as "9/16".
    Slashing, dashing, or comma-joining those day tokens is undone here.
    Words and amounts that are not day numbers are left in place.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    flattened = _JOINED_DAYS.sub(_split_joined, raw)
    flattened = _COMMA_DAYS.sub(_split_commas, flattened)
    flattened = re.sub(r"(\d[A-Za-z]*)\s*,\s*(?=[A-Za-z$])", r"\1 ", flattened)
    return _SPACES.sub(" ", flattened).strip()


_DAY_TOKEN = re.compile(r"(?<![\d$.])(?P<day>[1-9]|[12]\d|3[01])(?P<mark>h)?(?![\d.])", re.I)
_JOB = re.compile(
    r"(?P<name>[A-Za-z][^$|]{0,80}?)\s*\$\s*(?P<amt>\d+(?:\.\d{1,2})?)"
)


def _money_label(amount: float) -> str:
    if abs(amount - round(amount)) < 0.001:
        return f"${int(round(amount))}"
    return f"${amount:.2f}"


def parse_work_marks(text: str) -> list[dict]:
    """Split a work cell into mow days, hedge days, priced jobs, and leftover notes.

    A bare day such as 9 is a mowing visit. 15h is a hedge visit. A phrase with
    a written dollar amount is its own job. Other words, such as "paid", stay
    as notes and are not jobs.
    """
    raw = normalize_work_marks(text)
    if not raw:
        return []
    jobs: list[tuple[int, int, dict]] = []
    for match in _JOB.finditer(raw):
        name = re.sub(r"\s+", " ", match.group("name")).strip(" ;,|")
        if not name:
            continue
        jobs.append((
            match.start(),
            match.end(),
            {"kind": "custom", "name": name, "amount": float(match.group("amt")), "day": None},
        ))

    def covered(start: int, end: int) -> bool:
        return any(start < job_end and end > job_start for job_start, job_end, _job in jobs)

    marks: list[tuple[int, dict]] = [(start, item) for start, _end, item in jobs]
    claimed: list[tuple[int, int]] = [(start, end) for start, end, _item in jobs]
    for match in _DAY_TOKEN.finditer(raw):
        if covered(match.start(), match.end()):
            continue
        day = int(match.group("day"))
        if match.group("mark"):
            marks.append((match.start(), {"kind": "hedge", "day": day, "name": "", "amount": None}))
        else:
            marks.append((match.start(), {"kind": "mow", "day": day, "name": "", "amount": None}))
        claimed.append((match.start(), match.end()))

    claimed.sort()
    cursor = 0
    for start, end in claimed:
        _take_note(raw[cursor:start], marks, cursor)
        cursor = max(cursor, end)
    _take_note(raw[cursor:], marks, cursor)
    marks.sort(key=lambda item: item[0])
    ordered = [item for _pos, item in marks]
    collapsed: list[dict] = []
    for mark in ordered:
        if (
            mark["kind"] == "note"
            and str(mark.get("name") or "").lower() == "h"
            and collapsed
            and collapsed[-1]["kind"] == "mow"
        ):
            collapsed[-1] = {**collapsed[-1], "kind": "hedge"}
            continue
        collapsed.append(mark)
    return collapsed


def _take_note(chunk: str, marks: list[tuple[int, dict]], pos: int) -> None:
    text = re.sub(r"\s+", " ", chunk).strip(" ;,|")
    if not text or not re.search(r"[A-Za-z]", text):
        return
    marks.append((pos, {"kind": "note", "day": None, "name": text, "amount": None}))


def review_boxes(text: str) -> list[str]:
    """One label per box shown side by side in the review row."""
    boxes: list[str] = []
    for mark in parse_work_marks(text):
        kind = mark["kind"]
        if kind == "mow":
            boxes.append(str(mark["day"]))
        elif kind == "hedge":
            boxes.append(f"{mark['day']}h")
        elif kind == "custom":
            boxes.append(f"{mark['name']} {_money_label(float(mark['amount']))}")
        elif mark.get("name"):
            boxes.append(str(mark["name"]))
    return boxes

