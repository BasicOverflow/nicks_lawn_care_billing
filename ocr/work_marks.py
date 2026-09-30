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
_DAY_TOKEN = re.compile(r"(?<![\d$.])(?P<day>[1-9]|[12]\d|3[01])(?P<mark>h)?(?![\d.])", re.I)
# A house number after a price ("50 12 Main St") is an address, not a visit.
_HOUSE_AFTER = re.compile(
    r"\s+(?P<num>\d{1,4})\s+[A-Za-z][A-Za-z0-9 .'-]*"
    r"(?:\s(?:St|Street|Rd|Road|Ave|Avenue|Dr|Drive|Ln|Lane|Ct|Court|Way|Blvd|Pl|Place|Neck)\.?)\b",
    re.I,
)


def _split_joined(match: re.Match[str]) -> str:
    parts = re.split(r"\s*[/-]\s*", match.group(0))
    days = [part for part in parts if not re.fullmatch(_YEAR, part)]
    return " ".join(days)


def _split_commas(match: re.Match[str]) -> str:
    parts = re.split(r"\s*,\s*", match.group(0))
    return " ".join(parts)


def normalize_work_marks(text: str) -> str:
    """Keep each month-day as its own token, and keep any job wording."""
    raw = (text or "").strip()
    if not raw:
        return ""
    flattened = _JOINED_DAYS.sub(_split_joined, raw)
    flattened = _COMMA_DAYS.sub(_split_commas, flattened)
    flattened = re.sub(r"(\d[A-Za-z]*)\s*,\s*(?=[A-Za-z$])", r"\1 ", flattened)
    flattened = re.sub(r"(\d{1,2})\s+h\b", r"\1h", flattened, flags=re.I)
    flattened = re.sub(r"(\d{1,2})\s+hedge\s*(?=\||\s*$)", r"\1h", flattened, flags=re.I)
    flattened = re.sub(r"\$\s*(\d)", r" \1", flattened)
    return _SPACES.sub(" ", flattened).strip()


def _money_label(amount: float) -> str:
    if abs(amount - round(amount)) < 0.001:
        return f"${int(round(amount))}"
    return f"${amount:.2f}"


def _strip_address_tail(text: str) -> str:
    house = _HOUSE_AFTER.search(text)
    if not house:
        return text
    before = text[:house.start()].rstrip()
    if re.search(r"\d(?:\.\d{1,2})?$", before):
        return before
    return text


def _job_from_chunk(chunk: str) -> tuple[dict | None, str]:
    """Return a custom job parsed from the end of chunk, plus any leftover day text."""
    text = _strip_address_tail(chunk.strip())
    if not text or not re.search(r"[A-Za-z]", text):
        return None, chunk.strip()
    trailing = re.search(r"\s+(?P<trail>[1-9]|[12]\d|3[01])(?!\d)\s*$", text)
    trailing_day = trailing.group("trail") if trailing else ""
    core = text[: trailing.start()].strip() if trailing else text
    amt_match = re.search(r"\s+\$?\s*(?P<amt>\d+(?:\.\d{1,2})?)\s*$", core)
    if not amt_match:
        return None, text
    end = amt_match.end()
    house = _HOUSE_AFTER.match(core, end)
    if house:
        end = house.end()
    head = core[: amt_match.start()].strip()
    split = re.match(
        r"^(?:(?P<prefix>(?:(?:[1-9]|[12]\d|3[01])[hH]?\s+)+))(?P<name>[A-Za-z].+)$",
        head,
    )
    if split:
        prefix = split.group("prefix").strip()
        name = split.group("name").strip()
    else:
        prefix = ""
        name = head
    name = re.sub(r"\s+", " ", name).strip(" ;,|")
    if not name or not re.search(r"[A-Za-z]", name):
        return None, chunk.strip()
    day = None
    leftover_prefix = prefix
    if prefix and re.search(r"(?<![\d.])(?:[1-9]|[12]\d|3[01])[hH]\s*$", prefix):
        day = None
    else:
        day_match = re.search(r"(?<![\d.])(?P<day>[1-9]|[12]\d|3[01])(?!\d)\s*$", prefix)
        if day_match:
            day = int(day_match.group("day"))
            leftover_prefix = prefix[: day_match.start()].strip()
    amount = float(amt_match.group("amt"))
    if amount <= 0:
        return None, chunk.strip()
    leftover = " ".join(part for part in (leftover_prefix, trailing_day) if part).strip()
    return {
        "kind": "custom",
        "name": name,
        "amount": amount,
        "day": day,
    }, leftover


def _day_marks(text: str) -> list[dict]:
    """Bare days and hedge marks from text with no priced job at the end."""
    marks: list[dict] = []
    for match in _DAY_TOKEN.finditer(text):
        day = int(match.group("day"))
        if match.group("mark"):
            marks.append({"kind": "hedge", "day": day, "name": "", "amount": None})
        else:
            marks.append({"kind": "mow", "day": day, "name": "", "amount": None})
    return marks


def _dedupe_job_mow_days(marks: list[dict]) -> list[dict]:
    """A day written for a custom job is not also a separate mowing day."""
    job_days = [int(mark["day"]) for mark in marks if mark["kind"] == "custom" and mark.get("day")]
    if not job_days:
        return marks
    out: list[dict] = []
    for day in job_days:
        removed = False
        kept: list[dict] = []
        for mark in marks:
            if (
                not removed
                and mark["kind"] == "mow"
                and int(mark.get("day") or 0) == day
            ):
                removed = True
                continue
            kept.append(mark)
        marks = kept
    return marks


def _parse_chunk(chunk: str) -> list[dict]:
    raw = chunk.strip()
    if not raw:
        return []
    marks: list[dict] = []
    bare = re.match(r"^(?P<dollar>\$)?\s*(?P<amt>\d+(?:\.\d{1,2})?)(?!\d)", raw)
    if bare:
        dayish = bool(re.fullmatch(r"(?:[1-9]|[12]\d|3[01])", bare.group("amt")))
        house = _HOUSE_AFTER.match(raw, bare.end())
        if bare.group("dollar") or not dayish:
            end = house.end() if house else bare.end()
            marks.append({
                "kind": "custom",
                "name": "Custom",
                "amount": float(bare.group("amt")),
                "day": None,
            })
            raw = raw[end:].strip()
    job, leftover = _job_from_chunk(raw)
    if job:
        marks.append(job)
        marks.extend(_day_marks(leftover))
        return marks
    marks.extend(_day_marks(raw))
    note = re.sub(r"\s+", " ", raw).strip(" ;,|")
    if note and re.search(r"[A-Za-z]", note):
        marks.append({"kind": "note", "day": None, "name": note, "amount": None})
    return marks


def _attach_trailing_job_days(segments: list[list[dict]]) -> None:
    """When a job segment has no day, use a duplicated last day from the segment before it."""
    for index in range(len(segments) - 1):
        left = segments[index]
        right = segments[index + 1]
        if len(right) != 1 or right[0]["kind"] != "custom" or right[0].get("day"):
            continue
        mow_days = [int(m["day"]) for m in left if m["kind"] == "mow" and m.get("day")]
        if len(mow_days) < 2:
            continue
        last = mow_days[-1]
        if mow_days.count(last) < 2:
            continue
        right[0]["day"] = last


def parse_work_marks(text: str) -> list[dict]:
    """Split a work cell into mow days, hedge days, priced jobs, and leftover notes."""
    raw = normalize_work_marks(text)
    if not raw:
        return []
    parts = [part.strip() for part in re.split(r"\s*\|\s*", raw) if part.strip()]
    if not parts:
        return []
    segments = [_parse_chunk(part) for part in parts]
    _attach_trailing_job_days(segments)
    marks = [mark for segment in segments for mark in segment]
    marks = _dedupe_job_mow_days(marks)
    collapsed: list[dict] = []
    for mark in marks:
        if (
            mark["kind"] == "note"
            and re.fullmatch(r"h|hedge|hedging", str(mark.get("name") or "").strip(), re.I)
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
            dated = f"{mark['day']} " if mark.get("day") else ""
            boxes.append(f"{dated}{mark['name']} {_money_label(float(mark['amount']))}")
        elif mark.get("name"):
            boxes.append(str(mark["name"]))
    return boxes
