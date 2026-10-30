"""Months, feeding plan, colony names, sheet codes, roster and arena layout (no drawing).

A printed set covers one calendar month: two shared feeding sheets (days 1-15 and 16-end; all
colonies are fed in one round, so one line per day with a dead / left-over pair per colony) and
one sheet per colony (weekly paint and count, dead tag IDs, comments, events).

Colony names are what the lab writes: numbers ("22"), numbered splits ("10#1") or words ("Mix").
"""
import calendar
import csv
import re
from collections import namedtuple
from datetime import date, timedelta
from pathlib import Path

MAX_COLONY = 999
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
FEED_ALL, FEED_JELLY, FEED_WEEKEND = "MW RR J", "J", "J*"
FEED_PLAN = (FEED_ALL, FEED_JELLY, FEED_ALL, FEED_JELLY, FEED_ALL, FEED_WEEKEND, FEED_WEEKEND)
FIRST_PAGE_DAYS = 15            # feeding sheet 1 = days 1-15, sheet 2 = 16 to the end of the month
ARENA_FIELDS = ("valid_from", "left", "right")
ROSTER_FIELDS = ("colony", "care_from", "care_until")
ISO_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
MONTH_RE = re.compile(r"^([0-9]{4})-([0-9]{2})$")
# "22", "10#1" (no leading zeros; 1-999, split 1-99), or up to 8 letters, digits, '+' and '-' with
# at least one letter, starting with a letter or digit ("Mix", "4A", "13tag-R", "6+4B" - names
# of legacy boxes too). Short names keep the QR code at version 1 and fit a feeding-sheet column.
COLONY_NAME = re.compile(r"^(?:[1-9][0-9]{0,2}(?:#[1-9][0-9]?)?"
                         r"|(?=[A-Za-z0-9+\-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9+\-]{0,7})$")

# QR payloads use upper case, digits, space, '-' and '.' only: alphanumeric mode keeps them at
# version 1 (21 modules), ~1 mm per module, which still reads in blurred or distant photos.
CODE_RE = re.compile(r"(?:CARE1 C([0-9A-Z.+\-]+) ([0-9]{4}-[0-9]{2})|FEED1 ([0-9]{4}-[0-9]{2}) P([12]))")
SheetId = namedtuple("SheetId", "kind colony month page")   # kind "colony" (colony = key) | "feeding"


class InputError(ValueError):
    """Bad input from the user or a table (reported as a message, not a traceback)."""


# --- months ---------------------------------------------------------------------------------

def parse_month(text):
    """'2026-10' -> date(2026, 10, 1)."""
    m = MONTH_RE.match(text.strip())
    try:
        if not m:
            raise ValueError
        return date(int(m.group(1)), int(m.group(2)), 1)
    except ValueError:
        raise InputError("month must be YYYY-MM, got %r" % text) from None


def month_end(month):
    return month.replace(day=calendar.monthrange(month.year, month.month)[1])


def feeding_pages(month):
    """[(first, last)] of the two feeding sheets of a month."""
    split = month.replace(day=FIRST_PAGE_DAYS)
    return [(month, split), (split + timedelta(days=1), month_end(month))]


def week_ranges(month):
    """Rows of a colony sheet: days 1-7, 8-14, 15-21, 22-28 and the rest of the month."""
    end, out, first = month_end(month), [], month
    while first <= end:
        out.append((first, min(first + timedelta(days=6), end)))
        first += timedelta(days=7)
    return out


def planned_food(day):
    """Food planned for a day: everything Mon/Wed/Fri, jelly Tue/Thu and on Saturday or Sunday."""
    return FEED_PLAN[day.weekday()]


# --- colony names and sheet codes -----------------------------------------------------------

def check_colony_name(name, where="colony"):
    if not COLONY_NAME.fullmatch(name):
        raise InputError("%s: %r is not a colony name (e.g. 22, 10#1, Mix, 4A; at most 8 characters)"
                         % (where, name))
    return name


def colony_key(name):
    """Case-free, QR-safe form of a colony name: '10#1' -> '10.1', 'Mix' -> 'MIX'."""
    return name.upper().replace("#", ".")


def parse_colonies(spec):
    """'8,Mix,22-24' -> ['8', 'Mix', '22', '23', '24']; each colony once (ignoring case)."""
    out = []
    for part in (p.strip() for p in spec.split(",")):
        m = re.fullmatch(r"([0-9]+)-([0-9]+)", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo < 1 or hi < lo or hi > MAX_COLONY:
                raise InputError("bad colony range %r (numbers are 1-%d)" % (part, MAX_COLONY))
            out.extend(str(n) for n in range(lo, hi + 1))
        else:
            out.append(check_colony_name(part, "colony list %r" % spec))
    keys = [colony_key(n) for n in out]
    if len(set(keys)) != len(keys):
        raise InputError("colony listed twice in %r" % spec)
    return out


def colony_code(colony, month):
    return "CARE1 C%s %s" % (colony_key(colony), month.strftime("%Y-%m"))


def feeding_code(month, page):
    return "FEED1 %s P%d" % (month.strftime("%Y-%m"), page)


def parse_sheet_code(text):
    """QR text of a sheet -> SheetId; InputError for anything else."""
    m = CODE_RE.fullmatch(text.strip())
    if not m:
        raise InputError("not a care-sheet code: %r" % text)
    if m.group(1):
        return SheetId("colony", m.group(1), parse_month(m.group(2)), None)
    return SheetId("feeding", None, parse_month(m.group(3)), int(m.group(4)))


# --- tables ---------------------------------------------------------------------------------

def read_rows(path, fields, hint=""):
    """Stripped, non-blank rows of a CSV with the required columns (BOM-tolerant)."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:      # Excel saves a BOM
        reader = csv.DictReader(f)
        missing = [k for k in fields if k not in (reader.fieldnames or [])]
        if missing:
            raise InputError("%s: missing column(s) %s%s" % (path, ", ".join(missing), hint))
        out = []
        for line, raw in enumerate(reader, start=2):
            row = {k: (v or "").strip() for k, v in raw.items() if k is not None}
            if any(row.values()):
                out.append(("%s line %d" % (path, line), row))
    return out


def check_date(v, field, where):
    try:
        if not ISO_DATE.match(v):
            raise ValueError
        return date.fromisoformat(v)
    except ValueError:
        raise InputError("%s: %s %r is not a YYYY-MM-DD date" % (where, field, v)) from None


def read_roster(path, month):
    """Colonies on the care sheets in a month, in file order (colonies.csv care_from/care_until).

    A colony is cared for from care_from to care_until (blank = still); a row without care_from
    (e.g. an analysis-only colony) is not on the sheets. Every row is checked (name, no
    duplicates), also rows not on the sheets, so a bad name anywhere stops printing.
    """
    rows = read_rows(path, ROSTER_FIELDS, " (add care_from and care_until to list the reared colonies)")
    first, last = month, month_end(month)
    out, seen = [], set()
    for where, row in rows:
        name = check_colony_name(row["colony"], where)
        if colony_key(name) in seen:
            raise InputError("%s: colony %s appears twice" % (where, name))
        seen.add(colony_key(name))
        if not row["care_from"]:
            continue
        start = check_date(row["care_from"], "care_from", where)
        until = check_date(row["care_until"], "care_until", where) if row["care_until"] else None
        if until and until < start:
            raise InputError("%s: care_until %s is before care_from %s" % (where, until, start))
        if start <= last and (until is None or until >= first):
            out.append(name)
    return out


def read_collected(path):
    """{colony key: collection date} from colonies.csv (collected_on; blank = not known)."""
    out = {}
    for where, row in read_rows(path, ("colony", "collected_on")):
        if row["collected_on"]:
            out[colony_key(row["colony"])] = check_date(row["collected_on"], "collected_on", where)
    return out


def read_arena(path):
    """Validated rows of arena.csv (valid_from,left,right[,note]), values stripped.

    A missing file raises: the caller asked for it, and silently printing "Not in the arena" on
    every sheet would make carers skip the times. A half holds one colony name or nothing.
    """
    rows, seen = [], set()
    for where, row in read_rows(path, ARENA_FIELDS):
        check_date(row["valid_from"], "valid_from", where)
        for side in ("left", "right"):
            if row[side]:
                check_colony_name(row[side], "%s: %s %r" % (where, side, row[side]))
        if row["left"] and colony_key(row["left"]) == colony_key(row["right"]):
            raise InputError("%s: colony %s is on both sides" % (where, row["left"]))
        if row["valid_from"] in seen:
            raise InputError("%s: valid_from %s appears twice" % (where, row["valid_from"]))
        seen.add(row["valid_from"])
        rows.append(row)
    return rows


# --- arena layout ---------------------------------------------------------------------------

def _row_in_force(rows, on):
    in_force = [r for r in rows if date.fromisoformat(r["valid_from"]) <= on]
    return max(in_force, key=lambda r: date.fromisoformat(r["valid_from"])) if in_force else None


def arena_side(rows, colony, on):
    """(side, valid_from) of the arena layout in force on a date, or None if not in the arena."""
    row = _row_in_force(rows, on)
    if row is None:
        return None
    for side in ("left", "right"):
        if row.get(side) and colony_key(row[side]) == colony_key(colony):
            return side, row["valid_from"]
    return None


def _change_days(rows, start, end):
    """start, then every valid_from strictly inside start..end."""
    inside = sorted({d for d in (date.fromisoformat(r["valid_from"]) for r in rows) if start < d <= end})
    return [start] + inside


def arena_layouts(rows, start, end):
    """[(first_day, left, right)] for each layout in force during start..end ('' = empty half)."""
    out = []
    for day in _change_days(rows, start, end):
        row = _row_in_force(rows, day)
        layout = (row.get("left", ""), row.get("right", "")) if row else ("", "")
        if not out or tuple(map(colony_key, out[-1][1:])) != tuple(map(colony_key, layout)):
            out.append((day,) + layout)
    return out


def arena_colonies(rows, start, end):
    """Colonies in the arena at any time in start..end, in order of appearance ([] if unknown)."""
    out, seen = [], set()
    for _, left, right in (arena_layouts(rows, start, end) if rows is not None else []):
        for c in (left, right):
            if c and colony_key(c) not in seen:
                seen.add(colony_key(c))
                out.append(c)
    return out


def arena_header(rows, colony, start, end):
    """(text, in_arena) for a colony sheet header; None when arena.csv was not read."""
    if rows is None:
        return None
    states = []
    for day in _change_days(rows, start, end):
        side = arena_side(rows, colony, day)
        if not states or _half(side) != _half(states[-1][1]):
            states.append((day, side))
    if len(states) == 1:
        side = states[0][1]
        if side is None:
            return "Not in the arena", False
        return "IN ARENA: %s half since %s - write times to the minute" % side, True
    parts = ["%s %s" % (day.strftime("%m/%d"), "%s half" % side[0] if side else "out")
             for day, side in states]
    return "ARENA: %s - write times to the minute while in" % ", ".join(parts), True


def feeding_arena_line(rows, start, end):
    """(text, any_in_arena) for a feeding sheet header; None when arena.csv was not read."""
    if rows is None:
        return None
    layouts = arena_layouts(rows, start, end)
    if all(not left and not right for _, left, right in layouts):
        return "No colony in the arena", False

    def describe(left, right):
        return "left %s, right %s" % (left or "-", right or "-")

    if len(layouts) == 1:
        text = describe(*layouts[0][1:])
    else:
        text = "; ".join("from %s %s" % (day.strftime("%m/%d"), describe(left, right))
                         for day, left, right in layouts)
    return "In the arena: %s - write the feeding time to the minute" % text, True


def _half(side):
    return side[0] if side else None
