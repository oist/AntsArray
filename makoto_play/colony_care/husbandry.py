"""husbandry_log.csv: reading and checking it, and what it says about each colony (no drawing).

One row per care action or event, read from the filled care sheets (README.md). Adults are not
counted every week: they are computed from the last full count (census adult) plus the new
adults painted since, minus the ants found dead and removed since.
"""
import re
from collections import namedtuple

import sheet_model as M

FIELDS = ("date", "time", "colony", "event", "what", "n", "ids", "block", "by", "source", "note",
          "provisional")
EVENTS = ("collected", "fed", "not_fed", "leftover", "water", "dead", "painted", "census",
          "tray_changed", "nest_changed", "arena_in", "arena_out", "tagged", "tag_lost", "removed", "note")
ALL = "all"                                     # colony of a feeding round given to every colony
FOR_ALL = ("fed",)
CENSUS_ITEMS = ("adult", "pupa")
LEFTOVER_CODES = ("0", "1", "2", "3", "y")      # y = some left, amount not recorded (Forms records)
MALE = "male"                                   # dead what=male: males are not counted as adults
RECENT_DAYS = 28
TRAY_DUE_DAYS, TRAY_OVERDUE_DAYS = 60, 90
TIME_RE = re.compile(r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$")

Entry = namedtuple("Entry", "line date time colony event what n ids block by source note provisional")
Estimate = namedtuple("Estimate", "n base_date base_n added lost unknown provisional")
Summary = namedtuple("Summary", "colony adults pupae last_fed last_leftover last_water dead_28d "
                                "new_28d last_paint tray history")


# --- reading --------------------------------------------------------------------------------

def read_log(path):
    """Checked entries of husbandry_log.csv, in file order (blank lines skipped)."""
    return [_entry(where, row) for where, row in M.read_rows(path, FIELDS)]


def _entry(where, row):
    day = M.check_date(row["date"], "date", where)
    if row["time"] and not TIME_RE.match(row["time"]):
        raise M.InputError("%s: time %r is not HH:MM" % (where, row["time"]))
    event, colony = row["event"], row["colony"]
    if event not in EVENTS:
        raise M.InputError("%s: event %r is not one of %s" % (where, event, ", ".join(EVENTS)))
    if colony == ALL:
        if event not in FOR_ALL:
            raise M.InputError("%s: colony 'all' is only for %s" % (where, ", ".join(FOR_ALL)))
    else:
        M.check_colony_name(colony, where)
    n = _count(row["n"], where)
    ids = _ids(row["ids"], where)
    _check_event_fields(event, row["what"], n, where)
    if row["provisional"] not in ("", "y"):
        raise M.InputError("%s: provisional must be 'y' or empty, got %r" % (where, row["provisional"]))
    return Entry(where, day, row["time"], colony, event, row["what"], n, ids, row["block"], row["by"],
                 row["source"], row["note"], row["provisional"] == "y")


def _count(text, where):
    if not text:
        return None
    if not (text.isascii() and text.isdigit()):
        raise M.InputError("%s: n %r is not a whole number" % (where, text))
    return int(text)


def _ids(text, where):
    out = []
    for tok in text.split():
        if not (tok.isascii() and tok.isdigit() and int(tok) <= 99):
            raise M.InputError("%s: ids must be tag IDs 0-99 separated by spaces, got %r" % (where, text))
        out.append(int(tok))
    return tuple(out)


def _check_event_fields(event, what, n, where):
    if event == "census":
        if what not in CENSUS_ITEMS:
            raise M.InputError("%s: census needs what = %s" % (where, " or ".join(CENSUS_ITEMS)))
        if n is None:
            raise M.InputError("%s: census needs a number (n)" % where)
    if event == "leftover" and what not in LEFTOVER_CODES:
        raise M.InputError("%s: leftover needs what = 0-3 (or y), got %r" % (where, what))


# --- per colony ---------------------------------------------------------------------------

def _of(entries, colony):
    key = M.colony_key(colony)
    return [e for e in entries if e.colony != ALL and M.colony_key(e.colony) == key]


def _is_male(e):
    return e.event == "dead" and e.what == MALE


def _size(e):
    """Number of ants an entry stands for: n, else the number of IDs, else None."""
    return e.n if e.n is not None else (len(e.ids) or None)


def _after(e, base):
    """Whether an event comes after a full count: a later day, or the same day at a later time.

    An event without a time on the day of the count is taken to be in the count.
    """
    if e.date != base.date:
        return e.date > base.date
    return bool(e.time and base.time and e.time > base.time)


def adult_estimate(entries, colony, on):
    """Adults on a day: last full count on or before it + painted - dead - removed after the count.

    Of two counts at the same date and time the later row in the file wins (a recount). None
    without any full count. Changes of unknown size, and a result below zero, are listed in
    `unknown`.
    """
    own = [e for e in _of(entries, colony) if e.date <= on]
    counts = [(e.date, e.time, i, e) for i, e in enumerate(own) if e.event == "census" and e.what == "adult"]
    if not counts:
        return None
    base = max(counts, key=lambda c: c[:3])[3]
    after = [e for e in own if _after(e, base) and e.event in ("painted", "dead", "removed")
             and not _is_male(e)]
    added = lost = 0
    unknown = []
    for e in after:
        size = _size(e)
        if size is None:
            unknown.append("%s %s: %s" % (e.date, e.event, e.note or "number not recorded"))
        elif e.event == "painted":
            added += size
        else:
            lost += size
    n = base.n + added - lost
    if n < 0:
        unknown.append("computed number below zero: check the full count and the losses")
    provisional = base.provisional or any(e.provisional for e in after)
    return Estimate(n, base.date, base.n, added, lost, tuple(unknown), provisional)


Totals = namedtuple("Totals", "feedings dead males notes")


def totals(entries, colony):
    """Whole-record totals of a colony: feedings, dead females, dead males, dated notes."""
    own = _of(entries, colony)
    return Totals(
        feedings=sum(1 for e in own if e.event == "fed"),
        dead=sum(_size(e) or 0 for e in own if e.event == "dead" and not _is_male(e)),
        males=sum(_size(e) or 0 for e in own if _is_male(e)),
        notes=tuple("%s %s" % (e.date, e.what) for e in own if e.event == "note" and e.what),
    )


def tray_status(last, on):
    if last is None:
        return "unknown"
    days = (on - last).days
    return "ok" if days < TRAY_DUE_DAYS else ("due" if days <= TRAY_OVERDUE_DAYS else "overdue")


def _last(items):
    return max(items, key=lambda e: (e.date, e.time)) if items else None


def colony_summary(entries, colony, on):
    """What the log says about a colony on a day (history newest first, feeding rounds left out)."""
    own = [e for e in _of(entries, colony) if e.date <= on]
    recent = [e for e in own if (on - e.date).days < RECENT_DAYS]
    not_fed = {e.date for e in own if e.event == "not_fed"}
    fed = [e.date for e in entries if e.event == "fed" and e.date <= on and e.date not in not_fed
           and (e.colony == ALL or M.colony_key(e.colony) == M.colony_key(colony))]
    pupae = _last([e for e in own if e.event == "census" and e.what == "pupa"])
    leftover = _last([e for e in own if e.event == "leftover"])
    paint = _last([e for e in own if e.event == "painted"])
    tray = _last([e for e in own if e.event == "tray_changed"])
    water = _last([e for e in own if e.event == "water"])
    tray_date = tray.date if tray else None
    return Summary(
        colony=colony,
        adults=adult_estimate(entries, colony, on),
        pupae=(pupae.date, pupae.n) if pupae else None,
        last_fed=max(fed) if fed else None,
        last_leftover=(leftover.date, leftover.what) if leftover else None,
        last_water=water.date if water else None,
        dead_28d=sum(_size(e) or 0 for e in recent if e.event == "dead" and not _is_male(e)),
        new_28d=sum(e.n or 0 for e in recent if e.event == "painted"),
        last_paint=(paint.date, paint.what) if paint else None,
        tray=(tray_date, tray_status(tray_date, on)),
        history=sorted(own, key=lambda e: (e.date, e.time), reverse=True),
    )
