"""What husbandry_log.csv already says for the cells of a month's care sheets (no drawing).

Sheets printed when part of a month is already in the log (a reprint, a lost sheet, the rest
of the month) carry those records, so the carers write only what is still missing.
"""
from collections import namedtuple

import husbandry as H
import sheet_model as M

# Counts (dead, new_adults) are a number, None when not written, or "?" / "3+?" when the log
# has a row whose number was not recorded.
FeedingDay = namedtuple("FeedingDay", "time by cells")          # cells: {colony key: FeedingCell}
FeedingCell = namedtuple("FeedingCell", "dead left not_fed")
ColonyWeek = namedtuple("ColonyWeek", "date by colour new_adults pupae dead_ids comments")
ColonyEvent = namedtuple("ColonyEvent", "date time kind details by")
EMPTY_WEEK = ColonyWeek(None, "", "", None, None, (), ())
Records = namedtuple("Records", "entries collected")            # collected: {colony key: date}
SHEET_LEFTOVER = ("0", "1", "2", "3")                           # L codes read from a feeding sheet

# log event -> event type on the colony sheet (census adult = "count"); the collection date is in
# the sheet header instead
EVENT_KINDS = {"tray_changed": "tray", "nest_changed": "nest", "tagged": "tagging #",
               "arena_in": "arena in", "arena_out": "arena out", "removed": "removal", "tag_lost": "other"}


class ColonyFill(namedtuple("ColonyFill", "collected last_tray weeks events")):
    @property
    def has_records(self):
        """Whether the log has anything for the weeks or events of this month."""
        return bool(self.events) or any(w != EMPTY_WEEK for w in self.weeks)


def _when(e):
    return e.date, e.time


def _unique(values):
    """Non-empty values once each, in order."""
    return tuple(dict.fromkeys(v for v in values if v))


def _total(entries):
    """Ants the entries stand for: a number, '?' or 'N+?' when some were not counted."""
    sizes = [H.entry_size(e) for e in entries]
    known = sum(s for s in sizes if s is not None)
    if None in sizes:
        return "%d+?" % known if known else "?"
    return known


# --- feeding sheets ---------------------------------------------------------------------------

def feeding_days(entries, first, last):
    """{day: FeedingDay} for the days in first..last with anything for the feeding sheet."""
    out = {}
    for day in sorted({e.date for e in entries if first <= e.date <= last}):
        today = [e for e in entries if e.date == day]
        fed = [e for e in today if e.event == "fed" and e.colony == H.ALL] or \
              [e for e in today if e.event == "fed"]
        fed.sort(key=_when)
        cells = _feeding_cells(today)
        if fed or cells:
            out[day] = FeedingDay(next((e.time for e in fed if e.time), ""),
                                  " ".join(_unique(e.by for e in fed)), cells)
    return out


def _feeding_cells(today):
    """{colony key: FeedingCell} of one day: dead females found, food left over, not fed."""
    rows = {}
    for e in today:
        if e.colony != H.ALL and e.event in ("dead", "leftover", "not_fed") and e.what != H.MALE:
            rows.setdefault(M.colony_key(e.colony), []).append(e)
    cells = {}
    for key, own in rows.items():
        dead = [e for e in own if e.event == "dead"]
        left = [e.what for e in sorted(own, key=_when) if e.event == "leftover"]
        left = left[-1] if left else ""
        # D and L are written as a pair on the sheet and dead rows are logged only when D > 0, so
        # a sheet's L code without a dead row means D 0 (Forms 'y' rows had no such pair)
        n_dead = _total(dead) if dead else (0 if left in SHEET_LEFTOVER else None)
        cells[key] = FeedingCell(n_dead, left, any(e.event == "not_fed" for e in own))
    return cells


# --- colony sheets ----------------------------------------------------------------------------

def colony_fill(entries, colony, month, collected=None):
    """ColonyFill of one colony for a month: header dates, the weekly lines and the events."""
    own = sorted(H.of_colony(entries, colony), key=_when)
    end = M.month_end(month)
    trays = [e.date for e in own if e.event == "tray_changed" and e.date <= end]
    weeks = [_week([e for e in own if first <= e.date <= last]) for first, last in M.week_ranges(month)]
    events = [ColonyEvent(e.date, e.time, _kind(e), _details(e), e.by)
              for e in own if month <= e.date <= end and _kind(e)]
    return ColonyFill(collected, max(trays) if trays else None, weeks, events)


def _week(own):
    """ColonyWeek from the entries of one week (in time order)."""
    paint_day = [e for e in own if e.event == "painted" or (e.event == "census" and e.what == "pupa")]
    painted = [e for e in paint_day if e.event == "painted"]
    pupae = [e.n for e in paint_day if e.event == "census"]
    dead_ids = sorted({i for e in own if e.event == "dead" for i in e.ids})
    comments = _unique(e.note or " ".join(_unique([e.what, "" if e.n is None else str(e.n)]))
                       for e in own if e.event == "note")
    if not (paint_day or dead_ids or comments):
        return EMPTY_WEEK
    return ColonyWeek((painted or paint_day)[0].date if paint_day else None,
                      " ".join(_unique(e.by for e in paint_day)),
                      " ".join(_unique(e.what for e in painted)),
                      _total(painted) if painted else (0 if paint_day else None),
                      pupae[-1] if pupae else None,
                      tuple(dead_ids), comments)


def _kind(e):
    if e.event == "census":
        return "count" if e.what == "adult" else None
    return EVENT_KINDS.get(e.event)


def _details(e):
    if e.event == "census":
        head = "%d adults" % e.n
    else:
        label = e.event.replace("_", " ") if EVENT_KINDS[e.event] == "other" else ""
        head = " ".join(_unique([label, e.what, "" if e.n is None else str(e.n)]))
        if e.ids:
            head += ": " + " ".join(str(i) for i in e.ids)
    return "; ".join(_unique([head, e.note]))
