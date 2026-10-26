#!/usr/bin/env python3
"""One-time import of the Microsoft Forms feeding records into the husbandry log (legacy data).

Before the paper care sheets each feeding was recorded with a Microsoft Form, one response per
box, exported as "Check of Ants Box.xlsx":

    sheet Form1     Date, Name2, Box number, Dead number (total) (females + males), Feeding,
                    Water exchang, Remarks, Dead number of male
    sheet Box List  boxes 1-6: collection date, place, size, queen, males, remarks

Each response becomes husbandry_log.csv rows with source "forms ID <n>": fed / not_fed, dead
(females = total - males; before the male column existed the males come from remarks such as
"male 3 died"), dead what=male, water, and from the remarks: leftover y ("food left"), tray_changed,
nest_changed, and a note when a colony died out. Box names are unified (full-width digits,
mix/MIX/miz, the 13tag-R spellings, "4 with queen", ...). Box 10 is 10#1 and 10q is 10#2 from
the first day 10q was recorded; before that 10 is the undivided colony 10. colonies.csv gets
every colony with its first and last record as care_from / care_until (colonies on the current
roster keep a blank care_until) and the Box List details. A second import is refused.

    python makoto_play/colony_care/import_forms.py --xlsx "Check of Ants Box.xlsx" \
        --colonies-dir Z:/ReiterU/Ants/basler/_colonies            (dry run: prints the plan)
    ... --yes        writes; the old tables are kept in _colonies/backups/<time>/
"""
import argparse
import contextlib
import csv
import io
import re
import shutil
import sys
import traceback
import unicodedata
from collections import Counter, namedtuple
from datetime import date, datetime, timedelta
from pathlib import Path

import husbandry as H
import sheet_model as M
import xlsx_read as X

# Form1 columns (the start-time column is named in Japanese by Forms)
ID, START, DATE, NAME, BOX = "ID", "\u958b\u59cb\u6642\u523b", "Date", "Name2", "Box number"
DEAD, FEED, WATER, REMARKS, MALE = ("Dead number (total)", "Feeding", "Water exchang", "Remarks",
                                    "Dead number of male")
FORM_COLUMNS = (ID, DATE, BOX, DEAD, FEED, WATER)
# Box List columns: number, collection date, place, size (ants), queen, males, remarks
BOX_NUMBER, COLLECTED, PLACE = "Box Number", "\u63a1\u53d6\u65e5", "\u63a1\u53d6\u5834\u6240"
SIZE, QUEEN, MALES, BOX_REMARKS = ("\u30b5\u30a4\u30ba(\u5339\u6570)", "\u5973\u738b\u30a2\u30ea\u6709\u7121",
                                   "\u2642\u30a2\u30ea\u6709\u7121", "remarks")
# words in the remarks
LEFTOVER_WORD = "\u6b8b\u990c"                          # food left over
TRAY_WORD = "\u30c8\u30ec\u30a4\u4ea4\u63db"            # tray changed
NEST_WORD = "\u5de3\u306e\u4ea4\u63db"                  # nest changed
WIPED_WORD = "\u5168\u6ec5"                              # colony died out
HONEY_WORD = "\u306f\u3061\u307f\u3064"                  # honey
DEATH_WORD = "\u6b7b"                                    # died
MALE_DEATHS = re.compile("(?:\u96c4|\u30aa\u30b9)\u304c?\\s*([0-9]+)")   # "male 3" / "the male 1" in a remark
FOOD_CODES = {"mealworm": "MW", "jelly": "J", "paste": "Paste", "pellets": "Pellets"}
FOOD_ORDER = ("MW", "J", "Paste", "HoneyPaste", "Pellets")
SOURCE = "forms ID %s"
SOURCE_PREFIX = "forms "
LEGACY_DIR = Path("evidence") / "legacy"

Plan = namedtuple("Plan", "xlsx split_day log_rows colony_rows fieldnames eol repeats skipped")


def _norm(text):
    return unicodedata.normalize("NFKC", text or "").strip()


# --- conversion rules -----------------------------------------------------------------------

def box_name(raw, day, split_day):
    """Colony name for a Forms box name on a day (see the module docstring)."""
    s = _norm(raw)
    low = s.lower()
    if low in ("mix", "miz"):
        return "Mix"
    if re.fullmatch(r"10-?q", low):
        return "10#2"
    if s == "10":
        return "10#1" if split_day and day >= split_day else "10"
    if re.fullmatch(r"13-?tag[-_(]?r\)?", low):
        return "13tag-R"
    if low in ("13tag", "13t"):
        return "13tag"
    m = re.fullmatch(r"([0-9]+) .+", s)                 # "4 with queen", "4 untagged", ...
    if m:
        s = m.group(1)
    try:
        return M.check_colony_name(s, "box %r" % raw)
    except M.InputError:
        raise M.InputError("box %r: no colony name for it; add a rule to box_name()" % raw) from None


def feeding(raw):
    """('fed', food codes) or ('not_fed', '') for the Feeding cell."""
    s = _norm(raw)
    if s.lower() == "no":
        return "not_fed", ""
    if s.lower() in ("", "give"):
        return "fed", "given"
    codes = []
    for tok in (t.strip() for t in s.split(";")):
        if not tok:
            continue
        code = FOOD_CODES.get(tok.lower()) or ("HoneyPaste" if HONEY_WORD in tok else
                                               re.sub(r"[^A-Za-z0-9]+", "", tok) or "other")
        if code not in codes:
            codes.append(code)
    codes.sort(key=lambda c: FOOD_ORDER.index(c) if c in FOOD_ORDER else len(FOOD_ORDER))
    return "fed", "+".join(codes)


def count(raw, label):
    """(number, note): a plain number, or the sum of the numbers in a free-text cell (noted)."""
    s = _norm(raw)
    if not s:
        return None, ""
    if s.isdigit():
        return int(s), ""
    numbers = re.findall(r"[0-9]+", s)
    return (sum(map(int, numbers)) if numbers else None), "%s: %s" % (label, s)


def male_deaths_in(remarks):
    """Dead males written in a (normalised) remark, e.g. 'male 3 died'; None when not stated.

    A remark about males without a death ('a male appeared', '2 males') says nothing about deaths.
    """
    if DEATH_WORD not in remarks:
        return None
    m = MALE_DEATHS.search(remarks)
    return int(m.group(1)) if m else None


def _event(base, event, what="", n=None, note=""):
    return dict(base, event=event, what=what, n="" if n is None else str(n), note=note)


def convert(row, split_day):
    """husbandry_log rows (dicts) for one Form1 response."""
    day_text = (row.get(DATE) or "")[:10]
    day = date.fromisoformat(day_text)
    raw_box = (row.get(BOX) or "").strip()
    colony = box_name(raw_box, day, split_day)
    start = row.get(START) or ""
    remark_raw = (row.get(REMARKS) or "").strip()
    remarks = _norm(remark_raw)
    total, dead_note = count(row.get(DEAD), "dead")         # Forms counted males in the total
    male, male_note = count(row.get(MALE), "male")
    if male is None:
        male = male_deaths_in(remarks)                     # before the male column existed
    female, odd = (None, "") if total is None else (total - (male or 0), "")
    if female is not None and female < 0:
        female, odd = 0, "dead: %d in total but %d males" % (total, male)
    notes = [n for n in (remark_raw, "box %s" % raw_box if raw_box != colony else "", dead_note, male_note) if n]
    base = dict(date=day_text, time=start[11:16] if start[:10] == day_text and len(start) >= 16 else "",
                colony=colony, ids="", block="", by=_norm(row.get(NAME)), source=SOURCE % row.get(ID),
                provisional="")
    kind, what = feeding(row.get(FEED))
    events = [_event(base, kind, what, note="; ".join(notes))]
    if female:
        events.append(_event(base, "dead", n=female))
    if male:
        events.append(_event(base, "dead", H.MALE, male, note=odd))
    if _norm(row.get(WATER)).lower() == "did":
        events.append(_event(base, "water"))
    for word, event, value in ((LEFTOVER_WORD, "leftover", "y"), (TRAY_WORD, "tray_changed", ""),
                               (NEST_WORD, "nest_changed", ""), (WIPED_WORD, "note", "wiped out")):
        if word in remarks:
            events.append(_event(base, event, value))
    return events


def box_list(rows, split_day):
    """({colony: (collected_on, details)}, events) from the Box List sheet."""
    head, info, events = rows[0], {}, []
    for r in rows[1:]:
        cell = dict(zip(head, r))
        if not cell.get(BOX_NUMBER):
            continue
        collected = (cell.get(COLLECTED) or "")[:10]
        day = date.fromisoformat(collected) if collected else date.min
        colony = box_name(cell[BOX_NUMBER], day, split_day)
        size = _norm(cell.get(SIZE))
        details = ["collected at %s" % cell.get(PLACE) if cell.get(PLACE) else "",
                   "size %s" % size if size else "", "queen %s" % cell.get(QUEEN) if cell.get(QUEEN) else "",
                   "males %s" % cell.get(MALES) if cell.get(MALES) else "", cell.get(BOX_REMARKS) or ""]
        info[colony] = (collected, "; ".join(d for d in details if d))
        if collected:
            base = dict(date=collected, time="", colony=colony, ids="", block="", by="",
                        source="forms Box List", provisional="")
            n = int(size) if size.isdigit() else None
            events.append(_event(base, "collected", cell.get(PLACE) or "", n))
            if n is not None:
                events.append(_event(base, "census", "adult", n, note="size at collection"))
    return info, events


# --- plan and apply -------------------------------------------------------------------------

def _form_rows(xlsx):
    form = X.read_sheet(xlsx, "Form1")
    head = form[0]
    missing = [c for c in FORM_COLUMNS if c not in head]
    if missing:
        raise M.InputError("%s, sheet Form1: missing column(s) %s" % (xlsx, ", ".join(missing)))
    return [dict(zip(head, r)) for r in form[1:] if any(r)]


def _split_day(rows):
    days = [r[DATE][:10] for r in rows if re.fullmatch(r"10-?q", _norm(r.get(BOX)).lower())]
    return date.fromisoformat(min(days)) if days else None


def _deduplicate(rows, split_day):
    """Rows minus exact repeats (same day, colony and values), and the list of repeats."""
    seen, kept, repeats = {}, [], []
    for r in rows:
        day = date.fromisoformat(r[DATE][:10])
        key = (r[DATE][:10], box_name(r.get(BOX), day, split_day)) + tuple(
            _norm(r.get(c)) for c in (DEAD, FEED, WATER, REMARKS, MALE))
        if key in seen:
            repeats.append("%s repeats %s (%s, box %s)" % (SOURCE % r.get(ID), SOURCE % seen[key],
                                                           key[0], (r.get(BOX) or "").strip()))
            continue
        seen[key] = r.get(ID)
        kept.append(r)
    return kept, repeats


def _read_colonies(path):
    raw = Path(path).read_bytes()
    eol = "\r\n" if b"\r\n" in raw else "\n"
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    return list(reader.fieldnames), [dict(r) for r in reader], eol


def _roster(path, log_rows, info, split_day):
    """colonies.csv rows with care_from / care_until and Box List details from the records."""
    fields, rows, eol = _read_colonies(path)
    first, last = {}, {}
    for e in log_rows:
        c = e["colony"]
        first[c] = min(first.get(c, e["date"]), e["date"])
        last[c] = max(last.get(c, e["date"]), e["date"])
    if split_day and "10" in last:
        last["10"] = (split_day - timedelta(days=1)).isoformat()
    known = {M.colony_key(r["colony"]) for r in rows}
    for r in rows:
        c = r["colony"]
        if c in first:
            current = bool(r["care_from"]) and not r["care_until"]
            r["care_from"] = min(r["care_from"] or first[c], first[c])
            if not current:
                r["care_until"] = max(r["care_until"], last[c])
        if c in info:
            r["collected_on"] = r["collected_on"] or info[c][0]
            r["note"] = "; ".join(n for n in (r["note"], info[c][1]) if n)
    for c in sorted((c for c in first if M.colony_key(c) not in known), key=lambda c: (first[c], c)):
        collected, details = info.get(c, ("", ""))
        note = "; ".join(n for n in ("legacy box from the Forms records (%s..%s)" % (first[c], last[c]), details) if n)
        rows.append(dict({k: "" for k in fields}, colony=c, collected_on=collected, note=note,
                         care_from=first[c], care_until=last[c]))
    return fields, rows, eol


def make_plan(xlsx, colonies_dir):
    """Everything the import would write, without writing it."""
    d = Path(colonies_dir)
    existing = H.read_log(d / "husbandry_log.csv")
    if any(e.source.startswith(SOURCE_PREFIX) for e in existing):
        raise M.InputError("the Forms records are already imported into %s" % (d / "husbandry_log.csv"))
    rows = _form_rows(xlsx)
    skipped = ["%s: no Date" % (SOURCE % r.get(ID)) for r in rows if not r.get(DATE)]
    rows = [r for r in rows if r.get(DATE)]
    split_day = _split_day(rows)
    rows, repeats = _deduplicate(rows, split_day)
    info, log_rows = box_list(X.read_sheet(xlsx, "Box List"), split_day)
    for r in rows:
        log_rows.extend(convert(r, split_day))
    fields, colony_rows, eol = _roster(d / "colonies.csv", log_rows, info, split_day)
    return Plan(Path(xlsx), split_day, log_rows, colony_rows, fields, eol, repeats, skipped)


def _write_part(path, text, check=None):
    part = path.with_name(path.name + ".part")
    part.write_bytes(text.encode("utf-8"))
    if check:
        check(part)
    part.replace(path)


def apply(plan, colonies_dir, stamp=None):
    """Back up the two tables, copy the workbook to evidence/legacy/, and write the import."""
    d = Path(colonies_dir)
    backup = d / "backups" / (stamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup.mkdir(parents=True, exist_ok=True)
    for name in ("colonies.csv", "husbandry_log.csv"):
        shutil.copy2(d / name, backup / name)
    (d / LEGACY_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan.xlsx, d / LEGACY_DIR / plan.xlsx.name)
    log_path = d / "husbandry_log.csv"
    old = log_path.read_bytes().decode("utf-8")
    eol = "\r\n" if "\r\n" in old else "\n"
    out = io.StringIO()
    out.write(old if not old or old.endswith(("\n", "\r")) else old + eol)
    csv.DictWriter(out, fieldnames=H.FIELDS, lineterminator=eol).writerows(plan.log_rows)
    _write_part(log_path, out.getvalue(), check=H.read_log)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=plan.fieldnames, lineterminator=plan.eol)
    writer.writeheader()
    writer.writerows(plan.colony_rows)
    _write_part(d / "colonies.csv", out.getvalue())
    return backup


# --- command line ---------------------------------------------------------------------------

class _Tee:
    """Write to several streams at once (the console and the run log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def report(plan):
    print("workbook %s; 10 -> 10#1 / 10q -> 10#2 from %s" % (plan.xlsx, plan.split_day))
    print("rows to add to husbandry_log.csv: %d" % len(plan.log_rows))
    for event, n in sorted(Counter(e["event"] for e in plan.log_rows).items()):
        print("  %-13s %6d" % (event, n))
    print("repeated responses left out: %d" % len(plan.repeats))
    for r in plan.repeats:
        print("  " + r)
    for s in plan.skipped:
        print("  skipped: " + s)
    print("colonies.csv after the import (colony, care_from, care_until, collected_on):")
    for r in plan.colony_rows:
        print("  %-8s %-10s %-10s %s" % (r["colony"], r["care_from"], r["care_until"], r["collected_on"]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xlsx", type=Path, required=True, help="the Forms export (Check of Ants Box.xlsx)")
    p.add_argument("--colonies-dir", type=Path, required=True, help="basler/_colonies")
    p.add_argument("--yes", action="store_true", help="write (default: dry run)")
    a = p.parse_args(argv)
    logs = a.colonies_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / ("import_forms_log_%s.txt" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        print("import_forms %s" % " ".join(argv if argv is not None else sys.argv[1:]))
        try:
            plan = make_plan(a.xlsx, a.colonies_dir)
            report(plan)
            if not a.yes:
                print("dry run: nothing written (add --yes to write)")
                return 0
            backup = apply(plan, a.colonies_dir)
            print("written; previous tables kept in %s" % backup)
            return 0
        except (M.InputError, OSError, ValueError) as e:
            print("error: %s" % e)
            return 2
        except Exception:
            traceback.print_exc(file=sys.stdout)
            raise


if __name__ == "__main__":
    sys.exit(main())
