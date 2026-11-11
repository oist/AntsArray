#!/usr/bin/env python3
"""Inbox folder names and updates to basler/_colonies/tag_assignments.csv.

The folder a photo sits in says which colony and what was found, because both colonies reuse tag
IDs 0-99 and a photo cannot tell them apart:

    C<colony>_dead      ant found dead              end_reason found_dead
    C<colony>_tag       loose tag found             end_reason tag_lost  (C<colony>_dropped too)
    C<colony>_removed   ant taken out of the colony end_reason removed
    C<colony>_check     live ants photographed to check who wears which tag; never written
    (Colony<colony>_... and any letter case also work)

A finding ends the one active row (blank ended_on) of its (colony, tag_id): ended_on = the date
found, end_reason from the kind, and the evidence photo names are appended. A row already ended
from the same photos is reported as already recorded (a rerun). Anything else - no row, ended
for another reason, two active rows, found before it was tagged, one ID photographed as two
kinds - is reported and left for a person to resolve.

The table is shared and edited by hand, so rows that do not change are written back exactly as
they were read (quoting, line endings, missing final newline); only changed rows are re-encoded.
"""
import codecs
import csv
import io
import os
import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

KINDS = {"dead": ("found_dead", "founddead"),      # kind: (end_reason, evidence file tag)
         "tag": ("tag_lost", "tagfound"),
         "removed": ("removed", "removed")}
ALIASES = {"dropped": "tag"}
CHECK = "check"         # photos of live ants: read and reviewed, never written to the table
FOLDER = re.compile(r"^c(?:olony)?(\d+)_(dead|tag|dropped|removed|check)$", re.IGNORECASE)
LINE_END = re.compile(r"(?<=\n)")
TOOL = "tag_photos"


@dataclass(frozen=True)
class Finding:
    colony: int
    tag_id: int
    kind: str
    date: str               # YYYY-MM-DD found
    photos: tuple           # file names the photos have (or will have) in evidence/


@dataclass(frozen=True)
class Update:
    index: int              # data row in tag_assignments.csv
    colony: int
    tag_id: int
    kind: str
    ended_on: str
    end_reason: str
    evidence_add: str
    photos: tuple
    already: bool = False   # ended earlier from these same photos: nothing to write


@dataclass(frozen=True)
class Table:
    header: list
    rows: list              # dicts, one per data record
    header_raw: str
    records: list           # exact source text of each data record (blank lines before it included)
    tail: str               # blank lines after the last record
    newline: str
    bom: bool
    raw: bytes              # file content as read, to detect a concurrent edit before writing


def parse_folder(name):
    """(colony, kind) from an inbox folder name, or None."""
    m = FOLDER.match(name)
    if not m:
        return None
    kind = m[2].lower()
    return int(m[1]), ALIASES.get(kind, kind)


def evidence_name(colony, kind, photo):
    """Default evidence/ file name of a photo (existing Colony21_founddead_IMG_x style)."""
    return f"Colony{colony}_{KINDS[kind][1]}_{photo}"


def _split_records(text):
    """[(source text, values)] per CSV record, and trailing blank lines."""
    pending = []

    def lines():
        for ln in LINE_END.split(text):
            if ln:
                pending.append(ln)
                yield ln

    out = []
    for values in csv.reader(lines()):
        if values:                       # a blank line stays attached to the next record
            out.append(("".join(pending), values))
            pending.clear()
    return out, "".join(pending)


def read_table(path):
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"{path}: not UTF-8 (saved by Excel as plain CSV?); save it as 'CSV UTF-8' "
                         "and run again") from None
    records, tail = _split_records(text)
    if not records:
        raise ValueError(f"{path}: empty table")
    (header_raw, header), data = records[0], records[1:]
    for n, (_, values) in enumerate(data, 1):
        if len(values) != len(header):
            raise ValueError(f"{path}: data row {n} has {len(values)} fields but the header has "
                             f"{len(header)}; fix the table first")
    rows = [dict(zip(header, values)) for _, values in data]
    newline = "\r\n" if "\r\n" in text else "\n"
    return Table(header, rows, header_raw, [r for r, _ in data], tail, newline,
                 raw.startswith(codecs.BOM_UTF8), raw)


def _line(table, row):
    buf = io.StringIO()
    csv.writer(buf, lineterminator=table.newline).writerow([row.get(h, "") for h in table.header])
    return buf.getvalue()


def _format_row(table, row, old_raw):
    blank = re.match(r"(?:\r?\n)*", old_raw).group(0)
    line = _line(table, row)
    return blank + (line if old_raw.endswith("\n") else line.rstrip("\r\n"))


def _order(row):
    key = _key(row)
    return key if key is not None else (float("inf"), float("inf"))


def insert_rows(table, new_rows):
    """A new Table with new_rows placed in (colony, tag_id) order, after any existing rows of the
    same key; every existing record keeps its source text. Write it with write_table(path, t, t.rows)."""
    records, rows = list(table.records), list(table.rows)
    for new in sorted(new_rows, key=_order):
        i = next((j for j, r in enumerate(rows) if _order(r) > _order(new)), len(rows))
        if i == len(rows) and records and not records[-1].endswith("\n"):
            records[-1] += table.newline
        records.insert(i, _line(table, new))
        rows.insert(i, {h: new.get(h, "") for h in table.header})
    return replace(table, rows=rows, records=records)


def write_table(path, table, rows):
    """Write rows back; unchanged rows keep their source bytes. Temporary file + rename, so a
    failure never leaves half a table (the temporary file is removed again)."""
    parts = [table.header_raw]
    for old_raw, old, new in zip(table.records, table.rows, rows, strict=True):
        parts.append(old_raw if new == old else _format_row(table, new, old_raw))
    parts.append(table.tail)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes((codecs.BOM_UTF8 if table.bom else b"") + "".join(parts).encode("utf-8"))
    try:
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _key(row):
    try:
        return int(row["colony"]), int(row["tag_id"])
    except (KeyError, TypeError, ValueError):
        return None


def _group(findings):
    groups = {}
    for f in findings:
        groups.setdefault((f.colony, f.tag_id), []).append(f)
    return groups


def _mentions(text, name):
    return re.search(rf"(?<![\w.]){re.escape(name)}(?![\w.])", text or "") is not None


def _found_before_tagged(row, found):
    """Problem text when tagged_on is malformed or after the date found, else None."""
    tagged = (row.get("tagged_on") or "").strip()
    if not tagged:
        return None
    try:
        t = date.fromisoformat(tagged)
    except ValueError:
        return f"tagged_on '{tagged}' is not YYYY-MM-DD"
    return f"found {found}, before it was tagged ({tagged})" if t > date.fromisoformat(found) else None


def _plan_one(rows, colony, tag_id, group):
    """(Update, None) or (None, problem) for all findings of one (colony, tag_id)."""
    label = f"colony {colony} tag {tag_id}"
    kinds = sorted({f.kind for f in group})
    if len(kinds) > 1:
        return None, f"{label}: photographed as both {' and '.join(kinds)}; fix the folders and rerun"
    same = [i for i, r in enumerate(rows) if _key(r) == (colony, tag_id)]
    if not same:
        return None, f"{label}: no row in tag_assignments.csv"
    photos = tuple(sorted({p for f in group for p in f.photos}))
    add = f"photo{'s' if len(photos) > 1 else ''} {', '.join(photos)} ({TOOL})"
    # A row that already cites these photos was ended by an earlier run; never end another row
    # (e.g. the same ID put on a new ant since) with the same photos.
    for i in same:
        if all(_mentions(rows[i].get("evidence"), p) for p in photos):
            return Update(i, colony, tag_id, kinds[0], rows[i]["ended_on"], rows[i]["end_reason"], add,
                          photos, already=True), None
    active = [i for i in same if not (rows[i]["ended_on"] or "").strip()]
    if not active:
        last = rows[same[-1]]
        return None, f"{label}: already ended {last['ended_on']} ({last['end_reason']})"
    if len(active) > 1:
        return None, f"{label}: {len(active)} active rows"
    found = min(f.date for f in group)
    problem = _found_before_tagged(rows[active[0]], found)
    if problem:
        return None, f"{label}: {problem}"
    return Update(active[0], colony, tag_id, kinds[0], found, KINDS[kinds[0]][0], add, photos), None


def plan_updates(rows, findings):
    """(updates, problems): one update per (colony, tag_id) that can be ended unambiguously."""
    updates, problems = [], []
    for (colony, tag_id), group in _group(findings).items():
        update, problem = _plan_one(rows, colony, tag_id, group)
        if update:
            updates.append(update)
        else:
            problems.append(problem)
    return updates, problems


def apply_updates(rows, updates):
    """New row list with the updates applied; the input rows are not modified."""
    by_index = {u.index: u for u in updates if not u.already}
    out = []
    for i, r in enumerate(rows):
        u = by_index.get(i)
        if u is None:
            out.append(r)
            continue
        ev = (r.get("evidence") or "").strip()
        out.append({**r, "ended_on": u.ended_on, "end_reason": u.end_reason,
                    "evidence": f"{ev}; {u.evidence_add}" if ev else u.evidence_add})
    return out
