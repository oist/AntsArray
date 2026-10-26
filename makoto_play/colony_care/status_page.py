#!/usr/bin/env python3
"""Colony status page: one self-contained HTML with an overview of every colony on the care
roster and a section per colony (computed adults, feeding, pupae, deaths, tray, history).

    python makoto_play/colony_care/status_page.py --colonies-dir Z:/ReiterU/Ants/basler/_colonies

writes <colonies-dir>/status.html (open it in a browser; it needs nothing else) and a run log
in <colonies-dir>/logs/.
"""
import argparse
import contextlib
import csv
import html
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

import husbandry as H
import sheet_model as M

CSS = """
:root { --bg:#ffffff; --fg:#1d1d1f; --muted:#6e6e73; --line:#d2d2d7; --head:#f5f5f7; --prov:#fff4d6; --warn:#b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg:#161617; --fg:#f5f5f7; --muted:#a1a1a6; --line:#3a3a3c;
  --head:#232325; --prov:#3a3220; --warn:#ff8a80; } }
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0; }
main { max-width:1400px; margin:0 auto; padding:16px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:18px; margin:28px 0 6px; }
.muted { color:var(--muted); } .warn { color:var(--warn); }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; margin:8px 0; }
th, td { border-bottom:1px solid var(--line); padding:4px 8px; text-align:left; vertical-align:top; white-space:nowrap; }
td.note { white-space:normal; min-width:16em; }
th { background:var(--head); position:sticky; top:0; }
tr.prov td { background:var(--prov); }
td.adults { font-weight:600; }
a { color:inherit; }
dl { display:grid; grid-template-columns:max-content 1fr; gap:2px 16px; margin:6px 0; }
dt { color:var(--muted); }
"""
OVERVIEW_HEAD = ("Colony", "Arena", "Adults", "Last full count", "Pupae", "New adults 28 d",
                 "Dead 28 d", "Last fed", "Left over", "Water", "Tray", "Last paint", "Tags in use")
HISTORY_HEAD = ("Date", "Time", "Event", "What", "n", "IDs", "Block", "By", "Note", "Source")
HISTORY_ROWS = 200          # newest rows shown per colony; the rest stay in husbandry_log.csv


def _e(value):
    return html.escape("" if value is None else str(value))


def tags_in_use(path):
    """{colony key: number of tag rows without ended_on} from tag_assignments.csv ({} if absent)."""
    path = Path(path)
    if not path.is_file():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("colony") or "").strip() and not (r.get("ended_on") or "").strip():
                key = M.colony_key(r["colony"].strip())
                out[key] = out.get(key, 0) + 1
    return out


def _adults_text(est):
    if est is None:
        return "-"
    return "%d%s%s" % (est.n, "*" if est.provisional else "", " -?" if est.unknown else "")


def _dated(pair):
    return "%s: %s" % pair if pair else "-"


def _arena(arena, colony, on):
    side = M.arena_side(arena, colony, on) if arena is not None else None
    return "%s half" % side[0] if side else ""


def _tray(s):
    return "%s (%s)" % (s.tray[0], s.tray[1]) if s.tray[0] else "unknown"


def _overview_row(s, arena, tags, on):
    est = s.adults
    key = M.colony_key(s.colony)
    cells = [
        ("name", '<a href="#c-%s">%s</a>' % (_e(key), _e(s.colony))),
        ("arena", _e(_arena(arena, s.colony, on))),
        ("adults", _e(_adults_text(est))),
        ("count", _e("%s: %d" % (est.base_date, est.base_n) if est else "no full count yet")),
        ("pupae", _e(_dated(s.pupae))),
        ("new", _e(s.new_28d)),
        ("dead", _e(s.dead_28d)),
        ("fed", _e(s.last_fed or "-")),
        ("leftover", _e(_dated(s.last_leftover))),
        ("water", _e(s.last_water or "-")),
        ("tray", _e(_tray(s))),
        ("paint", _e(_dated(s.last_paint))),
        ("tags", _e(tags.get(key, "-"))),
    ]
    return "<tr>%s</tr>" % "".join('<td class="%s">%s</td>' % c for c in cells)


def _table(head, rows):
    th = "".join("<th>%s</th>" % _e(h) for h in head)
    return '<div class="scroll"><table><thead><tr>%s</tr></thead><tbody>\n%s\n</tbody></table></div>' % (
        th, "\n".join(rows))


def _adults_sentence(est):
    if est is None:
        return "no full count yet - record one (colony sheet, Events: count) to start the computed number"
    text = "%d%s = %d (full count %s) + %d new - %d dead or removed" % (
        est.n, "*" if est.provisional else "", est.base_n, est.base_date, est.added, est.lost)
    if est.unknown:
        text += ". Not included (number unknown): " + "; ".join(est.unknown)
    return text


def _history(entries):
    if not entries:
        return '<p class="muted">No records yet.</p>'
    rows = []
    for e in entries[:HISTORY_ROWS]:
        cells = [e.date, e.time, e.event, e.what, e.n, " ".join(map(str, e.ids)), e.block, e.by]
        tds = "".join("<td>%s</td>" % _e(v) for v in cells)
        tds += '<td class="note">%s</td><td class="note">%s</td>' % (_e(e.note), _e(e.source))
        rows.append('<tr%s>%s</tr>' % (' class="prov"' if e.provisional else "", tds))
    more = len(entries) - HISTORY_ROWS
    tail = ('<p class="muted">%d older rows not shown (see husbandry_log.csv).</p>' % more) if more > 0 else ""
    return _table(HISTORY_HEAD, rows) + tail


def _section(s, arena, tags, on):
    key = M.colony_key(s.colony)
    facts = [("Adults", _adults_sentence(s.adults)),
             ("Arena", _arena(arena, s.colony, on) or "not in the arena"),
             ("Pupae (last count)", _dated(s.pupae)),
             ("Last 28 days", "%d new adults, %d dead" % (s.new_28d, s.dead_28d)),
             ("Last fed", s.last_fed or "-"),
             ("Left over (last)", _dated(s.last_leftover)),
             ("Water changed (last)", s.last_water or "-"),
             ("Tray", _tray(s)),
             ("Last paint", _dated(s.last_paint)),
             ("Tags in use", tags.get(key, "-"))]
    dl = "".join("<dt>%s</dt><dd>%s</dd>" % (_e(k), _e(v)) for k, v in facts)
    return '<section id="c-%s"><h2>Colony %s</h2><dl>%s</dl>%s</section>' % (
        _e(key), _e(s.colony), dl, _history(s.history))


LEGACY_HEAD = ("Colony", "Cared for", "Collected", "Feedings", "Dead (females)", "Dead males", "Notes")


def _legacy(table_rows, entries, month):
    """Table of the colonies whose care ended before the month (legacy records)."""
    ended = [r for r in table_rows if r["care_from"] and r["care_until"] and r["care_until"] < month.isoformat()]
    if not ended:
        return ""
    rows = []
    for r in sorted(ended, key=lambda r: (r["care_from"], r["colony"])):
        t = H.totals(entries, r["colony"])
        cells = [r["colony"], "%s - %s" % (r["care_from"], r["care_until"]), r["collected_on"], t.feedings,
                 t.dead, t.males]
        rows.append("<tr>%s<td class=\"note\">%s</td></tr>" % ("".join("<td>%s</td>" % _e(v) for v in cells),
                                                             _e("; ".join(t.notes + (r["note"],)))))
    return '<section id="legacy"><h2>Legacy colonies</h2>%s</section>' % _table(LEGACY_HEAD, rows)


def build_html(colonies_dir, on):
    """The status page of the colonies on the care roster of on's month, as of `on`."""
    d = Path(colonies_dir)
    month = on.replace(day=1)
    roster = M.read_roster(d / "colonies.csv", month)
    table_rows = [r for _, r in M.read_rows(d / "colonies.csv", M.ROSTER_FIELDS)]
    entries = H.read_log(d / "husbandry_log.csv")
    arena = M.read_arena(d / "arena.csv") if (d / "arena.csv").is_file() else None
    tags = tags_in_use(d / "tag_assignments.csv")
    known = {M.colony_key(r["colony"]) for r in table_rows}
    stray = sorted({e.colony for e in entries if e.colony != H.ALL and M.colony_key(e.colony) not in known})
    summaries = [H.colony_summary(entries, c, on) for c in roster]
    warning = ('<p class="warn">Colonies in the log that are not in colonies.csv: %s</p>'
               % _e(", ".join(stray))) if stray else ""
    return "\n".join([
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
        ' initial-scale=1"><title>Colony status</title><style>%s</style></head><body><main>' % CSS,
        "<h1>Colony status</h1>",
        '<p class="muted">As of %s (generated %s). Adults = last full count + new adults - dead - '
        'removed since. * = includes provisional rows (entered before the paper records were merged). '
        '-? = some losses of unknown size are not included. 28 d = last 28 days.</p>'
        % (_e(on), _e(datetime.now().strftime("%Y-%m-%d %H:%M"))),
        warning,
        _table(OVERVIEW_HEAD, [_overview_row(s, arena, tags, on) for s in summaries]),
        '<p><a href="#legacy">Legacy colonies</a> (care ended before this month) are listed at the end.</p>',
        "\n".join(_section(s, arena, tags, on) for s in summaries),
        _legacy(table_rows, entries, month),
        "</main></body></html>",
    ])


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--colonies-dir", type=Path, required=True, help="basler/_colonies")
    p.add_argument("--on", type=date.fromisoformat, default=date.today(), help="as of YYYY-MM-DD (default today)")
    p.add_argument("--out", type=Path, default=None, help="output HTML (default <colonies-dir>/status.html)")
    a = p.parse_args(argv)
    out = a.out or a.colonies_dir / "status.html"
    logs = out.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / ("status_page_log_%s.txt" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        print("status_page %s" % " ".join(argv if argv is not None else sys.argv[1:]))
        try:
            page = build_html(a.colonies_dir, a.on)
            part = out.with_name(out.name + ".part")
            part.write_text(page, encoding="utf-8")
            part.replace(out)
            print("status page as of %s -> %s" % (a.on, out))
            return 0
        except (M.InputError, OSError) as e:
            print("error: %s" % e)
            return 2
        except Exception:
            traceback.print_exc(file=sys.stdout)
            raise


if __name__ == "__main__":
    sys.exit(main())
