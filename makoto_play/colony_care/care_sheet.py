#!/usr/bin/env python3
"""Print the colony care sheets of one month (one PDF, all A4 landscape).

The set is two shared feeding sheets (days 1-15 and 16-end; all colonies are fed in one round:
one line per day, a dead / left-over pair per colony) followed by one sheet per colony (weekly
paint + count of adults and pupae, dead tag IDs, comments, events). Photographs of filled sheets
become rows of basler/_colonies/husbandry_log.csv (README.md).

The colonies come from colonies.csv (care_from / care_until) in --colonies-dir, or --colonies.

    python makoto_play/colony_care/care_sheet.py --month 2026-10 \
        --colonies-dir Z:/ReiterU/Ants/basler/_colonies --out sheets/care_2026-10.pdf
"""
import argparse
import contextlib
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

import sheet_model as M
import sheet_pages as P


def render(colonies, month, out_pdf, arena=None, png_dir=None, feeding=True):
    """Write the two feeding sheets (unless feeding=False) and one page per colony to out_pdf.

    arena = rows from M.read_arena(), or None when arena.csv was not read (no arena lines).
    A PNG preview per page goes into png_dir. The PDF is written under a temporary name and
    renamed at the end, so a failed run never leaves a PDF that silently lacks pages.
    """
    if not colonies and not feeding:
        raise M.InputError("nothing to print: no colonies and no feeding sheet")
    tag, end = month.strftime("%Y-%m"), M.month_end(month)
    pages = []
    if feeding:
        P.feeding_slots(colonies)                       # fails early when too many colonies
        pages += [("feeding_%s_P%d.png" % (tag, page),
                   lambda page=page: P.draw_feeding_page(month, page, colonies, arena)) for page in (1, 2)]
    pages += [("care_C%s_%s.png" % (M.colony_key(c), tag),
               lambda c=c: P.draw_colony_page(c, month, M.arena_header(arena, c, month, end)))
              for c in colonies]
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if png_dir is not None:
        png_dir = Path(png_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
    part = out_pdf.with_name(out_pdf.name + ".part")
    try:
        with matplotlib.rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}), \
                PdfPages(part) as pdf:
            for png_name, draw in pages:
                fig = draw()
                pdf.savefig(fig)
                if png_dir is not None:
                    fig.savefig(png_dir / png_name, dpi=150)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    try:
        part.replace(out_pdf)
    except OSError as e:            # e.g. the old PDF is open in a viewer on Windows
        raise OSError("could not replace %s (%s); the new PDF is kept as %s" % (out_pdf, e, part)) from e
    return out_pdf


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
    p.add_argument("--month", default=date.today().strftime("%Y-%m"), help="YYYY-MM (default: this month)")
    p.add_argument("--colonies", default="",
                   help="colony names, e.g. 8,Mix,10#1,22-29 (default: the roster in colonies.csv)")
    p.add_argument("--no-feeding", action="store_true", help="leave out the shared feeding sheets")
    p.add_argument("--colonies-dir", type=Path, default=None,
                   help="basler/_colonies: colonies.csv (roster) and arena.csv (who is in the arena)")
    p.add_argument("--out", type=Path, default=None, help="output PDF")
    p.add_argument("--png-dir", type=Path, default=None, help="also write a PNG preview per page")
    a = p.parse_args(argv)
    out = a.out or Path("care_sheets_%s.pdf" % a.month)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.parent / ("care_sheet_log_%s.txt" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(log_path, "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        print("care_sheet %s" % " ".join(argv if argv is not None else sys.argv[1:]))
        try:
            return _run(a, out)
        except (M.InputError, OSError) as e:
            print("error: %s" % e)
            return 2
        except Exception:
            traceback.print_exc(file=sys.stdout)
            raise


def _run(a, out):
    month = M.parse_month(a.month)
    arena = M.read_arena(a.colonies_dir / "arena.csv") if a.colonies_dir else None
    if a.colonies.strip():
        colonies = M.parse_colonies(a.colonies)
    elif a.colonies_dir:
        colonies = M.read_roster(a.colonies_dir / "colonies.csv", month)
    else:
        colonies = []
    render(colonies, month, out, arena=arena, png_dir=a.png_dir, feeding=not a.no_feeding)
    print("care sheets %s -> %s" % (month.strftime("%Y-%m"), out))
    print("  colony sheets: %s" % (", ".join(colonies) or "-"))
    if arena is None:
        print("  arena.csv not read (no --colonies-dir): sheets carry no arena line")
        return 0
    end = M.month_end(month)
    if not a.no_feeding:
        for page, (first, last) in enumerate(M.feeding_pages(month), start=1):
            print("  feeding sheet %d: %s" % (page, M.feeding_arena_line(arena, first, last)[0]))
    keys = {M.colony_key(c) for c in colonies}
    for c in M.arena_colonies(arena, month, end):
        if M.colony_key(c) not in keys:
            print("  WARNING: colony %s is in the arena this month but has no colony sheet" % c)
    for c in colonies:
        text, in_arena = M.arena_header(arena, c, month, end)
        if in_arena:
            print("  colony %s: %s" % (c, text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
