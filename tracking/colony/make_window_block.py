#!/usr/bin/env python3
"""Materialize a chunk-range window of a block as a sibling "view" block.

Why: the tracking pipeline, the analysis scripts and the curation GUI are all
block-scoped -- they read ``<block>/data/`` and write ``<block>/{tracks,
stitched,...}``. A long recording processed in disjoint spans (the first and
last 24 h of a 99 h block, say) cannot be tracked as one block: the two spans
would race on the same state files and overwrite each other's stitched track
IDs. So each finished span becomes its own block-shaped directory next to the
real one::

    20260810/block02/                 the real block: raw videos, data/, contract
    20260810/block02-w000-031/        view: symlinks + copied contract
      cam*.mkv, *.json, sess_*.txt    -> ../block02/...      (GUI playback, sidecars)
      data/<chunk 000..031 files>     -> ../../block02/data/... (original names)
      data/PIPELINE_STATE.json        copy, plus a "window" key
      WINDOW.json                     provenance
      tracks/ stitched/ ...           written by the tracking pipeline as usual

Nothing is renamed or re-indexed: chunk indices and the timestamps inside file
names stay what the recorder produced, so stitching anchors global frames at
the true position (chunk_idx x frames_per_chunk) and analysis derives the
correct wall-clock time. Every link is RELATIVE, so the view resolves on deigo,
on saion and on a laptop mount alike.

The windows themselves are declared once, in ``<block>/data/WINDOWS.json``, and
every command reads them from there -- retyping a range is how a window ends
up covering the wrong hours.

Commands (deigo login; system python3 is enough)::

    make_window_block.py init        --block <dir> --ranges 0-31,32-70,71-109,110-148,149-197
    make_window_block.py status      --block <dir>
    make_window_block.py materialize --block <dir> --window w149-197 [--force] [--dry-run]
"""
import argparse
import datetime
import json
import os
import re
import sys

# pipeline_state lives with the detection pipeline; this script ships in the
# same checkout (tracking/colony/), so locate it relative to ourselves rather
# than depending on PYTHONPATH.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "detection_pipeline", "lib"))
import pipeline_state as ps  # noqa: E402

WINDOWS_BASENAME = "WINDOWS.json"
WINDOW_MARKER = "WINDOW.json"
SCHEMA = 1

# Detection output files carry their chunk index right after the video name:
# <vname>_<NNN>[_aruco_tracks|_aruco_detections|_sleap_data].<h5|slp|...>.
# The video name itself holds a timestamp like 2026-08-10-18-43-30, whose
# segments are '-'-separated, so "_NNN" followed by "_" or "." is unambiguous.
_DATA_FILE_RE = re.compile(r"^(?P<v>.+?)_(?P<i>\d{3})(?P<rest>(?:_[A-Za-z_]+)?\.[A-Za-z0-9]+)$")
_CONTROL_FILES = ("PIPELINE_STATE", "MULTIUSER_PLAN", "WINDOWS")

# Outputs whose presence means a chunk is done, mirroring track_trigger.sh:
# aruco gates on _aruco_tracks.h5, sleap on _sleap_data.h5.
_GATE_STAGES = ("trk", "sdat")


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def window_name(lo, hi):
    return "w%03d-%03d" % (lo, hi)


def view_dir_for(block_dir, lo, hi):
    block_dir = os.path.abspath(block_dir)
    return os.path.join(os.path.dirname(block_dir),
                        "%s-%s" % (os.path.basename(block_dir), window_name(lo, hi)))


# ---------------------------------------------------------------------------
# WINDOWS.json
# ---------------------------------------------------------------------------
def windows_path(block_dir):
    return os.path.join(block_dir, "data", WINDOWS_BASENAME)


def load_windows(block_dir):
    p = windows_path(block_dir)
    if not os.path.isfile(p):
        raise ValueError("%s not found; run 'init' first" % p)
    with open(p) as f:
        doc = json.load(f)
    if doc.get("schema") != SCHEMA or not isinstance(doc.get("windows"), list):
        raise ValueError("%s is not a windows file this code understands" % p)
    return doc


def block_chunk_total(block_dir):
    """(chunk indices declared by the contract, chunk_sec, state)."""
    state = ps.load(os.path.join(block_dir, "data"))
    if state is None:
        raise ValueError("%s/data has no PIPELINE_STATE.json; the block's chunk count "
                         "is unknown, so windows cannot be validated" % block_dir)
    videos = state.get("chunking", {}).get("videos", {})
    total = max(int(v.get("n_chunks", 0)) for v in videos.values()) if videos else 0
    if total <= 0:
        raise ValueError("contract declares no chunks")
    return total, int(state["chunking"]["chunk_sec"]), state


def parse_ranges(text):
    """'0-31,32-70' -> [(0, 31), (32, 70)], sorted; overlap is an error."""
    ranges = []
    for tok in (t.strip() for t in text.split(",")):
        if not tok:
            continue
        lo, hi = ps.parse_range(tok)
        ranges.append((lo, hi))
    ranges.sort()
    for (a_lo, a_hi), (b_lo, b_hi) in zip(ranges, ranges[1:]):
        if b_lo <= a_hi:
            raise ValueError("windows overlap: %s and %s" % (window_name(a_lo, a_hi),
                                                             window_name(b_lo, b_hi)))
    if not ranges:
        raise ValueError("no ranges given")
    return ranges


def coverage_gaps(ranges, total):
    """Chunk indices in 0..total-1 that no window claims."""
    claimed = set()
    for lo, hi in ranges:
        claimed.update(range(lo, hi + 1))
    return sorted(set(range(total)) - claimed)


def cmd_init(args):
    block_dir = os.path.abspath(args.block)
    total, chunk_sec, _state = block_chunk_total(block_dir)
    ranges = parse_ranges(args.ranges)
    if ranges[-1][1] >= total:
        raise ValueError("window %s runs past the block's last index %d"
                         % (window_name(*ranges[-1]), total - 1))
    gaps = coverage_gaps(ranges, total)
    if gaps and not args.allow_gaps:
        raise ValueError("windows leave %d chunk(s) unclaimed (%s); pass --allow-gaps "
                         "if that is intended" % (len(gaps), ps.fmt_ranges(gaps)))

    p = windows_path(block_dir)
    if os.path.exists(p) and not args.force:
        raise ValueError("%s exists; pass --force to overwrite" % p)
    doc = {
        "schema": SCHEMA,
        "block": block_dir,
        "chunk_sec": chunk_sec,
        "n_chunks": total,
        "created": _utcnow(),
        "windows": [{"name": window_name(lo, hi), "range": [lo, hi],
                     "hours": round((hi - lo + 1) * chunk_sec / 3600.0, 2)}
                    for lo, hi in ranges],
    }
    with open(p, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    _group_share(p, args.group, is_dir=False)
    sys.stderr.write("[OK] wrote %s: %d windows over %d chunks (%ds each)\n"
                     % (p, len(ranges), total, chunk_sec))
    for w in doc["windows"]:
        sys.stderr.write("     %-10s %3d-%3d  %5.1f h\n"
                         % (w["name"], w["range"][0], w["range"][1], w["hours"]))
    if gaps:
        sys.stderr.write("[WARN] unclaimed: %s\n" % ps.fmt_ranges(gaps))
    return 0


# ---------------------------------------------------------------------------
# completeness (from data/, same gate as track_trigger.sh)
# ---------------------------------------------------------------------------
def window_completeness(block_dir, lo, hi, state):
    """{stage: (done, expected, missing_indices)} for trk/sdat over lo..hi."""
    want = ps.wave_indices(state, {"chunk_range": [lo, hi]})
    expected = sum(len(s) for s in want.values())
    found = ps.scan_outputs(os.path.join(block_dir, "data"))
    out = {}
    for stage in _GATE_STAGES:
        obs = found.get(stage, {})
        done = 0
        missing = set()
        for vname, idxs in want.items():
            have = obs.get(vname, set()) & idxs
            done += len(have)
            missing.update(idxs - have)
        out[stage] = (done, expected, sorted(missing))
    return out


def cmd_status(args):
    block_dir = os.path.abspath(args.block)
    doc = load_windows(block_dir)
    _total, _cs, state = block_chunk_total(block_dir)
    for w in doc["windows"]:
        lo, hi = w["range"]
        comp = window_completeness(block_dir, lo, hi, state)
        complete = all(d == e and e > 0 for d, e, _m in comp.values())
        view = view_dir_for(block_dir, lo, hi)
        view_state = "view:materialized" if os.path.isdir(view) else "view:-"
        parts = ["%s %d/%d" % (s, comp[s][0], comp[s][1]) for s in _GATE_STAGES]
        print("%-10s %3d-%3d  %-8s  %s  %s"
              % (w["name"], lo, hi, "DONE" if complete else "pending",
                 "  ".join(parts), view_state))
        for s in _GATE_STAGES:
            if comp[s][2]:
                print("             missing %s at chunks %s"
                      % (s, ps.fmt_ranges(comp[s][2])))
    return 0


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------
def _group_share(path, group, is_dir):
    """Best-effort chgrp + group-writable, mirroring detection_pipeline/lib/perms.sh."""
    if not group:
        return
    try:
        import grp
        gid = grp.getgrnam(group).gr_gid
        os.chown(path, -1, gid)
        os.chmod(path, 0o2775 if is_dir else 0o664)
    except (KeyError, OSError, ImportError):
        pass


def _link(target_rel, link_path, dry_run, made):
    """Create a relative symlink; skip if an identical one already exists."""
    if os.path.islink(link_path):
        if os.readlink(link_path) == target_rel:
            made["kept"] += 1
            return
        raise ValueError("%s exists and points elsewhere (%s)" % (link_path, os.readlink(link_path)))
    if os.path.exists(link_path):
        raise ValueError("%s exists and is not a symlink" % link_path)
    if not dry_run:
        os.symlink(target_rel, link_path)
    made["new"] += 1


def data_files_in_window(data_dir, lo, hi):
    """Names in data/ whose chunk index falls in lo..hi (outputs and .slp)."""
    out = []
    for name in sorted(os.listdir(data_dir)):
        if name.startswith(_CONTROL_FILES):
            continue
        m = _DATA_FILE_RE.match(name)
        if m is None:
            continue
        idx = int(m.group("i"))
        if lo <= idx <= hi:
            out.append(name)
    return out


def materialize(block_dir, lo, hi, state, group, force=False, dry_run=False):
    block_dir = os.path.abspath(block_dir)
    block_base = os.path.basename(block_dir)
    view = view_dir_for(block_dir, lo, hi)
    name = window_name(lo, hi)

    comp = window_completeness(block_dir, lo, hi, state)
    incomplete = dict((s, v) for s, v in comp.items() if v[0] != v[1])
    if incomplete and not force:
        raise ValueError(
            "%s is not complete (%s); tracking would stop at the first hole. "
            "Finish detection or pass --force to materialize anyway."
            % (name, ", ".join("%s %d/%d" % (s, v[0], v[1]) for s, v in sorted(incomplete.items()))))

    if os.path.exists(view) and not force:
        raise ValueError("%s exists; pass --force to reconcile (adds missing links only)" % view)

    made = {"new": 0, "kept": 0}
    if not dry_run:
        os.makedirs(os.path.join(view, "data"), exist_ok=True)
        _group_share(view, group, is_dir=True)
        _group_share(os.path.join(view, "data"), group, is_dir=True)

    # Top-level files (raw videos, sidecars, conductor log): the GUI opens the
    # video from the block dir, and analysis finds sess_*.txt by walking up.
    for entry in sorted(os.listdir(block_dir)):
        src = os.path.join(block_dir, entry)
        if not os.path.isfile(src):
            continue
        _link(os.path.join("..", block_base, entry), os.path.join(view, entry), dry_run, made)

    # data/: only this window's chunks, original names.
    data_dir = os.path.join(block_dir, "data")
    names = data_files_in_window(data_dir, lo, hi)
    for n in names:
        _link(os.path.join("..", "..", block_base, "data", n),
              os.path.join(view, "data", n), dry_run, made)

    # The contract is COPIED: tracking reads chunk_sec/fps from it, and a copy
    # can never let a stray write reach the real block's contract.
    view_state = dict(state)
    view_state["window"] = {"source_block": block_dir, "name": name,
                            "chunk_range": [lo, hi], "materialized": _utcnow()}
    marker = {"schema": SCHEMA, "source_block": block_dir, "name": name,
              "chunk_range": [lo, hi], "materialized": _utcnow(),
              "n_data_links": len(names), "note": "view of a chunk window; "
              "every file under data/ and every video is a relative symlink into the source block"}
    if not dry_run:
        sp = os.path.join(view, "data", ps.STATE_BASENAME)
        with open(sp, "w") as f:
            json.dump(view_state, f, indent=2, sort_keys=True)
            f.write("\n")
        _group_share(sp, group, is_dir=False)
        mp = os.path.join(view, WINDOW_MARKER)
        with open(mp, "w") as f:
            json.dump(marker, f, indent=2)
            f.write("\n")
        _group_share(mp, group, is_dir=False)
    return view, made, len(names)


def cmd_materialize(args):
    block_dir = os.path.abspath(args.block)
    doc = load_windows(block_dir)
    _total, _cs, state = block_chunk_total(block_dir)
    by_name = dict((w["name"], tuple(w["range"])) for w in doc["windows"])
    if args.all:
        targets = list(by_name)
    else:
        if not args.window:
            raise ValueError("pass --window <name> or --all")
        if args.window not in by_name:
            raise ValueError("no window %r in %s (have: %s)"
                             % (args.window, windows_path(block_dir), ", ".join(by_name)))
        targets = [args.window]
    rc = 0
    for name in targets:
        lo, hi = by_name[name]
        try:
            view, made, n = materialize(block_dir, lo, hi, state, args.group,
                                        force=args.force, dry_run=args.dry_run)
        except ValueError as e:
            sys.stderr.write("[SKIP] %s: %s\n" % (name, e))
            rc = 1
            continue
        sys.stderr.write("[OK]%s %s -> %s (%d data files; links new=%d kept=%d)\n"
                         % (" (dry-run)" if args.dry_run else "", name, view, n,
                            made["new"], made["kept"]))
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")   # required= is 3.7+; deigo runs 3.6

    p = sub.add_parser("init", help="declare the windows in <block>/data/WINDOWS.json")
    p.add_argument("--block", required=True)
    p.add_argument("--ranges", required=True, help="comma-separated A-B chunk ranges")
    p.add_argument("--allow-gaps", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--group", default="reiteruni")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("status", help="detection completeness per window")
    p.add_argument("--block", required=True)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("materialize", help="create the sibling view block(s)")
    p.add_argument("--block", required=True)
    p.add_argument("--window")
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--group", default="reiteruni")
    p.set_defaults(fn=cmd_materialize)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.error("a subcommand is required (init|status|materialize)")
    try:
        return args.fn(args)
    except (ValueError, OSError) as e:
        sys.stderr.write("[ERR] %s\n" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
