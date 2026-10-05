#!/usr/bin/env python3
"""Move finished re-run outputs from a rerun view (make_rerun_view.py) into the original block.

  python promote_rerun.py --view <block>-<suffix> [--chunks A-B] [--cap 96] [--yes]

Without --yes it only reports. For every listed (camera, chunk) whose .slp and _sleap_data.h5 are
real files in the view's data/ (links -- the chunks not re-run -- are never moved):
  * the new h5's expected_frames must equal the original's; a mismatch moves nothing for that chunk
    and makes the run exit 1;
  * frames holding >= --cap instances are counted and recorded (re-run those at a higher cap);
  * the original pair moves to <block>/data/_superseded_cap20/, the new pair into <block>/data/.
Chunks without both new files are reported as pending and left alone. Every promotion is appended
to <block>/data/REPROCESS_<suffix>.json. Needs h5py (e.g. /apps/unit/ReiterU/ant_tracking/venv).
"""
import argparse
import datetime as dt
import json
import os
import sys

SUPERSEDED = "_superseded_cap20"


def h5_stats(path):
    """(expected_frames or None, per-frame instance counts) of a _sleap_data.h5."""
    import h5py
    import numpy as np
    with h5py.File(path, "r") as h5:
        exp = h5.attrs.get("expected_frames")
        d = h5["sleap_data"]
        if d.shape[0] == 0:
            return exp, np.zeros(0, dtype=int)
        fr, inst = d["Frame"][:], d["Instance"][:]
    pairs = np.unique(np.stack([fr, inst], 1), axis=0)
    return exp, np.unique(pairs[:, 0], return_counts=True)[1]


def parse_range(text):
    if not text:
        return None
    lo, _, hi = text.partition("-")
    return int(lo), int(hi or lo)


def append_record(path, entries):
    rec = {"promoted": []}
    if os.path.exists(path):
        with open(path) as fh:
            rec = json.load(fh)
    rec["promoted"].extend(entries)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, path)


def check_chunk(view_data, block_data, vname, chunk, cap):
    """(status, detail): 'pending' | 'error' | 'ok' (detail = stats dict for ok)."""
    names = ("%s_%03d.slp" % (vname, chunk), "%s_%03d_sleap_data.h5" % (vname, chunk))
    new = [os.path.join(view_data, n) for n in names]
    if not all(os.path.isfile(p) and not os.path.islink(p) and os.path.getsize(p) > 0 for p in new):
        return "pending", None
    new_exp, per_frame = h5_stats(new[1])
    old_h5 = os.path.join(block_data, names[1])
    old_exp = h5_stats(old_h5)[0] if os.path.exists(old_h5) else None
    if new_exp is None or (old_exp is not None and int(old_exp) != int(new_exp)):
        return "error", "expected_frames new=%s old=%s" % (new_exp, old_exp)
    stats = {"frames_at_cap": int((per_frame >= cap).sum()),
             "max_per_frame": int(per_frame.max()) if per_frame.size else 0,
             "mean_per_frame": round(float(per_frame.mean()), 2) if per_frame.size else 0.0,
             "names": names}
    return "ok", stats


def move_pair(view_data, block_data, names):
    """Old pair -> _superseded_cap20/, new pair -> data/. Returns whether an old pair existed."""
    sup = os.path.join(block_data, SUPERSEDED)
    if not os.path.isdir(sup):
        os.makedirs(sup)
        try:
            os.chmod(sup, 0o2775)
        except OSError:
            pass
    had_old = False
    for n in names:
        old, keep = os.path.join(block_data, n), os.path.join(sup, n)
        if os.path.lexists(old):
            if os.path.lexists(keep):
                raise RuntimeError("%s already holds %s; refusing to overwrite" % (sup, n))
            os.rename(old, keep)
            had_old = True
    for n in names:
        os.rename(os.path.join(view_data, n), os.path.join(block_data, n))
    return had_old


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view")
    ap.add_argument("--chunks", help="only chunks in A-B (inclusive)")
    ap.add_argument("--cap", type=int, default=96, help="the re-run's max_instances")
    ap.add_argument("--yes", action="store_true", help="move files (default: report only)")
    a = ap.parse_args(argv)
    if not a.view:
        ap.error("--view is required")
    view = os.path.abspath(a.view.rstrip("/"))
    with open(os.path.join(view, "VIEW_INFO.json")) as fh:
        info = json.load(fh)
    block_data = os.path.join(info["source_block"], "data")
    view_data = os.path.join(view, "data")
    rng = parse_range(a.chunks)
    record = os.path.join(block_data, "REPROCESS_%s.json" % info.get("suffix", "rerun"))
    now = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    n_ok = n_pending = n_err = 0
    entries = []
    for cam, p in sorted(info["cams"].items()):
        for chunk in p["rerun_chunks"]:
            if rng and not rng[0] <= chunk <= rng[1]:
                continue
            status, detail = check_chunk(view_data, block_data, p["vname"], chunk, a.cap)
            if status == "pending":
                n_pending += 1
                print("pending  %s %03d" % (cam, chunk))
                continue
            if status == "error":
                n_err += 1
                print("ERROR    %s %03d: %s; nothing moved" % (cam, chunk, detail))
                continue
            capped = " -- %d frames at cap %d" % (detail["frames_at_cap"], a.cap) if detail["frames_at_cap"] else ""
            print("%s %s %03d (mean %.1f, max %d)%s" % ("promote " if a.yes else "would   ", cam, chunk,
                  detail["mean_per_frame"], detail["max_per_frame"], capped))
            if not a.yes:
                n_ok += 1
                continue
            had_old = move_pair(view_data, block_data, detail["names"])
            n_ok += 1
            entries.append({"cam": cam, "vname": p["vname"], "chunk": chunk, "promoted_at": now,
                            "view": view, "cap": a.cap, "frames_at_cap": detail["frames_at_cap"],
                            "max_per_frame": detail["max_per_frame"],
                            "mean_per_frame": detail["mean_per_frame"], "superseded": had_old})
    if entries:
        append_record(record, entries)
    n_cap = sum(1 for e in entries if e["frames_at_cap"])
    print("%s %d, pending %d, errors %d%s" % ("promoted" if a.yes else "ready", n_ok, n_pending, n_err,
          ("; %d chunks with frames at cap" % n_cap) if n_cap else ""))
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
