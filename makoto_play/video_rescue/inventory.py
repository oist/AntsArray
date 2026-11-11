#!/usr/bin/env python3
"""Inventory the recorder output on /hpacquire and group it into sessions.

Runs on the cluster's python3 (3.6): stdlib only, no f-strings.

A recorder file is named <camtag>_cam<N>_<YYYY-MM-DD-HH-MM-SS>.mkv with an
optional <name>.mkv.diag.json sibling. All cameras of one recording start
within a few seconds of each other, so files are chained into a session
whenever consecutive start times are closer than --session-gap-sec. Each
session gets a destination <YYYYMMDD>/blockNN in start order (per date),
mirroring the layout already used under /bucket/StephensU/<user>/.

Files below --min-video-bytes are "stubs" (a recorder restart loop leaves
thousands of sub-MB files); they are copied verbatim, never transcoded. A
session that holds only stubs is placed under <YYYYMMDD>/short_HHMMSS.

Writes:
  <out-dir>/manifest.tsv   one row per file:
      session_id  dest_rel  kind(video|stub|diag)  bytes  mtime_epoch  src_path
  <out-dir>/sessions.tsv   one row per session (oldest mtime first)
"""
import argparse
import os
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime

NAME_RE = re.compile(
    r"^(?P<cam>[A-Za-z0-9]+_cam\d+)_(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.mkv$")
SESS_RE = re.compile(r"^sess_(?P<ts>\d{8}_\d{6})\.txt$")   # orchestrator session log
TS_FMT = "%Y-%m-%d-%H-%M-%S"
SESS_TS_FMT = "%Y%m%d_%H%M%S"
SESS_LEAD_SEC = 600   # a session log is opened shortly before its cameras start


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory holding the recorder .mkv files")
    ap.add_argument("--out-dir", required=True, help="where manifest.tsv / sessions.tsv go")
    ap.add_argument("--session-gap-sec", type=int, default=120,
                    help="max gap between consecutive start times inside one session")
    ap.add_argument("--min-video-bytes", type=int, default=10 ** 9,
                    help="smaller .mkv files are stubs: copied verbatim, not transcoded")
    ap.add_argument("--purge-days", type=int, default=21,
                    help="assumed auto-purge age, used only for the risk column")
    ap.add_argument("--dest-rel", default=None,
                    help="destination (YYYYMMDD/blockNN) for the single session found; use when "
                         "--src is an existing block directory whose number auto-numbering would not reproduce")
    return ap.parse_args()


DEST_REL_RE = re.compile(r"^\d{8}/[A-Za-z0-9_-]+$")


def override_dest(sessions, dest_rel):
    """Return sessions with the one session renamed to dest_rel; anything else is a usage error."""
    if not DEST_REL_RE.match(dest_rel):
        sys.exit("[ERR] --dest-rel must look like YYYYMMDD/blockNN, got %r" % dest_rel)
    if len(sessions) != 1:
        sys.exit("[ERR] --dest-rel names one session, but %d were found under --src" % len(sessions))
    sid, _auto_dest, files = sessions[0]
    return [(sid, dest_rel, files)]


def scan(src):
    """Return recorder videos [(name, ts, bytes, mtime)], session logs [(name, ts)], leftovers."""
    videos, sess_logs, others = [], [], []
    for name in sorted(os.listdir(src)):
        path = os.path.join(src, name)
        if not os.path.isfile(path):
            continue
        m = NAME_RE.match(name)
        if m:
            st = os.stat(path)
            videos.append((name, datetime.strptime(m.group("ts"), TS_FMT), st.st_size, st.st_mtime))
            continue
        m = SESS_RE.match(name)
        if m:
            sess_logs.append((name, datetime.strptime(m.group("ts"), SESS_TS_FMT)))
            continue
        if not name.endswith(".diag.json"):
            others.append(name)
    videos.sort(key=lambda v: (v[1], v[0]))
    return videos, sess_logs, others


def attach_session_logs(sessions, sess_logs):
    """Map each sess_*.txt to the session that started within SESS_LEAD_SEC after it."""
    attached, orphans = {}, []
    for name, ts in sess_logs:
        best = None
        for sid, dest_rel, files in sessions:
            lead = (files[0][1] - ts).total_seconds()
            if 0 <= lead <= SESS_LEAD_SEC and (best is None or lead < best[0]):
                best = (lead, sid)
        if best is None:
            orphans.append(name)
        else:
            attached.setdefault(best[1], []).append(name)
    return attached, orphans


def group_sessions(videos, gap_sec):
    """Chain files whose start times are within gap_sec of the previous file."""
    sessions = []
    for v in videos:
        if sessions and (v[1] - sessions[-1][-1][1]).total_seconds() <= gap_sec:
            sessions[-1].append(v)
        else:
            sessions.append([v])
    return sessions


def assign_dest(sessions, min_video_bytes):
    """Name sessions <date>/blockNN (has video) or <date>/short_HHMMSS (stubs only)."""
    per_date = {}
    out = []
    for files in sessions:
        start = files[0][1]
        date = start.strftime("%Y%m%d")
        has_video = any(f[2] >= min_video_bytes for f in files)
        if has_video:
            per_date[date] = per_date.get(date, 0) + 1
            dest_rel = "%s/block%02d" % (date, per_date[date])
        else:
            dest_rel = "%s/short_%s" % (date, start.strftime("%H%M%S"))
        out.append((start.strftime("%Y%m%d_%H%M%S"), dest_rel, files))
    return out


def write_manifest(path, src, sessions, min_video_bytes, sess_by_sid):
    n = 0
    with open(path, "w") as fh:
        for sid, dest_rel, files in sessions:
            for name in sess_by_sid.get(sid, []):
                st = os.stat(os.path.join(src, name))
                fh.write("\t".join([sid, dest_rel, "diag", str(st.st_size), "%d" % st.st_mtime,
                                    os.path.join(src, name)]) + "\n")
                n += 1
            for name, _ts, size, mtime in files:
                kind = "video" if size >= min_video_bytes else "stub"
                fh.write("\t".join([sid, dest_rel, kind, str(size), "%d" % mtime,
                                    os.path.join(src, name)]) + "\n")
                n += 1
                diag = os.path.join(src, name + ".diag.json")
                if os.path.isfile(diag):
                    dst = os.stat(diag)
                    fh.write("\t".join([sid, dest_rel, "diag", str(dst.st_size),
                                        "%d" % dst.st_mtime, diag]) + "\n")
                    n += 1
    return n


def write_sessions(path, sessions, min_video_bytes, purge_days):
    now = time.time()
    rows = []
    for sid, dest_rel, files in sessions:
        n_video = sum(1 for f in files if f[2] >= min_video_bytes)
        n_stub = len(files) - n_video
        total = sum(f[2] for f in files)
        oldest = min(f[3] for f in files)
        age_days = (now - oldest) / 86400.0
        rows.append((oldest, sid, dest_rel, n_video, n_stub, total, age_days))
    rows.sort()
    with open(path, "w") as fh:
        fh.write("session_id\tdest_rel\tn_video\tn_stub\tbytes\toldest_mtime\tage_days\tdays_to_purge\n")
        for oldest, sid, dest_rel, n_video, n_stub, total, age in rows:
            fh.write("%s\t%s\t%d\t%d\t%d\t%s\t%.1f\t%.1f\n" % (
                sid, dest_rel, n_video, n_stub, total,
                datetime.fromtimestamp(oldest).strftime("%Y-%m-%d %H:%M"), age, purge_days - age))
    return rows


def main():
    args = parse_args()
    if not os.path.isdir(args.src):
        sys.exit("[ERR] not a directory: %s" % args.src)
    videos, sess_logs, others = scan(args.src)
    if not videos:
        sys.exit("[ERR] no recorder .mkv files under %s" % args.src)
    sessions = assign_dest(group_sessions(videos, args.session_gap_sec), args.min_video_bytes)
    if args.dest_rel:
        sessions = override_dest(sessions, args.dest_rel)
    sess_by_sid, orphans = attach_session_logs(sessions, sess_logs)
    others.extend(orphans)
    os.makedirs(args.out_dir, exist_ok=True)
    n_rows = write_manifest(os.path.join(args.out_dir, "manifest.tsv"), args.src, sessions,
                            args.min_video_bytes, sess_by_sid)
    rows = write_sessions(os.path.join(args.out_dir, "sessions.tsv"), sessions,
                          args.min_video_bytes, args.purge_days)
    print("%-16s %-24s %6s %6s %9s  %-16s %6s %6s" % (
        "session", "dest", "video", "stub", "TB", "oldest mtime", "age d", "purge"))
    for oldest, sid, dest_rel, n_video, n_stub, total, age in rows:
        print("%-16s %-24s %6d %6d %9.2f  %-16s %6.1f %6.1f" % (
            sid, dest_rel, n_video, n_stub, total / 1e12,
            datetime.fromtimestamp(oldest).strftime("%Y-%m-%d %H:%M"), age, args.purge_days - age))
    total_bytes = sum(r[5] for r in rows)
    print("[OK] %d sessions, %d videos, %d session logs (%d manifest rows), %.2f TB -> %s" % (
        len(rows), len(videos), sum(len(v) for v in sess_by_sid.values()), n_rows,
        total_bytes / 1e12, args.out_dir))
    if others:
        print("[WARN] %d files do not look like recorder output and were skipped: %s" % (
            len(others), ", ".join(others[:5])))


if __name__ == "__main__":
    main()
