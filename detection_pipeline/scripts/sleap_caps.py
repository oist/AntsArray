#!/usr/bin/env python3
"""Per-camera SLEAP instance caps: the large cap for nest cameras, the small one elsewhere.

The per-frame instance cap is baked into the exported TensorRT engine, and inference time
grows with the cap whether or not the slots are used (~9.6 ms + 0.63 ms x cap per frame), so
cap 96 costs ~3x cap 20 on every camera. Only nest cameras come near 20 ants per frame. This
maps every (video, chunk) to a cap from a nest-camera table:

  effective_from<TAB>nest_cams<TAB>note
  2026-07-01 00:00<TAB>cam01,cam02,cam03<TAB>arena layout of July 2026

Times are JST. The latest row at or before a moment applies, so a row is added when nests
are added or moved. A chunk whose wall-clock span [start, start + chunk_sec) touches any
period in which its camera is a nest camera gets the nest cap: a nest added mid-chunk must
not truncate the rest of that chunk. Chunk start = sidecar firstEncodedFrame.hostEpochMs
+ derived.frameOffset / fps (a split tail keeps the original sidecar) + chunk x chunk_sec.

  sleap_caps.py table    --worklist W --manifest M --chunk-sec S (--uniform N | --nest-cams T) --out OUT
  sleap_caps.py describe --manifest M --chunk-sec S (--uniform N | --nest-cams T)

`table` writes vname<TAB>chunk<TAB>cap for each worklist row (the bridge picks an engine per
row from it); `describe` prints the one-line value recorded in the processing contract.
Exit 2 on a malformed table, a cap the engine cannot build, or a chunk no row covers.
"""
import argparse
import csv
import io
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))
CAM_RE = re.compile(r"^cam\d{2}$")
MAX_ENGINE_CROPS = 192   # cap x batch above this cannot build (see export_sleap_trt.sh)
TIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d")
# A chunk's real span can sit a GOP (60 s) off the nominal one (seek landing, off-grid
# keyframes) and host clocks differ by seconds, so each chunk is checked over its span
# widened by this much on both sides: an extra large-cap chunk costs little, a truncated
# nest chunk is lost data.
MARGIN_MS = 120 * 1000


def read_text(path):
    """UTF-8 with or without a BOM (the table may be saved on Windows, notes in Japanese);
    deigo's Python 3.6 would otherwise decode with the locale (ASCII under C)."""
    return io.open(str(path), encoding="utf-8-sig")


def parse_when(text):
    """JST 'YYYY-MM-DD[ HH:MM]' -> epoch ms."""
    for fmt in TIME_FORMATS:
        try:
            t = datetime.strptime(text.strip(), fmt).replace(tzinfo=JST)
        except ValueError:
            continue
        return int(round(t.timestamp() * 1000))
    raise ValueError("bad time %r (want YYYY-MM-DD[ HH:MM], JST)" % text)


def format_when(ms):
    return datetime.fromtimestamp(ms / 1000.0, JST).strftime("%Y-%m-%d %H:%M")


def load_table(path):
    """Rows (effective_ms, frozenset of cameras, note), oldest first."""
    rows = []
    with read_text(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\r\n")   # the table may be edited on Windows (Z:)
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split("\t")
            if fields[0].strip().lower() == "effective_from":
                continue
            if len(fields) < 2:
                raise ValueError("%s:%d: want effective_from<TAB>nest_cams[<TAB>note]" % (path, n))
            try:
                when = parse_when(fields[0])
            except ValueError as e:
                raise ValueError("%s:%d: %s" % (path, n, e))
            cams = [c.strip() for c in fields[1].split(",") if c.strip()]
            for c in cams:
                if not CAM_RE.match(c):
                    raise ValueError("%s:%d: bad camera %r (want camNN)" % (path, n, c))
            rows.append((when, frozenset(cams), fields[2].strip() if len(fields) > 2 else ""))
    if not rows:
        raise ValueError("%s: no rows" % path)
    rows.sort(key=lambda r: r[0])
    return rows


def rows_during(table, t0, t1):
    """The row in force at t0 plus every row that starts inside (t0, t1)."""
    at_start = [r for r in table if r[0] <= t0]
    if not at_start:
        raise ValueError("no nest-camera row covers %s (first row starts %s); add one or pass --uniform"
                         % (format_when(t0), format_when(table[0][0])))
    return [at_start[-1]] + [r for r in table if t0 < r[0] < t1]


def cams_during(table, t0, t1):
    """Every camera that is a nest camera at some moment of [t0 - margin, t1 + margin).
    The table must cover t0 itself; the margin only ever adds cameras."""
    rows = rows_during(table, t0, t1 + MARGIN_MS)
    before = [r for r in table if r[0] <= t0 - MARGIN_MS]
    if before:
        rows.append(before[-1])
    return frozenset().union(*(cams for _, cams, _ in rows))


def read_manifest(path):
    with read_text(path) as fh:
        return {r["vname"]: r for r in csv.DictReader(fh)}


def video_start_ms(row):
    side = row["source_path"] + ".diag.json"
    with read_text(side) as fh:
        j = json.load(fh)
    t0 = (j.get("provenance") or {}).get("firstEncodedFrame", {}).get("hostEpochMs")
    if t0 is None:
        raise ValueError("%s has no provenance.firstEncodedFrame.hostEpochMs" % side)
    off = int((j.get("derived") or {}).get("frameOffset") or 0)
    return t0 + off * 1000.0 / float(row["fps"])


def camera_of(vname):
    """camNN of a grid video name. Anything else is refused: its chunks would never match a
    nest row and would silently get the small cap."""
    cam = vname.split("_", 1)[0]
    if not CAM_RE.match(cam):
        raise ValueError("%s: video name does not start with camNN_, so its camera is unknown "
                         "(pass --sleap-max-instances / --uniform for one cap)" % vname)
    return cam


def read_worklist(path):
    with read_text(path) as fh:
        return [line.rstrip("\r\n").split("\t")[:2] for line in fh if line.strip()]


def chunk_caps(worklist, manifest, chunk_sec, table, nest_cap, other_cap):
    """[(vname, chunk, cap)] for every worklist row; table None = every chunk gets nest_cap."""
    rows = read_worklist(worklist)
    if table is None:
        return [(v, c, nest_cap) for v, c in rows]
    videos = read_manifest(manifest)
    starts = {}
    out = []
    for vname, chunk in rows:
        if vname not in starts:
            if vname not in videos:
                raise ValueError("%s is in the worklist but not in %s" % (vname, manifest))
            starts[vname] = video_start_ms(videos[vname])
        t0 = starts[vname] + int(chunk) * chunk_sec * 1000
        nest = camera_of(vname) in cams_during(table, t0, t0 + chunk_sec * 1000)
        out.append((vname, chunk, nest_cap if nest else other_cap))
    return out


def describe(manifest, chunk_sec, table, nest_cap, other_cap):
    """Contract value: every table row in force during the block, then the other cap."""
    if table is None:
        return str(nest_cap)
    spans = []
    for row in read_manifest(manifest).values():
        t0 = video_start_ms(row)
        spans.append((t0, t0 + int(row["n_chunks"]) * chunk_sec * 1000))
    rows = rows_during(table, min(s for s, _ in spans), max(e for _, e in spans))
    parts = ["nest%d=%s@%s" % (nest_cap, ",".join(sorted(cams)), format_when(when).replace(" ", "T"))
             for when, cams, _ in rows]
    return ";".join(parts + ["other%d" % other_cap])


def check_cap(name, cap):
    if not 1 <= cap <= MAX_ENGINE_CROPS:
        raise ValueError("%s %d: must be 1..%d (cap x batch above %d crops cannot build)"
                         % (name, cap, MAX_ENGINE_CROPS, MAX_ENGINE_CROPS))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["table", "describe"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--chunk-sec", type=int, required=True)
    ap.add_argument("--worklist", help="table: the SLEAP worklist (vname<TAB>chunk<TAB>frames)")
    ap.add_argument("--out", help="table: output vname<TAB>chunk<TAB>cap")
    how = ap.add_mutually_exclusive_group(required=True)
    how.add_argument("--uniform", type=int, help="one cap for every camera")
    how.add_argument("--nest-cams", help="nest-camera table (effective_from<TAB>nest_cams<TAB>note)")
    ap.add_argument("--nest-cap", type=int, default=96)
    ap.add_argument("--other-cap", type=int, default=20)
    a = ap.parse_args(argv)
    try:
        if a.uniform is not None:
            check_cap("--uniform", a.uniform)
            table, nest_cap, other_cap = None, a.uniform, a.uniform
        else:
            check_cap("--nest-cap", a.nest_cap)
            check_cap("--other-cap", a.other_cap)
            if a.nest_cap < a.other_cap:
                raise ValueError("--nest-cap %d is below --other-cap %d" % (a.nest_cap, a.other_cap))
            table, nest_cap, other_cap = load_table(a.nest_cams), a.nest_cap, a.other_cap
        if a.mode == "describe":
            print(describe(a.manifest, a.chunk_sec, table, nest_cap, other_cap))
            return 0
        if not (a.worklist and a.out):
            ap.error("table needs --worklist and --out")
        rows = chunk_caps(a.worklist, a.manifest, a.chunk_sec, table, nest_cap, other_cap)
    except (OSError, ValueError, KeyError) as e:
        print("[ERR] sleap_caps: %s" % e, file=sys.stderr)
        return 2
    tmp = a.out + ".tmp"
    with open(tmp, "w", newline="\n") as fh:   # read by bash (cut, awk): never CRLF
        fh.writelines("%s\t%s\t%d\n" % r for r in rows)
    os.replace(tmp, a.out)
    for cap in sorted({r[2] for r in rows}, reverse=True):
        cams = sorted({v.split("_", 1)[0] for v, _, c in rows if c == cap})
        print("cap %d: %d chunks (%s)" % (cap, sum(1 for r in rows if r[2] == cap), ",".join(cams)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
