#!/usr/bin/env python3
"""Re-cut one camera's per-chunk detection outputs onto the block's chunk grid.

Why: chunks are stream-copied, so they can only be cut on keyframes. Recorders
before PylonRecorder v0.78 let NVENC insert a scene-cut IDR, which restarts the
GOP and moves every later keyframe of that ONE camera off the 1440-frame grid
(20260928/block01 cam19: an extra IDR at frame 457). Its chunk 000 then runs
past the boundary to the next keyframe and every later chunk starts late::

    other cams   chunk i = true frames [i*P, (i+1)*P)
    this camera  chunk 0 = [0, P+off),  chunk c>=1 = [c*P+off, (c+1)*P+off)

Tracking places a chunk's frames at chunk_idx * P + local frame for every camera
alike (stitch_tracks.py), so this camera would sit `off` frames out of sync with
the rest in every chunk >= 1. Nothing downstream can take a per-camera offset,
so the fix is in the files: move the records that crossed a boundary into the
neighbouring chunk, renumber the local frames, and write chunks that each cover
exactly the true frames of their index.

The source outputs are never modified. The result is a sibling view block, the
same shape make_window_block.py builds (relative symlinks for every other
camera and for the raw videos, a copied contract), in which only this camera's
_aruco_tracks.h5 / _aruco_detections.h5 / _sleap_data.h5 are real, re-cut
files. Its .slp files are left out: they are on the shifted grid, and tracking
never reads them. REALIGN.json records the piece map and the record counts.

Where each true frame comes from is derived from the source chunks themselves
(cumulative ArUco num_frames), then checked against --offset and the contract's
frame_count; a SLEAP chunk whose inference was capped short of its chunk
(expected_frames attr) is refused, because its missing frames cannot be
re-cut into existence.

Run on deigo with the tracking venv (needs h5py, pandas, tables)::

    /apps/unit/ReiterU/ant_tracking/venv/bin/python tracking/colony/realign_offgrid_camera.py \\
        --block /bucket/ReiterU/Ants/basler/20260928/block01 \\
        --video cam19_cam1_2026-09-28-13-57-17 --offset 457 [--dry-run]

Frame drop (--sidecar-gaps). A camera that lost trigger pulses on the way to the
PC (diag.json provenance.gaps, e.g. 20261113/block01 cam03 + cam06 lost pulses
1,682,048-1,682,049) has fewer file frames than the others, and every file frame
from the gap on belongs to a later pulse. Its chunks were cut on its own file
grid, so its chunk c >= the drop chunk holds the wrong pulses. The re-cut puts
every record on its pulse, leaves the lost pulses as empty frames (no records,
zero dense rows, listed in the hole_frames attr), and moves the spill-over into
the next chunk, so every chunk again covers exactly [c*P, (c+1)*P) and the
camera has as many frames as the others. This is done IN PLACE, in two steps:

    # 1. compute node: re-cut only the chunks that change, into a staging dir
    python realign_offgrid_camera.py --block <block> --video <video> [<video> ...] \\
        --sidecar-gaps --stage /flash/.../realign_stage
    # 2. a host that can write the bucket (deigo login): verify and install
    python realign_offgrid_camera.py --block <block> --install /flash/.../realign_stage

Install moves the replaced originals to data/_pre_realign/ (never deletes them),
puts the re-cut files in data/, and writes <block>/REALIGN.json: the sidecar gap
records verbatim, the unrecorded pulses, the video-frame map, and the sha256 of
every source and re-cut file. Each re-cut file carries the record's id in its
realign_id attr, so a later re-detection that overwrites one is detectable.
Install is resumable and idempotent; it refuses if a source file changed after
staging. The .slp files and the raw videos are not touched: they keep their file
frame numbering, which the record's video_frame_map translates.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import uuid

import h5py
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "detection_pipeline", "lib"))
sys.path.insert(0, os.path.join(_REPO, "detection_pipeline", "scripts"))
import make_window_block as mwb  # noqa: E402
import pipeline_state as ps  # noqa: E402
from aruco_output import save_aruco_outputs  # noqa: E402

MARKER = "REALIGN.json"
SCHEMA = 1
TRK, DET, SDAT = "_aruco_tracks.h5", "_aruco_detections.h5", "_sleap_data.h5"
KINDS = (TRK, DET, SDAT)
ORIGINALS = "_pre_realign"
IN_PLACE_NOTE = ("Only the _aruco_tracks/_aruco_detections/_sleap_data exports of the listed chunks "
                 "were re-cut; the replaced files are in data/%s. The files tracking reads "
                 "(_aruco_tracks.h5, _sleap_data.h5) carry realign_id; _aruco_detections.h5 is covered "
                 "by its sha256 here only. SLEAP Instance is renumbered as a running index per file. "
                 "The .slp files and the raw videos keep their own (file) frame numbering: use "
                 "video_frame_map." % ORIGINALS)


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Log:
    """stderr plus a timestamped log file (opened once the view dir exists)."""

    def __init__(self):
        self.fh = None

    def open(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.fh = open(os.path.join(log_dir, "realign_%s.log" % stamp), "a")

    def __call__(self, msg):
        line = "[%s] %s\n" % (datetime.datetime.now().strftime("%F %T"), msg)
        sys.stderr.write(line)
        if self.fh:
            self.fh.write(line)
            self.fh.flush()


# ---------------------------------------------------------------------------
# frame geometry
# ---------------------------------------------------------------------------
def source_layout(lengths, per_chunk, offset, total):
    """True start of every source chunk, checked against the expected off-grid layout.

    lengths: frames in each source chunk, in index order. The camera's chunk c >= 1
    must start at c*per_chunk + offset and the chunks must tile [0, total).
    """
    starts = [0]
    for n in lengths[:-1]:
        starts.append(starts[-1] + n)
    bad = [(c, s, c * per_chunk + offset) for c, s in enumerate(starts)
           if c > 0 and s != c * per_chunk + offset]
    if bad:
        c, s, want = bad[0]
        raise ValueError("source chunk %03d starts at true frame %d, not %d (offset %d): "
                         "the chunks do not follow a constant off-grid shift"
                         % (c, s, want, offset))
    if starts[-1] + lengths[-1] != total:
        raise ValueError("source chunks cover %d frames, the contract declares %d"
                         % (starts[-1] + lengths[-1], total))
    return starts


def source_runs(starts, lengths, gaps=()):
    """[(src_chunk, src_lo, src_hi, true_lo)]: stretches contiguous in the file AND in true frames.

    gaps = [(file_frame, missing)] from the sidecar: file_frame is the first frame recorded
    after `missing` trigger pulses were lost, so file frame f is true frame f plus every
    pulse lost at or before it. Without gaps there is one run per source chunk.
    """
    end = starts[-1] + lengths[-1]
    gaps = sorted((int(g), int(m)) for g, m in gaps)
    for g, m in gaps:
        if not 0 < g < end or m < 1:
            raise ValueError("a gap of %d frame(s) at file frame %d is outside this camera's %d frames"
                             % (m, g, end))
    runs = []
    for c, (s, n) in enumerate(zip(starts, lengths)):
        cuts = sorted(set([s, s + n] + [g for g, _m in gaps if s < g < s + n]))
        for a, b in zip(cuts[:-1], cuts[1:]):
            runs.append((c, a - s, b - s, a + sum(m for g, m in gaps if g <= a)))
    return runs


def hole_spans(gaps):
    """True-frame spans [lo, hi) that no recorded frame covers (the lost pulses)."""
    out, lost = [], 0
    for g, m in sorted((int(g), int(m)) for g, m in gaps):
        out.append((g + lost, g + lost + m))
        lost += m
    return out


def chunk_holes(i, per_chunk, total, holes):
    """The holes inside true chunk i, as local [lo, hi) frame spans."""
    lo, hi = i * per_chunk, min((i + 1) * per_chunk, total)
    return [[max(lo, a) - lo, min(hi, b) - lo] for a, b in holes if max(lo, a) < min(hi, b)]


def pieces_for(i, runs, per_chunk, total, holes=()):
    """[(src_chunk, src_lo, src_hi, dst_lo)] that fill true chunk i, in frame order.

    Whatever the pieces leave uncovered must be exactly the declared holes.
    """
    lo, hi = i * per_chunk, min((i + 1) * per_chunk, total)
    out = []
    for c, s_lo, s_hi, t_lo in runs:
        a, b = max(lo, t_lo), min(hi, t_lo + s_hi - s_lo)
        if a < b:
            out.append((c, s_lo + a - t_lo, s_lo + b - t_lo, a - lo))
    out.sort(key=lambda p: p[3])
    if any(a[3] + a[2] - a[1] > b[3] for a, b in zip(out, out[1:])):
        raise ValueError("true chunk %03d: source runs overlap" % i)
    empty = sum(b - a for a, b in chunk_holes(i, per_chunk, total, holes))
    if sum(p[2] - p[1] for p in out) + empty != hi - lo:
        raise ValueError("true chunk %03d is not fully covered by the source chunks" % i)
    return out


def sidecar_gaps(block_dir, vname):
    """(gaps [(file_frame, missing)], the sidecar's gap records verbatim, frames in the file).

    Reads <video>.mkv.diag.json (schemaVersion 2: a Basler blockId per frame) and refuses
    anything a pure loss of trigger pulses cannot explain: a camera that did not start at
    blockId 0, resets / wraps / invalid IDs, an overflowed event log, or gap records whose
    blockIds disagree with the file frame they are attached to.
    """
    path = os.path.join(block_dir, vname + ".mkv.diag.json")
    with open(path) as f:
        doc = json.load(f)
    prov, frames = doc["provenance"], int(doc["recorder"]["framesEncoded"])
    raw = list(prov.get("gaps") or [])
    first, last = prov["firstEncodedFrame"], prov["lastEncodedFrame"]
    if (first["blockId"] != 0 or first["fileFrameIndex"] != 0
            or last["fileFrameIndex"] != frames - 1
            or any(prov.get(k) for k in ("resetEvents", "invalidIdEvents", "wrapEvents"))
            or prov.get("eventOverflowCount", 0)):
        raise ValueError("%s does not describe one clean trigger run" % path)
    gaps, lost = [], 0
    for g in sorted(raw, key=lambda g: int(g["fileFrameIndex"])):
        f, m = int(g["fileFrameIndex"]), int(g["blocksMissing"])
        lost += m
        if (g.get("discontinuity") != "forward_gap" or m < 1 or int(g["blockId"]) != f + lost
                or int(g["previousBlockId"]) != f + lost - m - 1):
            raise ValueError("%s: gap record %s is not a pure loss of trigger pulses" % (path, g))
        gaps.append((f, m))
    if int(last["blockId"]) != frames - 1 + lost:
        raise ValueError("%s: last blockId is %s, but %d frames + %d lost pulses end at %d"
                         % (path, last["blockId"], frames, lost, frames - 1 + lost))
    return gaps, raw, frames


def video_frame_map(gaps):
    """[[first_pulse, lost]]: from first_pulse on, video/file frame = pulse - lost."""
    out, lost = [[0, 0]], 0
    for g, m in sorted((int(g), int(m)) for g, m in gaps):
        lost += m
        out.append([g + lost, lost])
    return out


# ---------------------------------------------------------------------------
# reading sources
# ---------------------------------------------------------------------------
def _name(vname, idx, suffix):
    return "%s_%03d%s" % (vname, idx, suffix)


def read_aruco(path):
    with h5py.File(path, "r") as h:
        if int(h.attrs.get("aruco_detection_schema_version", 0)) < 2:
            raise ValueError("%s is not a schema-2 lossless ArUco file" % path)
        n = int(h.attrs["num_frames"])
        rec = h["aruco_detections"][:]
        tracks = h["aruco_tracks"][:]
        conf = h["aruco_confidences"][:]
    if tracks.shape[0] != n or conf.shape[0] != n:
        raise ValueError("%s: dense arrays hold %d frames, num_frames says %d"
                         % (path, tracks.shape[0], n))
    return {"n": n, "rec": rec, "tracks": tracks, "conf": conf}


def read_sleap(path, src_len):
    with h5py.File(path, "r") as h:
        rec = h["sleap_data"][:]
        attrs = dict((k, h.attrs[k]) for k in h.attrs)
    # expected_frames is the cap SLEAP ran with, not the chunk's length: the worklist gives the
    # last chunk the true-grid residual, which is LONGER than this camera's shifted last chunk.
    # Only a cap below the chunk's length means frames were never inferred.
    cap = attrs.get("expected_frames")
    if cap is not None and int(cap) < src_len:
        raise ValueError("%s: SLEAP ran over %d frames but the chunk holds %d; re-run SLEAP "
                         "on it with --frames 0-%d first" % (path, int(cap), src_len, src_len - 1))
    if len(rec) and int(rec["Frame"].max()) >= src_len:
        raise ValueError("%s has frames past its chunk length %d" % (path, src_len))
    return {"rec": rec, "attrs": attrs}


def source_lengths(data_dir, vname, n_chunks):
    lengths = []
    for c in range(n_chunks):
        p = os.path.join(data_dir, _name(vname, c, TRK))
        if not os.path.isfile(p):
            raise ValueError("missing %s" % p)
        with h5py.File(p, "r") as h:
            lengths.append(int(h.attrs["num_frames"]))
    return lengths


class SourceCache:
    """Hold at most the few source chunks the current true chunk needs."""

    def __init__(self, data_dir, vname, lengths):
        self.data_dir, self.vname, self.lengths = data_dir, vname, lengths
        self.aruco, self.sleap = {}, {}

    def get(self, c):
        if c not in self.aruco:
            self.aruco[c] = read_aruco(os.path.join(self.data_dir, _name(self.vname, c, TRK)))
            self.sleap[c] = read_sleap(os.path.join(self.data_dir, _name(self.vname, c, SDAT)),
                                       self.lengths[c])
        return self.aruco[c], self.sleap[c]

    def drop_below(self, c):
        for k in [k for k in self.aruco if k < c]:
            del self.aruco[k], self.sleap[k]


# ---------------------------------------------------------------------------
# re-cutting one true chunk
# ---------------------------------------------------------------------------
def _slice_records(rec, lo, hi, shift):
    keep = rec[(rec["Frame"] >= lo) & (rec["Frame"] < hi)].copy()
    keep["Frame"] = (keep["Frame"] + shift).astype(rec.dtype["Frame"])
    return keep


def build_chunk(pieces, cache, dst_len):
    """Fill one true chunk from its pieces: aruco dense + records, sleap records.

    Frames no piece covers (lost pulses) stay empty: zero dense rows, no records.
    """
    trk = conf = None
    a_rec, s_rec, sources = [], [], []
    for c, lo, hi, dst_lo in pieces:
        aru, slp = cache.get(c)
        if trk is None:
            trk = np.zeros((dst_len,) + aru["tracks"].shape[1:], dtype=aru["tracks"].dtype)
            conf = np.zeros((dst_len,) + aru["conf"].shape[1:], dtype=aru["conf"].dtype)
        elif aru["tracks"].shape[1:] != trk.shape[1:] or aru["conf"].shape[1:] != conf.shape[1:]:
            raise ValueError("source chunk %03d has a different ArUco dictionary size" % c)
        shift = dst_lo - lo
        trk[dst_lo:dst_lo + hi - lo] = aru["tracks"][lo:hi]
        conf[dst_lo:dst_lo + hi - lo] = aru["conf"][lo:hi]
        a_rec.append(_slice_records(aru["rec"], lo, hi, shift))
        s_rec.append(_slice_records(slp["rec"], lo, hi, shift))
        sources.append({"src_chunk": c, "src_frames": [lo, hi], "dst_first": dst_lo,
                        "sleap_attrs": slp["attrs"]})
    return {"tracks": trk, "conf": conf,
            "aruco": np.concatenate(a_rec), "sleap": renumber_instances(np.concatenate(s_rec)),
            "sources": sources}


def renumber_instances(rec):
    """SLEAP Instance as sleap2h5.py writes it: a running index over the file, frame by frame.

    Pieces from two source chunks would otherwise repeat Instance values within one file.
    (ArUco Instance is the tag ID and is never renumbered.)
    """
    if len(rec):
        key = rec["Frame"].astype(np.int64) * (int(rec["Instance"].max()) + 1) + rec["Instance"]
        _, rank = np.unique(key, return_inverse=True)
        rec["Instance"] = rank.astype(rec.dtype["Instance"])
    return rec


def _realigned_from(chunk):
    return json.dumps([{k: s[k] for k in ("src_chunk", "src_frames", "dst_first")}
                       for s in chunk["sources"]])


def tag_aruco(path, chunk, holes, realign_id=None):
    """Provenance attrs on a re-cut _aruco_tracks.h5 (save_aruco_outputs knows none of it)."""
    with h5py.File(path, "r+") as h:
        h.attrs["realigned_from"] = _realigned_from(chunk)
        h.attrs["hole_frames"] = json.dumps(holes)
        if realign_id:
            h.attrs["realign_id"] = realign_id


def _sleap_frame_count(chunk, dst_len, src_lengths):
    """frame_count keeps its source meaning: every frame if the sources counted every frame."""
    full = all(int(s["sleap_attrs"].get("frame_count", -1)) == src_lengths[s["src_chunk"]]
               for s in chunk["sources"])
    return dst_len if full else int(len(np.unique(chunk["sleap"]["Frame"])))


def write_sleap(path, chunk, dst_len, src_lengths, src_names, holes=(), realign_id=None):
    """frame_count stays the chunk's length when the sources covered every frame: hole frames
    count as frames without predictions; hole_frames names them."""
    attrs = [s["sleap_attrs"] for s in chunk["sources"]]
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as h:
        h.attrs["source_file"] = "; ".join(src_names)
        h.attrs["frame_count"] = _sleap_frame_count(chunk, dst_len, src_lengths)
        h.attrs["expected_frames"] = dst_len
        h.attrs["instance_count"] = max(int(a.get("instance_count", 0)) for a in attrs)
        h.attrs["node_count"] = int(attrs[0].get("node_count", 0))
        h.attrs["realigned_from"] = _realigned_from(chunk)
        h.attrs["hole_frames"] = json.dumps(list(holes))
        if realign_id:
            h.attrs["realign_id"] = realign_id
        if len(chunk["sleap"]) == 0:
            h.create_dataset("sleap_data", data=chunk["sleap"])
        else:
            h.create_dataset("sleap_data", data=chunk["sleap"], compression="gzip",
                             shuffle=True, chunks=True)
    os.replace(tmp, path)


def record_counts(data_dir, vname, c):
    with h5py.File(os.path.join(data_dir, _name(vname, c, TRK)), "r") as h:
        a = int(h["aruco_detections"].shape[0])
    with h5py.File(os.path.join(data_dir, _name(vname, c, SDAT)), "r") as h:
        s = int(h["sleap_data"].shape[0])
    return a, s


def realign_video(data_dir, out_dir, vname, n_chunks, per_chunk, offset, total, log, dry_run,
                  gaps=(), only_changed=False, realign_id=None):
    """Re-cut one camera onto the true grid. total = true frames (the clean cameras' count);
    gaps = the sidecar's lost pulses. only_changed writes (and reads) only the chunks whose
    content moves; the others are counted for the conservation check but left alone."""
    lengths = source_lengths(data_dir, vname, n_chunks)
    missing = sum(int(m) for _g, m in gaps)
    starts = source_layout(lengths, per_chunk, offset, total - missing)
    runs = source_runs(starts, lengths, gaps)
    holes = hole_spans(gaps)
    cache = SourceCache(data_dir, vname, lengths)
    rows = []
    for i in range(n_chunks):
        pieces = pieces_for(i, runs, per_chunk, total, holes)
        if not pieces:
            raise ValueError("true chunk %03d has no recorded frame" % i)
        dst_len = min(per_chunk, total - i * per_chunk)
        local_holes = chunk_holes(i, per_chunk, total, holes)
        changed = bool(local_holes) or pieces != [(i, 0, lengths[i], 0)] or dst_len != lengths[i]
        row = {"chunk": i, "frames": dst_len, "pieces": [list(p) for p in pieces],
               "holes": local_holes, "changed": changed}
        rows.append(row)
        if only_changed and not changed:
            row["aruco_records"], row["sleap_records"] = record_counts(data_dir, vname, i)
            continue
        cache.drop_below(pieces[0][0])
        chunk = build_chunk(pieces, cache, dst_len)
        if not dry_run:
            name = "%s_%03d" % (vname, i)
            det = pd.DataFrame.from_records(chunk["aruco"])
            save_aruco_outputs(out_dir, name, chunk["tracks"], chunk["conf"], det, "h5")
            tag_aruco(os.path.join(out_dir, name + TRK), chunk, local_holes, realign_id)
            write_sleap(os.path.join(out_dir, name + SDAT), chunk, dst_len, lengths,
                        [_name(vname, p[0], SDAT) for p in pieces], local_holes, realign_id)
        row["aruco_records"] = int(len(chunk["aruco"]))
        row["sleap_records"] = int(len(chunk["sleap"]))
        log("chunk %03d: %s -> %d frames%s, aruco %d, sleap %d"
            % (i, " + ".join("src%03d[%d:%d]" % p[:3] for p in pieces), dst_len,
               "".join(" (empty %d-%d)" % (a, b - 1) for a, b in local_holes),
               row["aruco_records"], row["sleap_records"]))
    return lengths, rows


def check_conservation(data_dir, vname, lengths, rows):
    """Every source record lies inside [0, total), so the re-cut must keep them all."""
    a_in = s_in = 0
    for c in range(len(lengths)):
        a, s = record_counts(data_dir, vname, c)
        a_in += a
        s_in += s
    a_out = sum(r["aruco_records"] for r in rows)
    s_out = sum(r["sleap_records"] for r in rows)
    if (a_in, s_in) != (a_out, s_out):
        raise ValueError("record count changed: aruco %d -> %d, sleap %d -> %d"
                         % (a_in, a_out, s_in, s_out))
    return {"aruco_records": a_in, "sleap_records": s_in}


# ---------------------------------------------------------------------------
# the view block
# ---------------------------------------------------------------------------
def link_view(block_dir, view, vname, group, dry_run):
    """Relative links for the raw files and for every data/ file of the OTHER videos."""
    base = os.path.basename(block_dir)
    made = {"new": 0, "kept": 0}
    for entry in sorted(os.listdir(block_dir)):
        if os.path.isfile(os.path.join(block_dir, entry)):
            mwb._link(os.path.join("..", base, entry), os.path.join(view, entry), dry_run, made)
    n_other = n_skipped = 0
    for name in sorted(os.listdir(os.path.join(block_dir, "data"))):
        if name.startswith(mwb._CONTROL_FILES):
            continue
        m = mwb._DATA_FILE_RE.match(name)
        if m is None:
            continue
        if m.group("v") == vname:
            n_skipped += 1
            continue
        mwb._link(os.path.join("..", "..", base, "data", name),
                  os.path.join(view, "data", name), dry_run, made)
        n_other += 1
    return made, n_other, n_skipped


def write_control(view, state, marker, group):
    sp = os.path.join(view, "data", ps.STATE_BASENAME)
    with open(sp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    mp = os.path.join(view, MARKER)
    with open(mp, "w") as f:
        json.dump(marker, f, indent=2)
        f.write("\n")
    for p in (sp, mp):
        mwb._group_share(p, group, is_dir=False)


def contract_geometry(state, vname):
    videos = state.get("chunking", {}).get("videos", {})
    if vname not in videos:
        raise ValueError("%s is not in the block's contract" % vname)
    v = videos[vname]
    per_chunk = int(round(float(v["fps"]) * int(state["chunking"]["chunk_sec"])))
    return int(v["n_chunks"]), per_chunk, int(v["frame_count"])


def require_complete(block_dir, total_chunks, state, allow_incomplete):
    comp = mwb.window_completeness(block_dir, 0, total_chunks - 1, state)
    holes = dict((s, v) for s, v in comp.items() if v[0] != v[1])
    if holes and not allow_incomplete:
        raise ValueError("detection is not complete (%s); tracking would stop at the first hole"
                         % ", ".join("%s %d/%d" % (s, v[0], v[1]) for s, v in sorted(holes.items())))


def write_json(path, doc, group):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    mwb._group_share(path, group, is_dir=False)


def run(args, log):
    block_dir = os.path.abspath(args.block)
    total_chunks, _cs, state = mwb.block_chunk_total(block_dir)
    n_chunks, per_chunk, total = contract_geometry(state, args.video)
    require_complete(block_dir, total_chunks, state, args.allow_incomplete)
    view = args.view or os.path.join(os.path.dirname(block_dir),
                                     os.path.basename(block_dir) + "-sync")
    if os.path.exists(view) and not args.force:
        raise ValueError("%s exists; pass --force to rebuild its re-cut files" % view)
    out_dir = os.path.join(view, "data")
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
        for d in (view, out_dir):
            mwb._group_share(d, args.group, is_dir=True)
        log.open(os.path.join(view, "realign_logs"))
    log("block %s, %s: %d chunks of %d frames, total %d, offset %d -> %s%s"
        % (block_dir, args.video, n_chunks, per_chunk, total, args.offset, view,
           " (dry-run)" if args.dry_run else ""))
    data_dir = os.path.join(block_dir, "data")
    lengths, rows = realign_video(data_dir, out_dir, args.video, n_chunks, per_chunk,
                                  args.offset, total, log, args.dry_run)
    counts = check_conservation(data_dir, args.video, lengths, rows)
    log("conservation OK: %d aruco, %d sleap records" % (counts["aruco_records"], counts["sleap_records"]))
    made, n_other, n_skipped = link_view(block_dir, view, args.video, args.group, args.dry_run)
    log("links new=%d kept=%d (%d data files of other videos; %d source files of %s not linked)"
        % (made["new"], made["kept"], n_other, n_skipped, args.video))
    realign = {"video": args.video, "offset": args.offset, "per_chunk": per_chunk,
               "source_block": block_dir, "created": _utcnow()}
    view_state = dict(state)
    view_state["realign"] = realign
    marker = dict(realign, schema=SCHEMA, source_lengths=lengths, chunks=rows, conserved=counts,
                  note="only %s's _aruco_tracks/_aruco_detections/_sleap_data files are real; "
                       "everything else is a relative symlink into the source block" % args.video)
    if not args.dry_run:
        write_control(view, view_state, marker, args.group)
    log("done: %s" % view)
    return 0


# ---------------------------------------------------------------------------
# frame drop: stage, then install in place
# ---------------------------------------------------------------------------
def plan_drop(block_dir, state, vname, all_videos):
    """Geometry of one dropping camera, checked against its sidecar and the clean cameras."""
    n_chunks, per_chunk, file_total = contract_geometry(state, vname)
    gaps, raw, frames = sidecar_gaps(block_dir, vname)
    if not gaps:
        raise ValueError("%s: its sidecar records no lost pulse; nothing to realign" % vname)
    if frames != file_total:
        raise ValueError("%s: the sidecar has %d frames, the contract %d" % (vname, frames, file_total))
    total = file_total + sum(m for _g, m in gaps)
    if -(-total // per_chunk) != n_chunks:
        raise ValueError("%s: %d frames need %d chunks but it has %d; a re-cut that adds a chunk "
                         "is not supported" % (vname, total, -(-total // per_chunk), n_chunks))
    others = set(int(v["frame_count"]) for name, v in state["chunking"]["videos"].items()
                 if name not in all_videos)
    if others != {total}:
        raise ValueError("%s: %d recorded + %d lost = %d frames, but the other cameras have %s"
                         % (vname, file_total, total - file_total, total, sorted(others)))
    return {"video": vname, "n_chunks": n_chunks, "per_chunk": per_chunk, "file_total": file_total,
            "gaps": gaps, "raw": raw, "total": total}


def stage_entry(p, block_dir, data_dir, out_data, lengths, rows, counts):
    """The record of one camera: sidecar evidence, holes, video map and every file's sha256."""
    chunks = []
    for r in rows:
        if r["changed"]:
            files = {}
            for s in KINDS:
                n = _name(p["video"], r["chunk"], s)
                files[s] = {"name": n, "source_sha256": sha256_file(os.path.join(data_dir, n)),
                            "sha256": sha256_file(os.path.join(out_data, n))}
            chunks.append(dict(r, files=files))
    sidecar = p["video"] + ".mkv.diag.json"
    return {"camera": p["video"].split("_")[0], "sidecar": sidecar,
            "sidecar_sha256": sha256_file(os.path.join(block_dir, sidecar)),
            "sidecar_gaps": p["raw"], "recorded_frames": p["file_total"],
            "unrecorded_frames": [list(h) for h in hole_spans(p["gaps"])],
            "video_frame_map": {"entries": video_frame_map(p["gaps"]),
                                "rule": "video (and .slp) frame = pulse - lost, from the last entry "
                                        "[first_pulse, lost] with first_pulse <= pulse; pulses in "
                                        "unrecorded_frames have no video frame"},
            "source_lengths": lengths, "conserved": counts, "chunks": chunks}


def stage(args, log):
    block_dir = os.path.abspath(args.block)
    data_dir = os.path.join(block_dir, "data")
    total_chunks, _cs, state = mwb.block_chunk_total(block_dir)
    require_complete(block_dir, total_chunks, state, args.allow_incomplete)
    for done in (os.path.join(block_dir, MARKER), os.path.join(data_dir, ORIGINALS)):
        if os.path.exists(done):
            raise ValueError("%s exists: this block was already realigned (restore data/%s first "
                             "to redo it)" % (done, ORIGINALS))
    out = os.path.abspath(args.stage)
    if os.path.exists(os.path.join(out, MARKER)) and not args.force:
        raise ValueError("%s already holds a staged realignment; pass --force to redo it" % out)
    plans = [plan_drop(block_dir, state, v, args.video) for v in args.video]
    record = {"schema": SCHEMA, "kind": "in_place", "id": uuid.uuid4().hex, "block": block_dir,
              "created": _utcnow(), "tool": "tracking/colony/realign_offgrid_camera.py",
              "tool_sha256": sha256_file(os.path.abspath(__file__)),
              "per_chunk": plans[0]["per_chunk"], "total_frames": plans[0]["total"],
              "originals_dir": "data/" + ORIGINALS, "note": IN_PLACE_NOTE, "videos": {}}
    out_data = os.path.join(out, "data")
    os.makedirs(out_data, exist_ok=True)
    log.open(os.path.join(out, "realign_logs"))
    for p in plans:
        log("stage %s: %d file frames, lost (file_frame, n) %s -> %d frames, %d chunks of %d"
            % (p["video"], p["file_total"], p["gaps"], p["total"], p["n_chunks"], p["per_chunk"]))
        lengths, rows = realign_video(data_dir, out_data, p["video"], p["n_chunks"], p["per_chunk"],
                                      0, p["total"], log, False, gaps=p["gaps"], only_changed=True,
                                      realign_id=record["id"])
        counts = check_conservation(data_dir, p["video"], lengths, rows)
        entry = stage_entry(p, block_dir, data_dir, out_data, lengths, rows, counts)
        record["videos"][p["video"]] = entry
        log("conservation OK: %d aruco, %d sleap records; %d chunk(s) staged"
            % (counts["aruco_records"], counts["sleap_records"], len(entry["chunks"])))
    write_json(os.path.join(out, MARKER), record, args.group)
    log("staged realignment %s in %s" % (record["id"], out))
    return 0


def install_plan(record, block_dir, stage_data):
    """[(name, needs_move, sha256)] still to install. Verifies every file before anything moves."""
    data_dir = os.path.join(block_dir, "data")
    orig_dir = os.path.join(data_dir, ORIGINALS)
    todo = []
    for vname in sorted(record["videos"]):
        entry = record["videos"][vname]
        if sha256_file(os.path.join(block_dir, entry["sidecar"])) != entry["sidecar_sha256"]:
            raise ValueError("%s changed after staging" % entry["sidecar"])
        for chunk in entry["chunks"]:
            for s in KINDS:
                f = chunk["files"][s]
                name = f["name"]
                if sha256_file(os.path.join(stage_data, name)) != f["sha256"]:
                    raise ValueError("staged %s changed after staging" % name)
                cur, old = os.path.join(data_dir, name), os.path.join(orig_dir, name)
                cur_sha = sha256_file(cur) if os.path.isfile(cur) else None
                old_sha = sha256_file(old) if os.path.isfile(old) else None
                if cur_sha == f["sha256"] and old_sha == f["source_sha256"]:
                    continue                                    # already installed
                if cur_sha == f["source_sha256"] and old_sha is None:
                    todo.append((name, True, f["sha256"]))
                elif cur_sha is None and old_sha == f["source_sha256"]:
                    todo.append((name, False, f["sha256"]))     # an earlier install stopped here
                else:
                    raise ValueError("%s: data/ and data/%s/ match neither the staged source nor the "
                                     "re-cut file (re-detected after staging?)" % (name, ORIGINALS))
    return todo


def keep_logs(stage_dir, orig_dir, log):
    """Copy the stage/install logs next to the originals; the install itself is already done."""
    src = os.path.join(stage_dir, "realign_logs")
    try:
        if os.path.isdir(src) and os.path.isdir(orig_dir):
            dst = os.path.join(orig_dir, "realign_logs")
            if not os.path.isdir(dst):
                os.makedirs(dst)
            for n in os.listdir(src):
                shutil.copyfile(os.path.join(src, n), os.path.join(dst, n))
    except OSError as e:
        log("[WARN] the install is complete, but its logs were not copied to %s: %s" % (orig_dir, e))


def install(args, log):
    block_dir = os.path.abspath(args.block)
    stage_dir = os.path.abspath(args.install)
    data_dir = os.path.join(block_dir, "data")
    orig_dir = os.path.join(data_dir, ORIGINALS)
    with open(os.path.join(stage_dir, MARKER)) as f:
        record = json.load(f)
    if record.get("schema") != SCHEMA or record.get("kind") != "in_place":
        raise ValueError("%s is not a staged in-place realignment" % stage_dir)
    if os.path.realpath(record["block"]) != os.path.realpath(block_dir):
        raise ValueError("staged for %s, not %s" % (record["block"], block_dir))
    target = os.path.join(block_dir, MARKER)
    if os.path.exists(target):
        with open(target) as f:
            if json.load(f).get("id") != record["id"]:
                raise ValueError("%s holds a different realignment" % target)
    log.open(os.path.join(stage_dir, "realign_logs"))
    todo = install_plan(record, block_dir, os.path.join(stage_dir, "data"))
    try:
        if todo and not os.path.isdir(orig_dir):
            os.makedirs(orig_dir)
            mwb._group_share(orig_dir, args.group, is_dir=True)
        for name, move, new_sha in todo:
            cur = os.path.join(data_dir, name)
            if move:
                os.rename(cur, os.path.join(orig_dir, name))
            tmp = cur + ".realign_tmp"
            shutil.copyfile(os.path.join(stage_dir, "data", name), tmp)
            if sha256_file(tmp) != new_sha:
                raise ValueError("the copy of %s does not match its sha256" % name)
            os.replace(tmp, cur)
            mwb._group_share(cur, args.group, is_dir=False)
            log("installed %s (original -> data/%s/)" % (name, ORIGINALS))
        if not os.path.exists(target):
            record["installed"] = _utcnow()
            write_json(target, record, args.group)
    except (ValueError, OSError):
        log("[ERR] INCOMPLETE INSTALL: data/ now mixes re-cut and original files and %s is not "
            "written (tracking refuses this block); rerun --install to finish" % MARKER)
        raise
    log("%s: %d file(s) installed now; record %s (id %s)" % (block_dir, len(todo), target, record["id"]))
    keep_logs(stage_dir, orig_dir, log)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--block", required=True, help="the real block dir (holds data/)")
    ap.add_argument("--video", nargs="+",
                    help="video name of the off-grid camera (view mode: exactly one), "
                         "or of each dropping camera (--stage)")
    ap.add_argument("--offset", type=int,
                    help="view mode: frames by which its chunks >= 1 start late (its off-grid keyframe)")
    ap.add_argument("--sidecar-gaps", action="store_true",
                    help="realign the lost trigger pulses recorded in each video's diag.json")
    ap.add_argument("--stage", help="--sidecar-gaps: write the re-cut chunks + REALIGN.json here only")
    ap.add_argument("--install", metavar="STAGE_DIR",
                    help="install a staged realignment into the block's data/ (originals -> data/%s)"
                         % ORIGINALS)
    ap.add_argument("--view", help="output view dir (default: sibling <block>-sync)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="build even if some chunk of some camera is missing (testing)")
    ap.add_argument("--force", action="store_true", help="reuse an existing view / stage dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--group", default="reiteruni")
    args = ap.parse_args(argv)
    log = Log()
    try:
        if args.install:
            if args.video or args.offset is not None or args.sidecar_gaps or args.stage:
                raise ValueError("--install takes only --block (and --group)")
            return install(args, log)
        if args.stage or args.sidecar_gaps:
            if not (args.stage and args.sidecar_gaps and args.video) or args.offset is not None:
                raise ValueError("a frame drop needs --video, --sidecar-gaps and --stage (no --offset)")
            return stage(args, log)
        if not args.video or len(args.video) != 1 or args.offset is None:
            raise ValueError("the view mode needs one --video and --offset")
        args.video = args.video[0]
        return run(args, log)
    except (ValueError, OSError, KeyError) as e:
        log("[ERR] %s" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
