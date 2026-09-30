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
"""
import argparse
import datetime
import json
import os
import sys

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


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def pieces_for(i, starts, lengths, per_chunk, total):
    """[(src_chunk, src_lo, src_hi, dst_lo)] that tile true chunk i, in frame order."""
    lo, hi = i * per_chunk, min((i + 1) * per_chunk, total)
    out = []
    for c, (s, n) in enumerate(zip(starts, lengths)):
        a, b = max(lo, s), min(hi, s + n)
        if a < b:
            out.append((c, a - s, b - s, a - lo))
    if sum(p[2] - p[1] for p in out) != hi - lo:
        raise ValueError("true chunk %03d is not fully covered by the source chunks" % i)
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


def build_chunk(pieces, cache):
    """Concatenate the pieces of one true chunk: aruco dense + records, sleap records."""
    trk, conf, a_rec, s_rec, sources = [], [], [], [], []
    for c, lo, hi, dst_lo in pieces:
        aru, slp = cache.get(c)
        shift = dst_lo - lo
        trk.append(aru["tracks"][lo:hi])
        conf.append(aru["conf"][lo:hi])
        a_rec.append(_slice_records(aru["rec"], lo, hi, shift))
        s_rec.append(_slice_records(slp["rec"], lo, hi, shift))
        sources.append({"src_chunk": c, "src_frames": [lo, hi], "dst_first": dst_lo,
                        "sleap_attrs": slp["attrs"]})
    return {"tracks": np.concatenate(trk), "conf": np.concatenate(conf),
            "aruco": np.concatenate(a_rec), "sleap": np.concatenate(s_rec),
            "sources": sources}


def _sleap_frame_count(chunk, dst_len, src_lengths):
    """frame_count keeps its source meaning: every frame if the sources counted every frame."""
    full = all(int(s["sleap_attrs"].get("frame_count", -1)) == src_lengths[s["src_chunk"]]
               for s in chunk["sources"])
    return dst_len if full else int(len(np.unique(chunk["sleap"]["Frame"])))


def write_sleap(path, chunk, dst_len, src_lengths, src_names):
    attrs = [s["sleap_attrs"] for s in chunk["sources"]]
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as h:
        h.attrs["source_file"] = "; ".join(src_names)
        h.attrs["frame_count"] = _sleap_frame_count(chunk, dst_len, src_lengths)
        h.attrs["expected_frames"] = dst_len
        h.attrs["instance_count"] = max(int(a.get("instance_count", 0)) for a in attrs)
        h.attrs["node_count"] = int(attrs[0].get("node_count", 0))
        h.attrs["realigned_from"] = json.dumps([{k: s[k] for k in ("src_chunk", "src_frames", "dst_first")}
                                                for s in chunk["sources"]])
        if len(chunk["sleap"]) == 0:
            h.create_dataset("sleap_data", data=chunk["sleap"])
        else:
            h.create_dataset("sleap_data", data=chunk["sleap"], compression="gzip",
                             shuffle=True, chunks=True)
    os.replace(tmp, path)


def realign_video(data_dir, out_dir, vname, n_chunks, per_chunk, offset, total, log, dry_run):
    lengths = source_lengths(data_dir, vname, n_chunks)
    starts = source_layout(lengths, per_chunk, offset, total)
    cache = SourceCache(data_dir, vname, lengths)
    rows = []
    for i in range(n_chunks):
        pieces = pieces_for(i, starts, lengths, per_chunk, total)
        dst_len = min(per_chunk, total - i * per_chunk)
        cache.drop_below(pieces[0][0])
        chunk = build_chunk(pieces, cache)
        if chunk["tracks"].shape[0] != dst_len:
            raise ValueError("true chunk %03d came out %d frames, want %d"
                             % (i, chunk["tracks"].shape[0], dst_len))
        if not dry_run:
            name = "%s_%03d" % (vname, i)
            det = pd.DataFrame.from_records(chunk["aruco"])
            save_aruco_outputs(out_dir, name, chunk["tracks"], chunk["conf"], det, "h5")
            write_sleap(os.path.join(out_dir, name + SDAT), chunk, dst_len, lengths,
                        [_name(vname, p[0], SDAT) for p in pieces])
        rows.append({"chunk": i, "frames": dst_len, "aruco_records": int(len(chunk["aruco"])),
                     "sleap_records": int(len(chunk["sleap"])),
                     "pieces": [list(p) for p in pieces]})
        log("chunk %03d: %s -> %d frames, aruco %d, sleap %d"
            % (i, " + ".join("src%03d[%d:%d]" % p[:3] for p in pieces), dst_len,
               rows[-1]["aruco_records"], rows[-1]["sleap_records"]))
    return lengths, rows


def check_conservation(data_dir, vname, lengths, rows):
    """Every source record lies inside [0, total), so the re-cut must keep them all."""
    a_in = s_in = 0
    for c in range(len(lengths)):
        with h5py.File(os.path.join(data_dir, _name(vname, c, TRK)), "r") as h:
            a_in += h["aruco_detections"].shape[0]
        with h5py.File(os.path.join(data_dir, _name(vname, c, SDAT)), "r") as h:
            s_in += h["sleap_data"].shape[0]
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


def run(args, log):
    block_dir = os.path.abspath(args.block)
    total_chunks, _cs, state = mwb.block_chunk_total(block_dir)
    n_chunks, per_chunk, total = contract_geometry(state, args.video)
    comp = mwb.window_completeness(block_dir, 0, total_chunks - 1, state)
    holes = dict((s, v) for s, v in comp.items() if v[0] != v[1])
    if holes and not args.allow_incomplete:
        raise ValueError("detection is not complete (%s); tracking would stop at the first hole"
                         % ", ".join("%s %d/%d" % (s, v[0], v[1]) for s, v in sorted(holes.items())))
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--block", required=True, help="the real block dir (holds data/)")
    ap.add_argument("--video", required=True, help="video name of the off-grid camera")
    ap.add_argument("--offset", type=int, required=True,
                    help="frames by which its chunks >= 1 start late (its off-grid keyframe)")
    ap.add_argument("--view", help="output view dir (default: sibling <block>-sync)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="build even if some chunk of some camera is missing (testing)")
    ap.add_argument("--force", action="store_true", help="reuse an existing view dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--group", default="reiteruni")
    args = ap.parse_args(argv)
    log = Log()
    try:
        return run(args, log)
    except (ValueError, OSError, KeyError) as e:
        log("[ERR] %s" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
