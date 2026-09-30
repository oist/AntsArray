#!/usr/bin/env python3
"""Tests for realign_offgrid_camera.py.

Run with pytest, or directly::

    python3 tracking/colony/test_realign_offgrid_camera.py

A synthetic block with 10-frame chunks and 37 frames. The off-grid camera's
keyframe grid is shifted by 3, so its source chunks hold [0,13) [13,23) [23,33)
[33,37). Every ArUco and SLEAP record stores its TRUE frame in X (and the dense
ArUco summary stores it too), so after the re-cut a record is correct exactly
when chunk_idx * 10 + Frame == X -- the same arithmetic stitch_tracks.py uses.
"""
import json
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "detection_pipeline", "lib"))
sys.path.insert(0, os.path.join(REPO, "detection_pipeline", "scripts"))

import pipeline_state as ps  # noqa: E402
import realign_offgrid_camera as roc  # noqa: E402
from aruco_output import save_aruco_outputs  # noqa: E402

P, OFF, TOTAL, N = 10, 3, 37, 4
IDS = 5
BAD = "cam19_cam1_2099-01-01-00-00-00"
GOOD = "cam18_cam0_2099-01-01-00-00-00"
SRC_LENGTHS = [P + OFF, P, P, TOTAL - (3 * P + OFF)]
SLEAP_DTYPE = [("Frame", "<i4"), ("Instance", "<i4"), ("Bodypoint", "<i4"),
               ("X", "<f4"), ("Y", "<f4"), ("Score_node", "<f4")]


def _symlinks_supported():
    d = tempfile.mkdtemp(prefix="rltest_")
    try:
        open(os.path.join(d, "t"), "w").close()
        os.symlink("t", os.path.join(d, "l"))
        return True
    except OSError:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


SYMLINKS = _symlinks_supported()


def _worklist_cap(idx):
    """What detection_pipeline/lib/worklist.py records as expected_frames: the TRUE-grid length,
    so the off-grid camera's last chunk (4 frames here) carries a cap of 7."""
    return P if idx < N - 1 else TOTAL - (N - 1) * P


def _write_chunk(data, vname, idx, first_true, n, sleap_cap=None):
    """One chunk whose records sit on true frames first_true .. first_true+n-1."""
    name = "%s_%03d" % (vname, idx)
    frames = np.arange(n)
    rows = [(f, (first_true + f) % IDS, float(first_true + f), 1.0) for f in frames]
    rows += [(f, (first_true + f + 1) % IDS, float(first_true + f), 2.0) for f in frames[::3]]
    tracks = np.zeros((n, IDS, 2), dtype=np.float32)
    conf = np.zeros((n, IDS), dtype=np.float32)
    for f, marker, x, y in rows:
        tracks[f, marker] = (x, y)
        conf[f, marker] = 1.0
    det = pd.DataFrame(rows, columns=["Frame", "Instance", "X", "Y"]).astype(
        {"Frame": "int32", "Instance": "int32", "X": "float32", "Y": "float32"})
    det["Confidence"] = np.ones(len(det), dtype=np.float32)
    save_aruco_outputs(data, name, tracks, conf, det, "h5")

    rec = np.array([(f, i, b, float(first_true + f), 0.0, 0.9)
                    for f in frames for i in range(2) for b in range(3)], dtype=SLEAP_DTYPE)
    with h5py.File(os.path.join(data, name + "_sleap_data.h5"), "w") as h:
        h.attrs["source_file"] = name + ".slp"
        h.attrs["frame_count"] = n
        h.attrs["expected_frames"] = n if sleap_cap is None else sleap_cap
        h.attrs["instance_count"] = 2
        h.attrs["node_count"] = 3
        h.create_dataset("sleap_data", data=rec)
    open(os.path.join(data, name + ".slp"), "w").close()


class Block:
    def __init__(self, sleap_cap_chunk0=None, skip_good_last=False):
        self.date = tempfile.mkdtemp(prefix="rltest_date_")
        self.root = os.path.join(self.date, "block01")
        self.data = os.path.join(self.root, "data")
        os.makedirs(self.data)
        for v in (BAD, GOOD):
            open(os.path.join(self.root, v + ".mkv"), "w").close()
        videos = dict((v, {"n_chunks": N, "fps": 1.0, "frame_count": TOTAL}) for v in (BAD, GOOD))
        ps.write(self.data, ps.new_state(self.root, P, "mkv", videos, {}))
        start = 0
        for c, n in enumerate(SRC_LENGTHS):
            # chunk 0 was re-run over its full length (the rescue); the rest carry worklist caps
            cap = (sleap_cap_chunk0 or n) if c == 0 else _worklist_cap(c)
            _write_chunk(self.data, BAD, c, start, n, cap)
            start += n
        for c in range(N if not skip_good_last else N - 1):
            _write_chunk(self.data, GOOD, c, c * P, min(P, TOTAL - c * P))

    def cleanup(self):
        shutil.rmtree(self.date, ignore_errors=True)


def _quiet(_msg):
    pass


# ---------------------------------------------------------------------------
def test_pieces_tile_the_true_grid():
    starts = roc.source_layout(SRC_LENGTHS, P, OFF, TOTAL)
    assert starts == [0, 13, 23, 33]
    assert roc.pieces_for(0, starts, SRC_LENGTHS, P, TOTAL) == [(0, 0, 10, 0)]
    assert roc.pieces_for(1, starts, SRC_LENGTHS, P, TOTAL) == [(0, 10, 13, 0), (1, 0, 7, 3)]
    assert roc.pieces_for(3, starts, SRC_LENGTHS, P, TOTAL) == [(2, 7, 10, 0), (3, 0, 4, 3)]


def test_layout_refuses_an_inconsistent_shift():
    # chunk 002 starts one frame late; the lengths no longer add up to the contract
    for lengths in ([13, 11, 10, 3], [13, 10, 10, 5]):
        try:
            roc.source_layout(lengths, P, OFF, TOTAL)
        except ValueError:
            continue
        raise AssertionError("accepted %s" % lengths)


def test_recut_puts_every_record_on_its_true_frame():
    b = Block()
    out = tempfile.mkdtemp(prefix="rltest_out_")
    try:
        lengths, rows = roc.realign_video(b.data, out, BAD, N, P, OFF, TOTAL, _quiet, False)
        assert [r["frames"] for r in rows] == [10, 10, 10, 7]
        for i in range(N):
            name = os.path.join(out, "%s_%03d" % (BAD, i))
            with h5py.File(name + "_aruco_tracks.h5", "r") as h:
                n = int(h.attrs["num_frames"])
                assert n == rows[i]["frames"] == h["aruco_tracks"].shape[0]
                rec = h["aruco_detections"][:]
                assert np.all(rec["Frame"] + i * P == rec["X"])
                dense = h["aruco_tracks"][:]
                f, m = np.nonzero(dense[:, :, 0])
                assert np.all(f + i * P == dense[f, m, 0])
            assert len(pd.read_hdf(name + "_aruco_detections.h5", "detections")) == len(rec)
            with h5py.File(name + "_sleap_data.h5", "r") as h:
                s = h["sleap_data"][:]
                assert np.all(s["Frame"] + i * P == s["X"])
                assert int(h.attrs["expected_frames"]) == rows[i]["frames"]
                assert int(h.attrs["frame_count"]) == rows[i]["frames"]
        counts = roc.check_conservation(b.data, BAD, lengths, rows)
        assert counts["sleap_records"] == TOTAL * 6
    finally:
        b.cleanup()
        shutil.rmtree(out, ignore_errors=True)


def test_refuses_a_sleap_chunk_capped_short():
    b = Block(sleap_cap_chunk0=P)   # the pre-rescue cam19_000: 43,200 of 43,657
    out = tempfile.mkdtemp(prefix="rltest_out_")
    try:
        try:
            roc.realign_video(b.data, out, BAD, N, P, OFF, TOTAL, _quiet, False)
        except ValueError as e:
            assert "re-run SLEAP" in str(e)
        else:
            raise AssertionError("accepted a capped SLEAP chunk")
    finally:
        b.cleanup()
        shutil.rmtree(out, ignore_errors=True)


def test_view_block_end_to_end():
    if not SYMLINKS:
        print("skip: no symlink support here")
        return
    b = Block()
    try:
        assert roc.main(["--block", b.root, "--video", BAD, "--offset", str(OFF), "--group", ""]) == 0
        view = b.root + "-sync"
        vdata = os.path.join(view, "data")
        names = sorted(os.listdir(vdata))
        assert not any(n.startswith(BAD) and n.endswith(".slp") for n in names)
        for i in range(N):
            for suf in ("_aruco_tracks.h5", "_aruco_detections.h5", "_sleap_data.h5"):
                p = os.path.join(vdata, "%s_%03d%s" % (BAD, i, suf))
                assert os.path.isfile(p) and not os.path.islink(p)
                q = os.path.join(vdata, "%s_%03d%s" % (GOOD, i, suf))
                assert os.path.islink(q) and not os.path.isabs(os.readlink(q))
        with open(os.path.join(vdata, ps.STATE_BASENAME)) as f:
            assert json.load(f)["realign"]["offset"] == OFF
        with open(os.path.join(view, roc.MARKER)) as f:
            marker = json.load(f)
        assert marker["source_lengths"] == SRC_LENGTHS
        assert os.path.islink(os.path.join(view, BAD + ".mkv"))
        # the source block is untouched
        assert os.path.isfile(os.path.join(b.data, "%s_001.slp" % BAD))
        assert roc.main(["--block", b.root, "--video", BAD, "--offset", str(OFF), "--group", ""]) == 2
    finally:
        b.cleanup()


def test_refuses_incomplete_detection():
    b = Block(skip_good_last=True)
    try:
        assert roc.main(["--block", b.root, "--video", BAD, "--offset", str(OFF),
                         "--group", "", "--dry-run"]) == 2
        assert not os.path.exists(b.root + "-sync")
    finally:
        b.cleanup()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
