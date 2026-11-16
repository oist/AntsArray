#!/usr/bin/env python3
"""Tests for realign_offgrid_camera.py.

Run with pytest, or directly::

    python3 tracking/colony/test_realign_offgrid_camera.py

A synthetic block with 10-frame chunks and 37 frames. The off-grid camera's
keyframe grid is shifted by 3, so its source chunks hold [0,13) [13,23) [23,33)
[33,37). Every ArUco and SLEAP record stores its TRUE frame in X (and the dense
ArUco summary stores it too), so after the re-cut a record is correct exactly
when chunk_idx * 10 + Frame == X -- the same arithmetic stitch_tracks.py uses.

The frame-drop block: the dropping camera lost 2 pulses (true frames 24-25) at
file frame 24, so it has 35 file frames cut on the file grid [10,10,10,5]; its
file frame f is true frame f, or f + 2 from file frame 24 on.
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


def _write_chunk(data, vname, idx, first_true, n, sleap_cap=None, true_frames=None):
    """One chunk whose local frame f holds records of true frame true_frames[f]
    (default first_true + f)."""
    name = "%s_%03d" % (vname, idx)
    frames = np.arange(n)
    true = np.arange(first_true, first_true + n) if true_frames is None else np.asarray(true_frames)
    rows = [(f, int(true[f]) % IDS, float(true[f]), 1.0) for f in frames]
    rows += [(f, int(true[f] + 1) % IDS, float(true[f]), 2.0) for f in frames[::3]]
    tracks = np.zeros((n, IDS, 2), dtype=np.float32)
    conf = np.zeros((n, IDS), dtype=np.float32)
    for f, marker, x, y in rows:
        tracks[f, marker] = (x, y)
        conf[f, marker] = 1.0
    det = pd.DataFrame(rows, columns=["Frame", "Instance", "X", "Y"]).astype(
        {"Frame": "int32", "Instance": "int32", "X": "float32", "Y": "float32"})
    det["Confidence"] = np.ones(len(det), dtype=np.float32)
    save_aruco_outputs(data, name, tracks, conf, det, "h5")

    # Instance is a file-wide running index, as sleap2h5.py writes it
    rec = np.array([(f, 2 * f + i, b, float(true[f]), 0.0, 0.9)
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


# --- frame-drop block --------------------------------------------------------
DROP = "cam03_cam2_2099-01-01-00-00-00"
CLEAN = "cam01_cam0_2099-01-01-00-00-00"
GAP_FILE, GAP_MISSING = 24, 2
FILE_TOTAL = TOTAL - GAP_MISSING            # 35 file frames on the dropping camera
DROP_LENGTHS = [P, P, P, FILE_TOTAL - 3 * P]


def _true_of_file(f):
    return f + (GAP_MISSING if f >= GAP_FILE else 0)


def _sidecar(frames, gaps):
    missing = sum(g["blocksMissing"] for g in gaps)
    return {"schemaVersion": 2, "context": {"fps": 1.0}, "recorder": {"framesEncoded": frames},
            "provenance": {"firstEncodedFrame": {"blockId": 0, "fileFrameIndex": 0},
                           "lastEncodedFrame": {"blockId": frames - 1 + missing,
                                                "fileFrameIndex": frames - 1},
                           "gaps": gaps, "resetEvents": [], "invalidIdEvents": [],
                           "wrapEvents": [], "eventOverflowCount": 0}}


DROP_GAP = {"blockId": GAP_FILE + GAP_MISSING, "previousBlockId": GAP_FILE - 1,
            "blocksMissing": GAP_MISSING, "fileFrameIndex": GAP_FILE,
            "discontinuity": "forward_gap", "hostEpochMs": 1, "cameraTimestampTicks": 2}


class DropBlock:
    def __init__(self):
        self.date = tempfile.mkdtemp(prefix="rltest_drop_")
        self.root = os.path.join(self.date, "block01")
        self.data = os.path.join(self.root, "data")
        os.makedirs(self.data)
        videos = {DROP: {"n_chunks": N, "fps": 1.0, "frame_count": FILE_TOTAL},
                  CLEAN: {"n_chunks": N, "fps": 1.0, "frame_count": TOTAL}}
        ps.write(self.data, ps.new_state(self.root, P, "mkv", videos, {}))
        for v, frames, gaps in ((DROP, FILE_TOTAL, [DROP_GAP]), (CLEAN, TOTAL, [])):
            open(os.path.join(self.root, v + ".mkv"), "w").close()
            self.write_sidecar(v, _sidecar(frames, [dict(g) for g in gaps]))
        start = 0
        for c, n in enumerate(DROP_LENGTHS):
            _write_chunk(self.data, DROP, c, 0, n,
                         true_frames=[_true_of_file(start + f) for f in range(n)])
            start += n
        for c in range(N):
            _write_chunk(self.data, CLEAN, c, c * P, min(P, TOTAL - c * P))

    def sidecar_path(self, v):
        return os.path.join(self.root, v + ".mkv.diag.json")

    def write_sidecar(self, v, doc):
        with open(self.sidecar_path(v), "w") as f:
            json.dump(doc, f)

    def cleanup(self):
        shutil.rmtree(self.date, ignore_errors=True)


def _assert_on_true_grid(path_prefix, i, holes=()):
    """Every record of re-cut chunk i sits on its true frame; hole frames are empty."""
    with h5py.File(path_prefix + "_aruco_tracks.h5", "r") as h:
        rec = h["aruco_detections"][:]
        assert np.all(rec["Frame"] + i * P == rec["X"])
        dense = h["aruco_tracks"][:]
        f, m = np.nonzero(dense[:, :, 0])
        assert np.all(f + i * P == dense[f, m, 0])
        for lo, hi in holes:
            assert not np.any(dense[lo:hi]) and not np.any(h["aruco_confidences"][lo:hi])
            assert not np.any((rec["Frame"] >= lo) & (rec["Frame"] < hi))
    with h5py.File(path_prefix + "_sleap_data.h5", "r") as h:
        s = h["sleap_data"][:]
        assert np.all(s["Frame"] + i * P == s["X"])
        keys = s["Frame"].astype(np.int64) * 10 ** 6 + s["Instance"] * 10 + s["Bodypoint"]
        assert len(np.unique(keys)) == len(s)
        for lo, hi in holes:
            assert not np.any((s["Frame"] >= lo) & (s["Frame"] < hi))


def _quiet(_msg):
    pass


# ---------------------------------------------------------------------------
def test_pieces_tile_the_true_grid():
    starts = roc.source_layout(SRC_LENGTHS, P, OFF, TOTAL)
    assert starts == [0, 13, 23, 33]
    runs = roc.source_runs(starts, SRC_LENGTHS)
    assert roc.pieces_for(0, runs, P, TOTAL) == [(0, 0, 10, 0)]
    assert roc.pieces_for(1, runs, P, TOTAL) == [(0, 10, 13, 0), (1, 0, 7, 3)]
    assert roc.pieces_for(3, runs, P, TOTAL) == [(2, 7, 10, 0), (3, 0, 4, 3)]


def test_drop_runs_pieces_and_holes():
    starts = roc.source_layout(DROP_LENGTHS, P, 0, FILE_TOTAL)
    assert starts == [0, 10, 20, 30]
    gaps = [(GAP_FILE, GAP_MISSING)]
    runs = roc.source_runs(starts, DROP_LENGTHS, gaps)
    assert runs == [(0, 0, 10, 0), (1, 0, 10, 10), (2, 0, 4, 20), (2, 4, 10, 26), (3, 0, 5, 32)]
    holes = roc.hole_spans(gaps)
    assert holes == [(24, 26)]
    assert roc.pieces_for(1, runs, P, TOTAL, holes) == [(1, 0, 10, 0)]
    assert roc.pieces_for(2, runs, P, TOTAL, holes) == [(2, 0, 4, 0), (2, 4, 8, 6)]
    assert roc.pieces_for(3, runs, P, TOTAL, holes) == [(2, 8, 10, 0), (3, 0, 5, 2)]
    assert roc.chunk_holes(2, P, TOTAL, holes) == [[4, 6]]
    assert roc.chunk_holes(3, P, TOTAL, holes) == []


def test_drop_refuses_undeclared_holes_and_impossible_gaps():
    starts = roc.source_layout(DROP_LENGTHS, P, 0, FILE_TOTAL)
    runs = roc.source_runs(starts, DROP_LENGTHS, [(GAP_FILE, GAP_MISSING)])
    try:
        roc.pieces_for(2, runs, P, TOTAL)          # the 2 empty frames were not declared
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an undeclared hole")
    for gaps in ([(0, 2)], [(FILE_TOTAL, 2)], [(GAP_FILE, 0)]):
        try:
            roc.source_runs(starts, DROP_LENGTHS, gaps)
        except ValueError:
            continue
        raise AssertionError("accepted gap %s" % gaps)


def test_sidecar_gaps_are_read_and_checked():
    b = DropBlock()
    try:
        gaps, raw, frames = roc.sidecar_gaps(b.root, DROP)
        assert gaps == [(GAP_FILE, GAP_MISSING)] and raw == [DROP_GAP] and frames == FILE_TOTAL
        assert roc.sidecar_gaps(b.root, CLEAN)[0] == []
        bad = [("lastEncodedFrame", {"blockId": FILE_TOTAL - 1, "fileFrameIndex": FILE_TOTAL - 1}),
               ("resetEvents", [{"blockId": 3}]),
               ("gaps", [dict(DROP_GAP, blockId=GAP_FILE + GAP_MISSING + 1)])]
        for key, value in bad:
            doc = _sidecar(FILE_TOTAL, [dict(DROP_GAP)])
            doc["provenance"][key] = value
            b.write_sidecar(DROP, doc)
            try:
                roc.sidecar_gaps(b.root, DROP)
            except ValueError:
                continue
            raise AssertionError("accepted a sidecar with a bad %s" % key)
    finally:
        b.cleanup()


def test_drop_recut_puts_every_record_on_its_true_frame():
    b = DropBlock()
    out = tempfile.mkdtemp(prefix="rltest_out_")
    try:
        lengths, rows = roc.realign_video(b.data, out, DROP, N, P, 0, TOTAL, _quiet, False,
                                          gaps=[(GAP_FILE, GAP_MISSING)])
        assert lengths == DROP_LENGTHS
        assert [r["frames"] for r in rows] == [10, 10, 10, 7]
        assert [r["holes"] for r in rows] == [[], [], [[4, 6]], []]
        assert [r["changed"] for r in rows] == [False, False, True, True]
        for i in range(N):
            prefix = os.path.join(out, "%s_%03d" % (DROP, i))
            _assert_on_true_grid(prefix, i, rows[i]["holes"])
            with h5py.File(prefix + "_sleap_data.h5", "r") as h:
                assert int(h.attrs["frame_count"]) == int(h.attrs["expected_frames"]) == rows[i]["frames"]
                assert json.loads(h.attrs["hole_frames"]) == rows[i]["holes"]
            with h5py.File(prefix + "_aruco_tracks.h5", "r") as h:
                assert int(h.attrs["num_frames"]) == rows[i]["frames"]
                assert json.loads(h.attrs["hole_frames"]) == rows[i]["holes"]
        counts = roc.check_conservation(b.data, DROP, lengths, rows)
        assert counts["sleap_records"] == FILE_TOTAL * 6
    finally:
        b.cleanup()
        shutil.rmtree(out, ignore_errors=True)


def test_only_changed_chunks_are_written():
    b = DropBlock()
    out = tempfile.mkdtemp(prefix="rltest_out_")
    try:
        lengths, rows = roc.realign_video(b.data, out, DROP, N, P, 0, TOTAL, _quiet, False,
                                          gaps=[(GAP_FILE, GAP_MISSING)], only_changed=True,
                                          realign_id="abc")
        written = sorted(os.listdir(out))
        assert written == sorted("%s_%03d%s" % (DROP, i, s) for i in (2, 3)
                                 for s in ("_aruco_tracks.h5", "_aruco_detections.h5", "_sleap_data.h5"))
        with h5py.File(os.path.join(out, "%s_002_sleap_data.h5" % DROP), "r") as h:
            assert h.attrs["realign_id"] == "abc"
        with h5py.File(os.path.join(out, "%s_002_aruco_tracks.h5" % DROP), "r") as h:
            assert h.attrs["realign_id"] == "abc"
        roc.check_conservation(b.data, DROP, lengths, rows)
    finally:
        b.cleanup()
        shutil.rmtree(out, ignore_errors=True)


def _stage(b, stage):
    return roc.main(["--block", b.root, "--video", DROP, "--sidecar-gaps",
                     "--stage", stage, "--group", ""])


def test_stage_and_install_in_place():
    b = DropBlock()
    stage = tempfile.mkdtemp(prefix="rltest_stage_")
    try:
        assert _stage(b, stage) == 0
        with open(os.path.join(stage, roc.MARKER)) as f:
            record = json.load(f)
        assert record["kind"] == "in_place" and record["total_frames"] == TOTAL
        entry = record["videos"][DROP]
        assert entry["camera"] == "cam03" and entry["sidecar_gaps"] == [DROP_GAP]
        assert entry["unrecorded_frames"] == [[24, 26]] and entry["recorded_frames"] == FILE_TOTAL
        assert [c["chunk"] for c in entry["chunks"]] == [2, 3]
        assert not os.path.exists(os.path.join(b.root, roc.MARKER))       # staging writes nothing
        assert not os.path.exists(os.path.join(b.data, roc.ORIGINALS))

        assert roc.main(["--block", b.root, "--install", stage, "--group", ""]) == 0
        with open(os.path.join(b.root, roc.MARKER)) as f:
            installed = json.load(f)
        assert installed["id"] == record["id"] and installed["installed"]
        for i in range(N):
            prefix = os.path.join(b.data, "%s_%03d" % (DROP, i))
            _assert_on_true_grid(prefix, i, [[4, 6]] if i == 2 else [])
            with h5py.File(prefix + "_sleap_data.h5", "r") as h:
                assert (h.attrs.get("realign_id") == record["id"]) == (i >= 2)
        for i in (2, 3):
            for s in ("_aruco_tracks.h5", "_aruco_detections.h5", "_sleap_data.h5"):
                assert os.path.isfile(os.path.join(b.data, roc.ORIGINALS, "%s_%03d%s" % (DROP, i, s)))
            assert os.path.isfile(os.path.join(b.data, "%s_%03d.slp" % (DROP, i)))  # .slp stay
        assert not any(n.startswith(CLEAN) for n in os.listdir(os.path.join(b.data, roc.ORIGINALS)))
        # installing again is a no-op
        assert roc.main(["--block", b.root, "--install", stage, "--group", ""]) == 0
    finally:
        b.cleanup()
        shutil.rmtree(stage, ignore_errors=True)


def test_install_refuses_a_source_changed_after_staging():
    b = DropBlock()
    stage = tempfile.mkdtemp(prefix="rltest_stage_")
    try:
        assert _stage(b, stage) == 0
        with h5py.File(os.path.join(b.data, "%s_003_sleap_data.h5" % DROP), "r+") as h:
            h.attrs["touched"] = 1
        assert roc.main(["--block", b.root, "--install", stage, "--group", ""]) == 2
        assert not os.path.exists(os.path.join(b.root, roc.MARKER))
        assert not os.path.exists(os.path.join(b.data, roc.ORIGINALS))
    finally:
        b.cleanup()
        shutil.rmtree(stage, ignore_errors=True)


def test_stage_refuses_a_drop_that_does_not_match_the_clean_cameras():
    b = DropBlock()
    stage = tempfile.mkdtemp(prefix="rltest_stage_")
    try:
        gap = dict(DROP_GAP, blocksMissing=3, blockId=GAP_FILE + 3)
        b.write_sidecar(DROP, _sidecar(FILE_TOTAL, [gap]))
        assert _stage(b, stage) == 2
        assert not os.path.exists(os.path.join(stage, roc.MARKER))
    finally:
        b.cleanup()
        shutil.rmtree(stage, ignore_errors=True)


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
