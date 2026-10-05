"""Rerun views: SLEAP re-processing of selected (camera, chunk) pairs of an existing block.

make_rerun_view.py builds <block>-<suffix>/ next to a block: links to the target cameras' videos,
and in data/ links to the outputs of every chunk that is NOT being re-run, so the pipeline's
bucket-aware skip sends only the listed chunks to the GPU. promote_rerun.py later moves each
finished output into the original block's data/ (the superseded file goes to a side folder).
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import tempfile

import h5py
import numpy as np
import pytest


def _can_symlink():
    d = tempfile.mkdtemp()
    try:
        os.symlink(d, os.path.join(d, "probe"))
        return True
    except (OSError, NotImplementedError, AttributeError):
        return False


# The views are symlink trees on Linux (deigo/saion); unprivileged Windows cannot create them.
pytestmark = pytest.mark.skipif(not _can_symlink(), reason="needs symlinks: run on deigo/saion")

ROOT = Path(__file__).resolve().parent
MAKE = ROOT / "scripts/make_rerun_view.py"
PROMOTE = ROOT / "scripts/promote_rerun.py"
VID = {"cam01": "cam01_cam0_2099-01-01-00-00-00", "cam04": "cam04_cam3_2099-01-01-00-00-00",
       "cam13": "cam13_cam4_2099-01-01-00-00-08"}


def write_sdat(path, per_frame, expected=100):
    """A _sleap_data.h5 holding per_frame[i] instances in frame i (2 bodypoints each)."""
    dt = np.dtype([("Frame", "<i8"), ("Instance", "<i8"), ("Bodypoint", "<i8"),
                   ("X", "<f8"), ("Y", "<f8"), ("Score_node", "<f8")])
    rows = [(f, i, b, 1.0, 2.0, 0.9) for f, n in enumerate(per_frame) for i in range(n) for b in range(2)]
    with h5py.File(str(path), "w") as h5:
        h5.attrs["expected_frames"] = expected
        h5.create_dataset("sleap_data", data=np.array(rows, dtype=dt))


def make_block(tmp_path, chunks=3):
    """basler/<date>/block01 with three cameras, a sess log and complete outputs for every chunk."""
    blk = tmp_path / "basler" / "20990101" / "block01"
    (blk / "data").mkdir(parents=True)
    (blk / "sess_20990101_000000.txt").write_text("sess\n")
    for v in VID.values():
        (blk / (v + ".mkv")).write_bytes(b"video")
        (blk / (v + ".mkv.diag.json")).write_text("{}")
        for c in range(chunks):
            (blk / "data" / ("%s_%03d.slp" % (v, c))).write_bytes(b"old-slp")
            write_sdat(blk / "data" / ("%s_%03d_sleap_data.h5" % (v, c)), [20, 20, 20])
            (blk / "data" / ("%s_%03d_aruco_tracks.h5" % (v, c))).write_bytes(b"aruco")
    return blk


def write_rerun(tmp_path, rows, name="rerun.tsv"):
    p = tmp_path / name
    p.write_text("".join("%s\t%s\t%03d\n" % r for r in rows))
    return p


def run(script, *args):
    return subprocess.run([sys.executable, str(script)] + [str(a) for a in args],
                          text=True, capture_output=True, timeout=60)


def test_view_links_target_cameras_and_every_chunk_not_being_rerun(tmp_path):
    blk = make_block(tmp_path)
    rerun = write_rerun(tmp_path, [("20990101/block01", "cam04", 1), ("20990101/block01", "cam04", 2),
                                   ("20990101/block01", "cam01", 0), ("20990102/block01", "cam13", 0)])
    r = run(MAKE, "--block", blk, "--rerun", rerun, "--suffix", "k96")
    assert r.returncode == 0, r.stderr
    view = blk.parent / "block01-k96"
    # videos + sidecars of the two target cameras and the session log, as relative links that resolve
    names = sorted(p.name for p in view.iterdir() if p.is_symlink())
    assert names == sorted([VID["cam01"] + ".mkv", VID["cam01"] + ".mkv.diag.json",
                            VID["cam04"] + ".mkv", VID["cam04"] + ".mkv.diag.json",
                            "sess_20990101_000000.txt"])
    for p in list(view.iterdir()) + list((view / "data").iterdir()):
        if p.is_symlink():
            assert not os.path.isabs(os.readlink(str(p))) and p.exists(), p
    linked = sorted(p.name for p in (view / "data").iterdir())
    want = []
    for cam, done in (("cam01", (1, 2)), ("cam04", (0,))):
        for c in done:
            want += ["%s_%03d.slp" % (VID[cam], c), "%s_%03d_sleap_data.h5" % (VID[cam], c)]
    assert linked == sorted(want)          # rerun chunks absent; cam13 (other block's row) absent
    info = json.loads((view / "VIEW_INFO.json").read_text())
    assert info["cams"]["cam04"]["rerun_chunks"] == [1, 2]
    assert info["cams"]["cam01"]["rerun_chunks"] == [0]
    assert set(info["cams"]) == {"cam01", "cam04"}


def test_existing_view_is_refused(tmp_path):
    blk = make_block(tmp_path)
    rerun = write_rerun(tmp_path, [("20990101/block01", "cam04", 1)])
    assert run(MAKE, "--block", blk, "--rerun", rerun, "--suffix", "k96").returncode == 0
    r = run(MAKE, "--block", blk, "--rerun", rerun, "--suffix", "k96")
    assert r.returncode != 0 and "exists" in r.stderr


def test_no_rows_for_the_block_is_refused(tmp_path):
    blk = make_block(tmp_path)
    rerun = write_rerun(tmp_path, [("20990102/block01", "cam04", 1)])
    r = run(MAKE, "--block", blk, "--rerun", rerun, "--suffix", "k96")
    assert r.returncode != 0
    assert not (blk.parent / "block01-k96").exists()


def finish(view, cam, chunk, per_frame, expected=100):
    """What the pipeline leaves behind for one re-run chunk: real files in the view's data/."""
    v = VID[cam]
    (view / "data" / ("%s_%03d.slp" % (v, chunk))).write_bytes(b"new-slp")
    write_sdat(view / "data" / ("%s_%03d_sleap_data.h5" % (v, chunk)), per_frame, expected)


def setup_view(tmp_path, rows=((("20990101/block01", "cam04", 0)), ("20990101/block01", "cam04", 1),
                                ("20990101/block01", "cam04", 2))):
    blk = make_block(tmp_path)
    rerun = write_rerun(tmp_path, list(rows))
    assert run(MAKE, "--block", blk, "--rerun", rerun, "--suffix", "k96").returncode == 0
    return blk, blk.parent / "block01-k96"


def test_promote_moves_finished_chunks_and_keeps_the_superseded_ones(tmp_path):
    blk, view = setup_view(tmp_path)
    finish(view, "cam04", 0, [35, 41, 38])
    finish(view, "cam04", 1, [30, 30, 33])
    r = run(PROMOTE, "--view", view, "--yes")
    assert r.returncode == 0, r.stderr
    v = VID["cam04"]
    for c in (0, 1):
        assert (blk / "data" / ("%s_%03d.slp" % (v, c))).read_bytes() == b"new-slp"
        assert (blk / "data" / "_superseded_cap20" / ("%s_%03d.slp" % (v, c))).read_bytes() == b"old-slp"
        assert not (view / "data" / ("%s_%03d.slp" % (v, c))).exists()
    # chunk 2 not finished yet: untouched on both sides, reported as pending
    assert (blk / "data" / ("%s_002.slp" % v)).read_bytes() == b"old-slp"
    assert "pending" in r.stdout
    rec = json.loads((blk / "data" / "REPROCESS_k96.json").read_text())
    assert sorted(e["chunk"] for e in rec["promoted"]) == [0, 1]
    assert rec["promoted"][0]["cam"] == "cam04"


def test_promote_refuses_a_chunk_whose_frame_count_changed(tmp_path):
    blk, view = setup_view(tmp_path)
    finish(view, "cam04", 0, [35, 41, 38], expected=99)
    r = run(PROMOTE, "--view", view, "--yes")
    assert r.returncode != 0
    assert "expected_frames" in r.stdout + r.stderr
    assert (blk / "data" / ("%s_000.slp" % VID["cam04"])).read_bytes() == b"old-slp"


def test_promote_records_chunks_that_reach_the_new_cap(tmp_path):
    blk, view = setup_view(tmp_path)
    finish(view, "cam04", 0, [96, 41, 38])
    r = run(PROMOTE, "--view", view, "--cap", "96", "--yes")
    assert r.returncode == 0, r.stderr
    assert "at cap" in r.stdout
    rec = json.loads((blk / "data" / "REPROCESS_k96.json").read_text())
    assert rec["promoted"][0]["frames_at_cap"] == 1


def test_promote_dry_run_and_chunk_range_move_nothing_else(tmp_path):
    blk, view = setup_view(tmp_path)
    finish(view, "cam04", 0, [35, 41, 38])
    finish(view, "cam04", 1, [30, 30, 33])
    assert run(PROMOTE, "--view", view).returncode == 0          # no --yes = dry run
    assert (blk / "data" / ("%s_000.slp" % VID["cam04"])).read_bytes() == b"old-slp"
    r = run(PROMOTE, "--view", view, "--chunks", "1-1", "--yes")
    assert r.returncode == 0, r.stderr
    assert (blk / "data" / ("%s_000.slp" % VID["cam04"])).read_bytes() == b"old-slp"
    assert (blk / "data" / ("%s_001.slp" % VID["cam04"])).read_bytes() == b"new-slp"


def test_promote_never_moves_the_view_links(tmp_path):
    blk, view = setup_view(tmp_path, rows=[("20990101/block01", "cam01", 2)])
    r = run(PROMOTE, "--view", view, "--yes")
    assert r.returncode == 0, r.stderr
    assert (view / "data" / ("%s_000.slp" % VID["cam01"])).is_symlink()
    assert (blk / "data" / ("%s_000.slp" % VID["cam01"])).read_bytes() == b"old-slp"
    assert not (blk / "data" / "_superseded_cap20").exists()
