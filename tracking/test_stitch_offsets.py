#!/usr/bin/env python3
"""Regression tests for block-global frame offsets in stitch_tracks.py.

Run with pytest, or directly -- the tracking venv ships no pytest, so this file
carries its own runner::

    /apps/unit/ReiterU/ant_tracking/venv/bin/python tracking/test_stitch_offsets.py

The load-bearing test is ``test_subset_keeps_block_frame_numbering``.

``stitch_group`` had two ways to place a chunk on the block's timeline, and both
were anchored to the set of files present: ``ref_dt`` is ``min()`` over them, and
``running_frame_offset`` accumulates over them. Stitching chunks 5-6 alone
therefore renumbered them from frame 0, and nothing looked wrong -- the parquet
is well formed and the track is continuous, only the clock is off by 216,000
frames. Every speed and sleep-bout figure downstream is computed on that clock.

The timestamp path never applied to this pipeline anyway: it needs
``YYYYMMDD-HHMMSS`` while colony videos are named ``2099-01-01-00-00-00``.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking import stitch_tracks as st  # noqa: E402

FPS = 24.0
CHUNK_SEC = 1800
CHUNK_FRAMES = int(FPS * CHUNK_SEC)  # 43200
TRACK_ID = 7
SUFFIX = "left"


def write_chunk_parquet(dirpath, chunk_idx, n_frames=CHUNK_FRAMES, rows=50):
    """One per-chunk per-track parquet, frames LOCAL to the chunk (0-based)."""
    df = pd.DataFrame({
        "Frame": np.arange(rows, dtype=np.int64),
        "TrackID": np.full(rows, TRACK_ID, dtype=np.int64),
        "TrackX": np.linspace(0.0, 100.0, rows),
        "TrackY": np.linspace(0.0, 50.0, rows),
    })
    fp = Path(dirpath) / ("arena00_chunk%03d_%s.parquet" % (chunk_idx, SUFFIX))
    st.write_parquet_with_num_frames(
        df, fp, num_frames=n_frames, engine="pyarrow", compression="zstd")
    return fp


def write_state(dirpath, chunk_sec=CHUNK_SEC, fps=FPS):
    state = {
        "schema": 1,
        "chunking": {
            "chunk_sec": chunk_sec,
            "chunk_ext": "mkv",
            "total_rows": 10,
            "videos": {
                "cam01_cam0_2099-01-01-00-00-00": {
                    "n_chunks": 10, "fps": fps, "frame_count": 432000},
            },
        },
        "detection": {},
        "waves": [],
    }
    fp = Path(dirpath) / "PIPELINE_STATE.json"
    with open(fp, "w") as f:
        json.dump(state, f)
    return fp


class Work(object):
    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="stitchtest_")
        self.chunks = Path(self.root) / "chunks"
        self.out = Path(self.root) / "out"
        self.chunks.mkdir()
        self.out.mkdir()

    def stitch(self, idxs, chunk_frames=None):
        files = [write_chunk_parquet(self.chunks, i) for i in idxs]
        per_track = self.out / "per_track"
        per_track.mkdir(exist_ok=True)
        st.stitch_group(
            files, SUFFIX, per_track, ["Frame", "TrackID", "TrackX", "TrackY"],
            fps=FPS, chunk_frames=chunk_frames)
        return pd.read_parquet(st.stitched_parquet_path(per_track, TRACK_ID, SUFFIX))

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_parse_chunk_index_reads_the_token():
    assert st.parse_chunk_index(Path("arena00_chunk005_left.parquet")) == 5
    assert st.parse_chunk_index(Path("chunk000_right.parquet")) == 0
    assert st.parse_chunk_index(Path("arena00_chunk197_left.parquet")) == 197
    assert st.parse_chunk_index(Path("no_chunk_token_here.parquet")) is None


def test_chunk_frames_from_state():
    w = Work()
    try:
        fp = write_state(w.root)
        frames, fps = st.chunk_frames_from_state(fp, fallback_fps=1.0)
        assert frames == 43200
        assert fps == 24.0
    finally:
        w.close()


def test_chunk_frames_from_state_rejects_a_contract_without_chunking():
    w = Work()
    try:
        fp = Path(w.root) / "bad.json"
        with open(fp, "w") as f:
            json.dump({"schema": 1, "chunking": {}}, f)
        try:
            st.chunk_frames_from_state(fp, fallback_fps=24.0)
        except RuntimeError:
            return
        raise AssertionError("accepted a contract with no chunk_sec")
    finally:
        w.close()


# ---------------------------------------------------------------------------
# offsets -- the regression
# ---------------------------------------------------------------------------
def test_subset_keeps_block_frame_numbering():
    """Chunks 5-6 alone must land at 216000, not at 0."""
    w = Work()
    try:
        out = w.stitch([5, 6], chunk_frames=CHUNK_FRAMES)
        starts = sorted(out.groupby("source_file")["Frame"].min())
        assert starts == [5 * CHUNK_FRAMES, 6 * CHUNK_FRAMES], starts
    finally:
        w.close()


def test_subset_without_an_anchor_renumbers_from_zero():
    """Pins the old behaviour, so the reason for the anchor stays visible.

    Note what makes it hard to spot: the chunks keep their correct spacing
    (43200 apart), so the track looks perfectly well formed. The whole subset is
    simply translated to the front of the block -- chunk 5 claims frame 0
    instead of 216000.
    """
    w = Work()
    try:
        out = w.stitch([5, 6], chunk_frames=None)
        starts = sorted(out.groupby("source_file")["Frame"].min())
        assert starts == [0, CHUNK_FRAMES], starts
    finally:
        w.close()


def test_full_stitch_is_unchanged_by_the_anchor():
    """Backward compatibility: with every chunk present, both agree."""
    w = Work()
    try:
        anchored = w.stitch([0, 1, 2], chunk_frames=CHUNK_FRAMES)
        starts = sorted(anchored.groupby("source_file")["Frame"].min())
        assert starts == [0, CHUNK_FRAMES, 2 * CHUNK_FRAMES]
    finally:
        w.close()


def test_a_gap_does_not_shift_later_chunks():
    """The anchor also fixes a defect the old code had on complete-looking runs.

    With chunk 1 absent, the running offset packed chunk 2 into chunk 1's slot.
    """
    w = Work()
    try:
        out = w.stitch([0, 2], chunk_frames=CHUNK_FRAMES)
        starts = sorted(out.groupby("source_file")["Frame"].min())
        assert starts == [0, 2 * CHUNK_FRAMES], starts
    finally:
        w.close()


def test_num_frames_metadata_spans_the_block_not_the_subset():
    w = Work()
    try:
        w.stitch([5, 6], chunk_frames=CHUNK_FRAMES)
        fp = st.stitched_parquet_path(w.out / "per_track", TRACK_ID, SUFFIX)
        assert st.parquet_num_frames(fp) == 7 * CHUNK_FRAMES
    finally:
        w.close()


# ---------------------------------------------------------------------------
# refusal
# ---------------------------------------------------------------------------
def test_main_refuses_a_non_contiguous_selection_without_an_anchor():
    """Chunks 5-6 alone: not 0-based, so their placement cannot be inferred."""
    w = Work()
    try:
        for i in (5, 6):
            write_chunk_parquet(w.chunks, i)
        try:
            st.main(w.chunks, w.out, ["Frame", "TrackID"], fps=FPS,
                    string=".parquet", chunks_filter={"005", "006"})
        except RuntimeError as e:
            assert "absolute frame anchor" in str(e), e
            return
        raise AssertionError("stitched a non-contiguous selection with no anchor")
    finally:
        w.close()


def test_main_refuses_a_hole_without_an_anchor():
    """A chunk excluded upstream as incomplete used to slide 2 into 1's slot."""
    w = Work()
    try:
        for i in (0, 2):
            write_chunk_parquet(w.chunks, i)
        try:
            st.main(w.chunks, w.out, ["Frame", "TrackID"], fps=FPS,
                    string=".parquet", chunks_filter={"000", "002"})
        except RuntimeError as e:
            assert "absolute frame anchor" in str(e), e
            return
        raise AssertionError("stitched across a hole with no anchor")
    finally:
        w.close()


def test_main_still_accepts_a_full_block_without_an_anchor():
    """Backward compatibility: every production stitch today passes chunks=all."""
    w = Work()
    try:
        for i in (0, 1, 2):
            write_chunk_parquet(w.chunks, i)
        st.main(w.chunks, w.out, ["Frame", "TrackID", "TrackX", "TrackY"], fps=FPS,
                string=".parquet", chunks_filter={"000", "001", "002"},
                write_track_pngs=False)
        out = pd.read_parquet(
            st.stitched_parquet_path(w.out / "per_track", TRACK_ID, SUFFIX))
        assert sorted(out.groupby("source_file")["Frame"].min()) == [
            0, CHUNK_FRAMES, 2 * CHUNK_FRAMES]
    finally:
        w.close()


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            sys.stdout.write(".")
        except Exception as e:  # a test runner reports everything
            failed.append((name, e))
            sys.stdout.write("F")
        sys.stdout.flush()
    sys.stdout.write("\n")
    for name, e in failed:
        sys.stdout.write("FAIL %s: %s: %s\n" % (name, type(e).__name__, e))
    sys.stdout.write("%d passed, %d failed (of %d)\n"
                     % (len(tests) - len(failed), len(failed), len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
