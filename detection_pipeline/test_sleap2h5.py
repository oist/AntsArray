#!/usr/bin/env python3
"""Tests for the .slp -> _sleap_data.h5 conversion (scripts/sleap2h5.py).

Run with pytest, or directly under a python that has h5py + pandas (the
cluster's ant_tracking venv does; the login system python does not)::

    /apps/unit/ReiterU/ant_tracking/venv/bin/python detection_pipeline/test_sleap2h5.py

The case that matters is the EMPTY chunk: a .slp with zero instances must
convert to a valid zero-row h5 carrying expected_frames, because a missing
h5 reads as "never converted" everywhere downstream. It used to raise
ZeroDivisionError (cam12 chunk 197 of 20260810/block02), which stranded the
chunk and stopped the tracking preflight at that index.
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts"))

try:
    import h5py  # noqa: E402
    import numpy as np  # noqa: E402
    import sleap2csv  # noqa: E402
    import sleap2h5  # noqa: E402
    DEPS = True
except (ImportError, SyntaxError):  # pragma: no cover - laptop without h5py / py3.6
    DEPS = False


def _write_slp(path, frame_idx, frame_id, point_id_start, x, y, score, nodes):
    """Minimal .slp: only the groups/attrs import_slp + flatten_data read."""
    with h5py.File(path, "w") as f:
        f.attrs["tracksjson"] = json.dumps({"nodes": nodes})
        f.create_dataset("frames/frame_idx", data=np.asarray(frame_idx, dtype=np.int64))
        f.create_dataset("instances/frame_id", data=np.asarray(frame_id, dtype=np.int64))
        f.create_dataset("instances/instance_id", data=np.asarray(frame_id, dtype=np.int64))
        f.create_dataset("instances/point_id_start", data=np.asarray(point_id_start, dtype=np.int64))
        f.create_dataset("pred_points/x", data=np.asarray(x, dtype=np.float64))
        f.create_dataset("pred_points/y", data=np.asarray(y, dtype=np.float64))
        f.create_dataset("pred_points/score", data=np.asarray(score, dtype=np.float64))


def test_empty_slp_converts_to_a_valid_zero_row_h5():
    if not DEPS:
        print("SKIP(no h5py)", end=" ")
        return
    d = tempfile.mkdtemp(prefix="s2h5_")
    try:
        slp = os.path.join(d, "cam12_cam3_2099-01-01-00-00-00_197.slp")
        # nodes absent as well: that is what made the old code divide by zero.
        _write_slp(slp, [], [], [], [], [], [], nodes=[])
        out = sleap2h5.slp2h5(slp, d, expected_frames=8640)
        assert os.path.basename(str(out)) == "cam12_cam3_2099-01-01-00-00-00_197_sleap_data.h5"
        with h5py.File(str(out), "r") as f:
            ds = f["sleap_data"]
            assert ds.shape == (0,)
            assert set(ds.dtype.names) == {"Frame", "Instance", "Bodypoint", "X", "Y", "Score_node"}
            assert int(f.attrs["expected_frames"]) == 8640
            assert int(f.attrs["instance_count"]) == 0
            assert int(f.attrs["frame_count"]) == 0
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_flatten_empty_matches_nonempty_schema():
    if not DEPS:
        print("SKIP(no h5py)", end=" ")
        return
    d = tempfile.mkdtemp(prefix="s2h5_")
    try:
        slp = os.path.join(d, "cam01_cam0_2099-01-01-00-00-00_000.slp")
        # one instance on frame 5 with two nodes -> two rows
        _write_slp(slp, [5], [0], [0], [1.0, 2.0], [3.0, 4.0], [0.9, 0.8], nodes=["a", "b"])
        full = sleap2csv.flatten_data(sleap2csv.import_slp(pathlib.Path(slp)))
        empty = sleap2csv._empty_flat()
        assert list(full.columns) == list(empty.columns)
        assert [str(t) for t in full.dtypes] == [str(t) for t in empty.dtypes]
        assert full["Frame"].tolist() == [5, 5]
        assert full["Bodypoint"].tolist() == [0, 1]
        # and the non-empty path still writes the chunked/compressed dataset
        out = sleap2h5.slp2h5(slp, d, expected_frames=43200)
        with h5py.File(str(out), "r") as f:
            assert f["sleap_data"].shape == (2,)
            assert f["sleap_data"].compression == "gzip"
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
