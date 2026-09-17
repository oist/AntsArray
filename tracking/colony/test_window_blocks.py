#!/usr/bin/env python3
"""Tests for chunk-window views of a block.

Run with pytest, or directly (the cluster pythons ship no pytest)::

    python3 tracking/colony/test_window_blocks.py

Two things are pinned here:

``discover_complete_input_chunks`` must accept a data/ that starts at a chunk
other than 000 (a window view of a partially processed block) and still stop
at the first hole -- and a real block starting at 000 must behave exactly as
before.

``make_window_block.py`` must build a view that is byte-for-byte what the
tracking pipeline expects: original file names, relative links, a copied
contract, and a refusal to materialize a window whose detection is not done.
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "detection_pipeline", "lib"))

import make_window_block as mwb  # noqa: E402
import pipeline_state as ps  # noqa: E402

try:
    import panorama_io  # noqa: E402  (needs pandas + python >= 3.7)
except (ImportError, SyntaxError):  # pragma: no cover
    panorama_io = None

VID = ["cam%02d_cam%d_2099-01-01-00-00-00" % (i, i % 8) for i in (1, 2, 3)]
SUFFIXES = ("_aruco_tracks.h5", "_aruco_detections.h5", "_sleap_data.h5", ".slp")


def _symlinks_supported():
    d = tempfile.mkdtemp(prefix="wbtest_")
    try:
        open(os.path.join(d, "t"), "w").close()
        os.symlink("t", os.path.join(d, "l"))
        return True
    except OSError:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


SYMLINKS = _symlinks_supported()


class Block:
    """A temp <date>/blockNN with a contract, raw-video stand-ins and outputs."""

    def __init__(self, n_chunks=10, done=(), cams=VID):
        self.date = tempfile.mkdtemp(prefix="wbtest_date_")
        self.root = os.path.join(self.date, "block02")
        self.data = os.path.join(self.root, "data")
        os.makedirs(self.data)
        for v in cams:
            open(os.path.join(self.root, v + ".mkv"), "w").close()
            open(os.path.join(self.root, v + ".json"), "w").close()
        open(os.path.join(self.root, "sess_20990101_000000.txt"), "w").close()
        videos = dict((v, {"n_chunks": n_chunks, "fps": 24.0, "frame_count": n_chunks * 43200})
                      for v in cams)
        ps.write(self.data, ps.new_state(self.root, 1800, "mkv", videos, {}))
        self.touch(done, cams)

    def touch(self, indices, cams=VID, suffixes=SUFFIXES):
        for v in cams:
            for i in indices:
                for s in suffixes:
                    open(os.path.join(self.data, "%s_%03d%s" % (v, i, s)), "w").close()

    def close(self):
        shutil.rmtree(self.date, ignore_errors=True)


def raises(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), "expected %r in %r" % (needle, str(e))
        return
    raise AssertionError("expected ValueError mentioning %r" % needle)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def test_discovery_still_starts_at_000_for_a_real_block():
    if panorama_io is None:
        print("SKIP(no pandas/py>=3.7)", end=" ")
        return
    b = Block(n_chunks=6, done=range(0, 4))
    try:
        from pathlib import Path
        chunks, summary = panorama_io.discover_complete_input_chunks(Path(b.data))
        assert chunks == ["000", "001", "002", "003"]
        assert summary["first_chunk"] == "000"
        assert summary["first_incomplete"]["chunk"] == "004"
    finally:
        b.close()


def test_discovery_accepts_a_window_that_starts_past_000():
    if panorama_io is None:
        print("SKIP(no pandas/py>=3.7)", end=" ")
        return
    b = Block(n_chunks=200, done=range(149, 153))
    try:
        from pathlib import Path
        chunks, summary = panorama_io.discover_complete_input_chunks(Path(b.data))
        assert chunks == ["149", "150", "151", "152"]
        assert summary["first_chunk"] == "149"
        assert summary["reference_camera_count"] == len(VID)
    finally:
        b.close()


def test_discovery_stops_at_the_first_hole_inside_a_window():
    if panorama_io is None:
        print("SKIP(no pandas/py>=3.7)", end=" ")
        return
    b = Block(n_chunks=200, done=(149, 150, 152))     # 151 missing entirely
    try:
        from pathlib import Path
        chunks, summary = panorama_io.discover_complete_input_chunks(Path(b.data))
        assert chunks == ["149", "150"]
        assert summary["first_incomplete"] == {"chunk": "151", "reason": "missing chunk"}
    finally:
        b.close()


def test_discovery_rejects_a_camera_short_of_one_modality_in_the_first_chunk():
    if panorama_io is None:
        print("SKIP(no pandas/py>=3.7)", end=" ")
        return
    b = Block(n_chunks=200, done=range(149, 151))
    try:
        os.remove(os.path.join(b.data, "%s_149_sleap_data.h5" % VID[2]))
        from pathlib import Path
        chunks, summary = panorama_io.discover_complete_input_chunks(Path(b.data))
        assert chunks == []
        assert summary["first_incomplete"]["missing_sleap_cameras"] == [3]
    finally:
        b.close()


def test_non_000_start_is_refused_unless_the_dir_is_a_window_view():
    if panorama_io is None:
        print("SKIP(no pandas/py>=3.7)", end=" ")
        return
    from pathlib import Path
    b = Block(n_chunks=200, done=range(149, 152))
    try:
        data = Path(b.data)
        chunks, _ = panorama_io.discover_complete_input_chunks(data)
        # An ordinary block whose leading chunks are simply missing: hard stop.
        raised = False
        try:
            panorama_io.require_contiguous_from_start(chunks, data)
        except FileNotFoundError as e:
            raised = True
            assert "not a window view" in str(e)
        assert raised
        # Positive evidence 1: WINDOW.json beside data/ (what materialize writes).
        with open(os.path.join(b.root, mwb.WINDOW_MARKER), "w") as f:
            json.dump({"schema": 1, "chunk_range": [149, 151]}, f)
        assert panorama_io.require_contiguous_from_start(chunks, data) == "149"
        os.remove(os.path.join(b.root, mwb.WINDOW_MARKER))
        # Positive evidence 2: a "window" key in the (copied) contract.
        state = ps.load(b.data)
        state["window"] = {"chunk_range": [149, 151]}
        ps.write(b.data, state)
        assert panorama_io.require_contiguous_from_start(chunks, data) == "149"
        # And a genuine 000 start needs no evidence at all.
        assert panorama_io.require_contiguous_from_start(["000", "001"], Path(b.date)) == "000"
    finally:
        b.close()


# ---------------------------------------------------------------------------
# windows file
# ---------------------------------------------------------------------------
def test_init_validates_overlap_coverage_and_extent():
    b = Block(n_chunks=10)
    try:
        raises(lambda: mwb.parse_ranges("0-4,4-9"), "overlap")
        rc = mwb.main(["init", "--block", b.root, "--ranges", "0-4,6-9"])
        assert rc == 2                                   # gap at 5, no --allow-gaps
        rc = mwb.main(["init", "--block", b.root, "--ranges", "0-4,5-12"])
        assert rc == 2                                   # past the last index
        rc = mwb.main(["init", "--block", b.root, "--ranges", "5-9,0-4", "--group", ""])
        assert rc == 0                                   # unsorted input is fine
        doc = mwb.load_windows(b.root)
        assert [w["name"] for w in doc["windows"]] == ["w000-004", "w005-009"]
        assert doc["windows"][0]["hours"] == 2.5
        assert doc["n_chunks"] == 10 and doc["chunk_sec"] == 1800
        assert mwb.main(["init", "--block", b.root, "--ranges", "0-9"]) == 2  # exists
    finally:
        b.close()


def test_status_reports_completeness_per_gate_stage():
    b = Block(n_chunks=10, done=range(0, 5))
    try:
        assert mwb.main(["init", "--block", b.root, "--ranges", "0-4,5-9", "--group", ""]) == 0
        _t, _c, state = mwb.block_chunk_total(b.root)
        comp = mwb.window_completeness(b.root, 0, 4, state)
        assert comp["trk"][:2] == (15, 15) and comp["sdat"][:2] == (15, 15)
        comp = mwb.window_completeness(b.root, 5, 9, state)
        assert comp["trk"][:2] == (0, 15)
        assert comp["trk"][2] == list(range(5, 10))
    finally:
        b.close()


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------
def test_materialize_refuses_an_incomplete_window_without_force():
    b = Block(n_chunks=10, done=range(0, 3))
    try:
        assert mwb.main(["init", "--block", b.root, "--ranges", "0-4,5-9", "--group", ""]) == 0
        rc = mwb.main(["materialize", "--block", b.root, "--window", "w000-004", "--dry-run"])
        assert rc == 1                                   # SKIP: not complete
        assert not os.path.exists(mwb.view_dir_for(b.root, 0, 4))
    finally:
        b.close()


def test_materialize_builds_a_block_shaped_view_with_relative_links():
    if not SYMLINKS:
        print("SKIP(no symlink privilege)", end=" ")
        return
    b = Block(n_chunks=10, done=range(0, 10))
    try:
        assert mwb.main(["init", "--block", b.root, "--ranges", "0-4,5-9", "--group", ""]) == 0
        assert mwb.main(["materialize", "--block", b.root, "--window", "w005-009",
                         "--group", ""]) == 0
        view = mwb.view_dir_for(b.root, 5, 9)
        assert os.path.basename(view) == "block02-w005-009"

        # data/: only chunks 5..9, original names, relative links that resolve.
        names = sorted(os.listdir(os.path.join(view, "data")))
        linked = [n for n in names if n != ps.STATE_BASENAME]
        assert len(linked) == len(VID) * 5 * len(SUFFIXES)
        for n in linked:
            m = mwb._DATA_FILE_RE.match(n)
            assert m and 5 <= int(m.group("i")) <= 9, n
        one = os.path.join(view, "data", linked[0])
        assert os.path.islink(one)
        assert os.readlink(one) == os.path.join("..", "..", "block02", "data", linked[0])
        assert os.path.isfile(one)                       # resolves through the link

        # Top level: videos, sidecars and the conductor log are linked; dirs are not.
        assert os.path.islink(os.path.join(view, VID[0] + ".mkv"))
        assert os.path.islink(os.path.join(view, "sess_20990101_000000.txt"))
        assert not os.path.exists(os.path.join(view, "data", "data"))

        # The contract is a COPY carrying the window, not a link.
        sp = os.path.join(view, "data", ps.STATE_BASENAME)
        assert not os.path.islink(sp)
        with open(sp) as f:
            copied = json.load(f)
        assert copied["window"]["chunk_range"] == [5, 9]
        assert copied["chunking"]["chunk_sec"] == 1800     # tracking's frame anchor
        with open(os.path.join(view, mwb.WINDOW_MARKER)) as f:
            assert json.load(f)["n_data_links"] == len(linked)

        # Re-running without --force refuses; with --force it keeps existing links.
        assert mwb.main(["materialize", "--block", b.root, "--window", "w005-009",
                         "--group", ""]) == 1
        assert mwb.main(["materialize", "--block", b.root, "--window", "w005-009",
                         "--force", "--group", ""]) == 0

        # --force never REPLACES: a link that points elsewhere is an error, and
        # a real file in the way is never touched.
        os.remove(one)
        os.symlink(os.path.join("..", "..", "elsewhere", linked[0]), one)
        assert mwb.main(["materialize", "--block", b.root, "--window", "w005-009",
                         "--force", "--group", ""]) == 1
        assert os.readlink(one).endswith(os.path.join("elsewhere", linked[0]))
        os.remove(one)
        open(one, "w").close()                          # a regular file, not a link
        assert mwb.main(["materialize", "--block", b.root, "--window", "w005-009",
                         "--force", "--group", ""]) == 1
        assert os.path.isfile(one) and not os.path.islink(one)
        os.remove(one)
        assert mwb.main(["materialize", "--block", b.root, "--window", "w005-009",
                         "--force", "--group", ""]) == 0   # restored

        # The tracking preflight accepts the view and sees chunks 005..009 only.
        if panorama_io is not None:
            from pathlib import Path
            chunks, summary = panorama_io.discover_complete_input_chunks(Path(view) / "data")
            assert chunks == ["005", "006", "007", "008", "009"]
            assert summary["first_chunk"] == "005"
    finally:
        b.close()


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
