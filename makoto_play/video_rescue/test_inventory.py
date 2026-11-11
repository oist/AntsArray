"""Tests for inventory.py session naming (run: pytest video_rescue/test_inventory.py)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INVENTORY = os.path.join(HERE, "inventory.py")


def make_source(tmp_path, starts):
    """Create recorder-named stub videos for each start timestamp (one file per camera)."""
    src = tmp_path / "src"
    src.mkdir()
    for ts in starts:
        for cam in ("cam01_cam0", "cam02_cam1"):
            (src / ("%s_%s.mkv" % (cam, ts))).write_bytes(b"x" * 16)
    return src


def run_inventory(src, out, *extra):
    return subprocess.run(
        [sys.executable, INVENTORY, "--src", str(src), "--out-dir", str(out),
         "--min-video-bytes", "1"] + list(extra),
        capture_output=True, text=True)


def read_dest_rels(out):
    lines = (out / "sessions.tsv").read_text().splitlines()[1:]
    return [line.split("\t")[1] for line in lines]


def test_single_session_defaults_to_block01(tmp_path):
    src = make_source(tmp_path, ["2026-08-10-18-43-25"])
    out = tmp_path / "work"

    result = run_inventory(src, out)

    assert result.returncode == 0, result.stderr
    assert read_dest_rels(out) == ["20260810/block01"]


def test_dest_rel_override_renames_the_only_session(tmp_path):
    src = make_source(tmp_path, ["2026-08-10-18-43-25"])
    out = tmp_path / "work"

    result = run_inventory(src, out, "--dest-rel", "20260810/block02")

    assert result.returncode == 0, result.stderr
    assert read_dest_rels(out) == ["20260810/block02"]
    manifest = (out / "manifest.tsv").read_text()
    assert "20260810/block02" in manifest
    assert "20260810/block01" not in manifest


def test_dest_rel_override_refuses_multiple_sessions(tmp_path):
    src = make_source(tmp_path, ["2026-08-10-18-43-25", "2026-08-11-09-00-00"])
    out = tmp_path / "work"

    result = run_inventory(src, out, "--dest-rel", "20260810/block02")

    assert result.returncode != 0
    assert "[ERR] --dest-rel" in result.stderr
    assert "2 were found" in result.stderr
    assert not (out / "sessions.tsv").exists()


def test_dest_rel_override_rejects_malformed_value(tmp_path):
    src = make_source(tmp_path, ["2026-08-10-18-43-25"])
    out = tmp_path / "work"

    result = run_inventory(src, out, "--dest-rel", "block02")

    assert result.returncode != 0
    assert "[ERR] --dest-rel must look like" in result.stderr
    assert not (out / "sessions.tsv").exists()
