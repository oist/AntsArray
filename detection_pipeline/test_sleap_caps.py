"""Per-camera SLEAP instance caps: nest cameras get the large cap, the rest the small one.

scripts/sleap_caps.py turns a nest-camera table (rows: effective_from JST, cameras, note)
into one cap per (video, chunk), using each chunk's wall-clock span.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import sleap_caps  # noqa: E402

JST_0701 = 1782831600000      # 2026-07-01 00:00 JST in epoch ms
HOUR = 3600 * 1000
CHUNK_SEC = 1800


def ms(text):
    return sleap_caps.parse_when(text)


def write_table(path, rows):
    lines = ["# nest cameras", "effective_from\tnest_cams\tnote"]
    lines += ["%s\t%s\t%s" % r for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def make_block(tmp_path, start_ms, cams=("cam01", "cam11"), n_chunks=4, frame_offset=0):
    """Manifest + sidecars for videos that all start at start_ms."""
    rows = ["vname,source_path,ext,fps,frame_count,duration_sec,n_chunks"]
    for cam in cams:
        vname = "%s_cam0_2026-09-15-15-52-47" % cam
        video = tmp_path / (vname + ".mkv")
        video.write_bytes(b"")
        side = {"provenance": {"firstEncodedFrame": {"hostEpochMs": start_ms}}}
        if frame_offset:
            side["derived"] = {"frameOffset": frame_offset}
        Path(str(video) + ".diag.json").write_text(json.dumps(side))
        rows.append("%s,%s,.mkv,24.0,%d,%d,%d" % (vname, video.as_posix(), n_chunks * 43200,
                                                   n_chunks * CHUNK_SEC, n_chunks))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows) + "\n")
    return manifest


def write_worklist(path, manifest, chunks):
    vnames = [line.split(",")[0] for line in manifest.read_text().splitlines()[1:]]
    path.write_text("".join("%s\t%03d\t43200\n" % (v, c) for v in vnames for c in chunks))
    return path


def caps_by_cam(rows):
    return {(v.split("_")[0], c): cap for v, c, cap in rows}


def test_parse_when_is_jst():
    assert ms("2026-07-01 00:00") == JST_0701
    assert ms("2026-07-01") == JST_0701
    assert ms("2026-07-01T09:30") == JST_0701 + 9.5 * HOUR


def test_table_rejects_bad_rows(tmp_path):
    with pytest.raises(ValueError, match="camera"):
        sleap_caps.load_table(write_table(tmp_path / "t.tsv", [("2026-07-01", "cam1,cam09", "")]))
    with pytest.raises(ValueError, match="time"):
        sleap_caps.load_table(write_table(tmp_path / "t.tsv", [("July 1", "cam01", "")]))


def test_nest_cameras_get_the_nest_cap(tmp_path):
    table = sleap_caps.load_table(write_table(tmp_path / "t.tsv", [("2026-07-01 00:00", "cam01,cam09", "A")]))
    manifest = make_block(tmp_path, JST_0701 + 24 * HOUR)
    wl = write_worklist(tmp_path / "wl.txt", manifest, [0, 1])

    rows = sleap_caps.chunk_caps(wl, manifest, CHUNK_SEC, table, nest_cap=96, other_cap=20)

    assert caps_by_cam(rows) == {("cam01", "000"): 96, ("cam01", "001"): 96,
                                 ("cam11", "000"): 20, ("cam11", "001"): 20}


def test_the_latest_row_at_the_chunk_applies_and_an_overlap_counts(tmp_path):
    # cam11 becomes a nest camera 45 min into the recording: chunk 000 (0-30 min) stays
    # small, chunk 001 (30-60 min) overlaps the change, chunk 002 is fully after it.
    start = JST_0701 + 24 * HOUR
    change = sleap_caps.format_when(start + 45 * 60 * 1000)
    table = sleap_caps.load_table(write_table(tmp_path / "t.tsv", [
        ("2026-07-01", "cam01", "A"), (change, "cam01,cam11", "new nest")]))
    manifest = make_block(tmp_path, start)
    wl = write_worklist(tmp_path / "wl.txt", manifest, [0, 1, 2])

    got = caps_by_cam(sleap_caps.chunk_caps(wl, manifest, CHUNK_SEC, table, 96, 20))

    assert [got[("cam11", c)] for c in ("000", "001", "002")] == [20, 96, 96]
    assert [got[("cam01", c)] for c in ("000", "001", "002")] == [96, 96, 96]


def test_a_split_tail_starts_at_its_frame_offset(tmp_path):
    # 0916/block02-style tail: the sidecar's first frame is the ORIGINAL start.
    start = JST_0701 + 24 * HOUR
    table = sleap_caps.load_table(write_table(tmp_path / "t.tsv", [
        ("2026-07-01", "cam01", "A"), (sleap_caps.format_when(start + 10 * HOUR), "cam11", "B")]))
    manifest = make_block(tmp_path, start, frame_offset=24 * 3600 * 20)   # tail begins 20 h in
    wl = write_worklist(tmp_path / "wl.txt", manifest, [0])

    got = caps_by_cam(sleap_caps.chunk_caps(wl, manifest, CHUNK_SEC, table, 96, 20))

    assert got == {("cam01", "000"): 20, ("cam11", "000"): 96}


def test_a_table_saved_on_windows_with_a_non_ascii_note_loads(tmp_path):
    # BOM + CRLF + a non-ASCII note: deigo's Python 3.6 decodes by locale (ASCII under C).
    note = "new nest – Fluon-coated, µ-film"
    path = tmp_path / "t.tsv"
    path.write_bytes(("﻿effective_from\tnest_cams\tnote\r\n"
                      "2026-07-01 00:00\tcam01,cam09\t%s\r\n" % note).encode("utf-8"))

    rows = sleap_caps.load_table(path)

    assert rows == [(JST_0701, frozenset({"cam01", "cam09"}), note)]


def test_a_video_not_named_camNN_is_refused(tmp_path):
    # Its chunks would otherwise never match a nest row and silently get the small cap.
    table = sleap_caps.load_table(write_table(tmp_path / "t.tsv", [("2026-07-01", "cam01", "A")]))
    manifest = make_block(tmp_path, JST_0701 + 24 * HOUR, cams=("Cam01",))
    wl = write_worklist(tmp_path / "wl.txt", manifest, [0])

    with pytest.raises(ValueError, match="camNN"):
        sleap_caps.chunk_caps(wl, manifest, CHUNK_SEC, table, 96, 20)


def test_a_nest_starting_just_after_the_chunk_still_counts(tmp_path):
    # Chunk boundaries can sit a GOP (60 s) off nominal and host clocks differ by seconds,
    # so a nest that starts 30 s after the computed chunk end still gets the large cap.
    start = JST_0701 + 24 * HOUR
    table = sleap_caps.load_table(write_table(tmp_path / "t.tsv", [
        ("2026-07-01", "cam01", "A"),
        (sleap_caps.format_when(start + CHUNK_SEC * 1000 + 60 * 1000), "cam01,cam11", "new nest")]))
    manifest = make_block(tmp_path, start)
    wl = write_worklist(tmp_path / "wl.txt", manifest, [0])

    got = caps_by_cam(sleap_caps.chunk_caps(wl, manifest, CHUNK_SEC, table, 96, 20))

    assert got[("cam11", "000")] == 96


def test_a_nest_removed_just_before_the_chunk_still_counts(tmp_path):
    start = JST_0701 + 24 * HOUR
    table = sleap_caps.load_table(write_table(tmp_path / "t.tsv", [
        ("2026-07-01", "cam01,cam11", "A"),
        (sleap_caps.format_when(start + CHUNK_SEC * 1000 - 60 * 1000), "cam01", "nest removed")]))
    manifest = make_block(tmp_path, start)
    wl = write_worklist(tmp_path / "wl.txt", manifest, [1, 2])

    got = caps_by_cam(sleap_caps.chunk_caps(wl, manifest, CHUNK_SEC, table, 96, 20))

    assert [got[("cam11", c)] for c in ("001", "002")] == [96, 20]


def test_a_missing_sidecar_names_the_file(tmp_path):
    table = sleap_caps.load_table(write_table(tmp_path / "t.tsv", [("2026-07-01", "cam01", "A")]))
    manifest = make_block(tmp_path, JST_0701 + 24 * HOUR, cams=("cam01",))
    next(tmp_path.glob("*.diag.json")).unlink()

    with pytest.raises(OSError, match="diag.json"):
        sleap_caps.describe(manifest, CHUNK_SEC, table, 96, 20)


def test_a_chunk_before_every_row_is_refused(tmp_path):
    table = sleap_caps.load_table(write_table(tmp_path / "t.tsv", [("2026-07-01", "cam01", "A")]))
    manifest = make_block(tmp_path, JST_0701 - 24 * HOUR)
    wl = write_worklist(tmp_path / "wl.txt", manifest, [0])

    with pytest.raises(ValueError, match="no nest-camera row"):
        sleap_caps.chunk_caps(wl, manifest, CHUNK_SEC, table, 96, 20)


def test_describe_names_every_row_the_block_spans(tmp_path):
    start = JST_0701 + 24 * HOUR
    table = sleap_caps.load_table(write_table(tmp_path / "t.tsv", [
        ("2026-06-01", "cam05", "May layout"), ("2026-07-01", "cam01,cam09", "A"),
        (sleap_caps.format_when(start + HOUR), "cam01,cam09,cam20", "new nest")]))
    manifest = make_block(tmp_path, start, n_chunks=4)

    text = sleap_caps.describe(manifest, CHUNK_SEC, table, 96, 20)

    assert text == ("nest96=cam01,cam09@2026-07-01T00:00;nest96=cam01,cam09,cam20@%s;other20"
                    % sleap_caps.format_when(start + HOUR).replace(" ", "T"))


def test_cli_writes_the_caps_table_and_uniform_mode_needs_no_table(tmp_path):
    manifest = make_block(tmp_path, JST_0701 + 24 * HOUR)
    wl = write_worklist(tmp_path / "wl.txt", manifest, [3])
    out = tmp_path / "caps.tsv"
    script = ROOT / "scripts" / "sleap_caps.py"

    r = subprocess.run([sys.executable, str(script), "table", "--worklist", str(wl), "--manifest", str(manifest),
                        "--chunk-sec", str(CHUNK_SEC), "--uniform", "96", "--out", str(out)],
                       capture_output=True, text=True)

    assert r.returncode == 0, r.stderr
    assert [line.split("\t")[1:] for line in out.read_text().splitlines()] == [["003", "96"], ["003", "96"]]
    assert "cap 96: 2 chunks" in r.stdout


def test_cli_refuses_a_cap_the_engine_cannot_build(tmp_path):
    manifest = make_block(tmp_path, JST_0701 + 24 * HOUR)
    table = write_table(tmp_path / "t.tsv", [("2026-07-01", "cam01", "A")])
    script = ROOT / "scripts" / "sleap_caps.py"

    r = subprocess.run([sys.executable, str(script), "describe", "--manifest", str(manifest),
                        "--chunk-sec", str(CHUNK_SEC), "--nest-cams", str(table), "--nest-cap", "200"],
                       capture_output=True, text=True)

    assert r.returncode == 2
    assert "192" in r.stderr
