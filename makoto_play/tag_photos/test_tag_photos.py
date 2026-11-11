"""End-to-end tests for tag_photos.py: scan an inbox of photos, review, apply to the colony table."""
import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import colony_tables as T  # noqa: E402
import photo_decode as P  # noqa: E402
import synth_scene as S  # noqa: E402
import tag_photos as TP  # noqa: E402

NPZ = P.DEFAULT_NPZ
HEADER = "colony,tag_id,role,paint,tagged_on,ended_on,end_reason,evidence,note"
ROWS = ["21,1,worker,Y,,,,note 2026-09-28 (Y73),",
        "22,1,worker,Y,,,,note 2026-09-25 (Y75),",
        "22,13,worker,B,2026-10-01,,,user 2026-10-01,newborn",
        "22,40,worker,Y,,,,note 2026-09-25 (Y75),"]
PHOTO = "C22_dead/IMG_0001.JPEG"


def save_photo(path, gray, taken=None, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.dstack([gray] * 3))
    if taken is None:
        img.save(path)
        return
    exif = Image.Exif()
    exif[306] = taken
    exif.get_ifd(0x8769)[36867] = taken
    img.save(path, quality=quality, exif=exif)


@pytest.fixture(scope="module")
def scene():
    dic = P.load_decoder(NPZ).dictionary
    return S.scene_with_tags(dic, [(1, (400, 400), 90, 15, 0.0), (13, (1100, 600), 90, 60, 0.05)])


def make_colonies(tmp_path, scene, rows=ROWS):
    col = tmp_path / "_colonies"
    (col / "evidence").mkdir(parents=True)
    (col / "colonies.csv").write_text("colony,gamergate_tag,collected_on,census_on,census,note\n"
                                      "21,0,,,,\n22,0,,,,\n")
    (col / "tag_assignments.csv").write_bytes(("\r\n".join([HEADER] + rows) + "\r\n").encode())
    save_photo(col / "evidence" / "inbox" / PHOTO, scene, "2026:10:20 09:15:00")
    return col


@pytest.fixture
def colonies(tmp_path, scene):
    col = make_colonies(tmp_path, scene)
    inbox = col / "evidence" / "inbox"
    save_photo(inbox / "misc" / "IMG_0002.JPEG", scene, "2026:10:20 09:16:00")
    save_photo(inbox / "C99_dead" / "IMG_0003.JPEG", scene, "2026:10:20 09:17:00")
    return col


def inbox(col):
    return col / "evidence" / "inbox"


def scan(col):
    assert TP.main(["scan", "--colonies", str(col)]) == 0
    found = sorted((inbox(col) / "_review").glob("*/review.csv"))
    return found[-1]


def read_rows(path, encoding="utf-8-sig"):
    with open(path, newline="", encoding=encoding) as fh:
        return list(csv.DictReader(fh))


def write_rows(path, rows, encoding="utf-8-sig"):
    with open(path, "w", newline="", encoding=encoding) as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def edit_review(rev, **by_tag):
    """by_tag = {"t13": {"accept": "n"}, ...} -> rewrite review.csv."""
    rows = read_rows(rev)
    for r in rows:
        r.update(by_tag.get(f"t{r['tag_id']}", {}))
    write_rows(rev, rows)


def table_rows(col):
    return {(r["colony"], r["tag_id"]): r for r in read_rows(col / "tag_assignments.csv")}


def apply(rev, yes=True):
    return TP.main(["apply", "--review", str(rev)] + (["--yes"] if yes else []))


def test_scan_then_apply(colonies):
    rev = scan(colonies)
    rows = read_rows(rev)
    assert sorted(int(r["tag_id"]) for r in rows) == [1, 13]
    assert all(r["accept"] == "y" and r["colony"] == "22" and r["kind"] == "dead" for r in rows)
    assert all(r["photo"] == PHOTO and r["taken"].startswith("2026-10-20") for r in rows)
    assert (rev.parent / "review.html").read_text(encoding="utf-8").count("IMG_0001") >= 2
    assert len(list((rev.parent / "crops").iterdir())) >= 6
    log = next(rev.parent.glob("tag_photos_scan_log_*.txt")).read_text(encoding="utf-8")
    assert "misc" in log and "C99_dead" in log

    before = (colonies / "tag_assignments.csv").read_bytes()
    assert apply(rev, yes=False) == 0
    assert (colonies / "tag_assignments.csv").read_bytes() == before

    assert apply(rev) == 0
    t = table_rows(colonies)
    for key in (("22", "1"), ("22", "13")):
        assert (t[key]["ended_on"], t[key]["end_reason"]) == ("2026-10-20", "found_dead")
        assert t[key]["evidence"].endswith("; photo Colony22_founddead_IMG_0001.JPEG (tag_photos)")
    assert t[("21", "1")]["ended_on"] == "" and t[("22", "40")]["ended_on"] == ""
    after = (colonies / "tag_assignments.csv").read_bytes().decode()
    assert "21,1,worker,Y,,,,note 2026-09-28 (Y73),\r\n" in after          # untouched rows keep CRLF
    assert (rev.parent / "tag_assignments.before.csv").read_bytes() == before
    assert (colonies / "evidence" / "Colony22_founddead_IMG_0001.JPEG").exists()
    assert not (inbox(colonies) / PHOTO).exists()
    assert list(rev.parent.glob("tag_photos_apply_log_*.txt"))


def test_check_photos_are_reviewed_but_never_written(tmp_path, scene):
    col = make_colonies(tmp_path, scene)
    (inbox(col) / PHOTO).unlink()
    check = inbox(col) / "C22_check" / "IMG_0005.JPEG"
    save_photo(check, scene, "2026:10:20 15:00:00")
    rev = scan(col)
    assert {r["photo"] for r in read_rows(rev)} == {"C22_check/IMG_0005.JPEG"}
    before = (col / "tag_assignments.csv").read_bytes()
    assert apply(rev) == 0
    assert (col / "tag_assignments.csv").read_bytes() == before
    assert check.exists()
    log = next(rev.parent.glob("tag_photos_apply_log_*.txt")).read_text(encoding="utf-8")
    assert "[CHECK] C22_check/IMG_0005.JPEG" in log


def test_rejected_rows_and_id_override(colonies):
    rev = scan(colonies)
    rows = read_rows(rev)
    for r in rows:
        if r["tag_id"] == "13":
            r["accept"] = "n"
        else:
            r["accept"], r["id_override"], r["note"] = "", "40", "\uff21\uff22"
    write_rows(rev, rows, encoding="cp932")                            # as Excel saves it
    assert apply(rev) == 0
    t = table_rows(colonies)
    assert t[("22", "40")]["ended_on"] == "2026-10-20"
    assert t[("22", "1")]["ended_on"] == "" and t[("22", "13")]["ended_on"] == ""


def test_missed_tag_added_by_hand(colonies):
    rev = scan(colonies)
    rows = read_rows(rev)
    rows.append({**{k: "" for k in rows[0]}, "photo": PHOTO, "n": "added", "id_override": "40"})
    write_rows(rev, rows)
    assert apply(rev) == 0
    t = table_rows(colonies)
    assert [t[("22", i)]["ended_on"] for i in ("1", "13", "40")] == ["2026-10-20"] * 3
    assert not (inbox(colonies) / PHOTO).exists()


def test_evidence_name_taken_by_another_photo(colonies):
    rev = scan(colonies)
    other = colonies / "evidence" / "Colony22_founddead_IMG_0001.JPEG"
    other.write_bytes(b"an older, different photo")
    assert apply(rev) == 0
    t = table_rows(colonies)
    assert "Colony22_founddead_IMG_0001_20261020.JPEG" in t[("22", "1")]["evidence"]
    assert (colonies / "evidence" / "Colony22_founddead_IMG_0001_20261020.JPEG").exists()
    assert other.read_bytes() == b"an older, different photo"


def test_photo_replaced_after_scan_is_skipped(colonies, scene):
    rev = scan(colonies)
    save_photo(inbox(colonies) / PHOTO, scene, "2026:10:21 10:00:00", quality=80)
    before = (colonies / "tag_assignments.csv").read_bytes()
    assert apply(rev) == 1
    assert (colonies / "tag_assignments.csv").read_bytes() == before
    log = next(rev.parent.glob("tag_photos_apply_log_*.txt")).read_text(encoding="utf-8")
    assert "changed since the scan" in log


def test_photo_without_exif_needs_date_override(tmp_path, scene):
    col = make_colonies(tmp_path, scene)
    (inbox(col) / PHOTO).unlink()
    save_photo(inbox(col) / "C22_dead" / "IMG_0009.png", scene)        # no EXIF
    rev = scan(col)
    assert apply(rev) == 1
    assert table_rows(col)[("22", "1")]["ended_on"] == ""
    edit_review(rev, t1={"date_override": "2026/10/19"}, t13={"date_override": "2026-10-19"})
    assert apply(rev) == 0
    assert table_rows(col)[("22", "1")]["ended_on"] == "2026-10-19"


def test_unknown_accept_value_is_reported(colonies):
    rev = scan(colonies)
    edit_review(rev, t1={"accept": "o"})
    assert apply(rev) == 1
    t = table_rows(colonies)
    assert t[("22", "1")]["ended_on"] == "" and t[("22", "13")]["ended_on"] == "2026-10-20"
    assert (inbox(colonies) / PHOTO).exists()                          # row 1 unresolved: photo stays


def test_broken_photo_is_listed_and_the_rest_reviewed(colonies):
    (inbox(colonies) / "C22_dead" / "IMG_bad.JPEG").write_bytes(b"")
    rev = scan(colonies)
    assert len(read_rows(rev)) == 2
    meta = (rev.parent / "scan.json").read_text(encoding="utf-8")
    assert "IMG_bad.JPEG" in meta
    assert "could not be read" in (rev.parent / "review.html").read_text(encoding="utf-8")


def test_table_open_in_excel_undoes_the_copies(colonies, monkeypatch):
    rev = scan(colonies)
    before = (colonies / "tag_assignments.csv").read_bytes()

    def locked(*_):
        raise PermissionError("locked")
    monkeypatch.setattr(T.os, "replace", locked)
    assert apply(rev) == 2
    assert (colonies / "tag_assignments.csv").read_bytes() == before
    assert not list((colonies / "evidence").glob("Colony22_*"))
    assert not (colonies / "tag_assignments.csv.tmp").exists()
    assert (inbox(colonies) / PHOTO).exists()


def test_photo_with_an_unresolved_tag_stays_in_the_inbox(tmp_path, scene):
    col = make_colonies(tmp_path, scene, rows=[r for r in ROWS if not r.startswith("22,13,")])
    rev = scan(col)
    assert apply(rev) == 1                                             # tag 13 has no row
    assert table_rows(col)[("22", "1")]["ended_on"] == "2026-10-20"
    assert (col / "evidence" / "Colony22_founddead_IMG_0001.JPEG").exists()
    assert (inbox(col) / PHOTO).exists()


def test_failed_copy_leaves_no_partial_file(colonies, monkeypatch):
    rev = scan(colonies)
    before = (colonies / "tag_assignments.csv").read_bytes()

    def half_copy(src, dst):
        Path(dst).write_bytes(Path(src).read_bytes()[:100])
        raise OSError("network drive went away")
    monkeypatch.setattr(TP.shutil, "copy2", half_copy)
    assert apply(rev) == 2
    assert not list((colonies / "evidence").glob("Colony22_*"))
    assert (colonies / "tag_assignments.csv").read_bytes() == before


def test_any_error_before_the_table_is_written_undoes_the_copies(colonies, monkeypatch):
    rev = scan(colonies)

    def broken(*_):
        raise ValueError("unexpected")
    monkeypatch.setattr(T, "write_table", broken)
    assert apply(rev) == 2
    assert not list((colonies / "evidence").glob("Colony22_*"))
    assert (inbox(colonies) / PHOTO).exists()


def test_undecided_row_keeps_the_photo_in_the_inbox(colonies):
    rev = scan(colonies)
    edit_review(rev, t13={"accept": ""})
    assert apply(rev) == 1
    t = table_rows(colonies)
    assert t[("22", "1")]["ended_on"] == "2026-10-20" and t[("22", "13")]["ended_on"] == ""
    assert (inbox(colonies) / PHOTO).exists()


def test_rerun_never_ends_a_retagged_row(colonies):
    rev = scan(colonies)
    assert apply(rev) == 0
    shutil.copy2(colonies / "evidence" / "Colony22_founddead_IMG_0001.JPEG", inbox(colonies) / PHOTO)
    with open(colonies / "tag_assignments.csv", "ab") as fh:
        fh.write(b"22,1,worker,R,,,,user 2026-10-21 retag,\r\n")        # tag 1 put on a new ant
    assert apply(rev) == 0
    rows = read_rows(colonies / "tag_assignments.csv")
    assert [r["ended_on"] for r in rows if (r["colony"], r["tag_id"]) == ("22", "1")] == ["2026-10-20", ""]
    assert not (inbox(colonies) / PHOTO).exists()


def test_future_date_override_is_refused(tmp_path, scene):
    col = make_colonies(tmp_path, scene)
    rev = scan(col)
    edit_review(rev, t1={"date_override": "2099-01-01"}, t13={"accept": "n"})
    assert apply(rev) == 1
    assert table_rows(col)[("22", "1")]["ended_on"] == ""


def test_rerun_after_an_interrupted_apply_finishes_the_job(colonies):
    rev = scan(colonies)
    assert apply(rev) == 0
    shutil.copy2(colonies / "evidence" / "Colony22_founddead_IMG_0001.JPEG", inbox(colonies) / PHOTO)
    table = (colonies / "tag_assignments.csv").read_bytes()
    assert apply(rev) == 0                                             # both rows: already recorded
    assert (colonies / "tag_assignments.csv").read_bytes() == table
    assert not (inbox(colonies) / PHOTO).exists()


def test_photo_date_falls_back_to_mtime(tmp_path):
    p = tmp_path / "x.png"
    save_photo(p, np.zeros((10, 10), np.uint8))
    ts = datetime(2026, 10, 19, 12, 0).timestamp()
    os.utime(p, (ts, ts))
    assert TP.photo_date(p) == ("2026-10-19 12:00:00", "mtime")
    assert TP.exif_date(p) is None


def test_photo_date_reads_exif(tmp_path):
    p = tmp_path / "x.jpg"
    save_photo(p, np.zeros((10, 10), np.uint8), "2026:10:18 07:05:09")
    assert TP.photo_date(p) == ("2026-10-18 07:05:09", "exif")


@pytest.mark.parametrize("photo", ["../x/IMG.JPEG", "C:/IMG.JPEG", "IMG.JPEG", "misc/IMG.JPEG", "C22_dead/a/b.JPEG"])
def test_review_rows_cannot_point_outside_the_inbox(tmp_path, photo):
    assert TP.inbox_path({"photo": photo}, tmp_path) is None
