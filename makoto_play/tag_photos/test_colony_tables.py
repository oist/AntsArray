"""Tests for colony_tables.py: folder names and tag_assignments.csv updates."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import colony_tables as T  # noqa: E402

HEADER = "colony,tag_id,role,paint,tagged_on,ended_on,end_reason,evidence,note"
ROWS = [
    "21,0,gamergate,,,,,user,",
    '21,1,worker,Y,,,,note 2026-09-28 (Y73),"tagged before 2026-09-28; found dead 2026-10-01, colony not yet known"',
    '22,1,worker,Y,,,,note 2026-09-25 (Y75),"tagged before 2026-09-25; found dead 2026-10-01, colony not yet known"',
    "22,80,worker,Y,,2026-10-08,found_dead,note 2026-09-25 (Y75); photo Colony22_founddead_IMG_3587 (OpenCV decode),",
    "22,13,worker,B,2026-10-01,,,user 2026-10-01,newborn",
]


def table_text(newline="\n", rows=ROWS):
    return newline.join([HEADER] + rows) + newline


@pytest.fixture
def table(tmp_path):
    p = tmp_path / "tag_assignments.csv"
    p.write_bytes(table_text().encode())
    return T.read_table(p)


def F(colony, tag_id, kind="dead", date="2026-10-20", photo="P1.JPEG"):
    return T.Finding(colony=colony, tag_id=tag_id, kind=kind, date=date, photos=(photo,))


def round_trip(tmp_path, text, rows_fn=lambda rows: rows):
    p = tmp_path / "t.csv"
    p.write_bytes(text.encode())
    t = T.read_table(p)
    T.write_table(p, t, rows_fn(t.rows))
    return p.read_bytes().decode()


@pytest.mark.parametrize("name,expect", [
    ("C21_dead", (21, "dead")), ("c22_TAG", (22, "tag")), ("Colony22_dead", (22, "dead")),
    ("C21_removed", (21, "removed")), ("C22_dropped", (22, "tag")), ("C21_check", (21, "check")),
    ("C21", None), ("misc", None), ("C21_alive", None),
])
def test_parse_folder(name, expect):
    assert T.parse_folder(name) == expect


def test_evidence_name():
    assert T.evidence_name(22, "dead", "IMG_3587.JPEG") == "Colony22_founddead_IMG_3587.JPEG"
    assert T.evidence_name(21, "tag", "IMG_1.HEIC") == "Colony21_tagfound_IMG_1.HEIC"


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_round_trip_is_byte_identical(tmp_path, newline):
    assert round_trip(tmp_path, table_text(newline)) == table_text(newline)


def test_unchanged_rows_keep_their_bytes(tmp_path):
    text = (HEADER + "\r\n" + '21,0,gamergate,"",,,,"user",\r\n\r\n'
            + "22,1,worker,Y,,,,ev,\r\n" + "22,2,worker,Y,,,,ev,")          # quotes, blank line, no final newline
    def end_22_2(rows):
        return rows[:2] + [{**rows[2], "ended_on": "2026-10-20"}]
    out = round_trip(tmp_path, text, end_22_2)
    assert out == text.replace("22,2,worker,Y,,,,ev,", "22,2,worker,Y,,2026-10-20,,ev,")


def test_insert_rows_positions(tmp_path):
    p = tmp_path / "t.csv"
    p.write_bytes(table_text("\r\n").encode())
    t = T.insert_rows(T.read_table(p), [{"colony": "22", "tag_id": "3"}, {"colony": "23", "tag_id": "0"}])
    assert [(r["colony"], r["tag_id"]) for r in t.rows] == [
        ("21", "0"), ("21", "1"), ("22", "1"), ("22", "3"), ("22", "80"), ("22", "13"), ("23", "0")]
    T.write_table(p, t, t.rows)
    text = p.read_bytes().decode()
    assert text.startswith(table_text("\r\n").split("22,80")[0] + "22,3,,,,,,,\r\n22,80")
    assert text.endswith("22,13,worker,B,2026-10-01,,,user 2026-10-01,newborn\r\n23,0,,,,,,,\r\n")


def test_insert_at_end_adds_the_missing_final_newline(tmp_path):
    p = tmp_path / "t.csv"
    p.write_bytes((HEADER + "\n21,0,gamergate,,,,,user,").encode())
    t = T.insert_rows(T.read_table(p), [{"colony": "21", "tag_id": "5"}])
    T.write_table(p, t, t.rows)
    assert p.read_bytes().decode() == HEADER + "\n21,0,gamergate,,,,,user,\n21,5,,,,,,,\n"


def test_short_row_is_refused(tmp_path):
    p = tmp_path / "t.csv"
    p.write_bytes(table_text(rows=ROWS + ["21,2,worker"]).encode())
    with pytest.raises(ValueError, match="data row 6 has 3 fields"):
        T.read_table(p)


def test_update_ends_only_the_matching_colony_row(table):
    updates, problems = T.plan_updates(table.rows, [F(22, 1)])
    assert problems == [] and len(updates) == 1
    new = T.apply_updates(table.rows, updates)
    assert new[2]["ended_on"] == "2026-10-20" and new[2]["end_reason"] == "found_dead"
    assert new[2]["evidence"].endswith("; photo P1.JPEG (tag_photos)")
    assert new[1] == table.rows[1]                 # colony 21's tag 1 untouched
    assert table.rows[2]["ended_on"] == ""         # input not mutated


def test_kind_sets_end_reason(table):
    for kind, reason in (("tag", "tag_lost"), ("removed", "removed")):
        updates, _ = T.plan_updates(table.rows, [F(21, 1, kind=kind)])
        assert T.apply_updates(table.rows, updates)[1]["end_reason"] == reason


def test_unknown_id_is_a_problem(table):
    updates, problems = T.plan_updates(table.rows, [F(21, 55)])
    assert updates == [] and "no row" in problems[0]


def test_already_ended_is_a_problem(table):
    updates, problems = T.plan_updates(table.rows, [F(22, 80)])
    assert updates == [] and "already ended 2026-10-08 (found_dead)" in problems[0]


def test_rerun_with_the_same_photo_is_already_recorded(table):
    updates, _ = T.plan_updates(table.rows, [F(22, 1)])
    rows = T.apply_updates(table.rows, updates)
    again, problems = T.plan_updates(rows, [F(22, 1)])
    assert problems == [] and again[0].already
    assert T.apply_updates(rows, again) == rows
    other, problems = T.plan_updates(rows, [F(22, 1, photo="P11.JPEG")])
    assert other == [] and "already ended" in problems[0]


def test_two_photos_of_one_ant_make_one_update(table):
    fs = [F(22, 1, date="2026-10-21", photo="P2.JPEG"), F(22, 1, date="2026-10-20", photo="P1.JPEG")]
    updates, problems = T.plan_updates(table.rows, fs)
    assert problems == [] and len(updates) == 1
    row = T.apply_updates(table.rows, updates)[2]
    assert row["ended_on"] == "2026-10-20"
    assert row["evidence"].endswith("; photos P1.JPEG, P2.JPEG (tag_photos)")


def test_conflicting_kinds_are_a_problem(table):
    updates, problems = T.plan_updates(table.rows, [F(22, 1), F(22, 1, kind="tag", photo="P2.JPEG")])
    assert updates == [] and "both" in problems[0]


def test_found_before_tagged_is_a_problem(table):
    updates, problems = T.plan_updates(table.rows, [F(22, 13, date="2026-09-30")])
    assert updates == [] and "before it was tagged" in problems[0]


def test_malformed_tagged_on_is_a_problem(tmp_path):
    p = tmp_path / "t.csv"
    p.write_bytes(table_text(rows=["22,13,worker,B,1 Oct,,,user,"]).encode())
    updates, problems = T.plan_updates(T.read_table(p).rows, [F(22, 13)])
    assert updates == [] and "not YYYY-MM-DD" in problems[0]


def test_two_active_rows_are_a_problem(table):
    rows = table.rows + [dict(table.rows[4])]
    updates, problems = T.plan_updates(rows, [F(22, 13)])
    assert updates == [] and "2 active rows" in problems[0]
