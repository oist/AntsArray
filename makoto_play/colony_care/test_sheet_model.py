"""Tests for sheet_model.py: months, feeding plan, colony names, sheet codes, roster, arena."""
import sys
from datetime import date
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sheet_model as M  # noqa: E402

OCT = date(2026, 10, 1)
OCT_END = date(2026, 10, 31)
ARENA = [{"valid_from": "2026-09-28", "left": "22", "right": "21", "note": ""}]
CHANGED = ARENA + [{"valid_from": "2026-10-16", "left": "Mix", "right": "10#1", "note": ""}]


# --- months -------------------------------------------------------------------------------

def test_parse_month():
    assert M.parse_month("2026-10") == OCT


@pytest.mark.parametrize("text", ["2026-13", "2026-1", "202610", "2026-10-01", "", "0000-05"])
def test_parse_month_rejects_bad_text(text):
    with pytest.raises(M.InputError):
        M.parse_month(text)


@pytest.mark.parametrize("month,end", [(OCT, OCT_END), (date(2027, 2, 1), date(2027, 2, 28)),
                                       (date(2026, 12, 1), date(2026, 12, 31))])
def test_month_end(month, end):
    assert M.month_end(month) == end


def test_feeding_pages_split_a_month_at_the_15th():
    assert M.feeding_pages(OCT) == [(OCT, date(2026, 10, 15)), (date(2026, 10, 16), OCT_END)]
    feb = date(2027, 2, 1)
    assert M.feeding_pages(feb) == [(feb, date(2027, 2, 15)), (date(2027, 2, 16), date(2027, 2, 28))]


def test_week_ranges_cut_a_month_into_7_day_rows():
    assert M.week_ranges(OCT) == [(date(2026, 10, a), date(2026, 10, b))
                                  for a, b in ((1, 7), (8, 14), (15, 21), (22, 28), (29, 31))]
    assert len(M.week_ranges(date(2027, 2, 1))) == 4


@pytest.mark.parametrize("day,plan", [
    (date(2026, 10, 5), "MW RR J"), (date(2026, 10, 6), "J"), (date(2026, 10, 7), "MW RR J"),
    (date(2026, 10, 1), "J"), (date(2026, 10, 2), "MW RR J"), (date(2026, 10, 3), "J*"),
    (date(2026, 10, 4), "J*"),
])
def test_planned_food_follows_the_weekly_feeding_plan(day, plan):
    assert M.planned_food(day) == plan


# --- colony names -------------------------------------------------------------------------

@pytest.mark.parametrize("spec,expected", [
    ("21", ["21"]), ("8,Mix,10#1,10#2", ["8", "Mix", "10#1", "10#2"]),
    ("22-24,8", ["22", "23", "24", "8"]), (" 22 , Mix ", ["22", "Mix"]),
])
def test_parse_colonies_accepts_names_lists_and_ranges(spec, expected):
    assert M.parse_colonies(spec) == expected


def test_parse_colonies_accepts_legacy_box_names():
    assert M.parse_colonies("4A,13tag-R,6+4B,18mix,Mix-2") == ["4A", "13tag-R", "6+4B", "18mix", "Mix-2"]


@pytest.mark.parametrize("spec", ["", "0", "021", "10#", "#1", "-2", "+4", "a b", "13tag_R", "7-5",
                                  "Mix,MIX", "21,21", "1-100000", "Abcdefghi", "1000", "1000#1", "10#100",
                                  "10#01",
"\uff12\uff11"])
def test_parse_colonies_rejects_bad_specs(spec):
    with pytest.raises(M.InputError):
        M.parse_colonies(spec)


@pytest.mark.parametrize("name,key", [("10#1", "10.1"), ("Mix", "MIX"), ("22", "22"),
                                      ("13tag-R", "13TAG-R"), ("6+4B", "6+4B")])
def test_colony_key_is_qr_alphanumeric(name, key):
    assert M.colony_key(name) == key


# --- sheet codes --------------------------------------------------------------------------

def test_colony_code_round_trips():
    code = M.colony_code("10#1", OCT)
    assert code == "CARE1 C10.1 2026-10"
    assert M.parse_sheet_code(code) == M.SheetId("colony", "10.1", OCT, None)


@pytest.mark.parametrize("name,key", [("13tag-R", "13TAG-R"), ("6+4B", "6+4B"), ("Mix", "MIX")])
def test_colony_codes_of_legacy_names_round_trip(name, key):
    assert M.parse_sheet_code(M.colony_code(name, OCT)) == M.SheetId("colony", key, OCT, None)


def test_a_name_with_a_trailing_newline_is_rejected():
    with pytest.raises(M.InputError):
        M.check_colony_name("22\n")


def test_feeding_code_round_trips():
    code = M.feeding_code(OCT, 2)
    assert code == "FEED1 2026-10 P2"
    assert M.parse_sheet_code(code) == M.SheetId("feeding", None, OCT, 2)


@pytest.mark.parametrize("code", [M.colony_code("Wwwwwwww", OCT), M.colony_code("999#99", OCT),
                                  M.colony_code("13tag-R", OCT), M.colony_code("6+4B", OCT),
                                  M.feeding_code(OCT, 2)])
def test_sheet_codes_stay_version_1_qr_codes(code):
    import cv2
    assert cv2.QRCodeEncoder.create().encode(code).shape == (25, 25)


@pytest.mark.parametrize("text", ["CARE9 C21 2026-10", "FEED1 2026-10 P3", "FEED1 2026-10 P0",
                                  "CARE1 C 2026-10", "FEED1 2026-13 P1", "hello"])
def test_parse_sheet_code_rejects_foreign_text(text):
    with pytest.raises(M.InputError):
        M.parse_sheet_code(text)


# --- roster (colonies.csv) ----------------------------------------------------------------

HEADER = "colony,gamergate_tag,collected_on,census_on,census,note,care_from,care_until\n"


def write_roster(tmp_path, body, header=HEADER, encoding="utf-8"):
    p = tmp_path / "colonies.csv"
    p.write_text(header + body, encoding=encoding)
    return p


def test_read_roster_lists_colonies_cared_for_in_the_month_in_file_order(tmp_path):
    p = write_roster(tmp_path, "8,,,,,,2026-10-01,\n21,0,,,,,,\n22,0,,,,,2026-10-01,\n"
                               "Mix,,,,,,2026-10-01,2026-10-20\n27,,,,,,2026-11-01,\n"
                               "26,,,,,,2026-09-01,2026-09-30\n,,,,,,,\n")
    assert M.read_roster(p, OCT) == ["8", "22", "Mix"]
    assert M.read_roster(p, date(2026, 11, 1)) == ["8", "22", "27"]


def test_read_roster_accepts_an_excel_bom(tmp_path):
    p = write_roster(tmp_path, "10#1,,,,,,2026-10-01,\n", encoding="utf-8-sig")
    assert M.read_roster(p, OCT) == ["10#1"]


@pytest.mark.parametrize("body,match", [
    ("8,,,,,,2026-10-1,\n", "care_from"),
    ("8,,,,,,2026-10-01,2026-09-01\n", "before care_from"),
    ("C 8,,,,,,2026-10-01,\n", "colony name"),
    ("8,,,,,,2026-10-01,\n8,,,,,,2026-10-01,\n", "twice"),
])
def test_read_roster_rejects_bad_rows(tmp_path, body, match):
    with pytest.raises(M.InputError, match=match):
        M.read_roster(write_roster(tmp_path, body), OCT)


def test_read_collected_gives_the_collection_date_per_colony_key(tmp_path):
    p = write_roster(tmp_path, "8,,2024-10-17,,,,2024-10-30,\nMix,,,,,,2025-01-08,\n10#1,,2024-10-17,,,,,\n")
    assert M.read_collected(p) == {"8": date(2024, 10, 17), "10.1": date(2024, 10, 17)}


def test_read_collected_rejects_a_bad_date(tmp_path):
    with pytest.raises(M.InputError, match="collected_on"):
        M.read_collected(write_roster(tmp_path, "8,,2024/10/17,,,,2024-10-30,\n"))


def test_read_roster_explains_missing_care_columns(tmp_path):
    p = write_roster(tmp_path, "8,,,,,\n", header="colony,gamergate_tag,collected_on,census_on,census,note\n")
    with pytest.raises(M.InputError, match="care_from"):
        M.read_roster(p, OCT)


# --- arena.csv ----------------------------------------------------------------------------

def write_arena(tmp_path, body, encoding="utf-8"):
    p = tmp_path / "arena.csv"
    p.write_text("valid_from,left,right,note\n" + body, encoding=encoding)
    return p


def test_read_arena_returns_stripped_rows(tmp_path):
    p = write_arena(tmp_path, "2026-09-28, 22 ,Mix,x\n")
    assert M.read_arena(p) == [{"valid_from": "2026-09-28", "left": "22", "right": "Mix", "note": "x"}]


def test_read_arena_accepts_an_excel_bom(tmp_path):
    rows = M.read_arena(write_arena(tmp_path, "2026-10-01,21,22,\n", encoding="utf-8-sig"))
    assert M.arena_side(rows, "21", OCT) == ("left", "2026-10-01")


def test_read_arena_raises_when_the_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        M.read_arena(tmp_path / "arena.csv")


@pytest.mark.parametrize("body,match", [
    ("2026-9-1,22,21,\n", "line 2"),
    ("20261001,22,21,\n", "not a YYYY-MM-DD"),
    ("2026-02-30,22,21,\n", "not a YYYY-MM-DD"),
    ("2026-09-28,22,21,\n2026-09-28,5,6,\n", "line 3.*twice"),
    ("2026-09-28,021,22,\n", "left '021'"),
    ("2026-09-28,22,21 23,\n", "right '21 23'"),
    ("2026-09-28,mix,MIX,\n", "both sides"),
])
def test_read_arena_rejects_bad_rows(tmp_path, body, match):
    with pytest.raises(M.InputError, match=match):
        M.read_arena(write_arena(tmp_path, body))


def test_read_arena_allows_an_empty_half_and_skips_blank_lines(tmp_path):
    rows = M.read_arena(write_arena(tmp_path, "2026-09-28,,21,\n,,,\n\n"))
    assert rows == [{"valid_from": "2026-09-28", "left": "", "right": "21", "note": ""}]


def test_read_arena_rejects_missing_columns(tmp_path):
    p = tmp_path / "arena.csv"
    p.write_text("valid_from,left\n2026-09-28,22\n", encoding="utf-8")
    with pytest.raises(M.InputError, match="right"):
        M.read_arena(p)


@pytest.mark.parametrize("colony,on,expected", [
    ("21", OCT, ("right", "2026-09-28")),
    ("22", OCT, ("left", "2026-09-28")),
    ("8", OCT, None),
    ("21", date(2026, 9, 27), None),
])
def test_arena_side_uses_the_row_in_force(colony, on, expected):
    assert M.arena_side(ARENA, colony, on) == expected


def test_arena_side_matches_names_ignoring_case():
    assert M.arena_side(CHANGED, "MIX", OCT_END) == ("left", "2026-10-16")


def test_arena_header_is_absent_when_arena_csv_was_not_read():
    assert M.arena_header(None, "21", OCT, OCT_END) is None


@pytest.mark.parametrize("colony,expected", [
    ("21", ("IN ARENA: right half since 2026-09-28 - write times to the minute", True)),
    ("8", ("Not in the arena", False)),
])
def test_arena_header_for_a_layout_that_holds_for_the_whole_month(colony, expected):
    assert M.arena_header(ARENA, colony, OCT, OCT_END) == expected


def test_arena_header_lists_changes_inside_the_month():
    assert M.arena_header(CHANGED, "21", OCT, OCT_END) == (
        "ARENA: 10/01 right half, 10/16 out - write times to the minute while in", True)
    assert M.arena_header(CHANGED, "Mix", OCT, OCT_END) == (
        "ARENA: 10/01 out, 10/16 left half - write times to the minute while in", True)


def test_arena_layouts_ignore_a_change_of_letter_case_only():
    rows = CHANGED + [{"valid_from": "2026-10-20", "left": "mix", "right": "10#1", "note": ""}]
    assert M.arena_layouts(rows, OCT, OCT_END) == [(OCT, "22", "21"), (date(2026, 10, 16), "Mix", "10#1")]


def test_arena_layouts_and_colonies_in_the_arena():
    assert M.arena_layouts(CHANGED, OCT, OCT_END) == [(OCT, "22", "21"), (date(2026, 10, 16), "Mix", "10#1")]
    assert M.arena_layouts([], OCT, OCT_END) == [(OCT, "", "")]
    assert M.arena_colonies(CHANGED, OCT, OCT_END) == ["22", "21", "Mix", "10#1"]
    assert M.arena_colonies(None, OCT, OCT_END) == []


def test_feeding_arena_line():
    assert M.feeding_arena_line(None, OCT, OCT_END) is None
    assert M.feeding_arena_line([], OCT, OCT_END) == ("No colony in the arena", False)
    assert M.feeding_arena_line(ARENA, OCT, OCT_END) == (
        "In the arena: left 22, right 21 - write the feeding time to the minute", True)
    assert M.feeding_arena_line(CHANGED, OCT, OCT_END) == (
        "In the arena: from 10/01 left 22, right 21; from 10/16 left Mix, right 10#1"
        " - write the feeding time to the minute", True)
