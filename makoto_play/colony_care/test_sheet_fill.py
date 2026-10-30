"""Tests for sheet_fill.py: what husbandry_log.csv already says for the cells of a month's sheets."""
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import husbandry as H  # noqa: E402
import sheet_fill as F  # noqa: E402

OCT = date(2026, 10, 1)
SECOND_HALF = (date(2026, 10, 16), date(2026, 10, 31))


def log(tmp_path, *rows):
    p = tmp_path / "husbandry_log.csv"
    p.write_text(",".join(H.FIELDS) + "\n" + "".join(rows), encoding="utf-8")
    return H.read_log(p)


def row(date_, colony, event, what="", n="", ids="", time="", by="", note=""):
    return ",".join([date_, time, colony, event, what, str(n), ids, "", by, "src", note, ""]) + "\n"


# --- feeding sheets ---------------------------------------------------------------------------

def test_feeding_day_has_the_round_and_a_dead_left_pair_per_colony(tmp_path):
    entries = log(tmp_path,
                  row("2026-10-15", "all", "fed", "J", time="09:00", by="MH"),
                  row("2026-10-23", "all", "fed", "MW+RR+J", time="12:55", by="YK"),
                  row("2026-10-23", "8", "dead", n=1, time="12:55"),
                  row("2026-10-23", "8", "leftover", "3", time="12:55"),
                  row("2026-10-23", "Mix", "leftover", "0", time="12:55"))
    days = F.feeding_days(entries, *SECOND_HALF)
    assert list(days) == [date(2026, 10, 23)]
    day = days[date(2026, 10, 23)]
    assert (day.time, day.by) == ("12:55", "YK")
    # a recorded left-over means the pair was filled in: no dead row = 0 dead
    assert day.cells == {"8": F.FeedingCell(1, "3", False), "MIX": F.FeedingCell(0, "0", False)}


def test_feeding_cell_counts_dead_females_and_marks_a_colony_not_fed(tmp_path):
    entries = log(tmp_path,
                  row("2026-10-23", "26", "dead", n=4),
                  row("2026-10-23", "26", "dead", "male", n=1),
                  row("2026-10-23", "26", "dead", ids="5 6"),
                  row("2026-10-23", "10#1", "not_fed"))
    day = F.feeding_days(entries, *SECOND_HALF)[date(2026, 10, 23)]
    assert (day.time, day.by) == ("", "")
    assert day.cells == {"26": F.FeedingCell(6, "", False), "10.1": F.FeedingCell(None, "", True)}


def test_feeding_round_without_an_all_row_comes_from_the_colony_feedings(tmp_path):
    entries = log(tmp_path,
                  row("2026-10-02", "8", "fed", "MW+J", time="13:14", by="MH"),
                  row("2026-10-02", "10#1", "fed", "MW+J", time="13:12", by="YK"),
                  row("2026-10-02", "14", "fed", "MW", time="13:15", by="YK"))
    day = F.feeding_days(entries, OCT, date(2026, 10, 15))[date(2026, 10, 2)]
    assert (day.time, day.by) == ("13:12", "YK MH")


# --- colony sheets ----------------------------------------------------------------------------

def test_colony_week_has_the_paint_day_counts_dead_tag_ids_and_comments(tmp_path):
    entries = log(tmp_path,
                  row("2026-10-29", "22", "painted", "G+G", n=4, by="YK"),
                  row("2026-10-29", "22", "census", "pupa", n=4, by="YK"),
                  row("2026-10-30", "22", "dead", n=2, ids="28 13"),
                  row("2026-10-30", "22", "dead", n=1),
                  row("2026-10-29", "22", "note", "queen", note="queen present"),
                  row("2026-10-29", "22", "note", "male", n=2),
                  row("2026-10-29", "23", "census", "pupa", n=9))
    weeks = F.colony_fill(entries, "22", OCT).weeks
    assert len(weeks) == 5 and weeks[:4] == [F.EMPTY_WEEK] * 4
    assert weeks[4] == F.ColonyWeek(date(2026, 10, 29), "YK", "G+G", 4, 4, (13, 28),
                                    ("queen present", "male 2"))


def test_colony_week_without_painting_has_zero_new_adults(tmp_path):
    entries = log(tmp_path, row("2026-10-29", "mix", "census", "pupa", n=0, by="YK"))
    week = F.colony_fill(entries, "Mix", OCT).weeks[4]
    assert (week.date, week.by, week.colour, week.new_adults, week.pupae) == (
        date(2026, 10, 29), "YK", "", 0, 0)


def test_colony_week_lists_two_paint_days_and_colours(tmp_path):
    entries = log(tmp_path,
                  row("2026-10-23", "25", "painted", "Gy", n=2, by="YK"),
                  row("2026-10-22", "25", "painted", "Gy", n=20, by="MH"),
                  row("2026-10-24", "25", "painted", "W", n=1, by="MH"),
                  row("2026-10-22", "25", "census", "pupa", n=20, by="MH"),
                  row("2026-10-24", "25", "census", "pupa", n=18, by="MH"))
    week = F.colony_fill(entries, "25", OCT).weeks[3]
    assert (week.date, week.by, week.colour, week.new_adults, week.pupae) == (
        date(2026, 10, 22), "MH YK", "Gy W", 23, 18)


def test_last_tray_change_is_the_latest_up_to_the_end_of_the_month(tmp_path):
    entries = log(tmp_path,
                  row("2026-08-20", "22", "tray_changed", "plastic"),
                  row("2026-10-05", "22", "tray_changed"),
                  row("2026-11-02", "22", "tray_changed"))
    assert F.colony_fill(entries, "22", OCT).last_tray == date(2026, 10, 5)
    assert F.colony_fill([], "22", OCT).last_tray is None


def test_colony_fill_carries_the_collection_date(tmp_path):
    assert F.colony_fill([], "8", OCT, collected=date(2024, 10, 17)).collected == date(2024, 10, 17)
    assert F.colony_fill([], "8", OCT).collected is None


def test_events_of_the_month_with_their_sheet_type_and_details(tmp_path):
    entries = log(tmp_path,
                  row("2026-10-08", "22", "nest_changed", "new nests", time="15:44"),
                  row("2026-10-08", "22", "removed", "foragers", time="18:31", note="number not recorded"),
                  row("2026-10-01", "22", "tagged", "B", n=6, ids="13 28"),
                  row("2026-10-20", "22", "tag_lost", n=3, ids="3 85 98", note="loose tags found"),
                  row("2026-09-28", "22", "census", "adult", n=99),
                  row("2026-10-29", "22", "census", "adult", n=80, time="10:00", by="MH"),
                  row("2026-10-29", "22", "census", "pupa", n=4),
                  row("2026-10-29", "22", "water"))
    events = F.colony_fill(entries, "22", OCT).events
    assert [(e.date.day, e.kind, e.details) for e in events] == [
        (1, "tagging #", "B 6: 13 28"),
        (8, "nest", "new nests"),
        (8, "removal", "foragers; number not recorded"),
        (20, "other", "tag lost 3: 3 85 98; loose tags found"),
        (29, "count", "80 adults")]
    assert events[4] == F.ColonyEvent(date(2026, 10, 29), "10:00", "count", "80 adults", "MH")


def test_colony_fill_knows_whether_the_log_has_anything_for_the_sheet(tmp_path):
    assert not F.colony_fill([], "22", OCT, collected=date(2024, 10, 17)).has_records
    entries = log(tmp_path, row("2026-10-29", "22", "census", "pupa", n=4))
    assert F.colony_fill(entries, "22", OCT).has_records


def test_counts_that_were_not_recorded_are_a_question_mark(tmp_path):
    entries = log(tmp_path,
                  row("2026-10-06", "14", "painted", "Y", note="adults painted Y; not counted"),
                  row("2026-10-06", "15", "painted", "Y", n=3),
                  row("2026-10-06", "15", "painted", "G"),
                  row("2026-10-23", "8", "dead"),
                  row("2026-10-23", "8", "leftover", "1"),
                  row("2026-10-23", "17", "dead", n=2),
                  row("2026-10-23", "17", "dead"))
    assert F.colony_fill(entries, "14", OCT).weeks[0].new_adults == "?"
    assert F.colony_fill(entries, "15", OCT).weeks[0].new_adults == "3+?"
    cells = F.feeding_days(entries, *SECOND_HALF)[date(2026, 10, 23)].cells
    assert cells["8"].dead == "?" and cells["17"].dead == "2+?"


def test_no_dead_row_is_zero_dead_only_next_to_a_left_over_code_from_a_sheet(tmp_path):
    entries = log(tmp_path,
                  row("2026-10-21", "8", "leftover", "y"),      # Forms: some food left, dead not a pair
                  row("2026-10-23", "8", "leftover", "2"))      # sheet: D and L are written together
    days = F.feeding_days(entries, *SECOND_HALF)
    assert days[date(2026, 10, 21)].cells["8"] == F.FeedingCell(None, "y", False)
    assert days[date(2026, 10, 23)].cells["8"] == F.FeedingCell(0, "2", False)


def test_paint_day_is_the_day_of_painting_when_the_pupae_were_counted_earlier(tmp_path):
    entries = log(tmp_path,
                  row("2026-10-08", "22", "census", "pupa", n=5, by="MH"),
                  row("2026-10-10", "22", "painted", "R", n=2, by="YK"))
    week = F.colony_fill(entries, "22", OCT).weeks[1]
    assert (week.date, week.by, week.new_adults, week.pupae) == (date(2026, 10, 10), "MH YK", 2, 5)
