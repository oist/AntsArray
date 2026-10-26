"""Tests for husbandry.py: reading husbandry_log.csv and the per-colony summary."""
import sys
from datetime import date
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import husbandry as H  # noqa: E402
import sheet_model as M  # noqa: E402

HEADER = ",".join(H.FIELDS) + "\n"
ON = date(2026, 10, 22)


def write_log(tmp_path, body, encoding="utf-8"):
    p = tmp_path / "husbandry_log.csv"
    p.write_text(HEADER + body, encoding=encoding)
    return p


def row(date_, colony, event, what="", n="", ids="", time="", provisional="", note=""):
    return ",".join([date_, time, colony, event, what, str(n), ids, "", "", "src", note, provisional]) + "\n"


# --- reading --------------------------------------------------------------------------------

def test_read_log_returns_typed_entries_and_skips_blank_lines(tmp_path):
    p = write_log(tmp_path, row("2026-10-01", "22", "dead", n=3, ids="1 34 77", time="12:19")
                  + ",,,,,,,,,,,\n" + row("2026-10-02", "all", "fed", what="MW+RR+J"), encoding="utf-8-sig")
    entries = H.read_log(p)
    assert [e.event for e in entries] == ["dead", "fed"]
    first = entries[0]
    assert (first.date, first.time, first.colony, first.n, first.ids) == (
        date(2026, 10, 1), "12:19", "22", 3, (1, 34, 77))
    assert entries[1].n is None and entries[1].ids == ()


@pytest.mark.parametrize("body,match", [
    (row("2026-10-1", "22", "dead", n=1), "date"),
    (row("2026-10-01", "22", "dead", n=1, time="9:5"), "time"),
    (row("2026-10-01", "22", "died", n=1), "event"),
    (row("2026-10-01", "C 22", "dead", n=1), "colony name"),
    (row("2026-10-01", "all", "dead", n=1), "all"),
    (row("2026-10-01", "22", "census", n=5), "census"),
    (row("2026-10-01", "22", "census", what="adult"), "number"),
    (row("2026-10-01", "22", "leftover", what="4"), "0-3"),
    (row("2026-10-01", "22", "dead", n="x"), "n"),
    (row("2026-10-01", "22", "dead", n=1, ids="100"), "ids"),
    (row("2026-10-01", "22", "dead", n=1, provisional="maybe"), "provisional"),
])
def test_read_log_rejects_bad_rows_with_their_line(tmp_path, body, match):
    with pytest.raises(M.InputError, match=match):
        H.read_log(write_log(tmp_path, body))


def test_read_log_rejects_missing_columns(tmp_path):
    p = tmp_path / "husbandry_log.csv"
    p.write_text("date,colony,event\n2026-10-01,22,dead\n", encoding="utf-8")
    with pytest.raises(M.InputError, match="missing column"):
        H.read_log(p)


# --- adults -------------------------------------------------------------------------------

def entries(tmp_path, body):
    return H.read_log(write_log(tmp_path, body))


def test_adults_are_the_last_full_count_plus_new_minus_dead_and_removed(tmp_path):
    log = entries(tmp_path, row("2026-09-25", "22", "census", what="adult", n=96)
                  + row("2026-09-25", "22", "dead", n=2)                 # same day: in the count
                  + row("2026-10-01", "22", "painted", what="B", n=6)
                  + row("2026-10-01", "22", "dead", n=3, ids="1 34 77")
                  + row("2026-10-08", "22", "dead", ids="80 84")          # n from the IDs
                  + row("2026-10-09", "22", "removed", n=4)
                  + row("2026-10-30", "22", "dead", n=9)                 # after the day asked
                  + row("2026-10-02", "21", "dead", n=50))               # another colony
    est = H.adult_estimate(log, "22", ON)
    assert (est.n, est.base_date, est.base_n, est.added, est.lost) == (
        96 + 6 - 3 - 2 - 4, date(2026, 9, 25), 96, 6, 9)
    assert est.unknown == () and not est.provisional


def test_adults_report_losses_of_unknown_size_and_provisional_rows(tmp_path):
    log = entries(tmp_path, row("2026-09-28", "21", "census", what="adult", n=99, provisional="y")
                  + row("2026-10-08", "21", "removed", what="foragers", note="number not recorded"))
    est = H.adult_estimate(log, "21", ON)
    assert est.n == 99 and est.provisional
    assert est.unknown == ("2026-10-08 removed: number not recorded",)


def test_timed_events_after_a_timed_count_on_the_same_day_are_applied(tmp_path):
    log = entries(tmp_path, row("2026-10-01", "8", "census", what="adult", n=50, time="10:00")
                  + row("2026-10-01", "8", "dead", n=1, time="09:00")      # before the count: in it
                  + row("2026-10-01", "8", "dead", n=3, time="16:00")      # after the count
                  + row("2026-10-01", "8", "dead", n=5))                   # untimed: assumed in it
    assert H.adult_estimate(log, "8", ON).n == 47


def test_the_last_of_two_counts_on_one_day_wins(tmp_path):
    log = entries(tmp_path, row("2026-10-01", "8", "census", what="adult", n=50)
                  + row("2026-10-01", "8", "census", what="adult", n=48, note="recount"))
    assert H.adult_estimate(log, "8", ON).base_n == 48


def test_new_adults_without_a_number_are_reported(tmp_path):
    log = entries(tmp_path, row("2026-10-01", "8", "census", what="adult", n=50)
                  + row("2026-10-05", "8", "painted", what="G"))
    est = H.adult_estimate(log, "8", ON)
    assert est.n == 50 and est.unknown == ("2026-10-05 painted: number not recorded",)


def test_a_negative_estimate_is_flagged(tmp_path):
    log = entries(tmp_path, row("2026-10-01", "8", "census", what="adult", n=2)
                  + row("2026-10-05", "8", "dead", n=5))
    est = H.adult_estimate(log, "8", ON)
    assert est.n == -3 and any("below zero" in u for u in est.unknown)


def test_water_and_leftover_without_an_amount_are_valid_events(tmp_path):
    log = entries(tmp_path, row("2026-10-01", "8", "water") + row("2026-10-01", "8", "leftover", what="y"))
    assert [(e.event, e.what) for e in log] == [("water", ""), ("leftover", "y")]


def test_dead_males_are_not_adults_lost(tmp_path):
    log = entries(tmp_path, row("2026-10-01", "8", "census", what="adult", n=50)
                  + row("2026-10-05", "8", "dead", n=2)
                  + row("2026-10-05", "8", "dead", what="male", n=3))
    assert H.adult_estimate(log, "8", ON).n == 48
    assert H.colony_summary(log, "8", ON).dead_28d == 2


def test_summary_has_the_last_water_change(tmp_path):
    log = entries(tmp_path, row("2026-10-01", "8", "water") + row("2026-10-12", "8", "water"))
    assert H.colony_summary(log, "8", ON).last_water == date(2026, 10, 12)


def test_adults_are_unknown_without_a_full_count(tmp_path):
    assert H.adult_estimate(entries(tmp_path, row("2026-10-01", "8", "dead", n=1)), "8", ON) is None


def test_a_newer_full_count_restarts_the_sum(tmp_path):
    log = entries(tmp_path, row("2026-09-01", "8", "census", what="adult", n=50)
                  + row("2026-09-10", "8", "dead", n=5)
                  + row("2026-10-01", "8", "census", what="adult", n=40)
                  + row("2026-10-05", "8", "painted", what="G", n=2))
    assert H.adult_estimate(log, "8", ON).n == 42


# --- summary --------------------------------------------------------------------------------

def test_colony_summary(tmp_path):
    log = entries(tmp_path, row("2026-10-01", "22", "census", what="adult", n=90)
                  + row("2026-10-01", "22", "census", what="pupa", n=5)
                  + row("2026-10-15", "22", "census", what="pupa", n=7)
                  + row("2026-10-20", "all", "fed", what="MW+RR+J")
                  + row("2026-10-21", "all", "fed", what="J")
                  + row("2026-10-21", "22", "not_fed")
                  + row("2026-10-21", "22", "leftover", what="2")
                  + row("2026-10-20", "22", "dead", n=2)
                  + row("2026-09-01", "22", "dead", n=7)                 # older than 28 days
                  + row("2026-10-10", "22", "painted", what="G", n=3)
                  + row("2026-07-01", "22", "tray_changed"))
    s = H.colony_summary(log, "22", ON)
    assert s.pupae == (date(2026, 10, 15), 7)
    assert s.last_fed == date(2026, 10, 20)
    assert s.last_leftover == (date(2026, 10, 21), "2")
    assert (s.dead_28d, s.new_28d) == (2, 3)
    assert s.last_paint == (date(2026, 10, 10), "G")
    assert s.tray == (date(2026, 7, 1), "overdue")
    assert [e.event for e in s.history][:2] == ["not_fed", "leftover"]      # newest first


@pytest.mark.parametrize("last,status", [(None, "unknown"), (date(2026, 9, 1), "ok"),
                                         (date(2026, 8, 1), "due"), (date(2026, 7, 1), "overdue")])
def test_tray_status(last, status):
    assert H.tray_status(last, ON) == status
