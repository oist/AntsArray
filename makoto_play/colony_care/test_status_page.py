"""Tests for status_page.py: the colony status page (overview + one section per colony)."""
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import husbandry as H  # noqa: E402
import status_page as SP  # noqa: E402

ON = date(2026, 10, 22)
LOG = (",".join(H.FIELDS) + "\n"
       "2026-09-28,,21,census,adult,99,,,,note photo,,y\n"
       "2026-10-08,18:31,21,removed,foragers,,,20261008/block01,,session NOTE,number not recorded,y\n"
       "2026-10-20,,21,dead,,10,1 7 9 10 12 14 17 19 37 88,,,tag_photos,<b>dish</b>,y\n"
       "2026-10-20,,all,fed,MW+RR+J,,,,MH,sheet,,\n"
       "2026-10-01,,Mix,dead,,2,,,YT,sheet,,\n")


def make_dir(tmp_path, log=LOG, roster="21,0,,,,,2026-10-01,\nMix,,,,,,2026-10-01,\n8,,,,,,2026-10-01,\n"):
    (tmp_path / "colonies.csv").write_text(
        "colony,gamergate_tag,collected_on,census_on,census,note,care_from,care_until\n" + roster,
        encoding="utf-8")
    (tmp_path / "arena.csv").write_text("valid_from,left,right,note\n2026-09-28,22,21,\n", encoding="utf-8")
    (tmp_path / "tag_assignments.csv").write_text(
        "colony,tag_id,role,paint,tagged_on,ended_on,end_reason,evidence,note\n"
        "21,0,gamergate,,,,,,\n21,1,worker,Y,,2026-10-20,found_dead,,\n21,2,worker,Y,,,,,\n",
        encoding="utf-8")
    if log is not None:
        (tmp_path / "husbandry_log.csv").write_text(log, encoding="utf-8")
    return tmp_path


def test_status_page_lists_every_colony_and_links_its_section(tmp_path):
    html = SP.build_html(make_dir(tmp_path), ON)
    for name, key in (("21", "21"), ("Mix", "MIX"), ("8", "8")):
        assert 'href="#c-%s"' % key in html and 'id="c-%s"' % key in html
        assert ">%s<" % name in html


def test_status_page_shows_computed_adults_with_their_caveats(tmp_path):
    html = SP.build_html(make_dir(tmp_path), ON)
    assert '<td class="adults">89*' in html                 # 99 - 10, provisional
    assert "2026-10-08 removed: number not recorded" in html
    assert "no full count yet" in html                      # Mix and 8


def test_status_page_shows_tags_in_use_arena_and_last_feeding(tmp_path):
    html = SP.build_html(make_dir(tmp_path), ON)
    assert "right half" in html                             # colony 21 in the arena
    assert '<td class="fed">2026-10-20</td>' in html        # last fed (round for all)
    assert '<td class="tags">2</td>' in html                # tags in use for 21 (0 and 2)


def test_status_page_escapes_text_from_the_log(tmp_path):
    html = SP.build_html(make_dir(tmp_path), ON)
    assert "<b>dish</b>" not in html and "&lt;b&gt;dish&lt;/b&gt;" in html


def test_status_page_warns_about_log_colonies_missing_from_colonies_csv(tmp_path):
    log = LOG + "2026-10-02,,99,dead,,1,,,,sheet,,\n"
    html = SP.build_html(make_dir(tmp_path, log=log), ON)
    assert "not in colonies.csv: 99" in html


def test_legacy_colonies_are_listed_but_not_warned_about(tmp_path):
    roster = ("21,0,,,,,2026-10-01,\nMix,,,,,,2026-10-01,\n"
              "4A,,2024-03-21,,,legacy box,2024-11-01,2025-04-16\n")
    log = LOG + ("2025-04-10,09:00,4A,fed,MW+J,,,,,forms ID 1,,\n"
                 "2025-04-10,09:00,4A,dead,,2,,,,forms ID 1,,\n"
                 "2025-04-10,09:00,4A,dead,male,1,,,,forms ID 1,,\n"
                 "2025-04-16,09:00,4A,note,wiped out,,,,,forms ID 2,,\n")
    html = SP.build_html(make_dir(tmp_path, log=log, roster=roster), ON)
    assert "not in colonies.csv" not in html
    assert 'id="legacy"' in html
    assert "<td>4A</td><td>2024-11-01 - 2025-04-16</td><td>2024-03-21</td><td>1</td><td>2</td><td>1</td>" in html
    assert "2025-04-16 wiped out" in html


def test_long_histories_show_the_newest_rows_and_say_how_many_are_left_out(tmp_path):
    many = "".join("2026-%02d-%02d,,8,water,,,,,,forms,,\n" % (m, d) for m in range(1, 10) for d in range(1, 29))
    html = SP.build_html(make_dir(tmp_path, log=LOG + many), ON)
    assert "2026-09-28" in html and "2026-01-01" not in html
    assert "%d older rows not shown" % (9 * 28 - SP.HISTORY_ROWS) in html


def test_main_writes_the_page_and_a_log(tmp_path):
    d = make_dir(tmp_path)
    out = tmp_path / "out" / "status.html"
    assert SP.main(["--colonies-dir", str(d), "--on", "2026-10-22", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
    [log] = list((out.parent / "logs").glob("status_page_log_*.txt"))
    assert "status page" in log.read_text(encoding="utf-8")


def test_main_fails_without_the_husbandry_log(tmp_path):
    d = make_dir(tmp_path, log=None)
    out = tmp_path / "out" / "status.html"
    assert SP.main(["--colonies-dir", str(d), "--on", "2026-10-22", "--out", str(out)]) == 2
    assert not out.exists()
