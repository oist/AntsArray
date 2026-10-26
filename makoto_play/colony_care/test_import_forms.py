"""Tests for xlsx_read.py and import_forms.py (Microsoft Forms feeding records -> husbandry log)."""
import csv
import sys
import zipfile
from datetime import date
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import husbandry as H  # noqa: E402
import import_forms as F  # noqa: E402
import sheet_model as M  # noqa: E402
import xlsx_read as X  # noqa: E402

SPLIT = date(2026, 5, 13)
FORM_HEAD = ["ID", "\u958b\u59cb\u6642\u523b", "\u5b8c\u4e86\u6642\u523b", "\u30e1\u30fc\u30eb", "Date", "Name2",
             "Box number", "Dead number (total)", "Feeding", "Water exchang", "Remarks", "Dead number of male"]
LEFT = "\u6b8b\u990c\u3042\u308a"                      # "food left" remark
TRAY = "\u30c8\u30ec\u30a4\u4ea4\u63db"                # "tray changed" remark
WIPED = "\u5168\u6ec5"                                  # "wiped out" remark


# --- a minimal .xlsx writer for the tests ---------------------------------------------------

def _serial(d):
    return (d - date(1899, 12, 30)).days


def _cell(ref, value):
    if isinstance(value, date):
        return '<c r="%s" s="1"><v>%d</v></c>' % (ref, _serial(value))
    if isinstance(value, tuple):                       # (date, "HH:MM") -> date-time cell
        d, hm = value
        h, m = map(int, hm.split(":"))
        return '<c r="%s" s="2"><v>%.6f</v></c>' % (ref, _serial(d) + (h * 60 + m) / 1440)
    if isinstance(value, (int, float)):
        return '<c r="%s"><v>%s</v></c>' % (ref, value)
    text = str(value).replace("&", "&amp;").replace("<", "&lt;")
    return '<c r="%s" t="inlineStr"><is><t>%s</t></is></c>' % (ref, text)


def make_xlsx(path, sheets):
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rns = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/styles.xml", '<styleSheet %s><numFmts count="1"><numFmt numFmtId="164" '
                   'formatCode="m/d/yy h:mm:ss"/></numFmts><cellXfs count="3"><xf numFmtId="0"/>'
                   '<xf numFmtId="14"/><xf numFmtId="164"/></cellXfs></styleSheet>' % ns)
        sheets_xml, rels = [], []
        for i, (name, rows) in enumerate(sheets.items(), start=1):
            sheets_xml.append('<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (name, i, i))
            rels.append('<Relationship Id="rId%d" Target="worksheets/sheet%d.xml"/>' % (i, i))
            body = "".join('<row r="%d">%s</row>' % (r, "".join(
                _cell("%s%d" % (chr(65 + c), r), v) for c, v in enumerate(row) if v is not None))
                for r, row in enumerate(rows, start=1))
            z.writestr("xl/worksheets/sheet%d.xml" % i, "<worksheet %s><sheetData>%s</sheetData></worksheet>"
                       % (ns, body))
        z.writestr("xl/workbook.xml", "<workbook %s %s><sheets>%s</sheets></workbook>" % (ns, rns, "".join(sheets_xml)))
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/'
                   'package/2006/relationships">%s</Relationships>' % "".join(rels))
    return path


def form_row(i, day, box, dead=0, feeding="Mealworm;Jelly", water="Did", remarks=None, male=0,
             name=None, at=None):
    submitted = (day, at) if at else (date(2026, 10, 30), "09:00")
    return [i, submitted, submitted, "anonymous", day, name, box, dead, feeding, water, remarks, male]


# --- xlsx_read ------------------------------------------------------------------------------

def test_xlsx_read_returns_rows_with_dates_and_times(tmp_path):
    p = make_xlsx(tmp_path / "a.xlsx", {"Form1": [FORM_HEAD, form_row(1, date(2026, 10, 2), "8", at="13:14")]})
    rows = X.read_sheet(p, "Form1")
    assert rows[0] == FORM_HEAD
    assert rows[1][:8] == ["1", "2026-10-02 13:14", "2026-10-02 13:14", "anonymous", "2026-10-02", None, "8", "0"]


def test_xlsx_read_skips_phonetic_runs_in_shared_strings(tmp_path):
    p = make_xlsx(tmp_path / "a.xlsx", {"Form1": [["x"]]})
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    with zipfile.ZipFile(p, "a") as z:
        z.writestr("xl/sharedStrings.xml", '<sst %s><si><t>KANJI</t><rPh sb="0" eb="1"><t>kana</t></rPh></si>'
                   '<si><r><t>A</t></r><r><t>B</t></r></si></sst>' % ns)
        z.writestr("xl/worksheets/sheet2.xml", '<worksheet %s><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c>'
                   '<c r="B1" t="s"><v>1</v></c></row></sheetData></worksheet>' % ns)
    assert X._shared_strings(zipfile.ZipFile(p)) == ["KANJI", "AB"]


def test_xlsx_read_names_a_missing_sheet(tmp_path):
    p = make_xlsx(tmp_path / "a.xlsx", {"Form1": [FORM_HEAD]})
    with pytest.raises(M.InputError, match="Box List"):
        X.read_sheet(p, "Box List")


# --- conversion rules -----------------------------------------------------------------------

@pytest.mark.parametrize("raw,day,expected", [
    ("8", date(2026, 10, 2), "8"),
    ("\uff18", date(2025, 4, 28), "8"),                      # full-width digit
    ("10", date(2026, 5, 12), "10"),                         # before 10q existed
    ("10", SPLIT, "10#1"),
    ("10q", SPLIT, "10#2"), ("10Q", SPLIT, "10#2"), ("10-Q", SPLIT, "10#2"),
    ("mix", date(2026, 1, 1), "Mix"), ("MIX", date(2025, 1, 8), "Mix"), ("miz", date(2025, 4, 30), "Mix"),
    ("13tag_R", date(2025, 6, 2), "13tag-R"), ("13-tag(R)", date(2025, 7, 2), "13tag-R"),
    ("13tag(R)", date(2025, 5, 30), "13tag-R"), ("13T", date(2025, 5, 14), "13tag"),
    ("4 with queen", date(2024, 10, 1), "4"), ("\uff14A", date(2025, 2, 27), "4A"),
    ("6+4B", date(2025, 8, 1), "6+4B"), ("18mix", date(2026, 1, 23), "18mix"),
])
def test_box_names_map_to_colony_names(raw, day, expected):
    assert F.box_name(raw, day, SPLIT) == expected


def test_an_unusable_box_name_is_reported():
    with pytest.raises(M.InputError, match="box"):
        F.box_name("box ?!", date(2026, 1, 1), SPLIT)


@pytest.mark.parametrize("raw,expected", [
    ("Mealworm;Jelly", ("fed", "MW+J")), ("Jelly;Mealworm;", ("fed", "MW+J")),
    ("Mealworm;Jelly;\u306f\u3061\u307f\u3064\uff06Paste", ("fed", "MW+J+HoneyPaste")),
    ("Jelly;Paste", ("fed", "J+Paste")), ("Mealworm;Pellets ;", ("fed", "MW+Pellets")),
    ("Give", ("fed", "given")), ("No", ("not_fed", "")), ("NO", ("not_fed", "")),
])
def test_feeding_codes(raw, expected):
    assert F.feeding(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("0", (0, "")), ("\uff12", (2, "")), ("12", (12, "")), (None, (None, "")),
    ("about 10", (10, "dead: about 10")), ("2 + 12 on sunday", (14, "dead: 2 + 12 on sunday")),
    ("-", (None, "dead: -")),
])
def test_counts(raw, expected):
    assert F.count(raw, "dead") == expected


def test_a_form_row_becomes_feeding_dead_water_and_remark_events():
    row = dict(zip(FORM_HEAD, ["7", "2026-10-02 13:14", "2026-10-02 13:15", "anonymous", "2026-10-02", "Yutaka",
                               "10q", "2", "Mealworm;Jelly", "Did", LEFT + "\u3001" + TRAY, "1"]))
    events = F.convert(row, SPLIT)
    got = [(e["event"], e["what"], e["n"]) for e in events]
    assert got == [("fed", "MW+J", ""), ("dead", "", "1"), ("dead", "male", "1"), ("water", "", ""),
                   ("leftover", "y", ""), ("tray_changed", "", "")]
    first = events[0]
    assert (first["date"], first["time"], first["colony"], first["by"], first["source"]) == (
        "2026-10-02", "13:14", "10#2", "Yutaka", "forms ID 7")
    assert LEFT in first["note"] and "box 10q" in first["note"]
    assert all(e["provisional"] == "" for e in events)


MALE_WORD, FEMALE_WORD, DIED = "\u96c4", "\u96cc", "\u6b7b\u4ea1"       # male, female, died


def dead_events(total, male, remarks):
    row = dict(zip(FORM_HEAD, ["5", None, None, None, "2024-11-01", None, "15", total, "Give", "No", remarks, male]))
    return [(e["what"], e["n"], e["note"]) for e in F.convert(row, SPLIT) if e["event"] == "dead"]


@pytest.mark.parametrize("total,male,remarks,expected", [
    ("5", "1", None, [("", "4", ""), ("male", "1", "")]),                       # total includes males
    ("3", "3", None, [("male", "3", "")]),                                     # all males
    ("23", None, FEMALE_WORD + "17\u3001" + MALE_WORD + "6" + DIED, [("", "17", ""), ("male", "6", "")]),
    ("2", None, MALE_WORD + "\uff12" + DIED, [("male", "2", "")]),              # full-width digit
    ("10", None, "\u30aa\u30b9\u304c1\u5339\u6b7b\u3093\u3067\u3044\u305f", [("", "9", ""), ("male", "1", "")]),
    ("0", None, MALE_WORD + "1\u5339\u51fa\u73fe", []),                       # a male appeared: no death
    ("0", None, "Worms jelly 2 males", []),
    ("1", "2", None, [("male", "2", "dead: 1 in total but 2 males")]),
])
def test_dead_females_are_the_total_minus_the_males(total, male, remarks, expected):
    assert dead_events(total, male, remarks) == expected


def test_time_is_kept_only_when_the_form_was_filled_on_the_day():
    row = dict(zip(FORM_HEAD, ["9", "2024-04-18 16:32", "2024-04-18 16:33", "anonymous", "2024-03-25", None,
                               "5", "1", "Give", "No", None, "0"]))
    events = F.convert(row, SPLIT)
    assert [(e["event"], e["time"]) for e in events] == [("fed", ""), ("dead", "")]


def test_wiped_out_becomes_a_note_event():
    row = dict(zip(FORM_HEAD, ["3", None, None, None, "2024-05-22", None, "5", "0", "Give", "No", WIPED, "0"]))
    assert [(e["event"], e["what"]) for e in F.convert(row, SPLIT)] == [("fed", "given"), ("note", "wiped out")]


# --- the whole import -----------------------------------------------------------------------

BOX_HEAD = ["Box Number", "\u63a1\u53d6\u65e5", "\u63a1\u53d6\u5834\u6240", "\u30b5\u30a4\u30ba(\u5339\u6570)",
            "\u5973\u738b\u30a2\u30ea\u6709\u7121", "\u2642\u30a2\u30ea\u6709\u7121", "remarks"]


def make_colonies_dir(tmp_path):
    d = tmp_path / "_colonies"
    d.mkdir()
    (d / "colonies.csv").write_bytes(
        b"colony,gamergate_tag,collected_on,census_on,census,note,care_from,care_until\r\n"
        b"8,,,,,,2026-10-01,\r\n10#1,,,,,,2026-10-01,\r\n10#2,,,,,,2026-10-01,\r\n"
        b'21,0,,2026-09-28,Y73 G6 R19 Q1 = 99 tagged,"tag 85 unused, x",2026-10-01,\r\n')
    (d / "husbandry_log.csv").write_text(",".join(H.FIELDS) + "\n2026-09-28,,21,census,adult,99,,,,note,,y\n",
                                         encoding="utf-8")
    return d


def make_workbook(tmp_path, extra=()):
    rows = [FORM_HEAD,
            form_row(1, date(2024, 3, 18), "2", dead=1, feeding="Give", water="No"),
            form_row(2, date(2024, 5, 22), "2", feeding="Give", water="No", remarks=WIPED),
            form_row(3, date(2026, 4, 1), "10"),
            form_row(4, date(2026, 5, 13), "10"),
            form_row(5, date(2026, 5, 13), "10q", dead=2),
            form_row(6, date(2026, 10, 2), "8", remarks=LEFT, at="13:14"),
            form_row(7, date(2026, 10, 2), "8", remarks=LEFT, at="13:14")] + list(extra)   # 7 = repeat of 6
    boxes = [BOX_HEAD, [2, date(2024, 3, 18), "CDC, OIST", 56, "yes", "no", "the queen found dead on 2024/03/25"]]
    return make_xlsx(tmp_path / "Check of Ants Box.xlsx", {"Form1": rows, "Box List": boxes})


def test_plan_converts_every_row_and_reports_repeats(tmp_path):
    plan = F.make_plan(make_workbook(tmp_path), make_colonies_dir(tmp_path))
    assert plan.split_day == SPLIT
    assert plan.repeats == ["forms ID 7 repeats forms ID 6 (2026-10-02, box 8)"]
    colonies = {e["colony"] for e in plan.log_rows}
    assert colonies == {"2", "10", "10#1", "10#2", "8"}
    collected = [e for e in plan.log_rows if e["event"] in ("collected", "census")]
    assert [(e["event"], e["date"], e["n"]) for e in collected] == [("collected", "2024-03-18", "56"),
                                                                     ("census", "2024-03-18", "56")]


def test_plan_updates_the_roster(tmp_path):
    plan = F.make_plan(make_workbook(tmp_path), make_colonies_dir(tmp_path))
    rows = {r["colony"]: r for r in plan.colony_rows}
    assert (rows["2"]["care_from"], rows["2"]["care_until"], rows["2"]["collected_on"]) == (
        "2024-03-18", "2024-05-22", "2024-03-18")
    assert "CDC, OIST" in rows["2"]["note"] and "legacy" in rows["2"]["note"]
    assert (rows["10"]["care_from"], rows["10"]["care_until"]) == ("2026-04-01", "2026-05-12")
    assert (rows["10#1"]["care_from"], rows["10#1"]["care_until"]) == ("2026-05-13", "")
    assert (rows["8"]["care_from"], rows["8"]["care_until"]) == ("2026-10-01", "")
    assert rows["21"]["note"] == "tag 85 unused, x"


def test_apply_writes_backups_tables_and_the_workbook_copy(tmp_path):
    d = make_colonies_dir(tmp_path)
    plan = F.make_plan(make_workbook(tmp_path), d)
    F.apply(plan, d, stamp="20261023_120000")
    backup = d / "backups" / "20261023_120000"
    assert (backup / "colonies.csv").is_file() and (backup / "husbandry_log.csv").is_file()
    assert (d / "evidence" / "legacy" / "Check of Ants Box.xlsx").is_file()
    entries = H.read_log(d / "husbandry_log.csv")              # the new log passes all checks
    assert entries[0].source == "note" and any(e.source == "forms ID 6" for e in entries)
    with open(d / "colonies.csv", newline="", encoding="utf-8") as f:
        assert [r["colony"] for r in csv.DictReader(f)] == ["8", "10#1", "10#2", "21", "2", "10"]
    assert b"\r\n" in (d / "colonies.csv").read_bytes()


def test_a_second_import_is_refused(tmp_path):
    d = make_colonies_dir(tmp_path)
    F.apply(F.make_plan(make_workbook(tmp_path), d), d, stamp="a")
    with pytest.raises(M.InputError, match="already"):
        F.make_plan(make_workbook(tmp_path), d)


def test_main_is_a_dry_run_without_yes(tmp_path):
    d = make_colonies_dir(tmp_path)
    before = (d / "husbandry_log.csv").read_bytes()
    assert F.main(["--xlsx", str(make_workbook(tmp_path)), "--colonies-dir", str(d)]) == 0
    assert (d / "husbandry_log.csv").read_bytes() == before
    [log] = list((d / "logs").glob("import_forms_log_*.txt"))
    assert "dry run" in log.read_text(encoding="utf-8")
