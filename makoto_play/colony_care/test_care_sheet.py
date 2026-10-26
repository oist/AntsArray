"""Tests for care_sheet.py: the monthly set (two feeding sheets + one sheet per colony)."""
import re
import sys
from datetime import date
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import care_sheet as S  # noqa: E402
import sheet_model as M  # noqa: E402
import sheet_pages as P  # noqa: E402

OCT = date(2026, 10, 1)
ARENA = [{"valid_from": "2026-09-28", "left": "22", "right": "21", "note": ""}]
ROSTER = ["8", "Mix", "10#1", "10#2", "14", "15", "17", "20", "22", "23", "24", "25", "26", "27",
          "28", "29"]


def page_count(pdf):
    return len(re.findall(rb"/Type\s*/Page(?!s)", pdf.read_bytes()))


def test_render_writes_two_feeding_sheets_then_one_page_per_colony(tmp_path):
    out = tmp_path / "sheets.pdf"
    S.render(["22", "10#1"], OCT, out, arena=ARENA, png_dir=tmp_path / "png")
    assert page_count(out) == 4
    assert sorted(p.name for p in (tmp_path / "png").iterdir()) == [
        "care_C10.1_2026-10.png", "care_C22_2026-10.png",
        "feeding_2026-10_P1.png", "feeding_2026-10_P2.png"]


def test_render_handles_the_full_roster(tmp_path):
    out = tmp_path / "sheets.pdf"
    S.render(ROSTER, OCT, out, arena=ARENA)
    assert page_count(out) == 2 + len(ROSTER)


def test_feeding_sheet_has_a_column_pair_per_colony_and_room_for_new_ones():
    assert P.feeding_slots(["22", "10#1"]) == ["22", "10#1"] + [""] * (P.FEEDING_SLOTS - 2)
    assert P.feeding_slots(ROSTER) == ROSTER
    with pytest.raises(M.InputError, match="at most"):
        P.feeding_slots([str(n) for n in range(1, P.MAX_FEEDING_COLONIES + 2)])


@pytest.mark.parametrize("name", ["10#1", "Mix", "999#99", "13tag-R", "WWWWWWWW"])
@pytest.mark.parametrize("n_colonies", [16, P.MAX_FEEDING_COLONIES])
def test_feeding_column_names_fit_their_column(name, n_colonies):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    colonies = [name] + [str(k) for k in range(1, n_colonies)]
    fig = P.draw_feeding_page(OCT, 1, colonies, None)
    ax = fig.axes[0]
    renderer = FigureCanvasAgg(fig).get_renderer()
    [label] = [t for t in ax.texts if t.get_text() == name]
    xs = ax.transData.inverted().transform(label.get_window_extent(renderer))[:, 0]
    slot_w = (P.A4_LONG - P.MARGIN - P.MARGIN - sum(w for _, _, w in P.FEEDING_FIXED)) / n_colonies
    assert xs.max() - xs.min() < slot_w - 0.5


def test_colony_sheet_asks_for_new_adults_and_pupae_but_no_adult_count():
    titles = [title for title, _, _ in P.COLONY_COLS]
    assert "Adults" not in titles
    assert "New adults" in titles and "Pupae" in titles


def test_event_types_include_a_full_count_and_fit_the_type_column():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    assert "count" in P.EVENT_TYPES
    fig = P.draw_colony_page("22", OCT, None)
    ax = fig.axes[0]
    renderer = FigureCanvasAgg(fig).get_renderer()
    type_x, type_w = [(x, w) for x, w, t, _ in P._columns(P.MARGIN, P.A4_LONG - P.MARGIN, P.EVENT_COLS)
                      if t == "Type"][0]
    labels = [t for t in ax.texts if t.get_text() == P.EVENT_TYPES[-1]]
    right = max(ax.transData.inverted().transform(t.get_window_extent(renderer))[:, 0].max() for t in labels)
    assert right < type_x + type_w


def test_feeding_sheet_takes_seventeen_colonies():
    colonies = ROSTER[:8] + ["21"] + ROSTER[8:]
    assert P.feeding_slots(colonies) == colonies


def test_colony_title_ends_before_the_header_information():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = P.draw_colony_page("WWWWWWWW", OCT, None)
    ax = fig.axes[0]
    renderer = FigureCanvasAgg(fig).get_renderer()
    [title] = [t for t in ax.texts if t.get_text() == "COLONY WWWWWWWW"]
    assert ax.transData.inverted().transform(title.get_window_extent(renderer))[:, 0].max() < P.INFO_X - 2


def test_render_can_leave_out_the_feeding_sheets(tmp_path):
    out = tmp_path / "sheets.pdf"
    S.render(["22"], OCT, out, feeding=False)
    assert page_count(out) == 1


def test_render_needs_at_least_one_page(tmp_path):
    with pytest.raises(M.InputError, match="nothing to print"):
        S.render([], OCT, tmp_path / "s.pdf", feeding=False)


@pytest.mark.parametrize("name,expected", [
    ("care_C10.1_2026-10.png", M.SheetId("colony", "10.1", OCT, None)),
    ("feeding_2026-10_P2.png", M.SheetId("feeding", None, OCT, 2)),
])
def test_rendered_sheets_are_machine_readable(tmp_path, name, expected):
    import cv2
    S.render(["10#1"], OCT, tmp_path / "s.pdf", png_dir=tmp_path)
    img = cv2.imread(str(tmp_path / name))
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    assert M.parse_sheet_code(text) == expected
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                       cv2.aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert sorted(ids.ravel().tolist()) == [0, 1, 2, 3]


@pytest.mark.parametrize("text", [
    "IN ARENA: right half since 2026-09-28 - write times to the minute",
    "ARENA: 10/01 out, 10/09 right half, 10/16 out, 10/23 left half - write times to the minute while in",
    "ARENA: " + ", ".join("10/%02d right half" % d for d in range(1, 30)) + " - write times",
])
def test_colony_arena_line_stays_clear_of_the_qr_code_and_readable(text):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = P.draw_colony_page("22", OCT, (text, True))
    ax = fig.axes[0]
    renderer = FigureCanvasAgg(fig).get_renderer()
    [line] = [t for t in ax.texts if t.get_text().replace("\n", " ") == text]
    right_mm = ax.transData.inverted().transform(line.get_window_extent(renderer))[:, 0].max()
    qr_x = P.A4_LONG - P.MARGIN - P.MARKER - P.QR_GAP - P.QR
    assert right_mm < qr_x - 1.0
    assert line.get_fontsize() >= P.MIN_STATUS_PT


def test_render_leaves_no_pdf_when_a_page_fails(tmp_path, monkeypatch):
    real = P.draw_colony_page

    def fail_on_22(colony, *args):
        if colony == "22":
            raise RuntimeError("boom")
        return real(colony, *args)

    monkeypatch.setattr(P, "draw_colony_page", fail_on_22)
    with pytest.raises(RuntimeError):
        S.render(["8", "22"], OCT, tmp_path / "s.pdf")
    assert list(tmp_path.iterdir()) == []


def test_render_keeps_the_new_pdf_when_the_old_one_cannot_be_replaced(tmp_path, monkeypatch):
    out = tmp_path / "s.pdf"
    out.write_bytes(b"old")
    real_replace = Path.replace

    def locked(self, target):
        if Path(target) == out:
            raise PermissionError("in use")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", locked)
    with pytest.raises(OSError, match="kept as .*s.pdf.part"):
        S.render(["22"], OCT, out)
    assert out.read_bytes() == b"old"
    assert page_count(tmp_path / "s.pdf.part") == 3


# --- command line ---------------------------------------------------------------------------

def logs(tmp_path):
    found = list(tmp_path.glob("care_sheet_log_*.txt"))
    assert len(found) == 1
    return found[0].read_text(encoding="utf-8")


def write_tables(tmp_path, roster="8,,,,,,2026-10-01,\nMix,,,,,,2026-10-01,\n22,0,,,,,2026-10-01,\n"
                                  "21,0,,,,,,\n"):
    (tmp_path / "arena.csv").write_text("valid_from,left,right,note\n2026-09-28,22,21,\n", encoding="utf-8")
    (tmp_path / "colonies.csv").write_text(
        "colony,gamergate_tag,collected_on,census_on,census,note,care_from,care_until\n" + roster,
        encoding="utf-8")


def test_main_prints_the_roster_of_the_month_and_logs(tmp_path):
    write_tables(tmp_path)
    out = tmp_path / "s.pdf"
    assert S.main(["--month", "2026-10", "--out", str(out), "--colonies-dir", str(tmp_path)]) == 0
    assert page_count(out) == 2 + 3
    log = logs(tmp_path)
    assert "colony sheets: 8, Mix, 22" in log
    assert "colony 22: IN ARENA: left half" in log
    assert "WARNING: colony 21 is in the arena" in log


def test_main_colonies_option_overrides_the_roster(tmp_path):
    write_tables(tmp_path)
    out = tmp_path / "s.pdf"
    assert S.main(["--month", "2026-10", "--colonies", "10#1", "--out", str(out),
                   "--colonies-dir", str(tmp_path)]) == 0
    assert page_count(out) == 3


def test_main_without_any_colony_prints_only_the_feeding_sheets(tmp_path):
    out = tmp_path / "s.pdf"
    assert S.main(["--month", "2026-10", "--out", str(out)]) == 0
    assert page_count(out) == 2


def test_main_logs_the_traceback_of_an_unexpected_error(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(S, "render", broken)
    with pytest.raises(RuntimeError):
        S.main(["--month", "2026-10", "--out", str(tmp_path / "s.pdf")])
    assert "Traceback" in logs(tmp_path) and "RuntimeError: bug" in logs(tmp_path)


@pytest.mark.parametrize("case,message", [
    ("no_arena", "arena.csv"),
    ("no_pages", "nothing to print"),
    ("bad_month", "2026-13"),
])
def test_main_fails_and_logs_errors(tmp_path, case, message):
    out = tmp_path / "s.pdf"
    args = {"no_arena": ["--colonies", "22", "--colonies-dir", str(tmp_path / "nowhere")],
            "no_pages": ["--no-feeding"],
            "bad_month": ["--month", "2026-13", "--colonies", "22"]}[case]
    if case != "bad_month":
        args += ["--month", "2026-10"]
    assert S.main(["--out", str(out)] + args) == 2
    assert not out.exists()
    log = logs(tmp_path)
    assert "error:" in log and message in log
