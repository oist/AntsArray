"""Tests for generate_cuttable_sheet.py: ID-subset (re-tagging) sheets and ink-spread compensation."""

import re
import struct
import zlib
from collections import Counter
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
import pytest

from generate_cuttable_sheet import (
    COMPARE_SHRINK_MM,
    DEFAULT_DPI,
    default_cols,
    generate_cuttable_sheet,
    generate_cuttable_svg,
    load_custom_dictionary,
    main,
    mm_to_px,
    parse_ids,
    tag_sequence,
)

HERE = Path(__file__).resolve().parent
NPZ_A = HERE / "custom_4x4_A100_d4_20260410_103938.npz"


@pytest.fixture(scope="module")
def dict_a():
    return load_custom_dictionary(NPZ_A)


def detect_ids(png_path, dictionary):
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    params = aruco.DetectorParameters()
    params.errorCorrectionRate = 0.0   # a rendered sheet must read exactly, no bit fixing
    _, ids, _ = aruco.ArucoDetector(dictionary, params).detectMarkers(img)
    return Counter() if ids is None else Counter(int(i) for i in ids.ravel())


def test_parse_ids_accepts_lists_and_ranges_sorted_unique():
    assert parse_ids("85", 100) == [85]
    assert parse_ids("28, 3,13,57-59,3", 100) == [3, 13, 28, 57, 58, 59]


@pytest.mark.parametrize("spec", ["100", "-1", "5-2", "a", "", "3,,4"])
def test_parse_ids_rejects_ids_outside_the_dictionary_or_malformed(spec):
    with pytest.raises(ValueError):
        parse_ids(spec, 100)


def test_tag_sequence_repeats_each_id_consecutively():
    assert tag_sequence(None, 4, copies=1) == [0, 1, 2, 3]
    assert tag_sequence([3, 13], 100, copies=2) == [3, 3, 13, 13]
    with pytest.raises(ValueError):
        tag_sequence([3], 100, copies=0)
    for bad in ([-1], [100]):          # -1 would silently print marker 99
        with pytest.raises(ValueError):
            tag_sequence(bad, 100)


def test_svg_title_is_escaped(tmp_path, dict_a):
    dictionary, n_markers, min_d, _ = dict_a
    out = tmp_path / "t.svg"
    generate_cuttable_svg(dictionary, n_markers, min_d, out, ids=[3], cols=1, title="L&R <test>")
    assert "L&amp;R &lt;test&gt;" in out.read_text(encoding="utf-8")


@pytest.mark.parametrize("copies,cols", [(1, 10), (2, 10), (3, 9), (4, 8), (12, 12)])
def test_default_cols_keeps_all_copies_of_an_id_on_one_row(copies, cols):
    assert default_cols(copies) == cols
    assert default_cols(copies) % copies == 0


def test_subset_png_is_wide_enough_for_its_title(tmp_path, dict_a):
    dictionary, n_markers, min_d, _ = dict_a
    out = tmp_path / "narrow.png"
    title = "20260928/block01 LEFT retag  19 IDs x2  (title must not be clipped)"
    generate_cuttable_sheet(dictionary, n_markers, min_d, out, ids=[3, 3], cols=2, title=title)
    width = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE).shape[1]
    font_scale = mm_to_px(2.0, DEFAULT_DPI) / 30
    title_px = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, DEFAULT_DPI // 300)[0][0]
    assert width >= title_px + 2 * mm_to_px(8.0, DEFAULT_DPI)


def test_function_defaults_print_the_same_2p1mm_tags_as_the_cli(tmp_path, dict_a):
    """Callers that skip marker/margin (tag_roster) must get the tags the lab prints: 1.5 + 2x0.3 mm."""
    dictionary, n_markers, min_d, _ = dict_a
    out = tmp_path / "pitch.png"
    generate_cuttable_sheet(dictionary, n_markers, min_d, out, ids=[3, 13], cols=2)
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    corners, _, _ = aruco.ArucoDetector(dictionary, aruco.DetectorParameters()).detectMarkers(img)
    xs = sorted(float(c[0][:, 0].mean()) for c in corners)
    assert xs[1] - xs[0] == pytest.approx(mm_to_px(2.1 + 1.2, DEFAULT_DPI), abs=2)   # tag + column gap
    svg = tmp_path / "pitch.svg"
    generate_cuttable_svg(dictionary, n_markers, min_d, svg, ids=[3], cols=1)
    assert "Each tag: 2.1mm x 2.1mm" in svg.read_text(encoding="utf-8")


def test_subset_png_prints_exactly_the_requested_ids_and_copies(tmp_path, dict_a):
    dictionary, n_markers, min_d, _ = dict_a
    out = tmp_path / "retag.png"
    seq = tag_sequence([3, 13, 85], n_markers, copies=2)
    generate_cuttable_sheet(dictionary, n_markers, min_d, out, ids=seq, cols=2,
                            title="test LEFT retag")
    assert detect_ids(out, dictionary) == Counter({3: 2, 13: 2, 85: 2})


def test_full_sheet_still_holds_every_id_once(tmp_path, dict_a):
    dictionary, n_markers, min_d, _ = dict_a
    out = tmp_path / "full.png"
    generate_cuttable_sheet(dictionary, n_markers, min_d, out)
    assert detect_ids(out, dictionary) == Counter(range(n_markers))


def test_subset_svg_has_one_marker_and_label_per_tag(tmp_path, dict_a):
    dictionary, n_markers, min_d, _ = dict_a
    out = tmp_path / "retag.svg"
    seq = tag_sequence([7, 40], n_markers, copies=3)
    generate_cuttable_svg(dictionary, n_markers, min_d, out, ids=seq, cols=3,
                          title="test RIGHT retag")
    svg = out.read_text(encoding="utf-8")
    assert svg.count('fill="black"/>') == len(seq)          # one black marker square per tag
    assert svg.count('text-anchor="middle">7</text>') == 3
    assert svg.count('text-anchor="middle">40</text>') == 3
    assert "test RIGHT retag" in svg


# --- ink-spread compensation (black_shrink_mm) ---

def sheet_pair(tmp_path, dictionary, n_markers, min_d, mm, ids=(3, 13, 85)):
    """The same subset sheet as designed and with `mm` black shrink, read as grayscale arrays."""
    out = []
    for shrink in (0.0, mm):
        path = tmp_path / f"s{shrink:g}.png"
        generate_cuttable_sheet(dictionary, n_markers, min_d, path, ids=list(ids), cols=3,
                                title="shrink test", black_shrink_mm=shrink)
        out.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
    return out


def test_png_is_pure_black_and_white_and_records_its_dpi(tmp_path, dict_a):
    dictionary, n_markers, min_d, _ = dict_a
    out = tmp_path / "bw.png"
    generate_cuttable_sheet(dictionary, n_markers, min_d, out, ids=[3, 13], cols=2, title="L&R 0.05")
    assert set(np.unique(cv2.imread(str(out), cv2.IMREAD_UNCHANGED))) == {0, 255}
    png = out.read_bytes()
    at = png.index(b"pHYs")
    (length,) = struct.unpack(">I", png[at - 4:at])
    x_ppm, y_ppm, unit = struct.unpack(">IIB", png[at + 4:at + 13])
    (crc,) = struct.unpack(">I", png[at + 13:at + 17])
    assert length == 9 and crc == zlib.crc32(png[at:at + 13])  # decoders skip a bad ancillary chunk
    assert (x_ppm, y_ppm, unit) == (100000, 100000, 1)        # 2540 dpi, unit = metre
    assert at < png.index(b"IDAT")


def test_shrink_only_removes_marker_black_and_keeps_the_layout(tmp_path, dict_a):
    dictionary, n_markers, min_d, _ = dict_a
    plain, shrunk = sheet_pair(tmp_path, dictionary, n_markers, min_d, 0.05)
    assert plain.shape == shrunk.shape                        # same physical sheet
    body = slice(0, plain.shape[0] - mm_to_px(8 + 2, DEFAULT_DPI))   # above the footer text
    changed = plain[body] != shrunk[body]
    assert np.all(plain[body][changed] == 0) and np.all(shrunk[body][changed] == 255)

    det = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    boxes = {}
    for name, img in (("plain", plain), ("shrunk", shrunk)):
        corners, ids, _ = det.detectMarkers(img)
        boxes[name] = {int(i): c[0] for i, c in zip(ids.ravel(), corners)}
    marker_px, shrink_px = mm_to_px(1.5, DEFAULT_DPI), 5
    inside = np.zeros_like(changed)
    for mid, c in boxes["plain"].items():
        s = boxes["shrunk"][mid]
        assert s.mean(axis=0) == pytest.approx(c.mean(axis=0), abs=0.5)        # tag did not move
        x0, y0 = np.round(c.min(axis=0)).astype(int)
        box = (slice(y0, y0 + marker_px), slice(x0, x0 + marker_px))
        inside[box] = True
        ys, xs = np.nonzero(shrunk[box] == 0)    # pixels, not detector corners (they snap inward)
        assert (xs.min(), xs.max(), ys.min(), ys.max()) == (
            shrink_px, marker_px - 1 - shrink_px, shrink_px, marker_px - 1 - shrink_px)  # 0.05 mm per side
    assert not np.any(changed & ~inside)                      # labels, crop marks, margins untouched


@pytest.mark.parametrize("mm", COMPARE_SHRINK_MM[:3])
def test_shrinks_up_to_0p05_mm_still_read_back_every_id(tmp_path, dict_a, mm):
    dictionary, n_markers, min_d, _ = dict_a
    report = generate_cuttable_sheet(dictionary, n_markers, min_d, tmp_path / "all.png",
                                     black_shrink_mm=mm)
    assert report.readable and report.decoded == n_markers
    assert report.plan.edge_px == round(mm * 100)            # 100 px/mm


def test_a_shrink_too_strong_to_decode_is_reported(tmp_path, dict_a, capsys):
    dictionary, n_markers, min_d, _ = dict_a
    report = generate_cuttable_sheet(dictionary, n_markers, min_d, tmp_path / "x.png",
                                     ids=[0, 3], cols=2, black_shrink_mm=0.08)
    assert not report.readable and report.missing and not report.unexpected
    assert "WARNING: black shrink 0.08 mm" in capsys.readouterr().out


def test_a_dpi_too_coarse_for_the_shrink_is_warned(tmp_path, dict_a, capsys):
    dictionary, n_markers, min_d, _ = dict_a
    report = generate_cuttable_sheet(dictionary, n_markers, min_d, tmp_path / "600.png",
                                     ids=[3], cols=1, dpi=600, black_shrink_mm=0.05)
    assert report.plan.edge_px == 1                           # 1.18 px rounds to one 0.042 mm pixel
    assert "WARNING: at 600 dpi" in capsys.readouterr().out


def svg_rects(svg, fill):
    """(x, y, width, height, rounding attributes) of each marker rect of one colour, not the page."""
    return [(float(x), float(y), float(w), float(h), r) for x, y, w, h, r in re.findall(
        r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"'
        r'((?: rx="[\d.]+" ry="[\d.]+")?) fill="' + fill + '"', svg)]


def test_svg_draws_the_disc_erosion_exactly(tmp_path, dict_a):
    dictionary, n_markers, min_d, _ = dict_a
    rects = {}
    for d in (0.0, 0.05):
        out = tmp_path / f"d{d}.svg"
        generate_cuttable_svg(dictionary, n_markers, min_d, out, ids=[0], cols=1, black_shrink_mm=d)
        svg = out.read_text(encoding="utf-8")
        assert f"Black shrink: {d:.3f}mm" in svg
        rects[d] = (svg_rects(svg, "black"), svg_rects(svg, "white"))
    (black0,), white0 = rects[0.0]
    (black, ), white = rects[0.05]
    d = 0.05
    assert black[:4] == pytest.approx((black0[0] + d, black0[1] + d, 1.5 - 2 * d, 1.5 - 2 * d))
    assert black0[2:4] == (1.5, 1.5) and black0[4] == black[4] == ""
    assert white0 and len(white) == len(white0)
    for (x0, y0, w0, h0, r0), (x, y, w, h, r) in zip(white0, white):
        assert (w0, h0, r0) == (0.25, 0.25, "")
        assert (x, y, w, h) == pytest.approx((x0 - d, y0 - d, w0 + 2 * d, h0 + 2 * d))
        assert r == ' rx="0.05" ry="0.05"'                   # corners of the disc


def test_compare_shrink_writes_one_sheet_per_value_with_preview_and_log(tmp_path):
    main(["--npz", str(NPZ_A), "--output", str(tmp_path / "uv.png"), "--ids", "3", "--compare-shrink"])
    for mm in ("0", "0.03", "0.05", "0.08"):
        png = tmp_path / f"uv_shrink{mm}mm.png"
        assert png.exists() and png.with_suffix(".svg").exists()
    preview = cv2.imread(str(tmp_path / "uv_shrink_preview.png"))
    red = np.all(preview == (0, 0, 255), axis=2)
    assert preview.shape[1] > 4 * 600 and red.sum() > 1000   # removed black shown in red
    assert not red[:, :preview.shape[1] // 4].any()            # ...but not on the design panel
    log = next(tmp_path.glob("generate_cuttable_sheet_log_*.txt")).read_text(encoding="utf-8")
    assert "Black shrink summary" in log and "WARNING: black shrink 0.08 mm" in log


def test_an_invalid_shrink_fails_before_any_sheet_is_written_and_is_logged(tmp_path):
    with pytest.raises(ValueError, match="half a module"):
        main(["--npz", str(NPZ_A), "--output", str(tmp_path / "bad.png"), "--ids", "3",
              "--black-shrink-mm", "0.05", "0.2"])
    assert not list(tmp_path.glob("*.png"))
    log = next(tmp_path.glob("generate_cuttable_sheet_log_*.txt")).read_text(encoding="utf-8")
    assert "ValueError" in log and "half a module" in log


def test_compare_shrink_refuses_explicit_shrinks(tmp_path):
    with pytest.raises(SystemExit):
        main(["--npz", str(NPZ_A), "--output", str(tmp_path / "x.png"), "--compare-shrink",
              "--black-shrink-mm", "0.04"])
