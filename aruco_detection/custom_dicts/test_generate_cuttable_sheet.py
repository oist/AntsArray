"""Tests for the ID-subset (re-tagging) mode of generate_cuttable_sheet.py."""

from collections import Counter
from pathlib import Path

import cv2
import cv2.aruco as aruco
import pytest

from generate_cuttable_sheet import (
    default_cols,
    generate_cuttable_sheet,
    generate_cuttable_svg,
    load_custom_dictionary,
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
    font_scale = mm_to_px(2.0, 600) / 30
    title_px = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]
    assert width >= title_px + 2 * mm_to_px(8.0, 600)


def test_function_defaults_print_the_same_2p1mm_tags_as_the_cli(tmp_path, dict_a):
    """Callers that skip marker/margin (tag_roster) must get the tags the lab prints: 1.5 + 2x0.3 mm."""
    dictionary, n_markers, min_d, _ = dict_a
    out = tmp_path / "pitch.png"
    generate_cuttable_sheet(dictionary, n_markers, min_d, out, ids=[3, 13], cols=2)
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    corners, _, _ = aruco.ArucoDetector(dictionary, aruco.DetectorParameters()).detectMarkers(img)
    xs = sorted(float(c[0][:, 0].mean()) for c in corners)
    assert xs[1] - xs[0] == pytest.approx(mm_to_px(2.1 + 1.2, 600), abs=2)   # tag + column gap
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
