"""Tests for the black ink-spread pre-compensation (ink_spread.py)."""

from pathlib import Path

import cv2.aruco as aruco
import numpy as np
import pytest

from generate_cuttable_sheet import load_custom_dictionary
from ink_spread import ShrinkPlan, check_shrink, shrink_black

HERE = Path(__file__).resolve().parent
NPZ_A = HERE / "custom_4x4_A100_d4_20260410_103938.npz"
MODULE_PX = 25            # 0.25 mm at 2540 dpi: the raster is exactly the vector design
MARKER_PX = 6 * MODULE_PX


@pytest.fixture(scope="module")
def dictionary():
    return load_custom_dictionary(NPZ_A)[0]


def marker(dictionary, mid):
    return aruco.generateImageMarker(dictionary, mid, MARKER_PX)


def exact_erosion(dictionary, mid, shrink_px):
    """Black where a pixel centre lies >= shrink_px from all white, from the module geometry alone."""
    bits = aruco.generateImageMarker(dictionary, mid, 6)          # 1 px per module
    c = np.arange(MARKER_PX) + 0.5
    x, y = np.meshgrid(c, c)
    dist = np.minimum.reduce([x, y, MARKER_PX - x, MARKER_PX - y])  # to the white quiet zone
    for r, col in zip(*np.nonzero(bits == 255)):                  # to each white module square
        x0, y0 = col * MODULE_PX, r * MODULE_PX
        dx = np.maximum.reduce([x0 - x, np.zeros_like(x), x - x0 - MODULE_PX])
        dy = np.maximum.reduce([y0 - y, np.zeros_like(y), y - y0 - MODULE_PX])
        dist = np.minimum(dist, np.hypot(dx, dy))
    return np.where((bits.repeat(MODULE_PX, 0).repeat(MODULE_PX, 1) == 0) & (dist >= shrink_px), 0, 255)


@pytest.mark.parametrize("mid", [0, 3, 57, 85])
@pytest.mark.parametrize("shrink_px", [3.0, 4.72, 5.0, 8.0])
def test_shrink_is_the_exact_disc_erosion_of_the_designed_marker(dictionary, mid, shrink_px):
    np.testing.assert_array_equal(shrink_black(marker(dictionary, mid), shrink_px),
                                  exact_erosion(dictionary, mid, shrink_px))


@pytest.mark.parametrize("shrink_px,moved", [(0.4, 0), (0.71, 1), (1.18, 1), (2.83, 3), (3.0, 3),
                                             (3.5, 3), (4.72, 5), (7.56, 8)])   # a half rounds down
def test_a_straight_edge_moves_in_by_the_rounded_shrink(shrink_px, moved):
    band = np.full((40, 60), 255, np.uint8)
    band[:, 20:40] = 0                                  # a 20 px wide black line
    out = shrink_black(band, shrink_px)
    assert (out[20] == 0).sum() == 20 - 2 * moved       # thinner by the shrink on both sides
    assert ShrinkPlan(shrink_px * 25.4 / 2540, 2540).edge_px == moved


def test_output_is_binary_and_only_ever_turns_black_white(dictionary):
    m = marker(dictionary, 85)
    out = shrink_black(m, 5.0)
    assert set(np.unique(out)) == {0, 255}
    assert np.all(m[out == 0] == 0)                     # no new black anywhere
    assert np.all(out[m == 255] == 255)                 # the white base is never touched


def test_adjacent_black_modules_stay_joined(dictionary):
    out = shrink_black(marker(dictionary, 0), 5.0)
    row = out[MODULE_PX // 2]                           # through the middle of the top border
    assert np.all(row[5:MARKER_PX - 5] == 0)            # one unbroken line across 6 modules
    assert np.all(row[:5] == 255) and np.all(row[MARKER_PX - 5:] == 255)


def test_zero_shrink_returns_the_marker_unchanged(dictionary):
    m = marker(dictionary, 3)
    np.testing.assert_array_equal(shrink_black(m, 0.0), m)


def test_plan_reports_where_the_pixel_grid_puts_the_edge():
    exact = ShrinkPlan(0.03, 2540)
    assert exact.edge_px == 3 and exact.error_mm == pytest.approx(0, abs=1e-9)
    coarse = ShrinkPlan(0.05, 600)                      # 1.18 px -> 1 px = 0.0423 mm
    assert coarse.edge_px == 1 and coarse.error_mm == pytest.approx(-0.0077, abs=1e-4)
    assert ShrinkPlan(0.05, 2540).lone_module_mm(0.25) == pytest.approx((0.15, 0.35))


@pytest.mark.parametrize("mm", [-0.01, 0.125, 0.2, float("nan")])
def test_shrink_that_would_erase_a_lone_module_is_rejected(mm):
    with pytest.raises(ValueError):
        check_shrink(mm, 0.25)
