"""Tests for photo_decode.py on synthetic close-up photos of A100 tags."""
import sys
from collections import Counter
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import photo_decode as P  # noqa: E402
import synth_scene as S  # noqa: E402

NPZ = P.DEFAULT_NPZ


@pytest.fixture(scope="module")
def dec():
    return P.load_decoder(NPZ)


def found(reads, statuses=("ok", "check")):
    return {r.tag_id for r in reads if r.status in statuses}


def signed_area(q):
    x, y = q[:, 0], q[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def test_reference_bits_follow_opencv_rendering(dec):
    assert dec.refs.shape == (100, 4, 4, 4)
    img = aruco.generateImageMarker(dec.dictionary, 7, 6, borderBits=1)
    np.testing.assert_array_equal(dec.refs[7, 0], (img[1:5, 1:5] < 128).astype(np.uint8))


def test_nearest_exact_and_one_bit_off(dec):
    bits = np.rot90(dec.refs[42, 0], 3).copy()
    tag_id, d, d2, _ = P.nearest(bits, dec.refs)
    assert (tag_id, d) == (42, 0) and d2 >= 3
    bits[1, 2] ^= 1
    tag_id, d, _, _ = P.nearest(bits, dec.refs)
    assert (tag_id, d) == (42, 1)


def test_read_grid_on_a_rendered_marker(dec):
    patch, cell = S.tag_patch(dec.dictionary, 55, 60)
    pad = int(round(S.MARGIN_CELLS * cell))
    m = 6 * cell
    quad = np.float32([[pad, pad], [pad + m, pad], [pad + m, pad + m], [pad, pad + m]])
    g = P.read_grid(patch, quad)
    np.testing.assert_array_equal(g.bits, dec.refs[55, 0])
    assert g.border_black == 1.0 and g.margin_white == 1.0


def test_clockwise_reverses_counterclockwise_quads():
    cw = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])
    np.testing.assert_array_equal(P.clockwise(cw), cw)
    fixed = P.clockwise(cw[::-1].copy())
    assert signed_area(fixed) > 0
    assert sorted(map(tuple, fixed)) == sorted(map(tuple, cw))


def test_decodes_every_tag_in_a_scene(dec):
    tags = [(3, (200, 200), 30, 0, 0.0), (28, (600, 220), 45, 17, 0.05),
            (39, (1100, 250), 60, 33, -0.05), (80, (300, 700), 80, 52, 0.08),
            (84, (800, 750), 100, 71, 0.0), (99, (1300, 800), 140, 88, 0.1)]
    gray = S.scene_with_tags(dec.dictionary, tags)
    reads = P.decode_photo(gray, dec)
    assert found(reads, ("ok",)) == {3, 28, 39, 80, 84, 99}
    assert len([r for r in reads if r.status in ("ok", "check")]) == 6


def test_glare_spot_keeps_the_right_id(dec):
    gray = S.scene_with_tags(dec.dictionary, [(77, (400, 300), 90, 20, 0.0)], shape=(600, 800))
    cv2.circle(gray, (395, 290), 9, 255, -1)
    reads = P.decode_photo(gray, dec)
    assert found(reads) == {77}


def test_small_tag_is_read(dec):
    gray = S.scene_with_tags(dec.dictionary, [(61, (300, 300), 24, 10, 0.0)], shape=(600, 800),
                             blur=0.7)
    assert found(P.decode_photo(gray, dec)) == {61}


def test_clutter_alone_gives_no_ids(dec):
    gray = S.finish(S.background((1200, 1600), seed=5), seed=5)
    assert found(P.decode_photo(gray, dec)) == set()


def test_tag_seen_from_behind_is_flagged(dec):
    gray = S.scene_with_tags(dec.dictionary, [(8, (400, 300), 90, 25, 0.0, True)], shape=(600, 800))
    reads = [r for r in P.decode_photo(gray, dec) if r.status != "unreadable"]
    assert [(r.tag_id, r.status, r.mirrored) for r in reads] == [(8, "check", True)]
    assert "mirrored" in reads[0].alt


def test_touching_tags_are_both_kept(dec):
    side = 70
    tags = [(28, (300, 300), side, 0, 0.0), (39, (300 + 1.45 * side, 300), side, 0, 0.0)]
    gray = S.scene_with_tags(dec.dictionary, tags, shape=(600, 800))
    assert found(P.decode_photo(gray, dec), ("ok",)) == {28, 39}


def test_tag_cut_by_the_image_edge_does_not_crash(dec):
    gray = S.scene_with_tags(dec.dictionary, [(5, (10, 300), 80, 0, 0.0), (6, (400, 300), 80, 0, 0.0)],
                             shape=(600, 800))
    assert 6 in found(P.decode_photo(gray, dec))


@pytest.mark.parametrize("votes,expect", [
    (Counter({(5, 0, False): 4}), (5, "ok", False)),
    (Counter({(5, 0, False): 1}), (5, "check", False)),
    (Counter({(5, 0, False): 4, (9, 1, False): 3}), (5, "check", False)),
    (Counter({(5, 0, False): 8, (9, 1, False): 1}), (5, "ok", False)),
    (Counter({(5, 1, False): 3}), (5, "check", False)),
    (Counter({(5, 0, True): 4}), (5, "check", True)),
    (Counter(), (None, "unreadable", False)),
])
def test_status_rules(votes, expect):
    tag_id, status, _, _, _, mirrored = P.summarize(votes)
    assert (tag_id, status, mirrored) == expect
