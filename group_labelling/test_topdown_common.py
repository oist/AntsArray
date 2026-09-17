#!/usr/bin/env python3
"""Regression tests for the primitives shared by the top-down entry points.

Run with pytest, or directly -- the sleap-nn environment ships no pytest, so this
file carries a small shim and its own runner::

    /apps/unit/ReiterU/sleap-nn/0.3.1/tools/sleap-nn/bin/python test_topdown_common.py

The video-alignment pair is the reason this file exists. ``align_videos`` and
``match_videos_by_overlap`` answer different questions and were once used
interchangeably, which produced scores that looked plausible while pairing the wrong
ants. ``test_overlap_pair_is_not_containment`` pins the distinction: the same input
that best-overlap accepts must make containment refuse.

Labels are stubbed rather than built with sleap_io. The functions under test only
walk ``labels.videos``, ``frame.video``, ``frame.frame_idx`` and
``frame.user_instances``, and stubbing keeps the tests free of video files.
"""

from __future__ import annotations

import numpy as np

try:
    import pytest
except ModuleNotFoundError:  # the sleap-nn env has no pytest; keep the file runnable
    import contextlib
    import math
    import types

    class _Approx:
        def __init__(self, expected, abs=None, rel=None):  # noqa: A002 - pytest's name
            self.expected, self.abs, self.rel = expected, abs, rel

        def __eq__(self, other):
            if self.abs is not None:
                return abs(other - self.expected) <= self.abs
            return math.isclose(other, self.expected, rel_tol=self.rel or 1e-9, abs_tol=1e-12)

        def __repr__(self):
            return f"approx({self.expected!r})"

    @contextlib.contextmanager
    def _raises(exception):
        try:
            yield
        except exception:
            return
        raise AssertionError(f"did not raise {exception.__name__}")

    pytest = types.SimpleNamespace(approx=_Approx, raises=_raises, main=None)

from topdown_common import (
    align_videos,
    antenna_indices,
    body_frame_offsets,
    classify_points,
    disagreement,
    edge_margin_of,
    match_frame,
    match_videos_by_overlap,
    normalise,
    resolve_nodes,
    stats_of,
)

# The real skeleton, irregular spellings and all. 'aruco ' has a trailing space,
# 'antenna L_2' a space where its siblings use an underscore, 'antenna_L3' none.
NODE_NAMES = [
    "aruco ",
    "occiput",
    "petiole",
    "gaster_tip",
    "antenna_L_1",
    "antenna L_2",
    "antenna_L3",
    "antenna_R_1",
    "antenna R_2",
    "antenna_R3",
]
ARUCO, OCCIPUT, PETIOLE, GASTER = 0, 1, 2, 3
IMAGE_HW = (3036, 4024)


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name


class _Skeleton:
    def __init__(self, names: list[str]) -> None:
        self.nodes = [_Node(name) for name in names]


class _Frame:
    def __init__(self, video: str, frame_idx: int, n_user: int = 1) -> None:
        self.video = video
        self.frame_idx = frame_idx
        self.user_instances = [object() for _ in range(n_user)]


class _Labels:
    def __init__(self, videos: list[str], frames: list[_Frame]) -> None:
        self.videos = videos
        self.skeletons = [_Skeleton(NODE_NAMES)]
        self._frames = frames

    def __iter__(self):
        return iter(self._frames)


def labels_with(video_frames: dict[str, list[int]], n_user: int = 1) -> _Labels:
    videos = list(video_frames)
    frames = [
        _Frame(video, idx, n_user) for video, indices in video_frames.items() for idx in indices
    ]
    return _Labels(videos, frames)


def antenna() -> list[int]:
    return antenna_indices(labels_with({"v": [0]}))


def complete_instance() -> np.ndarray:
    """A fully visible ant in the middle of the frame, pointing up the image."""
    points = np.zeros((len(NODE_NAMES), 2))
    points[GASTER] = [2000, 1600]
    points[PETIOLE] = [2000, 1560]
    points[ARUCO] = [2000, 1540]
    points[OCCIPUT] = [2000, 1500]
    for i in range(4, 10):
        points[i] = [2000 + (i - 6) * 5, 1480]
    return points


# ------------------------------------------------------------- skeleton nodes


def test_normalise_folds_the_irregular_spellings():
    assert normalise("aruco ") == "aruco"
    assert normalise("antenna L_2") == "antenna_l_2"
    assert normalise("antenna_L3") == "antenna_l3"


def test_resolve_nodes_finds_irregular_names_without_rewriting_them():
    labels = labels_with({"v0": [0]})
    assert resolve_nodes(labels, ["aruco", "antenna L_2", "antenna_l3"], "t") == [0, 5, 6]
    # The skeleton itself is untouched: renaming head channels breaks warm start.
    assert [node.name for node in labels.skeletons[0].nodes] == NODE_NAMES


def test_resolve_nodes_rejects_an_unknown_node():
    with pytest.raises(SystemExit):
        resolve_nodes(labels_with({"v0": [0]}), ["thorax"], "t")


def test_antenna_indices_catches_all_six_spellings():
    assert antenna() == [4, 5, 6, 7, 8, 9]


# ------------------------------------------------------------ video alignment


def test_align_videos_accepts_containment():
    # A prediction exists only where a label anchored it, so its frame set is a
    # strict subset of the source's.
    src = labels_with({"a": [1, 2, 3, 4], "b": [10, 11]})
    pred = labels_with({"pa": [1, 2, 3], "pb": [10]})
    assert align_videos(src, pred, "t") == {0: 0, 1: 1}


def test_overlap_pair_is_not_containment():
    """The distinction the two relations exist for; conflating them broke a run.

    Two edited versions of the same labels overlap without either containing the
    other -- a review may empty a frame or add one. Best-overlap must accept it;
    containment must refuse rather than pair the wrong videos.
    """
    before = labels_with({"a": [1, 2, 3], "b": [50, 51, 52]})
    after = labels_with({"a2": [2, 3, 4]})

    assert match_videos_by_overlap(before, after, min_jaccard=0.4) == {0: 0}
    with pytest.raises(SystemExit):
        align_videos(before, after, "t")


def test_match_videos_by_overlap_refuses_below_the_floor():
    before = labels_with({"a": [1, 2, 3]})
    after = labels_with({"a2": [90, 91, 92]})
    with pytest.raises(SystemExit):
        match_videos_by_overlap(before, after, min_jaccard=0.8)


def test_match_videos_by_overlap_refuses_an_ambiguous_pairing():
    before = labels_with({"a": [1, 2, 3], "b": [1, 2, 3]})
    after = labels_with({"a2": [1, 2, 3]})
    with pytest.raises(SystemExit):
        match_videos_by_overlap(before, after, min_jaccard=0.8)


# --------------------------------------------------------- instance matching


def test_match_frame_pairs_by_anchor_not_by_position():
    gt = np.array([[[0.0, 0.0]], [[100.0, 0.0]], [[200.0, 0.0]]])
    pred = np.array([[[200.0, 1.0]], [[0.0, 1.0]], [[100.0, 1.0]]])
    pairing = match_frame(gt, pred, anchor_idx=0, max_dist=10.0)
    assert {k: v[0] for k, v in pairing.items()} == {0: 1, 1: 2, 2: 0}


def test_match_frame_drops_pairs_beyond_the_gate():
    gt = np.array([[[0.0, 0.0]]])
    pred = np.array([[[500.0, 0.0]]])
    assert match_frame(gt, pred, anchor_idx=0, max_dist=100.0) == {}


def test_match_frame_handles_an_empty_side():
    gt = np.array([[[0.0, 0.0]]])
    assert match_frame(gt, np.zeros((0, 1, 2)), anchor_idx=0, max_dist=100.0) == {}


# ------------------------------------------------------------ truncation rule


def test_interior_complete_instance_is_kept():
    assert classify_points(complete_instance(), antenna(), IMAGE_HW, 100.0, 3) is None


def test_near_edge_but_complete_is_kept():
    """45% of near-edge instances carry a full skeleton; position alone over-drops."""
    points = complete_instance()
    points[:, 0] -= 1980  # slide the whole ant to the left border
    assert edge_margin_of(points, IMAGE_HW) < 100.0
    assert classify_points(points, antenna(), IMAGE_HW, 100.0, 3) is None


def test_near_edge_and_hidden_is_truncated():
    points = complete_instance()
    points[:, 0] -= 1980
    points[[4, 5, 6]] = np.nan
    assert classify_points(points, antenna(), IMAGE_HW, 100.0, 3) == "edge_and_hidden"


def test_interior_with_every_antenna_hidden_is_a_label_defect():
    points = complete_instance()
    points[4:10] = np.nan
    assert classify_points(points, antenna(), IMAGE_HW, 100.0, 3) == "centre_no_antennae"


def test_a_fully_hidden_instance_is_rejected():
    points = np.full((len(NODE_NAMES), 2), np.nan)
    assert classify_points(points, antenna(), IMAGE_HW, 100.0, 3) == "no_visible_nodes"


# ---------------------------------------------------------------- body frame


def rotate(points: np.ndarray, degrees: float, about: np.ndarray) -> np.ndarray:
    theta = np.radians(degrees)
    matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (points - about) @ matrix.T + about


def antenna_pair() -> np.ndarray:
    """Head at the origin of the body frame, one node 10px to each side."""
    points = np.full((len(NODE_NAMES), 2), np.nan)
    points[GASTER] = [0.0, 0.0]
    points[OCCIPUT] = [0.0, -100.0]
    points[4] = [10.0, -100.0]
    points[7] = [-10.0, -100.0]
    return points


def test_body_frame_offsets_are_measured_from_the_head_along_the_body_axis():
    offsets = body_frame_offsets(antenna_pair(), [4], (GASTER, OCCIPUT))
    assert offsets[4]["body_length_px"] == pytest.approx(100.0)
    assert offsets[4]["along_px"] == pytest.approx(0.0, abs=1e-9)
    assert abs(offsets[4]["perp_px"]) == pytest.approx(10.0)
    assert abs(offsets[4]["perp_norm"]) == pytest.approx(0.1)


def test_left_and_right_nodes_land_on_opposite_signs():
    offsets = body_frame_offsets(antenna_pair(), [4, 7], (GASTER, OCCIPUT))
    assert offsets[4]["perp_px"] * offsets[7]["perp_px"] < 0


def test_body_frame_is_invariant_to_the_ant_s_orientation():
    """Posture must not leak into the convention measurement."""
    points = antenna_pair()
    points[4] = [10.0, -80.0]
    upright = body_frame_offsets(points, [4], (GASTER, OCCIPUT))

    for degrees in (37.0, 90.0, 213.0):
        turned = body_frame_offsets(
            rotate(points, degrees, np.array([500.0, 500.0])), [4], (GASTER, OCCIPUT)
        )
        assert turned[4]["along_px"] == pytest.approx(upright[4]["along_px"], abs=1e-6)
        assert turned[4]["perp_px"] == pytest.approx(upright[4]["perp_px"], abs=1e-6)


def test_normalised_offsets_are_invariant_to_apparent_size():
    """Two cameras at different apparent scale must not read as a convention gap."""
    points = antenna_pair()
    points[4] = [10.0, -80.0]
    small = body_frame_offsets(points, [4], (GASTER, OCCIPUT))
    large = body_frame_offsets(points * 2.5, [4], (GASTER, OCCIPUT))

    assert large[4]["perp_norm"] == pytest.approx(small[4]["perp_norm"])
    assert large[4]["perp_px"] == pytest.approx(small[4]["perp_px"] * 2.5)


def test_body_frame_is_undefined_when_an_axis_node_is_hidden():
    points = antenna_pair()
    points[OCCIPUT] = np.nan
    assert body_frame_offsets(points, [4], (GASTER, OCCIPUT)) is None


def test_a_hidden_target_node_is_skipped_not_zeroed():
    points = antenna_pair()
    points[4] = np.nan
    offsets = body_frame_offsets(points, [4, 7], (GASTER, OCCIPUT))
    assert 4 not in offsets and 7 in offsets


# ----------------------------------------------------------------- error/PCK


def test_disagreement_weights_the_anchor_higher():
    gt = np.zeros((len(NODE_NAMES), 2))
    pred = np.zeros((len(NODE_NAMES), 2))
    pred[ARUCO] = [3.0, 0.0]
    score, per_node, n_scored = disagreement(gt, pred, [ARUCO, OCCIPUT], [2.0, 1.0])
    assert n_scored == 2
    assert per_node[ARUCO] == 3.0
    assert score == pytest.approx((3.0 * 2.0 + 0.0) / 3.0)


def test_disagreement_ignores_nodes_hidden_on_either_side():
    gt = np.zeros((len(NODE_NAMES), 2))
    pred = np.zeros((len(NODE_NAMES), 2))
    gt[OCCIPUT] = np.nan
    score, _, n_scored = disagreement(gt, pred, [ARUCO, OCCIPUT], [1.0, 1.0])
    assert n_scored == 1 and score == 0.0


def test_disagreement_returns_none_when_nothing_is_comparable():
    gt = np.full((len(NODE_NAMES), 2), np.nan)
    pred = np.zeros((len(NODE_NAMES), 2))
    score, _, n_scored = disagreement(gt, pred, [ARUCO], [1.0])
    assert score is None and n_scored == 0


def test_stats_of_drops_nan_and_reports_pck():
    stats = stats_of(np.array([1.0, 3.0, 9.0, np.nan]), [5.0])
    assert stats["n"] == 3
    assert stats["median"] == 3.0
    # Reported values are rounded for the JSON report, so compare at that precision.
    assert stats["pck@5"] == pytest.approx(2 / 3, abs=1e-4)


def test_stats_of_handles_an_empty_set():
    assert stats_of(np.array([np.nan]), [5.0]) == {"n": 0}


def run_without_pytest() -> int:
    """Definition-order runner for the environment that has no pytest."""
    tests = [
        (name, func)
        for name, func in list(globals().items())
        if name.startswith("test_") and callable(func)
    ]
    failed = []
    for name, func in tests:
        try:
            func()
        except Exception as exc:  # noqa: BLE001 - a test runner reports everything
            failed.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if getattr(pytest, "main", None) is not None:
        raise SystemExit(pytest.main([__file__, "-q"]))
    raise SystemExit(run_without_pytest())
