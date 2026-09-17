import json

import numpy as np
import pytest

from tracking.gui.contact_distance import FinishedTrackClip, skeleton_pair_distances, validate_distance


def test_crossing_segments_hit_even_when_all_nodes_are_far_apart():
    poses = {2: np.array([[-10, 0], [10, 0]]), 7: np.array([[0, -10], [0, 10]])}
    row, = skeleton_pair_distances(poses, mm_per_pixel=.016, edges=[(0, 1)])
    assert row.distance_mm == 0
    assert row.is_hit(0)
    assert row.point_a == row.point_b == (0, 0)


def test_threshold_units_and_body_only_contacts():
    poses = {2: np.array([[0, 0], [10, 0]]), 7: np.array([[0, 5], [10, 5]])}
    row, = skeleton_pair_distances(poses, mm_per_pixel=.016, edges=[(0, 1)])
    assert row.distance_mm == pytest.approx(.08)
    assert not row.is_hit(.079)
    assert row.is_hit(.08) and row.is_hit(.1)


def test_missing_nodes_do_not_bridge_edges_or_become_zero_distance():
    poses = {2: np.array([[-10, 0], [np.nan, np.nan], [10, 0]]),
             7: np.array([[0, 0], [np.nan, np.nan], [np.nan, np.nan]]),
             8: np.full((3, 2), np.nan)}
    rows = skeleton_pair_distances(poses, mm_per_pixel=1, edges=[(0, 1), (1, 2)])
    assert rows[0].distance_mm == 10
    assert not rows[0].is_hit(.1)
    assert all(np.isnan(row.distance_mm) and not row.is_hit(100) for row in rows[1:])


def test_no_temporal_or_duplicate_pose_filter_and_no_double_count():
    poses = {2: np.array([[0, 0], [0, 0]]), 7: np.array([[0, 0], [0, 0]])}
    for _ in range(5):
        rows = skeleton_pair_distances(poses, mm_per_pixel=.016, edges=[(0, 1)])
        assert len(rows) == 1
        assert rows[0].is_hit(0)
    poses[7] = np.array([[1000, 0], [1000, 0]])
    row, = skeleton_pair_distances(poses, mm_per_pixel=.016, edges=[(0, 1)])
    assert row.distance_mm == 16
    assert not row.is_hit(.1) and row.is_hit(16)  # No center-radius cutoff.


@pytest.mark.parametrize("value", ["", "bad", "-1", "NaN", "inf", None])
def test_invalid_threshold(value):
    with pytest.raises(ValueError):
        validate_distance(value)


def test_pose_reader_needs_no_interaction_caches(tmp_path):
    manifest = dict(start_frame=100, stop_frame=102, target=2, fps=24)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    np.savez(tmp_path / "geometry.npz", ids=np.array([2, 7]),
             xy=np.zeros((2, 2, 10, 2)), anchors=np.zeros((2, 2, 2)))
    clip = FinishedTrackClip(tmp_path)
    assert clip.ids.tolist() == [2, 7]
    assert clip.start == 100 and clip.stop == 102
    assert not hasattr(clip, "hits") and not hasattr(clip, "trial_bouts")
