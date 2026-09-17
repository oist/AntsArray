"""Contact geometry, duplicate suppression, persistence and separation tests."""

import numpy as np
import pandas as pd

from analysis.contact_review_utils import AllPairContactReview, ContactReview, detection_bouts, pair_geometry


def test_reciprocal_frames_and_persistence():
    bouts = detection_bouts([1, 1, 2, 2, 8, 10, 20], gap_frames=2, min_hits=2)
    assert bouts.start_frame.tolist() == [1, 8]
    assert bouts.n_detection_frames.tolist() == [2, 2]
    assert bouts.end_frame.tolist() == [2, 10]


def test_legacy_antenna_to_antenna_is_not_body_contact():
    focal = np.full((1, 10, 2), np.nan)
    partner = focal.copy()
    focal[0, 0] = [0, 0]
    focal[0, 4] = [10, 0]
    partner[0, 0] = [20, 0]
    partner[0, 4] = [10.1, 0]
    distances, endpoints, identical = pair_geometry(focal, partner)
    np.testing.assert_allclose(distances[0], [0.1, 10, 20])
    assert endpoints[0, 0].tolist() == [4, 4]
    assert not identical.item()


def test_missing_tracking_is_not_verified_separation():
    review = ContactReview.__new__(ContactReview)
    review.start, review.stop, review.target, review.fps = 0, 30, 1, 1
    review.ids = np.array([1, 2])
    review.target_index = 0
    review.xy = np.zeros((30, 2, 10, 2))
    review.distances = np.full((30, 2, 3), np.inf)
    review.distances[[5, 6, 15, 16, 25], 1] = 0.05
    review.available = np.ones((30, 2), dtype=bool)
    review.available[14, 1] = False
    review.duplicate = np.zeros((30, 2), dtype=bool)
    review.apply(gap_seconds=2, min_hits=2)
    assert review.trial_bouts.start_frame.tolist() == [5, 15, 25]
    assert review.trial_bouts.counted.tolist() == [True, False, False]
    expected = review.trial_bouts.copy()
    review.apply(gap_seconds=2, min_hits=2)
    pd.testing.assert_frame_equal(review.trial_bouts, expected)
    review.duplicate[5:7, 1] = True
    review.apply(gap_seconds=2, min_hits=2)
    assert 5 not in review.trial_bouts.start_frame.tolist()
    review.duplicate[:] = False
    review.xy[4, 1, 4:] = np.nan
    review.apply(gap_seconds=2, min_hits=2)
    assert not review.trial_bouts.loc[review.trial_bouts.start_frame == 5, "counted"].item()


def test_identical_pose_is_flagged_but_no_data_is_not():
    focal = np.ones((2, 10, 2))
    focal[1] = np.nan
    _, _, duplicate = pair_geometry(focal, focal.copy())
    assert duplicate.tolist() == [True, False]


def test_all_pairs_include_nonfocal_and_tracking_gate_is_optional():
    from types import SimpleNamespace

    r = AllPairContactReview.__new__(AllPairContactReview)
    class Source:
        def apply(self, **kwargs): self.settings = kwargs
    r.source = Source()
    r.ids = np.array([1, 2, 3])
    r.start, r.stop, r.fps, r.target = 100, 130, 1, 1
    r.pair_indices = np.array([[0, 1], [1, 2]])
    r.distances = np.full((30, 2, 3), np.inf)
    r.distances[10:13, 1] = 0.08  # A 2-3 contact, with neither ant the focal ID.
    r.distances[20:23, 0] = 0.2
    r.duplicate = np.zeros((30, 2), dtype=bool)
    r.body_observed = np.ones((30, 2), dtype=bool)
    r.antenna_observed = np.ones((30, 2), dtype=bool)
    r.body_observed[9, 1] = False
    r.apply()
    assert r.trial_bouts[["ant_a", "ant_b"]].values.tolist() == [[2, 3]]
    assert r.trial_bouts.counted.item()
    r.apply(require_observed_separation=True)
    assert not r.trial_bouts.counted.item()
    r.apply(distance_mm=.05)
    assert r.trial_bouts.empty
