"""Checks for leakage, exposure, missingness, and feature-block balance."""
import unittest

import numpy as np
import pandas as pd

from analysis.behavior_landscape_features import contact_vectors, extract_ant, mapped_source, second_speed, FeatureSettings
from analysis.behavior_landscape_utils import BlockScaler, other_ant_mean, score_predictions, OUTCOMES, select_training_model, ModelSettings
from analysis.behavior_landscape_spatial import frame_histograms


class BehaviorLandscapeTests(unittest.TestCase):
    def test_self_never_contributes_to_cluster_forecast(self):
        values = np.array([[100.], [2.], [4.]])
        predicted = other_ant_mean(values, np.ones((3, 3)))
        np.testing.assert_allclose(predicted.ravel(), [3, 52, 51])
        self.assertTrue(np.isnan(other_ant_mean(values, np.eye(3))).all())

    def test_missing_outcomes_are_not_zero(self):
        values = np.array([[np.nan, 0.], [2., np.nan], [4., 1.]])
        predicted = other_ant_mean(values, np.ones((3, 3)))
        np.testing.assert_allclose(predicted, [[3., 1.], [4., .5], [2., 0.]])

    def test_test_day_cannot_change_fitted_scaling(self):
        train = {"activity": np.array([[0., np.nan], [1., 1.], [2., 2.], [3., 3.], [4., 4.]])}
        scaler = BlockScaler().fit(train, ["activity"])
        before = scaler.transform(train).copy()
        test = {"activity": np.full((5, 2), 10000.)}
        scaler.transform(test)
        np.testing.assert_allclose(before, scaler.transform(train))

    def test_duplicating_a_family_does_not_increase_its_weight(self):
        rng = np.random.default_rng(5)
        a = rng.normal(size=(20, 2))
        b = rng.normal(size=(20, 7))
        original = {"activity": a, "timing": b}
        repeated = {"activity": np.tile(a, (1, 10)), "timing": b}
        x = BlockScaler().fit(original, original).transform(original)
        y = BlockScaler().fit(repeated, repeated).transform(repeated)
        np.testing.assert_allclose(x @ x.T, y @ y.T, atol=1e-12)

    def test_negative_control_shuffled_future_loses_predictability(self):
        # Two clear groups in every outcome, independent of clustering machinery.
        labels = np.repeat([0, 1], 20)
        values = labels[:, None].astype(float)
        train = {name: values.copy() for name in OUTCOMES}
        identity = pd.DataFrame({"track_id": np.arange(40)})
        weights = labels[:, None] == labels[None, :]
        true = pd.DataFrame(score_predictions(train, train, weights, identity, {}))
        shuffle = np.random.default_rng(0).permutation(40)
        null = {name: values[shuffle] for name in OUTCOMES}
        fake = pd.DataFrame(score_predictions(train, null, weights, identity, {}))
        self.assertEqual(true.squared_error.sum(), 0)
        self.assertGreater(fake.squared_error.mean(), fake.baseline_squared_error.mean())

    def test_unobserved_neighbor_target_uses_baseline_instead_of_dropping_target(self):
        values = np.array([[np.nan], [1.], [2.], [3.], [4.]])
        train = {name: values.copy() for name in OUTCOMES}
        test = {name: np.arange(5.)[:, None] for name in OUTCOMES}
        weights = np.zeros((5, 5))
        weights[:, 0] = 1
        identity = pd.DataFrame({"track_id": np.arange(5)})
        result = pd.DataFrame(score_predictions(train, test, weights, identity, {}))
        self.assertTrue(result.n_outcome_features.eq(1).all())
        np.testing.assert_allclose(result.squared_error, result.baseline_squared_error)

    def test_contact_bout_union_does_not_double_count_overlap(self):
        bouts = pd.DataFrame(dict(side=["left"] * 3, ant_a=[1, 1, 7], ant_b=[2, 3, 8],
                                  start_frame=[2, 4, 0], end_frame=[5, 7, 9]))
        selected, active = contact_vectors(bouts, "left", 1, 10, 1)
        self.assertEqual(len(selected), 2)
        np.testing.assert_array_equal(np.flatnonzero(active), np.arange(2, 8))

    def test_global_speed_offset_and_frame_weighted_exposure(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "speed.npy"
            np.save(path, np.array([2., np.nan, 6., 8.]))
            values, counts = second_speed(path, dict(fps=2., frame_min=1, n_frames=4), 4, 2.)
        np.testing.assert_allclose(values[:3], [2., 6., 8.])
        self.assertTrue(np.isnan(values[3]))
        np.testing.assert_array_equal(counts, [1, 1, 1, 0])

    def test_windows_have_disjoint_positions_speed_and_contact_onsets(self):
        import json
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        day, n = 86400, 172800
        with tempfile.TemporaryDirectory() as root:
            block = Path(root)
            name = "TrackID_0001_all_000000_left.parquet"
            context_path = block / "context.npz"
            np.savez(context_path, inside=np.ones(n, np.int8), state=np.zeros(n, np.int8),
                     x_mm=np.repeat([.5, 1.5], day), y_mm=np.full(n, .5),
                     position_count=np.ones(n, np.int16), resource_seconds=np.zeros(n),
                     body_speed=np.repeat([1., 100.], day), antenna_speed=np.ones(n))
            speed_root = block / "stitched/speed_vectors/per_track" / Path(name).stem
            speed_root.mkdir(parents=True)
            np.save(speed_root / "speed_mm_s.npy", np.repeat([1., 100.], day))
            (speed_root / "speed_metadata.json").write_text(json.dumps(dict(fps=1, n_frames=n, frame_min=0)))
            meta_path = block / "grid.json"
            meta_path.write_text(json.dumps(dict(input_x_origin_px=0, y_origin_px=0, mm_per_px=1,
                arena_bounds_px=dict(x_min_px=0, x_max_px=2, y_min_px=0, y_max_px=1))))
            np.save(block / "x.npy", [0., 1., 2.])
            np.save(block / "y.npy", [0., 1.])
            row = SimpleNamespace(track_name=name, side="left", track_id=1, cluster_id="left_0",
                                  x_edges_path=block / "x.npy", y_edges_path=block / "y.npy", metadata_path=meta_path)
            bouts = pd.DataFrame(dict(side=["left"], ant_a=[1], ant_b=[2], start_frame=[day - 2],
                                      end_frame=[day + 2], is_new_onset=[True]))
            daily, profiles, hist = extract_ant(row, block, dict(fps=1, frame_start=0,
                 context_settings=dict(position_bin_seconds=1)), context_path, bouts,
                 {"left": np.ones(n, bool)}, FeatureSettings())
        np.testing.assert_allclose(hist, [[1., 0.], [0., 1.]])
        self.assertEqual([r["speed"] for r in daily], [1., 100.])
        self.assertEqual([r["onsets"] for r in daily], [1, 0])
        self.assertTrue(all(np.isnan(r["median_bout_seconds"]) for r in daily))
        self.assertEqual(len(profiles), 24)

    def test_exact_grid_counts_outside_detections_in_denominator(self):
        meta = dict(input_x_origin_px=100, y_origin_px=200, mm_per_px=.1,
                    arena_bounds_px=dict(x_min_px=100, x_max_px=115, y_min_px=200, y_max_px=215))
        x, y = np.array([100., 115., 118., np.nan]), np.array([200., 215., 210., 202.])
        hist, detected = frame_histograms(np.arange(4), x, y, meta, np.array([0., 1., 2.]),
                                          np.array([0., 1., 2.]), [(0, 4), (0, 2), (2, 4)])
        np.testing.assert_allclose(hist[0], [1/3, 0, 0, 1/3])
        np.testing.assert_allclose(hist[1], [.5, 0, 0, .5])
        np.testing.assert_allclose(hist[2], 0)
        np.testing.assert_array_equal(detected, [3, 2, 1])

    def test_model_selection_uses_only_training_quality(self):
        table = pd.DataFrame(dict(k=[2, 3, 4], admissible=[True, True, False],
                                  stability_median=[.9, .3, .99], silhouette=[.2, .8, .9],
                                  cross_day_ari=[0., 1., 1.]))
        self.assertEqual(select_training_model(table, ModelSettings()), 2)
        table["stability_median"] = 0
        self.assertEqual(select_training_model(table, ModelSettings()), 1)

    def test_cross_block_remapping_is_rejected(self):
        with self.assertRaises(ValueError):
            mapped_source('/stage/20260723/block01/context.npz', '/data/20260724/block01')

    def test_invalid_settings_fail(self):
        with self.assertRaises(ValueError):
            FeatureSettings(profile_bin_seconds=7000).validate()


if __name__ == "__main__":
    unittest.main()
