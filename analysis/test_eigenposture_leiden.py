"""Measurement tests: no gap bridging, unsigned antennas, and overlap counting."""
import unittest
import numpy as np
from analysis.eigenposture_wavelets import morlet_power, clip_features
from analysis.eigenposture_leiden_features import union_exposure, make_bins, FAMILIES, balanced_matrix


class FeatureTests(unittest.TestCase):
    def test_wavelet_zero_dc_frequency_response_and_missing_support(self):
        fs = 12
        time = np.arange(120) / fs
        waves = np.stack([np.sin(2 * np.pi * f * time) for f in (2, 3, 4)])[:, :, None]
        energy, counts = morlet_power(waves)
        np.testing.assert_array_equal(energy[:, 0].argmax(axis=1), [0, 1, 2])
        constant, _ = morlet_power(np.ones((2, 28, 4)))
        self.assertLess(np.nanmax(constant), 1e-10)
        damaged = np.ones((1, 28, 4));damaged[:, ::2] = np.nan
        power, count = morlet_power(damaged)
        self.assertTrue(np.isnan(power).all())
        self.assertEqual(count.sum(), 0)
        cameras = np.arange(28)[None, :]
        power, count = morlet_power(np.ones((1, 28, 4)), cameras)
        self.assertTrue(np.isnan(power).all())
        self.assertEqual(count.sum(), 0)

    def test_antenna_velocity_unsigned_physical_scale_and_stationary(self):
        x = np.zeros((1, 28, 12))
        x[:, :, 0::2] = np.arange(28)[None, :, None] / 12
        velocity = np.zeros((1, 28, 2))
        cameras = np.zeros((1, 28))
        features, _ = clip_features(x, velocity, cameras, np.zeros(12), np.eye(12), .5)
        np.testing.assert_allclose(features[0, [36, 37, 38, 40, 41]], .5, atol=1e-6)
        self.assertEqual(features[0, 39], 0)
        mirror, _ = clip_features(-x, velocity, cameras, np.zeros(12), np.eye(12), .5)
        np.testing.assert_allclose(mirror[:, 36:], features[:, 36:])
        still, _ = clip_features(x * 0, velocity + 2, cameras, np.zeros(12), np.eye(12), .5)
        np.testing.assert_allclose(still[:, 36:], 0)
        self.assertGreater(still[0, 0], 0)

    def test_bout_union_is_exact_and_does_not_double_count(self):
        result = union_exposure([-4, 2, 9, 9, 29], [1, 8, 12, 16, 35], 3, 10)
        np.testing.assert_allclose(result, [.8, .6, .1])
        np.testing.assert_array_equal(union_exposure([], [], 3, 10), [0, 0, 0])

    def test_bin_support_and_wavelet_missingness(self):
        data = np.full((1, 10, 42), np.nan)
        data[:, :3] = 1
        data[:, :3, 24:36] = np.nan
        data[:, 1, 24:36] = 2
        data[:, 2, 0] = 4
        data[:, 2, 37] = 5
        social = np.zeros((1, 2, 3))
        values, counts, feature_counts = make_bins(data, social, 5)
        self.assertEqual(values.shape, (1, 2, 49))
        self.assertEqual(counts[0, 0], 3)
        np.testing.assert_allclose(values[0, 0, 24:36], 2)
        self.assertEqual(values[0, 0, 0], 4)
        self.assertEqual(values[0, 0, 44], 5)
        self.assertTrue(np.isnan(values[0, 1]).all())
        self.assertTrue((feature_counts[0, 0, 24:36] == 1).all())

    def test_balanced_families_have_equal_variance(self):
        rng = np.random.default_rng(1)
        raw = rng.random((50, 49))
        x, center, scale = balanced_matrix(raw, np.arange(49))
        for family in set(FAMILIES):
            columns = np.array(FAMILIES) == family
            self.assertAlmostEqual(float(x[:, columns].var(axis=0).sum()), 1, places=5)


if __name__ == '__main__':
    unittest.main()
