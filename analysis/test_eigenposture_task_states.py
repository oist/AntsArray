"""Tests for pooled state counts, missing coverage and unsigned motion."""

import ast
from pathlib import Path
import unittest
import warnings

import numpy as np
import pandas as pd

from analysis.eigenposture_unsigned_velocity import unsigned_peaks
from analysis.eigenposture_interactions import count_onsets


def interactive_helper(name, **settings):
    # Import just the tested function: never run data loading, plotting or fitting.
    source = Path(__file__).with_name("full_behavior_roles.py").read_text()
    source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = dict(np=np, warnings=warnings, **settings)
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(__file__), "exec"),
        namespace,
    )
    return namespace[name]


class TaskStateTests(unittest.TestCase):
    def test_unsigned_peaks_precede_reduction_and_share_validity(self):
        velocity = np.full((3, 10, 2), np.nan)
        velocity[0, :4] = [1, -2]
        velocity[0, 4:8] = [-3, 4]
        velocity[0, 8] = [10000, np.nan]  # invalid paired motion sample
        velocity[2, :7] = [2, -5]  # insufficient samples
        before = velocity.copy()
        peaks, counts = unsigned_peaks(velocity)
        np.testing.assert_array_equal(counts, [8, 0, 7])
        np.testing.assert_allclose(peaks[0], [3, 4])
        self.assertTrue(np.isnan(peaks[1:]).all())
        np.testing.assert_array_equal(velocity, before)

    def test_pooled_proportions_keep_missing_observations_unknown(self):
        proportions = interactive_helper("proportions")
        result = proportions(np.array([[2, 6], [0, 0], [3, 0]]))
        np.testing.assert_allclose(result[[0, 2]], [[.25, .75], [1, 0]])
        self.assertTrue(np.isnan(result[1]).all())

    def test_observed_time_counts_replace_equal_hour_weighting(self):
        proportions = interactive_helper("proportions")
        # Ten observed samples of state 0 and two of state 1 count directly,
        # even when they came from hours with different tracking coverage.
        result = proportions(np.array([[10, 2]]))
        np.testing.assert_allclose(result, [[10 / 12, 2 / 12]])

    def test_interactions_count_both_ants_and_preserve_missing_coverage(self):
        ants = pd.DataFrame(dict(side=["left", "left", "right"], track_id=[0, 1, 0]))
        bouts = pd.DataFrame(dict(side=["left"] * 6, ant_a=[0] * 6, ant_b=[1] * 6,
                                  start_frame=[9, 10, 19, 20, 25, 30],
                                  is_new_onset=[True, True, True, True, False, True]))
        coverage = {"left": np.ones(40, bool), "right": np.ones(40, bool)}
        coverage["right"][21] = False
        counts = count_onsets(bouts, ants, coverage, 10, 2, 10, 1)
        np.testing.assert_array_equal(counts[:2], [[2, 1], [2, 1]])
        self.assertEqual(counts[2, 0], 0)
        self.assertTrue(np.isnan(counts[2, 1]))


if __name__ == "__main__":
    unittest.main()
