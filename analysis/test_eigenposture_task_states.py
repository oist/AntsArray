"""Tests for missing-bin handling and unsigned motion before aggregation."""

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
    source = Path(__file__).with_name("eigenposture_interactive.py").read_text()
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

    def test_bins_use_common_clips_and_never_fill_missing_with_zero(self):
        make_bins = interactive_helper("bin_features", BIN_MINUTES=5, MIN_CLIPS=3)
        data = np.full((2, 10, 6), np.nan)
        data[0, :3] = [[1, 2, 1, 2, 3, 4], [3, 1, 3, 4, 5, 6], [2, 4, 5, 6, 7, 8]]
        data[0, 3] = [999, 999, np.nan, 1, 1, 1]
        data[0, 5:7] = 2
        before = data.copy()
        binned, counts = make_bins(data)
        np.testing.assert_allclose(binned[0, 0], [3, 4, 3, 4, 5, 6])
        np.testing.assert_array_equal(counts, [[3, 2], [0, 0]])
        self.assertTrue(np.isnan(binned[0, 1]).all())
        self.assertTrue(np.isnan(binned[1]).all())
        np.testing.assert_array_equal(data, before)

    def test_task_fractions_exclude_missing_bins(self):
        proportions = interactive_helper("task_proportions", task_k=3)
        result = proportions(
            np.array([[0, 1, -1, 1], [2, 2, -1, -1], [-1, -1, -1, -1]])
        )
        np.testing.assert_allclose(result[:2], [[1 / 3, 2 / 3, 0], [0, 0, 1]])
        self.assertTrue(np.isnan(result[2]).all())

    def test_transitions_preserve_order_without_crossing_gaps(self):
        transitions = interactive_helper("transition_counts", task_k=2)
        # Equal state fractions, different time organization.
        labels = np.array([[0, 0, 1, 1, -1, 0], [0, 1, 0, 1, -1, 0]])
        counts = transitions(labels)
        np.testing.assert_array_equal(counts[0], [[1, 1], [0, 1]])
        np.testing.assert_array_equal(counts[1], [[0, 2], [1, 0]])
        self.assertEqual(counts.sum(), 6)
        self.assertEqual(transitions(np.full((1, 6), -1)).sum(), 0)

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
