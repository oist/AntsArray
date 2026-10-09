"""Tests for time support, training-only transforms, and parameter selection."""

import joblib
import numpy as np
import pandas as pd
from analysis.eigenposture_parameters import (
    Config,
    Representation,
    aggregate,
    rank_screen,
)
from analysis.eigenposture_analysis import profiles


def test_hourly_quartiles_reproduce_existing_measurements():
    rng = np.random.default_rng(34)
    minute = rng.normal(size=(3, 1440, 2))
    minute[0, :57] = np.nan
    bins = minute.reshape(3, 24, 60, 2)
    expected = np.nanmean(bins, axis=2)
    expected[np.isfinite(bins).sum(axis=2) < 5] = np.nan
    np.testing.assert_allclose(aggregate(minute, Config()), profiles(expected))


def test_disjoint_time_samples_cannot_leak_values():
    minute = np.ones((2, 1440, 2))
    mask = np.arange(1440) % 2 == 0
    minute[:, ~mask] = 999
    result = aggregate(minute, Config(5, "mean_sd"), mask)
    np.testing.assert_allclose(result[:, :2], 1)
    np.testing.assert_allclose(result[:, 2:], 0)


def test_short_bins_preserve_burst_variability_hidden_by_hourly_means():
    minute = np.zeros((2, 1440, 1))
    minute[:, np.arange(1440) % 60 < 5] = 12
    short = aggregate(minute, Config(5, "mean_sd"))
    long = aggregate(minute, Config(60, "mean_sd"))
    np.testing.assert_allclose(short[:, 0], long[:, 0])
    assert (short[:, 1] > 3).all() and (long[:, 1] == 0).all()


def test_family_weighting_preserves_relative_mode_amplitudes():
    rng = np.random.default_rng(31)
    names = np.array(
        ["forward_mean", "posture_pc1_mean", "posture_pc2_mean", "posture_pc1_rate"]
    )
    fam = np.array(["velocity", "posture", "posture", "dynamics"])
    x = rng.normal(size=(40, 8))
    x[:, [3, 7]] = np.abs(x[:, [3, 7]])
    x[:, 2] *= 0.02
    x[:, 6] *= 0.02
    t = Representation(Config(5, "mean_sd", "family"), names, fam).fit(x)
    assert t.scale[1] == t.scale[2] == t.scale[5] == t.scale[6]
    # Tiny higher-mode amplitude must not get inflated by separate standardization.
    assert np.std(x[:, 2] / t.scale[2]) < np.std(x[:, 1] / t.scale[1]) * 0.05


def test_transform_is_frozen_for_new_ants_and_serializable(tmp_path):
    rng = np.random.default_rng(13)
    names = np.array(["forward_mean", "posture_pc1_mean"])
    fam = np.array(["velocity", "posture"])
    x = rng.normal(size=(30, 4))
    x[0, 0] = np.nan
    t = Representation(Config(5, "mean_sd", "family"), names, fam).fit(x)
    before = t.transform(x)
    query = x.copy()
    query[1:] += 100
    np.testing.assert_allclose(before[:1], t.transform(query)[:1])
    joblib.dump(t, tmp_path / "model.joblib")
    np.testing.assert_allclose(
        before, joblib.load(tmp_path / "model.joblib").transform(x)
    )


def test_screen_rejects_stable_split_without_one_group_baseline_support():
    rows = []
    for config, bic in [("unsupported", -2), ("supported", 3)]:
        for side in ["left", "right"]:
            rows.append(
                dict(
                    config=config,
                    side=side,
                    bic_gain=bic,
                    bootstrap_median=0.9,
                    bootstrap_p10=0.8,
                    minute_ari=0.9,
                    block30_ari=0.9,
                    smallest_group=10,
                )
            )
    ranked = rank_screen(pd.DataFrame(rows))
    assert ranked.iloc[0].config == "supported"
    assert not ranked.loc[ranked.config.eq("unsupported"), "admissible"].iloc[0]
