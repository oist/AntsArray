"""Regression checks for combining cached trip events and behavioral budgets."""
import numpy as np
import pandas as pd

from analysis import long_timescale_tasks as tasks


def test_trip_crossing_bin_boundary_counted_once_time_split():
    template = pd.DataFrame(dict(bin_start_frame=[0, 60, 120, 180], bin_stop_frame=[60, 120, 180, 240]))
    events = pd.DataFrame(dict(exit_frame=[50], duration_minutes=[.5]))
    positions = pd.DataFrame(dict(elapsed_seconds=np.arange(50, 80)))
    observed = np.r_[np.ones(180, dtype=bool), np.zeros(60, dtype=bool)]
    out = tasks.bin_trip_events(template, events, positions, observed, fps=1)
    assert out.trip_count[:3].tolist() == [1, 0, 0]
    assert np.isnan(out.trip_count.iloc[3])
    np.testing.assert_allclose(out.trip_observed_outside_seconds, [10, 20, 0, 0])
    np.testing.assert_allclose(out.trip_rate[:3], [60, 0, 0])
    assert out.trip_duration_minutes.iloc[0] == .5
    assert out.trip_duration_minutes[1:].isna().all()
    np.testing.assert_allclose(out.trip_investment_percent[:3], [100/6, 100/3, 0])


def test_no_completed_trips_is_zero_only_with_observations():
    template = pd.DataFrame(dict(bin_start_frame=[0, 60], bin_stop_frame=[60, 120]))
    out = tasks.bin_trip_events(template, pd.DataFrame(), pd.DataFrame(), np.r_[np.ones(60), np.zeros(60)], fps=1)
    assert out.trip_count.iloc[0] == 0
    assert np.isnan(out.trip_count.iloc[1])
    assert out.trip_duration_minutes.isna().all()


def test_daily_values_keep_counts_and_use_metric_specific_denominators():
    rows = []
    for metric, values, weights in [("trip_count", [1, 3], [1, 1]),
                                    ("trip_duration", [10, 2], [1, 3]),
                                    ("colony", [100, 0], [.1, .9]),
                                    ("sleep", [np.nan, np.nan], [0, 0])]:
        for i in range(2):
            rows.append(dict(side="left", track_id=1, ant="left:001", calendar_date="2026-07-24",
                             phase="light", metric=metric, value=values[i], weight=weights[i], valid_hours=.1))
    out = tasks.daily_profiles(pd.DataFrame(rows)).query("phase == 'all'").set_index("metric")
    assert out.loc["trip_count", "value"] == 4
    assert out.loc["trip_duration", "value"] == 4
    assert out.loc["colony", "value"] == 10
    assert np.isnan(out.loc["sleep", "value"])
