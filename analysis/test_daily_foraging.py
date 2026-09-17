"""Daily foraging: light–dark cycles, exposure, paired ants, and cache reuse."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis import trip_phenotyping_utils as trip


def _counts(observed, inside):
    return observed, inside, observed - inside, np.zeros(len(observed)), np.zeros(len(observed))


def _task(n, **changes):
    return dict(fps=1.0, frame_max=n - 1, start_clock_seconds=0.0,
                light_on_hour=5.5, light_off_hour=19.5, **changes)


def test_daily_positions_split_lights_on_and_exclude_partial_cycles(monkeypatch):
    observed = np.ones(86400 * 2, dtype=int)
    inside = observed.copy()
    inside[63000 - 10:63000 + 10] = 0  # Next lights-on, after a noon recording start.
    observed[70000:73600] = 0
    inside[70000:73600] = 0
    monkeypatch.setattr(trip, "_scan_position_counts", lambda task: _counts(observed, inside))
    task = _task(len(observed))
    task["start_clock_seconds"] = 43200
    daily = trip._daily_position_summary(task)
    assert daily.full_day.tolist() == [False, True, False]
    assert daily.available_seconds.tolist() == [63000, 86400, 23400]
    assert daily.outside_seconds.tolist() == [10, 10, 0]
    assert daily.observed_seconds.tolist() == [63000, 82800, 23400]
    assert daily.loc[1, "coverage"] == pytest.approx(23 / 24)
    assert daily.loc[1, "light_coverage"] == pytest.approx(13 / 14)
    assert daily.loc[1, "dark_coverage"] == 1
    assert daily.cycle_start_elapsed_seconds.tolist() == [-23400, 63000, 149400]
    assert daily.cycle_stop_elapsed_seconds.tolist() == [63000, 149400, 235800]


def test_daily_positions_do_not_impute_missing_bins_and_clip_final_bin(monkeypatch):
    observed = np.array([2, 0, 2])
    inside = np.array([1, 0, 0])
    monkeypatch.setattr(trip, "_scan_position_counts", lambda task: _counts(observed, inside))
    task = _task(3)
    task.update(fps=2.0, frame_max=4)  # 2.5 seconds; final position bin is half a second.
    daily = trip._daily_position_summary(task)
    assert daily.available_seconds.item() == 2.5
    assert daily.observed_seconds.item() == 1.5
    assert daily.outside_seconds.item() == 1.0
    assert not daily.full_day.item()


def test_late_window_excludes_unrecorded_days_and_clips_first_bin(monkeypatch):
    observed = np.ones(86400 * 3, dtype=int)
    monkeypatch.setattr(trip, "_scan_position_counts", lambda task: _counts(observed, np.zeros_like(observed)))
    task = _task(len(observed), frame_min=86400 * 2)
    task["start_clock_seconds"] = 19800
    daily = trip._daily_position_summary(task)
    assert daily.day_index.tolist() == [2]
    assert daily.full_day.item()
    assert daily.available_seconds.item() == 86400
    assert daily.observed_seconds.item() == daily.outside_seconds.item() == 86400
    task.update(frame_min=86400 * 2 + 1, fps=2, frame_max=86400 * 2 + 4)
    daily = trip._daily_position_summary(task)
    assert daily.available_seconds.sum() == 2.0
    assert daily.observed_seconds.sum() == 2.0


def test_trip_frequency_uses_window_duration_not_global_frame_stop():
    summary = pd.DataFrame([dict(side="left", track_id=1, track_name="ant", frame_min=2 * 86400,
                                 frame_max=3 * 86400 - 1)])
    trips = pd.DataFrame(dict(track_name=["ant"] * 3, duration_minutes=[1, 2, 3]))
    result = trip.compute_trip_investment_confidence(summary, trips, fps=1, n_bootstrap=20)
    assert result.recording_days.item() == 1
    assert result.completed_trips_per_day.item() == 3


def _regions():
    return pd.DataFrame([
        dict(region_type="colony", shape="rectangle", side=side, name=side,
             tracking_x_min_px=0, tracking_x_max_px=10, tracking_y_min_px=0,
             tracking_y_max_px=10, mm_per_pixel=1)
        for side in ("left", "right")
    ])


def test_daily_loader_zero_days_phase_coverage_and_cache_invalidation(tmp_path, monkeypatch):
    tracks = pd.DataFrame([
        dict(side=side, track_id=1, track_name=f"{side}.parquet", frame_max=172799)
        for side in ("left", "right")
    ])
    for name in tracks.track_name:
        (tmp_path / name).touch()
    trips = pd.DataFrame([dict(track_name="right.parquet", exit_frame=86390, return_frame=86410)])
    scanned = []

    def scan(task):
        scanned.append(task["track_path"])
        observed = np.ones(172800, dtype=int)
        inside = observed.copy()
        inside[86390:86410] = 0
        if "left.parquet" in task["track_path"]:
            # Adequate cycle coverage, but no second-cycle dark observations.
            observed[86400 + 50400:] = 0
            inside[observed == 0] = 0
        return _counts(observed, inside)

    monkeypatch.setattr(trip, "_scan_position_counts", scan)
    kwargs = dict(fps=1.0, start_clock_seconds=19800, light_on_hour=5.5, light_off_hour=19.5,
                  min_coverage=0.5, recording_date="20260515", max_workers=1)
    regions = _regions()

    def load(**changes):
        return trip.load_daily_foraging_investment(
            tracks, trips, regions, tmp_path, tmp_path / "summary", **(kwargs | changes)
        )

    daily = load()
    assert len(scanned) == 2
    assert daily.completed_trip_departures.tolist() == [0, 0, 1, 0]
    assert daily.eligible.tolist() == [True, False, True, True]
    assert daily.day_label.tolist() == ["May 15", "May 16", "May 15", "May 16"]
    assert daily.loc[1, "coverage"] > 0.5
    assert daily.loc[1, "dark_coverage"] == 0
    assert daily.loc[2, "trips_per_observed_day"] == 1
    assert daily.loc[3, "outside_seconds"] == 10  # Time after lights-on is in the next cycle.
    assert daily.loc[0, "outside_hours_per_observed_day"] == pytest.approx(10 / 3600)
    assert daily.day_definition.eq("lights_on_to_lights_on").all()
    assert daily.light_on_hour.eq(5.5).all()
    pd.testing.assert_frame_equal(load(), daily)
    assert len(scanned) == 2
    load(min_coverage=0.9)  # Filtering doesn't invalidate raw observations.
    assert len(scanned) == 2
    regions.loc[regions.side == "left", "tracking_x_max_px"] = 11
    load()
    assert len(scanned) == 3  # Only the changed colony needs rescanning.
    (tmp_path / "right.parquet").write_bytes(b"changed source")
    load()
    assert len(scanned) == 4
    # A cache from the old midnight-based implementation must never be reused.
    cache_meta = tmp_path / "summary/daily_position_cache/right.json"
    old_settings = json.loads(cache_meta.read_text())
    old_settings["version"] = 1
    old_settings.pop("day_definition")
    cache_meta.write_text(json.dumps(old_settings))
    load()
    assert len(scanned) == 5
    load(light_on_hour=6.0)
    assert len(scanned) == 7  # Changed cycle boundaries require rescanning both ants.


def test_predawn_start_and_departures_follow_lights_on_not_midnight(tmp_path, monkeypatch):
    (tmp_path / "left.parquet").touch()
    tracks = pd.DataFrame([dict(side="left", track_id=1, track_name="left.parquet", frame_max=172799)])
    observed = np.ones(172800, dtype=int)
    monkeypatch.setattr(trip, "_scan_position_counts", lambda task: _counts(observed, observed))
    # Recording starts at 04:00. The first lights-on is 05:30 (+5400 s),
    # midnight is +72000 s, and the following lights-on is +91800 s.
    trips = pd.DataFrame(dict(track_name="left.parquet", exit_frame=[5399, 5400, 71999, 72000, 91800]))
    daily = trip.load_daily_foraging_investment(
        tracks, trips, _regions(), tmp_path, tmp_path / "summary", fps=1.0,
        start_clock_seconds=14400, light_on_hour=5.5, light_off_hour=19.5, recording_date="20260515",
    )
    assert daily.day_index.tolist() == [-1, 0, 1]
    assert daily.day_label.tolist() == ["May 14", "May 15", "May 16"]
    assert daily.completed_trip_departures.tolist() == [1, 3, 1]
    assert daily.full_day.tolist() == [False, True, False]
    assert daily.eligible.tolist() == [False, True, False]
    assert daily.available_seconds.sum() == 172800


def test_light_phase_crossing_midnight_stays_in_one_cycle(monkeypatch):
    observed = np.zeros(86400, dtype=int)
    observed[:36000] = 1  # Lights-on 19:30 through lights-off 05:30.
    monkeypatch.setattr(trip, "_scan_position_counts", lambda task: _counts(observed, observed))
    task = _task(len(observed))
    task.update(start_clock_seconds=70200, light_on_hour=19.5, light_off_hour=5.5)
    daily = trip._daily_position_summary(task)
    assert daily.day_index.tolist() == [0]
    assert daily.full_day.item()
    assert daily.light_coverage.item() == 1
    assert daily.dark_coverage.item() == 0
    assert daily.coverage.item() == pytest.approx(10 / 24)


def _paired_daily():
    return pd.DataFrame([
        dict(side=side, track_id=ant, track_name=f"{side}_{ant}", day_index=day,
             day_label=f"Day {day + 1}", full_day=True, eligible=True, min_coverage=0.7,
             light_on_hour=5.5, light_off_hour=19.5,
             outside_hours_per_observed_day=ant if side == "left" or day == 0 else 4 - ant,
             trips_per_observed_day=2 * (ant if side == "left" or day == 0 else 4 - ant))
        for side in ("left", "right") for ant in range(5) for day in (0, 1)
    ])


def test_repeatability_matches_ants_within_colony_and_retains_zeros():
    daily = _paired_daily()
    daily.loc[(daily.track_name == "right_4") & (daily.day_index == 1), "eligible"] = False
    result = trip.compute_daily_foraging_repeatability(daily, n_bootstrap=200)
    assert result.n_matched_ants.tolist() == [5, 5, 4, 4]
    np.testing.assert_allclose(result.spearman_rho, [1, 1, -1, -1])
    np.testing.assert_allclose(result.rho_ci_low, result.spearman_rho)
    np.testing.assert_allclose(result.rho_ci_high, result.spearman_rho)
    pd.testing.assert_frame_equal(result, trip.compute_daily_foraging_repeatability(daily, n_bootstrap=200))


def test_repeatability_constant_values_missing_days_and_small_cohorts():
    daily = _paired_daily()
    daily["trips_per_observed_day"] = 0
    result = trip.compute_daily_foraging_repeatability(daily, n_bootstrap=100)
    constant = result[result.metric == "trips_per_observed_day"]
    assert constant.spearman_rho.isna().all()
    assert constant.status.eq("constant values").all()
    small = trip.compute_daily_foraging_repeatability(daily[daily.track_id < 2], n_bootstrap=100)
    assert small.spearman_rho.isna().all()
    assert trip.compute_daily_foraging_repeatability(daily[daily.day_index == 0]).empty
    daily.loc[daily.day_index == 1, "day_index"] = 2
    assert trip.compute_daily_foraging_repeatability(daily).empty  # Not adjacent days.


def test_daily_figures_handle_missing_values_and_save(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    daily = _paired_daily()
    daily.loc[daily.track_name == "right_4", "eligible"] = False
    initial = set(plt.get_fignums())
    try:
        trip.plot_daily_foraging_investment(daily)
        stats = trip.compute_daily_foraging_repeatability(daily, n_bootstrap=100)
        trip.plot_daily_foraging_repeatability(daily, stats)
        created = set(plt.get_fignums()) - initial
        assert len(created) == 2
        for number in created:
            assert "05:30–05:30" in plt.figure(number)._suptitle.get_text()
            path = tmp_path / f"figure_{number}.png"
            plt.figure(number).savefig(path, dpi=50)
            assert path.stat().st_size > 1000
    finally:
        for number in set(plt.get_fignums()) - initial:
            plt.close(number)


def test_shared_position_scanner_preserves_completed_trip_extraction(tmp_path):
    path = tmp_path / "ant.parquet"
    pd.DataFrame(dict(Frame=np.arange(20), Bodypoint=0, TrackX=[5] * 5 + [15] * 10 + [5] * 5,
                      TrackY=5)).to_parquet(path)
    task = _task(20)
    task.update(track_path=str(path), position_bin_seconds=1, bodypoint=0,
                rectangle=trip.colony_rectangles_from_regions(_regions())["left"],
                max_state_gap_seconds=2, min_state_run_seconds=2, min_trip_seconds=3,
                min_colony_anchor_seconds=3, min_trip_coverage=0.7, max_path_gap_seconds=3,
                trip_entropy_bin_mm=10, side="left", track_id=1, track_name=path.name, cluster_id="left_1")
    trips, positions, diagnostic = trip._scan_track_to_position_bins(task)
    assert trips.exit_frame.tolist() == [5]
    assert trips.return_frame.tolist() == [15]
    assert trips.duration_seconds.tolist() == [10]
    assert positions.elapsed_seconds.tolist() == list(range(5, 15))
    assert diagnostic["n_observed_position_bins"] == 20
