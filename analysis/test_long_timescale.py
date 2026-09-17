"""Longitudinal time alignment, shared cohorts, denominators and censoring."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import long_timescale_utils as lt


def test_calendar_uses_actual_global_frame_offset_for_window_view():
    info = dict(start_time="2026-07-30 21:48:50", frame_start=3024000, fps=24,
                light_on_hour=5.5, light_off_hour=19.5)
    data = pd.DataFrame(dict(bin_start_frame=[3024000], bin_stop_frame=[3067200],
                             clock_hour=[22.], side=["left"], track_id=[3], outside_percent=[25.]))
    out = lt.calendar_bins(data, info)
    assert out.timestamp.iloc[0] == pd.Timestamp("2026-07-30 22:03:50")
    assert out.calendar_date.iloc[0] == "2026-07-30"
    assert out.colony_percent.iloc[0] == 75
    assert out.phase.iloc[0] == "dark"


def profiles():
    rows = []
    for recording in ("a", "b", "c"):
        for side in ("left", "right"):
            for ant in (1, 2):
                for slot in range(6):
                    value = slot + (10 if side == "right" else 0)
                    if (recording == "a" and slot >= 4) or (ant == 2 and recording == "c"):
                        value = np.nan
                    if slot >= 4 and recording != "a":
                        value = 1000  # Different observed hours must not create drift.
                    rows.append(dict(recording=recording, side=side, track_id=ant, ant=f"{side}:{ant}", metric="speed",
                                     zt_bin=slot, clock_hour=slot+.5, phase="light" if slot < 2 else "dark",
                                     value=value, bin_minutes=60.))
    return pd.DataFrame(rows)


def test_every_recording_shares_identical_slots_and_side_identity():
    options = lt.Settings(min_shared_hours=3, min_phase_hours=2, bootstrap=100)
    standard, slots = lt.standardize_all_recordings(profiles(), ["a", "b", "c"], options)
    included = standard.query("included and phase == 'all'")
    assert set(included.track_id) == {1}
    assert len(included) == 6
    assert included.shared_hours.eq(4).all()
    assert included[included.side.eq("left")].value.eq(1.5).all()
    assert included[included.side.eq("right")].value.eq(11.5).all()
    assert slots.zt_bin.max() == 3
    assert not standard[standard.track_id.eq(2)].included.any()
    assert standard.query("track_id == 1 and phase != 'all'").shared_hours.eq(2).all()


def test_resource_denominator_unknowns_overlap_and_global_offsets():
    context = dict(position_count=np.array([0, 0, 2, 2, 0, 0]))
    bins = pd.DataFrame(dict(bin_start_frame=[0, 4, 8], bin_stop_frame=[4, 8, 12]))
    # Two overlapping water annotations hit the same frame: count it once.
    hits = pd.DataFrame(dict(frame=[4, 4, 5], region_type=["water", "water", "food"]))
    out = lt.bin_resources(context, hits, bins, fps=2, width=2)
    assert out.resource_coverage.tolist() == [0, 1, 0]
    np.testing.assert_allclose(out.water_percent, [np.nan, 25, np.nan], equal_nan=True)
    np.testing.assert_allclose(out.food_percent, [np.nan, 25, np.nan], equal_nan=True)
    with pytest.raises(ValueError, match="align"):
        lt.interval_sum(context["position_count"], [1], [4], 2)


def test_sleep_bouts_touching_unknown_or_recording_edges_are_censored():
    info = dict(start_time="2026-07-23 12:00", frame_start=0, frame_stop=12, fps=1,
                context_settings=dict(position_bin_seconds=1), light_on_hour=5.5, light_off_hour=19.5)
    context = dict(state=np.array([1, 1, 0, 1, 1, 0, -1, 1, 0, 1, 1, 1]))
    out = pd.DataFrame(lt.sleep_bouts(context, info, dict(ant="left:1")))
    assert out.complete.tolist() == [False, True, False, False]
    assert out.duration_seconds.tolist() == [2, 2, 1, 3]
    assert out.left_censored.tolist() == [True, False, True, False]
    assert out.right_censored.tolist() == [False, False, False, True]


def test_phase_requires_hours_per_ant_and_unknown_is_not_inactivity():
    data = pd.DataFrame(dict(side="left", track_id=1, ant="left:1", metric="sleep",
                             calendar_date="2026-07-23", phase="light", full_recording_bin=True, in_recording_bin=True,
                             value=[0., 100., np.nan], valid_hours=[.5, .5, 0.], bin_minutes=60.))
    out = lt.phase_table(data, lt.Settings(min_phase_hours=2, bootstrap=100))
    assert out.value.iloc[0] == 50 and out.covered_clock_hours.iloc[0] == 2
    assert out.valid_hours.iloc[0] == 1
    out = lt.phase_table(data, lt.Settings(min_phase_hours=3, bootstrap=100))
    assert not out.included.iloc[0] and np.isnan(out.value.iloc[0])


def test_slopes_use_actual_days_and_all_recordings():
    options = lt.Settings(min_shared_hours=3, min_phase_hours=2, bootstrap=100)
    standard, _ = lt.standardize_all_recordings(profiles(), ["a", "b", "c"], options)
    standard["value"] += standard.recording.map(dict(a=0, b=2, c=14))
    times = dict(a="2026-07-23", b="2026-07-24", c="2026-07-30")
    ants, _ = lt.longitudinal_slopes(standard, times, options)
    np.testing.assert_allclose(ants.loc[ants.included, "slope_per_day"], 2)
    assert not ants.loc[ants.track_id.eq(2), "included"].any()


def test_discovery_skips_untracked_and_continuous_duplicate(tmp_path):
    for name in ["block01", "block02", "continuous_stitched"]:
        path = tmp_path / "20260723" / name / "stitched/per_track"
        path.mkdir(parents=True)
        if name != "block02":
            (path / "ant.parquet").touch()
    blocks, audit = lt.discover_blocks(tmp_path, ["20260723"])
    assert [b.name for b in blocks] == ["block01"]
    assert len(audit) == 2


def test_clock_folding_equal_cycles_then_equal_ants():
    data = pd.DataFrame(dict(side="left", track_id=[1, 1, 2], ant=["left:1", "left:1", "left:2"],
                             metric="sleep", recording="a", phase="dark", zt_bin=0, clock_hour=.25,
                             full_recording_bin=True, in_recording_bin=True, value=[0, 100, 100], valid_hours=[.25,.5,.5], bin_minutes=30.))
    p = lt.clock_profiles(data)
    assert p.loc[p.track_id.eq(1), "value"].iloc[0] == 50
    summary = lt.summarize_ants(p, ["side", "metric"], lt.Settings(bootstrap=100))
    assert summary["mean"].iloc[0] == 75 and summary.n_ants.iloc[0] == 2


def test_full_49_hour_recording_and_unclustered_ant_survive_display_binning(tmp_path, monkeypatch):
    """A complete central cycle must not suppress the first/last partial days."""
    n = 49 * 3600 + 9 * 60 + 26
    inventory, speeds, sleeps = [], [], []
    for ant in (1, 2):
        name = f"TrackID_{ant:04d}_all_093110_left.parquet"
        speed = np.ones(n, dtype=np.float32)
        state = np.zeros(n, dtype=np.int8)
        if ant == 2:
            speed[:] = np.nan
            state[:] = -1
            speed[1800], state[1800] = 2., 1  # One observation must remain visible.
        speed_path, state_path = tmp_path / f"speed{ant}.npy", tmp_path / f"state{ant}.npy"
        np.save(speed_path, speed)
        np.save(state_path, state)
        identity = dict(side="left", track_id=ant, track_name=name, fps=1., frame_min=0, frame_max=n-1)
        speeds.append(dict(identity, speed_path=speed_path))
        sleeps.append(dict(identity, state_path=state_path))
        inventory.append(dict(side="left", track_id=ant, track_name=name,
                              cluster_id="left_0" if ant == 1 else None, selected=ant == 1))
    monkeypatch.setattr(lt.go, "load_speed_tracks", lambda _: pd.DataFrame(speeds))
    monkeypatch.setattr(lt.sma, "load_sleep_label_tracks", lambda *args: pd.DataFrame(sleeps))
    info = dict(block=str(tmp_path), start_time="2026-07-24 09:31:10", frame_start=0, frame_stop=n,
                fps=1., bin_minutes=30., cached_min_coverage=.5, light_on_hour=5.5, light_off_hour=19.5)
    bins = lt.all_recording_clock_bins(dict(info=info, inventory=pd.DataFrame(inventory)), lambda p: Path(p), 1)
    bins["outside_percent"] = 0.
    bins = lt.calendar_bins(bins, info)
    observed = bins[bins.in_recording_bin & bins.track_id.eq(1)]
    assert observed.timestamp.min() == pd.Timestamp("2026-07-24 09:45")
    assert observed.timestamp.max() == pd.Timestamp("2026-07-26 10:45")
    assert len(observed) == 99
    assert observed.mean_speed_mm_s.eq(1).all()
    assert observed.sleep_percent.eq(0).all()
    assert not observed.complete_cycle.all()
    sparse = bins[bins.track_id.eq(2) & bins.mean_speed_mm_s.notna()]
    assert len(sparse) == 1 and sparse.mean_speed_mm_s.iloc[0] == 2
    assert sparse.sleep_percent.iloc[0] == 100
    assert sparse.speed_coverage.iloc[0] < .001


def test_default_views_keep_partial_bins_and_low_coverage():
    settings = lt.Settings()
    assert settings.min_bin_coverage == settings.min_phase_hours == 0
    bins = pd.DataFrame(dict(side="left", track_id=1, ant="left:1", timestamp=pd.to_datetime(["2026-07-24 09:15", "2026-07-24 09:45"]),
                             calendar_date="2026-07-24", recording="20260724", source_block="block", phase="light",
                             clock_hour=[9.25, 9.75], zt_bin=[7, 8], bin_minutes=30., min_bin_coverage=0.,
                             full_recording_bin=False, in_recording_bin=True))
    for value, coverage, _, _ in lt.METRICS.values():
        bins[value] = [1., np.nan]
        bins[coverage] = [.0001, 0.]
    long = lt.to_long(bins, settings)
    assert long[long.timestamp.eq(pd.Timestamp("2026-07-24 09:15"))].valid.all()
    assert not long[long.timestamp.eq(pd.Timestamp("2026-07-24 09:45"))].valid.any()
    phase = lt.phase_table(long, settings)
    assert phase.included.all() and phase.value.eq(1).all()
