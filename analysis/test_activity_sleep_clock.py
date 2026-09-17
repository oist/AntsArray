"""Clock matrices preserve physical units, unknowns, cycle boundaries and identity."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis import sleep_motion_analysis_utils as sma


def _inputs(tmp_path):
    clusters, speeds, sleeps = [], [], []
    for side in ("left", "right"):
        name = f"ant_{side}.parquet"
        clusters.append(dict(side=side, TrackID=1, track_name=name, cluster_id=f"{side}_0"))
        speed = np.r_[np.full(86400, 2.0), np.full(86400, 6.0)].astype(np.float32)
        speed[:10800] = np.nan
        speed[21600:43200] = 0
        speed[86400 + 21600:86400 + 43200] = 0
        speed_path = tmp_path / f"speed_{side}.npy"
        np.save(speed_path, speed[3600:])  # Frame offset must be honored, not shifted to zero.
        state = np.r_[np.zeros(86400), np.ones(86400)].astype(np.int8)
        state[:10800] = -1
        state_path = tmp_path / f"state_{side}.npy"
        np.save(state_path, state)
        identity = dict(side=side, track_id=1, track_name=name, fps=1.0, frame_max=172799)
        speeds.append(dict(**identity, frame_min=3600, speed_path=speed_path))
        sleeps.append(dict(**identity, frame_min=0, state_path=state_path))
    return pd.DataFrame(clusters), pd.DataFrame(speeds), pd.DataFrame(sleeps)


def _compute(inputs, **changes):
    options = dict(fps=1.0, recording_stop_frame=172800, start_clock_seconds=19800,
                   light_on_hour=5.5, light_off_hour=19.5, bin_minutes=360,
                   min_bin_coverage=0.5, min_cycles=2, recording_date="20260723", max_workers=1)
    return sma.compute_activity_sleep_clock_profiles(*inputs, **(options | changes))


def test_physical_units_offsets_unknowns_and_equal_cycle_weight(tmp_path):
    bins, profiles, audit = _compute(_inputs(tmp_path))
    assert bins.cycle_label.unique().tolist() == ["Jul 23", "Jul 24"]
    assert bins.zt_hour.unique().tolist() == [3, 9, 15, 21]
    assert bins.clock_hour.unique().tolist() == [8.5, 14.5, 20.5, 2.5]
    first = bins[(bins.side == "left") & (bins.cycle_index == 0) & (bins.zt_bin == 0)].iloc[0]
    assert first.n_speed_frames == first.n_sleep_frames == 10800
    assert first.speed_coverage == first.sleep_coverage == 0.5
    assert first.mean_speed_mm_s == 2
    assert first.sleep_percent == 0  # Unknowns do not count as waking.
    folded = profiles[(profiles.side == "left") & (profiles.zt_bin == 0)].iloc[0]
    assert folded.mean_speed_mm_s == 4  # Equal cycles, not a frame-weighted 4.67.
    assert folded.sleep_percent == 50  # Equal cycles, not a frame-weighted 66.67%.
    assert folded.n_speed_cycles == folded.n_sleep_cycles == 2
    assert profiles.loc[profiles.zt_bin == 1, "mean_speed_mm_s"].eq(0).all()
    assert audit.speed_status.eq("ok").all() and audit.sleep_status.eq("ok").all()


def test_coverage_and_min_cycles_are_applied_per_ant_time_measure(tmp_path):
    inputs = _inputs(tmp_path)
    bins, profiles, _ = _compute(inputs, min_bin_coverage=0.75)
    first = bins[(bins.cycle_index == 0) & (bins.zt_bin == 0)]
    assert first.mean_speed_mm_s.isna().all() and first.sleep_percent.isna().all()
    folded = profiles[profiles.zt_bin == 0]
    assert folded.n_speed_cycles.eq(1).all() and folded.n_sleep_cycles.eq(1).all()
    assert folded.mean_speed_mm_s.isna().all() and folded.sleep_percent.isna().all()
    assert profiles.loc[profiles.zt_bin == 2, "mean_speed_mm_s"].notna().all()


def test_missing_sleep_or_speed_keeps_ant_as_unknown_and_audits(tmp_path):
    clusters, speed, sleep = _inputs(tmp_path)
    Path(speed.loc[speed.side == "right", "speed_path"].item()).unlink()
    bins, profiles, audit = _compute((clusters, speed, None))
    assert profiles.sleep_percent.isna().all()
    assert profiles.loc[profiles.side == "right", "mean_speed_mm_s"].isna().all()
    assert profiles.loc[profiles.side == "left", "mean_speed_mm_s"].notna().all()
    assert audit.sleep_status.eq("missing metadata").all()
    assert audit.loc[audit.side == "right", "speed_status"].item() == "missing vector"
    assert bins.n_sleep_frames.eq(0).all()


def test_partial_cycles_excluded_and_bins_start_exactly_at_lights_on(tmp_path):
    inputs = _inputs(tmp_path)
    bins, profiles, _ = _compute(inputs, start_clock_seconds=14400)  # 04:00; first full cycle starts +5400 s.
    assert bins.cycle_index.unique().tolist() == [0]
    assert bins.bin_start_frame.min() == 5400
    assert bins.bin_stop_frame.max() == 91800
    assert profiles.mean_speed_mm_s.isna().all()  # Cannot infer a two-cycle mean from one.
    with pytest.raises(ValueError, match="No complete"):
        _compute(inputs, recording_stop_frame=3600)


def test_partial_cycle_display_never_uses_unrecorded_frames(tmp_path):
    clusters, speed, sleep = _inputs(tmp_path)
    # Deliberately use vectors extending beyond the requested recording span:
    # those frames must not leak into the partial-cycle display.
    bins, profiles, _ = _compute((clusters.drop(columns="track_name"), speed, sleep),
                                recording_stop_frame=3600, include_partial_cycles=True, min_cycles=1)
    assert not bins.complete_cycle.any()
    assert bins.n_speed_frames.sum() == 0  # All speed in the first hour is missing.
    assert bins.n_sleep_frames.sum() == 0  # Same for classified sleep.
    assert profiles.sleep_percent.isna().all()
    bins, profiles, _ = _compute((clusters, speed, sleep),
                                recording_stop_frame=21600, include_partial_cycles=True, min_cycles=1)
    assert bins.loc[bins.zt_bin > 0, "n_speed_frames"].eq(0).all()
    assert profiles.loc[profiles.zt_bin == 0, "mean_speed_mm_s"].eq(2).all()
    assert profiles.loc[profiles.zt_bin > 0, "mean_speed_mm_s"].isna().all()
    assert profiles.n_speed_cycles.max() == 1


def test_inconsistent_labels_and_frame_rates_raise(tmp_path):
    clusters, speed, sleep = _inputs(tmp_path)
    wrong = sleep.copy()
    wrong.loc[0, "track_name"] = "different_recording.parquet"
    with pytest.raises(ValueError, match="identity mismatch"):
        _compute((clusters, speed, wrong))
    wrong = sleep.copy()
    wrong.loc[0, "fps"] = 24
    with pytest.raises(ValueError, match="inconsistent frame rates"):
        _compute((clusters, speed, wrong))
    np.save(sleep.state_path.iloc[0], np.full(172800, 2, dtype=np.int8))
    with pytest.raises(ValueError, match="Unrecognized sleep-state"):
        _compute((clusters, speed, sleep))


def test_late_recording_window_does_not_count_unrecorded_cycles(tmp_path):
    inputs = _inputs(tmp_path)
    bins, profiles, _ = _compute(inputs, recording_start_frame=86400, min_cycles=1)
    assert bins.cycle_index.unique().tolist() == [1]
    assert bins.cycle_label.unique().tolist() == ["Jul 24"]
    assert bins.recording_hours.eq(24).all()
    assert profiles.n_sleep_cycles.eq(1).all()
    assert profiles.sleep_percent.eq(100).all()
    with pytest.raises(ValueError, match="No complete"):
        _compute(inputs, recording_start_frame=86401)


def test_late_partial_cycle_clips_frames_before_window(tmp_path):
    bins, profiles, _ = _compute(_inputs(tmp_path), recording_start_frame=93600,
                                recording_stop_frame=97200, include_partial_cycles=True,
                                min_cycles=1, min_bin_coverage=0.1)
    assert bins.cycle_index.unique().tolist() == [1]
    first = bins[bins.zt_bin == 0]
    assert first.n_sleep_frames.eq(3600).all()
    assert first.n_speed_frames.eq(3600).all()
    assert bins.loc[bins.zt_bin != 0, "n_sleep_frames"].eq(0).all()
    assert bins.recording_hours.eq(1).all()


def test_cluster_sleep_bins_start_at_observed_window(tmp_path):
    clusters, _, sleep = _inputs(tmp_path)
    sleep = sleep.merge(clusters[["side", "cluster_id"]], on="side")
    summary, bins = sma.cluster_sleep_timeseries(sleep, bin_seconds=3600,
                                                recording_start_frame=86400, recording_stop_frame=93600)
    assert bins.bin_start_frame.unique().tolist() == [86400, 90000]
    assert bins.classified_fraction.eq(1).all()
    assert summary.mean_sleep_fraction.eq(1).all()


def test_clock_matrices_have_shared_scales_cycle_columns_and_no_row_scaling(tmp_path):
    bins, profiles, _ = _compute(_inputs(tmp_path))
    figures = sma.plot_activity_sleep_clock_matrices(bins, profiles, speed_vmax=7)
    try:
        assert len(figures) == 2
        for index, fig in enumerate(figures):
            heatmaps = [ax for ax in fig.axes if ax.images]
            assert len(heatmaps) == 6
            assert all(ax.images[0].get_clim() == (0, 7) for ax in heatmaps[:3])
            assert all(ax.images[0].get_clim() == (0, 100) for ax in heatmaps[3:])
            assert heatmaps[2].images[0].get_array()[0, 0] == 4
            assert "Across-cycle mean" in heatmaps[2].get_title()
            assert list(heatmaps[0].lines[0].get_xdata()) == [14, 14]
            path = tmp_path / f"clock_{index}.png"
            fig.savefig(path, dpi=50)
            assert path.stat().st_size > 1000
    finally:
        for fig in figures:
            plt.close(fig)


def test_grid_contains_clock_matrix_cell_and_uses_existing_saver():
    script = Path(__file__).with_name("grid_occupancy.py").read_text()
    cell = next(cell for cell in script.split("# %%") if "# Individual activity/sleep versus time of day" in cell)
    tree = ast.parse(cell)
    names = [node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert "compute_activity_sleep_clock_profiles" in names
    assert "plot_activity_sleep_clock_matrices" in names
    assert "show" in names
    assert "install_auto_savefig" not in cell  # Do not redirect later figures.


@pytest.mark.parametrize("start_hour,hours,complete,min_cycles,partial", [
    (19.5, 14, 0, 1, True),
    (14.34, 66, 2, 2, False),
    (14.34, 30, 0, 1, True),
    (5.5, 24, 1, 1, False),
    (5.5, 48, 2, 2, False),
])
def test_grid_clock_defaults_follow_available_complete_light_dark_cycles(
    start_hour, hours, complete, min_cycles, partial,
):
    script = Path(__file__).with_name("grid_occupancy.py").read_text()
    cell = next(cell for cell in script.split("# %%") if "# Individual activity/sleep versus time of day" in cell)
    names = {"clock_speed_tracks", "clock_fps", "clock_recording_start_frame", "clock_recording_stop_frame",
             "clock_cycle_offset_seconds", "clock_n_complete_cycles",
             "CLOCK_MATRIX_MIN_CYCLES", "CLOCK_MATRIX_INCLUDE_PARTIAL_CYCLES"}
    assignments = [node for node in ast.parse(cell).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    namespace = dict(math=math, go=SimpleNamespace(load_speed_tracks=lambda _: pd.DataFrame({"fps": [24]})),
                     SPEED_ROOT=None, recording_start_frame=0, recording_stop_frame=int(hours * 3600 * 24),
                     LIGHT_ON_HOUR=5.5, experiment_start_clock_seconds=start_hour * 3600)
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "grid_clock_settings", "exec"), namespace)
    assert namespace["clock_n_complete_cycles"] == complete
    assert namespace["CLOCK_MATRIX_MIN_CYCLES"] == min_cycles
    assert namespace["CLOCK_MATRIX_INCLUDE_PARTIAL_CYCLES"] is partial
