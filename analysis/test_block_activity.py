"""Clock matching, identity, missingness and cache-only worker comparisons."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis import block_activity_utils as ba


SETTINGS = ba.ComparisonSettings(min_matched_hours=1, n_bootstrap=100)


def _bins(values=(1., 2., 3.), *, side="left", track_id=1, coverage=1., full=True):
    n = len(values)
    result = pd.DataFrame(dict(side=side, track_id=track_id, cycle_index=0,
                               zt_bin=np.arange(n), zt_hour=(np.arange(n)+.5)/2,
                               clock_hour=5.5+(np.arange(n)+.5)/2, bin_minutes=30.,
                               bin_start_frame=np.arange(n)*43200,
                               bin_stop_frame=(np.arange(n)+1)*43200,
                               full_recording_bin=full, min_bin_coverage=.5))
    for value_col, coverage_col, _, _ in ba.METRICS.values():
        result[value_col] = values
        result[coverage_col] = coverage
    return result


def _compare(early, late, settings=SETTINGS):
    return ba.pair_profiles(ba.fold_profiles(early, settings), ba.fold_profiles(late, settings), settings)


def test_clock_matching_removes_unequal_recording_hour_confound():
    early = _bins([1., 2., np.nan])
    late = _bins([1., 2., 90.])
    paired, ants = _compare(early, late)
    assert ants.included.all()
    assert ants.delta.eq(0).all()
    assert ants.early.eq(1.5).all()
    assert ants.unmatched_clock_mean_late.eq(31).all()
    assert ants.matched_clock_hours.eq(1).all()
    assert ants.matched_valid_hours_late.eq(1).all()
    assert not paired.loc[paired.zt_bin.eq(2), "matched"].any()


def test_identity_is_side_and_id_not_filename_or_cluster_number():
    a = pd.concat([_bins([1, 2], side="left"), _bins([10, 20], side="right")])
    b = pd.concat([_bins([3, 4], side="left"), _bins([7, 17], side="right")])
    a["track_name"], a["cluster_id"] = "timestamp_early.parquet", "cluster_2"
    b["track_name"], b["cluster_id"] = "timestamp_late.parquet", "cluster_9"
    _, ants = _compare(a, b)
    assert ants.loc[ants.side.eq("left"), "delta"].eq(2).all()
    assert ants.loc[ants.side.eq("right"), "delta"].eq(-3).all()
    profile = ba.fold_profiles(a, SETTINGS)
    with pytest.raises(ValueError, match="Duplicate"):
        ba.pair_profiles(pd.concat([profile, profile]), profile, SETTINGS)


def test_partial_bins_unknowns_and_metric_specific_coverage():
    a = _bins([1., 2., 99.], full=[True, True, False])
    b = _bins([1., 3., 99.])
    a.loc[0, "sleep_coverage"] = .49
    _, ants = _compare(a, b)
    sleep = ants[ants.metric.eq("sleep")].iloc[0]
    assert not sleep.included and sleep.matched_clock_hours == .5
    assert ants.loc[ants.metric.ne("sleep"), "included"].all()
    assert ants.loc[ants.metric.ne("sleep"), "delta"].eq(.5).all()
    assert ants.loc[ants.metric.ne("sleep"), "n_matched_bins"].eq(2).all()
    with pytest.raises(ValueError, match="below the cached"):
        ba.fold_profiles(a, replace(SETTINGS, min_bin_coverage=.25))


def test_equal_clock_bin_and_cycle_weight_not_observation_weight():
    a = _bins([0., 10.], coverage=[.5, 1.])
    b = _bins([0., 10.], coverage=[1., .5])
    _, ants = _compare(a, b)
    assert ants.early.eq(5).all() and ants.late.eq(5).all()
    assert ants.matched_valid_hours_early.eq(.75).all()
    second = a.copy()
    second["cycle_index"] = 1
    second["mean_speed_mm_s"] = [10., 0.]
    profiles = ba.fold_profiles(pd.concat([a, second]), SETTINGS)
    speed = profiles[profiles.metric.eq("speed")]
    assert speed.value.eq(5).all() and speed.n_cycles.eq(2).all()


def test_context_binning_preserves_global_offsets_and_unknowns():
    template = pd.DataFrame(dict(bin_start_frame=[-4, 0, 4, 8, 12], bin_stop_frame=[0, 4, 8, 12, 16]))
    context = dict(body_speed=np.array([np.nan, np.nan, 2., 4., np.nan, 6.]),
                   antenna_speed=np.array([np.nan, np.nan, 0., 0., np.nan, np.nan]),
                   inside=np.array([-1, -1, 0, 1, -1, 0]))
    result = ba.bin_context(context, template, bin_frames=2, n_context_bins=6)
    np.testing.assert_allclose(result.body_motion_mm_s, [np.nan, np.nan, 3, 6, np.nan], equal_nan=True)
    np.testing.assert_allclose(result.body_coverage, [0, 0, 1, .5, 0])
    np.testing.assert_allclose(result.outside_percent, [np.nan, np.nan, 50, 100, np.nan], equal_nan=True)
    assert result.antenna_motion_mm_s.iloc[2] == 0  # Genuine stillness is retained.
    with pytest.raises(ValueError, match="align"):
        ba.bin_context(context, template + 1, bin_frames=2, n_context_bins=6)
    with pytest.raises(ValueError, match="span"):
        ba.bin_context(context, template, bin_frames=2, n_context_bins=5)
    context["inside"][0] = 2
    with pytest.raises(ValueError, match="Unrecognized"):
        ba.bin_context(context, template, bin_frames=2, n_context_bins=6)


def test_bootstrap_pairs_and_handles_ties_and_small_cohorts():
    a = pd.concat([_bins([i, i+1], track_id=i) for i in range(6)])
    b = pd.concat([_bins([i+2, i+3], track_id=i) for i in range(6)])
    _, ants = _compare(a, b)
    result = ba.summarize_changes(ants, SETTINGS)
    left = result[result.side.eq("left")]
    assert left.mean_delta.eq(2).all()
    assert left.delta_ci_low.eq(2).all() and left.delta_ci_high.eq(2).all()
    np.testing.assert_allclose(left.spearman_rho, 1)
    pd.testing.assert_frame_equal(result, ba.summarize_changes(ants, SETTINGS))
    assert result.loc[result.side.eq("right"), "n_ants"].eq(0).all()
    assert np.isnan(ba._bootstrap_rho(np.ones((2, 3)), np.ones((2, 3)))).all()
    small = ba.summarize_changes(ants[ants.track_id.le(3)], SETTINGS)
    assert small.delta_ci_low.isna().all()


def test_identity_audit_keeps_unselected_workers_and_arbitrary_cluster_labels():
    a = pd.DataFrame(dict(side="left", track_id=[1, 2, 3, 4], selected=[True, True, True, False],
                          cluster_id=["left_0", "left_0", "left_1", None], detection_fraction=[.8]*4))
    b = pd.DataFrame(dict(side="left", track_id=[1, 2, 3, 5], selected=True,
                          cluster_id=["left_9", "left_9", "left_8", "left_8"], detection_fraction=[.9]*4))
    _, ants = _compare(_bins(), _bins())
    audit = ba.identity_audit(a, b, ants)
    assert audit.track_id.tolist() == [1, 2, 3, 4, 5]
    assert audit.selected_both.sum() == 3
    assert audit.loc[audit.track_id.eq(4), "detection_fraction_late"].isna().all()
    counts, scores = ba.cluster_overlap(audit)
    assert counts.n_ants.sum() == 3
    assert scores.adjusted_rand_index.item() == 1  # Renumbering is not switching.


def _window_fixture(tmp_path, monkeypatch):
    block = tmp_path / "20260810" / "block"
    grid = block / "stitched" / "grid_occupancy_histograms_arena"
    name = "TrackID_0001_all_184336_left.parquet"
    regions = pd.DataFrame(dict(region_id=[1], region_type=["colony"], side=["left"]))
    bounds = {"left": {"x": [0, 10], "y": [0, 10]}}

    def write(relative, data):
        path = block / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    region_path = write("regions.csv", {})
    monkeypatch.setattr(ba.go, "panorama_regions_path", lambda _: region_path)
    monkeypatch.setattr(ba.rs, "panorama_regions_path", lambda _: region_path)
    monkeypatch.setattr(ba.go, "load_panorama_regions", lambda _: regions)
    monkeypatch.setattr(ba.arena, "arena_bounds_from_regions", lambda _: bounds)
    grid_meta = write("grid.json", dict(frame_min=0, frame_max=86399, n_detected_frames=80000,
                                       side="left", track_id=1, track_name=name,
                                       bounds_source="panorama_arena_regions", arena_bounds_px=bounds["left"]))
    monkeypatch.setattr(ba.go, "metadata_paths", lambda _: [grid_meta])
    track_path = write(f"stitched/per_track/{name}", {})
    motion_path = write(f"stitched/sleep_motion/per_track/{Path(name).stem}/{ba.rs.sm.METADATA_FILENAME}", {})
    classifier = {"body_percentile": 75}
    label_meta = write("labels/metadata.json", {"classifier_parameters": classifier})
    state_path = label_meta.with_name("state.npy")
    np.save(state_path, np.zeros(86400, dtype=np.int8))
    labels = pd.DataFrame([dict(side="left", track_id=1, track_name=name, fps=24.,
                                frame_min=0, frame_max=86399, metadata_path=str(label_meta),
                                state_path=str(state_path), classifier_parameters=json.dumps(classifier))])
    monkeypatch.setattr(ba.sma, "load_sleep_label_tracks", lambda _: labels)
    settings = ba.rs.ReturnSettings()
    write("stitched/grid_occupancy_histograms_arena/sleep_motion_analysis/return_response/settings.json",
          dict(settings=asdict(settings), sleep_classifier_parameters=classifier))
    pd.DataFrame([dict(side="left", TrackID=1, track_name=name, cluster_id="left_0")]).to_csv(grid / "track_cluster_ids.csv", index=False)
    speed_meta = write(f"stitched/speed_vectors/per_track/{Path(name).stem}/speed_metadata.json",
                       dict(track_name=name, fps=24., mm_per_px=.016, bodypoint_filter=0,
                            max_interp_gap_frames=5, smooth_sigma_frames=2., max_speed_mm_s=5.,
                            speed_path="/flash/not/available/speed_mm_s.npy"))
    np.save(speed_meta.with_name("speed_mm_s.npy"), np.ones(86400, dtype=np.float32))
    bins = _bins([1., 2.])
    bins["track_name"], bins["cluster_id"] = name, "left_0"
    bins["light_on_hour"], bins["light_off_hour"] = 5.5, 19.5
    clock = grid / "activity_sleep_clock" / "ant_cycle_time_bins.parquet"
    clock.parent.mkdir(parents=True)
    bins.to_parquet(clock)
    row = next(labels.itertuples())
    root = block / "stitched" / "analysis_cache" / "return_sleep"
    context, crossing = ba.rs.context_cache_paths(row, block, regions, settings, 3600, root)
    context.parent.mkdir(parents=True)
    np.savez_compressed(context, body_speed=np.ones(3600), antenna_speed=np.ones(3600)*2, inside=np.ones(3600))
    return SimpleNamespace(block=block, row=row, root=root, regions=regions, settings=settings,
                           context=context, crossing=crossing, track=track_path, motion=motion_path,
                           label_meta=label_meta, region_path=region_path, speed_meta=speed_meta, clock=clock)


def test_context_resolver_preserves_existing_key_and_invalidates_changes(tmp_path, monkeypatch):
    f = _window_fixture(tmp_path, monkeypatch)
    sources = dict(version=2, fps=24., bin_frames=24, min_position_fraction=.25, n_bins=3600,
                   regions=ba.rs.fingerprint(f.region_path), region_sides=f.regions.to_dict("records"),
                   track=ba.rs.fingerprint(f.track), labels=ba.rs.fingerprint(f.label_meta),
                   motion=ba.rs.fingerprint(f.motion))
    stem = Path(f.row.track_name).stem + "_" + ba.rs.cache_key(sources)
    assert f.context.name == stem + ".npz"
    assert f.crossing.name == stem + "_v1.npz"
    stamp = f.motion.stat().st_mtime_ns + 1_000_000_000
    os.utime(f.motion, ns=(stamp, stamp))
    changed, _ = ba.rs.context_cache_paths(f.row, f.block, f.regions, f.settings, 3600, f.root)
    assert changed != f.context


def test_window_loader_reuses_cache_and_does_not_rebuild_missing_context(tmp_path, monkeypatch):
    f = _window_fixture(tmp_path, monkeypatch)
    root = tmp_path / "comparison"
    first = ba.load_window(f.block, root, label="early")
    assert first["bins"].body_motion_mm_s.eq(1).all()
    assert first["inventory"].selected.all()
    assert first["info"]["start_time"] == "2026-08-10 18:43:36"
    def forbidden(*args, **kwargs):
        raise AssertionError("Must reuse the comparison cache")
    monkeypatch.setattr(ba, "bin_context", forbidden)
    cached = ba.load_window(f.block, root, label="early")
    pd.testing.assert_frame_equal(first["bins"].reset_index(drop=True), cached["bins"])
    f.context.unlink()
    with pytest.raises(FileNotFoundError, match="Missing current motion context"):
        ba.load_window(f.block, root, label="early")


def test_loader_rejects_stale_vectors_and_absent_inputs(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="Missing comparison inputs"):
        ba.load_window(tmp_path / "missing", tmp_path / "cache", label="early")
    f = _window_fixture(tmp_path, monkeypatch)
    stamp = f.clock.stat().st_mtime_ns + 1_000_000_000
    os.utime(f.speed_meta, ns=(stamp, stamp))
    with pytest.raises(ValueError, match="predate updated vectors"):
        ba.load_window(f.block, tmp_path / "cache", label="early")


def test_plotting_shared_scales_and_missing_ids(tmp_path):
    a = pd.concat([_bins([i, i+1], side=side, track_id=i) for side in ("left", "right") for i in range(6)])
    paired, ants = _compare(a, a)
    inventory = a[ba.IDENTITY].drop_duplicates().assign(selected=True, cluster_id="cluster_0", detection_fraction=.8)
    audit = ba.identity_audit(inventory, inventory, ants)
    summary = ba.summarize_changes(ants, SETTINGS)
    counts, scores = ba.cluster_overlap(audit)
    info = dict(bin_minutes=30, light_on_hour=5.5, light_off_hour=19.5)
    figures = [ba.plot_paired_activity(ants, summary), ba.plot_changes_by_id(ants, audit),
               ba.plot_coverage(audit, 1), ba.plot_cluster_overlap(counts, scores),
               ba.plot_clock_changes(paired, ants, info)[0],
               ba.plot_ant_profiles(ba.fold_profiles(a, SETTINGS), ba.fold_profiles(a, SETTINGS),
                                    side="left", track_id=1, info=info)]
    try:
        for i, fig in enumerate(figures):
            fig.canvas.draw()
            assert np.asarray(fig.canvas.buffer_rgba())[..., :3].std() > 5
            fig.savefig(tmp_path / f"figure{i}.png", dpi=40)
        heatmaps = [ax.images[0] for ax in figures[1].axes if ax.images]
        assert len(heatmaps) == 10
        assert heatmaps[0].get_clim() == heatmaps[5].get_clim()
        ax = figures[0].axes[0]
        figures[0].canvas.callbacks.process("pick_event", SimpleNamespace(artist=ax.collections[0], ind=[0]))
        assert any("left T0" in text.get_text() and text.get_visible() for text in ax.texts)
        renderer = figures[-1].canvas.get_renderer()
        boxes = [ax.yaxis.label.get_window_extent(renderer) for ax in figures[-1].axes]
        assert not any(a.overlaps(b) for a, b in zip(boxes, boxes[1:]))
        with pytest.raises(ValueError, match="No eligible"):
            ba.plot_ant_profiles(ba.fold_profiles(a, SETTINGS), ba.fold_profiles(a, SETTINGS),
                                 side="left", track_id=99, info=info)
    finally:
        plt.close("all")
