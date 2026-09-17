import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis import arena_grid_utils as arena
from analysis import compute_track_grid_occupancy as compute
from analysis import grid_occupancy_utils as go


def _dataset(tmp_path):
    block = tmp_path / "20260810" / "block02"
    track_root = block / "stitched" / "per_track"
    track_root.mkdir(parents=True)
    grid = block / "stitched" / "grid_occupancy_histograms"
    rows = []
    for side, x0, x1, y0, y1 in (("left", 10, 120, 20, 180), ("right", 300, 440, 100, 260)):
        name = f"TrackID_0001_all_184324_{side}.parquet"
        pd.DataFrame(dict(Frame=np.arange(1234000, 1234004), Bodypoint=0,
                          TrackX=[x0 + 10, x1, x1 + 1, 1000],
                          TrackY=[y0 + 10, y1, y1 + 1, 1000])).to_parquet(track_root / name)
        folder = grid / "per_track" / name.removesuffix(".parquet")
        folder.mkdir(parents=True)
        (folder / "grid_occupancy_metadata.json").write_text(json.dumps(dict(
            track_name=name, track_id=1, side=side, mm_per_px=.016, grid_size_mm=1.0,
            frame_min=1234000, frame_max=1234003, n_observed_frames=4,
            x_col="TrackX", y_col="TrackY", input_x_is_side_local=False, bodypoint_filter=0)))
        rows.append(dict(region_id=side, name=f"arena_{side}", semantic_label="arena", shape="rectangle",
                         tracking_x_min_px=x0, tracking_x_max_px=x1,
                         tracking_y_min_px=y0, tracking_y_max_px=y1,
                         tracking_center_x_px=(x0+x1)/2, tracking_center_y_px=(y0+y1)/2,
                         radius_px=np.nan, area_mm2=(x1-x0)*(y1-y0)*.016**2, mm_per_pixel=.016))
    regions_path = block / "panorama_regions.csv"
    pd.DataFrame(rows).to_csv(regions_path, index=False)
    return block, grid, regions_path


def test_rebuilds_exact_independent_arena_bounds_and_preserves_frame_clock(tmp_path, monkeypatch):
    block, source, regions_path = _dataset(tmp_path)
    output = arena.prepare_arena_grid_cache(block, source, regions_path)
    assert output != source
    tracks = go.load_grid_tracks(output)
    assert tracks.frame_min.eq(1234000).all() and tracks.frame_max.eq(1234003).all()
    assert tracks.n_in_grid_frames.eq(2).all()  # Outliers in partial edge bins are excluded too.
    assert tracks.n_out_of_grid_detected_frames.eq(2).all()
    np.testing.assert_allclose(tracks.occupancy_sum, .5)
    assert tracks.input_x_origin_px.tolist() == [10, 300]
    assert tracks.y_origin_px.tolist() == [20, 100]
    assert tracks.grid_size_mm.eq(.25).all()
    assert tracks.histogram_shape_yx.tolist() == [(11, 8), (11, 9)]
    for row in tracks.itertuples():
        metadata = json.loads(row.metadata_path.read_text())
        assert metadata["grid_pad_mm"] == 0
        assert metadata["bounds_source"] == "panorama_arena_regions"
        assert metadata["arena_regions_path"] == str(regions_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    go.plot_single_histogram(tracks, row_number=0)
    ax = plt.gcf().axes[0]
    np.testing.assert_allclose(ax.get_xlim(), [0, 110 * .016])
    np.testing.assert_allclose(ax.get_ylim(), [0, 160 * .016])
    plt.close("all")


def test_cache_reuses_unchanged_annotations_and_rebuilds_changed_side_only(tmp_path, monkeypatch):
    block, source, regions_path = _dataset(tmp_path)
    arena.prepare_arena_grid_cache(block, source, regions_path)
    read = compute.load_track_xy
    calls = []
    def counted(path, **kwargs):
        calls.append(path.name)
        return read(path, **kwargs)
    monkeypatch.setattr(compute, "load_track_xy", counted)
    regions = pd.read_csv(regions_path)
    regions.to_csv(regions_path, index=False)
    arena.prepare_arena_grid_cache(block, source, regions_path)
    assert calls == []
    regions.loc[regions.region_id == "left", "tracking_x_max_px"] = 125
    regions.to_csv(regions_path, index=False)
    arena.prepare_arena_grid_cache(block, source, regions_path)
    assert len(calls) == 1 and calls[0].endswith("left.parquet")


def test_missing_arenas_fail_before_any_rebuild(tmp_path):
    block, source, regions_path = _dataset(tmp_path)
    regions = pd.read_csv(regions_path)
    regions["semantic_label"] = "colony"
    regions.to_csv(regions_path, index=False)
    with pytest.raises(ValueError, match="exactly two rectangular regions labelled arena"):
        arena.prepare_arena_grid_cache(block, source, regions_path)
    assert not (block / "stitched" / "grid_occupancy_histograms_arena").exists()


def test_explicit_combined_arena_rebuild_uses_direct_paths_and_reuses_cache(tmp_path, monkeypatch):
    block, source, regions_path = _dataset(tmp_path)
    combined = block.parent / "continous_stitched"
    (block / "stitched").rename(combined)
    regions_path = regions_path.rename(combined / regions_path.name)
    source = combined / source.name
    output = arena.prepare_arena_grid_cache(combined, source, regions_path)
    assert output == combined / "grid_occupancy_histograms_arena_0p25mm"
    assert len(go.load_grid_tracks(output)) == 2
    assert not (combined / "stitched").exists()
    read = compute.load_track_xy
    calls = []
    def counted(path, **kwargs):
        calls.append(path.name)
        return read(path, **kwargs)
    monkeypatch.setattr(compute, "load_track_xy", counted)
    arena.prepare_arena_grid_cache(combined, source, regions_path)
    assert calls == []


def test_arena_positions_override_colony_based_divider_and_preserve_small_overlap(tmp_path):
    _, _, path = _dataset(tmp_path)
    data = pd.read_csv(path)
    data.loc[data.region_id == "left", "tracking_x_max_px"] = 302
    data.loc[data.region_id == "left", "tracking_center_x_px"] = 156
    data.to_csv(path, index=False)
    regions = go.load_panorama_regions(path)
    assert regions.side_split_x_px.eq(301).all()
    bounds = arena.arena_bounds_from_regions(regions)
    assert bounds["left"]["x_max_px"] == 302
    data.loc[data.region_id == "left", "tracking_x_max_px"] = 310
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlap in x"):
        go.load_panorama_regions(path)


def test_finer_grid_rereads_positions_preserves_mass_and_keeps_legacy_cache(tmp_path, monkeypatch):
    block, source, regions_path = _dataset(tmp_path)
    coarse = arena.prepare_arena_grid_cache(block, source, regions_path, grid_size_mm=1.0)
    legacy = coarse.rename(block / "stitched/grid_occupancy_histograms_arena")
    old_metadata = {p: p.read_bytes() for p in go.metadata_paths(legacy)}
    read = compute.load_track_xy
    calls = []
    def counted(path, **kwargs):
        calls.append(path.name)
        return read(path, **kwargs)
    monkeypatch.setattr(compute, "load_track_xy", counted)
    output = arena.prepare_arena_grid_cache(block, source, regions_path, grid_size_mm=.25)
    assert output.name == "grid_occupancy_histograms_arena_0p25mm"
    assert len(calls) == 2
    tracks = go.load_grid_tracks(output)
    assert tracks.grid_size_mm.eq(.25).all()
    assert tracks.histogram_shape_yx.tolist() == [(11, 8), (11, 9)]
    np.testing.assert_allclose(tracks.occupancy_sum, .5)
    assert tracks.n_in_grid_frames.eq(2).all() and tracks.n_out_of_grid_detected_frames.eq(2).all()
    assert all(p.read_bytes() == content for p, content in old_metadata.items())
    calls.clear()
    assert arena.prepare_arena_grid_cache(block, source, regions_path, grid_size_mm=.25) == output
    assert calls == []


def test_block_preference_is_persistent_and_explicit_size_takes_precedence(tmp_path):
    block, source, regions_path = _dataset(tmp_path)
    assert arena.configured_grid_size_mm(block) == compute.DEFAULT_GRID_SIZE_MM == .25
    (block / arena.GRID_SETTINGS_FILENAME).write_text(json.dumps(dict(grid_size_mm=.5)))
    assert arena.configured_grid_size_mm(block) == .5
    assert arena.configured_grid_size_mm(block / "stitched") == .5
    assert arena.configured_grid_size_mm(block, .125) == .125
    output = arena.prepare_arena_grid_cache(block, source, regions_path)
    assert output.name == "grid_occupancy_histograms_arena_0p5mm"
    assert go.load_grid_tracks(output).grid_size_mm.eq(.5).all()


@pytest.mark.parametrize("size", [0, -.25, float("nan"), float("inf"), True, "invalid"])
def test_invalid_grid_spacing_fails_before_writing(tmp_path, size):
    block, source, regions_path = _dataset(tmp_path)
    with pytest.raises(ValueError, match="positive finite"):
        arena.prepare_arena_grid_cache(block, source, regions_path, grid_size_mm=size)
    assert not list((block / "stitched").glob("grid_occupancy_histograms_arena*"))
