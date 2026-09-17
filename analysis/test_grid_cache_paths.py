import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from analysis import grid_occupancy_utils as go
from analysis import return_sleep_utils as rs
from analysis import arena_grid_utils as arena


def _grid(block, name):
    root = block / "stitched" / name
    metadata = root / "per_track" / "ant_left" / "grid_occupancy_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.touch()
    return root


def test_pipeline_grid_preferred_over_pilot_name(tmp_path):
    legacy = _grid(tmp_path, "grid_occupancy_histograms_0p5mm_inferred_bounds")
    assert go.resolve_grid_root(tmp_path) == legacy
    canonical = _grid(tmp_path, "grid_occupancy_histograms")
    assert go.resolve_grid_root(tmp_path) == canonical
    assert go.resolve_grid_root(tmp_path, legacy.name) == legacy


def test_missing_and_ambiguous_grids_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="compute_track_grid_occupancy"):
        go.resolve_grid_root(tmp_path)
    _grid(tmp_path, "grid_occupancy_histograms_trial1")
    _grid(tmp_path, "grid_occupancy_histograms_trial2")
    with pytest.raises(ValueError, match="Multiple alternate"):
        go.resolve_grid_root(tmp_path)
    with pytest.raises(FileNotFoundError, match="requested cache"):
        go.resolve_grid_root(tmp_path, "missing")


def test_interactions_prefer_published_pipeline_and_support_legacy(tmp_path):
    canonical = tmp_path / "interactions"
    assert rs.resolve_interaction_root(tmp_path) == canonical
    legacy = tmp_path / "interactions_skeleton_0p1mm"
    legacy.mkdir()
    (legacy / "run_manifest.json").touch()
    assert rs.resolve_interaction_root(tmp_path) == legacy
    canonical.mkdir()
    (canonical / "run_manifest.json").touch()
    assert rs.resolve_interaction_root(tmp_path) == canonical


@pytest.mark.parametrize("name", ["continous_stitched", "continuous_stitched"])
def test_continuous_selection_is_explicit_and_accepts_both_spellings(tmp_path, name):
    combined = tmp_path / name
    (combined / "per_track").mkdir(parents=True)
    grid = combined / "grid_occupancy_histograms"
    metadata = grid / "per_track" / "ant_left" / "grid_occupancy_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.touch()
    assert go.resolve_analysis_dataset(tmp_path) == tmp_path
    assert go.resolve_analysis_dataset(tmp_path, continuous=True) == combined
    assert go.resolve_analysis_dataset(combined, continuous=True) == combined
    assert go.resolve_grid_root(combined) == grid
    assert go.resolve_grid_root(combined, grid.name) == grid
    assert go.resolve_stitched_root(combined) == combined
    assert go.infer_speed_root(grid) == combined / "speed_vectors"


def test_continuous_groups_require_an_unambiguous_selection(tmp_path):
    container = tmp_path / "continous_stitched"
    groups = [container / "block02_block03", container / "block06_block07"]
    with pytest.raises(FileNotFoundError, match="No continuous recording"):
        go.resolve_analysis_dataset(tmp_path, continuous=True)
    for group in groups:
        group.mkdir(parents=True)
        (group / "block_combination.json").write_text("{}")
    with pytest.raises(ValueError, match="Multiple continuous recordings"):
        go.resolve_analysis_dataset(tmp_path, continuous=True)
    assert go.resolve_analysis_dataset(groups[1], continuous=True) == groups[1]
    (groups[1] / "block_combination.json").unlink()
    assert go.resolve_analysis_dataset(container, continuous=True) == groups[0]


def test_combination_metadata_controls_clock_date_and_full_frame_span(tmp_path):
    combined = tmp_path / "continous_stitched" / "block02_block03"
    combined.mkdir(parents=True)
    (combined / "block_combination.json").write_text(json.dumps(dict(
        start_datetime="2026-05-15T14:20:47", num_frames=9013355, fps=24,
        blocks=[{"name": "block02"}, {"name": "block03"}],
    )))
    # The first ant can start late, stop early, or have no clock in its name.
    tracks = pd.DataFrame(dict(track_name=["TrackID_0001_left.parquet"], frame_min=[100], frame_max=[1000]))
    context = go.recording_context(combined, tracks)
    assert context == dict(start_clock_seconds=14*3600+20*60+47, recording_date="20260515",
                           frame_min=0, frame_stop=9013355, fps=24, blocks=["block02", "block03"])


def test_single_block_clock_and_direct_stitched_paths_remain_supported(tmp_path):
    block = tmp_path / "20260810" / "block02"
    grid = _grid(block, "grid_occupancy_histograms")
    assert go.resolve_grid_root(block / "stitched") == grid
    tracks = pd.DataFrame(dict(track_name=["TrackID_0001_all_184324_left.parquet"],
                               frame_min=[1234000], frame_max=[1234003]))
    context = go.recording_context(block, tracks)
    assert context == dict(start_clock_seconds=18*3600+43*60+24, recording_date="20260810",
                           frame_min=1234000, frame_stop=1234004, blocks=[])
    assert go.recording_context(block / "stitched", tracks) == context


def test_script_reuses_combined_histograms_without_reading_tracks(tmp_path, monkeypatch):
    combined = tmp_path / "continous_stitched"
    cache = combined / "grid_occupancy_histograms"
    metadata = cache / "per_track" / "ant_left" / "grid_occupancy_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.touch()
    # No raw tracks or annotations are available: cached spatial plots need neither.
    def unexpected(*args, **kwargs):
        raise AssertionError("Combined occupancy must not rebuild existing histograms")
    monkeypatch.setattr(arena, "prepare_arena_grid_cache", unexpected)
    names = {"SOURCE_GRID_ROOT", "USE_ANNOTATED_ARENA_BOUNDS", "PANORAMA_REGIONS_PATH", "GRID_ROOT"}
    script = Path(__file__).with_name("grid_occupancy.py")
    statements = [node for node in ast.parse(script.read_text()).body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in names]
    namespace = dict(go=go, arena_go=arena, DATASET_ROOT=combined, ANNOTATION_ROOT=combined,
                     GRID_OUTPUT_NAME=None, args=SimpleNamespace(grid_workers=2))
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(script), "exec"), namespace)
    assert namespace["GRID_ROOT"] == cache
    assert namespace["USE_ANNOTATED_ARENA_BOUNDS"] is False
    assert not go.is_combined_recording(tmp_path / "block02")
