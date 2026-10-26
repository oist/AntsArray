"""Report regression cases from the September 15/16 pipeline failures."""
import json
import os
from pathlib import Path
import runpy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from analysis.camera_motion import MOTION_VERSION
from analysis.eigenposture_inputs import (
    MOTION_OUTPUT_NAME, report_grid_root, sha, validate_preprocessing, validate_report_inputs,
)


@pytest.fixture
def packet(tmp_path, monkeypatch):
    monkeypatch.delenv('ANTS_GRID_OUTPUT_NAME', raising=False)
    block = tmp_path / 'block'; source = block / 'analysis_outputs/eigenposture'
    (source / 'models').mkdir(parents=True)
    np.savez(source / 'models/dictionary.npz', state=[0, 1])
    name = 'TrackID_0001_all_155247_right.parquet'
    pd.DataFrame([dict(track_name=name)]).to_csv(source / 'inventory.csv', index=False)
    (source / 'pipeline_run.json').write_text(json.dumps(dict(grid_size_mm=.5)))
    grid = block / 'stitched/grid_occupancy_histograms_arena_0p5mm'
    (grid / 'per_track' / Path(name).stem).mkdir(parents=True)
    (grid / 'per_track' / Path(name).stem / 'grid_occupancy_metadata.json').write_text('{}')
    (grid / 'track_cluster_ids.csv').write_text('TrackID,cluster_id\n1,right_0\n')
    motion = source / MOTION_OUTPUT_NAME
    folder = motion / 'per_track' / Path(name).stem; folder.mkdir(parents=True)
    pd.DataFrame(dict(ant=['right:001'], state=[0], body=[.01], antenna=[.02])).to_parquet(motion / 'calibration_samples.parquet')
    (motion / 'calibration_state_summary.csv').write_text('state\n0\n')
    definition = dict(dictionary_sha256=sha(source / 'models/dictionary.npz'),
                      inventory_sha256=sha(source / 'inventory.csv'),
                      calibration_samples_sha256=sha(motion / 'calibration_samples.parquet'))
    (motion / 'definition.json').write_text(json.dumps(definition))
    definition_hash = sha(motion / 'definition.json')
    (motion / 'complete.json').write_text(json.dumps(dict(n_tracks=1, definition_sha256=definition_hash)))
    np.save(folder / 'sleep_state.npy', np.array([-1, 0, 1], dtype=np.int8))
    (folder / 'sleep_motion_label_metadata.json').write_text(json.dumps(dict(
        definition_sha256=definition_hash, summary=dict(n_frames=3), files=dict(sleep_state='sleep_state.npy'),
        measurement=dict(motion_metadata=dict(motion_version=MOTION_VERSION)))))
    return block, source, motion, grid


def test_report_uses_producer_grid_resolution_and_honors_explicit_selection(packet, monkeypatch):
    block, source, motion, grid = packet
    assert validate_report_inputs(block, source, motion) == grid
    monkeypatch.setenv('ANTS_GRID_OUTPUT_NAME', 'missing_grid')
    with pytest.raises(FileNotFoundError, match='requested cache'):
        report_grid_root(block, source)
    assert report_grid_root(block, source, grid.name) == grid


@pytest.mark.parametrize('missing', ['definition.json', 'complete.json', 'calibration_samples.parquet'])
def test_report_rejects_unbuilt_sleep_prerequisites(packet, missing):
    block, source, motion, _ = packet
    (motion / missing).unlink()
    with pytest.raises(ValueError, match='generate S0 motion-sleep'):
        validate_report_inputs(block, source, motion)


def test_report_rejects_labels_from_previous_dictionary(packet):
    block, source, motion, _ = packet
    np.savez(source / 'models/dictionary.npz', state=[0, 1, 2])
    with pytest.raises(ValueError, match='Stale S0'):
        validate_report_inputs(block, source, motion)


def test_report_rejects_incomplete_label_arrays(packet):
    block, source, motion, _ = packet
    folder = next((motion / 'per_track').iterdir())
    np.save(folder / 'sleep_state.npy', np.array([1], dtype=np.int8))
    with pytest.raises(ValueError, match='Truncated'):
        validate_report_inputs(block, source, motion)


def test_actual_interactive_entry_checks_inputs_before_plotting(packet, monkeypatch):
    block, source, motion, _ = packet
    (motion / 'definition.json').unlink()
    monkeypatch.setenv('ANTS_DATASET_ROOT', str(block))
    monkeypatch.setenv('ANTS_BEHAVIOR_SOURCE', str(source))
    monkeypatch.setenv('ANTS_MOTION_SLEEP', str(motion))
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    with patch('matplotlib.pyplot.figure', side_effect=AssertionError('plotted before validation')):
        with pytest.raises(ValueError, match='generate S0 motion-sleep'):
            runpy.run_path(str(Path(__file__).with_name('eigenposture_interactive.py')))


def test_preprocessing_rejects_stale_tracking_and_old_motion(tmp_path, monkeypatch):
    monkeypatch.delenv('ANTS_MOTION_CACHE_ROOT', raising=False)
    track = tmp_path / 'stitched/per_track/TrackID_0001_right.parquet'
    track.parent.mkdir(parents=True)
    pd.DataFrame(dict(Frame=[0], Bodypoint=[0], X=[1.], Y=[1.])).to_parquet(track)
    motion = tmp_path / 'stitched/sleep_motion/per_track' / track.stem; motion.mkdir(parents=True)
    pose = tmp_path / 'stitched/posture_interpolated/per_track' / track.stem; pose.mkdir(parents=True)
    (pose / track.name).write_bytes(track.read_bytes()); (pose / '_SUCCESS').write_text('complete')
    np.save(motion / 'speed.npy', [0., 1.])
    meta = dict(motion_version=MOTION_VERSION, source_size_bytes=track.stat().st_size,
                source_mtime_ns=track.stat().st_mtime_ns, files=dict(speed='speed.npy'))
    sidecar = motion / 'sleep_motion_metadata.json'; sidecar.write_text(json.dumps(meta))
    (pose / 'posture_interpolation_metadata.json').write_text(json.dumps(dict(
        source=dict(size=track.stat().st_size, mtime_ns=track.stat().st_mtime_ns),
        parameters=dict(version=1, max_gap_seconds=1.))))
    validate_preprocessing(tmp_path)
    sidecar.write_text(json.dumps({**meta, 'motion_version': 'old_cross_camera'}))
    with pytest.raises(ValueError, match='Stale or incompatible'):
        validate_preprocessing(tmp_path)
    sidecar.write_text(json.dumps(meta))
    os.utime(track, ns=(track.stat().st_atime_ns, track.stat().st_mtime_ns + 1000000))
    with pytest.raises(ValueError, match='Stale or incompatible'):
        validate_preprocessing(tmp_path)
