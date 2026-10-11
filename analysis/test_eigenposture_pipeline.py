"""Publication must wait for complete, current tracking inputs and a real report."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from analysis.eigenposture_pipeline import validate_interactions, run
from scripts.eigenposture_after_tracking import ready, watch
from tracking.colony.interaction_one_chunk import interaction_parameters


def test_standard_pipeline_gates_publication_on_eigenposture(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    block = tmp_path / '20260916/block02'
    (block / 'data').mkdir(parents=True)
    hmats = tmp_path / 'H.npz'; hmats.touch()
    output = tmp_path / 'scratch'
    command = ['bash', str(repo / 'tracking/colony/submit_blocks_pipeline.sh'),
               '--blocks_root', str(block.parent), '--block_glob', block.name,
               '--hmats', str(hmats), '--output_root', str(output), '--dry_run']
    subprocess.run(command, check=True, capture_output=True)
    state = output / 'jobs/block02/state'
    config = json.loads((state / 'eigenposture_watch.json').read_text())
    assert len(config['prerequisites']) == 5
    assert not any('eigenposture_complete' in p for p in config['prerequisites'])
    assert str(state / 'interactions_complete_block02.ok') in config['prerequisites']
    manifest = (state / 'per_track_analysis_markers_block02.tsv').read_text()
    assert 'analysis_outputs/eigenposture/eigenposture_complete.ok' in manifest
    assert 'posture_interpolated/posture_interpolation_complete.ok' in manifest
    assert 'sleep_predictions/' not in manifest
    assert 'sleep_motion_labels/' not in manifest
    assert 'colony_presence_vectors/' not in manifest
    assert str(block / 'analysis_outputs/eigenposture') in (state / 'transfer_manifest_block02.tsv').read_text()
    job = Path(config['job_script']).read_text()
    assert 'analysis.eigenposture_pipeline' in job and str(block) in job
    # Execute the generated cache submitter with a harmless fan-out stand-in:
    # the four per-track outputs may be launched in the standard mode.
    calls = tmp_path / 'calls.txt'
    stub = tmp_path / 'fanout.sh'
    stub.write_text('printf "%s\\n" "$*" >> "' + str(calls) + '"\n')
    submitter = output / 'jobs/block02/scripts/submit_per_track_analysis_block02.sbatch'
    content = submitter.read_text().replace(str(repo / 'scripts/per_track_slurm_fanout.sh'), str(stub))
    local = tmp_path / 'submit.sh'; local.write_text(content)
    subprocess.run(['bash', str(local)], check=True, capture_output=True)
    launched = calls.read_text().splitlines()
    assert len(launched) == 4
    assert all(any('--operation_name ' + name + ' ' in line for line in launched)
               for name in ('speed_vector', 'sleep_motion', 'grid_occupancy', 'posture_interpolation'))
    subprocess.run(command + ['--no_eigenposture'], check=True, capture_output=True)
    assert 'eigenposture' not in (state / 'per_track_analysis_markers_block02.tsv').read_text()


def test_watcher_requires_every_fresh_marker_and_submits_once(tmp_path):
    a, b = tmp_path / 'stitch.ok', tmp_path / 'contacts.ok'
    a.write_text('done'); b.write_text('done')
    os.utime(b, (10, 10))
    assert not ready([a, b], 20)
    config = tmp_path / 'watch.json'
    config.write_text(json.dumps(dict(epoch=20, prerequisites=[str(a), str(b)],
                                    job_script='job.sbatch', sbatch_bin='sbatch', poll_seconds=1)))
    with patch('scripts.eigenposture_after_tracking.time.time', return_value=21), \
         patch('scripts.eigenposture_after_tracking.time.sleep', side_effect=lambda _: os.utime(b, None)), \
         patch('scripts.eigenposture_after_tracking.subprocess.check_output', return_value='12345\n') as submit:
        watch(config)
        watch(config)
    submit.assert_called_once()
    assert config.with_name('eigenposture_job_id.txt').read_text().strip() == '12345'


def test_native_interactions_require_matching_source_and_complete_parameters(tmp_path):
    tracks, contacts = tmp_path / 'tracks', tmp_path / 'interactions'
    tracks.mkdir(); contacts.mkdir()
    name = '20260916_225431_chunk000_right.parquet'
    track, output = tracks / name, contacts / name
    pd.DataFrame({'Frame': [0]}).to_parquet(track)
    pd.DataFrame({'Frame': [0], 'ant_a': [1], 'ant_b': [2]}).to_parquet(output)
    params = interaction_parameters(mm_per_px=.016, interaction_radius_mm=8,
        micro_interaction_distance_mm=.1, antenna_bodypoints=None, frame_start=0, max_frames=None, frame_step=1)
    meta = dict(parameters=params, input_size=track.stat().st_size,
                input_mtime_ns=track.stat().st_mtime_ns, output_size=output.stat().st_size, n_pair_detections=1)
    sidecar = output.with_suffix('.metadata.json')
    sidecar.write_text(json.dumps(meta))
    validate_interactions(tmp_path)
    manifest = json.loads((contacts / 'run_manifest.json').read_text())
    assert manifest['expected_files'] == [name]
    assert json.loads((contacts / 'transfer_complete.ok').read_text())['run_id'] == manifest['run_id']
    meta['input_size'] += 1; sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='stale or incomplete'):
        validate_interactions(tmp_path)


def test_failed_report_never_marks_pipeline_complete(tmp_path):
    block = tmp_path / 'block'; block.mkdir()
    annotations = block / 'panorama_regions.csv'; annotations.write_text('annotations')
    output = block / 'analysis_outputs/eigenposture'
    output.mkdir(parents=True)
    marker = output / 'eigenposture_complete.ok'; marker.write_text('old success')
    calls = []
    def command(args, **kwargs):
        calls.append((args, kwargs['env'].copy()))
        if 'analysis.eigenposture_task_report' in args:
            raise subprocess.CalledProcessError(1, args)
    with patch('analysis.eigenposture_pipeline.validate_interactions'), \
         patch('analysis.eigenposture_pipeline.subprocess.run', side_effect=command):
        with pytest.raises(subprocess.CalledProcessError):
            run(block, block, output, 4, .25)
    assert not marker.exists()
    assert calls[-1][1]['ANTS_GRID_OUTPUT_NAME'] == 'grid_occupancy_histograms_arena_0p25mm'
    assert calls[-1][1]['ANTS_DATASET_ROOT'] == str(block)
    assert [c[0][2] for c in calls].index('analysis.full_behavior_roles') < [c[0][2] for c in calls].index('analysis.grid_occupancy')
