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
from scripts.eigenposture_after_tracking import ready, watch, failure_reason
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
    assert len(config['prerequisites']) == 6
    assert any('posture_interpolation_complete.ok' in p for p in config['prerequisites'])
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
    snapshot = Path(config['repo'])
    assert snapshot != repo and (snapshot / 'PIPELINE_SOURCE.json').is_file()
    content = submitter.read_text().replace(str(snapshot / 'scripts/per_track_slurm_fanout.sh'), str(stub))
    local = tmp_path / 'submit.sh'; local.write_text(content)
    subprocess.run(['bash', str(local)], check=True, capture_output=True)
    launched = calls.read_text().splitlines()
    assert len(launched) == 4
    assert all(any('--operation_name ' + name + ' ' in line for line in launched)
               for name in ('speed_vector', 'sleep_motion', 'grid_occupancy', 'posture_interpolation'))
    # A restitch or changed analysis parameter invalidates a bare _SUCCESS.
    assert all('--skip_existing' not in line for line in launched)
    transfer = output / 'jobs/block02/scripts/transfer_to_bucket_block02.sh'
    subprocess.run(['bash', '-n', str(transfer)], check=True)
    assert '--check-failure' in transfer.read_text()
    subprocess.run(command + ['--no_eigenposture'], check=True, capture_output=True)
    assert 'eigenposture' not in (state / 'per_track_analysis_markers_block02.tsv').read_text()


def test_watcher_requires_every_fresh_marker_and_submits_once(tmp_path):
    a, b = tmp_path / 'stitch.ok', tmp_path / 'contacts.ok'
    a.write_text('done'); b.write_text('done')
    os.utime(b, (10, 10))
    assert not ready([a, b], 20)
    config = tmp_path / 'watch.json'
    config.write_text(json.dumps(dict(epoch=20, prerequisites=[str(a), str(b)],
                                    job_script='job.sbatch', sbatch_bin='sbatch', poll_seconds=1, output=str(tmp_path))))
    def submit_job(*args, **kwargs):
        (tmp_path / 'eigenposture_complete.ok').write_text('done')
        return '12345\n'
    with patch('scripts.eigenposture_after_tracking.time.time', return_value=21), \
         patch('scripts.eigenposture_after_tracking.time.sleep', side_effect=lambda _: os.utime(b, None)), \
         patch('scripts.eigenposture_after_tracking.subprocess.check_output', side_effect=submit_job) as submit:
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
         patch('analysis.eigenposture_pipeline.validate_preprocessing'), \
         patch('analysis.eigenposture_pipeline.validate_report_inputs'), \
         patch('analysis.eigenposture_pipeline.subprocess.run', side_effect=command):
        with pytest.raises(subprocess.CalledProcessError):
            run(block, block, output, 4, .25)
    assert not marker.exists()
    status = json.loads((output / 'pipeline_status.json').read_text())
    assert status['state'] == 'failed' and 'analysis.eigenposture_task_report' in status['stage']
    assert calls[-1][1]['ANTS_GRID_OUTPUT_NAME'] == 'grid_occupancy_histograms_arena_0p25mm'
    assert calls[-1][1]['ANTS_DATASET_ROOT'] == str(block)
    assert [c[0][2] for c in calls].index('analysis.full_behavior_roles') < [c[0][2] for c in calls].index('analysis.grid_occupancy')


def test_reports_follow_fresh_calibration_and_use_separate_colonies(tmp_path):
    block = tmp_path / 'block'; block.mkdir()
    (block / 'panorama_regions.csv').write_text('annotations')
    output = block / 'analysis_outputs/eigenposture'
    calls = []
    def command(args, **kwargs):
        calls.append((args, kwargs['env'].copy()))
        if 'analysis.eigenposture_task_report' in args:
            side=kwargs['env']['ANTS_COLONY_SIDE']
            assert Path(kwargs['env']['ANTS_ROLE_MODELS']).name == ('roles' if side=='right' else 'roles_left')
            folder=Path(args[-1]);folder.mkdir(parents=True)
            (folder/'summary.json').write_text('{}')
            (folder/(side+'_behavior_classes.pdf')).write_bytes(b'%PDF-1.4\n')
    with patch('analysis.eigenposture_pipeline.validate_interactions'), \
         patch('analysis.eigenposture_pipeline.validate_preprocessing'), \
         patch('analysis.eigenposture_pipeline.validate_report_inputs'), \
         patch('analysis.eigenposture_pipeline.subprocess.run', side_effect=command):
        run(block,block,output,4,.25)
    stages=[args[2] for args,_ in calls]
    assert stages.index('analysis.s0_motion_sleep') < stages.index('analysis.eigenposture_task_report')
    assert all(env['ANTS_MOTION_SLEEP'] == str(output / 'sleep_from_s0_motion_20261011') for _, env in calls)
    assert '--posture-root' in calls[0][0]
    assert [env['ANTS_COLONY_SIDE'] for args,env in calls if args[2]=='analysis.eigenposture_task_report']==['right','left']
    assert (output/'eigenposture_complete.ok').exists()


def test_report_retry_keeps_completed_models_and_runs_both_sides(tmp_path):
    block = tmp_path / 'block'; block.mkdir()
    (block / 'panorama_regions.csv').write_text('annotations')
    output = block / 'analysis_outputs/eigenposture'; output.mkdir(parents=True)
    original = dict(analysis_block=str(block), grid_size_mm=.25, commands=['original preprocessing'])
    (output / 'pipeline_run.json').write_text(json.dumps(original))
    calls = []
    def command(args, **kwargs):
        assert args[2] == 'analysis.eigenposture_task_report'
        side = kwargs['env']['ANTS_COLONY_SIDE']; calls.append(side)
        folder = Path(args[-1]); folder.mkdir(parents=True)
        (folder / 'summary.json').write_text('{}')
        (folder / (side + '_behavior_classes.pdf')).write_bytes(b'%PDF-1.4\n')
    with patch('analysis.eigenposture_pipeline.validate_interactions'), \
         patch('analysis.eigenposture_pipeline.validate_preprocessing'), \
         patch('analysis.eigenposture_pipeline.validate_report_inputs'), \
         patch('analysis.eigenposture_pipeline.subprocess.run', side_effect=command):
        run(block, block, output, 4, .25, reports_only=True)
    assert calls == ['right', 'left']
    provenance = json.loads((output / 'pipeline_run.json').read_text())
    assert provenance['commands'] == original['commands']
    assert len(provenance['report_retries']) == 1


@pytest.mark.parametrize('state', ['FAILED', 'OUT_OF_MEMORY', 'TIMEOUT', 'CANCELLED by 1234', 'COMPLETED'])
def test_failed_or_markerless_job_is_reported(tmp_path, state):
    config = tmp_path / 'watch.json'
    config.write_text(json.dumps(dict(epoch=0, output=str(tmp_path))))
    (tmp_path / 'eigenposture_job_id.txt').write_text('12345\n')
    result = subprocess.CompletedProcess([], 0, f'12345|{state}|1:0\n', '')
    with patch('scripts.eigenposture_after_tracking.subprocess.run', return_value=result):
        assert '12345' in failure_reason(config)


def test_dead_dependency_is_reported_and_scheduler_outage_is_not_failure(tmp_path):
    config = tmp_path / 'watch.json'
    jobfile = tmp_path / 'motion_job.txt'; jobfile.write_text('456\n')
    config.write_text(json.dumps(dict(epoch=0, output=str(tmp_path), upstream_jobs={'motion': str(jobfile)})))
    results = [subprocess.CompletedProcess([], 0, '456|PENDING|0:0\n', ''),
               subprocess.CompletedProcess([], 0, '456|PENDING|DependencyNeverSatisfied\n', '')]
    with patch('scripts.eigenposture_after_tracking.subprocess.run', side_effect=results):
        assert 'DependencyNeverSatisfied' in failure_reason(config)
    with patch('scripts.eigenposture_after_tracking.subprocess.run', side_effect=OSError('scheduler offline')):
        assert failure_reason(config) is None


def test_watcher_records_failure_without_resubmitting(tmp_path):
    config = tmp_path / 'watch.json'
    config.write_text(json.dumps(dict(epoch=0, prerequisites=[], output=str(tmp_path), poll_seconds=1)))
    (tmp_path / 'eigenposture_job_id.txt').write_text('12345\n')
    with patch('scripts.eigenposture_after_tracking.failure_reason', return_value='job 12345 FAILED'), \
         patch('scripts.eigenposture_after_tracking.subprocess.check_output') as submit:
        with pytest.raises(RuntimeError, match='12345 FAILED'):
            watch(config)
    submit.assert_not_called()
    assert json.loads((tmp_path / 'eigenposture_watch_status.json').read_text())['state'] == 'failed'


def test_transfer_stops_on_report_failure_before_running_rsync(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    block = tmp_path / 'dataset/block01'; (block / 'data').mkdir(parents=True)
    hmats = tmp_path / 'H.npz'; hmats.touch()
    output = tmp_path / 'scratch'
    subprocess.run(['bash', str(repo / 'tracking/colony/submit_blocks_pipeline.sh'),
        '--blocks_root', str(block.parent), '--hmats', str(hmats), '--output_root', str(output), '--dry_run'],
        check=True, capture_output=True)
    report = output / 'block01/analysis_outputs/eigenposture'; report.mkdir(parents=True)
    (report / 'pipeline_status.json').write_text(json.dumps(dict(state='failed', stage='calibration', error='missing input')))
    transfer = output / 'jobs/block01/scripts/transfer_to_bucket_block01.sh'
    result = subprocess.run(['bash', str(transfer)], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and 'calibration' in result.stdout
    assert not (output / 'jobs/block01/state/transfer_to_bucket_block01.ok').exists()


def test_snapshot_keeps_relative_dataset_paths_anchored_to_submission_directory(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / 'dataset/block01/data').mkdir(parents=True)
    (tmp_path / 'H.npz').touch()
    (tmp_path / 'bounds.json').write_text('{}')
    (tmp_path / 'local_venv').mkdir()
    (tmp_path / 'local_venv/python').symlink_to(sys.executable)
    subprocess.run(['bash', str(repo / 'tracking/colony/submit_blocks_pipeline.sh'),
        '--blocks_root', 'dataset', '--hmats', 'H.npz', '--output_root', 'scratch',
        '--python_bin', 'local_venv/python', '--eigenposture_python_bin', 'local_venv/python',
        '--grid_bounds_json', 'bounds.json', '--dry_run'],
        cwd=tmp_path, check=True, capture_output=True)
    state = tmp_path / 'scratch/jobs/block01/state'
    config = json.loads((state / 'eigenposture_watch.json').read_text())
    assert config['output'] == str(tmp_path / 'scratch/block01/analysis_outputs/eigenposture')
    assert str(tmp_path / 'dataset/block01') in Path(config['job_script']).read_text()
    assert config['python'] == str(tmp_path / 'local_venv/python')
    submitter = state.parent / 'scripts/submit_per_track_analysis_block01.sbatch'
    assert str(tmp_path / 'bounds.json') in submitter.read_text()


@pytest.mark.parametrize('previous_success', [False, True])
def test_transfer_cannot_report_success_after_skipping_a_missing_source(tmp_path, previous_success):
    repo = Path(__file__).resolve().parents[1]
    block = tmp_path / 'dataset/block01'; (block / 'data').mkdir(parents=True)
    hmats = tmp_path / 'H.npz'; hmats.touch()
    output = tmp_path / 'scratch'
    subprocess.run(['bash', str(repo / 'tracking/colony/submit_blocks_pipeline.sh'),
        '--blocks_root', str(block.parent), '--hmats', str(hmats), '--output_root', str(output),
        '--no_eigenposture', '--dry_run'], check=True, capture_output=True)
    state = output / 'jobs/block01/state'
    markers = [state / 'stitch_block01.ok', state / 'interactions_complete_block01.ok']
    markers += [Path(line.split('\t')[1]) for line in (state / 'per_track_analysis_markers_block01.tsv').read_text().splitlines()]
    for path in markers:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('done')
    complete = state / 'transfer_to_bucket_block01.ok'
    if previous_success:
        complete.write_text('completed previous run\n')
        os.utime(complete, (1, 1))
    script = output / 'jobs/block01/scripts/transfer_to_bucket_block01.sh'
    result = subprocess.run(['bash', str(script)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and 'expected transfer source is missing' in result.stdout
    assert not complete.exists()
