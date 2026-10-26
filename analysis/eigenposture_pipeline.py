"""Build the inputs and run eigenposture_interactive for a finished colony block."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from analysis.eigenposture_inputs import MOTION_OUTPUT_NAME, validate_preprocessing, validate_report_inputs


def write_status(output, state, stage, **details):
    path = output / 'pipeline_status.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(dict(state=state, stage=stage, updated_epoch=time.time(),
        job_id=os.environ.get('SLURM_JOB_ID'), **details), indent=2) + '\n')
    temporary.replace(path)


def validate_interactions(block: Path) -> None:
    """Give native pipeline contacts the same validated manifest as rebuilds."""
    import pyarrow.parquet as pq
    from tracking.colony.interaction_one_chunk import cache_matches
    from tracking.colony.skeleton_contacts import CONTACT_GEOMETRY

    root = block / 'interactions'
    tracks = sorted((block / 'tracks').glob('*_chunk*_*.parquet'))
    if not tracks:
        raise ValueError(f'No chunk tracks in {block / "tracks"}')
    records = []
    parameters = None
    for track in tracks:
        output = root / track.name
        meta = json.loads(output.with_suffix('.metadata.json').read_text())
        params = meta['parameters']
        if (params['geometry'] != CONTACT_GEOMETRY or params['directed'] is not False
                or params['micro_interaction_distance_mm'] != .1
                or params['frame_start'] != 0 or params['frame_step'] != 1
                or params['max_frames'] is not None):
            raise ValueError(f'Expected complete 0.1 mm skeleton contacts: {output}')
        if parameters is not None and params != parameters:
            raise ValueError(f'Interaction parameters differ: {output}')
        if not cache_matches(output, track, params):
            raise ValueError(f'Interaction cache is stale or incomplete: {output}')
        if pq.ParquetFile(output).metadata.num_rows != meta['n_pair_detections']:
            raise ValueError(f'Interaction row count differs: {output}')
        parameters = params
        records.append(dict(file=track.name, metadata=meta))
    run_id = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:20]
    manifest = dict(run_id=run_id, parameters=parameters,
                    expected_files=[p.name for p in tracks], validation='native pipeline sidecars and parquet footers')
    (root / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (root / 'transfer_complete.ok').write_text(json.dumps(dict(run_id=run_id, n_chunks=len(tracks),
        validation='complete compute-side outputs; bucket publication is handled by the pipeline')) + '\n')


def _run(block: Path, source_block: Path, output: Path, workers: int, grid_size_mm: float,
        grid_output_name: str = 'grid_occupancy_histograms', reports_only: bool = False) -> None:
    from camera_cal.region_paths import panorama_regions_path
    from analysis.arena_grid_utils import arena_grid_output_name

    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'eigenposture_complete.ok'
    validate_preprocessing(block)
    regions = panorama_regions_path(source_block, search_earlier_dates=True)
    if not regions.is_file():
        raise FileNotFoundError(f'No current or earlier panorama annotations for {source_block}')
    if block.resolve() != source_block.resolve():
        shutil.copy2(regions, block / 'panorama_regions.csv')
    validate_interactions(block)
    env = os.environ.copy()
    env.update(MPLBACKEND='Agg', ANTS_DATASET_ROOT=str(block),
               ANTS_BEHAVIOR_SOURCE=str(output), ANTS_ROLE_OUTPUT=str(output / 'report'),
               ANTS_MOTION_SLEEP=str(output / MOTION_OUTPUT_NAME))
    # The grid preparation starts with the pipeline cache; the report uses its
    # annotated-arena derivative, at the requested resolution.
    env['ANTS_GRID_OUTPUT_NAME'] = grid_output_name
    commands = [
        ['analysis.full_behavior_extract', '--block', str(block), '--output', str(output), '--workers', str(workers),
         '--posture-root', str(block / 'stitched/posture_interpolated/per_track')],
        ['analysis.full_behavior_states', 'features', '--root', str(output), '--workers', str(workers)],
        ['analysis.full_behavior_states', 'learn', '--root', str(output)],
        ['analysis.full_behavior_states', 'assign', '--root', str(output)],
        ['analysis.full_behavior_profiles', '--root', str(output)],
        ['analysis.full_behavior_roles', '--root', str(output)],
        ['analysis.full_behavior_roles', '--root', str(output), '--side', 'left', '--output', str(output / 'roles_left')],
        ['analysis.s0_motion_sleep', 'run', '--source', str(output), '--block', str(block),
         '--work', str(output / 'motion_work'), '--output', str(output / MOTION_OUTPUT_NAME), '--workers', str(workers)],
        ['analysis.grid_occupancy', str(block), '--headless', '--occupancy-only',
         '--grid-workers', str(workers), '--grid-size-mm', str(grid_size_mm),
         '--figure-root', str(output / 'spatial_reference')],
        ['analysis.eigenposture_task_report', '--source', str(output), '--output', str(output / 'report')],
        ['analysis.eigenposture_task_report', '--source', str(output), '--output', str(output / 'report_left')],
    ]
    provenance = dict(source_block=str(source_block), analysis_block=str(block),
                      annotations=str(regions), annotation_sha256=hashlib.sha256(regions.read_bytes()).hexdigest(),
                      commands=commands, entrypoint='analysis/eigenposture_interactive.py',
                      state_cohort='both', ant_classes='separate left and right', posture='interpolated <=1 second',
                      motion='same camera and recording', sleep_window_seconds=1, grid_size_mm=grid_size_mm)
    repo = Path(__file__).resolve().parents[1]
    snapshot = repo / 'PIPELINE_SOURCE.json'
    if snapshot.is_file():
        provenance['code_snapshot'] = json.loads(snapshot.read_text())
    if reports_only:
        previous = json.loads((output / 'pipeline_run.json').read_text())
        if previous['analysis_block'] != str(block) or previous['grid_size_mm'] != grid_size_mm:
            raise ValueError('Report retry must use the original analysis block and grid resolution.')
        retry = dict(epoch=time.time(), commands=commands[-2:], code_snapshot=provenance.get('code_snapshot'))
        previous.setdefault('report_retries', []).append(retry)
        provenance = previous
        commands = commands[-2:]
    (output / 'pipeline_run.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for command in commands:
        write_status(output, 'running', ' '.join(command), command=command)
        if command[0] == 'analysis.eigenposture_task_report':
            env['ANTS_GRID_OUTPUT_NAME'] = arena_grid_output_name(grid_size_mm)
            side='left' if command[-1] == str(output / 'report_left') else 'right'
            env['ANTS_COLONY_SIDE']=side
            env['ANTS_ROLE_MODELS']=str(output / ('roles_left' if side=='left' else 'roles'))
            validate_report_inputs(block, output, output / MOTION_OUTPUT_NAME, env['ANTS_GRID_OUTPUT_NAME'])
        print('RUN', command, flush=True)
        subprocess.run([sys.executable, '-m', *command], cwd=repo, env=env, check=True)
    for side,folder in [('right','report'),('left','report_left')]:
        for name in ('summary.json', side+'_behavior_classes.pdf'):
            if not (output / folder / name).is_file() or not (output / folder / name).stat().st_size:
                raise FileNotFoundError(output / folder / name)
    marker.write_text('completed eigenposture_interactive\n')


def run(block: Path, source_block: Path, output: Path, workers: int, grid_size_mm: float,
        grid_output_name: str = 'grid_occupancy_histograms', reports_only: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'eigenposture_complete.ok'
    marker.unlink(missing_ok=True)
    write_status(output, 'running', 'validate_preprocessing')
    try:
        _run(block, source_block, output, workers, grid_size_mm, grid_output_name, reports_only)
    except Exception as error:
        marker.unlink(missing_ok=True)
        stage = json.loads((output / 'pipeline_status.json').read_text())['stage']
        write_status(output, 'failed', stage, error=f'{type(error).__name__}: {error}')
        raise
    write_status(output, 'complete', 'complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block', type=Path, required=True, help='Writable scratch block with tracking and caches')
    parser.add_argument('--source-block', type=Path, required=True, help='Original recording, for annotation fallback')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--grid-size-mm', type=float, default=.25)
    parser.add_argument('--grid-output-name', default='grid_occupancy_histograms')
    parser.add_argument('--reports-only', action='store_true', help='Validate existing inputs and rerun both reports without refitting models')
    args = parser.parse_args()
    run(args.block, args.source_block, args.output, args.workers, args.grid_size_mm, args.grid_output_name, args.reports_only)


if __name__ == '__main__':
    main()
