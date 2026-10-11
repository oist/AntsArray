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


def run(block: Path, source_block: Path, output: Path, workers: int, grid_size_mm: float,
        grid_output_name: str = 'grid_occupancy_histograms') -> None:
    from camera_cal.region_paths import panorama_regions_path
    from analysis.arena_grid_utils import arena_grid_output_name

    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'eigenposture_complete.ok'
    marker.unlink(missing_ok=True)
    regions = panorama_regions_path(source_block, search_earlier_dates=True)
    if not regions.is_file():
        raise FileNotFoundError(f'No current or earlier panorama annotations for {source_block}')
    if block.resolve() != source_block.resolve():
        shutil.copy2(regions, block / 'panorama_regions.csv')
    validate_interactions(block)
    env = os.environ.copy()
    env.update(MPLBACKEND='Agg', ANTS_DATASET_ROOT=str(block),
               ANTS_BEHAVIOR_SOURCE=str(output), ANTS_ROLE_OUTPUT=str(output / 'report'))
    # The grid preparation starts with the pipeline cache; the report uses its
    # annotated-arena derivative, at the requested resolution.
    env['ANTS_GRID_OUTPUT_NAME'] = grid_output_name
    commands = [
        ['analysis.full_behavior_extract', '--block', str(block), '--output', str(output), '--workers', str(workers)],
        ['analysis.full_behavior_states', 'features', '--root', str(output), '--workers', str(workers)],
        ['analysis.full_behavior_states', 'learn', '--root', str(output)],
        ['analysis.full_behavior_states', 'assign', '--root', str(output)],
        ['analysis.full_behavior_profiles', '--root', str(output)],
        ['analysis.full_behavior_roles', '--root', str(output)],
        ['analysis.grid_occupancy', str(block), '--headless', '--occupancy-only',
         '--grid-workers', str(workers), '--grid-size-mm', str(grid_size_mm),
         '--figure-root', str(output / 'spatial_reference')],
        ['analysis.eigenposture_task_report', '--source', str(output), '--output', str(output / 'report')],
    ]
    provenance = dict(source_block=str(source_block), analysis_block=str(block),
                      annotations=str(regions), annotation_sha256=hashlib.sha256(regions.read_bytes()).hexdigest(),
                      commands=commands, entrypoint='analysis/eigenposture_interactive.py',
                      state_cohort='both', ant_classes='right', grid_size_mm=grid_size_mm)
    (output / 'pipeline_run.json').write_text(json.dumps(provenance, indent=2) + '\n')
    repo = Path(__file__).resolve().parents[1]
    for command in commands:
        if command[0] == 'analysis.eigenposture_task_report':
            env['ANTS_GRID_OUTPUT_NAME'] = arena_grid_output_name(grid_size_mm)
        print('RUN', command, flush=True)
        subprocess.run([sys.executable, '-m', *command], cwd=repo, env=env, check=True)
    for name in ('summary.json', 'right_behavior_classes.pdf'):
        if not (output / 'report' / name).is_file():
            raise FileNotFoundError(output / 'report' / name)
    marker.write_text('completed eigenposture_interactive\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block', type=Path, required=True, help='Writable scratch block with tracking and caches')
    parser.add_argument('--source-block', type=Path, required=True, help='Original recording, for annotation fallback')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--grid-size-mm', type=float, default=.25)
    parser.add_argument('--grid-output-name', default='grid_occupancy_histograms')
    args = parser.parse_args()
    run(args.block, args.source_block, args.output, args.workers, args.grid_size_mm, args.grid_output_name)


if __name__ == '__main__':
    main()
