"""Shared input contract for the pipeline and interactive/headless reports."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MOTION_OUTPUT_NAME = 'sleep_from_s0_motion_20261011'


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f'Missing or invalid eigenposture input: {path}') from error


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_files(paths, remedy):
    missing = [str(p) for p in paths if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise ValueError(f'{remedy}\nMissing inputs:\n' + '\n'.join(missing[:10]))


def validate_preprocessing(block: Path):
    """Reject missing, obsolete or stale caches before expensive model fitting."""
    from analysis.camera_motion import MOTION_VERSION
    from analysis.compute_track_posture_interpolation import VERSION
    from analysis.sleep_motion_utils import motion_cache_root

    tracks = sorted((block / 'stitched/per_track').glob('*.parquet'))
    if not tracks:
        raise ValueError(f'No stitched tracks under {block}')
    for track in tracks:
        motion_root = motion_cache_root(block) / 'per_track' / track.stem
        pose_root = block / 'stitched/posture_interpolated/per_track' / track.stem
        require_files([motion_root / 'sleep_motion_metadata.json',
                       pose_root / 'posture_interpolation_metadata.json', pose_root / '_SUCCESS',
                       pose_root / track.name], 'Rebuild motion and posture caches before eigenposture.')
        motion = read_json(motion_root / 'sleep_motion_metadata.json')
        pose = read_json(pose_root / 'posture_interpolation_metadata.json')
        stat = track.stat()
        if (motion.get('motion_version') != MOTION_VERSION
                or motion.get('source_size_bytes') != stat.st_size
                or motion.get('source_mtime_ns') != stat.st_mtime_ns
                or pose.get('source', {}).get('size') != stat.st_size
                or pose.get('source', {}).get('mtime_ns') != stat.st_mtime_ns
                or pose.get('parameters', {}).get('version') != VERSION
                or pose.get('parameters', {}).get('max_gap_seconds') != 1.):
            raise ValueError(f'Stale or incompatible motion/posture cache for {track}; rebuild these caches.')
        require_files([motion_root / filename for filename in motion['files'].values()],
                      f'Rebuild incomplete motion cache for {track}.')


def report_grid_root(block: Path, source: Path, output_name: str | None = None):
    """Use the explicit grid or the resolution recorded by its producer."""
    from analysis import grid_occupancy_utils as go
    from analysis.arena_grid_utils import arena_grid_output_name

    output_name = output_name or os.environ.get('ANTS_GRID_OUTPUT_NAME')
    if not output_name:
        manifest = source / 'pipeline_run.json'
        if manifest.is_file():
            output_name = arena_grid_output_name(read_json(manifest)['grid_size_mm'])
        else:
            # Older manually prepared packets predate pipeline_run.json.
            candidates = [name for name in (arena_grid_output_name(.25), 'grid_occupancy_histograms_arena')
                          if go.metadata_paths(go.resolve_stitched_root(block) / name)]
            if len(candidates) != 1:
                raise ValueError('Set ANTS_GRID_OUTPUT_NAME to the prepared arena grid; '
                                 'no unique report grid was found.')
            output_name = candidates[0]
    return go.resolve_grid_root(block, output_name)


def validate_report_inputs(block: Path, source: Path, motion_root: Path,
                           grid_output_name: str | None = None):
    """Fail before plotting if calibrated sleep or the spatial reference is absent."""
    import pandas as pd
    import numpy as np
    from analysis.camera_motion import MOTION_VERSION

    grid = report_grid_root(block, source, grid_output_name)
    require_files([grid / 'track_cluster_ids.csv'], 'Run spatial reference preparation before the report.')
    require_files([source / 'inventory.csv', source / 'models/dictionary.npz',
                   *[motion_root / name for name in ('definition.json', 'complete.json',
                       'calibration_samples.parquet', 'calibration_state_summary.csv')]],
                  'Run analysis.eigenposture_pipeline to generate S0 motion-sleep calibration and labels before the report.')
    definition = read_json(motion_root / 'definition.json')
    complete = read_json(motion_root / 'complete.json')
    definition_sha = sha(motion_root / 'definition.json')
    if (complete.get('definition_sha256') != definition_sha
            or definition.get('dictionary_sha256') != sha(source / 'models/dictionary.npz')
            or definition.get('inventory_sha256') != sha(source / 'inventory.csv')
            or definition.get('calibration_samples_sha256') != sha(motion_root / 'calibration_samples.parquet')):
        raise ValueError(f'Stale S0 motion-sleep calibration in {motion_root}; rerun calibration and labels.')
    inventory = pd.read_csv(source / 'inventory.csv')
    if complete.get('n_tracks') != len(inventory):
        raise ValueError(f'Incomplete S0 motion-sleep labels in {motion_root}')
    for name in inventory.track_name:
        folder = motion_root / 'per_track' / Path(name).stem
        meta = read_json(folder / 'sleep_motion_label_metadata.json')
        if (meta.get('definition_sha256') != definition_sha
                or meta.get('measurement', {}).get('motion_metadata', {}).get('motion_version') != MOTION_VERSION):
            raise ValueError(f'Stale S0 motion-sleep labels in {folder}; rebuild motion and labels.')
        require_files([folder / name for name in meta['files'].values()], 'Rebuild incomplete S0 motion-sleep labels.')
        state = np.load(folder / meta['files']['sleep_state'], mmap_mode='r')
        if len(state) != meta['summary']['n_frames']:
            raise ValueError(f'Truncated S0 motion-sleep labels in {folder}')
    return grid
