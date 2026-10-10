"""Post hoc comparison of frozen behavior states with existing motion sleep labels.

This module cannot choose state features, resolution, or the low-motion candidate.
Sleep labels are movement-based comparators, not independent arousal validation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.eigenposture_leiden_features import START_FRAME, STOP_FRAME, FPS
from analysis.sleep_motion_analysis_utils import interval_counts


def compare(block, output):
    frozen_path = output / 'minimal_selection_frozen.json'
    before = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    frozen = json.loads(frozen_path.read_text())
    width = frozen['minutes'] * 60 * FPS
    starts = np.arange(START_FRAME, STOP_FRAME, width)
    stops = starts + width
    assignments = pd.read_csv(output / 'ant_roles.csv')
    with np.load(output / 'states.npz') as z:
        np.testing.assert_array_equal(z['ants'], assignments.ant)
        tasks = z['tasks']
        values = z['values']
    known = np.zeros(tasks.shape, dtype=np.int64)
    asleep = np.zeros(tasks.shape, dtype=np.int64)
    parameters, provenance = [], []
    for i, ant in enumerate(assignments.itertuples()):
        root = block / 'stitched/sleep_motion_labels/per_track' / Path(ant.track_name).stem
        metadata_path = root / 'sleep_motion_label_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        if metadata['fps'] != FPS or metadata['summary']['track_name'] != ant.track_name:
            raise ValueError('Sleep-label clock or identity mismatch')
        motion_meta_path = block / 'stitched/sleep_motion/per_track' / Path(ant.track_name).stem / 'sleep_motion_metadata.json'
        motion_meta = json.loads(motion_meta_path.read_text())
        if (motion_meta_path.stat().st_size, motion_meta_path.stat().st_mtime_ns) != (metadata['source_metadata_size_bytes'], metadata['source_metadata_mtime_ns']):
            raise ValueError(f'Sleep labels use changed motion metadata: {motion_meta_path}')
        raw = block / 'stitched/per_track' / ant.track_name
        if (raw.stat().st_size, raw.stat().st_mtime_ns) != (motion_meta['source_size_bytes'], motion_meta['source_mtime_ns']):
            raise ValueError(f'Sleep motion cache uses changed track: {raw}')
        state_path = root / metadata['files']['sleep_state']
        state = np.load(state_path, mmap_mode='r')
        if state.dtype != np.int8 or len(state) != metadata['frame_max'] - metadata['frame_min'] + 1:
            raise ValueError('Sleep-label array/metadata mismatch')
        known[i] = interval_counts(state >= 0, metadata['frame_min'], starts, stops)
        asleep[i] = interval_counts(state == 1, metadata['frame_min'], starts, stops)
        parameters.append(json.dumps(metadata['classifier_parameters'], sort_keys=True))
        provenance.append(dict(ant=ant.ant, state_path=str(state_path), size=state_path.stat().st_size,
            mtime_ns=state_path.stat().st_mtime_ns, metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest()))
    if len(set(parameters)) != 1:
        raise ValueError('Mixed sleep classifier definitions')
    fraction = np.divide(asleep, known, out=np.full(tasks.shape, np.nan), where=known >= .5 * width)
    valid = (tasks >= 0) & np.isfinite(fraction)
    rows, ant_rows = [], []
    rng = np.random.default_rng(7241010)
    for state in range(frozen['n_states']):
        selected = valid & (tasks == state)
        ant_known = np.where(selected, known, 0).sum(axis=1)
        ant_sleep = np.where(selected, asleep, 0).sum(axis=1)
        ant_fraction = np.divide(ant_sleep, ant_known, out=np.full(len(tasks), np.nan),
                                 where=(ant_known > 0) & (selected.sum(axis=1) >= 3))
        usable = np.flatnonzero(np.isfinite(ant_fraction))
        for i in usable:
            ant_rows.append(dict(ant=assignments.ant.iloc[i], side=assignments.side.iloc[i],
                                 state=state, sleep_fraction=ant_fraction[i], n_bins=int(selected[i].sum())))
        observed = ant_fraction[usable]
        draws = observed[rng.integers(len(observed), size=(1000, len(observed)))].mean(axis=1) if len(observed) else np.full(1000, np.nan)
        # Immediately adjacent observed bins only; no concatenation across gaps.
        pair_valid = (tasks[:, :-1] == state) & (tasks[:, 1:] >= 0)
        stays = pair_valid & (tasks[:, 1:] == state)
        rows.append(dict(state=state, n_ants=len(usable), n_bins=int(selected.sum()),
            sleep_fraction=float(observed.mean()), lower=float(np.quantile(draws, .025)),
            upper=float(np.quantile(draws, .975)),
            persistence=float(stays.sum() / max(pair_valid.sum(), 1)),
            body_forward_peak=float(np.median(values[:, :, 0][tasks == state])),
            body_lateral_peak=float(np.median(values[:, :, 1][tasks == state])),
            antenna_unsigned_mean=float(np.median(values[:, :, 43][tasks == state]))))
    pd.DataFrame(rows).to_csv(output / 'sleep_by_state.csv', index=False)
    pd.DataFrame(ant_rows).to_csv(output / 'sleep_by_ant_state.csv', index=False)
    np.savez_compressed(output / 'sleep_comparison.npz', ants=assignments.ant.to_numpy(str),
                        sleep_fraction=fraction, classified_frames=known, sleep_frames=asleep)
    candidate = frozen['low_motion_candidate']
    baseline_known = np.where(valid, known, 0).sum(axis=1)
    baseline_sleep = np.where(valid, asleep, 0).sum(axis=1)
    baseline_per_ant = np.divide(baseline_sleep, baseline_known, out=np.full(len(tasks), np.nan), where=baseline_known > 0)
    baseline = np.nanmean(baseline_per_ant)
    # Paired comparison controls for different ants contributing to each state.
    paired = []
    for i, ant in enumerate(assignments.itertuples()):
        candidate_bins = valid[i] & (tasks[i] == candidate)
        other_bins = valid[i] & (tasks[i] != candidate)
        if candidate_bins.sum() < 3 or other_bins.sum() < 3:
            continue
        candidate_fraction = asleep[i, candidate_bins].sum() / known[i, candidate_bins].sum()
        other_fraction = asleep[i, other_bins].sum() / known[i, other_bins].sum()
        paired.append(dict(ant=ant.ant, side=ant.side, candidate_fraction=candidate_fraction,
                           other_fraction=other_fraction, difference=candidate_fraction-other_fraction))
    paired = pd.DataFrame(paired, columns=['ant', 'side', 'candidate_fraction', 'other_fraction', 'difference'])
    paired.to_csv(output / 'sleep_candidate_contrast.csv', index=False)
    difference = paired.difference.to_numpy()
    draws = difference[rng.integers(len(difference), size=(1000, len(difference)))].mean(axis=1) if len(difference) else np.full(1000, np.nan)
    summary = dict(candidate_state=candidate, candidate_rule='Lowest combined body-peak and mean-antenna motion, fixed before reading sleep labels',
        candidate_sleep_fraction=rows[candidate]['sleep_fraction'], equal_ant_baseline=float(baseline),
        paired_ants=len(paired), paired_sleep_difference=float(difference.mean()) if len(difference) else None,
        paired_sleep_difference_ci=np.quantile(draws, [.025, .975]).tolist(),
        classifier_parameters=json.loads(parameters[0]), minimum_classified_bin_fraction=.5,
        caveat='Sleep labels use overlapping movement measurements. Agreement is descriptive, not independent physiological/arousal validation. Relative antennal speeds differ from the global-landmark speed used by the sleep rule.',
        bin_minutes=frozen['minutes'], frozen_selection_sha256=before, sources=provenance)
    (output / 'sleep_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == before
    print('SLEEP CANDIDATE', candidate, rows[candidate], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    compare(a.block, a.output)
