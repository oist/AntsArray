"""Bin cached posture, velocity, wavelet and social measurements without spatial labels."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

BINS = (2, 5, 15)
START_FRAME = 41520
STOP_FRAME = 4188720
FPS = 24
FAMILIES = (['velocity'] * 8 + ['posture'] * 8 + ['variation'] * 4 +
            ['rate'] * 4 + ['wavelet'] * 12 + ['variation'] * 4 + ['interaction'] * 3 + ['antenna'] * 6)
N_FEATURES = len(FAMILIES)


def union_exposure(starts, stops, n_bins, width):
    """Exact union of half-open frame intervals, clipped to this analysis clock."""
    starts = np.clip(np.asarray(starts, dtype=np.int64), 0, n_bins * width)
    stops = np.clip(np.asarray(stops, dtype=np.int64), 0, n_bins * width)
    keep = stops > starts
    starts, stops = starts[keep], stops[keep]
    result = np.zeros(n_bins, dtype=float)
    if not len(starts):
        return result
    order = np.argsort(starts)
    starts, stops = starts[order], stops[order]
    ends = np.maximum.accumulate(stops)
    new = np.r_[0, np.flatnonzero(starts[1:] > ends[:-1]) + 1]
    starts, stops = starts[new], np.maximum.reduceat(stops, new)
    first, last = starts // width, (stops - 1) // width
    same = first == last
    np.add.at(result, first[same], stops[same] - starts[same])
    first, last, starts, stops = first[~same], last[~same], starts[~same], stops[~same]
    np.add.at(result, first, (first + 1) * width - starts)
    np.add.at(result, last, stops - last * width)
    difference = np.zeros(n_bins + 1)
    np.add.at(difference, first + 1, width)
    np.add.at(difference, last, -width)
    result += np.cumsum(difference[:-1])
    assert np.all((result >= 0) & (result <= width))
    return result / width


def social_features(bouts, ants, coverage, minutes):
    width = minutes * 60 * FPS
    n_bins = (STOP_FRAME - START_FRAME) // width
    result = np.zeros((len(ants), n_bins, 3), dtype=np.float32)
    for i, ant in enumerate(ants.itertuples()):
        selected = bouts[bouts.side.eq(ant.side) & (bouts.ant_a.eq(ant.track_id) | bouts.ant_b.eq(ant.track_id))]
        starts = selected.start_frame.to_numpy() - START_FRAME
        stops = selected.end_frame.to_numpy() + 1 - START_FRAME
        onset = selected.is_new_onset.to_numpy() & (starts >= 0) & (starts < n_bins * width)
        result[i, :, 0] = np.bincount(starts[onset] // width, minlength=n_bins)
        result[i, :, 2] = union_exposure(starts, stops, n_bins, width)
        partner = np.where(selected.ant_a.eq(ant.track_id), selected.ant_b, selected.ant_a)
        for other in np.unique(partner):
            keep = (partner == other) & (stops > 0) & (starts < n_bins * width)
            lo = np.clip(starts[keep], 0, n_bins * width - 1) // width
            hi = (np.clip(stops[keep], 1, n_bins * width) - 1) // width + 1
            difference = np.zeros(n_bins + 1, dtype=int)
            np.add.at(difference, lo, 1)
            np.add.at(difference, hi, -1)
            result[i, :, 1] += np.cumsum(difference[:-1]) > 0
        seconds = coverage[ant.side][START_FRAME // FPS:STOP_FRAME // FPS]
        covered = seconds.reshape(n_bins, minutes * 60).all(axis=1)
        result[i, ~covered] = np.nan
    return result


def make_bins(minute_values, social, minutes):
    """Core >=60% clips (at least two); wavelets >=20%, without imputation.

    Long convolution supports are stricter than derivative/mean supports.
    Each wavelet band uses only valid clips from the same bin and common core
    support. Counts are saved separately; clips are never joined across gaps.
    """
    data = minute_values.copy()
    core_columns = np.r_[0:24, 36:42]
    core_valid = np.isfinite(data[:, :, core_columns]).all(axis=2)
    data[~core_valid] = np.nan
    blocks = data.reshape(len(data), -1, minutes, data.shape[-1])
    feature_counts = np.isfinite(blocks).sum(axis=2)
    counts = core_valid.reshape(len(data), -1, minutes).sum(axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        values = np.nanmean(blocks, axis=2)
        values[:, :, :2] = np.nanmax(blocks[:, :, :, :2], axis=2)
        values[:, :, 37] = np.nanmax(blocks[:, :, :, 37], axis=2)
        between = np.nanstd(blocks[:, :, :, 8:12], axis=2)
    values[:, :, 24:36][feature_counts[:, :, 24:36] < max(1, int(np.ceil(.2 * minutes)))] = np.nan
    values = np.concatenate([values[:, :, :36], between, social, values[:, :, 36:]], axis=2)
    values[counts < max(2, int(np.ceil(.6 * minutes)))] = np.nan
    return values, counts, feature_counts


def balanced_matrix(raw, columns):
    """Log nonnegative quantities, preserve PC means, equal family variance."""
    transformed = np.asarray(raw, dtype=float).copy()
    scales = np.ones(N_FEATURES)
    scales[:6] = .1
    scales[6:8] = 1.
    scales[16:20] = .05
    scales[20:24] = .5
    scales[24:36] = .0025
    scales[36:40] = .05
    scales[43:] = .1
    positive = np.r_[0:8, 16:42, 43:49]  # contact fraction stays linear
    transformed[:, positive] = np.log1p(transformed[:, positive] / scales[positive])
    transformed = transformed[:, columns]
    center = transformed.mean(axis=0)
    scale = np.ones(len(columns))
    families = np.asarray(FAMILIES)[columns]
    for family in np.unique(families):
        mask = families == family
        scale[mask] = max(np.sqrt(transformed[:, mask].var(axis=0).sum()), 1e-6)
    return ((transformed - center) / scale).astype(np.float32), center, scale


def prepare(block, source, clips, output):
    from analysis.behavior_landscape_features import load_contacts
    output.mkdir(parents=True, exist_ok=True)
    ants = pd.read_csv(source / 'all_ant_coverage.csv')
    with np.load(clips / 'clip_features.npz') as z:
        np.testing.assert_array_equal(z['ants'], ants.ant)
        values = z['values']
        clip_names = list(z['names'])
        names = clip_names[:36] + [f'posture_PC{i}_between_std' for i in range(1, 5)]
    names += ['interaction_onsets', 'interaction_partners', 'contact_bout_fraction'] + clip_names[36:]
    info = dict(start_time='2026-07-24 10:00:00', frame_start=START_FRAME,
                frame_stop=STOP_FRAME, fps=24., context_settings=dict(fps=24.))
    bouts, coverage, sources, run_id = load_contacts(block, info)
    for minutes in BINS:
        social = social_features(bouts, ants, coverage, minutes)
        binned, counts, feature_counts = make_bins(values, social, minutes)
        if minutes == 5:
            old = source.parent / 'task_states_interactions_20261010/interaction_counts.npz'
            with np.load(old) as z:
                np.testing.assert_allclose(social[:, :, 0], z['counts'], equal_nan=True)
        np.savez_compressed(output / f'bins_{minutes:02d}.npz', ants=ants.ant.to_numpy(str),
                            values=binned, counts=counts, clip_feature_counts=feature_counts, social=social, names=names,
                            families=FAMILIES, minutes=minutes)
        print(f'{minutes} minutes: {np.isfinite(binned).all(axis=2).sum()} common valid bins', flush=True)
    (output / 'social_provenance.json').write_text(json.dumps(dict(
        run_id=run_id, bout_cache=str(sources[-1]),
        bout_cache_sha256=hashlib.sha256(sources[-1].read_bytes()).hexdigest(),
        onset='New pair-contact bouts; count for both ants; left-censored starts excluded',
        partners='Distinct partners whose bouts overlap the bin, including earlier starts',
        contact_bout_fraction='Union of observed bout intervals, clipped to bin; includes merged gaps up to 2 s',
        common_support='Core features share valid clips; >=60% (minimum 2) per bin. Wavelets use full-valid supports from those clips, >=20% (minimum 1); counts saved; no imputation',
        bin_minutes=BINS,
    ), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('block', 'source', 'clips', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    a = parser.parse_args()
    prepare(a.block, a.source, a.clips, a.output)
