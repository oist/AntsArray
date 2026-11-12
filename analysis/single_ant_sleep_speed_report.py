"""Ordered single-ant sleep/speed report and conditional speed-proxy validation.

Use observed posture states, never motion-inferred negatives, as validation
targets. Extract once, then render small cached summaries without rereading tracks.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPEED_CUTS = np.linspace(0, 20, 4001)
BOUT_SPEEDS = np.array([.02, .05, .1, .2, .5, 1.])
BOUT_SECONDS = np.array([0, 1, 5, 10, 30, 60, 300])
COLORS = ['#8452b8', '#d99728', '#b8bcc6']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def support_mask(manifest):
    mask = np.zeros(manifest['n_frames'], bool)
    for segment in manifest['recording_segments']:
        start = int(segment['frame_offset'])
        mask[start:start + int(segment['num_frames'])] = True
    return mask


def true_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def long_bout_mask(mask, minimum_frames):
    """Retrospective entire bouts; a missing speed observation breaks a bout."""
    starts, stops = true_runs(mask)
    keep = stops - starts >= minimum_frames
    delta = np.zeros(len(mask) + 1, np.int32)
    delta[starts[keep]] = 1
    delta[stops[keep]] = -1
    return np.cumsum(delta[:-1]) > 0


def state_groups(n_states, sleep_states=(0, 1)):
    sleep = np.array(sorted(set(int(s) for s in sleep_states)), dtype=int)
    if not len(sleep) or (sleep < 0).any() or (sleep >= n_states).any():
        raise ValueError('Sleep states must be nonempty and inside the fitted dictionary')
    other = np.setdiff1d(np.arange(n_states), sleep)
    if not len(other):
        raise ValueError('At least one other observed state is required for speed-proxy validation')
    return sleep, other


def state_label(states):
    return '/'.join(f'S{int(s)}' for s in states)


def enrich_metadata(table, dataset, registry=None):
    """Join by cup and resident tag; repeated tags in different cups stay distinct."""
    table = table.copy()
    identities = pd.read_csv(dataset / 'cup_identities.csv')
    key = 'TrackID' if 'TrackID' in identities else 'CupID'
    lookup = identities.set_index(key)
    if lookup.index.duplicated().any():
        raise ValueError('Resident track IDs must be unique')
    records = pd.read_csv(registry, sep='\t') if registry is not None else None
    for i, row in table.iterrows():
        if pd.isna(row.TrackID):
            continue
        identity = lookup.loc[int(row.TrackID)]
        if int(identity.get('CupID', int(row.TrackID))) != int(row.CupID):
            raise ValueError('Track/cup identity mismatch')
        tag = int(identity.ArucoID)
        if 'ExpectedID' not in row or pd.isna(row.ExpectedID):
            table.loc[i, 'ExpectedID'] = tag
        if records is not None:
            matched = records.loc[records.cup.eq(row.CupID) & records.tag_id.eq(tag)]
            if len(matched) != 1:
                raise ValueError(f'Expected one documented resident for cup {row.CupID}, tag {tag}')
            resident = matched.iloc[0]
            for field, source in [('Colony', 'colony'), ('Role', 'role'), ('ResidentStatus', 'status'),
                                  ('ResidentFrom', 'in_cup_from'), ('ResidentUntil', 'in_cup_until'), ('ResidentNote', 'note')]:
                table.loc[i, field] = resident.get(source, '')
    for field in ['Colony', 'Role']:
        if field not in table:
            table[field] = 'unknown'
        table[field] = table[field].fillna('unknown')
    if records is not None:
        table['_role_order'] = table.Role.map({'nurse': 0, 'forager': 1}).fillna(2)
        table = table.sort_values(['Colony', '_role_order', 'CupID']).drop(columns='_role_order')
    table['ExpectedID'] = pd.to_numeric(table['ExpectedID']).astype('Int64')
    return table.reset_index(drop=True)


def bout_confusions(quiet, positive, negative, durations, boundaries=()):
    """Exact whole-bout counts without materializing one mask per duration."""
    starts, stops = true_runs(quiet)
    cuts = np.asarray(boundaries, dtype=int)
    cuts = cuts[(cuts > 0) & (cuts < len(quiet))]
    cuts = cuts[quiet[cuts - 1] & quiet[cuts]]
    if len(cuts):
        starts = np.sort(np.r_[starts, cuts]); stops = np.sort(np.r_[stops, cuts])
    p = np.r_[0, np.cumsum(positive, dtype=np.int64)]
    q = np.r_[0, np.cumsum(negative, dtype=np.int64)]
    hits_p, hits_q = p[stops] - p[starts], q[stops] - q[starts]
    result = []
    for duration in durations:
        keep = stops - starts >= duration
        tp, fp = int(hits_p[keep].sum()), int(hits_q[keep].sum())
        result.append([tp, fp, int(p[-1]) - tp, int(q[-1]) - fp])
    return np.asarray(result, dtype=np.int64)


def extract(dataset, metadata, output, speed_root=None, sleep_states=(0, 1), state_root=None,
            registry=None, bin_seconds=600):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset / 'manifest.json').read_text())
    recorded = support_mask(manifest)
    state_root = state_root or dataset / 'analysis/sleep_s01_exclusion'
    if output.resolve() in {dataset.resolve(), metadata.resolve(), state_root.resolve()}:
        raise ValueError('Use a separate report directory to preserve source states and metadata')
    source_definition = json.loads((state_root / 'definition.json').read_text())
    n_states = source_definition.get('model_selection', {}).get('n_states')
    if n_states is None:
        with np.load(dataset / 'analysis/eigenposture/models/dictionary.npz') as dictionary:
            n_states = int(dictionary['state'].max()) + 1
    n_states = int(n_states)
    sleep, other = state_groups(n_states, sleep_states)
    table = enrich_metadata(pd.read_csv(metadata / 'per_ant_summary.csv'), dataset, registry)
    from tracking.single_ant.state_labels import recording_bins
    ranges = recording_bins(manifest, seconds=bin_seconds)
    starts, stops, rec = ranges.T
    bins = dict(start_frame=starts, stop_frame=stops, recorded=rec.astype(bool),
                time_h=(starts + stops) / 2 / manifest['fps'] / 3600, cup_ids=table.CupID.to_numpy())
    output_bins = dict(bins, speed=np.full((len(table), len(starts)), np.nan),
                       coverage=np.full((len(table), len(starts)), np.nan))
    fps = float(manifest['fps'])
    n = len(recorded)
    parity = (np.arange(n) // int(fps * 1800)) % 2
    hist = np.zeros((len(table), 2, n_states + 1, len(SPEED_CUTS)), np.int64)
    early_hist = np.zeros((len(table), n_states + 1, len(SPEED_CUTS)), np.int64)
    bout_counts = np.zeros((len(table), len(BOUT_SPEEDS), len(BOUT_SECONDS), 4), np.int64)
    fractions = np.full((len(table), len(bins['time_h']), 3), np.nan)
    rows = []
    files = []
    for i, row in table.iterrows():
        result = row.to_dict()
        identity_ok = row.IdentityStatus == 'matched'
        has_track = pd.notna(row.TrackID)
        category = np.full(n, -1, np.int8)
        observed = np.full(n, -1, np.int8)
        evidence = np.zeros(n, np.uint8)
        speed = np.full(n, np.nan, np.float32)
        if has_track:
            stem = f'TrackID_{int(row.TrackID):04d}_all_single'
            statefile = state_root / 'per_track' / stem / 'labels.npz'
            with np.load(statefile) as data:
                category, observed, evidence = [data[k] for k in ['category', 'observed', 'evidence']]
            files.append(dict(path=str(statefile), sha256=digest(statefile)))
            if identity_ok:
                speedfile = (speed_root or dataset / 'analysis/speed_vectors') / 'per_track' / stem / 'speed_mm_s.npy'
                speed = np.load(speedfile, mmap_mode='r')
                files.append(dict(path=str(speedfile), size=speedfile.stat().st_size,
                                  mtime_ns=speedfile.stat().st_mtime_ns))
        assert len(category) == n and len(observed) == n and len(speed) == n
        assert np.all(observed[~recorded] < 0)
        assert np.all((observed >= -1) & (observed < n_states))
        if not identity_ok:
            assert np.all(observed < 0), 'Identity-excluded cup has observed state labels'
        known = recorded & (observed >= 0) & identity_ok & ((evidence & 1) > 0)
        category = np.full(n, -1, np.int8)
        category[known] = np.isin(observed[known], sleep).astype(np.int8)
        observed = np.where(known, observed, -1).astype(np.int8)
        evidence = known.astype(np.uint8)
        folder = output / 'per_track' / (stem if has_track else f'TrackID_{int(row.CupID):04d}_all_single')
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'labels.npz', category=category, observed=observed, evidence=evidence)
        totals = np.array([(recorded & (category == v)).sum() for v in [1, 0, -1]])
        assert totals.sum() == recorded.sum()
        for j, (start, stop, rec) in enumerate(zip(bins['start_frame'], bins['stop_frame'], bins['recorded'])):
            if rec:
                fractions[i, j] = [(category[start:stop] == v).mean() for v in [1, 0, -1]]
        valid_speed = recorded & np.isfinite(speed) & (speed >= 0) & (speed <= 20)
        if speed_root is not None:
            result['PreviousFusedMeanSpeed'] = result.get('MeanSpeed')
            result.update(RawMeanSpeed=np.nan, RawAbove20Fraction=np.nan)
        finite = speed[valid_speed]
        result.update(MeanSpeed=float(np.mean(finite)) if len(finite) else np.nan,
            MedianSpeed=float(np.median(finite)) if len(finite) else np.nan,
            P95Speed=float(np.quantile(finite, .95)) if len(finite) else np.nan,
            ValidSpeedFraction=float(valid_speed.sum() / recorded.sum()), ValidSpeedFrames=int(valid_speed.sum()))
        first = recorded & (np.arange(n) < fps * 6 * 3600)
        early_valid = first & valid_speed
        result['First6hCoverage'] = float(early_valid.sum() / first.sum())
        result['First6hMean'] = float(speed[early_valid].mean()) if result['First6hCoverage'] >= .5 else np.nan
        for j, (start, stop, rec) in enumerate(zip(bins['start_frame'], bins['stop_frame'], bins['recorded'])):
            if rec:
                coverage = valid_speed[start:stop].mean()
                output_bins['coverage'][i, j] = coverage
                if coverage >= .5:
                    output_bins['speed'][i, j] = float(speed[start:stop][valid_speed[start:stop]].mean())
        joint = valid_speed & known
        result.update(SleepPercent=100 * totals[0] / recorded.sum(),
                      NotSleepPercent=100 * totals[1] / recorded.sum(),
                      AmbiguousPercent=100 * totals[2] / recorded.sum(),
                      MotionInferredNotPercent=100 * ((evidence & 6 > 0) & recorded).sum() / recorded.sum(),
                      ObservedStatePercent=100 * known.sum() / recorded.sum(),
                      JointSpeedStatePercent=100 * joint.sum() / recorded.sum(),
                      JointSpeedStateFrames=int(joint.sum()),
                      ObservedStateFrames=int(known.sum()), RecordedFrames=int(recorded.sum()),
                      StateCohort=bool(known.any()))
        result.pop('NotS01Percent', None)
        if sleep.tolist() == [0, 1]:
            result['NotS01Percent'] = result['NotSleepPercent']
        state_index = np.where(observed >= 0, observed, n_states).astype(np.int64)
        vi = np.flatnonzero(valid_speed)
        speed_index = np.searchsorted(SPEED_CUTS, speed[vi], side='left')
        index = (parity[vi] * (n_states + 1) + state_index[vi]) * len(SPEED_CUTS) + speed_index
        hist[i] = np.bincount(index, minlength=2 * (n_states + 1) * len(SPEED_CUTS)).reshape(2, n_states + 1, -1)
        early = vi < fps * 6 * 3600
        index = state_index[vi[early]] * len(SPEED_CUTS) + speed_index[early]
        early_hist[i] = np.bincount(index, minlength=(n_states + 1) * len(SPEED_CUTS)).reshape(n_states + 1, -1)
        if joint.any():
            positive = joint & np.isin(observed, sleep)
            negative = joint & np.isin(observed, other)
            for a, limit in enumerate(BOUT_SPEEDS):
                quiet = valid_speed & (speed <= limit)
                bout_counts[i, a] = bout_confusions(quiet, positive, negative, np.rint(BOUT_SECONDS * fps).astype(int),
                                                   [s['frame_offset'] for s in manifest['recording_segments']])
        rows.append(result)
        print(f"Cup {int(row.CupID):02d}: observed {result['ObservedStatePercent']:.1f}%, joint {result['JointSpeedStatePercent']:.1f}%", flush=True)
    pd.DataFrame(rows).to_csv(output / 'per_ant_summary.csv', index=False)
    np.savez_compressed(output / 'summaries.npz', hist=hist, early_hist=early_hist,
                        bout_counts=bout_counts, fractions=fractions, thresholds=SPEED_CUTS,
                        bout_speeds=BOUT_SPEEDS, bout_seconds=BOUT_SECONDS,
                        **output_bins)
    write_json(output / 'provenance.json', dict(dataset=str(dataset),
        origin=manifest['start_datetime'], fps=fps, n_frames=n, recorded_frames=int(recorded.sum()),
        segments=manifest['recording_segments'],
        manifest_sha256=digest(dataset / 'manifest.json'),
        sleep_states=sleep.tolist(), other_states=other.tolist(), n_states=n_states,
        sleep_definition=dict(sleep_states=sleep.tolist(), source_definition=source_definition,
                              positive_rule='Observed full-posture state belongs to the user-selected putative sleep set',
                              unknown_rule='Unobserved posture, excluded identity, and unrecorded intervals remain unknown'),
        state_source=str(state_root), state_definition_sha256=digest(state_root / 'definition.json'),
        registry_source=str(registry) if registry is not None else None,
        registry_sha256=digest(registry) if registry is not None else None,
        bin_seconds=bin_seconds,
        metadata_file=str(metadata / 'per_ant_summary.csv'),
        metadata_sha256=digest(metadata / 'per_ant_summary.csv'),
        speed_definition=json.loads((speed_root / 'definition.json').read_text()) if speed_root is not None else
            dict(label='Fused single-tag position', description='TrackX/TrackY only: decoded ArUco position, or SLEAP tag anchor when ArUco is missing; no antenna/body speed aggregation. Short gaps ≤5 frames interpolated; coordinate Gaussian smoothing σ=2 frames; >20 mm/s excluded.'),
        extraction_code_sha256=digest(__file__), inputs=files,
        validation_target=f'Observed {state_label(sleep)} versus observed {state_label(other)}; unknown and inferred labels excluded',
        threshold_grid_mm_s=SPEED_CUTS.tolist(),
        bout_rule='Retrospective entire low-speed runs; missing speed or recording gaps break runs'))
    write_json(output / 'definition.json', dict(sleep_states=sleep.tolist(), other_states=other.tolist(),
               n_states=n_states, categories={'1':state_label(sleep), '0':'other observed states', '-1':'unknown'},
               model_supported=source_definition.get('model_supported'), state_source=str(state_root),
               source_definition_sha256=digest(state_root / 'definition.json'),
               dictionary_sha256=source_definition.get('reference_dictionary_sha256'),
               motion_inferred_labels=False))


def confusion_from_hist(hist, sleep_states=(0, 1)):
    """Axis -2: original state; -1: speed's first inclusive upper threshold."""
    sleep, other = state_groups(hist.shape[-2] - 1, sleep_states)
    pos = hist[..., sleep, :].sum(axis=-2)
    neg = hist[..., other, :].sum(axis=-2)
    tp, fp = pos.cumsum(axis=-1), neg.cumsum(axis=-1)
    return np.stack([tp, fp, pos.sum(axis=-1)[..., None] - tp,
                     neg.sum(axis=-1)[..., None] - fp], axis=-1)


def metrics(counts):
    tp, fp, fn, tn = np.moveaxis(np.asarray(counts, float), -1, 0)
    def div(a, b):
        return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=b > 0)
    sensitivity = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    return dict(sensitivity=sensitivity, specificity=specificity, precision=div(tp, tp + fp),
                balanced_accuracy=(sensitivity + specificity) / 2,
                accuracy=div(tp + tn, tp + fp + fn + tn), f1=div(2 * tp, 2 * tp + fp + fn),
                predicted_sleep_fraction=div(tp + fp, tp + fp + fn + tn))


def ant_weights(counts):
    total = counts.sum(axis=-1)
    return np.divide(counts, total[..., None], out=np.zeros_like(counts, dtype=float), where=total[..., None] > 0)


def choose_cut(counts):
    score = np.nanmean(metrics(counts)['balanced_accuracy'], axis=0)
    return int(np.nanargmax(score))


def cv_counts(counts):
    indices, predictions = [], []
    for i in range(len(counts)):
        cut = choose_cut(np.delete(counts, i, axis=0))
        indices.append(cut)
        predictions.append(counts[i, cut])
    return np.array(indices), np.array(predictions)


def bootstrap_macro(values, seed=20261102):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), lower=float(np.quantile(samples, .025)), upper=float(np.quantile(samples, .975)))


def analyse(output):
    provenance = json.loads((output / 'provenance.json').read_text())
    sleep, other = state_groups(provenance.get('n_states', 6), provenance.get('sleep_states', [0, 1]))
    n_states = provenance.get('n_states', 6)
    provenance['analysis_code_sha256'] = digest(__file__)
    write_json(output / 'provenance.json', provenance)
    table = pd.read_csv(output / 'per_ant_summary.csv')
    z = np.load(output / 'summaries.npz')
    hist = z['hist'].sum(axis=1)
    counts_all = confusion_from_hist(hist, sleep)
    use = (hist[:, sleep].sum(axis=(1, 2)) > 0) & (hist[:, other].sum(axis=(1, 2)) > 0)
    counts = counts_all[use]
    selected = table.loc[use].reset_index(drop=True)
    cuts, heldout = cv_counts(counts)
    result = selected[['CupID', 'ExpectedID', 'Colony', 'Role', 'JointSpeedStatePercent']].copy()
    result['LOAOThreshold'] = SPEED_CUTS[cuts]
    for key, value in metrics(heldout).items():
        result[key] = value
    from sklearn.metrics import auc
    per_metrics = metrics(counts)
    # Include the (0,0) origin omitted by the first threshold when speed is exactly zero.
    result['AUROC'] = [auc(np.r_[0, 1 - sp], np.r_[0, se])
                       for sp, se in zip(per_metrics['specificity'], per_metrics['sensitivity'])]
    result.to_csv(output / 'proxy_per_ant.csv', index=False)
    macro = {k: bootstrap_macro(v) for k, v in result.items() if k in [*metrics(heldout), 'AUROC']}
    fixed_rows = []
    for cutoff in [.01, .02, .05, .1, .2, .5, .66, 1.]:
        c = counts[:, int(round(cutoff / .005))]
        m = metrics(c)
        row = dict(speed_threshold_mm_s=cutoff, n_ants=len(c))
        row.update({k: float(np.nanmean(v)) for k, v in m.items()})
        row['false_sleep_fraction_of_predicted'] = 1 - row['precision']
        row['pooled_sleep_precision'] = float(metrics(c.sum(axis=0))['precision'])
        row['equal_ant_sleep_precision'] = float(metrics(ant_weights(c).mean(axis=0))['precision'])
        all_valid = hist[use].sum(axis=1).cumsum(axis=-1)[:, int(round(cutoff / .005))]
        unknown = hist[use, n_states].cumsum(axis=-1)[:, int(round(cutoff / .005))]
        row['stationary_time_without_observed_state_fraction'] = float(np.mean(np.divide(unknown, all_valid, out=np.zeros_like(unknown, dtype=float), where=all_valid > 0)))
        row['ants_with_sleep_predictions'] = int(((c[:, 0] + c[:, 1]) > 0).sum())
        fixed_rows.append(row)
    pd.DataFrame(fixed_rows).to_csv(output / 'stationarity_thresholds.csv', index=False)
    # Duration and threshold are both selected within each training set.
    bouts = z['bout_counts'][use].reshape(len(counts), -1, 4)
    bout_cuts, bout_heldout = cv_counts(bouts)
    bout_result = selected[['CupID', 'ExpectedID', 'Colony', 'Role']].copy()
    bout_result['speed_threshold_mm_s'] = BOUT_SPEEDS[bout_cuts // len(BOUT_SECONDS)]
    bout_result['minimum_bout_seconds'] = BOUT_SECONDS[bout_cuts % len(BOUT_SECONDS)]
    for key, value in metrics(bout_heldout).items():
        bout_result[key] = value
    bout_result.to_csv(output / 'bout_proxy_per_ant.csv', index=False)
    bout_macro = {k: bootstrap_macro(v) for k, v in metrics(bout_heldout).items()}
    duration_rows = []
    for a, cutoff in enumerate(BOUT_SPEEDS):
        for b, seconds in enumerate(BOUT_SECONDS):
            row = dict(speed_threshold_mm_s=float(cutoff), minimum_bout_seconds=int(seconds))
            row.update({k: float(np.mean(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan
                        for k, v in metrics(z['bout_counts'][use, a, b]).items()})
            row['ants_with_sleep_predictions'] = int((z['bout_counts'][use, a, b, :2].sum(axis=1) > 0).sum())
            duration_rows.append(row)
    pd.DataFrame(duration_rows).to_csv(output / 'bout_thresholds.csv', index=False)
    # Separate temporal holdout; state dictionary stays fixed in all these tests.
    temporal = confusion_from_hist(z['hist'][use, 0], sleep)
    temporal_cut = choose_cut(temporal)
    temporal_test = confusion_from_hist(z['hist'][use, 1], sleep)[:, temporal_cut]
    early_counts = confusion_from_hist(z['early_hist'][use], sleep)
    early_use = (early_counts[:, -1, 0] > 0) & (early_counts[:, -1, 1] > 0)
    _, early_test = cv_counts(early_counts[early_use])
    summary = dict(n_ants=int(use.sum()), n_joint_frames=int(hist[use, :n_states].sum()),
        n_joint_ant_hours=float(hist[use, :n_states].sum() / provenance['fps'] / 3600),
        original_sleep_prevalence=float(np.mean(hist[use][:, sleep].sum(axis=(1, 2)) / hist[use, :n_states].sum(axis=(1, 2)))),
        sleep_states=sleep.tolist(), other_states=other.tolist(), n_states=n_states,
        loao_speed=macro, loao_bouts=bout_macro,
        speed_threshold_range_mm_s=[float(SPEED_CUTS[cuts].min()), float(SPEED_CUTS[cuts].max())],
        speed_threshold_median_mm_s=float(np.median(SPEED_CUTS[cuts])),
        temporal_holdout=dict(threshold_mm_s=float(SPEED_CUTS[temporal_cut]),
            metrics={k: bootstrap_macro(v) for k, v in metrics(temporal_test).items()}),
        first_6h_loao=dict(n_ants=int(early_use.sum()), metrics={k: bootstrap_macro(v) for k, v in metrics(early_test).items()}),
        uncertainty='95% percentile bootstrap of held-out ant scores, 5000 replicates; conditional on fixed learned dictionary and fitted fold thresholds',
        validation_target=f'Observed {state_label(sleep)} versus observed {state_label(other)}. Inferred labels and unknown frames excluded.',
        caveat='Learned states are exploratory and include motion features. This tests recovery of operational posture-state labels, not physiological sleep.')
    # Keep identities with no posture labels visible in the group denominator.
    groups = []
    for (colony, role), group in table.loc[table.IdentityStatus.eq('matched')].groupby(['Colony', 'Role'], sort=False):
        record = dict(Colony=colony, Role=role, Ants=len(group), AntsWithPostureStates=int(group.StateCohort.sum()))
        record.update({key: float(group[key].mean()) for key in ['SleepPercent', 'NotSleepPercent', 'AmbiguousPercent',
            'MotionInferredNotPercent', 'MeanSpeed', 'ValidSpeedFraction']})
        groups.append(record)
    pd.DataFrame(groups).to_csv(output / 'colony_role_summary.csv', index=False)
    write_json(output / 'proxy_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    from textwrap import fill

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    table = pd.read_csv(output / 'per_ant_summary.csv')
    z = np.load(output / 'summaries.npz')
    summary = json.loads((output / 'proxy_summary.json').read_text())
    provenance = json.loads((output / 'provenance.json').read_text())
    from datetime import datetime
    sleep, other = state_groups(provenance.get('n_states', 6), provenance.get('sleep_states', [0, 1]))
    sleep_name, other_name = state_label(sleep), state_label(other)
    n_states = provenance.get('n_states', 6)
    origin_label = datetime.fromisoformat(provenance['origin']).strftime('%Y-%m-%d %H:%M:%S %Z')
    dataset_label = ' + '.join(s.get('block', s.get('name', 'recording')) for s in provenance['segments'])
    recorded_hours = provenance['recorded_frames'] / provenance['fps'] / 3600
    gap_seconds = (provenance['n_frames'] - provenance['recorded_frames']) / provenance['fps']
    source_definition = provenance.get('sleep_definition', {}).get('source_definition', {})
    dictionary_status = 'exploratory' if source_definition.get('model_supported') is False else 'fitted'
    provenance['render_code_sha256'] = digest(__file__)
    write_json(output / 'provenance.json', provenance)
    fixed = pd.read_csv(output / 'stationarity_thresholds.csv')
    bouts = pd.read_csv(output / 'bout_thresholds.csv')
    per_ant = pd.read_csv(output / 'proxy_per_ant.csv')
    labels = [f"{r.Colony} · {r.Role} · cup {r.CupID:02d} / ID {r.ExpectedID}" for r in table.itertuples()]
    speed_label = provenance.get('speed_definition', {}).get('label', 'Fused single-tag position')
    speed_description = provenance.get('speed_definition', {}).get('description',
        'Fused single-tag TrackX/TrackY; ArUco detections bridged with the SLEAP tag anchor. No other node speed enters the predictor.')
    matched = table.IdentityStatus.eq('matched')
    full = table.loc[matched]
    percentages = full[['SleepPercent', 'NotSleepPercent', 'AmbiguousPercent']].mean()
    n = len(table)
    xedges = np.r_[z['start_frame'], z['stop_frame'][-1]] / provenance['fps'] / 3600
    centers = z['time_h']
    speed = z['speed']
    maximum_speed = float(np.nanmax(speed))
    boundaries = [i - .5 for i in range(1, n) if table.loc[i, 'Colony'] != table.loc[i - 1, 'Colony']]
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    gallery = []
    pdf = PdfPages(output / 'sleep_speed_summary.pdf')

    def save(fig, name, title, description):
        fig.savefig(figures / f'{name}.png', dpi=150, bbox_inches='tight')
        fig.savefig(figures / f'{name}.svg', bbox_inches='tight')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        gallery.append(dict(name=name, title=title, description=description))

    def pct(key, method='loao_speed'):
        result = summary[method][key]
        return f"{100 * result['mean']:.1f}% (95% CI {100 * result['lower']:.1f}–{100 * result['upper']:.1f}%)"

    quiet = fixed.loc[np.isclose(fixed.speed_threshold_mm_s, .1)].iloc[0]
    quiet05 = fixed.loc[np.isclose(fixed.speed_threshold_mm_s, .05)].iloc[0]
    conclusion = (
        f"Using {sleep_name} as putative sleep, speed discrimination has mean held-out-ant AUROC "
        f"{summary['loao_speed']['AUROC']['mean']:.3f}. "
        f"At ≤0.10 mm/s, {100 * (1 - quiet.precision):.1f}% of speed-defined sleep is another observed state "
        f"(mean across ants). Thresholds tuned on other ants give {pct('balanced_accuracy')} balanced accuracy. "
        f"These results quantify agreement with the fixed {dictionary_status} posture-state dictionary; physiological sleep is unverified.")
    methods = [
        f'Dataset: {dataset_label}. Start {origin_label}; {recorded_hours:.3f} recorded hours at '
        f"{provenance['fps']:g} Hz. Unrecorded intervals total {gap_seconds:.2f} seconds and remain unknown.",
        'Metadata: actual resident identity from cup history; ordered by origin colony, then recorded nurse/forager role, '
        'then cup. Colony “#14 or #10-2” and unknown metadata are separate groups. Raw tag IDs are not unique across cups.',
        f'Display: purple = observed {sleep_name} (putative sleep); amber = observed {other_name}; gray = unknown. '
        'Percentages use recorded time, including missing observations. Identity-excluded cups are retained in displays and excluded from valid summaries.',
        'Only the user-selected state grouping changes. Original state assignments and the fitted dictionary are preserved. '
        'Missing posture stays unknown; no speed-based or motion-inferred decisions fill the label vector. Resident-history notes are retained for review.',
        f'Speed predictor: {speed_label}. {speed_description} Ten-minute means require ≥50% valid speed; missing bins stay blank. '
        'The full-record per-ant mean uses all finite standard speed samples.',
        f'Proxy targets: observed {sleep_name} versus observed {other_name}, with finite tag-position speed and valid identity. '
        'Ambiguous and motion-inferred negatives never enter validation. All jointly observed frames contribute; no random frame split.',
        'Validation: leave one whole ant out; choose the speed threshold on the other ants to maximize equal-ant balanced accuracy '
        'on a 0.005 mm/s grid from 0 to 20. Evaluate the held-out ant; report macro ant scores. AUROC approximates ranking by negative '
        'speed using the same 0.005 mm/s bins (within-bin values are tied). '
        'Confidence intervals resample held-out ant scores 5,000 times, conditional on the fixed dictionary and fold fits. '
        'They treat ants as sampling units and do not model shared source-colony effects.',
        'Stationary-bout proxy: thresholds 0.02/0.05/0.1/0.2/0.5/1 mm/s and minimum durations 0/1/5/10/30/60/300 s. '
        'Select both parameters using training ants only. Credit the entire qualifying bout retrospectively, favoring this proxy; '
        'missing speed or an unrecorded gap breaks a bout. All labeled frames, including moving frames, remain in its test denominator.',
        'Sensitivity: thresholds trained on even 30-minute blocks are evaluated on odd blocks; a separate leave-one-ant-out analysis '
        'uses the first six hours. Neither test retrains the posture-state dictionary.',
        f'Interpretation: {sleep_name} are user-selected putative sleep labels from a {dictionary_status} {n_states}-state model. The state model itself '
        'includes motion features, so agreement with speed is not independent biological validation. Whole-ant holdout applies to '
        'the speed threshold, not the learned state dictionary. Coverage selection and declining/weak ants limit generalization; '
        'stationarity can include waking behavior, poor tracking, or immobility. The comparison does not isolate static posture '
        'from antennal motion or other dynamic features, and does not test more complex temporal speed models. '
        'No claim of physiological necessity follows from labels alone.'
    ]
    fig = plt.figure(figsize=(12, 11))
    fig.text(.07, .94, 'Single-ant sleep and speed', fontsize=24, weight='bold')
    fig.text(.07, .905, dataset_label + ' · colony → role → cup', fontsize=11, color='#555')
    text = (f"Speed: {speed_label}; no antenna or other-node speed in the predictor.\n"
            f"{int(matched.sum())} identity-matched ants · {n} cups shown · {summary['n_ants']} ants in posture/speed validation\n"
            f"Matched-ant recorded time: {percentages.iloc[0]:.1f}% sleep, {percentages.iloc[1]:.1f}% not {sleep_name}, "
            f"{percentages.iloc[2]:.1f}% ambiguous\n\n" + fill(conclusion, 112) + '\n\n'
            f"Optimized speed proxy, leave one ant out:\n"
            f"    Sensitivity: {pct('sensitivity')}\n    Specificity: {pct('specificity')}\n"
            f"    Precision: {pct('precision')}\n"
            f"    Threshold: median {summary['speed_threshold_median_mm_s']:.3f} mm/s "
            f"(fold range {summary['speed_threshold_range_mm_s'][0]:.3f}–{summary['speed_threshold_range_mm_s'][1]:.3f})\n\n"
            f"Optimized stationary-bout proxy: {pct('balanced_accuracy', 'loao_bouts')} balanced accuracy\n\n"
            f"At ≤0.05 mm/s: {100 * quiet05.precision:.1f}% precision; {100 * quiet05.sensitivity:.1f}% sensitivity.\n"
            f"At ≤0.10 mm/s: {100 * quiet.precision:.1f}% precision; {100 * quiet.sensitivity:.1f}% sensitivity.\n\n"
            'These comparisons exclude time without original state labels, including motion-filled negatives.\n\n'
            + fill(f'The {n_states}-state dictionary is {dictionary_status} and includes movement features. The measurements quantify how much '
                   'speed recovers the user-selected putative labels; they do not independently validate physiological sleep.', 112))
    fig.text(.07, .85, text, fontsize=12, va='top', linespacing=1.55)
    save(fig, '01_findings', 'Findings and scope', conclusion)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 14), sharey=True, gridspec_kw={'width_ratios': [1.45, 1]})
    y = np.arange(n)
    left = np.zeros(n)
    for column, color, title in zip(['SleepPercent', 'NotSleepPercent', 'AmbiguousPercent'], COLORS,
                                    [f'Putative sleep ({sleep_name})', 'Other observed states', 'Ambiguous']):
        values = table[column].to_numpy()
        axes[0].barh(y, values, left=left, color=color, height=.78, label=title)
        for row, (start, value) in enumerate(zip(left, values)):
            if value >= 7:
                axes[0].text(start + value / 2, row, f'{value:.0f}', ha='center', va='center', fontsize=8,
                             color='white' if color != COLORS[2] else '#333')
        left += values
    axes[0].set(yticks=y, yticklabels=labels, ylim=(n - .5, -.5), xlim=(0, 100), xlabel='% of recorded time')
    axes[0].tick_params(axis='y', labelsize=9)
    axes[0].legend(loc='lower center', bbox_to_anchor=(.5, 1.005), ncol=3, fontsize=10)
    axes[1].barh(y, table.MeanSpeed, color='#287c8e', height=.68)
    for i, row in table.iterrows():
        if pd.notna(row.MeanSpeed):
            axes[1].text(row.MeanSpeed + .04, i, f'{row.MeanSpeed:.2f}  ({100 * row.ValidSpeedFraction:.0f}%)', va='center', fontsize=8)
        else:
            axes[1].text(.04, i, row.IdentityStatus, va='center', fontsize=8, color='#777')
    axes[1].set(xlim=(0, float(table.MeanSpeed.max()) * 1.34), xlabel='Mean tag speed (mm/s); valid coverage in parentheses')
    for ax in axes:
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
        for boundary in boundaries:
            ax.axhline(boundary, color='#333', lw=1)
    for i, row in table.iterrows():
        if not row.StateCohort:
            axes[0].text(101, i, '†', va='center', fontsize=10)
    fig.suptitle(f'Each ant, ordered by colony then recorded role · {speed_label}', fontsize=17, y=.99)
    fig.text(.02, .014, '† No valid posture-state assignments. Gray includes missing data. '
             'Numbers inside bars are percentages of recorded time.', fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .975))
    save(fig, '02_ordered_ants', 'Sleep categories and speed per ant',
         'Identical ant ordering on both panels. Sleep percentages include ambiguous recorded time in their denominator; speed uses measured samples.')

    fig, axes = plt.subplots(1, 4, figsize=(20, 14), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1, 1.15]})
    for k, ax in enumerate(axes[:3]):
        cmap = LinearSegmentedColormap.from_list(f'category{k}', ['white', COLORS[k]])
        mesh = ax.pcolormesh(xedges, np.arange(n + 1) - .5, z['fractions'][:, :, k], cmap=cmap, vmin=0, vmax=1, shading='flat', rasterized=True)
        fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=.035, fraction=.025, label='Fraction of 10-min bin')
        ax.set_title([f'Putative sleep ({sleep_name})', 'Other observed states', 'Ambiguous'][k])
    mesh = axes[3].pcolormesh(xedges, np.arange(n + 1) - .5, speed, cmap='viridis', vmin=0, vmax=maximum_speed, rasterized=True)
    fig.colorbar(mesh, ax=axes[3], orientation='horizontal', pad=.035, fraction=.025, label='Mean speed (mm/s)')
    axes[3].set_title('Tag-position speed')
    for ax in axes:
        ax.set(xlim=(0, xedges[-1]), ylim=(n - .5, -.5), xlabel='Hours from ' + origin_label)
        for boundary in boundaries:
            ax.axhline(boundary, color='#555', lw=.7)
        for i, rec in enumerate(z['recorded']):
            if not rec:
                ax.axvspan(xedges[i], xedges[i + 1], color='white', hatch='////', lw=0)
    axes[0].set_yticks(y, labels, fontsize=8)
    fig.suptitle('Sleep composition and speed over the same recorded timeline', fontsize=20, y=.985)
    fig.text(.02, .01, 'Each fraction panel ranges from white (0%) to its category color (100%). '
             'Speed bins require ≥50% valid samples; blanks are missing. The narrow hatched gap was not recorded.', fontsize=11)
    fig.tight_layout(rect=(0, .03, 1, .97))
    save(fig, '03_timeline_overview', 'Aligned timeline overview',
         'Ten-minute category fractions and standard speed on identical rows and time axes; no missing samples are treated as zero speed.')

    # Per-colony pages: exact category fractions and a separate speed axis per ant.
    page = 4
    for colony, group in table.groupby('Colony', sort=False):
        indices = group.index.to_numpy()
        for offset in range(0, len(indices), 6):
            chunk = indices[offset:offset + 6]
            page_height = 2.25 * len(chunk) + 1.5
            fig, axes = plt.subplots(len(chunk) * 2, 1, figsize=(15, page_height), sharex=True,
                                      gridspec_kw={'height_ratios': [1, 1.2] * len(chunk), 'hspace': .14})
            axes = np.atleast_1d(axes)
            for a, i in enumerate(chunk):
                row = table.loc[i]
                top, bottom = axes[2 * a:2 * a + 2]
                lower = np.zeros(len(centers))
                for k, color in enumerate(COLORS):
                    values = z['fractions'][i, :, k]
                    top.bar(xedges[:-1], values, width=np.diff(xedges), bottom=lower, align='edge', color=color, linewidth=0)
                    lower += values
                top.set(ylim=(0, 1), yticks=[0, 1], yticklabels=['0', '100%'])
                top.set_title(f"Cup {int(row.CupID):02d} · resident ID {int(row.ExpectedID)} · {row.Role} · "
                              f"sleep {row.SleepPercent:.1f}% / not {row.NotSleepPercent:.1f}% / ambiguous {row.AmbiguousPercent:.1f}%",
                              loc='left', fontsize=10, pad=3)
                bottom.plot(centers, speed[i], color='#287c8e', lw=1.15)
                bottom.set(ylim=(0, maximum_speed * 1.05), ylabel='mm/s')
                bottom.grid(alpha=.15)
                if not row.StateCohort:
                    bottom.text(.99, .78, 'No original posture labels' if row.IdentityStatus == 'matched' else row.IdentityStatus,
                                transform=bottom.transAxes, ha='right', color='#666', fontsize=9)
                for ax in [top, bottom]:
                    for j, rec in enumerate(z['recorded']):
                        if not rec:
                            ax.axvspan(xedges[j], xedges[j + 1], color='#eee', hatch='////', lw=0)
            axes[-1].set(xlim=(0, xedges[-1]), xlabel='Hours from ' + origin_label)
            fig.suptitle(f'Colony {colony} · recorded role order · cups {offset + 1}–{offset + len(chunk)} of {len(indices)}',
                         y=1 - .08 / page_height, fontsize=16)
            fig.legend(handles=[Patch(color=c, label=t) for c, t in zip(COLORS, [f'Putative sleep ({sleep_name})', 'Other observed states', 'Ambiguous'])],
                       loc='upper center', bbox_to_anchor=(.5, 1 - .36 / page_height), ncol=3)
            fig.subplots_adjust(left=.07, right=.98, top=1 - 1.0 / page_height, bottom=.5 / page_height)
            save(fig, f'{page:02d}_colony_{colony.replace("#", "").replace(" ", "_")}_{offset // 6 + 1}',
                 f'Colony {colony}: ant time courses, part {offset // 6 + 1}',
                 'Within each ant: stacked sleep-category fractions above mean speed; all ants share the same speed scale. No time periods are silently filled.')
            page += 1

    hist = z['hist'].sum(axis=1)
    use = table.CupID.isin(per_ant.CupID).to_numpy(bool)
    counts = confusion_from_hist(hist[use], sleep)
    weighted = ant_weights(counts).mean(axis=0)
    curves = metrics(weighted)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes[0, 0].plot(np.r_[0, 1 - curves['specificity']], np.r_[0, curves['sensitivity']], color=COLORS[0], lw=2)
    axes[0, 0].plot([0, 1], [0, 1], '--', color='#aaa')
    axes[0, 0].scatter(1 - summary['loao_speed']['specificity']['mean'], summary['loao_speed']['sensitivity']['mean'],
                        color='#222', s=45, label='Held-out-ant mean at trained thresholds', zorder=3)
    axes[0, 0].set(xlim=(0, 1), ylim=(0, 1), xlabel=f'False-positive rate ({other_name} called sleep)',
                    ylabel=f'Sensitivity ({sleep_name} recovered)', title='Speed ranking: ROC')
    axes[0, 0].legend(fontsize=8, loc='lower right')
    axes[0, 1].plot(curves['sensitivity'], curves['precision'], color=COLORS[0], lw=2)
    axes[0, 1].axhline(summary['original_sleep_prevalence'], color='#999', ls='--', label='Observed sleep prevalence')
    axes[0, 1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Sensitivity', ylabel='Sleep precision', title='Precision–recall tradeoff')
    axes[0, 1].legend(fontsize=8)
    for key, title, color in [('sensitivity', 'Sensitivity', COLORS[0]), ('specificity', 'Specificity', '#287c8e'), ('precision', 'Precision', COLORS[1])]:
        axes[1, 0].plot(fixed.speed_threshold_mm_s, fixed[key], 'o-', color=color, label=title)
    axes[1, 0].set(xscale='log', ylim=(0, 1), xlabel='Stationarity threshold (mm/s)', ylabel='Mean ant metric', title='Fixed thresholds: all original labeled frames')
    axes[1, 0].legend(fontsize=9)
    p = per_ant  # Already follows the canonical metadata order, including role order.
    xx = np.arange(len(p))
    axes[1, 1].bar(xx, p.balanced_accuracy, color='#287c8e')
    axes[1, 1].axhline(.5, color='#888', ls='--')
    axes[1, 1].set(ylim=(0, 1), xticks=xx, xticklabels=[str(c) for c in p.CupID], xlabel='Held-out cup (colony then role)',
                   ylabel='Balanced accuracy', title='One whole ant held out per fold')
    axes[1, 1].tick_params(axis='x', labelsize=7, rotation=90)
    fig.suptitle('Can speed recover the posture-defined sleep labels?', fontsize=19)
    fig.text(.025, .015, 'Curves weight ants equally and describe the fixed state dictionary. Black ROC point and per-ant bars use thresholds '
             'trained on other ants. Ambiguous time and motion-inferred exclusions are not validation targets.', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .96))
    save(fig, f'{page:02d}_proxy_validation', 'Speed-proxy discrimination and errors',
         f'Observed {sleep_name} versus observed {other_name} only. Descriptive equal-ant curves and leave-one-ant-out threshold validation.')
    page += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for cutoff, group in bouts.groupby('speed_threshold_mm_s'):
        axes[0].plot(group.minimum_bout_seconds + 1, group.balanced_accuracy, 'o-', label=f'≤{cutoff:g} mm/s')
    axes[0].set(xscale='log', ylim=(.4, 1), xlabel='Minimum stationary bout (s; axis uses duration + 1)',
                ylabel='Mean ant balanced accuracy', title='Does requiring sustained stationarity help?')
    axes[0].legend(fontsize=9, ncol=2)
    names = ['Speed threshold\nLOAO', 'Threshold + bout\nLOAO', 'Speed threshold\nfirst 6 h LOAO', 'Speed threshold\ntime holdout']
    entries = [summary['loao_speed']['balanced_accuracy'], summary['loao_bouts']['balanced_accuracy'],
               summary['first_6h_loao']['metrics']['balanced_accuracy'], summary['temporal_holdout']['metrics']['balanced_accuracy']]
    for i, entry in enumerate(entries):
        axes[1].errorbar(i, entry['mean'], yerr=[[entry['mean'] - entry['lower']], [entry['upper'] - entry['mean']]],
                        fmt='o', color='#287c8e', capsize=4, markersize=8)
        axes[1].text(i, entry['upper'] + .02, f"{100 * entry['mean']:.1f}%", ha='center', fontsize=10)
    axes[1].set(xticks=np.arange(4), xticklabels=names, ylim=(.45, 1.02), xlim=(-.5, 3.5), ylabel='Balanced accuracy',
                title='Held-out tests and sensitivity checks')
    axes[1].axhline(.5, color='#999', ls='--')
    axes[1].tick_params(axis='x', labelsize=9)
    fig.suptitle('Stationary bouts and robustness checks', fontsize=18)
    fig.text(.025, .015, 'Left: descriptive parameter sweep. Right: train-only choice of threshold/bout parameters. '
             'Missing speed breaks a bout; full qualifying bouts are credited retrospectively. Error bars: ant bootstrap 95% CI.', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    save(fig, f'{page:02d}_bout_sensitivity', 'Sustained stationarity and robustness',
         'Speed-duration sweep, held-out whole-ant validation, first-six-hour sensitivity, and even/odd 30-minute temporal validation.')
    page += 1

    # Show speed distributions separately for every original state to expose overlap.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    binspeed = SPEED_CUTS
    for state in range(n_states):
        data = hist[use, state]
        totals = data.sum(axis=1)
        valid = totals > 0
        if not valid.any():
            continue
        cdf = (data[valid] / totals[valid, None]).mean(axis=0).cumsum()
        axes[0].plot(binspeed, cdf, label=f'S{state}' + (' (putative sleep)' if state in sleep else ''))
    axes[0].set_xscale('symlog', linthresh=.01)
    axes[0].set(xlim=(0, 20), ylim=(0, 1), xlabel='Speed (mm/s)', ylabel='Cumulative fraction, equal ant weight',
                title='Individual original states overlap in speed')
    axes[0].legend(fontsize=10)
    q = fixed[['speed_threshold_mm_s', 'precision', 'sensitivity', 'specificity', 'stationary_time_without_observed_state_fraction']]
    cell = [[f'{r.iloc[0]:g}'] + [f'{100 * v:.1f}%' for v in r.iloc[1:]] for _, r in q.iterrows()]
    axes[1].axis('off')
    tab = axes[1].table(cellText=cell, colLabels=['Cutoff\nmm/s', 'Sleep\nprecision', 'Sleep\nsensitivity', 'Non-sleep\nspecificity', 'Quiet time\nno state label'],
                        loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.1, 2.1)
    axes[1].set_title('Threshold tradeoffs and the unresolved denominator')
    fig.suptitle('Stationary does not uniquely specify a posture state', fontsize=18)
    fig.text(.025, .02, 'Quiet time without original state labels is shown separately (it may have a motion-inferred exclusion). '
             'State-specific distributions use equal ant weights within each state; table values are means of ant metrics.', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    save(fig, f'{page:02d}_state_speed_overlap', 'Speed overlap across posture states',
         'Full standard-speed range shown; fixed stationarity thresholds expose the precision–sensitivity tradeoff and unresolved quiet time.')
    page += 1

    for start in range(0, len(methods), 5):
        fig = plt.figure(figsize=(12, 11))
        fig.text(.07, .94, 'Methods and interpretation', fontsize=23, weight='bold')
        yy = .87
        for text in methods[start:start + 5]:
            wrapped = fill(text, 112)
            fig.text(.07, yy, wrapped, fontsize=11.5, va='top', linespacing=1.5)
            yy -= .025 * (wrapped.count('\n') + 1) + .025
        save(fig, f'{page:02d}_methods_{start // 5 + 1}', f'Methods, part {start // 5 + 1}',
             'Source definitions, denominators, exclusion rules, validation design, and limitations.')
        page += 1
    pdf.close()
    write_json(output / 'figure_manifest.json', gallery)
    # Offline HTML embeds plots. The PDF and machine-readable tables remain alongside it.
    sections = []
    for item in gallery:
        data = base64.b64encode((figures / (item['name'] + '.png')).read_bytes()).decode()
        sections.append(f'<section id="{item["name"]}"><h2>{html.escape(item["title"])}</h2>'
                        f'<p>{html.escape(item["description"])}</p><img src="data:image/png;base64,{data}" '
                        f'alt="{html.escape(item["title"])}"></section>')
    links = ['sleep_speed_summary.pdf', 'per_ant_summary.csv', 'colony_role_summary.csv', 'proxy_summary.json', 'proxy_per_ant.csv',
             'stationarity_thresholds.csv', 'bout_proxy_per_ant.csv', 'bout_thresholds.csv', 'provenance.json']
    table_columns = ['Colony', 'Role', 'CupID', 'ExpectedID', 'IdentityStatus', 'SleepPercent', 'NotSleepPercent',
                     'AmbiguousPercent', 'MotionInferredNotPercent', 'MeanSpeed', 'ValidSpeedFraction', 'JointSpeedStatePercent']
    doc = ('<!doctype html><meta charset="utf-8"><title>Single-ant sleep and speed summary</title>'
           '<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1550px;margin:35px auto;padding:0 24px;color:#222}'
           'h1{font-size:34px}h2{margin-top:1.8em}img{max-width:100%;height:auto}section{border-top:1px solid #ddd;margin-top:35px}'
           'nav a{display:inline-block;margin:6px 18px 6px 0}table{border-collapse:collapse;font-size:12px}td,th{padding:6px 9px;border-bottom:1px solid #ddd}'
           '.tablewrap{overflow:auto}.finding{background:#f5f0fa;border-left:5px solid #8452b8;padding:18px}details{margin:20px 0}</style>'
           f'<h1>Single-ant putative sleep and tag-position speed</h1><p>{html.escape(dataset_label)} · colony → role → cup · {html.escape(speed_label)}</p>'
           f'<p class="finding">{html.escape(conclusion)}</p><nav>'
           + ''.join(f'<a href="{name}">{html.escape(name)}</a>' for name in links) + '</nav>'
           + '<details><summary>Methods, validation design, and limits</summary><ol>'
           + ''.join(f'<li>{html.escape(text)}</li>' for text in methods) + '</ol></details>'
           + '<details><summary>All ants: numerical table, in plot order</summary><div class="tablewrap">'
           + table[table_columns].to_html(index=False, float_format=lambda x: f'{x:.3f}', na_rep='missing', border=0)
           + '</div></details>' + ''.join(sections))
    (output / 'sleep_speed_summary.html').write_text(doc)
    (output / 'README.txt').write_text('Single-ant sleep and speed report\n\n' + conclusion + '\n\n'
        + '\n\n'.join(methods) + '\n\nReproduce from this code snapshot:\n'
        + 'cd code\npython -m analysis.single_ant_sleep_speed_report all --dataset DATASET --metadata METADATA_REPORT --output OUTPUT '
        + '--sleep-states ' + ' '.join(map(str, sleep)) + f" --bin-seconds {provenance['bin_seconds']}"
        + (' --registry REGISTRY.tsv' if provenance.get('registry_source') else '') + '\n')
    print(f'Rendered {len(gallery)} pages in {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['extract', 'analyse', 'render', 'all'])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speed-root', type=Path, help='Alternative single-position speed cache with definition.json and per_track folders')
    parser.add_argument('--sleep-states', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--state-root', type=Path, help='Original per-track observed state exports; defaults to analysis/sleep_s01_exclusion')
    parser.add_argument('--registry', type=Path, help='Documented cup/tag/colony/role TSV')
    parser.add_argument('--bin-seconds', type=float, default=600)
    args = parser.parse_args()
    if args.stage in ['extract', 'all']:
        extract(args.dataset, args.metadata, args.output, args.speed_root, args.sleep_states,
                args.state_root, args.registry, args.bin_seconds)
    if args.stage in ['analyse', 'all']:
        analyse(args.output)
    if args.stage in ['render', 'all']:
        render(args.output)


if __name__ == '__main__':
    main()
