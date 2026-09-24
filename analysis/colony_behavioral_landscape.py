"""A single-recording spatial landscape with temporal validation.

Uses verified half-hour counts, never a previously fitted multi-recording model.
See colony_behavioral_landscape.md for the estimands and interpretation limits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture


def stamp(path):
    path = Path(path)
    st = path.stat()
    return dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns)


def features(maps, observed=None):
    """Square-root probability, with a bin for observed positions outside arena.

    An empty profile is missing data and must never become an 'outside' state.
    Euclidean distances divided by sqrt(2) are Hellinger distances.
    """
    maps = np.asarray(maps, dtype=float)
    p = maps.reshape(len(maps), -1)
    mass = p.sum(axis=1)
    if observed is None:
        empty = mass <= 0
    else:
        observed = np.asarray(observed)
        if observed.shape != mass.shape or not np.isfinite(observed).all():
            raise ValueError('Observed counts must match the profile rows')
        empty = observed <= 0
    if not np.isfinite(p).all() or (p < 0).any() or empty.any() or (mass > 1 + 1e-5).any():
        raise ValueError("Expected nonempty, finite occupancy probabilities with mass <= 1")
    p = np.column_stack([p, np.maximum(0, 1-mass)])
    return np.sqrt(p / p.sum(axis=1, keepdims=True))


def interval_frames(calendar_bin, start, frame_count, fps=24):
    """Exact frame exposure in integer half-hour bins, including partial edges."""
    start_frame = int(pd.Timestamp(start).value // 10**9) * fps
    left = np.asarray(calendar_bin, dtype=np.int64) * 1800 * fps
    return np.maximum(0, np.minimum(left + 1800*fps, start_frame+frame_count) - np.maximum(left, start_frame))


def matched_days(calendar_bin, recorded_frames, fps=24):
    """First two complete, clock-matched 24h periods; never split cached bins."""
    full = np.flatnonzero(np.asarray(recorded_frames) == 1800*fps)
    if not len(full):
        raise ValueError("No complete half-hour bins")
    first = int(calendar_bin[full[0]])
    masks = [(calendar_bin >= first+d*48) & (calendar_bin < first+(d+1)*48) for d in range(2)]
    if any(m.sum() != 48 or not np.all(recorded_frames[m] == 1800*fps) for m in masks):
        raise ValueError("Temporal validation needs two complete consecutive 24h intervals")
    return masks


def align(reference, prediction, fit_mask=None):
    """Permutation matching uses only the specified fitting observations."""
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    k = max(reference.max(), prediction.max()) + 1
    mask = np.ones(len(reference), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    counts = np.zeros((k, k), int)
    np.add.at(counts, (reference[mask], prediction[mask]), 1)
    rows, cols = linear_sum_assignment(-counts)
    lookup = np.empty(k, int)
    lookup[cols] = rows
    return lookup[prediction]


def kmeans(x, k, seed=0, weights=None):
    return KMeans(k, n_init=30, random_state=seed).fit(x, sample_weight=weights)


def bootstrap_candidates(x, repeats, seed):
    """Resample whole ants; predict all ants, and align using in-bag ants only."""
    rng = np.random.default_rng(seed)
    rows, fits, agreements = [], {}, {}
    for k in range(2, 7):
        fit = kmeans(x, k, seed)
        labels = fit.labels_
        ari, numerator, denominator = [], np.zeros(len(x)), np.zeros(len(x))
        for b in range(repeats):
            counts = np.bincount(rng.integers(len(x), size=len(x)), minlength=len(x))
            pred = kmeans(x, k, seed+b+1, counts).predict(x)
            ari.append(adjusted_rand_score(labels, pred))
            matched = align(labels, pred, counts > 0)
            absent = counts == 0
            numerator += absent & (matched == labels)
            denominator += absent
        sizes = np.bincount(labels, minlength=k)
        rows.append(dict(k=k, silhouette=silhouette_score(x, labels), bootstrap_ari_median=np.median(ari),
                         bootstrap_ari_p10=np.quantile(ari, .1), bootstrap_ari_p90=np.quantile(ari, .9),
                         smallest_group=int(sizes.min()), sizes=json.dumps(sizes.tolist())))
        fits[k] = fit
        agreements[k] = (numerator / np.maximum(denominator, 1), denominator)
    return pd.DataFrame(rows), fits, agreements


def lag_statistics(scores, max_lag=24):
    """Equal-ant lag correlations; gaps stay at their actual time indices.

    Pooled uses the grand mean; centered removes each ant's mean. Each ant's
    lag product and variance receive equal weight, irrespective of coverage.
    """
    means = np.nanmean(scores, axis=1)
    grand = means.mean()
    rows = []
    for lag in range(1, max_lag+1):
        row = dict(lag_hours=lag*.5)
        for name, mean in [('pooled', np.full(len(scores), grand)), ('within_ant', means)]:
            a, b = scores[:, :-lag]-mean[:, None], scores[:, lag:]-mean[:, None]
            ok = np.isfinite(a) & np.isfinite(b)
            n = ok.sum(axis=1)
            valid = n >= 4
            numerator = np.divide(np.where(ok, a*b, 0).sum(axis=1), n, out=np.full(len(n), np.nan), where=n > 0)
            va = np.divide(np.where(ok, a*a, 0).sum(axis=1), n, out=np.full(len(n), np.nan), where=n > 0)
            vb = np.divide(np.where(ok, b*b, 0).sum(axis=1), n, out=np.full(len(n), np.nan), where=n > 0)
            denom = np.sqrt(np.mean(va[valid])*np.mean(vb[valid])) if valid.any() else 0
            row[name] = float(np.mean(numerator[valid]) / denom) if denom > 0 else np.nan
            row['n_ants'] = int(valid.sum())
            row['n_pairs'] = int(n[valid].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def behavioral_means(bins):
    metrics = {'colony_percent':'position_coverage', 'mean_speed_mm_s':'speed_coverage',
               'sleep_percent':'sleep_coverage', 'food_percent':'resource_coverage'}
    out = []
    for ant, part in bins.groupby('ant', sort=True):
        row = dict(ant=ant)
        for metric, coverage in metrics.items():
            w = part.n_expected_frames.to_numpy() * part[coverage].to_numpy()
            good = np.isfinite(part[metric]) & (w > 0)
            row[metric] = np.average(part.loc[good, metric], weights=w[good]) if good.any() else np.nan
            row[coverage+'_hours'] = w[good].sum()/24/3600
        out.append(row)
    return pd.DataFrame(out)


def load_recording(block, cache, fine_name, output):
    """Validate provenance and load only the explicitly requested source block."""
    manifest_path = cache/'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    indices = [i for i, w in enumerate(manifest['windows']) if w['block'] == str(block)]
    if len(indices) != 1:
        raise ValueError("Exactly one cache source must match the requested block")
    index = indices[0]
    info = manifest['sources'][index]['info']
    duration_frames = info['frame_stop'] - info['frame_start']
    if info['frame_start'] != 0 or info['fps'] != 24:
        raise ValueError("This analysis currently requires a complete 24 fps block starting at frame 0")
    provenance = [stamp(manifest_path), stamp(block/'panorama_regions.csv')]
    # Reject stale behavior caches, remapping their original flash staging prefix.
    for saved in info['sources']:
        saved_path = Path(saved['path'])
        original_root = block if saved_path.is_relative_to(block) else Path(info['block'])
        suffix = saved_path.relative_to(original_root)
        actual = stamp(block/suffix)
        if (actual['size'], actual['mtime_ns']) != (saved['size'], saved['mtime_ns']):
            raise ValueError(f"Behavior source changed: {actual['path']}")
        provenance.append(actual)
    fine_root = block/'stitched'/fine_name
    labels = pd.read_csv(fine_root/'track_cluster_ids.csv').set_index('track_name')
    coarse_labels = pd.read_csv(block/'stitched/grid_occupancy_histograms_arena/track_cluster_ids.csv').set_index('track_name')
    provenance += [stamp(fine_root/'track_cluster_ids.csv'), stamp(block/'stitched/grid_occupancy_histograms_arena/track_cluster_ids.csv')]
    bins_path = cache/'task_bins.parquet'
    bins = pd.read_parquet(bins_path)
    bins = bins[bins.source_block.eq(str(block)) & bins.in_recording_bin].copy()
    if bins.empty or bins.duplicated(['ant','timestamp']).any():
        raise ValueError("Missing or duplicated within-recording behavior bins")
    bins['calendar_bin'] = bins.timestamp.astype('datetime64[ns]').astype('int64')//(1800*10**9)
    # Old label columns can refer to a different fit; discard before any analysis.
    bins = bins.drop(columns=['cluster_id'], errors='ignore')
    bins.to_parquet(output/'behavior_bins.parquet', index=False)
    provenance.append(stamp(bins_path))
    means = behavioral_means(bins).set_index('ant')
    inventory, data = [], {}
    atom_root = cache/'temporal_clusters_4h/atoms'/str(index)
    for path in sorted(atom_root.glob('*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        entry = meta['signature']['entry']
        if entry['block'] != str(block) or entry['block_index'] != index or not meta['original_histogram_exact']:
            raise ValueError(f"Wrong recording or unverified atoms: {path}")
        for saved in meta['signature']['inputs']:
            actual = stamp(saved['path'])
            if (actual['size'], actual['mtime_ns']) != (saved['size'], saved['mtime_ns']):
                raise ValueError(f"Stale atom input: {actual['path']}")
        ant, name = entry['ant'], entry['track_name']
        side = ant.split(':')[0]
        with np.load(path) as z:
            counts = z['counts'].astype(np.int64)
            detected = z['detected'].copy()
            calendar_bin = z['calendar_bin'].copy()
            edges = (z['x_edges'].copy(), z['y_edges'].copy())
        if counts.sum() > detected.sum() or (counts.sum(axis=(1,2)) > detected).any():
            raise ValueError("Spatial counts exceed detected frames")
        recorded = interval_frames(calendar_bin, entry['start'], duration_frames)
        if recorded.sum() != duration_frames or (detected > recorded).any():
            raise ValueError("Incorrect time exposure or duplicated frames")
        coarse = counts.sum(axis=0)/max(1, detected.sum())
        root = fine_root/'per_track'/Path(name).stem
        fm = json.loads((root/'grid_occupancy_metadata.json').read_text())
        st = (block/'stitched/per_track'/name).stat()
        source = fm['arena_cache_inputs']
        if (st.st_size, st.st_mtime_ns) != (source['source_size'], source['source_mtime_ns']):
            raise ValueError("Fine grid source changed")
        fine = np.load(root/'grid_occupancy_f4.npy')
        original = np.load(block/'stitched/grid_occupancy_histograms_arena/per_track'/Path(name).stem/'grid_occupancy_f4.npy')
        np.testing.assert_array_equal(coarse.astype(np.float32), original)
        if detected.sum() != fm['n_detected_frames']:
            raise ValueError("Fine and coarse grids use different position samples")
        sm_path = block/'stitched/speed_vectors/per_track'/Path(name).stem/'speed_metadata.json'
        sm = json.loads(sm_path.read_text())
        selected = sm['n_observed_frames']/sm['n_frames'] > .4
        if selected != (name in labels.index):
            raise ValueError("Saved reference cohort differs from the >40% detection rule")
        row = dict(ant=ant, side=side, track_id=int(ant.split(':')[1]), track_name=name, selected=selected,
                   speed_detection=sm['n_observed_frames']/sm['n_frames'], position_detection=detected.sum()/duration_frames,
                   old_fine_label=str(labels.loc[name,'cluster_id']) if selected else '',
                   old_coarse_label=str(coarse_labels.loc[name,'cluster_id']) if name in coarse_labels.index else '')
        if ant in means.index:
            row.update(means.loc[ant].to_dict())
        inventory.append(row)
        data[ant] = dict(counts=counts, detected=detected, calendar_bin=calendar_bin, recorded=recorded,
                         coarse=coarse, fine=fine, edges=edges, fine_edges=(np.load(root/'grid_x_edges_mm.npy'), np.load(root/'grid_y_edges_mm.npy')),
                         metadata=fm)
        provenance.extend([stamp(path), stamp(path.with_suffix('.json')), stamp(root/'grid_occupancy_f4.npy'), stamp(root/'grid_occupancy_metadata.json'), stamp(sm_path)])
    table = pd.DataFrame(inventory).sort_values(['side','track_id']).reset_index(drop=True)
    if len(table) != len(list((block/'stitched/per_track').glob('*.parquet'))):
        raise ValueError("Atom cache does not cover all finished tracks")
    table.to_csv(output/'inventory.csv', index=False)
    return table, data, bins, info, provenance


def analyze_side(side, inventory, data, bins, repeats, seed, output):
    table = inventory[inventory.side.eq(side) & inventory.selected].copy().reset_index(drop=True)
    ants = table.ant.to_list()
    ds = [data[a] for a in ants]
    maps = np.stack([d['coarse'] for d in ds])
    fine = np.stack([d['fine'] for d in ds])
    x, xf = features(maps), features(fine)
    pca, pca_fine = PCA(svd_solver='full').fit(x), PCA(svd_solver='full').fit(xf)
    z, zf = pca.transform(x), pca_fine.transform(xf)
    # Sign is arbitrary; orient toward more time outside the nest, after fitting.
    if spearmanr(z[:,0], table.colony_percent).statistic > 0:
        pca.components_[0] *= -1
        z[:,0] *= -1
    if spearmanr(zf[:,0], table.colony_percent).statistic > 0:
        pca_fine.components_[0] *= -1
        zf[:,0] *= -1
    candidates, fits, agreements = bootstrap_candidates(z, repeats, seed)
    raw = fits[2].labels_
    reorder = np.argsort([-np.nanmedian(table.colony_percent[raw == k]) for k in range(2)])
    labels = np.zeros(len(raw), int)
    for new, old in enumerate(reorder):
        labels[raw == old] = new
    centers = np.stack([x[labels == k].mean(axis=0) for k in range(2)])
    table['group'] = labels
    table['pc1'], table['pc2'] = z[:,0], z[:,1]
    table['fine_pc1'], table['fine_pc2'] = zf[:,0], zf[:,1]
    table['oob_agreement'], table['oob_trials'] = agreements[2]
    dwhole = cdist(x, centers)
    table['centroid_margin'] = abs(dwhole[:,0]-dwhole[:,1])/dwhole.max(axis=1)
    table['entropy_bits'] = -np.sum(np.square(x)*np.log2(np.maximum(np.square(x), 1e-30)), axis=1)
    fine_model = kmeans(zf, 2, seed)
    smooth_rows = []
    for sigma in (0., 1., 2.):
        smooth = gaussian_filter(maps, sigma=(0,sigma,sigma)) if sigma else maps
        ys = kmeans(PCA(svd_solver='full').fit_transform(features(smooth)), 2, seed).labels_
        smooth_rows.append(dict(side=side, comparison=f'1mm grid, smoothing sigma {sigma:g}mm', ari=adjusted_rand_score(labels,ys)))
    smooth_rows += [dict(side=side, comparison='0.25mm KMeans vs 1mm KMeans', ari=adjusted_rand_score(labels,fine_model.labels_)),
                   dict(side=side, comparison='saved 0.25mm Leiden vs 1mm KMeans', ari=adjusted_rand_score(labels,table.old_fine_label)),
                   dict(side=side, comparison='saved 1mm Leiden vs saved 0.25mm Leiden', ari=adjusted_rand_score(table.old_coarse_label,table.old_fine_label))]
    for cutoff in (.6,.8):
        good = table.speed_detection.to_numpy() > cutoff
        if good.sum() >= 8:
            yy = kmeans(z[good], 2, seed).labels_
            smooth_rows.append(dict(side=side, comparison=f'detection >{cutoff:.0%}',ari=adjusted_rand_score(labels[good],yy),
                                    n_ants=int(good.sum()),group0=int(((labels==0)&good).sum()),group1=int(((labels==1)&good).sum())))
    # Temporal validation: these models see day 1 only; day 2 never tunes them.
    cal, recorded = ds[0]['calendar_bin'], ds[0]['recorded']
    day_masks = matched_days(cal, recorded)
    day_features, day_coverage = [], []
    for mask in day_masks:
        count = np.stack([d['counts'][mask].sum(axis=0) for d in ds])
        observed = np.array([d['detected'][mask].sum() for d in ds])
        day_coverage.append(observed/recorded[mask].sum())
        day_features.append(features(count/np.maximum(observed[:,None,None],1),observed=observed))
    paired = (day_coverage[0] > .4) & (day_coverage[1] > .4)
    train, test = day_features[0][paired], day_features[1][paired]
    temporal_pca = PCA(svd_solver='full').fit(train)
    temporal_scores = temporal_pca.transform(train)
    test_scores = temporal_pca.transform(test)
    baseline_error = np.square(test-train.mean(axis=0)).sum(axis=1)
    validation, fitted_days = [], {}
    for k in range(1,7):
        fit = kmeans(temporal_scores, k, seed)
        # Cluster centers are returned to the full space before measuring loss.
        prototype = temporal_pca.inverse_transform(fit.cluster_centers_)
        dist = cdist(test, prototype)
        pred = dist.argmin(axis=1)
        day2fit = kmeans(test, k, seed)
        error = np.square(dist.min(axis=1))
        validation.append(dict(side=side, model=f'K={k}', complexity=k, n_ants=int(paired.sum()),
                               heldout_mse=error.mean(), explained_vs_train_mean=1-error.mean()/baseline_error.mean(),
                               day1_day2_partition_ari=adjusted_rand_score(fit.labels_, day2fit.labels_),
                               same_ant_assignment_fraction=float((fit.labels_ == pred).mean())))
        fitted_days[k] = (fit, prototype, pred, day2fit)
    for rank in (1,2,3,5,10):
        low = np.zeros_like(test_scores); low[:,:rank] = test_scores[:,:rank]
        reconstructed = temporal_pca.inverse_transform(low)
        error = np.square(test-reconstructed).sum(axis=1)
        validation.append(dict(side=side, model=f'PC{rank}', complexity=rank, n_ants=int(paired.sum()),
                               heldout_mse=error.mean(), explained_vs_train_mean=1-error.mean()/baseline_error.mean()))
    fit, prototype, pred, day2fit = fitted_days[2]
    day_sign = -1 if spearmanr(temporal_scores[:,0], table.loc[paired,'colony_percent']).statistic > 0 else 1
    paired_table = table.loc[paired,['ant']].copy()
    paired_table['day1_pc1'] = temporal_scores[:,0]*day_sign
    paired_table['day2_pc1'] = test_scores[:,0]*day_sign
    paired_table['day1_group'] = fit.labels_
    paired_table['day2_assigned_group'] = pred
    paired_table.to_csv(output/f'{side}_paired_days.csv',index=False)
    # Evaluate a deliberately simple continuous density alternative on PC1.
    gmm = [GaussianMixture(k, n_init=30, random_state=seed, reg_covar=1e-5).fit(z[:,:1]) for k in (1,2)]
    delta_bic = gmm[0].bic(z[:,:1])-gmm[1].bic(z[:,:1])
    # A matched covariance Gaussian null tests whether K=2 silhouette is special.
    # It is a geometric reference, not a biological null or calibrated p-value.
    rng = np.random.default_rng(seed+991)
    null_sil = []
    for _ in range(repeats):
        null = rng.normal(size=z.shape)*np.sqrt(pca.explained_variance_)
        y = kmeans(null,2,seed).labels_
        null_sil.append(silhouette_score(null,y))
    # Half-hour trajectories in the same 0724-only PCA basis.
    scores = np.full((len(ants),len(cal)),np.nan)
    time_rows = []
    for i, (ant,d) in enumerate(zip(ants,ds)):
        cov = d['detected']/np.maximum(recorded,1)
        nonempty = d['detected'] > 0
        tx = features(d['counts'][nonempty]/d['detected'][nonempty,None,None],observed=d['detected'][nonempty])
        tz = pca.transform(tx)
        distance = cdist(tx,centers)
        margin = abs(distance[:,0]-distance[:,1])/distance.max(axis=1)
        eligible = (cov[nonempty] >= .4) & (recorded[nonempty] >= .95*43200)
        scores[i,np.flatnonzero(nonempty)[eligible]] = tz[eligible,0]
        for j,t in enumerate(np.flatnonzero(nonempty)):
            time_rows.append(dict(ant=ant,side=side,calendar_bin=int(cal[t]),pc1=tz[j,0],pc2=tz[j,1],
                                  coverage=cov[t],recording_fraction=recorded[t]/43200,qualified=bool(eligible[j]),
                                  group=int(distance[j].argmin()),margin=margin[j]))
    time = pd.DataFrame(time_rows).merge(bins.drop(columns=['side','track_id','track_name'],errors='ignore'),on=['ant','calendar_bin'],how='left',validate='one_to_one')
    time['elapsed_hours'] = (time.calendar_bin-cal[0])*.5
    time.to_parquet(output/f'{side}_trajectories.parquet',index=False)
    lag = lag_statistics(scores)
    nulls = []
    for _ in range(repeats):
        shuffled = scores.copy()
        for row in shuffled:
            good = np.flatnonzero(np.isfinite(row)); row[good] = rng.permutation(row[good])
        nulls.append(lag_statistics(shuffled)[['pooled','within_ant']].to_numpy())
    nulls = np.stack(nulls)
    for j,key in enumerate(('pooled','within_ant')):
        lag[key+'_shuffle_low'],lag[key+'_shuffle_high'] = np.quantile(nulls[:,:,j],[.025,.975],axis=0)
    within = np.nanvar(scores,axis=1).mean()
    between = np.var(np.nanmean(scores,axis=1))
    events=[]
    for ant, part in time.groupby('ant'):
        p=part.set_index('calendar_bin').reindex(cal)
        valid=p.qualified.eq(True).to_numpy(bool) & p.margin.fillna(0).ge(.1).to_numpy()
        state=p.group.to_numpy()
        for t in range(3,len(cal)-2):
            if valid[t-3:t+3].all() and len(set(state[t-3:t]))==1 and len(set(state[t:t+3]))==1 and state[t-1]!=state[t]:
                events.append(dict(side=side,ant=ant,calendar_bin=int(cal[t]),from_group=int(state[t-1]),to_group=int(state[t]),
                                   first_new_bin=str(pd.Timestamp(int(cal[t])*1800*10**9))))
    summary=dict(side=side,n_ants=len(ants),all_tracks=int(inventory.side.eq(side).sum()),
                 groups=np.bincount(labels).tolist(),saved_fine_groups=table.old_fine_label.value_counts().sort_index().to_dict(),
                 saved_coarse_groups=table.old_coarse_label.value_counts().sort_index().to_dict(),
                 pc1_variance=float(pca.explained_variance_ratio_[0]),pc2_cumulative=float(pca.explained_variance_ratio_[:2].sum()),
                 dims90=int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),.9)+1),
                 fine_pc1_variance=float(pca_fine.explained_variance_ratio_[0]),
                 pc1_two_gaussian_delta_bic=float(delta_bic),silhouette_gaussian_null_median=float(np.median(null_sil)),
                 silhouette_gaussian_null_95=np.quantile(null_sil,[.025,.975]).tolist(),
                 temporal_validation_n=int(paired.sum()),day1_start=str(pd.Timestamp(int(cal[day_masks[0]][0])*1800*10**9)),
                 day2_start=str(pd.Timestamp(int(cal[day_masks[1]][0])*1800*10**9)),
                 day_pc1_spearman=float(spearmanr(paired_table.day1_pc1,paired_table.day2_pc1).statistic),
                 within_ant_pc1_variance_fraction=float(within/(within+between)),
                 sustained_reassignments=len(events),sustained_reassignment_ants=len(set(e['ant'] for e in events)),
                 qualified_halfhours=int(np.isfinite(scores).sum()),total_halfhours=int(scores.size))
    summary['qualified_margin_halfhours'] = int((time.qualified & time.margin.ge(.1)).sum())
    behavior=[]
    for metric in ('colony_percent','mean_speed_mm_s','sleep_percent','food_percent','speed_detection'):
        values=[table.loc[table.group.eq(k),metric].dropna().to_numpy() for k in range(2)]
        effects=np.array([np.median(rng.choice(values[1],len(values[1]),replace=True))-np.median(rng.choice(values[0],len(values[0]),replace=True)) for _ in range(1000)])
        behavior.append(dict(side=side,metric=metric,g0_median=np.median(values[0]),g1_median=np.median(values[1]),
                             difference_g1_minus_g0=np.median(values[1])-np.median(values[0]),
                             bootstrap_low=np.quantile(effects,.025),bootstrap_high=np.quantile(effects,.975),
                             pc1_spearman=spearmanr(table.pc1,table[metric],nan_policy='omit').statistic))
    table.to_csv(output/f'{side}_individuals.csv',index=False)
    candidates.assign(side=side).to_csv(output/f'{side}_model_selection.csv',index=False)
    pd.DataFrame(validation).to_csv(output/f'{side}_temporal_validation.csv',index=False)
    pd.DataFrame(smooth_rows).to_csv(output/f'{side}_sensitivity.csv',index=False)
    pd.DataFrame(behavior).to_csv(output/f'{side}_behavior.csv',index=False)
    lag.to_csv(output/f'{side}_lag_correlations.csv',index=False)
    pd.DataFrame(events,columns=['side','ant','calendar_bin','from_group','to_group','first_new_bin']).to_csv(output/f'{side}_sustained_reassignments.csv',index=False)
    np.savez_compressed(output/f'{side}_model.npz',ants=np.array(ants),pca_mean=pca.mean_,pca_components=pca.components_,
                        variance_ratio=pca.explained_variance_ratio_,centers=centers,coarse_maps=maps,fine_scores=zf,
                        x_edges=ds[0]['edges'][0],y_edges=ds[0]['edges'][1],half_hour_pc1=scores,
                        fine_pca_components=pca_fine.components_[:2],fine_variance_ratio=pca_fine.explained_variance_ratio_)
    print('SIDE_RESULT',json.dumps(summary),flush=True)
    return dict(summary=summary,table=table,maps=maps,fine=fine,pca=pca,pca_fine=pca_fine,z=z,zf=zf,
                distance=squareform(pdist(x))/np.sqrt(2),candidates=candidates,validation=pd.DataFrame(validation),
                sensitivity=pd.DataFrame(smooth_rows),behavior=pd.DataFrame(behavior),scores=scores,time=time,lag=lag,
                gmm=gmm,paired=paired_table,edges=ds[0]['edges'],fine_edges=ds[0]['fine_edges'],metadata=ds[0]['metadata'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fine-grid',default='grid_occupancy_histograms_arena_0p25mm')
    parser.add_argument('--bootstrap',type=int,default=100)
    parser.add_argument('--seed',type=int,default=24)
    args=parser.parse_args()
    if args.block.parent.name != '20260724' or args.block.name != 'block01':
        raise ValueError('This report is scoped to 20260724/block01; use an explicit path to that recording')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'COMPLETE.json').unlink(missing_ok=True)
    print('VALIDATING_SINGLE_RECORDING_INPUTS',str(args.block),flush=True)
    if args.bootstrap < 20:
        raise ValueError('Use at least 20 bootstrap replicates')
    inventory,data,bins,info,provenance=load_recording(args.block,args.cache,args.fine_grid,args.output)
    print('INPUTS_VALIDATED',len(inventory),'tracks;',int(inventory.selected.sum()),'selected ants',flush=True)
    results={side:analyze_side(side,inventory,data,bins,args.bootstrap,args.seed,args.output) for side in ('left','right')}
    from analysis.colony_behavioral_landscape_plots import render
    render(results,inventory,data,info,args.output)
    manifest=dict(block=str(args.block),cache=str(args.cache),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  summary={k:v['summary'] for k,v in results.items()},recording_info=info,input_fingerprints=provenance,
                  software={k:importlib.metadata.version(k) for k in ('numpy','pandas','scipy','scikit-learn','matplotlib')},
                  code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('colony_behavioral_landscape_plots.py'),Path(__file__).with_name('colony_behavioral_landscape_explorer.html'))})
    # Detect changed inputs rather than silently publish a mixture of cache states.
    if any(stamp(p['path']) != p for p in provenance):
        raise ValueError('Input changed during analysis')
    (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(complete=True,block=str(args.block)))+'\n')


if __name__=='__main__':
    main()
