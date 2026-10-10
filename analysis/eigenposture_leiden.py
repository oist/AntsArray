"""Leiden behavior states, feature ablation, and ant roles from state proportions.

The graph lives in the balanced feature space. UMAP layouts and spatial labels
never choose state assignments. All search decisions and rejected fits are saved.
"""
import argparse
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, silhouette_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set

from analysis.eigenposture_leiden_features import BINS, FAMILIES, N_FEATURES, balanced_matrix

SEED = 7241010
RESOLUTIONS = (.1, .25, .5, 1.)
GRAPH_NEIGHBORS = 30
MAX_GRAPH_BINS = 20000
CORE = np.array([0, 1, 8, 9, 10, 11, 40])
RETAIN = np.array([0, 1, 43])  # explicit body and antennal unsigned motion for sleep interpretation
PROTOCOL = dict(bin_minutes=BINS, broad_features=N_FEATURES, graph_neighbors=GRAPH_NEIGHBORS,
    graph_max_bins=MAX_GRAPH_BINS, dictionary_fraction=.8, resolutions=RESOLUTIONS,
    leiden_iterations=3, seed=SEED, seed_repeats=3,
    state_rule='Finest tested resolution passing seed ARI >=0.9, ant-subsample ARI >=0.8, smallest fraction >=0.02, >=10 ants/state, >=3 ants/colony/state',
    primary_rule='Most qualifying states, then ant-subsample ARI, then seed ARI; no role/spatial scores',
    fallback_rule='If no partition qualifies, use highest ant-subsample then seed stability; report provisional, not supported',
    reduction_rule='Same K, ARI >=0.9, minimum matched-state Jaccard >=0.75 against broad dictionary, seed ARI >=0.9; choose smallest tested prefix then greedy deletion',
    role_rule='Existing proportion-only tied-covariance GMM K=1..4; BIC, >=5 ants, median bootstrap ARI >=0.8 and temporal ARI >=0.6; within 2 BIC choose smaller K',
    waveform_rule='Within-clip only: omega0=3, 2/3/4 Hz, no interpolation or cross-clip wavelets',
    retained_physical_features=['forward_peak', 'lateral_peak', 'antenna_unsigned_mean'],
    sleep_rule='Post hoc comparison only; sleep labels never select features, resolution or state partition')


def graph_from_features(x, seed=SEED):
    random = np.random.RandomState(seed)
    indices, distances, search = nearest_neighbors(x, GRAPH_NEIGHBORS, 'euclidean', {},
        False, random, n_jobs=1, verbose=False)
    adjacency, _, _ = fuzzy_simplicial_set(x, GRAPH_NEIGHBORS, random, 'euclidean',
        knn_indices=indices, knn_dists=distances)
    upper = sparse.triu(adjacency, k=1).tocoo()
    graph = ig.Graph(n=len(x), edges=np.column_stack([upper.row, upper.col]))
    graph.es['weight'] = upper.data
    return graph, search, adjacency.tocsr()


def partition(graph, resolution, seed=SEED):
    fit = la.find_partition(graph, la.RBConfigurationVertexPartition, weights='weight',
                            resolution_parameter=resolution, n_iterations=3, seed=seed)
    return np.asarray(fit.membership, dtype=int)


def overlap_scores(reference, labels):
    ka, kb = int(reference.max() + 1), int(labels.max() + 1)
    counts = np.zeros((ka, kb), dtype=int)
    np.add.at(counts, (reference, labels), 1)
    union = counts.sum(axis=1)[:, None] + counts.sum(axis=0)[None, :] - counts
    jaccard = np.divide(counts, union, out=np.zeros(counts.shape), where=union > 0)
    a, b = linear_sum_assignment(-jaccard)
    return adjusted_rand_score(reference, labels), float(jaccard[a, b].min()) if ka == kb else 0.


def state_proportions(labels, k):
    counts = np.stack([(labels == state).sum(axis=1) for state in range(k)], axis=1)
    total = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, total, out=np.full(counts.shape, np.nan), where=total > 0)


@dataclass
class Dataset:
    minutes: int
    values: np.ndarray
    counts: np.ndarray
    names: np.ndarray
    valid: np.ndarray
    raw: np.ndarray
    ant_index: np.ndarray
    time_index: np.ndarray
    dictionary: np.ndarray


def load_dataset(path):
    with np.load(path) as z:
        values, counts, names, minutes = z['values'], z['counts'], z['names'], int(z['minutes'])
    valid = np.isfinite(values).all(axis=2)
    ant_index, time_index = np.where(valid)
    raw = values[valid]
    dictionary = np.sort(np.random.default_rng(SEED).choice(len(raw), min(MAX_GRAPH_BINS, int(.8 * len(raw))), replace=False))
    return Dataset(minutes, values, counts, names, valid, raw, ant_index, time_index, dictionary)


def sweep_dataset(data, ants, output):
    x, center, scale = balanced_matrix(data.raw, np.arange(N_FEATURES))
    nodes = data.dictionary
    graph, search, adjacency = graph_from_features(x[nodes])
    rng = np.random.default_rng(SEED + 11)
    retained = np.concatenate([rng.choice(ix, max(1, int(.8 * len(ix))), replace=False)
        for side in ('left', 'right')
        if len(ix := np.unique(data.ant_index[nodes][ants.side.to_numpy()[data.ant_index[nodes]] == side]))])
    subset = np.flatnonzero(np.isin(data.ant_index[nodes], retained))
    small_graph, _, _ = graph_from_features(x[nodes[subset]])
    fits, rows = {}, []
    silhouette_sample = np.random.default_rng(SEED).choice(len(nodes), min(2000, len(nodes)), replace=False)
    for resolution in RESOLUTIONS:
        labels = [partition(graph, resolution, SEED + i) for i in range(3)]
        seed_ari = np.median([adjusted_rand_score(a, b) for a, b in combinations(labels, 2)])
        reference = labels[0]
        sub = partition(small_graph, resolution)
        sub_ari = adjusted_rand_score(reference[subset], sub)
        k = reference.max() + 1
        sizes = np.bincount(reference)
        ant_counts, side_counts = [], []
        for state in range(k):
            ids = np.unique(data.ant_index[nodes][reference == state])
            ant_counts.append(len(ids))
            side_counts.append(min((ants.side.iloc[ids] == side).sum() for side in ('left', 'right')))
        silhouette = silhouette_score(x[nodes[silhouette_sample]], reference[silhouette_sample]) if 1 < k < len(silhouette_sample) else np.nan
        qualifies = k >= 2 and seed_ari >= .9 and sub_ari >= .8 and sizes.min() / len(nodes) >= .02 and min(ant_counts) >= 10 and min(side_counts) >= 3
        row = dict(minutes=data.minutes, resolution=resolution, k=int(k), seed_ari=seed_ari,
                   ant_subsample_ari=sub_ari, smallest_fraction=sizes.min() / len(nodes),
                   minimum_ants=min(ant_counts), minimum_ants_per_colony=min(side_counts),
                   silhouette=silhouette, qualifies=qualifies, n_bins=len(data.raw), dictionary_bins=len(nodes))
        rows.append(row); fits[resolution] = reference
        print('STATE', row, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / f'state_resolution_{data.minutes:02d}min.csv', index=False)
    candidates = table[table.qualifies]
    supported = not candidates.empty
    if candidates.empty:
        candidates = table[table.k >= 2]
        if candidates.empty:
            candidates = table
    order = ['k', 'ant_subsample_ari', 'seed_ari'] if supported else ['ant_subsample_ari', 'seed_ari', 'k']
    chosen = candidates.sort_values(order, ascending=False).iloc[0]
    np.savez_compressed(output / f'broad_{data.minutes:02d}min.npz', dictionary=nodes,
        labels=fits[chosen.resolution], columns=np.arange(N_FEATURES), center=center, scale=scale,
        resolution=chosen.resolution, supported=supported)
    return table, chosen.to_dict() | {'supported': supported}


def evaluate_subset(data, columns, resolution, reference):
    columns = np.sort(columns)
    x, center, scale = balanced_matrix(data.raw, columns)
    graph, search, adjacency = graph_from_features(x[data.dictionary])
    labels = partition(graph, resolution)
    ari, jaccard = overlap_scores(reference, labels)
    k = int(labels.max() + 1)
    preserves = k == int(reference.max() + 1) and ari >= .9 and jaccard >= .75
    seed_ari = np.nan
    if preserves:
        repeats = [labels] + [partition(graph, resolution, SEED + i) for i in (1, 2)]
        seed_ari = float(np.median([adjusted_rand_score(a, b) for a, b in combinations(repeats, 2)]))
        preserves = seed_ari >= .9
    row = dict(n_features=len(columns), columns=json.dumps(columns.tolist()), k=k, seed_ari=seed_ari,
               ari_to_broad=ari, minimum_state_jaccard=jaccard, preserves=preserves)
    return row, dict(x=x, center=center, scale=scale, graph=graph, search=search,
                     adjacency=adjacency, labels=labels, columns=columns)


def reduce_features(data, ants, broad, output):
    reference, resolution = broad['labels'], float(broad['resolution'])
    x, _, _ = balanced_matrix(data.raw, np.arange(N_FEATURES))
    nodes = data.dictionary
    rng = np.random.default_rng(SEED)
    ids = np.unique(data.ant_index[nodes])
    train_ants = rng.choice(ids, int(.7 * len(ids)), replace=False)
    train = np.flatnonzero(np.isin(data.ant_index[nodes], train_ants))
    test = np.flatnonzero(~np.isin(data.ant_index[nodes], train_ants))
    test = rng.choice(test, min(3000, len(test)), replace=False)
    model = ExtraTreesClassifier(n_estimators=150, min_samples_leaf=3,
                                 class_weight='balanced', random_state=SEED, n_jobs=1)
    model.fit(x[nodes[train]], reference[train])
    accuracy = balanced_accuracy_score(reference[test], model.predict(x[nodes[test]]))
    importance = permutation_importance(model, x[nodes[test]], reference[test],
        scoring='balanced_accuracy', n_repeats=3, random_state=SEED, n_jobs=1)
    ranking = np.argsort(-importance.importances_mean, kind='stable')
    table = pd.DataFrame(dict(feature=data.names, family=FAMILIES,
        permutation_importance=importance.importances_mean, permutation_sd=importance.importances_std))
    table['surrogate_balanced_accuracy'] = accuracy
    table.to_csv(output / 'feature_importance.csv', index=False)
    (output / 'surrogate_notes.json').write_text(json.dumps(dict(balanced_accuracy=float(accuracy), training_ants=len(train_ants), testing_ants=len(ids)-len(train_ants), role='Feature importance explains broad state labels; not causal importance or independent sleep validation'), indent=2))
    rows, kept = [], {}
    # Explicit leave-one-family-out tests establish which broad blocks matter.
    for family in sorted(set(FAMILIES)):
        columns = np.flatnonzero(np.asarray(FAMILIES) != family)
        row, result = evaluate_subset(data, columns, resolution, reference)
        row.update(kind='family_ablation', name=f'without_{family}')
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / 'feature_reduction.csv', index=False)
        print('ABLATION', row, flush=True)
    candidates = sorted(set([4, 6, 8, 12, 16, 24, 32, N_FEATURES]))
    chosen_columns = np.arange(N_FEATURES)
    for number in candidates:
        row, result = evaluate_subset(data, np.union1d(RETAIN, ranking[:number]), resolution, reference)
        row.update(kind='ranked_prefix', name=f'top_{number}_plus_physical')
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / 'feature_reduction.csv', index=False)
        print('REDUCTION', row, flush=True)
        if row['preserves'] or number == N_FEATURES:
            chosen_columns = result['columns']
            kept['chosen'] = result
            break
    # One greedy deletion pass, least important first, with full Leiden refits.
    for column in ranking[::-1]:
        if column in RETAIN or column not in chosen_columns or len(chosen_columns) <= len(RETAIN):
            continue
        remaining = chosen_columns[chosen_columns != column]
        row, result = evaluate_subset(data, remaining, resolution, reference)
        row.update(kind='greedy', name=f'drop_{data.names[column]}')
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / 'feature_reduction.csv', index=False)
        print('PRUNE', row, flush=True)
        if row['preserves']:
            chosen_columns = remaining
            kept['chosen'] = result
    selected = kept['chosen']
    repeats = [selected['labels']] + [partition(selected['graph'], resolution, SEED + i) for i in (1, 2)]
    seed_ari = float(np.median([adjusted_rand_score(a, b) for a, b in combinations(repeats, 2)]))
    pd.DataFrame(rows).to_csv(output / 'feature_reduction.csv', index=False)
    np.savez_compressed(output / 'minimal_dictionary.npz', dictionary=nodes,
        labels=selected['labels'], columns=chosen_columns, center=selected['center'],
        scale=selected['scale'], resolution=resolution, seed_ari=seed_ari)
    sparse.save_npz(output / 'minimal_graph.npz', selected['adjacency'])
    return selected, seed_ari


def assign_all(data, fit):
    # A small dictionary graph makes feature refits practical; every valid bin
    # is assigned by five-neighbor distance-weighted vote in the SAME space.
    neighbors, distances = fit['search'].query(fit['x'], k=5)
    labels = fit['labels']
    k = int(labels.max() + 1)
    votes = np.zeros((len(data.raw), k))
    weight = 1 / np.maximum(distances, 1e-6)
    np.add.at(votes, (np.arange(len(votes))[:, None], labels[neighbors]), weight)
    assigned = votes.argmax(axis=1)
    confidence = votes.max(axis=1) / votes.sum(axis=1)
    assigned[data.dictionary] = labels
    confidence[data.dictionary] = 1.
    speed = [data.raw[assigned == i, 0].mean() for i in range(k)]
    order = np.argsort(np.argsort(speed))
    assigned = order[assigned]
    tasks = np.full(data.valid.shape, -1, dtype=int)
    tasks[data.valid] = assigned
    return tasks, confidence, order


def fit_role_model(x, k, seed=SEED):
    model = GaussianMixture(k, covariance_type='tied', reg_covar=.001,
                            n_init=10, max_iter=500, random_state=seed).fit(x)
    if not model.converged_:
        raise RuntimeError('Role GMM failed to converge')
    return model


def fit_roles(tasks, minutes, ants, state_speeds, output, bootstraps=200):
    k_states = int(tasks.max() + 1)
    proportions = state_proportions(tasks, k_states)
    hours = (tasks >= 0).sum(axis=1) * minutes / 60
    eligible = hours >= 12
    groups = np.full(len(ants), -1, dtype=int)
    summary = {}
    for side in ('left', 'right'):
        indices = np.flatnonzero(ants.side.eq(side) & eligible)
        x = np.sqrt(proportions[indices])
        fits = {k: fit_role_model(x, k) for k in range(1, 5)}
        rows = []
        for k, model in fits.items():
            labels = model.predict(x)
            row = dict(k=k, bic=model.bic(x), smallest_group=np.bincount(labels, minlength=k).min(),
                       bootstrap_median=np.nan, bootstrap_p10=np.nan, temporal_median=np.nan)
            if k > 1:
                rng = np.random.default_rng(SEED)
                bootstrap = []
                for repeat in range(bootstraps):
                    ix = rng.integers(len(x), size=len(x))
                    sampled = fit_role_model(x[ix], k, SEED + repeat)
                    bootstrap.append(adjusted_rand_score(labels, sampled.predict(x)))
                temporal = []
                for block in (30, 60, 120):
                    predictions = []
                    for half in (0, 1):
                        mask = (np.arange(tasks.shape[1]) * minutes // block) % 2 == half
                        xx = np.sqrt(state_proportions(tasks[indices][:, mask], k_states))
                        assert np.isfinite(xx).all()
                        predictions.append(fit_role_model(xx, k).predict(xx))
                    temporal.append(adjusted_rand_score(*predictions))
                row.update(bootstrap_median=np.median(bootstrap), bootstrap_p10=np.quantile(bootstrap, .1), temporal_median=np.median(temporal))
            rows.append(row)
        table = pd.DataFrame(rows).set_index('k')
        table['qualifies'] = ((table.bic < table.loc[1, 'bic']) & (table.smallest_group >= 5)
            & (table.bootstrap_median >= .8) & (table.temporal_median >= .6))
        accepted = table[table.qualifies]
        chosen = 1 if accepted.empty else int(accepted[accepted.bic <= accepted.bic.min() + 2].index.min())
        labels = fits[chosen].predict(x)
        expected_speed = proportions[indices] @ state_speeds
        remap = np.argsort(np.argsort([expected_speed[labels == g].mean() for g in range(chosen)]))
        groups[indices] = remap[labels]
        table.to_csv(output / f'{side}_role_k.csv')
        summary[side] = dict(k=chosen, n=len(indices), sizes=np.bincount(groups[indices]).tolist())
        print('ROLES', side, summary[side], flush=True)
    assignments = ants.copy()
    assignments['observed_bin_hours'] = hours
    assignments['eligible'] = eligible
    assignments['role'] = groups
    assignments.to_csv(output / 'ant_roles.csv', index=False)
    pd.DataFrame(proportions, index=ants.ant, columns=[f'T{i}' for i in range(k_states)]).to_csv(output / 'ant_state_proportions.csv')
    return assignments, proportions, summary


def run(source, inputs, output):
    threadpool_limits(1)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'search_protocol.json').write_text(json.dumps(PROTOCOL, indent=2) + '\n')
    ants = pd.read_csv(source / 'all_ant_coverage.csv')
    chosen_rows, all_rows = [], []
    for minutes in BINS:
        data = load_dataset(inputs / f'bins_{minutes:02d}.npz')
        table, chosen = sweep_dataset(data, ants, output)
        all_rows.append(table)
        chosen_rows.append(chosen)
    sweep = pd.concat(all_rows, ignore_index=True)
    sweep.to_csv(output / 'state_resolution_sweep.csv', index=False)
    choices = pd.DataFrame(chosen_rows)
    viable = choices[choices.supported]
    supported = not viable.empty
    if viable.empty:
        viable = choices
    order = ['k', 'ant_subsample_ari', 'seed_ari'] if supported else ['ant_subsample_ari', 'seed_ari', 'k']
    primary = viable.sort_values(order, ascending=False).iloc[0]
    minutes = int(primary.minutes)
    (output / 'state_selection_frozen.json').write_text(json.dumps(primary.to_dict(), indent=2) + '\n')
    print('PRIMARY', primary.to_dict(), flush=True)
    data = load_dataset(inputs / f'bins_{minutes:02d}.npz')
    with np.load(output / f'broad_{minutes:02d}min.npz') as z:
        broad = dict(z)
    # The baseline is assessed at the same support and graph resolution.
    core_row, core_fit = evaluate_subset(data, CORE, float(broad['resolution']), broad['labels'])
    pd.DataFrame([core_row]).to_csv(output / 'core_comparison.csv', index=False)
    selected, seed_ari = reduce_features(data, ants, broad, output)
    finish(source, output, data, ants, broad, primary, selected, seed_ari)


def finish(source, output, data, ants, broad, primary, selected, seed_ari):
    minutes = data.minutes
    tasks, confidence, remap = assign_all(data, selected)
    k = int(tasks.max() + 1)
    state_means = np.stack([data.raw[tasks[data.valid] == state].mean(axis=0) for state in range(k)])
    means = pd.DataFrame(state_means, columns=data.names, index=pd.Index(range(k), name='state'))
    means['bins'] = np.bincount(tasks[data.valid])
    means.to_csv(output / 'state_profiles.csv')
    # Freeze the unsupervised dictionary BEFORE running role fits or opening space.
    final = dict(minutes=minutes, n_states=k, selected_features=data.names[selected['columns']].tolist(),
                 selected_columns=selected['columns'].tolist(), n_features=len(selected['columns']),
                 resolution=float(broad['resolution']), state_seed_ari=seed_ari,
                 broad_supported=bool(primary.supported), n_valid_bins=len(data.raw), n_dictionary_bins=len(data.dictionary))
    # Low-motion candidate fixed without opening the sleep-label arrays.
    motion_score = [float(np.median(np.log1p(data.raw[tasks[data.valid] == state, 0] / .1)
                    + np.log1p(data.raw[tasks[data.valid] == state, 1] / .1)
                    + np.log1p(data.raw[tasks[data.valid] == state, 43] / .1))) for state in range(k)]
    final['low_motion_candidate'] = int(np.argmin(motion_score))
    final['low_motion_scores'] = motion_score
    (output / 'minimal_selection_frozen.json').write_text(json.dumps(final, indent=2) + '\n')
    assignments, proportions, roles = fit_roles(tasks, minutes, ants, state_means[:, 0], output)
    # External spatial comparison is downstream and cannot select features/K.
    reference = pd.read_csv(source / 'spatial_reference.csv')
    assignments = assignments.merge(reference[['side', 'track_id', 'spatial_cluster']],
        on=['side', 'track_id'], how='left', validate='one_to_one')
    spatial_rows = []
    for side in ('left', 'right'):
        sub = assignments[assignments.side.eq(side) & assignments.eligible]
        assert sub.spatial_cluster.notna().all()
        spatial_rows.append(dict(side=side, n=len(sub), k=roles[side]['k'],
                                 ari=adjusted_rand_score(sub.spatial_cluster, sub.role)))
        pd.crosstab(sub.role, sub.spatial_cluster).to_csv(output / f'{side}_spatial_contingency.csv')
    assignments.to_csv(output / 'ant_roles.csv', index=False)
    pd.DataFrame(spatial_rows).to_csv(output / 'spatial_comparison.csv', index=False)
    np.savez_compressed(output / 'states.npz', ants=ants.ant.to_numpy(str), tasks=tasks,
        values=data.values, counts=data.counts, proportions=proportions, roles=assignments.role,
        confidence=confidence, selected_columns=selected['columns'], center=selected['center'],
        scale=selected['scale'], names=data.names, dictionary=data.dictionary,
        state_remap=remap, minutes=minutes)
    summary = final | dict(roles=roles, spatial=spatial_rows, bootstraps=200)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('DONE', summary, flush=True)


def resume_dictionary(source, inputs, output):
    """Complete exports/roles from the saved dictionary, without repeating refits."""
    threadpool_limits(1)
    primary = pd.Series(json.loads((output / 'state_selection_frozen.json').read_text()))
    data = load_dataset(inputs / f'bins_{int(primary.minutes):02d}.npz')
    ants = pd.read_csv(source / 'all_ant_coverage.csv')
    with np.load(output / 'minimal_dictionary.npz') as z:
        saved = dict(z)
    np.testing.assert_array_equal(data.dictionary, saved['dictionary'])
    x, center, scale = balanced_matrix(data.raw, saved['columns'])
    np.testing.assert_array_equal(center, saved['center'])
    np.testing.assert_array_equal(scale, saved['scale'])
    graph, search, adjacency = graph_from_features(x[data.dictionary])
    selected = dict(x=x, center=center, scale=scale, graph=graph, search=search,
        adjacency=adjacency, labels=saved['labels'], columns=saved['columns'])
    with np.load(output / f'broad_{data.minutes:02d}min.npz') as z:
        broad = dict(z)
    finish(source, output, data, ants, broad, primary, selected, float(saved['seed_ari']))


def check_convergence(inputs, output):
    """Audit the selected broad partition without changing any model choice."""
    threadpool_limits(1)
    selected = json.loads((output / 'state_selection_frozen.json').read_text())
    minutes = int(selected['minutes'])
    data = load_dataset(inputs / f'bins_{minutes:02d}.npz')
    x, _, _ = balanced_matrix(data.raw, np.arange(N_FEATURES))
    graph, _, _ = graph_from_features(x[data.dictionary])
    with np.load(output / f'broad_{minutes:02d}min.npz') as z:
        reference, resolution = z['labels'], float(z['resolution'])
    labels, rows = [], []
    for i in range(3):
        fit = la.find_partition(graph, la.RBConfigurationVertexPartition,
            weights='weight', resolution_parameter=resolution, n_iterations=-1, seed=SEED+i)
        label = np.asarray(fit.membership)
        labels.append(label)
        rows.append(dict(seed=SEED+i, k=int(label.max()+1),
            ari_to_three_iteration_reference=adjusted_rand_score(reference, label), quality=fit.quality()))
    result = dict(minutes=minutes, resolution=resolution, iterations='until convergence', fits=rows,
        seed_ari=float(np.median([adjusted_rand_score(a, b) for a, b in combinations(labels, 2)])))
    (output / 'convergence_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print('CONVERGENCE', result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'inputs', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--check-convergence', action='store_true', help='Audit the selected broad fit; leave selection unchanged')
    parser.add_argument('--resume-dictionary', action='store_true', help='Complete roles and exports from a saved minimal_dictionary.npz')
    a = parser.parse_args()
    if a.check_convergence:
        check_convergence(a.inputs, a.output)
    elif a.resume_dictionary:
        resume_dictionary(a.source, a.inputs, a.output)
    else:
        run(a.source, a.inputs, a.output)
