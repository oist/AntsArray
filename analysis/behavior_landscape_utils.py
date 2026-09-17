"""Balanced multi-view clustering and time-disjoint forecast comparisons.

All transformations and model selection use the training day only. Cluster
forecasts use other ants' training-day means, never the focal ant's own value.
The reverse time direction is a reproducibility check, not a prospective test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from analysis import grid_occupancy_utils as go


FEATURE_SETS = {
    "occupancy": ("occupancy",),
    "activity": ("activity",),
    "interactions": ("interactions",),
    "occupancy_activity": ("occupancy", "activity"),
    "occupancy_timing": ("occupancy", "timing"),
    "occupancy_interactions": ("occupancy", "interactions"),
    "full": ("occupancy", "activity", "timing", "interactions"),
    "coverage": ("coverage",),
    "occupancy_coverage": ("occupancy", "coverage"),
}
ACTIVITY = ["speed", "body", "antenna", "sleep"]
INTERACTIONS = ["contacts", "bout_presence", "partner_diversity", "median_bout_seconds"]
COVERAGE = ["position_coverage", "speed_coverage", "body_coverage", "antenna_coverage", "sleep_coverage"]
OUTCOMES = ("activity", "timing", "interactions", "space_use")


@dataclass(frozen=True)
class ModelSettings:
    max_clusters: int = 6
    min_cluster_size: int = 4
    stability_repeats: int = 30
    stability_fraction: float = .8
    min_stability: float = .75
    bootstrap_repeats: int = 2000
    seed: int = 724

    def validate(self):
        if self.max_clusters < 2 or self.min_cluster_size < 2:
            raise ValueError("Need max_clusters >=2 and min_cluster_size >=2")
        if self.stability_repeats < 2 or not 0 < self.stability_fraction < 1:
            raise ValueError("Need repeated proper ant subsamples for stability")
        if self.bootstrap_repeats < 100 or not 0 <= self.min_stability <= 1:
            raise ValueError("Invalid bootstrap count or stability threshold")


def nanmean(x, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def arrays_for_colony(features, side, min_coverage=None, min_profile_bins=None):
    tracks = features["tracks"].query("side == @side").reset_index(drop=True)
    audit = features["audit"].query("side == @side").copy()
    if min_coverage is not None:
        cols = COVERAGE + ["contacts_coverage"]
        ok = audit[cols].ge(min_coverage).all(axis=1)
        support = features["settings"]["min_profile_bins"] if min_profile_bins is None else min_profile_bins
        ok &= audit[[m + "_profile_bins" for m in ("speed", "sleep", "contacts")]].ge(support).all(axis=1)
        audit["included"] = ok.groupby(audit.track_id).transform("all")
    ids = audit.loc[audit.included, "track_id"].unique()
    keep = tracks.track_id.isin(ids).to_numpy()
    selected = tracks[keep].reset_index(drop=True)
    days, outcomes = [], []
    for day in (0, 1):
        daily = features["daily"].query("side == @side and day == @day").set_index("track_id").loc[selected.track_id]
        profiles = features["profiles"].query("side == @side and day == @day")
        timing = []
        for key in ("speed", "sleep", "contacts"):
            values = profiles.pivot(index="track_id", columns="slot", values=key).loc[selected.track_id].to_numpy()
            if key != "sleep":
                values = np.log1p(values)
            # Relative timing is separate from overall activity/contact amount.
            timing.append(values - nanmean(values, axis=1)[:, None])
        activity = daily[ACTIVITY].to_numpy().copy()
        activity[:, :3] = np.log1p(activity[:, :3])
        interactions = daily[INTERACTIONS].to_numpy().copy()
        interactions[:, [0, 3]] = np.log1p(interactions[:, [0, 3]])
        blocks = dict(occupancy=np.sqrt(features["spatial"][side][keep, day]),
                      activity=activity, timing=np.concatenate(timing, axis=1),
                      interactions=interactions, coverage=daily[COVERAGE].to_numpy())
        days.append(blocks)
        outcomes.append({**{k: blocks[k] for k in OUTCOMES if k != "space_use"},
                         "space_use": daily[["outside", "resource_fraction"]].to_numpy()})
    return selected, days, outcomes


class BlockScaler:
    """Train-only imputation/scaling; equal total variance per feature family.

    Square-root occupancy retains its original geometry: individual grid cells
    are not z-scored, which would amplify rare occupied cells. Other features
    are z-scored before total-block balancing. Missingness is not a zero value.
    """
    def fit(self, blocks, names):
        self.parameters = {}
        for name in names:
            raw = np.asarray(blocks[name], float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                std = np.nanstd(raw, axis=0)
                median = np.nanmedian(raw, axis=0)
            keep = (np.isfinite(raw).mean(axis=0) >= .8) & (std > 1e-10)
            if not keep.any():
                continue
            fill = median[keep]
            values = np.where(np.isfinite(raw[:, keep]), raw[:, keep], fill)
            center = values.mean(axis=0)
            scale = np.ones(keep.sum()) if name == "occupancy" else values.std(axis=0)
            values = (values - center) / scale
            energy = np.sqrt(np.mean(np.sum(values ** 2, axis=1)))
            self.parameters[name] = (keep, fill, center, scale, energy)
        if not self.parameters:
            raise ValueError("No variable, adequately observed training features")
        return self

    def transform(self, blocks):
        parts = []
        for name, (keep, fill, center, scale, energy) in self.parameters.items():
            raw = np.asarray(blocks[name], float)[:, keep]
            values = np.where(np.isfinite(raw), raw, fill)
            parts.append((values - center) / scale / energy)
        return np.concatenate(parts, axis=1) / np.sqrt(len(parts))


def compact_coordinates(x):
    """Exact isometry of training rows into <= n-ant coordinates; no truncation."""
    gram = x @ x.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > 1e-12
    return eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])


def fit_partition(x, k, seed):
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(compact_coordinates(x))


def partition_stability(blocks, names, labels, k, settings):
    rng = np.random.default_rng(settings.seed)
    n = len(labels)
    values = []
    for repeat in range(settings.stability_repeats):
        idx = np.sort(rng.choice(n, max(k + 1, int(settings.stability_fraction * n)), replace=False))
        subset = {key: value[idx] for key, value in blocks.items()}
        scaler = BlockScaler().fit(subset, names)
        resampled = fit_partition(scaler.transform(subset), k, settings.seed + repeat + 1)
        values.append(adjusted_rand_score(labels[idx], resampled))
    return float(np.median(values)), float(np.quantile(values, .1))


def other_ant_mean(values, weights):
    """NaN-aware weighted training means with zero self-weight enforced."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float).copy()
    np.fill_diagonal(weights, 0)
    valid = np.isfinite(values)
    sums = weights @ np.where(valid, values, 0.)
    counts = weights @ valid.astype(float)
    return np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0)


def score_predictions(train, test, weights, identity, metadata):
    """Per-ant losses in common, train-scaled held-out outcome families."""
    rows = []
    n = len(identity)
    for family in OUTCOMES:
        a, b = train[family], test[family]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            std = np.nanstd(a, axis=0)
        keep = (np.isfinite(a).mean(axis=0) >= .8) & (std > 1e-10)
        if not keep.any():
            continue
        a, b = a[:, keep] / std[keep], b[:, keep] / std[keep]
        baseline = other_ant_mean(a, np.ones((n, n)))
        predicted = a if weights is None else other_ant_mean(a, weights)
        # An entirely unobserved neighborhood target gets the training-colony
        # forecast. Do not improve a model's score by dropping its hard targets.
        predicted = np.where(np.isfinite(predicted), predicted, baseline)
        valid = np.isfinite(b) & np.isfinite(predicted) & np.isfinite(baseline)
        loss = nanmean(np.where(valid, (b - predicted) ** 2, np.nan), axis=1)
        base_loss = nanmean(np.where(valid, (b - baseline) ** 2, np.nan), axis=1)
        for i, ant in enumerate(identity.track_id):
            rows.append(dict(**metadata, track_id=int(ant), outcome=family, squared_error=loss[i],
                             baseline_squared_error=base_loss[i], n_outcome_features=int(valid[i].sum())))
    return rows


def select_training_model(table, settings):
    valid = table[table.admissible & table.stability_median.ge(settings.min_stability)]
    if valid.empty:
        return 1
    return int(valid.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k)


def evaluate_colony(identity, days, outcomes, side, settings):
    n = len(identity)
    if n < 2 * settings.min_cluster_size:
        raise ValueError(f"Too few eligible ants for {side}: {n}")
    candidates, losses, assignments, leiden = [], [], [], []
    for train_day in (0, 1):
        train, test = days[train_day], days[1 - train_day]
        metadata = dict(side=side, train_day=train_day, n_ants=n)
        losses += score_predictions(outcomes[train_day], outcomes[1 - train_day], np.ones((n, n)), identity,
                                    dict(**metadata, feature_set="colony_mean", method="mean", k=1))
        losses += score_predictions(outcomes[train_day], outcomes[1 - train_day], None, identity,
                                    dict(**metadata, feature_set="own_previous_day", method="persistence", k=0))
        for feature_set, names in FEATURE_SETS.items():
            scaler = BlockScaler().fit(train, names)
            x = scaler.transform(train)
            test_scaler = BlockScaler().fit(test, names)
            x_test = test_scaler.transform(test)
            # A continuous local-neighborhood forecast tests information lost by discretizing.
            distances = cdist(x, x)
            np.fill_diagonal(distances, np.inf)
            neighbors = np.argsort(distances, axis=1)[:, :min(5, n - 1)]
            weights = np.zeros((n, n))
            np.put_along_axis(weights, neighbors, 1, axis=1)
            losses += score_predictions(outcomes[train_day], outcomes[1 - train_day], weights, identity,
                                        dict(**metadata, feature_set=feature_set, method="neighbors", k=5))
            for k in range(2, min(settings.max_clusters, n // settings.min_cluster_size) + 1):
                labels = fit_partition(x, k, settings.seed)
                minimum = int(np.bincount(labels, minlength=k).min())
                admissible = minimum >= settings.min_cluster_size
                stability, low = partition_stability(train, names, labels, k, settings)
                # Independently fit the other day; its outcomes never select k.
                labels_test = fit_partition(x_test, k, settings.seed)
                row = dict(**metadata, feature_set=feature_set, k=k, min_cluster_size=minimum,
                           admissible=admissible, silhouette=float(silhouette_score(x, labels)),
                           stability_median=stability, stability_p10=low,
                           cross_day_ari=float(adjusted_rand_score(labels, labels_test)))
                candidates.append(row)
                for ant, label in zip(identity.track_id, labels):
                    assignments.append(dict(**metadata, feature_set=feature_set, k=k, track_id=int(ant), label=int(label)))
                if admissible:
                    weights = labels[:, None] == labels[None, :]
                    losses += score_predictions(outcomes[train_day], outcomes[1 - train_day], weights, identity,
                                                dict(**metadata, feature_set=feature_set, method="clusters", k=k))
            # Direct check with the canonical grid workflow's graph construction.
            if feature_set not in ("coverage", "occupancy_coverage"):
                for resolution in (.5, 1., 1.5, 2.):
                    labels = go.leiden_labels(x, n_neighbors=min(10, n - 1), resolution=resolution,
                                              random_state=settings.seed)
                    labels_test = go.leiden_labels(x_test, n_neighbors=min(10, n - 1),
                                                   resolution=resolution, random_state=settings.seed)
                    for ant, label in zip(identity.track_id, labels):
                        leiden.append(dict(**metadata, feature_set=feature_set, resolution=resolution,
                                           k=len(np.unique(labels)), track_id=int(ant), label=int(label),
                                           cross_day_ari=float(adjusted_rand_score(labels, labels_test))))
        print(f"{side}: trained on window {train_day + 1}; {n} ants", flush=True)
    table = pd.DataFrame(candidates)
    selected = []
    loss_table = pd.DataFrame(losses)
    for (day, feature_set), group in table.groupby(["train_day", "feature_set"]):
        k = select_training_model(group, settings)
        selected.append(dict(side=side, train_day=day, feature_set=feature_set, selected_k=k))
        if k == 1:
            part = loss_table[(loss_table.train_day == day) & (loss_table.feature_set == "colony_mean")].copy()
            part["feature_set"] = feature_set
        else:
            part = loss_table[(loss_table.train_day == day) & (loss_table.feature_set == feature_set)
                              & (loss_table.method == "clusters") & (loss_table.k == k)].copy()
        part["method"] = "selected_clusters"
        losses.extend(part.to_dict("records"))
    return table, pd.DataFrame(losses), pd.DataFrame(assignments), pd.DataFrame(selected), pd.DataFrame(leiden)


def summarize_scores(losses):
    keys = ["side", "train_day", "feature_set", "method", "k", "outcome"]
    summary = losses.groupby(keys, as_index=False).agg(mse=("squared_error", "mean"),
                 baseline_mse=("baseline_squared_error", "mean"), n_ants=("track_id", "nunique"))
    summary["heldout_r2"] = 1 - summary.mse / summary.baseline_mse
    return summary


def paired_gains(losses, settings):
    """Matched-k loss reductions, resampling ant identities across both directions.

    Intervals are conditional on these fitted clusters and these two windows;
    they are not independent-colony intervals or model-selection-adjusted tests.
    """
    rows = []
    rng = np.random.default_rng(settings.seed)
    source = losses[losses.method.isin(["clusters", "neighbors", "selected_clusters"])].copy()
    # Selected models can use different k by day; match the selection procedure.
    source["comparison_k"] = np.where(source.method.eq("selected_clusters"), 0, source.k)
    keys = ["side", "train_day", "method", "comparison_k", "track_id", "outcome"]
    baseline = source[source.feature_set.eq("occupancy")][keys + ["squared_error"]].rename(columns={"squared_error": "occupancy_error"})
    joined = source.merge(baseline, on=keys, validate="many_to_one")
    grouped = joined.groupby(["side", "method", "feature_set", "comparison_k", "outcome"])
    for (side, method, feature_set, k, outcome), all_days in grouped:
        if feature_set == "occupancy":
            continue
        for direction, day in (("forward", 0), ("reverse", 1), ("both", None)):
            group = all_days if day is None else all_days[all_days.train_day.eq(day)]
            if group.train_day.nunique() != (2 if day is None else 1):
                continue
            per_ant = group.groupby("track_id")[["squared_error", "occupancy_error"]].mean().dropna()
            a, b = per_ant.squared_error.to_numpy(), per_ant.occupancy_error.to_numpy()
            if len(a) < 4 or b.mean() <= 0:
                continue
            idx = rng.integers(0, len(a), (settings.bootstrap_repeats, len(a)))
            draws = 1 - a[idx].mean(axis=1) / b[idx].mean(axis=1)
            rows.append(dict(side=side, method=method, feature_set=feature_set, k=int(k), outcome=outcome,
                             direction=direction, n_ants=len(a), fractional_mse_reduction=float(1 - a.mean() / b.mean()),
                             ci_low=float(np.quantile(draws, .025)), ci_high=float(np.quantile(draws, .975))))
    return pd.DataFrame(rows)


def run_comparison(features, settings=ModelSettings(), *, min_coverage=None, min_profile_bins=None):
    settings.validate()
    outputs = []
    with threadpool_limits(limits=1):
        for side in ("left", "right"):
            identity, days, outcomes = arrays_for_colony(features, side, min_coverage, min_profile_bins)
            outputs.append(evaluate_colony(identity, days, outcomes, side, settings))
    names = ["cluster_diagnostics", "heldout_losses", "assignments", "selected_models", "leiden_assignments"]
    tables = {name: pd.concat([result[i] for result in outputs], ignore_index=True) for i, name in enumerate(names)}
    tables["heldout_scores"] = summarize_scores(tables["heldout_losses"])
    tables["paired_gains"] = paired_gains(tables["heldout_losses"], settings)
    return tables
