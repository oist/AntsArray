"""Fit physical ant phenotypes without space, then evaluate fixed spatial labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from analysis.velocity_posture_features import FEATURES, MIN_HOURS, SEED

FAMILIES = {
    "joint": ("velocity", "posture", "dynamics"),
    "velocity_only": ("velocity",),
    "posture_only": ("posture", "dynamics"),
}


def summarize_hours(hourly):
    """Hourly distributions, with all measured hours weighted equally."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            np.nanquantile(hourly, [0.25, 0.5, 0.75], axis=1)
            .transpose(1, 0, 2)
            .reshape(len(hourly), -1)
        )


class PhenotypeTransform:
    """Fit scaling and PCA only on the specified training ants/day."""

    def fit(self, x, families):
        groups = np.tile([f[3] for f in FEATURES], 3)
        allowed = np.isin(groups, families)
        raw = self.physical_transform(x)
        self.keep = (
            allowed & np.isfinite(raw).all(axis=0) & (np.std(raw, axis=0) > 1e-7)
        )
        z = raw[:, self.keep]
        self.center = np.median(z, axis=0)
        q = np.quantile(z, [0.25, 0.75], axis=0)
        self.scale = np.maximum(q[1] - q[0], np.std(z, axis=0) * 0.25)
        z = np.clip((z - self.center) / self.scale, -5, 5)
        groups = groups[self.keep]
        self.block_scale = np.ones(z.shape[1])
        for group in np.unique(groups):
            ix = groups == group
            self.block_scale[ix] = max(np.sqrt(z[:, ix].var(axis=0).sum()), 1e-8)
        z /= self.block_scale
        full = PCA(svd_solver="full").fit(z)
        needed = np.searchsorted(np.cumsum(full.explained_variance_ratio_), 0.9) + 1
        self.n_components = int(min(6, needed, len(z) - 1, z.shape[1]))
        self.pca = PCA(n_components=self.n_components, svd_solver="full").fit(z)
        return self

    @staticmethod
    def physical_transform(x):
        z = np.asarray(x, float).copy()
        for j in range(z.shape[1]):
            name, _, _, group = FEATURES[j % len(FEATURES)]
            if name in ("forward_velocity", "lateral_velocity"):
                z[:, j] = np.arcsinh(z[:, j] / 0.1)
            elif name in ("speed", "speed_p90", "lateral_magnitude"):
                z[:, j] = np.log1p(np.maximum(z[:, j], 0) / 0.1)
            elif group == "dynamics":
                z[:, j] = np.log1p(np.maximum(z[:, j], 0) / 30)
        return z

    def transform(self, x):
        z = self.physical_transform(x)[:, self.keep]
        return self.pca.transform(
            np.clip((z - self.center) / self.scale, -5, 5) / self.block_scale
        )


class BlockPhenotypeTransform:
    """One independently learned dominant axis per physical feature family."""

    def fit(self, x, families):
        self.blocks = []
        for family in families:
            t = PhenotypeTransform().fit(x, (family,))
            scale = float(np.sqrt(t.pca.explained_variance_[0]))
            self.blocks.append((family, t, scale))
        self.n_components = len(self.blocks)
        self.pca = SimpleNamespace(
            explained_variance_ratio_=np.array(
                [
                    t.pca.explained_variance_ratio_[0] / len(self.blocks)
                    for _, t, _ in self.blocks
                ]
            )
        )
        return self

    def transform(self, x):
        return np.column_stack(
            [t.transform(x)[:, 0] / scale for _, t, scale in self.blocks]
        )


class CommonPhenotypeTransform:
    """Dominant covariance pattern of equally standardized physical features.

    All hourly quartiles enter. PCA chooses their loadings without spatial labels.
    One component is an explicit simplicity constraint, not chosen using labels.
    """

    def fit(self, x, families):
        raw = PhenotypeTransform.physical_transform(x)
        groups = np.tile([f[3] for f in FEATURES], 3)
        self.keep = np.isin(groups, families) & (np.std(raw, axis=0) > 1e-7)
        self.center = raw[:, self.keep].mean(axis=0)
        self.scale = raw[:, self.keep].std(axis=0)
        self.pca = PCA(n_components=1, svd_solver="full").fit(
            (raw[:, self.keep] - self.center) / self.scale
        )
        self.n_components = 1
        return self

    def transform(self, x):
        raw = PhenotypeTransform.physical_transform(x)
        return self.pca.transform((raw[:, self.keep] - self.center) / self.scale)


def fitted(x, groups, k, seed=SEED, n_init=25):
    t = PhenotypeTransform().fit(x, groups)
    z = t.transform(x)
    m = KMeans(n_clusters=k, n_init=n_init, max_iter=500, random_state=seed).fit(z)
    return t, m, z


def predict(model, x):
    t, m = model
    return m.predict(t.transform(x))


def matched_accuracy(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    table = np.zeros((len(ua), len(ub)), int)
    np.add.at(table, (ia, ib), 1)
    ra, rb = linear_sum_assignment(-table)
    return (
        float(table[ra, rb].sum() / len(a)),
        table,
        {str(ua[i]): str(ub[j]) for i, j in zip(ra, rb)},
    )


def choose_k(table):
    allowed = table[
        (table.k >= 2)
        & (table.silhouette >= 0.25)
        & (table.bootstrap_ari_median >= 0.8)
        & (table.split_minute_ari >= 0.6)
        & (table.smallest_group >= table.minimum_group_size)
    ]
    if allowed.empty:
        return 1
    return int(
        allowed.loc[allowed.silhouette >= allowed.silhouette.max() - 0.02, "k"].min()
    )


def fit_models(output, bootstrap=100):
    qc = pd.read_csv(output / "all_ant_coverage.csv")
    with np.load(output / "physical_measurements.npz") as f:
        hourly = f["hourly"]
        even = f["hourly_even"]
        odd = f["hourly_odd"]
    x = summarize_hours(hourly[:, :24])
    test = summarize_hours(hourly[:, 24:])
    even_x = summarize_hours(even[:, :24])
    odd_x = summarize_hours(odd[:, :24])
    replicate_ok = (np.isfinite(even[:, :24]).sum(axis=1).min(axis=1) >= 8) & (
        np.isfinite(odd[:, :24]).sum(axis=1).min(axis=1) >= 8
    )
    candidates = []
    assignments = []
    summaries = []
    models = {}
    coordinates = []
    for side in ("left", "right"):
        ix = np.flatnonzero(qc.side.eq(side) & qc.day1_eligible)
        for family, groups in FAMILIES.items():
            rng = np.random.default_rng(SEED)
            print("FIT", side, family, "n", len(ix), flush=True)
            xt = x[ix]
            rep = np.flatnonzero(replicate_ok[ix])
            models_for_k = {}
            rows = []
            for k in range(1, min(5, len(ix) // 4) + 1):
                t, m, z = fitted(xt, groups, k)
                labels = m.labels_
                models_for_k[k] = (t, m, z)
                sizes = np.bincount(labels, minlength=k)
                row = dict(
                    side=side,
                    family=family,
                    k=k,
                    n_ants=len(ix),
                    n_components=t.n_components,
                    retained_variance=float(t.pca.explained_variance_ratio_.sum()),
                    smallest_group=int(sizes.min()),
                    minimum_group_size=max(4, int(np.ceil(0.1 * len(ix)))),
                    group_sizes=json.dumps(sizes.tolist()),
                    inertia=float(m.inertia_),
                    silhouette=np.nan,
                    bootstrap_ari_median=np.nan,
                    bootstrap_ari_p10=np.nan,
                    split_minute_ari=np.nan,
                    n_split_minute_ants=len(rep),
                )
                if k > 1:
                    row["silhouette"] = float(silhouette_score(z, labels))
                    ari = []
                    for repeat in range(bootstrap):
                        sample = rng.integers(len(xt), size=len(xt))
                        bt, bm, _ = fitted(
                            xt[sample], groups, k, seed=SEED + repeat + 1, n_init=10
                        )
                        ari.append(adjusted_rand_score(labels, predict((bt, bm), xt)))
                    row["bootstrap_ari_median"] = float(np.median(ari))
                    row["bootstrap_ari_p10"] = float(np.quantile(ari, 0.1))
                    if len(rep) >= max(12, 2 * k):
                        et, em, _ = fitted(even_x[ix[rep]], groups, k)
                        ot, om, _ = fitted(odd_x[ix[rep]], groups, k)
                        row["split_minute_ari"] = float(
                            adjusted_rand_score(em.labels_, om.labels_)
                        )
                rows.append(row)
            table = pd.DataFrame(rows)
            chosen = choose_k(table)
            table["selected"] = table.k.eq(chosen)
            candidates.append(table)
            t, m, z = models_for_k[chosen]
            # Activity label order follows median speed only, never spatial overlap.
            med_speed = x[ix, len(FEATURES) + 2]
            order = sorted(
                range(chosen), key=lambda c: np.median(med_speed[m.labels_ == c])
            )
            remap = np.argsort(order)
            key = f"{side}/{family}"
            models[key] = dict(transform=t, model=m, remap=remap, indices=ix)
            d2ok = qc.iloc[ix].day2_eligible.to_numpy(bool)
            d2labels = np.full(len(ix), -1, int)
            d2labels[d2ok] = remap[predict((t, m), test[ix[d2ok]])]
            d1labels = remap[m.labels_]
            # A fully independent day-2 refit is a validation diagnostic only.
            independent_ari = np.nan
            if chosen > 1 and d2ok.sum() >= max(12, 2 * chosen):
                _, independent, _ = fitted(test[ix[d2ok]], groups, chosen)
                independent_ari = float(
                    adjusted_rand_score(d1labels[d2ok], independent.labels_)
                )
            summaries.append(
                dict(
                    side=side,
                    family=family,
                    selected_k=chosen,
                    n_ants=len(ix),
                    day2_ants=int(d2ok.sum()),
                    day2_frozen_retention=(
                        float(np.mean(d2labels[d2ok] == d1labels[d2ok]))
                        if chosen > 1 and d2ok.any()
                        else np.nan
                    ),
                    day2_independent_ari=independent_ari,
                    selected_silhouette=float(
                        table.loc[table.k.eq(chosen), "silhouette"].iloc[0]
                    ),
                )
            )
            for pos, ant_ix in enumerate(ix):
                base = qc.iloc[ant_ix].to_dict()
                assignments.append(
                    dict(
                        **base,
                        family=family,
                        activity_cluster=int(d1labels[pos]),
                        day2_cluster=int(d2labels[pos]),
                        selected_k=chosen,
                    )
                )
                coordinates.append(
                    dict(
                        ant=base["ant"],
                        side=side,
                        family=family,
                        pc1=float(z[pos, 0]),
                        pc2=float(z[pos, 1]) if z.shape[1] > 1 else 0.0,
                        activity_cluster=int(d1labels[pos]),
                    )
                )
            print("SELECTED", side, family, chosen, flush=True)
    pd.concat(candidates, ignore_index=True).to_csv(
        output / "k_selection.csv", index=False
    )
    pd.DataFrame(assignments).to_csv(output / "activity_assignments.csv", index=False)
    pd.DataFrame(coordinates).to_csv(output / "activity_coordinates.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "unsupervised_summary.csv", index=False)
    joblib.dump(models, output / "phenotype_models.joblib")
    np.savez_compressed(
        output / "ant_profiles.npz",
        training=x,
        heldout=test,
        even=even_x,
        odd=odd_x,
        feature_names=np.array(
            [f"{q}_{f[0]}" for q in ("q25", "median", "q75") for f in FEATURES]
        ),
    )
    frozen = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        primary_family="joint",
        groups=FAMILIES,
        selection_rule="Smallest K within 0.02 of highest eligible silhouette; require silhouette >=0.25, bootstrap median ARI >=0.8, even/odd-minute ARI >=0.6, and each group >=max(4,10% of ants). Otherwise K=1.",
        spatial_labels_loaded=False,
        day2_used_for_selection=False,
        bootstrap_repeats=bootstrap,
        file_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                output / "physical_measurements.npz",
                output / "activity_assignments.csv",
                output / "k_selection.csv",
                output / "phenotype_models.joblib",
            ]
        },
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "UNSUPERVISED_FROZEN.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )


def evaluate(output, spatial_reference):
    frozen = json.loads((output / "UNSUPERVISED_FROZEN.json").read_text())
    for name, digest in frozen["file_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest, name
    spatial = pd.read_csv(spatial_reference).rename(
        columns={"TrackID": "track_id", "cluster_id": "spatial_cluster"}
    )
    # Grid occupancy exports globally unique strings such as left_0/right_1.
    # Here side is already a separate key; retain the within-colony integer.
    spatial["spatial_cluster"] = (
        spatial.spatial_cluster.astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    )
    assert not spatial.duplicated(["side", "track_id"]).any()
    assignments = pd.read_csv(output / "activity_assignments.csv")
    merged = assignments.merge(
        spatial[["side", "track_id", "spatial_cluster"]],
        on=["side", "track_id"],
        how="left",
        validate="many_to_one",
    )
    metrics = []
    for (side, family), part in merged.groupby(["side", "family"]):
        common = part.dropna(subset=["spatial_cluster"])
        k = int(part.selected_k.iloc[0])
        if len(common):
            accuracy, table, mapping = matched_accuracy(
                common.activity_cluster, common.spatial_cluster
            )
            ari = adjusted_rand_score(common.spatial_cluster, common.activity_cluster)
        else:
            accuracy = ari = np.nan
            mapping = {}
            table = []
        mask = (merged.side == side) & (merged.family == family)
        merged.loc[mask, "spatial_label_for_activity"] = (
            merged.loc[mask, "activity_cluster"].astype(str).map(mapping).astype(float)
        )
        metrics.append(
            dict(
                side=side,
                family=family,
                selected_k=k,
                n_model_ants=len(part),
                n_spatial_reference_ants=int(spatial.side.eq(side).sum()),
                n_compared=len(common),
                matched_accuracy=accuracy,
                adjusted_rand_index=ari,
                spatial_mapping=json.dumps(mapping),
                contingency=json.dumps(
                    table.tolist() if hasattr(table, "tolist") else table
                ),
            )
        )
    merged["spatial_agreement"] = np.where(
        merged.spatial_cluster.notna(),
        merged.spatial_cluster.eq(merged.spatial_label_for_activity),
        np.nan,
    )
    merged.to_csv(output / "assignments_with_spatial_comparison.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "spatial_agreement.csv", index=False)
    pd.read_csv(spatial_reference).to_csv(output / "spatial_reference.csv", index=False)
    (output / "spatial_reference_provenance.json").write_text(
        json.dumps(
            dict(
                path=str(spatial_reference),
                sha256=hashlib.sha256(spatial_reference.read_bytes()).hexdigest(),
                evaluated_at=datetime.now(timezone.utc).isoformat(),
                policy="Labels loaded only after UNSUPERVISED_FROZEN.json was written. Permutation aligns display names only.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        pd.DataFrame(metrics)
        .drop(columns=["spatial_mapping", "contingency"])
        .to_string(index=False),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["fit", "evaluate"], required=True)
    p.add_argument("--spatial-reference", type=Path)
    p.add_argument("--bootstrap", type=int, default=100)
    a = p.parse_args()
    if a.stage == "fit":
        fit_models(a.output, a.bootstrap)
    elif a.spatial_reference is None:
        p.error("--spatial-reference is required for evaluation")
    else:
        evaluate(a.output, a.spatial_reference)


if __name__ == "__main__":
    main()
