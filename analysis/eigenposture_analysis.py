"""Cluster hourly distributions of learned posture modes and body velocities."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from analysis.eigenposture_features import SEED

FAMILIES = {
    "joint": ("velocity", "posture", "dynamics"),
    "velocity_only": ("velocity",),
    "posture_only": ("posture", "dynamics"),
}


def profiles(hourly):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            np.nanquantile(hourly, [0.25, 0.5, 0.75], axis=1)
            .transpose(1, 0, 2)
            .reshape(len(hourly), -1)
        )


class ProfileAxis:
    """Common physical axis; no hand-defined shape feature enters scaling/PCA."""

    @staticmethod
    def values(x, names):
        z = x.copy()
        for j, name in enumerate(np.tile(names, 3)):
            if name in ("forward_mean", "lateral_mean"):
                z[:, j] = np.arcsinh(z[:, j] / 0.1)
            elif name in ("forward_rms", "lateral_rms"):
                z[:, j] = np.log1p(np.maximum(z[:, j], 0) / 0.1)
            elif name.endswith("_rate"):
                z[:, j] = np.log1p(np.maximum(z[:, j], 0))
        return z

    def fit(self, x, names, families, allowed):
        self.names = names
        raw = self.values(x, names)
        self.keep = (
            np.isin(np.tile(families, 3), allowed)
            & np.isfinite(raw).all(axis=0)
            & (raw.std(axis=0) > 1e-8)
        )
        raw = raw[:, self.keep]
        self.center = raw.mean(axis=0)
        self.scale = raw.std(axis=0)
        self.pca = PCA(1, svd_solver="full").fit((raw - self.center) / self.scale)
        return self

    def transform(self, x):
        return self.pca.transform(
            (self.values(x, self.names)[:, self.keep] - self.center) / self.scale
        )


def mixture(z, k, seed=SEED, n_init=20):
    candidates = [
        GaussianMixture(
            k,
            covariance_type="full",
            reg_covar=0.01,
            n_init=n_init,
            max_iter=500,
            random_state=seed,
        ).fit(z)
    ]
    if k > 1:
        groups = np.array_split(np.argsort(z[:, 0]), k)
        candidates.append(
            GaussianMixture(
                k,
                covariance_type="full",
                reg_covar=0.01,
                n_init=1,
                max_iter=500,
                random_state=seed,
                means_init=np.stack([z[g].mean(axis=0) for g in groups]),
                precisions_init=np.stack(
                    [
                        np.linalg.inv(
                            np.atleast_2d(np.cov(z[g], rowvar=False))
                            + np.eye(z.shape[1]) * 0.01
                        )
                        for g in groups
                    ]
                ),
                weights_init=np.array([len(g) / len(z) for g in groups]),
            ).fit(z)
        )
    fits = [m for m in candidates if m.converged_]
    if not fits:
        raise RuntimeError("Mixture did not converge")
    return max(fits, key=lambda m: m.lower_bound_)


def select(table):
    baseline = float(table.loc[table.k.eq(1), "bic"].iloc[0])
    eligible = table[
        (table.k > 1)
        & (table.bic < baseline)
        & (table.bootstrap_median_ari >= 0.8)
        & (table.split_minute_ari >= 0.6)
        & (table.smallest_group >= table.minimum_group_size)
    ]
    return (
        1
        if eligible.empty
        else int(eligible.loc[eligible.bic <= eligible.bic.min() + 2, "k"].min())
    )


def fit(output, bootstrap=100):
    q = pd.read_csv(output / "all_ant_coverage.csv")
    with np.load(output / "eigenposture_measurements.npz") as z:
        hourly = z["hourly"]
        names = z["feature_names"]
        families = z["families"]
        x = profiles(hourly[:, :24])
        test = profiles(hourly[:, 24:])
        even = profiles(z["hourly_even"][:, :24])
        odd = profiles(z["hourly_odd"][:, :24])
    np.savez_compressed(
        output / "ant_profiles.npz",
        training=x,
        heldout=test,
        even=even,
        odd=odd,
        feature_names=np.array(
            [f"{quart}_{name}" for quart in ("q25", "median", "q75") for name in names]
        ),
    )
    models = {}
    assignments = []
    summaries = []
    candidates = []
    coords = []
    for side in ["left", "right"]:
        ix = np.flatnonzero(q.side.eq(side) & q.day1_eligible)
        repeat = q.iloc[ix].even_eligible.to_numpy(bool) & q.iloc[
            ix
        ].odd_eligible.to_numpy(bool)
        for family, allowed in FAMILIES.items():

            def transform(data):
                return ProfileAxis().fit(data, names, families, allowed)

            print("FIT", side, family, "n", len(ix), flush=True)
            t = transform(x[ix])
            z = t.transform(x[ix])
            et = transform(even[ix[repeat]])
            ot = transform(odd[ix[repeat]])
            ez = et.transform(even[ix[repeat]])
            oz = ot.transform(odd[ix[repeat]])
            rng = np.random.default_rng(SEED)
            rows = []
            fits = {}
            for k in range(1, 5):
                m = mixture(z, k)
                fits[k] = m
                labels = m.predict(z)
                sizes = np.bincount(labels, minlength=k)
                row = dict(
                    side=side,
                    family=family,
                    k=k,
                    n_ants=len(ix),
                    bic=float(m.bic(z)),
                    smallest_group=int(sizes.min()),
                    minimum_group_size=max(4, int(np.ceil(0.1 * len(ix)))),
                    group_sizes=json.dumps(sizes.tolist()),
                    bootstrap_median_ari=np.nan,
                    bootstrap_p10_ari=np.nan,
                    split_minute_ari=np.nan,
                    split_minute_ants=int(repeat.sum()),
                    phenotype_pc_variance=float(t.pca.explained_variance_ratio_[0]),
                )
                if k > 1:
                    aris = []
                    for rep in range(bootstrap):
                        sample = rng.integers(len(ix), size=len(ix))
                        bt = transform(x[ix[sample]])
                        bm = mixture(
                            bt.transform(x[ix[sample]]),
                            k,
                            seed=SEED + rep + 1,
                            n_init=5,
                        )
                        aris.append(
                            adjusted_rand_score(labels, bm.predict(bt.transform(x[ix])))
                        )
                    row["bootstrap_median_ari"] = float(np.median(aris))
                    row["bootstrap_p10_ari"] = float(np.quantile(aris, 0.1))
                    em = mixture(ez, k)
                    om = mixture(oz, k)
                    row["split_minute_ari"] = float(
                        adjusted_rand_score(em.predict(ez), om.predict(oz))
                    )
                rows.append(row)
                print(
                    "K",
                    k,
                    "BIC",
                    round(row["bic"], 2),
                    "boot",
                    row["bootstrap_median_ari"],
                    "split",
                    row["split_minute_ari"],
                    flush=True,
                )
            table = pd.DataFrame(rows)
            chosen = select(table)
            table["selected"] = table.k.eq(chosen)
            candidates.append(table)
            m = fits[chosen]
            labels = m.predict(z)
            order = sorted(
                range(chosen),
                key=lambda c: np.median(x[ix, len(names) + 2][labels == c]),
            )
            remap = np.argsort(order)
            d1 = remap[labels]
            ok = q.iloc[ix].day2_eligible.to_numpy(bool)
            d2 = np.full(len(ix), -1, int)
            d2[ok] = remap[m.predict(t.transform(test[ix[ok]]))]
            independent = np.nan
            if chosen > 1:
                it = transform(test[ix[ok]])
                iz = it.transform(test[ix[ok]])
                im = mixture(iz, chosen)
                independent = float(adjusted_rand_score(d1[ok], im.predict(iz)))
            summaries.append(
                dict(
                    side=side,
                    family=family,
                    selected_k=chosen,
                    n_ants=len(ix),
                    day2_ants=int(ok.sum()),
                    day2_frozen_retention=(
                        float(np.mean(d1[ok] == d2[ok])) if chosen > 1 else np.nan
                    ),
                    day2_refit_ari=independent,
                    bic_improvement=float(
                        table.loc[table.k.eq(1), "bic"].iloc[0]
                        - table.loc[table.k.eq(chosen), "bic"].iloc[0]
                    ),
                )
            )
            models[f"{side}/{family}"] = dict(
                transform=t, mixture=m, remap=remap, indices=ix
            )
            for j, ai in enumerate(ix):
                base = q.iloc[ai].to_dict()
                assignments.append(
                    dict(
                        **base,
                        family=family,
                        cluster=int(d1[j]),
                        day2_cluster=int(d2[j]),
                        k=chosen,
                        posterior=float(m.predict_proba(z)[j].max()),
                    )
                )
                coords.append(
                    dict(
                        ant=base["ant"],
                        side=side,
                        family=family,
                        score=float(z[j, 0]),
                        cluster=int(d1[j]),
                    )
                )
            print("SELECTED", side, family, chosen, flush=True)
    pd.concat(candidates).to_csv(output / "k_selection.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "fit_summary.csv", index=False)
    pd.DataFrame(assignments).to_csv(output / "assignments.csv", index=False)
    pd.DataFrame(coords).to_csv(output / "phenotype_coordinates.csv", index=False)
    joblib.dump(models, output / "models.joblib")
    paths = [
        "eigenposture_measurements.npz",
        "landmark_pca.npz",
        "ant_profiles.npz",
        "assignments.csv",
        "models.joblib",
        "k_selection.csv",
    ]
    frozen = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        primary="joint",
        spatial_inputs_loaded=False,
        handcrafted_shape_descriptors_used=False,
        posture_rank=int(np.load(output / "landmark_pca.npz")["rank"]),
        bootstrap_repeats=bootstrap,
        selection_rule="K=1..4; multi-group fits must beat K=1 BIC, bootstrap median ARI >=0.8, independent even/odd-minute ARI >=0.6 and smallest group >=max(4,10% of ants); smallest K within two BIC units of best eligible fit, otherwise K=1.",
        limitations="Exploratory dataset already studied. Bootstrap phenotype stability conditions on landmark PCA; landmark-subspace bootstrap is reported separately. Day-2 phenotype refit uses the frozen day-1 landmark basis.",
        file_sha256={
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in paths
        },
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "UNSUPERVISED_FROZEN.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )


def evaluate(output, reference):
    frozen = json.loads((output / "UNSUPERVISED_FROZEN.json").read_text())
    for name, digest in frozen["file_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest, name
    spatial = pd.read_csv(reference).rename(columns={"TrackID": "track_id"})
    spatial["spatial_cluster"] = (
        spatial.cluster_id.astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    )
    a = pd.read_csv(output / "assignments.csv").merge(
        spatial[["side", "track_id", "spatial_cluster"]],
        on=["side", "track_id"],
        how="left",
        validate="many_to_one",
    )
    rows = []
    for (side, family), part in a.groupby(["side", "family"]):
        both = part.dropna(subset=["spatial_cluster"])
        table = pd.crosstab(both.cluster, both.spatial_cluster)
        ri, ci = linear_sum_assignment(-table.to_numpy())
        mapping = {int(table.index[i]): int(table.columns[j]) for i, j in zip(ri, ci)}
        accuracy = float(table.to_numpy()[ri, ci].sum() / len(both))
        mask = a.side.eq(side) & a.family.eq(family)
        a.loc[mask, "mapped_spatial_cluster"] = a.loc[mask, "cluster"].map(mapping)
        rows.append(
            dict(
                side=side,
                family=family,
                k=int(part.k.iloc[0]),
                n_eligible=len(part),
                n_compared=len(both),
                n_without_reference=len(part) - len(both),
                accuracy=accuracy,
                ari=float(adjusted_rand_score(both.spatial_cluster, both.cluster)),
                contingency=json.dumps(table.to_numpy().tolist()),
            )
        )
    a["agreement"] = np.where(
        a.spatial_cluster.notna(),
        a.spatial_cluster.eq(a.mapped_spatial_cluster),
        np.nan,
    )
    a.to_csv(output / "spatial_comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "spatial_metrics.csv", index=False)
    spatial.to_csv(output / "spatial_reference.csv", index=False)
    (output / "spatial_provenance.json").write_text(
        json.dumps(
            dict(
                path=str(reference),
                sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                evaluated_at=datetime.now(timezone.utc).isoformat(),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        pd.DataFrame(rows).drop(columns="contingency").to_string(index=False),
        flush=True,
    )


def diagnostics(output, reference):
    """Inspect rejected K=2 candidates; never replace selected assignments."""
    candidates = []
    sensitivity = []
    for label, folder in [
        ("90%", output),
        ("95%", output / "rank95"),
        ("100%", output / "full_rank"),
    ]:
        if not (folder / "UNSUPERVISED_FROZEN.json").exists():
            continue
        frozen = json.loads((folder / "UNSUPERVISED_FROZEN.json").read_text())
        for name, digest in frozen["file_sha256"].items():
            assert (
                hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
            ), name
        coords = pd.read_csv(folder / "phenotype_coordinates.csv")
        table = pd.read_csv(folder / "k_selection.csv")
        models = joblib.load(folder / "models.joblib")
        quality = pd.read_csv(folder / "all_ant_coverage.csv")
        with np.load(folder / "ant_profiles.npz") as data:
            heldout = data["heldout"]
        for side in ["left", "right"]:
            part = coords[coords.side.eq(side) & coords.family.eq("joint")]
            z = part.score.to_numpy()[:, None]
            m = mixture(z, 2)
            pred = m.predict(z)
            order = np.argsort(m.means_[:, 0])
            remap = np.argsort(order)
            for row, prediction in zip(part.itertuples(), remap[pred]):
                candidates.append(
                    dict(
                        variance_target=label,
                        ant=row.ant,
                        side=side,
                        candidate_cluster=int(prediction),
                    )
                )
            tab = table[table.side.eq(side) & table.family.eq("joint")]
            two = tab[tab.k.eq(2)].iloc[0]
            model = models[f"{side}/joint"]
            indices = model["indices"]
            ok = quality.iloc[indices].day2_eligible.to_numpy(bool)
            second = m.predict(model["transform"].transform(heldout[indices[ok]]))
            retention = float(np.mean(pred[ok] == second))
            sensitivity.append(
                dict(
                    variance_target=label,
                    side=side,
                    posture_rank=int(frozen["posture_rank"]),
                    selected_k=int(tab.loc[tab.selected, "k"].iloc[0]),
                    two_group_bic_improvement=float(
                        tab.loc[tab.k.eq(1), "bic"].iloc[0] - two.bic
                    ),
                    two_group_bootstrap_ari=two.bootstrap_median_ari,
                    two_group_split_ari=two.split_minute_ari,
                    candidate_k2_day2_retention=retention,
                    day2_ants=int(ok.sum()),
                )
            )
    candidate = pd.DataFrame(candidates)
    candidate.to_csv(output / "diagnostic_k2_assignments.csv", index=False)
    (output / "DIAGNOSTIC_K2_FROZEN.json").write_text(
        json.dumps(
            dict(
                created_at=datetime.now(timezone.utc).isoformat(),
                policy="Already examined dataset; these fixed K=2 diagnostics are not selected models and do not change assignments.",
                sha256=hashlib.sha256(
                    (output / "diagnostic_k2_assignments.csv").read_bytes()
                ).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    spatial = pd.read_csv(reference)
    spatial["ant"] = (
        spatial.side + ":" + spatial.TrackID.astype(int).map(lambda v: f"{v:03d}")
    )
    spatial["spatial_cluster"] = (
        spatial.cluster_id.astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    )
    candidate = candidate.merge(
        spatial[["ant", "spatial_cluster"]],
        on="ant",
        how="left",
        validate="many_to_one",
    )
    for row in sensitivity:
        part = candidate[
            candidate.variance_target.eq(row["variance_target"])
            & candidate.side.eq(row["side"])
        ].dropna(subset=["spatial_cluster"])
        table = pd.crosstab(part.candidate_cluster, part.spatial_cluster).to_numpy()
        ri, ci = linear_sum_assignment(-table)
        row["candidate_k2_accuracy"] = float(table[ri, ci].sum() / len(part))
        row["candidate_k2_ari"] = float(
            adjusted_rand_score(part.spatial_cluster, part.candidate_cluster)
        )
        row["n_compared"] = len(part)
    pd.DataFrame(sensitivity).to_csv(output / "rank_sensitivity.csv", index=False)
    candidate.to_csv(output / "diagnostic_k2_spatial_comparison.csv", index=False)
    print(pd.DataFrame(sensitivity).to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["fit", "evaluate", "diagnostics"], required=True)
    p.add_argument("--spatial-reference", type=Path)
    p.add_argument("--bootstrap", type=int, default=100)
    a = p.parse_args()
    if a.stage == "fit":
        fit(a.output, a.bootstrap)
    elif a.spatial_reference is None:
        p.error("Evaluation requires --spatial-reference")
    elif a.stage == "evaluate":
        evaluate(a.output, a.spatial_reference)
    else:
        diagnostics(a.output, a.spatial_reference)


if __name__ == "__main__":
    # Persist custom transformers under their importable module, not __main__.
    from analysis.eigenposture_analysis import main as cli_main

    cli_main()
