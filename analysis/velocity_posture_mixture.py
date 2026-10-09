"""Unequal-variance phenotype mixtures; spatial labels are never fit inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score

from analysis.velocity_posture_features import FEATURES, SEED
from analysis.velocity_posture_analysis import (
    PhenotypeTransform,
    BlockPhenotypeTransform,
    CommonPhenotypeTransform,
    summarize_hours,
    FAMILIES,
)


def mixture(z, k, seed=SEED, n_init=12, covariance="diag"):
    fits = [
        GaussianMixture(
            k,
            covariance_type=covariance,
            reg_covar=0.01,
            n_init=n_init,
            max_iter=500,
            random_state=seed,
        ).fit(z)
    ]
    if k > 1:
        for axis in range(z.shape[1]):
            groups = np.array_split(np.argsort(z[:, axis]), k)
            centers = np.stack([z[g].mean(axis=0) for g in groups])
            variances = np.stack([z[g].var(axis=0) + 0.01 for g in groups])
            precision = (
                1 / variances
                if covariance == "diag"
                else np.stack(
                    [
                        np.linalg.inv(
                            np.atleast_2d(np.cov(z[g], rowvar=False))
                            + np.eye(z.shape[1]) * 0.01
                        )
                        for g in groups
                    ]
                )
            )
            fits.append(
                GaussianMixture(
                    k,
                    covariance_type=covariance,
                    reg_covar=0.01,
                    n_init=1,
                    means_init=centers,
                    precisions_init=precision,
                    weights_init=np.array([len(g) / len(z) for g in groups]),
                    max_iter=500,
                    random_state=seed,
                ).fit(z)
            )
    good = [m for m in fits if m.converged_]
    if not good:
        raise RuntimeError("No converged Gaussian mixture")
    return max(good, key=lambda m: m.lower_bound_)


def select(table, minimum_bic_improvement=10):
    baseline = float(table.loc[table.k.eq(1), "bic"].iloc[0])
    allowed = table[
        (table.k >= 2)
        & (table.smallest_group >= table.minimum_group_size)
        & (table.bootstrap_ari_median >= 0.8)
        & (table.split_minute_ari >= 0.6)
        & (table.bic < baseline - minimum_bic_improvement)
    ]
    if allowed.empty:
        return 1
    return int(allowed.loc[allowed.bic <= allowed.bic.min() + 2, "k"].min())


def run(source, output, bootstrap=100, representation="common"):
    transformer = {
        "global": PhenotypeTransform,
        "block": BlockPhenotypeTransform,
        "common": CommonPhenotypeTransform,
    }[representation]
    covariance = "diag" if representation == "global" else "full"

    def fit_mixture(z, k, **kwargs):
        return mixture(z, k, covariance=covariance, **kwargs)

    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "physical_measurements.npz",
        "all_ant_coverage.csv",
        "measurement_manifest.json",
    ):
        target = output / name
        if not target.exists():
            target.symlink_to((source / name).resolve())
    qc = pd.read_csv(source / "all_ant_coverage.csv")
    with np.load(source / "physical_measurements.npz") as f:
        hourly = f["hourly"]
        even = f["hourly_even"]
        odd = f["hourly_odd"]
    x = summarize_hours(hourly[:, :24])
    test = summarize_hours(hourly[:, 24:])
    even_x = summarize_hours(even[:, :24])
    odd_x = summarize_hours(odd[:, :24])
    if not (output / "ant_profiles.npz").exists():
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
    replicate_ok = (np.isfinite(even[:, :24]).sum(axis=1).min(axis=1) >= 8) & (
        np.isfinite(odd[:, :24]).sum(axis=1).min(axis=1) >= 8
    )
    all_candidates = []
    assignments = []
    summaries = []
    models = {}
    coordinates = []
    for side in ("left", "right"):
        ix = np.flatnonzero(qc.side.eq(side) & qc.day1_eligible)
        for family, groups in FAMILIES.items():
            print("MIXTURE", side, family, len(ix), flush=True)
            rng = np.random.default_rng(SEED)
            xt = x[ix]
            rep = np.flatnonzero(replicate_ok[ix])
            t = transformer().fit(xt, groups)
            z = t.transform(xt)
            et = transformer().fit(even_x[ix[rep]], groups)
            ez = et.transform(even_x[ix[rep]])
            ot = transformer().fit(odd_x[ix[rep]], groups)
            oz = ot.transform(odd_x[ix[rep]])
            fits = {}
            rows = []
            for k in range(1, 5):
                m = fit_mixture(z, k)
                labels = m.predict(z)
                fits[k] = m
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
                    bic=float(m.bic(z)),
                    aic=float(m.aic(z)),
                    silhouette=(
                        float(silhouette_score(z, labels))
                        if k > 1 and len(np.unique(labels)) > 1
                        else np.nan
                    ),
                    bootstrap_ari_median=np.nan,
                    bootstrap_ari_p10=np.nan,
                    split_minute_ari=np.nan,
                )
                if k > 1:
                    ari = []
                    for repeat in range(bootstrap):
                        sample = rng.integers(len(xt), size=len(xt))
                        bt = transformer().fit(xt[sample], groups)
                        bm = fit_mixture(
                            bt.transform(xt[sample]),
                            k,
                            seed=SEED + repeat + 1,
                            n_init=4,
                        )
                        ari.append(
                            adjusted_rand_score(labels, bm.predict(bt.transform(xt)))
                        )
                    row["bootstrap_ari_median"] = float(np.median(ari))
                    row["bootstrap_ari_p10"] = float(np.quantile(ari, 0.1))
                    em = fit_mixture(ez, k)
                    om = fit_mixture(oz, k)
                    row["split_minute_ari"] = float(
                        adjusted_rand_score(em.predict(ez), om.predict(oz))
                    )
                rows.append(row)
                print(
                    "K",
                    k,
                    "BIC",
                    round(row["bic"], 2),
                    "bootstrap",
                    row["bootstrap_ari_median"],
                    flush=True,
                )
            table = pd.DataFrame(rows)
            chosen = select(
                table, minimum_bic_improvement=0 if representation == "common" else 10
            )
            table["selected"] = table.k.eq(chosen)
            all_candidates.append(table)
            m = fits[chosen]
            labels = m.predict(z)
            speed = x[ix, len(FEATURES) + 2]
            order = sorted(range(chosen), key=lambda c: np.median(speed[labels == c]))
            remap = np.argsort(order)
            models[f"{side}/{family}"] = dict(
                transform=t, model=m, remap=remap, indices=ix
            )
            d1 = remap[labels]
            d2ok = qc.iloc[ix].day2_eligible.to_numpy(bool)
            d2 = np.full(len(ix), -1, int)
            d2[d2ok] = remap[m.predict(t.transform(test[ix[d2ok]]))]
            independent_ari = np.nan
            if chosen > 1 and d2ok.sum() >= max(12, 2 * chosen):
                it = transformer().fit(test[ix[d2ok]], groups)
                iz = it.transform(test[ix[d2ok]])
                im = fit_mixture(iz, chosen)
                independent_ari = float(adjusted_rand_score(d1[d2ok], im.predict(iz)))
            summaries.append(
                dict(
                    side=side,
                    family=family,
                    selected_k=chosen,
                    n_ants=len(ix),
                    day2_ants=int(d2ok.sum()),
                    day2_frozen_retention=(
                        float(np.mean(d2[d2ok] == d1[d2ok])) if chosen > 1 else np.nan
                    ),
                    day2_independent_ari=independent_ari,
                    selected_bic=float(table.loc[table.k.eq(chosen), "bic"].iloc[0]),
                    delta_bic_vs_one=float(
                        table.loc[table.k.eq(1), "bic"].iloc[0]
                        - table.loc[table.k.eq(chosen), "bic"].iloc[0]
                    ),
                )
            )
            confidence = m.predict_proba(z).max(axis=1)
            for pos, ant_ix in enumerate(ix):
                base = qc.iloc[ant_ix].to_dict()
                assignments.append(
                    dict(
                        **base,
                        family=family,
                        activity_cluster=int(d1[pos]),
                        day2_cluster=int(d2[pos]),
                        selected_k=chosen,
                        posterior_confidence=float(confidence[pos]),
                    )
                )
                coordinates.append(
                    dict(
                        ant=base["ant"],
                        side=side,
                        family=family,
                        pc1=float(z[pos, 0]),
                        pc2=float(z[pos, 1]) if z.shape[1] > 1 else 0.0,
                        activity_cluster=int(d1[pos]),
                    )
                )
            print("SELECTED", side, family, chosen, flush=True)
    pd.concat(all_candidates, ignore_index=True).to_csv(
        output / "k_selection.csv", index=False
    )
    pd.DataFrame(assignments).to_csv(output / "activity_assignments.csv", index=False)
    pd.DataFrame(coordinates).to_csv(output / "activity_coordinates.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "unsupervised_summary.csv", index=False)
    joblib.dump(models, output / "phenotype_models.joblib")
    frozen = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        primary_family="joint",
        algorithm=f"Gaussian mixture; unequal {covariance} component covariances",
        representation=representation,
        groups=FAMILIES,
        reg_covar=0.01,
        bootstrap_repeats=bootstrap,
        selection_rule=f'Among K=2..4 with BIC improvement >{0 if representation=="common" else 10} over K=1, bootstrap median ARI >=0.8, split-minute ARI >=0.6, and minimum group size >=max(4,10% of ants), choose smallest K within 2 of best BIC. Otherwise K=1.',
        spatial_labels_loaded=False,
        day2_used_for_selection=False,
        rationale="One PCA axis of standardized hourly physical profiles captures the dominant shared velocity-posture variation, with fewer mixture parameters than a multidimensional fit. The common representation weights standardized input variables equally before learning PCA loadings. Earlier higher-dimensional representations are retained as exploratory history; their block weighting and stronger BIC gate differ from this final model.",
        exploratory_note="The dataset and the first model have already been examined. This is an exploratory unsupervised refinement, not a prospectively registered blind validation.",
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=100)
    p.add_argument(
        "--representation", choices=["global", "block", "common"], default="common"
    )
    a = p.parse_args()
    run(a.source, a.output, a.bootstrap, a.representation)


if __name__ == "__main__":
    main()
