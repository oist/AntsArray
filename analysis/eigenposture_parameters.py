"""Spatial-label-free parameter audit of landmark-PC and velocity phenotypes."""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

SEED = 7241010


@dataclass(frozen=True)
class Config:
    bin_minutes: int = 60
    summary: str = "quartiles"
    scaling: str = "standard"
    dimensions: int = 1
    covariance: str = "full"

    @property
    def key(self):
        return f"b{self.bin_minutes}_{self.summary}_{self.scaling}_d{self.dimensions}_{self.covariance}"


def aggregate(minute, cfg, mask=None):
    """Equal observed-minute means, then a distribution or ordered bin profile."""
    b = cfg.bin_minutes
    data = minute.copy()
    if mask is not None:
        data[:, ~mask] = np.nan
    bins = data.reshape(len(data), 1440 // b, b, data.shape[-1])
    count = np.isfinite(bins).sum(axis=2)
    threshold = max(1, int(np.ceil(b / 12 * (1 if mask is None else np.mean(mask)))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(bins, axis=2)
        means[count < threshold] = np.nan
        if cfg.summary == "quartiles":
            values = np.nanquantile(means, [0.25, 0.5, 0.75], axis=1).transpose(1, 0, 2)
        elif cfg.summary == "mean_sd":
            values = np.stack(
                [np.nanmean(means, axis=1), np.nanstd(means, axis=1)], axis=1
            )
        elif cfg.summary == "concatenated":
            values = means
        else:
            raise ValueError(cfg.summary)
    return values.reshape(len(data), -1)


def nonlinear(x, names, summary):
    z = x.copy()
    f = len(names)
    for j in range(z.shape[1]):
        name = names[j % f]
        is_sd = summary == "mean_sd" and j >= f
        if name in ("forward_mean", "lateral_mean"):
            z[:, j] = (
                np.log1p(np.maximum(z[:, j], 0) / 0.1)
                if is_sd
                else np.arcsinh(z[:, j] / 0.1)
            )
        elif name in ("forward_rms", "lateral_rms"):
            z[:, j] = np.log1p(np.maximum(z[:, j], 0) / 0.1)
        elif name.endswith("_rate"):
            z[:, j] = np.log1p(np.maximum(z[:, j], 0))
    return z


class Representation:
    def __init__(self, cfg, names, families):
        self.cfg, self.names, self.families = cfg, names, families

    def fit(self, x):
        raw = nonlinear(x, self.names, self.cfg.summary)
        # No missingness indicators; imputation is fitted on training ants only.
        self.keep = np.isfinite(raw).sum(axis=0) >= max(4, int(0.5 * len(raw)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.fill = np.nanmedian(raw[:, self.keep], axis=0)
        raw = np.where(np.isfinite(raw[:, self.keep]), raw[:, self.keep], self.fill)
        self.center = raw.mean(axis=0)
        std = raw.std(axis=0)
        if self.cfg.scaling == "standard":
            self.scale = np.maximum(std, 1e-6)
        else:
            # One scale per feature family, retaining relative physical mode amplitudes.
            fam = np.tile(self.families, x.shape[1] // len(self.names))[self.keep]
            self.scale = np.ones(len(std))
            for value in np.unique(fam):
                ix = fam == value
                self.scale[ix] = max(np.sqrt(np.sum(std[ix] ** 2)), 1e-6)
        self.pca = PCA(self.cfg.dimensions, svd_solver="full").fit(
            (raw - self.center) / self.scale
        )
        # Normalize score variance globally: covariance regularization must not depend on units.
        self.score_scale = np.sqrt(self.pca.explained_variance_[0])
        return self

    def transform(self, x):
        raw = nonlinear(x, self.names, self.cfg.summary)[:, self.keep]
        raw = np.where(np.isfinite(raw), raw, self.fill)
        return self.pca.transform((raw - self.center) / self.scale) / self.score_scale


def cluster(z, k, cfg, seed=SEED, n_init=5):
    models = []
    for means in [None] + (
        [np.stack([z[g].mean(axis=0) for g in np.array_split(np.argsort(z[:, 0]), k)])]
        if k > 1
        else []
    ):
        m = GaussianMixture(
            k,
            covariance_type=cfg.covariance,
            n_init=n_init if means is None else 1,
            means_init=means,
            reg_covar=0.001,
            max_iter=500,
            random_state=seed,
        ).fit(z)
        if m.converged_:
            models.append(m)
    if not models:
        raise RuntimeError("No converged mixture")
    return max(models, key=lambda m: m.lower_bound_)


def fit_model(x, cfg, names, families, k=2, seed=SEED):
    t = Representation(cfg, names, families).fit(x)
    z = t.transform(x)
    m = cluster(z, k, cfg, seed)
    return t, m, m.predict(z)


def bootstrap(x, cfg, names, families, labels, repeats, seed, k=2, fixed=None):
    rng = np.random.default_rng(seed)
    ari, predictions = [], []
    for rep in range(repeats):
        ix = rng.integers(len(x), size=len(x))
        if fixed is None:
            t, m, _ = fit_model(x[ix], cfg, names, families, k, seed + rep)
        else:
            t = fixed
            m = cluster(t.transform(x[ix]), k, cfg, seed + rep)
        y = m.predict(t.transform(x))
        ari.append(adjusted_rand_score(labels, y))
        predictions.append(y)
    return np.array(ari), np.array(predictions)


def load_source(source):
    q = pd.read_csv(source / "all_ant_coverage.csv")
    with np.load(source / "eigenposture_measurements.npz") as z:
        return q, z["minute"], z["feature_names"], z["families"]


def screen_one(task):
    source, cfg, repeats = task
    q, minute, names, families = load_source(source)
    x = aggregate(minute[:, :1440], cfg)
    splits = {}
    time = np.arange(1440)
    for kind, block in [("minute", 1), ("block30", 30)]:
        splits[kind] = [
            aggregate(minute[:, :1440], cfg, (time // block) % 2 == half)
            for half in [0, 1]
        ]
    rows = []
    for side in ["left", "right"]:
        ix = np.flatnonzero(q.side.eq(side) & q.day1_eligible)
        t, m, labels = fit_model(x[ix], cfg, names, families)
        z = t.transform(x[ix])
        aris, _ = bootstrap(x[ix], cfg, names, families, labels, repeats, SEED + 1000)
        row = dict(
            config=cfg.key,
            side=side,
            **asdict(cfg),
            n=len(ix),
            bic_gain=cluster(z, 1, cfg).bic(z) - m.bic(z),
            bootstrap_median=np.median(aris),
            bootstrap_p10=np.quantile(aris, 0.1),
            smallest_group=np.bincount(labels, minlength=2).min(),
            silhouette=(
                silhouette_score(z, labels) if len(np.unique(labels)) == 2 else np.nan
            ),
            explained=t.pca.explained_variance_ratio_.sum(),
        )
        for kind, parts in splits.items():
            ys = [fit_model(p[ix], cfg, names, families)[2] for p in parts]
            row[f"{kind}_ari"] = adjusted_rand_score(*ys)
        rows.append(row)
    return rows


def configurations():
    out = []
    for b in [1, 5, 15, 30, 60, 120, 240]:
        summaries = ["quartiles", "mean_sd"] + (["concatenated"] if b >= 15 else [])
        for summary in summaries:
            for scale in ["standard", "family"]:
                for d in [1, 2]:
                    for covariance in ["full", "tied"]:
                        out.append(Config(b, summary, scale, d, covariance))
    return out


def rank_screen(table):
    rows = []
    for key, group in table.groupby("config"):
        row = {"config": key}
        for col in [
            "bic_gain",
            "bootstrap_median",
            "bootstrap_p10",
            "minute_ari",
            "block30_ari",
            "smallest_group",
        ]:
            row[f"min_{col}"] = group[col].min()
        row["admissible"] = row["min_smallest_group"] >= 5 and row["min_bic_gain"] > 0
        row["robustness"] = np.mean(
            [row["min_bootstrap_median"], row["min_minute_ari"], row["min_block30_ari"]]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["admissible", "robustness", "min_bootstrap_p10"], ascending=False
    )


def screen(source, output, workers, repeats):
    output.mkdir(parents=True, exist_ok=True)
    configs = configurations()
    protocol = dict(
        source=str(source),
        source_sha256=hashlib.sha256(
            (source / "eigenposture_measurements.npz").read_bytes()
        ).hexdigest(),
        seed=SEED,
        bootstrap_repeats=repeats,
        configurations=[asdict(c) for c in configs],
        ranking="Require groups >=5 and positive K2-vs-K1 BIC gain in both colonies. Rank mean of worst-colony bootstrap median, alternating-minute ARI, alternating-30-minute ARI; tie break bootstrap p10.",
        spatial_labels="Never loaded by screen or validation; evaluate only after freeze.",
        caveats="Exploratory tuning on a previously examined dataset. Day2 already examined in earlier analysis; not a fresh prospective holdout.",
        cohort="Same 41 left +45 right day1-eligible ants for every configuration.",
        posture_rank=4,
        window_seconds=2.5,
        window_sampling="one clip per minute",
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "screen_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(
            pool.map(screen_one, [(source, c, repeats) for c in configs])
        ):
            results.extend(result)
            if i % 10 == 0:
                print("SCREEN", i + 1, "/", len(configs), flush=True)
            pd.DataFrame(results).to_csv(output / "screen.csv", index=False)
    ranked = rank_screen(pd.DataFrame(results))
    ranked.to_csv(output / "screen_ranked.csv", index=False)
    print(ranked.head(20).to_string(index=False), flush=True)


def validation_one(task):
    source, cfg, repeats = task
    q, minute, names, families = load_source(source)
    x = aggregate(minute[:, :1440], cfg)
    output, details = [], []
    for side in ["left", "right"]:
        ix = np.flatnonzero(q.side.eq(side) & q.day1_eligible)
        t, m, labels = fit_model(x[ix], cfg, names, families)
        z = t.transform(x[ix])
        aris, prediction = bootstrap(
            x[ix], cfg, names, families, labels, repeats, SEED + 30000
        )
        fixed, _ = bootstrap(
            x[ix], cfg, names, families, labels, repeats, SEED + 30000, fixed=t
        )
        loo = []
        for omit in range(len(ix)):
            subset = np.arange(len(ix)) != omit
            lt, lm, _ = fit_model(x[ix[subset]], cfg, names, families)
            loo.append(adjusted_rand_score(labels, lm.predict(lt.transform(x[ix]))))
        temporal = []
        rng = np.random.default_rng(SEED + 40000)
        for rep in range(30):
            # One randomly assigned half-hour in each hour to each disjoint split.
            choose = rng.integers(2, size=24)
            mask = np.repeat(np.column_stack([choose == 0, choose == 1]).ravel(), 30)
            ys = []
            for part in [mask, ~mask]:
                sx = aggregate(minute[ix, :1440], cfg, part)
                ys.append(fit_model(sx, cfg, names, families)[2])
            temporal.append(adjusted_rand_score(*ys))
        sizes = np.bincount(labels, minlength=2)
        row = dict(
            config=cfg.key,
            side=side,
            **asdict(cfg),
            n=len(ix),
            bic_gain=cluster(z, 1, cfg).bic(z) - m.bic(z),
            smallest_group=sizes.min(),
            group_sizes=json.dumps(sizes.tolist()),
            bootstrap_median=np.median(aris),
            bootstrap_p10=np.quantile(aris, 0.1),
            bootstrap_fixed_median=np.median(fixed),
            bootstrap_fixed_p10=np.quantile(fixed, 0.1),
            loo_median=np.median(loo),
            loo_min=np.min(loo),
            temporal_median=np.median(temporal),
            temporal_p10=np.quantile(temporal, 0.1),
            explained=t.pca.explained_variance_ratio_.sum(),
        )
        output.append(row)
        for kind, values in [
            ("bootstrap", aris),
            ("bootstrap_fixed", fixed),
            ("leave_one_out", loo),
            ("time_split", temporal),
        ]:
            details.extend(
                dict(config=cfg.key, side=side, kind=kind, repeat=i, ari=a)
                for i, a in enumerate(values)
            )
    return output, details


def validate(source, output, workers, repeats):
    ranked = pd.read_csv(output / "screen_ranked.csv")
    keys = set(ranked.head(8).config)
    keys.update(Config(b).key for b in [1, 5, 15, 30, 60, 120, 240])
    configs = [c for c in configurations() if c.key in keys]
    protocol = dict(
        configurations=[asdict(c) for c in configs],
        bootstrap_repeats=repeats,
        temporal_splits="30 random disjoint half-hour splits, paired within each hour",
        selection="Require both colonies: min group >=5, BIC gain >0, bootstrap median >=0.8, temporal median >=0.6. Rank average of worst-colony bootstrap p10 and temporal p10. Tie break worst-colony bootstrap median, then configuration key.",
        selection_independent_of="spatial labels, day2 behavior",
    )
    (output / "validation_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    rows, details = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (r, d) in enumerate(
            pool.map(validation_one, [(source, c, repeats) for c in configs])
        ):
            rows.extend(r)
            details.extend(d)
            pd.DataFrame(rows).to_csv(output / "validation.csv", index=False)
            pd.DataFrame(details).to_csv(
                output / "validation_replicates.csv", index=False
            )
            print("VALIDATE", i + 1, "/", len(configs), r[0]["config"], flush=True)
    table = pd.DataFrame(rows)
    ranking = []
    for key, g in table.groupby("config"):
        v = {
            col: g[col].min()
            for col in [
                "bic_gain",
                "smallest_group",
                "bootstrap_median",
                "bootstrap_p10",
                "temporal_median",
                "temporal_p10",
            ]
        }
        ranking.append(
            dict(
                config=key,
                **v,
                admissible=v["bic_gain"] > 0
                and v["smallest_group"] >= 5
                and v["bootstrap_median"] >= 0.8
                and v["temporal_median"] >= 0.6,
                score=(v["bootstrap_p10"] + v["temporal_p10"]) / 2,
            )
        )
    ranked = pd.DataFrame(ranking).sort_values(
        ["admissible", "score", "bootstrap_median", "config"],
        ascending=[False, False, False, True],
    )
    ranked.to_csv(output / "validation_ranked.csv", index=False)
    print(ranked.to_string(index=False), flush=True)
    chosen = ranked[ranked.admissible]
    if chosen.empty:
        raise RuntimeError(
            "No shared configuration passed validation; report the failed search without selecting one."
        )
    cfg = next(c for c in configs if c.key == chosen.iloc[0].config)
    (output / "SELECTED_FROZEN.json").write_text(
        json.dumps(
            dict(
                configuration=asdict(cfg),
                key=cfg.key,
                selection="validation_protocol.json",
                validation_sha256=hashlib.sha256(
                    (output / "validation.csv").read_bytes()
                ).hexdigest(),
                spatial_labels_loaded=False,
                day2_loaded_for_selection=False,
            ),
            indent=2,
        )
        + "\n"
    )


def match_labels(reference, labels):
    rvals = np.unique(reference)
    lvals = np.unique(labels)
    counts = np.array(
        [[(reference == r)[labels == l].sum() for l in lvals] for r in rvals]
    )
    row, col = linear_sum_assignment(-counts)
    return counts, counts[row, col].sum() / len(labels)


def finish(source, output):
    cfg = Config(
        **json.loads((output / "SELECTED_FROZEN.json").read_text())["configuration"]
    )
    q, minute, names, families = load_source(source)
    x = aggregate(minute[:, :1440], cfg)
    models, assignments, ks, replicates, loadings, summaries = {}, [], [], [], [], []
    for side in ["left", "right"]:
        ix = np.flatnonzero(q.side.eq(side) & q.day1_eligible)
        t, m, labels = fit_model(x[ix], cfg, names, families)
        z = t.transform(x[ix])
        kmodels = {2: m}
        for k in [1, 2, 3, 4]:
            km = m if k == 2 else cluster(z, k, cfg, n_init=20)
            kmodels[k] = km
            y = km.predict(z)
            sizes = np.bincount(y, minlength=k)
            aris, yp = (
                bootstrap(
                    x[ix],
                    cfg,
                    names,
                    families,
                    y,
                    500 if k == 2 else 100,
                    SEED + 60000,
                    k=k,
                )
                if k > 1
                else (np.array([1]), None)
            )
            temporal = []
            for block in [1, 5, 15, 30, 60, 120]:
                ys = []
                for half in [0, 1]:
                    p = aggregate(
                        minute[ix, :1440], cfg, (np.arange(1440) // block) % 2 == half
                    )
                    ys.append(fit_model(p, cfg, names, families, k=k)[2])
                temporal.append(adjusted_rand_score(*ys))
            ks.append(
                dict(
                    side=side,
                    k=k,
                    bic=km.bic(z),
                    smallest_group=sizes.min(),
                    group_sizes=json.dumps(sizes.tolist()),
                    bootstrap_median=np.median(aris),
                    bootstrap_p10=np.quantile(aris, 0.1),
                    temporal_median=np.median(temporal),
                    temporal_min=np.min(temporal),
                )
            )
            for rep, a in enumerate(aris):
                replicates.append(
                    dict(side=side, k=k, kind="bootstrap", repeat=rep, ari=a)
                )
            for block, a in zip([1, 5, 15, 30, 60, 120], temporal):
                replicates.append(
                    dict(side=side, k=k, kind="time_block", repeat=block, ari=a)
                )
            if k == 2:
                co = (yp[:, :, None] == yp[:, None, :]).mean(axis=0)
                np.save(output / f"{side}_coassignment.npy", co)
                # Align each resampled partition to the complete-data partition.
                aligned = []
                for yy in yp:
                    aligned.append(1 - yy if np.mean(yy == labels) < 0.5 else yy)
                confidence = (np.asarray(aligned) == labels).mean(axis=0)
        table = pd.DataFrame([v for v in ks if v["side"] == side])
        admissible = table[
            (table.k > 1)
            & (table.bic < table.loc[table.k == 1, "bic"].iloc[0])
            & (table.smallest_group >= 5)
            & (table.bootstrap_median >= 0.8)
            & (table.temporal_median >= 0.6)
        ]
        chosen = (
            1
            if admissible.empty
            else int(admissible[admissible.bic <= admissible.bic.min() + 2].k.min())
        )
        order = np.argsort(
            [np.nanmedian(minute[ix[labels == v], :1440, 2]) for v in [0, 1]]
        )
        remap = np.argsort(order)
        labels = remap[labels]
        models[side] = dict(
            transform=t, mixture=m, remap=remap, indices=ix, k_selected=chosen
        )
        for j, i in enumerate(ix):
            assignments.append(
                dict(
                    ant=q.iloc[i].ant,
                    side=side,
                    track_id=int(q.iloc[i].track_id),
                    track_name=q.iloc[i].track_name,
                    group=int(labels[j]),
                    score=float(z[j, 0]),
                    bootstrap_agreement=float(confidence[j]),
                    day2_eligible=bool(q.iloc[i].day2_eligible),
                )
            )
        stats = {"mean_sd": ["mean", "sd"], "quartiles": ["q25", "median", "q75"]}.get(
            cfg.summary, [f"bin{i}" for i in range(1440 // cfg.bin_minutes)]
        )
        fullnames = [f"{stat}:{name}" for stat in stats for name in names]
        for j, ind in enumerate(np.flatnonzero(t.keep)):
            loadings.append(
                dict(
                    side=side,
                    feature=fullnames[ind],
                    family=np.tile(families, len(stats))[ind],
                    coefficient=t.pca.components_[0, j],
                    scale=t.scale[j],
                )
            )
        summaries.append(
            dict(
                side=side,
                n=len(ix),
                k_selected=chosen,
                profile_variance=t.pca.explained_variance_ratio_[0],
            )
        )
    pd.DataFrame(ks).to_csv(output / "k_selection.csv", index=False)
    pd.DataFrame(replicates).to_csv(output / "final_replicates.csv", index=False)
    pd.DataFrame(assignments).to_csv(output / "assignments.csv", index=False)
    pd.DataFrame(loadings).to_csv(output / "profile_loadings.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "fit_summary.csv", index=False)
    joblib.dump(models, output / "models.joblib")
    frozen = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            output / "models.joblib",
            output / "assignments.csv",
            output / "k_selection.csv",
            output / "SELECTED_FROZEN.json",
        ]
    }
    (output / "MODELS_FROZEN.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print("FROZEN", cfg.key, flush=True)
    # Post-selection evaluation only: day2 and the spatial reference enter here.
    day2 = aggregate(minute[:, 1440:], cfg)
    reference = pd.read_csv(source / "spatial_reference.csv").rename(
        columns={"TrackID": "track_id"}
    )
    reference["spatial_group"] = (
        reference.cluster_id.astype(str).str.extract(r"(\d+)$").astype(int)
    )
    a = pd.DataFrame(assignments).merge(
        reference[["side", "track_id", "spatial_group"]],
        on=["side", "track_id"],
        validate="one_to_one",
    )
    evaluation = []
    for side in ["left", "right"]:
        mod = models[side]
        ix = mod["indices"]
        ss = a.side.eq(side)
        ok = q.iloc[ix].day2_eligible.to_numpy(bool)
        d1 = a.loc[ss, "group"].to_numpy()
        d2 = mod["remap"][
            mod["mixture"].predict(mod["transform"].transform(day2[ix[ok]]))
        ]
        a.loc[ss, "day2_group"] = -1
        a.loc[a.index[ss][ok], "day2_group"] = d2
        _, _, independent = fit_model(day2[ix[ok]], cfg, names, families)
        _, accuracy = match_labels(a.loc[ss, "spatial_group"].to_numpy(), d1)
        evaluation.append(
            dict(
                side=side,
                n=len(ix),
                spatial_accuracy=accuracy,
                spatial_correct=round(accuracy * len(ix)),
                spatial_ari=adjusted_rand_score(a.loc[ss, "spatial_group"], d1),
                day2_n=ok.sum(),
                day2_retained=int((d1[ok] == d2).sum()),
                day2_retention=np.mean(d1[ok] == d2),
                day2_frozen_ari=adjusted_rand_score(d1[ok], d2),
                day2_refit_ari=adjusted_rand_score(d1[ok], independent),
            )
        )
    a.to_csv(output / "assignments_evaluated.csv", index=False)
    pd.DataFrame(evaluation).to_csv(output / "evaluation.csv", index=False)
    print(pd.DataFrame(evaluation).to_string(index=False), flush=True)
    # Evaluate the complete frozen screen to show the stability/agreement tradeoff,
    # never to change the selected configuration.
    selected = {side: a[a.side.eq(side)].group.to_numpy() for side in ["left", "right"]}
    comparison = []
    for c in configurations():
        cx = aggregate(minute[:, :1440], c)
        for side in ["left", "right"]:
            ix = np.flatnonzero(q.side.eq(side) & q.day1_eligible)
            _, _, y = fit_model(cx[ix], c, names, families)
            spatial = a[a.side.eq(side)].spatial_group.to_numpy()
            _, acc = match_labels(spatial, y)
            comparison.append(
                dict(
                    config=c.key,
                    side=side,
                    spatial_accuracy=acc,
                    spatial_ari=adjusted_rand_score(spatial, y),
                    selected_ari=adjusted_rand_score(selected[side], y),
                )
            )
    pd.DataFrame(comparison).to_csv(output / "screen_posthoc_spatial.csv", index=False)
    # Physical feature differences remain descriptive, with ants as replicates.
    rows = []
    antvalues = np.nanmedian(minute[:, :1440], axis=1)
    for side in ["left", "right"]:
        ix = models[side]["indices"]
        groups = selected[side]
        for j, name in enumerate(names):
            for g in [0, 1]:
                v = antvalues[ix[groups == g], j]
                rows.append(
                    dict(
                        side=side,
                        group=g,
                        feature=name,
                        n=len(v),
                        median=np.nanmedian(v),
                        q25=np.nanquantile(v, 0.25),
                        q75=np.nanquantile(v, 0.75),
                    )
                )
    pd.DataFrame(rows).to_csv(output / "physical_features.csv", index=False)


def baseline_diagnosis(source, output):
    from analysis.eigenposture_analysis import ProfileAxis, profiles, mixture

    q, minute, names, families = load_source(source)
    with np.load(source / "eigenposture_measurements.npz") as z:
        x = profiles(z["hourly"][:, :24])
    rows, replicates, parameters = [], [], []
    for side in ["left", "right"]:
        ix = np.flatnonzero(q.side.eq(side) & q.day1_eligible)

        def transform(xx):
            return ProfileAxis().fit(
                xx, names, families, ("velocity", "posture", "dynamics")
            )

        t = transform(x[ix])
        z = t.transform(x[ix])
        old = mixture(z, 2)
        baseline = old.predict(z)
        for covariance in ["full", "tied"]:
            m = (
                old
                if covariance == "full"
                else GaussianMixture(
                    2,
                    covariance_type="tied",
                    n_init=20,
                    reg_covar=0.01,
                    max_iter=500,
                    random_state=SEED,
                ).fit(z)
            )
            labels = m.predict(z)
            pars = dict(
                side=side,
                covariance=covariance,
                means=json.dumps(m.means_.ravel().tolist()),
                variances=json.dumps(m.covariances_.ravel().tolist()),
                weights=json.dumps(m.weights_.tolist()),
                bic=m.bic(z),
            )
            parameters.append(pars)
            for fixed in [False, True]:
                rng = np.random.default_rng(SEED + 80000)
                aris = []
                for rep in range(300):
                    j = rng.integers(len(ix), size=len(ix))
                    bt = t if fixed else transform(x[ix[j]])
                    bz = bt.transform(x[ix[j]])
                    bm = (
                        mixture(bz, 2, seed=SEED + rep, n_init=5)
                        if covariance == "full"
                        else GaussianMixture(
                            2,
                            covariance_type="tied",
                            n_init=5,
                            reg_covar=0.01,
                            max_iter=500,
                            random_state=SEED + rep,
                        ).fit(bz)
                    )
                    ari = adjusted_rand_score(labels, bm.predict(bt.transform(x[ix])))
                    aris.append(ari)
                    replicates.append(
                        dict(
                            side=side,
                            covariance=covariance,
                            fixed_representation=fixed,
                            repeat=rep,
                            ari=ari,
                        )
                    )
                rows.append(
                    dict(
                        side=side,
                        covariance=covariance,
                        fixed_representation=fixed,
                        bootstrap_median=np.median(aris),
                        bootstrap_p10=np.quantile(aris, 0.1),
                        agreement_with_original=adjusted_rand_score(baseline, labels),
                    )
                )
        print("BASELINE", side, flush=True)
    pd.DataFrame(rows).to_csv(output / "baseline_diagnosis.csv", index=False)
    pd.DataFrame(parameters).to_csv(
        output / "baseline_mixture_parameters.csv", index=False
    )
    pd.DataFrame(replicates).to_csv(output / "baseline_replicates.csv", index=False)


def controls_one(task):
    source, cfg, allowed, repeats = task
    q, minute, names, families = load_source(source)
    keep = np.isin(families, allowed)
    minute = minute[:, :, keep]
    names = names[keep]
    families = families[keep]
    x = aggregate(minute[:, :1440], cfg)
    rows = []
    for side in ["left", "right"]:
        ix = np.flatnonzero(q.side.eq(side) & q.day1_eligible)
        t, m, y = fit_model(x[ix], cfg, names, families)
        z = t.transform(x[ix])
        aris, _ = bootstrap(x[ix], cfg, names, families, y, repeats, SEED + 90000)
        rows.append(
            dict(
                config=cfg.key,
                families="+".join(allowed),
                side=side,
                bin_minutes=cfg.bin_minutes,
                bootstrap_median=np.median(aris),
                bootstrap_p10=np.quantile(aris, 0.1),
                bic_gain=cluster(z, 1, cfg).bic(z) - m.bic(z),
                smallest_group=np.bincount(y, minlength=2).min(),
            )
        )
    return rows


def controls(source, output, workers, repeats):
    cfg = Config(
        **json.loads((output / "SELECTED_FROZEN.json").read_text())["configuration"]
    )
    tasks = [
        (
            source,
            Config(**(asdict(cfg) | {"bin_minutes": b})),
            ["velocity", "posture", "dynamics"],
            repeats,
        )
        for b in [1, 5, 15, 30, 60, 120, 240]
    ]
    tasks += [
        (source, cfg, f, repeats)
        for f in [
            ["velocity"],
            ["posture"],
            ["dynamics"],
            ["posture", "dynamics"],
            ["velocity", "posture"],
            ["velocity", "dynamics"],
        ]
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(controls_one, tasks):
            rows.extend(r)
    pd.DataFrame(rows).to_csv(output / "controls.csv", index=False)
    q, minute, names, families = load_source(source)
    x = aggregate(minute[:, :1440], cfg)
    models = joblib.load(output / "models.joblib")
    reference = pd.read_csv(output / "assignments_evaluated.csv")
    sensitivity = []
    for side in ["left", "right"]:
        mod = models[side]
        ix = mod["indices"]
        t = mod["transform"]
        z = t.transform(x[ix])
        y = mod["mixture"].predict(z)
        for reg in [0.0001, 0.001, 0.01, 0.1]:
            m = GaussianMixture(
                2,
                covariance_type=cfg.covariance,
                n_init=50,
                reg_covar=reg,
                tol=1e-6,
                max_iter=2000,
                random_state=SEED,
            ).fit(z)
            sensitivity.append(
                dict(
                    side=side,
                    check="regularization_and_convergence",
                    value=reg,
                    n=len(ix),
                    ari=adjusted_rand_score(y, m.predict(z)),
                )
            )
        for hours in [18, 20, 22]:
            good = (
                q.iloc[ix][
                    ["day1_posture_hours", "day1_dynamics_hours", "day1_velocity_hours"]
                ].min(axis=1)
                >= hours
            ).to_numpy()
            if good.sum() >= 15:
                ct, cm, cy = fit_model(x[ix[good]], cfg, names, families)
                sensitivity.append(
                    dict(
                        side=side,
                        check="minimum_observed_hours",
                        value=hours,
                        n=int(good.sum()),
                        ari=adjusted_rand_score(y[good], cy),
                    )
                )
        other = models["right" if side == "left" else "left"]
        yy = other["mixture"].predict(other["transform"].transform(x[ix]))
        sensitivity.append(
            dict(
                side=side,
                check="transfer_other_colony",
                value=0,
                n=len(ix),
                ari=adjusted_rand_score(y, yy),
            )
        )
        # Restrict to the first two large landmark PCs; do not reintroduce named geometry.
        keep = np.array([not ("pc3_" in name or "pc4_" in name) for name in names])
        ax = aggregate(minute[ix, :1440][:, :, keep], cfg)
        _, _, ay = fit_model(ax, cfg, names[keep], families[keep])
        sensitivity.append(
            dict(
                side=side,
                check="two_landmark_pcs",
                value=2,
                n=len(ix),
                ari=adjusted_rand_score(y, ay),
            )
        )
    pd.DataFrame(sensitivity).to_csv(output / "sensitivity.csv", index=False)
    # Post-selection interpretation of control partitions, not a new selection.
    evaluation = []
    for _, c, allowed, _ in tasks:
        keep = np.isin(families, allowed)
        xx = aggregate(minute[:, :1440, keep], c)
        for side in ["left", "right"]:
            ix = models[side]["indices"]
            _, _, yy = fit_model(xx[ix], c, names[keep], families[keep])
            ref = reference[reference.side.eq(side)]
            _, acc = match_labels(ref.spatial_group.to_numpy(), yy)
            evaluation.append(
                dict(
                    config=c.key,
                    families="+".join(allowed),
                    side=side,
                    spatial_accuracy=acc,
                    selected_ari=adjusted_rand_score(ref.group, yy),
                )
            )
    pd.DataFrame(evaluation).to_csv(output / "controls_posthoc.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "command", choices=["screen", "validate", "finish", "diagnose", "controls"]
    )
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=24)
    a = p.parse_args()
    if a.command == "controls":
        controls(a.source, a.output, a.workers, a.bootstrap)
    elif a.command in ["screen", "validate"]:
        (screen if a.command == "screen" else validate)(
            a.source, a.output, a.workers, a.bootstrap
        )
    else:
        (finish if a.command == "finish" else baseline_diagnosis)(a.source, a.output)


if __name__ == "__main__":
    from analysis.eigenposture_parameters import main as cli_main

    cli_main()
