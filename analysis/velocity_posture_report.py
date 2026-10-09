"""Direct physical phenotype figures and post-fit spatial validation.

All labels are read here only after the unsupervised fit has been frozen.
Resampling units are ants, never frames. This module does not refit clusters.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from analysis.velocity_posture_features import FEATURES, SEED
from analysis.velocity_posture_analysis import evaluate, summarize_hours

COLORS = ["#2577a8", "#e48b2d", "#8660a4", "#52a87c"]
SPEED_EDGES = np.array([0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 20.0001])
POSTURES = ["antenna_extension", "antenna_asymmetry", "antenna_motion", "head_bend"]
NAMES = [f[0] for f in FEATURES]


def bootstrap_difference(a, b, repeats=2000):
    """Median(b)-median(a), with independent ant bootstrap intervals."""
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    rng = np.random.default_rng(SEED)
    samples = np.median(rng.choice(b, (repeats, len(b))), axis=1) - np.median(
        rng.choice(a, (repeats, len(a))), axis=1
    )
    return float(np.median(b) - np.median(a)), *np.quantile(samples, [0.025, 0.975])


def cliffs_delta(a, b):
    """P(b>a)-P(b<a); ties contribute zero."""
    return float(np.sign(np.asarray(b)[:, None] - np.asarray(a)[None, :]).mean())


def conditional_posture(minute, qc, assignments, cameras=None):
    """Within-ant posture at comparable clip speeds, optionally within camera.

    Each contributing ant needs ten observed clips in a speed stratum. A row is
    one ant, so group curves cannot be dominated by extensively observed ants.
    """
    rows = []
    camera_rows = []
    for row in assignments.itertuples():
        ix = int(np.flatnonzero(qc.ant.eq(row.ant))[0])
        x = minute[ix, :1440]
        bins = (
            np.searchsorted(SPEED_EDGES, x[:, NAMES.index("speed")], side="right") - 1
        )
        for b in range(len(SPEED_EDGES) - 1):
            use = bins == b
            for feature in POSTURES:
                v = x[:, NAMES.index(feature)]
                ok = use & np.isfinite(v)
                base = dict(
                    ant=row.ant,
                    side=row.side,
                    spatial_cluster=int(row.spatial_cluster),
                    speed_bin=b,
                    speed_low=SPEED_EDGES[b],
                    speed_high=SPEED_EDGES[b + 1],
                    feature=feature,
                )
                if ok.sum() >= 10:
                    rows.append(
                        dict(**base, value=float(np.mean(v[ok])), n_clips=int(ok.sum()))
                    )
                if cameras is not None:
                    for camera in np.unique(cameras[ix, :1440][ok]):
                        keep = ok & (cameras[ix, :1440] == camera)
                        if camera >= 0 and keep.sum() >= 10:
                            camera_rows.append(
                                dict(
                                    **base,
                                    camera=int(camera),
                                    value=float(np.mean(v[keep])),
                                    n_clips=int(keep.sum()),
                                )
                            )
    return pd.DataFrame(rows), pd.DataFrame(camera_rows)


def extract_camera_controls(pose_cache, qc):
    """Observation control only; never a clustering input."""
    result = np.full((len(qc), 2880), -1, np.int16)
    lookup = {ant: i for i, ant in enumerate(qc.ant)}
    for path in pose_cache.glob("*.npz"):
        meta = json.loads(path.with_suffix(".json").read_text())
        ant = meta["signature"]["task"]["ant"]
        if ant not in lookup:
            continue
        with np.load(path) as z:
            cam = z["cameras"]
            duplicate = z["duplicate_frame"]
        for minute in range(len(cam)):
            c = cam[minute][
                np.isfinite(cam[minute]) & (cam[minute] >= 0) & ~duplicate[minute]
            ]
            if len(c):
                result[lookup[ant], minute] = np.bincount(c.astype(int)).argmax()
    return result


def build(output, spatial_reference, occupancy_root, pose_cache=None):
    evaluate(output, spatial_reference)
    q = pd.read_csv(output / "all_ant_coverage.csv")
    assignment = pd.read_csv(output / "assignments_with_spatial_comparison.csv")
    a = assignment[assignment.family.eq("joint")].copy()
    assert (
        a.spatial_cluster.notna().all()
    ), "Every eligible ant needs a spatial reference for these comparisons"
    with np.load(output / "physical_measurements.npz") as z:
        minute = z["minute"]
        hourly = z["hourly"]
        camera_counts = z["camera_counts"]
    med = summarize_hours(hourly[:, :24])[:, len(FEATURES) : 2 * len(FEATURES)]
    physical = q[["ant", "side", "track_id"]].copy()
    for j, feature in enumerate(NAMES):
        physical[feature] = med[:, j]
    physical = physical.merge(
        a[
            [
                "ant",
                "spatial_cluster",
                "activity_cluster",
                "day2_cluster",
                "posterior_confidence",
            ]
        ],
        on="ant",
    )
    physical.to_csv(output / "ant_physical_measurements.csv", index=False)
    effects = []
    for side, part in physical.groupby("side"):
        for name, label, unit, family in FEATURES:
            v0 = part.loc[part.spatial_cluster.eq(0), name].to_numpy()
            v1 = part.loc[part.spatial_cluster.eq(1), name].to_numpy()
            delta, lo, hi = bootstrap_difference(v0, v1)
            effects.append(
                dict(
                    side=side,
                    feature=name,
                    label=label,
                    unit=unit,
                    family=family,
                    n0=len(v0),
                    n1=len(v1),
                    median0=float(np.median(v0)),
                    median1=float(np.median(v1)),
                    difference_1_minus_0=delta,
                    bootstrap_low=lo,
                    bootstrap_high=hi,
                    cliffs_delta=cliffs_delta(v0, v1),
                )
            )
    effects = pd.DataFrame(effects)
    effects.to_csv(output / "physical_group_differences.csv", index=False)
    camera_file = output / "minute_camera_controls.npz"
    if camera_file.exists():
        with np.load(camera_file) as controls:
            np.testing.assert_array_equal(controls["ants"], q.ant.to_numpy(str))
            cameras = controls["cameras"]
    elif pose_cache:
        cameras = extract_camera_controls(pose_cache, q)
        np.savez_compressed(camera_file, cameras=cameras, ants=q.ant.to_numpy(str))
    else:
        cameras = None
    conditional, camera_conditional = conditional_posture(minute, q, a, cameras)
    conditional.to_csv(output / "posture_at_matched_speed_per_ant.csv", index=False)
    camera_conditional.to_csv(
        output / "posture_at_matched_speed_camera_per_ant.csv", index=False
    )
    within = []
    if len(camera_conditional):
        for keys, part in camera_conditional.groupby(
            ["side", "feature", "speed_bin", "camera"]
        ):
            v0 = part.loc[part.spatial_cluster.eq(0), "value"].to_numpy()
            v1 = part.loc[part.spatial_cluster.eq(1), "value"].to_numpy()
            if min(len(v0), len(v1)) >= 3:
                within.append(
                    dict(
                        zip(["side", "feature", "speed_bin", "camera"], keys),
                        n0=len(v0),
                        n1=len(v1),
                        median_difference_1_minus_0=float(
                            np.median(v1) - np.median(v0)
                        ),
                    )
                )
    pd.DataFrame(within).to_csv(output / "shared_camera_speed_strata.csv", index=False)
    summary = pd.read_csv(output / "unsupervised_summary.csv")
    agreement = pd.read_csv(output / "spatial_agreement.csv")
    selection = pd.read_csv(output / "k_selection.csv")
    models = joblib.load(output / "phenotype_models.joblib")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
            "font.family": "DejaVu Sans",
        }
    )
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    gallery = []
    pdf = PdfPages(output / "velocity_posture_figures.pdf")

    def save(fig, stem, caption):
        fig.savefig(figures / f"{stem}.png", bbox_inches="tight", facecolor="white")
        fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        gallery.append((stem, caption))

    fig, axs = plt.subplots(2, 6, figsize=(17, 7), layout="constrained")
    shown = [
        "forward_velocity",
        "lateral_magnitude",
        "antenna_extension",
        "antenna_asymmetry",
        "antenna_motion",
        "head_bend",
    ]
    for r, side in enumerate(["left", "right"]):
        part = physical[physical.side.eq(side)]
        for c, name in enumerate(shown):
            ax = axs[r, c]
            name, label, unit, _ = FEATURES[NAMES.index(name)]
            for group in [0, 1]:
                vals = part.loc[part.spatial_cluster.eq(group), name].to_numpy()
                jitter = np.random.default_rng(SEED).uniform(-0.13, 0.13, len(vals))
                ax.scatter(
                    group + jitter,
                    vals,
                    s=23,
                    c=COLORS[group],
                    alpha=0.8,
                    edgecolors="white",
                    linewidths=0.3,
                )
                ax.plot(
                    [group - 0.22, group + 0.22],
                    [np.median(vals)] * 2,
                    color="black",
                    lw=2,
                )
            ax.set_xticks([0, 1], ["S0", "S1"])
            ax.set_xlim(-0.5, 1.5)
            ax.set_title(label, fontsize=11)
            ax.set_ylabel(f"{side.capitalize()} colony · {unit}")
    fig.suptitle(
        "1  The spatial groups differ in velocity and antennal shape", fontsize=17
    )
    save(
        fig,
        "01_physical_differences",
        "One point = one ant; each value is its median hourly mean on day 1. Black bars are group medians. S0/S1 are the existing spatial classes, used only for this descriptive comparison. Left: 23/18 ants; right: 27/18. All 15 features and ant-bootstrap intervals are in physical_group_differences.csv.",
    )

    fig, axs = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    centers = (SPEED_EDGES[:-1] + SPEED_EDGES[1:]) / 2
    conditional_summary = []
    for r, side in enumerate(["left", "right"]):
        for c, feature in enumerate(POSTURES):
            ax = axs[r, c]
            for group in [0, 1]:
                medians = []
                lows = []
                highs = []
                xx = []
                for b in range(len(centers)):
                    vals = conditional.loc[
                        conditional.side.eq(side)
                        & conditional.feature.eq(feature)
                        & conditional.spatial_cluster.eq(group)
                        & conditional.speed_bin.eq(b),
                        "value",
                    ].to_numpy()
                    if len(vals) < 4:
                        continue
                    rng = np.random.default_rng(SEED)
                    bs = np.median(rng.choice(vals, (2000, len(vals))), axis=1)
                    lo, hi = np.quantile(bs, [0.025, 0.975])
                    v = float(np.median(vals))
                    xx.append(centers[b])
                    medians.append(v)
                    lows.append(lo)
                    highs.append(hi)
                    conditional_summary.append(
                        dict(
                            side=side,
                            feature=feature,
                            spatial_cluster=group,
                            speed_bin=b,
                            n_ants=len(vals),
                            median=v,
                            low=lo,
                            high=hi,
                        )
                    )
                ax.plot(xx, medians, "o-", color=COLORS[group], label=f"S{group}", ms=4)
                ax.fill_between(xx, lows, highs, color=COLORS[group], alpha=0.17)
            ax.set_xscale("log")
            ax.set_xlabel("Clip mean speed (mm/s)")
            ax.set_ylabel(f"{side.capitalize()} · {FEATURES[NAMES.index(feature)][2]}")
            ax.set_title(FEATURES[NAMES.index(feature)][1], fontsize=11)
            if r == 0 and c == 0:
                ax.legend(frameon=False)
    pd.DataFrame(conditional_summary).to_csv(
        output / "posture_at_matched_speed_summary.csv", index=False
    )
    fig.suptitle(
        "2  Antennal posture differs at low speeds and converges during fast movement",
        fontsize=16,
    )
    save(
        fig,
        "02_posture_at_matched_speed",
        "Speed is measured within each 2.5-second clip. Each ant contributes its mean posture in a speed bin (at least ten clips); curves show the median across ants with 95% ant-bootstrap intervals. Each plotted point needs at least four ants in its group; high-speed bins can lack one group. Bins have unequal widths; this is coarse speed matching, not an exact causal adjustment. Camera-stratified comparisons are provided separately.",
    )

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    loading_rows = []
    for r, side in enumerate(["left", "right"]):
        part = physical[physical.side.eq(side)]
        for c, column in enumerate(["activity_cluster", "spatial_cluster"]):
            ax = axs[r, c]
            for group, points in part.groupby(column):
                ax.scatter(
                    points.forward_velocity,
                    points.antenna_extension,
                    c=COLORS[int(group)],
                    s=43,
                    alpha=0.9,
                    edgecolors="white",
                    linewidths=0.5,
                    label=f'{"A" if c==0 else "S"}{int(group)} (n={len(points)})',
                )
            ax.set_xlabel("Forward velocity (mm/s)")
            ax.set_ylabel("Antennal straightness")
            ax.set_title(
                f'{side.capitalize()}: {"physical-feature clusters" if c==0 else "spatial reference"}'
            )
            ax.legend(frameon=False, fontsize=9)
        t = models[f"{side}/joint"]["transform"]
        weights = np.zeros(45)
        weights[t.keep] = t.pca.components_[0]
        sign = 1 if weights.reshape(3, 15)[:, 0].mean() >= 0 else -1
        w = (weights * sign).reshape(3, 15).mean(axis=0)
        for name, value in zip(NAMES, w):
            loading_rows.append(
                dict(side=side, feature=name, mean_quartile_loading=value)
            )
        ix = np.argsort(np.abs(w))[-8:]
        ax = axs[r, 2]
        ax.barh(
            np.arange(8),
            w[ix],
            color=["#2577a8" if v >= 0 else "#a4576b" for v in w[ix]],
        )
        ax.set_yticks(np.arange(8), [FEATURES[i][1] for i in ix], fontsize=9)
        ax.axvline(0, color=".5", lw=0.8)
        ax.set_xlabel("Mean PCA loading across hourly quartiles")
        ax.set_title(
            f"Physical PC1: {t.pca.explained_variance_ratio_[0]:.0%} of variance"
        )
    pd.DataFrame(loading_rows).to_csv(
        output / "physical_axis_loadings.csv", index=False
    )
    fig.suptitle(
        "3  Cluster a shared velocity–posture phenotype, then reveal space", fontsize=17
    )
    save(
        fig,
        "03_physical_clustering",
        "Identical physical coordinates in the first two columns; only point colors change. A0/A1 are unsupervised physical groups, ordered by speed. All 15 measurements, represented by three hourly quartiles each, enter a standardized PCA. One dominant component summarizes the shared physical variation; a Gaussian mixture groups ants on this axis. Position, spatial labels, camera and identity never enter the fit. Antennal straightness and asymmetry contribute to this axis, so it is not a speed-only grouping.",
    )

    fig, axs = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    for r, side in enumerate(["left", "right"]):
        tab = selection[selection.side.eq(side) & selection.family.eq("joint")]
        sel = tab[tab.selected].iloc[0]
        model = models[f"{side}/joint"]
        z = model["transform"].transform(
            summarize_hours(hourly[model["indices"], :24])
        )[:, 0]
        grid = np.linspace(z.min() - 1, z.max() + 1, 400)
        ax = axs[r, 0]
        ax.hist(z, bins=12, density=True, color=".8", edgecolor="white", label="Ants")
        ax.plot(
            grid,
            norm.pdf(grid, z.mean(), z.std()),
            "--",
            color=".35",
            lw=1,
            label="K=1",
        )
        mix = model["model"]
        ax.plot(
            grid,
            np.exp(mix.score_samples(grid[:, None])),
            color="black",
            label=f"K={mix.n_components}",
        )
        for component in range(mix.n_components):
            density = mix.weights_[component] * norm.pdf(
                grid,
                mix.means_[component, 0],
                np.sqrt(mix.covariances_[component, 0, 0]),
            )
            ax.fill_between(
                grid, density, color=COLORS[model["remap"][component]], alpha=0.3
            )
        ax.set_title(f"{side.capitalize()}: shared physical axis")
        ax.set_xlabel("Physical PC1 score")
        ax.set_ylabel("Density across ants")
        ax.legend(frameon=False, fontsize=8)
        ax = axs[r, 1]
        ax.plot(tab.k, tab.bic - tab.bic.min(), "o-", color=COLORS[r])
        ax.scatter(
            [sel.k],
            [sel.bic - tab.bic.min()],
            s=160,
            facecolors="none",
            edgecolors="black",
        )
        ax.set_xticks(tab.k)
        ax.set_xlabel("Number of physical groups K")
        ax.set_ylabel("BIC − minimum (lower is better)")
        ax.set_title(f"{side.capitalize()}: selected K={int(sel.k)}")
        ax = axs[r, 2]
        t = tab[tab.k.gt(1)]
        ax.plot(
            t.k, t.bootstrap_ari_median, "o-", label="Ant bootstrap", color=COLORS[0]
        )
        ax.plot(
            t.k, t.split_minute_ari, "s-", label="Even/odd minutes", color=COLORS[1]
        )
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(t.k)
        ax.set_xlabel("K")
        ax.set_ylabel("Adjusted Rand index")
        ax.set_title("Repeatability of the partition")
        ax.legend(frameon=False, fontsize=9)
        part = a[a.side.eq(side)]
        ct = pd.crosstab(part.activity_cluster, part.spatial_cluster)
        ax = axs[r, 3]
        ax.imshow(ct, cmap="Blues", vmin=0)
        for i in range(len(ct.index)):
            for j in range(len(ct.columns)):
                ax.text(
                    j,
                    i,
                    str(ct.iloc[i, j]),
                    ha="center",
                    va="center",
                    color=(
                        "white" if ct.iloc[i, j] > ct.to_numpy().max() / 2 else "black"
                    ),
                    fontsize=18,
                )
        ax.set_xticks(range(len(ct.columns)), [f"S{int(x)}" for x in ct.columns])
        ax.set_yticks(range(len(ct.index)), [f"A{int(x)}" for x in ct.index])
        ax.set_xlabel("Spatial reference")
        ax.set_ylabel("Physical cluster")
        metric = agreement[agreement.side.eq(side) & agreement.family.eq("joint")].iloc[
            0
        ]
        ax.set_title(
            f"{int(round(metric.matched_accuracy*metric.n_compared))}/{int(metric.n_compared)} agree · ARI {metric.adjusted_rand_index:.2f}"
        )
    fig.suptitle("4  Evidence for K, stability, and spatial agreement", fontsize=17)
    save(
        fig,
        "04_cluster_validation",
        "The first column shows ants on the fitted physical axis, the single-Gaussian baseline (dashed), and the selected mixture (black, with weighted components shaded). K is selected from physical data alone: lower BIC than K=1, median ant-bootstrap ARI ≥0.8, even/odd-minute ARI ≥0.6, and every group at least 10% of ants (minimum four). Among eligible solutions within two BIC units of the best, choose the smaller K. BIC improvement is moderate, not overwhelming. Spatial comparison comes after the fit is frozen. Bootstrap refits preprocessing and clustering; split-minute fits use independently measured clips.",
    )

    fig, axs = plt.subplots(2, 4, figsize=(13, 10), layout="constrained")
    occupancy_rows = []
    occupancy_arrays = {}
    for r, side in enumerate(["left", "right"]):
        part = a[a.side.eq(side)]
        maps = []
        keys = []
        titles = []
        for column, prefix in [
            ("spatial_cluster", "Spatial"),
            ("activity_cluster", "Physical"),
        ]:
            for group in [0, 1]:
                stack = []
                for row in part[part[column].eq(group)].itertuples():
                    folder = occupancy_root / "per_track" / Path(row.track_name).stem
                    hist = np.load(folder / "grid_occupancy_f4.npy").astype(float)
                    hist /= hist.sum()
                    stack.append(hist)
                    xe = np.load(folder / "grid_x_edges_mm.npy")
                    ye = np.load(folder / "grid_y_edges_mm.npy")
                    occupancy_rows.append(
                        dict(
                            ant=row.ant,
                            side=side,
                            label_source=column,
                            group=group,
                            path=str(folder),
                            sha256=hashlib.sha256(
                                (folder / "grid_occupancy_f4.npy").read_bytes()
                            ).hexdigest(),
                        )
                    )
                if stack:
                    maps.append(np.mean(stack, axis=0))
                    keys.append(f"{side}_{prefix.lower()}_{group}")
                    titles.append(f"{prefix} {group} · n={len(stack)}")
        vmax = max(np.max(m) for m in maps)
        for c, (m, key, title) in enumerate(zip(maps, keys, titles)):
            ax = axs[r, c]
            im = ax.imshow(
                m,
                origin="upper",
                extent=[xe[0], xe[-1], ye[-1], ye[0]],
                norm=PowerNorm(0.4, vmin=0, vmax=vmax),
                cmap="magma",
                aspect="equal",
            )
            ax.set_title(f"{side.capitalize()} · {title}")
            ax.set_xlabel("Arena x (mm)")
            ax.set_ylabel("Arena y (mm)")
            occupancy_arrays[key] = m
        occupancy_arrays[f"{side}_x_edges"] = xe
        occupancy_arrays[f"{side}_y_edges"] = ye
        fig.colorbar(
            im, ax=axs[r, :], shrink=0.6, label="Mean probability per 0.25 mm bin"
        )
    np.savez_compressed(output / "group_mean_occupancy.npz", **occupancy_arrays)
    pd.DataFrame(occupancy_rows).to_csv(
        output / "occupancy_validation_sources.csv", index=False
    )
    fig.suptitle(
        "5  Spatial segregation emerges from the physical grouping", fontsize=17
    )
    save(
        fig,
        "05_emergent_spatial_occupancy",
        "Each map averages normalized occupancy equally across ants. Spatial-reference maps and physical-group maps use exactly the same eligible cohort. Color scaling is shared within colony (power scale exponent 0.4); displayed coordinates and occupancy are used only for validation. These existing maps span the full recording, so they are not an independent held-out spatial test.",
    )

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for r, side in enumerate(["left", "right"]):
        part = a[a.side.eq(side)].sort_values(["activity_cluster", "track_id"])
        ix = [int(np.flatnonzero(q.ant.eq(ant))[0]) for ant in part.ant]
        for c, feature in enumerate(["forward_velocity", "antenna_extension"]):
            ax = axs[r, c]
            data = hourly[ix, :, NAMES.index(feature)]
            limits = np.nanquantile(data, [0.02, 0.98])
            im = ax.imshow(
                data,
                aspect="auto",
                interpolation="none",
                vmin=limits[0],
                vmax=limits[1],
                cmap="viridis",
            )
            ax.axvline(23.5, color="white", lw=2)
            ax.set_xticks(
                [0, 12, 24, 36, 47],
                ["Jul24 10h", "22h", "Jul25 10h", "22h", "Jul26 09h"],
            )
            boundaries = np.flatnonzero(np.diff(part.activity_cluster.to_numpy())) + 0.5
            for boundary in boundaries:
                ax.axhline(boundary, color="white", lw=1.5)
            ax.set_ylabel(
                f"{side.capitalize()}: {len(part)} ants, ordered by day-1 cluster"
            )
            ax.set_title(FEATURES[NAMES.index(feature)][1] + " · hourly means")
            fig.colorbar(im, ax=ax, shrink=0.8, label=FEATURES[NAMES.index(feature)][2])
    fig.suptitle("6  Hourly measurements and next-day persistence", fontsize=17)
    save(
        fig,
        "06_hourly_profiles",
        "Columns are successive one-hour bins; rows are ants ordered by physical group and ID. The white vertical line separates training day 1 from day 2. The frozen day-1 scaler, PCA and mixture assign day-2 profiles without refitting. White cells are missing measurements, not inactivity. Exact hourly values, per-feature observation counts and ant IDs are included in physical_measurements.npz and all_ant_coverage.csv.",
    )
    pdf.close()

    controls = []
    for side, part in a.groupby("side"):
        model = models[f"{side}/joint"]
        ix = model["indices"]
        profile = summarize_hours(hourly[ix, :24])
        score = model["transform"].transform(profile)[:, 0]
        for column in [
            "day1_observed_minutes",
            "day1_pose_samples",
            "day1_minimum_feature_hours",
        ]:
            controls.append(
                dict(
                    side=side,
                    control=column,
                    spearman_r=float(spearmanr(score, q.iloc[ix][column]).statistic),
                )
            )
    pd.DataFrame(controls).to_csv(output / "observation_controls.csv", index=False)
    a.groupby(
        ["side", "dominant_camera_day1", "spatial_cluster", "activity_cluster"]
    ).size().reset_index(name="n_ants").to_csv(
        output / "dominant_camera_contingency.csv", index=False
    )
    make_report(output, gallery, summary, agreement, selection, effects, within, q)


def make_report(output, gallery, summary, agreement, selection, effects, within, q):
    conditional = pd.read_csv(output / "posture_at_matched_speed_summary.csv")
    speed_statements = []
    for b, label in [(0, "Below 0.05 mm/s"), (5, "At 1–2 mm/s")]:
        comparisons = []
        for side in ["left", "right"]:
            part = conditional[
                conditional.side.eq(side)
                & conditional.feature.eq("antenna_extension")
                & conditional.speed_bin.eq(b)
            ]
            if set(part.spatial_cluster) == {0, 1}:
                values = part.set_index("spatial_cluster")["median"]
                comparisons.append(f"{values[0]:.3f} versus {values[1]:.3f} ({side})")
        speed_statements.append(
            f"{label}, median antennal straightness is "
            + ", ".join(comparisons)
            + " (spatial 0 versus 1)."
        )
    controls = pd.read_csv(output / "observation_controls.csv")
    control_values = controls.set_index(["side", "control"]).spearman_r
    control_statement = (
        f"The physical axis correlates with observed minutes in the left colony (Spearman r={control_values['left', 'day1_observed_minutes']:.2f}) "
        f"and with available pose samples in the right colony (r={control_values['right', 'day1_pose_samples']:.2f}), so observation effects cannot be dismissed."
    )
    lines = [
        "# Velocity and posture phenotypes — 2026-07-24",
        "",
        "Fresh analysis of block01. The question is whether physical behavior alone recovers the existing two spatial groups, and which measurable behaviors distinguish them.",
        "",
        "## Main result",
        "",
    ]
    metrics = []
    for side in ["left", "right"]:
        result = agreement[agreement.side.eq(side) & agreement.family.eq("joint")].iloc[
            0
        ]
        val = summary[summary.side.eq(side) & summary.family.eq("joint")].iloc[0]
        n = int(result.n_compared)
        correct = int(round(result.matched_accuracy * n))
        text = f"{side.capitalize()}: K={int(result.selected_k)}, {correct}/{n} ants agree with the spatial partition ({result.matched_accuracy:.1%}; ARI {result.adjusted_rand_index:.3f}). Day-2 frozen assignments retain {val.day2_frozen_retention:.1%} of {int(val.day2_ants)} measured ants; independent day-2 refit ARI {val.day2_independent_ari:.3f}."
        metrics.append(text)
        lines.append("- " + text)
    lines += [
        "",
        "Spatial group 1 has greater forward and lateral motion, straighter antennae and more similar straightness between the two antennae. Head bending is a much weaker discriminator. The antennal difference persists at low matched speeds and largely converges during fast movement. Differences are associations, not established intrinsic ant types or behavioral roles.",
        "",
        "| Colony | Physical measurement | Spatial 0 median | Spatial 1 median | Difference (95% ant-bootstrap interval) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in effects[
        effects.feature.isin(
            [
                "forward_velocity",
                "lateral_magnitude",
                "antenna_extension",
                "antenna_asymmetry",
                "head_bend",
            ]
        )
    ].itertuples():
        lines.append(
            f"| {row.side} | {row.label} ({row.unit}) | {row.median0:.3f} | {row.median1:.3f} | {row.difference_1_minus_0:+.3f} [{row.bootstrap_low:+.3f}, {row.bootstrap_high:+.3f}] |"
        )
    lines += [
        "",
        "## What was fit",
        "",
        "One random 2.5-second clip per minute per ant, drawn from the existing unfitted skeleton/velocity cache. The 114 source tracking signatures were checked against current file sizes and modification times. Day 1 is July 24 10:00–July 25 10:00 JST; day 2 is the next 24 hours. Local valid windows contribute; a missing landmark does not discard a whole clip. Missing observations never become zeros.",
        "",
        "Seven velocity measurements (signed forward and lateral velocity, speed, upper-tail speed, moving fraction, backward fraction and absolute lateral velocity), five static posture measurements (head/gaster bend, antenna spread, straightness and asymmetry), and three short-timescale angular-motion measurements (head, gaster, antennae). Forward/lateral axes follow the ant body. Antennal straightness is tip-to-head distance divided by the length of the three-segment antenna chain; 1 means straight. Measurements are body-relative; no absolute spatial coordinates enter clustering.",
        "",
        "Clips contribute equal weight to hourly means (at least five valid minutes per feature per hour). An ant needs at least 12 measured hours for every feature on day 1. Its 25th, 50th and 75th percentiles across hours form a 45-variable profile. Monotone transforms reduce heavy velocity tails; each variable is standardized using training ants. A single PCA component captures the dominant correlated physical pattern. A Gaussian mixture with unequal variances groups ants on that axis. Fits are separate for the two colonies. Posture contributes through learned PCA loadings; there is no spatial feature, supervised classifier or motif-frequency fitting.",
        "",
        f"Tracking contains {len(q)} identities: "
        + ", ".join(
            f"{side} {int(q.side.eq(side).sum())} total / {int((q.side.eq(side)&q.day1_eligible).sum())} eligible"
            for side in ["left", "right"]
        )
        + ". The 86 eligible ants match all 86 identities in the spatial reference. The other 28 identities fail the measured-hours rule; they are not treated as inactive or assigned by extrapolation. See all_ant_coverage.csv for every identity and reason.",
        "",
        "## Why two groups?",
        "",
        "Compare K=1–4 using BIC on the same one-dimensional physical representation. A multi-group solution must beat K=1, have median ant-bootstrap adjusted Rand index ≥0.8, even/odd-minute ARI ≥0.6, and at least max(4,10% of ants) in each group. Choose the smaller K within two BIC units of the best eligible solution. Spatial agreement is not a K-selection criterion. Day 2 is not used for fitting or selecting K. A one-axis representation is a simplicity choice and does not claim all posture variation is one-dimensional.",
        "",
    ]
    for side in ["left", "right"]:
        tab = selection[selection.side.eq(side) & selection.family.eq("joint")]
        selected = tab[tab.selected].iloc[0]
        delta = float(tab.loc[tab.k.eq(1), "bic"].iloc[0] - selected.bic)
        lines.append(
            f"- {side.capitalize()}: K={int(selected.k)}; BIC improvement over K=1 {delta:.2f}; bootstrap median ARI {selected.bootstrap_ari_median:.3f}, 10th percentile {selected.bootstrap_ari_p10:.3f}; split-minute ARI {selected.split_minute_ari:.3f}."
        )
    lines += [
        "",
        "The BIC evidence is moderate. These data support a useful approximate two-group physical description; they do not prove exactly two immutable biological classes. The result is exploratory: earlier representations were examined on this dataset before choosing this simpler shared axis. Numerical records of those attempts are retained in exploratory_history. This is not a prospectively blind discovery or a validation on independent colonies.",
        "",
        "## Does posture add information beyond speed?",
        "",
        "Figure 2 compares posture within speed bins, with each ant weighted equally. "
        + " ".join(speed_statements)
        + " Thus the posture signal is strongest during low-speed behavior and is not simply an overall difference in the amount of walking. Coarse speed bins and camera effects still prevent a causal interpretation. The per-ant values, intervals and group-specific sample counts are supplied as CSVs. Velocity-only and posture-only ablations use the same PCA/mixture workflow and the same eligible cohort:",
        "",
        "| Colony | Input family | Selected K | Spatial agreement | Day-2 retention |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        metric = agreement[
            agreement.side.eq(row.side) & agreement.family.eq(row.family)
        ].iloc[0]
        match = f"{metric.matched_accuracy:.1%}" if row.selected_k > 1 else "unsplit"
        retention = f"{row.day2_frozen_retention:.1%}" if row.selected_k > 1 else "n/a"
        lines.append(
            f"| {row.side} | {row.family} | {row.selected_k} | {match} | {retention} |"
        )
    lines += [
        "",
        "## Observation and camera controls",
        "",
        "Pose quality and camera assignment can covary with location. Camera never enters the fit. Dominant-camera contingency tables and correlations between the physical axis and observation coverage are included. "
        + control_statement
        + " Supplementary comparisons restrict to shared camera × speed strata with at least three ants from each spatial class and ten clips per ant. These are descriptive strata, not independent replicates or a complete correction for camera bias. Straightness is dimensionless but can still be affected by pose-estimation error.",
        "",
    ]
    w = pd.DataFrame(within)
    if len(w):
        for side in ["left", "right"]:
            part = w[w.side.eq(side) & w.feature.eq("antenna_extension")]
            if len(part):
                lines.append(
                    f"- {side.capitalize()}: spatial group 1 has straighter antennae in {int(part.median_difference_1_minus_0.gt(0).sum())}/{len(part)} shared camera × speed strata."
                )
    lines += [
        "",
        "Occupancy maps and the spatial reference span the full recording and share data with the physical analysis. Their agreement is a post-fit correspondence check, not an independent spatial holdout. Day-2 persistence uses the same ants; independent colonies are needed to assess generality. The source cache samples only 2.5 seconds each minute, so fast or rare events can be missed. The analysis describes hourly distributions, not the complete ordering of actions.",
        "",
        "## Figures and files",
        "",
        "- [All figures (PDF)](velocity_posture_figures.pdf)",
        "- [Per-ant physical measurements](ant_physical_measurements.csv)",
        "- [Group differences and ant-bootstrap intervals](physical_group_differences.csv)",
        "- [Cluster assignments and spatial comparison](assignments_with_spatial_comparison.csv)",
        "- [Coverage audit for all 114 identities](all_ant_coverage.csv)",
        "- [K selection](k_selection.csv)",
        "- [Frozen fit and provenance](UNSUPERVISED_FROZEN.json)",
        "- [Reproduction instructions](reproduction/README.md)",
        "",
    ]
    for stem, caption in gallery:
        lines += [
            f'### {stem.replace("_"," ")}',
            "",
            f"![{stem}](figures/{stem}.png)",
            "",
            caption,
            "",
        ]
    lines += [
        "Gaussian-mixture/BIC implementation: [scikit-learn mixture selection documentation](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html)."
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    cards = "".join(
        f'<section><a href="figures/{stem}.pdf"><img src="figures/{stem}.png" alt="{html.escape(stem)}"></a><p>{html.escape(caption)}</p></section>'
        for stem, caption in gallery
    )
    page = (
        '<!doctype html><html><head><meta charset="utf-8"><title>July 24 velocity and posture</title><style>body{max-width:1400px;margin:32px auto;padding:0 24px;font:17px/1.5 system-ui;color:#202c35;background:#f5f7fa}h1{font-size:32px}section{background:white;padding:20px;margin:28px 0;border-radius:8px}img{width:100%}a{color:#17659b}.note{padding:16px;background:#fff0d9}li{margin:8px}</style></head><body><h1>July 24: velocity and posture explain spatial segregation</h1><p>Fresh analysis · 86 eligible ants · two colonies · hourly physical measurements</p><ul>'
        + "".join("<li>" + html.escape(m) + "</li>" for m in metrics)
        + '</ul><p>Spatial group 1 moves faster and has straighter, more symmetric antennae. The physical grouping uses velocity and posture together; spatial information is introduced only after fitting.</p><p class="note">Exploratory analysis; moderate evidence for two groups. Group correspondence is approximate, and camera/observation effects remain possible. Same-ant day-2 checks are not independent-colony validation.</p><p><a href="REPORT.md">Full report and limitations</a> · <a href="velocity_posture_figures.pdf">All figures PDF</a> · <a href="assignments_with_spatial_comparison.csv">Ant assignments</a> · <a href="physical_group_differences.csv">Effect estimates</a> · <a href="all_ant_coverage.csv">114-ant coverage audit</a> · <a href="reproduction/README.md">Reproduce</a></p>'
        + cards
        + "</body></html>"
    )
    (output / "index.html").write_text(page)
    (output / "REPORT_COMPLETE.json").write_text(
        json.dumps(
            dict(
                created_at=datetime.now(timezone.utc).isoformat(),
                figures=len(gallery),
                eligible_ants=int(len(q[q.day1_eligible])),
                source_script_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--spatial-reference", type=Path, required=True)
    p.add_argument("--occupancy-root", type=Path, required=True)
    p.add_argument("--pose-cache", type=Path)
    a = p.parse_args()
    build(a.output, a.spatial_reference, a.occupancy_root, a.pose_cache)


if __name__ == "__main__":
    main()
