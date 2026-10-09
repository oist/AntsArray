"""Interpret frozen landmark modes and compare physical groups with space."""

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

from analysis.eigenposture_features import SEED, mean_missing
from analysis.eigenposture_analysis import evaluate, profiles, ProfileAxis

COLORS = ["#267ca5", "#e58b2d", "#956bb1", "#369976"]


def straightness(x):
    """Interpretation only: never called by extraction, PCA or cluster fitting."""
    chains = x.reshape(*x.shape[:-1], 2, 3, 2)
    origin = np.zeros((*chains.shape[:-2], 1, 2))
    length = np.linalg.norm(
        np.diff(np.concatenate([origin, chains], axis=-2), axis=-2), axis=-1
    ).sum(axis=-1)
    reach = np.linalg.norm(chains[..., -1, :], axis=-1)
    return np.mean(reach / length, axis=-1)


def weighted_correlation(x, y, w):
    x = x - np.sum(x * w)
    y = y - np.sum(y * w)
    return float(np.sum(w * x * y) / np.sqrt(np.sum(w * x * x) * np.sum(w * y * y)))


def heldout_samples(sample, q, day=1):
    eligible = q.day1_eligible.to_numpy(bool) & (
        q.day2_eligible.to_numpy(bool) if day == 1 else True
    )
    x = []
    weights = []
    ants = []
    for side in ["left", "right"]:
        ix = np.flatnonzero(q.side.eq(side) & eligible)
        for i in ix:
            raw = sample[i, day].reshape(-1, 12)
            ok = np.isfinite(raw).all(axis=1)
            raw = raw[ok]
            x.append(raw)
            weights.extend(np.full(len(raw), 0.5 / len(ix) / len(raw)))
            ants.extend([q.ant.iloc[i]] * len(raw))
    return np.concatenate(x), np.array(weights), np.array(ants)


def interpret_basis(output, basis, q):
    with np.load(output / "sampled_landmarks.npz") as f:
        samples = f["coordinates"]
    x, w, ants = heldout_samples(samples, q)
    c = basis["components"]
    center = basis["mean"]
    scores = (x - center) @ c.T
    target = straightness(x)
    variance = np.sum(w * (target - np.sum(w * target)) ** 2)
    rows = []
    for k in range(1, 13):
        reconstructed = center + scores[:, :k] @ c[:k]
        predicted = straightness(reconstructed)
        rows.append(
            dict(
                rank=k,
                straightness_r2=1
                - float(np.sum(w * (predicted - target) ** 2) / variance),
                coordinate_variance_explained=1
                - float(
                    np.sum(w[:, None] * (x - reconstructed) ** 2)
                    / np.sum(w[:, None] * (x - center) ** 2)
                ),
            )
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "heldout_reconstruction.csv", index=False)
    correlations = [
        dict(
            mode=i + 1,
            heldout_straightness_r=weighted_correlation(scores[:, i], target, w),
        )
        for i in range(12)
    ]
    pd.DataFrame(correlations).to_csv(output / "mode_interpretation.csv", index=False)
    rank = int(basis["rank"])
    predicted = straightness(center + scores[:, :rank] @ c[:rank])
    np.savez_compressed(
        output / "heldout_interpretation_samples.npz",
        scores=scores.astype(np.float32),
        straightness=target.astype(np.float32),
        reconstructed_straightness=predicted.astype(np.float32),
        weights=w,
        ants=ants,
    )
    return table, pd.DataFrame(correlations), target, predicted


def draw_shape(ax, vector, color, label=None, alpha=1.0, linewidth=1.8):
    p = vector.reshape(2, 3, 2)
    for antenna in range(2):
        points = np.vstack([np.zeros(2), p[antenna]])
        ax.plot(
            points[:, 1],
            points[:, 0],
            "o-",
            c=color,
            ms=3,
            lw=linewidth,
            alpha=alpha,
            label=label if antenna == 0 else None,
        )
    ax.set_aspect("equal")
    ax.axvline(0, color=".85", lw=0.6)
    ax.scatter([0], [0], c="black", s=18, zorder=5)


def build(output, reference, occupancy):
    evaluate(output, reference)
    q = pd.read_csv(output / "all_ant_coverage.csv")
    with np.load(output / "landmark_pca.npz") as f:
        basis = {k: f[k] for k in f.files}
    with np.load(output / "eigenposture_measurements.npz") as f:
        hourly = f["hourly"]
        names = f["feature_names"]
        families = f["families"]
    with np.load(output / "ant_coordinate_moments.npz") as f:
        antmeans = f["means"]
    rank = int(basis["rank"])
    reconstruction, interpretation, straight, predicted = interpret_basis(
        output, basis, q
    )
    assignment = pd.read_csv(output / "spatial_comparison.csv")
    a = assignment[assignment.family.eq("joint")].copy()
    summary = pd.read_csv(output / "fit_summary.csv")
    metrics = pd.read_csv(output / "spatial_metrics.csv")
    selection = pd.read_csv(output / "k_selection.csv")
    models = joblib.load(output / "models.joblib")
    x = profiles(hourly[:, :24])
    med = x[:, len(names) : 2 * len(names)]
    physical = q[["ant", "side", "track_id"]].copy()
    for j, name in enumerate(names):
        physical[name] = med[:, j]
    physical = physical.merge(
        a[["ant", "cluster", "spatial_cluster", "day2_cluster", "posterior"]], on="ant"
    )
    physical.to_csv(output / "ant_mode_measurements.csv", index=False)
    loadings = []
    shown_modes = {}
    for side in ["left", "right"]:
        model = models[f"{side}/joint"]
        t = model["transform"]
        w = np.zeros(3 * len(names))
        w[t.keep] = t.pca.components_[0]
        if w.reshape(3, -1)[:, 2].mean() < 0:
            w = -w
        w = w.reshape(3, -1).mean(axis=0)
        for name, family, value in zip(names, families, w):
            loadings.append(dict(side=side, feature=name, family=family, loading=value))
        chosen = np.argsort(np.abs(w[4 : 4 + rank]))[::-1][:2]
        shown_modes[side] = chosen
    loading = pd.DataFrame(loadings)
    loading.to_csv(output / "phenotype_loadings.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    gallery = []
    pdf = PdfPages(output / "eigenposture_figures.pdf")

    def save(fig, name, caption):
        for suffix in ["png", "pdf"]:
            fig.savefig(
                folder / f"{name}.{suffix}", bbox_inches="tight", facecolor="white"
            )
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        gallery.append((name, caption))

    ncols = max(4, rank)
    fig = plt.figure(figsize=(max(14, 2.9 * ncols), 8), layout="constrained")
    gs = fig.add_gridspec(2, ncols)
    ax = fig.add_subplot(gs[0, :2])
    variance = basis["variance_ratio"]
    ax.bar(np.arange(1, 13), variance * 100, color=".65", label="Individual variance")
    ax.plot(
        np.arange(1, 13),
        np.cumsum(variance) * 100,
        "o-",
        c=COLORS[0],
        label="Cumulative",
    )
    ax.axhline(90, color=".4", ls="--")
    ax.axvline(rank, color=COLORS[1], lw=2)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Learned posture mode")
    ax.set_ylabel("Coordinate variance (%)")
    ax.legend(frameon=False)
    ax.set_title(f"{rank} modes retain {variance[:rank].sum():.1%} of posture variance")
    ax = fig.add_subplot(gs[0, 2:])
    im = ax.imshow(
        basis["components"][:rank], aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6
    )
    ax.set_yticks(range(rank), [f"P{i+1}" for i in range(rank)])
    ax.set_xticks(
        range(12),
        [
            f"{a}{j} {axis}"
            for a in ["A", "B"]
            for j in [1, 2, 3]
            for axis in ["forward", "lateral"]
        ],
        rotation=60,
        ha="right",
        fontsize=8,
    )
    fig.colorbar(im, ax=ax, label="Coordinate loading")
    ax.set_title("PCA uses landmark coordinates directly")
    all_shapes = []
    for mode in range(rank):
        ax = fig.add_subplot(gs[1, mode])
        sd = np.sqrt(basis["eigenvalues"][mode])
        for value, color, label in [
            (-1.5, COLORS[0], "−1.5 SD"),
            (0, ".3", "Mean"),
            (1.5, COLORS[1], "+1.5 SD"),
        ]:
            shape = basis["mean"] + value * sd * basis["components"][mode]
            all_shapes.append(shape)
            draw_shape(ax, shape, color, label)
        ax.set_title(f"P{mode+1}: {variance[mode]:.1%}")
        ax.set_xlabel("Lateral / body-axis length")
        ax.set_ylabel("Anterior / body-axis length")
        if mode == 0:
            ax.legend(frameon=False, fontsize=8)
    points = np.stack(all_shapes).reshape(-1, 2)
    for ax in fig.axes[-rank:]:
        ax.set_xlim(points[:, 1].min() - 0.15, points[:, 1].max() + 0.15)
        ax.set_ylim(min(0, points[:, 0].min()) - 0.1, points[:, 0].max() + 0.15)
    fig.suptitle(
        "1  Learn antennal posture from positions, before defining behaviors",
        fontsize=17,
    )
    save(
        fig,
        "01_learned_posture_modes",
        "The inputs are six head-relative antennal landmarks (12 coordinates), aligned to the anterior body axis. Each ant is normalized by one fixed day-1 median body length; per-frame bending and extension are preserved. PCA weights colonies, ants, hours and minute clips equally. Lower panels are linear mean ±1.5 SD reconstructions, not measured trajectories; segment lengths can vary in this coordinate representation. No straightness, spread or spatial class enters PCA.",
    )

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    ax = axs[0]
    ax.plot(
        reconstruction["rank"],
        reconstruction.coordinate_variance_explained,
        "o-",
        label="Coordinate variance",
        c=COLORS[0],
    )
    ax.plot(
        reconstruction["rank"],
        reconstruction.straightness_r2,
        "s-",
        label="Straightness reconstruction R²",
        c=COLORS[1],
    )
    ax.axvline(rank, c=".3", ls="--")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Number of retained posture modes")
    ax.set_ylabel("Held-out reconstruction quality")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Does the learned geometry retain straightness?")
    ax = axs[1]
    ax.bar(
        interpretation["mode"][:rank],
        interpretation.heldout_straightness_r[:rank],
        color=COLORS[0],
    )
    ax.axhline(0, c=".5", lw=0.7)
    ax.set_ylim(-1, 1)
    ax.set_xticks(range(1, rank + 1))
    ax.set_xlabel("Learned posture mode")
    ax.set_ylabel("Correlation with straightness")
    ax.set_title("Interpretation after fitting")
    ax = axs[2]
    im = ax.hexbin(straight, predicted, gridsize=45, mincnt=1, cmap="magma", bins="log")
    ax.plot([0, 1], [0, 1], "--", c="cyan", lw=1)
    ax.set_xlabel("Straightness from measured landmarks")
    ax.set_ylabel(f"Straightness from {rank}-mode reconstruction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.colorbar(im, ax=ax, label="Sample count")
    ax.set_title("Day 2, using the frozen day-1 basis")
    fig.suptitle("2  Test the meaning of the modes after learning them", fontsize=17)
    save(
        fig,
        "02_posthoc_mode_interpretation",
        "Straightness is introduced here only to interpret the frozen coordinate basis. It is tip-to-head distance divided by the three-segment chain length, averaged across the two antennae. It never enters the ant profiles, clustering or K selection. Reconstruction R² and mode correlations use held-out day-2 samples with equal colony/ant weights. A nonlinear shape property need not coincide with one linear principal component. The hexbin shows sampled poses; it is not a frame-level significance test.",
    )

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for r, side in enumerate(["left", "right"]):
        part = physical[physical.side.eq(side)]
        mode = int(shown_modes[side][0]) + 1
        name = f"posture_pc{mode}_mean"
        for col, label in enumerate(["cluster", "spatial_cluster"]):
            ax = axs[r, col]
            for group, subset in part.groupby(label, dropna=False):
                color = ".65" if pd.isna(group) else COLORS[int(group)]
                text = (
                    "No spatial reference"
                    if pd.isna(group)
                    else f'{"A" if col==0 else "S"}{int(group)} (n={len(subset)})'
                )
                ax.scatter(
                    subset.forward_mean,
                    subset[name],
                    s=40,
                    c=color,
                    edgecolors="white",
                    linewidths=0.4,
                    label=text,
                )
            ax.set_xlabel("Median hourly forward velocity (mm/s)")
            ax.set_ylabel(f"Median hourly posture P{mode} amplitude")
            ax.set_title(
                f'{side.capitalize()}: {"learned-feature groups" if col==0 else "spatial reference"}'
            )
            ax.legend(frameon=False, fontsize=8)
        part = loading[loading.side.eq(side)]
        order = part.loading.abs().nlargest(8).index
        top = part.loc[order].sort_values("loading")
        ax = axs[r, 2]
        ax.barh(
            top.feature,
            top.loading,
            color=[COLORS[0] if f == "velocity" else COLORS[1] for f in top.family],
        )
        ax.set_xlabel("Mean ant-profile PCA loading across quartiles")
        ax.set_title("What drives the physical grouping?")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(
        "3  Group ants from learned posture dynamics and velocity", fontsize=17
    )
    save(
        fig,
        "03_ant_groups_in_learned_features",
        "One point is one ant. The first two columns show identical physical measurements colored by the unsupervised grouping or by spatial reference. For display, the posture mode with the largest absolute contribution to the ant-profile PCA is shown; this choice uses no spatial labels. All retained posture amplitudes, their RMS short-timescale derivatives, and signed mean/RMS anterior and lateral velocity enter the joint fit. The posture basis and the ant-profile PCA are distinct stages.",
    )

    fig, axs = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    for r, side in enumerate(["left", "right"]):
        model = models[f"{side}/joint"]
        z = model["transform"].transform(x[model["indices"]])[:, 0]
        m = model["mixture"]
        grid = np.linspace(z.min() - 1, z.max() + 1, 400)
        ax = axs[r, 0]
        ax.hist(z, bins=12, density=True, color=".8", edgecolor="white")
        ax.plot(grid, norm.pdf(grid, z.mean(), z.std()), "--", c=".5", label="K=1")
        ax.plot(
            grid,
            np.exp(m.score_samples(grid[:, None])),
            c="black",
            label=f"Selected K={m.n_components}",
        )
        for component in range(m.n_components):
            ax.fill_between(
                grid,
                m.weights_[component]
                * norm.pdf(
                    grid,
                    m.means_[component, 0],
                    np.sqrt(m.covariances_[component, 0, 0]),
                ),
                color=COLORS[model["remap"][component]],
                alpha=0.3,
            )
        ax.set_title(f"{side.capitalize()}: physical phenotype axis")
        ax.set_xlabel("Ant-profile PC1 score")
        ax.set_ylabel("Density across ants")
        ax.legend(frameon=False, fontsize=8)
        tab = selection[selection.side.eq(side) & selection.family.eq("joint")]
        chosen = tab[tab.selected].iloc[0]
        ax = axs[r, 1]
        ax.plot(tab.k, tab.bic - tab.bic.min(), "o-", color=COLORS[0])
        ax.scatter(
            [chosen.k],
            [chosen.bic - tab.bic.min()],
            s=150,
            facecolors="none",
            edgecolors="black",
        )
        ax.set_xticks(range(1, 5))
        ax.set_xlabel("K")
        ax.set_ylabel("BIC − minimum")
        ax.set_title(f"Selected K={int(chosen.k)}; lower BIC is better")
        ax = axs[r, 2]
        multi = tab[tab.k.gt(1)]
        ax.plot(multi.k, multi.bootstrap_median_ari, "o-", label="Ant bootstrap")
        ax.plot(multi.k, multi.split_minute_ari, "s-", label="Even/odd minutes")
        ax.set_xticks([2, 3, 4])
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlabel("K")
        ax.set_ylabel("Adjusted Rand index")
        ax.set_title("Repeatability")
        ax.legend(frameon=False, fontsize=8)
        part = a[a.side.eq(side)].dropna(subset=["spatial_cluster"])
        ct = pd.crosstab(part.cluster, part.spatial_cluster)
        ax = axs[r, 3]
        ax.imshow(ct, cmap="Blues", vmin=0)
        for i in range(len(ct)):
            for j in range(len(ct.columns)):
                ax.text(
                    j,
                    i,
                    str(ct.iloc[i, j]),
                    ha="center",
                    va="center",
                    fontsize=17,
                    color=(
                        "white" if ct.iloc[i, j] > ct.to_numpy().max() / 2 else "black"
                    ),
                )
        ax.set_xticks(range(len(ct.columns)), [f"S{int(g)}" for g in ct.columns])
        ax.set_yticks(range(len(ct)), [f"A{int(g)}" for g in ct.index])
        metric = metrics[metrics.side.eq(side) & metrics.family.eq("joint")].iloc[0]
        ax.set_title(
            f"{int(round(metric.accuracy*metric.n_compared))}/{int(metric.n_compared)} agree · ARI {metric.ari:.2f}"
            if metric.k > 1
            else "K=1: no supported split"
        )
        ax.set_xlabel("Spatial reference")
        ax.set_ylabel("Learned-feature group")
    fig.suptitle(
        "4  Test the number of groups and compare with space afterward", fontsize=17
    )
    save(
        fig,
        "04_k_and_spatial_agreement",
        "K=1–4 is evaluated using physical inputs only. The previously fixed rule requires a BIC improvement over K=1, median ant-bootstrap ARI ≥0.8, even/odd-minute ARI ≥0.6 and at least max(4,10% of ants) per group; choose the smaller eligible K within two BIC units of the best. A tiny BIC improvement is weak evidence even when the rule selects K=2. Bootstrap refits profile scaling/PCA/mixtures conditional on the landmark basis; even/odd fits additionally learn their coordinate PCA independently. Spatial labels are loaded only after fit hashes are frozen.",
    )

    maxk = int(a.k.max())
    fig, axs = plt.subplots(
        2, 2 + maxk, figsize=(3.2 * (2 + maxk), 10), layout="constrained", squeeze=False
    )
    mapdata = {}
    map_sources = []
    for r, side in enumerate(["left", "right"]):
        part = a[a.side.eq(side) & a.spatial_cluster.notna()]
        maps = []
        titles = []
        keys = []
        for column, prefix, groups in [
            ("spatial_cluster", "Spatial", [0, 1]),
            ("cluster", "Learned", range(int(part.k.iloc[0]))),
        ]:
            for group in groups:
                stack = []
                for row in part[part[column].eq(group)].itertuples():
                    path = occupancy / "per_track" / Path(row.track_name).stem
                    hist = np.load(path / "grid_occupancy_f4.npy").astype(float)
                    hist /= hist.sum()
                    stack.append(hist)
                    xe = np.load(path / "grid_x_edges_mm.npy")
                    ye = np.load(path / "grid_y_edges_mm.npy")
                    map_sources.append(
                        dict(
                            ant=row.ant,
                            side=side,
                            path=str(path),
                            sha256=hashlib.sha256(
                                (path / "grid_occupancy_f4.npy").read_bytes()
                            ).hexdigest(),
                        )
                    )
                maps.append(np.mean(stack, axis=0))
                titles.append(f"{prefix} {group} · n={len(stack)}")
                keys.append(f"{side}_{prefix.lower()}_{group}")
        vmax = max(m.max() for m in maps)
        for col, (data, title, key) in enumerate(zip(maps, titles, keys)):
            ax = axs[r, col]
            im = ax.imshow(
                data,
                origin="upper",
                extent=[xe[0], xe[-1], ye[-1], ye[0]],
                norm=PowerNorm(0.4, vmin=0, vmax=vmax),
                cmap="magma",
            )
            ax.set_title(side.capitalize() + " · " + title)
            ax.set_xlabel("Arena x (mm)")
            ax.set_ylabel("Arena y (mm)")
            mapdata[key] = data
        for col in range(len(maps), 2 + maxk):
            axs[r, col].axis("off")
        fig.colorbar(
            im, ax=axs[r, :], shrink=0.6, label="Mean probability per 0.25 mm bin"
        )
        mapdata[f"{side}_x_edges"] = xe
        mapdata[f"{side}_y_edges"] = ye
    np.savez_compressed(output / "mean_occupancy.npz", **mapdata)
    pd.DataFrame(map_sources).drop_duplicates("ant").to_csv(
        output / "occupancy_sources.csv", index=False
    )
    fig.suptitle(
        "5  Spatial occupancy of groups learned from posture and velocity", fontsize=17
    )
    save(
        fig,
        "05_spatial_occupancy",
        "Group maps average each ant’s normalized occupancy equally. The spatial and learned-feature maps use the same ants with an existing spatial reference; any extra eligible ants remain in the fit and are listed separately in the comparison CSV. The color scale is shared within colony. These full-recording occupancy maps are post-fit validation; they are not independent held-out spatial data.",
    )

    fig, axs = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    heatmap_order = []
    for r, side in enumerate(["left", "right"]):
        part = a[a.side.eq(side)].sort_values(["cluster", "track_id"])
        ix = [int(np.flatnonzero(q.ant.eq(ant))[0]) for ant in part.ant]
        for order, ant in enumerate(part.ant):
            heatmap_order.append(
                dict(
                    side=side, row=order, ant=ant, cluster=int(part.iloc[order].cluster)
                )
            )
        features = ["forward_mean"] + [
            f"posture_pc{int(mode)+1}_mean" for mode in shown_modes[side]
        ]
        for col, name in enumerate(features):
            ax = axs[r, col]
            data = hourly[ix, :, list(names).index(name)]
            low, high = np.nanquantile(data, [0.02, 0.98])
            im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=low, vmax=high)
            ax.axvline(23.5, c="white", lw=2)
            for boundary in np.flatnonzero(np.diff(part.cluster.to_numpy())) + 0.5:
                ax.axhline(boundary, c="white", lw=1.5)
            ax.set_xticks(
                [0, 12, 24, 36, 47],
                ["Jul24 10h", "22h", "Jul25 10h", "22h", "Jul26 09h"],
            )
            ax.set_ylabel(
                f"{side.capitalize()} · {len(ix)} ants ordered by learned group"
            )
            ax.set_title(name.replace("_", " "))
            fig.colorbar(im, ax=ax, shrink=0.75)
    pd.DataFrame(heatmap_order).to_csv(output / "heatmap_ant_order.csv", index=False)
    fig.suptitle(
        "6  Hourly posture and velocity profiles across training and held-out days",
        fontsize=17,
    )
    save(
        fig,
        "06_hourly_learned_profiles",
        "Rows are ants sorted by their day-1 physical group; the row-to-ant lookup is supplied. Columns are hourly means, with the white vertical line separating training and held-out days. White cells are missing, not zero. Displayed posture modes are selected by physical-profile loadings only. Clustering uses distributions of hourly measurements; it does not fit spatial occupancy or the order of hours.",
    )
    pdf.close()
    make_report(
        output,
        gallery,
        q,
        basis,
        reconstruction,
        interpretation,
        summary,
        metrics,
        selection,
    )


def make_report(
    output,
    gallery,
    q,
    basis,
    reconstruction,
    interpretation,
    summary,
    metrics,
    selection,
):
    rank = int(basis["rank"])
    rec = reconstruction[reconstruction["rank"].eq(rank)].iloc[0]
    basis_summary = json.loads((output / "landmark_pca_summary.json").read_text())
    boot = pd.read_csv(output / "landmark_basis_bootstrap.csv")
    findings = []
    for side in ["left", "right"]:
        m = metrics[metrics.side.eq(side) & metrics.family.eq("joint")].iloc[0]
        s = summary[summary.side.eq(side) & summary.family.eq("joint")].iloc[0]
        if m.k == 1:
            findings.append(
                f"{side.capitalize()}: no supported split (K=1); this does not recover the two spatial classes."
            )
        else:
            findings.append(
                f"{side.capitalize()}: selected K={int(m.k)}; {int(round(m.accuracy*m.n_compared))}/{int(m.n_compared)} spatial agreements ({m.accuracy:.1%}, ARI {m.ari:.3f}); BIC improvement over K=1 {s.bic_improvement:.2f}."
            )
    diagnostic_text = ""
    if (output / "rank_sensitivity.csv").exists():
        sensitivity = pd.read_csv(output / "rank_sensitivity.csv")
        primary = sensitivity[sensitivity.variance_target.eq("90%")]
        matches = "; ".join(
            f"{row.side} {int(round(row.candidate_k2_accuracy*row.n_compared))}/{int(row.n_compared)}"
            for row in primary.itertuples()
        )
        diagnostic_text = f"The fixed K=2 candidate resembles the spatial partition ({matches}), but the right-colony candidate fails the ant-bootstrap criterion. It is a diagnostic candidate, not the selected right-colony classification. Retaining 95% of coordinate variance (five modes) or all twelve modes selects K=1 in both colonies. Thus the learned posture structure is reproducible, but a unique two-class ant partition is not robust under this workflow."
    lines = (
        [
            "# Antennal eigenpostures and the two spatial classes",
            "",
            "This analysis replaces the predefined posture descriptors with modes learned directly from six antennal landmark positions. No straightness, spread, asymmetry, spatial position or spatial class enters fitting. The dataset has been studied previously, so this is exploratory validation, not a first look at independent data.",
            "",
            "## Result",
            "",
        ]
        + ["- " + f for f in findings]
        + [
            "",
            diagnostic_text,
            "",
            f'{rank} learned posture modes capture {basis["variance_ratio"][:rank].sum():.1%} of day-1 coordinate variance and {rec.coordinate_variance_explained:.1%} on held-out day 2. The largest even/odd-minute subspace angle is {max(basis_summary["split_minute_subspace_angles_degrees"]):.2f}°; the median maximum angle under ant bootstrap is {boot.max_angle_degrees.median():.2f}°. These checks concern the learned posture basis, not the number of ant classes.',
            "",
            "## Does straightness emerge?",
            "",
            f"Straightness is calculated only after the unsupervised model has been frozen. Reconstructing the landmarks from the {rank} learned modes preserves straightness with held-out weighted R²={rec.straightness_r2:.3f}. The per-mode correlations below show whether it corresponds to a single linear mode or a combination. This is an interpretation of the learned geometry, not a feature used to obtain the clusters.",
            "",
            "| Mode | Coordinate variance | Held-out correlation with straightness |",
            "|---|---:|---:|",
        ]
    )
    for mode in range(rank):
        lines.append(
            f'| P{mode+1} | {basis["variance_ratio"][mode]:.1%} | {interpretation.heldout_straightness_r.iloc[mode]:+.3f} |'
        )
    lines += [
        "",
        "A PCA mode is a linear displacement of landmark positions. A scalar such as straightness is nonlinear and can depend on several mode amplitudes. Coordinate PCA also retains real size differences and measurement-driven changes in segment length after fixed body-length normalization; it does not enforce a rigid articulated skeleton. The reconstruction plots make these variations visible.",
        "",
        "## Two separate PCA stages",
        "",
        "1. **Posture basis:** head-relative antennal landmarks in the body-axis frame, divided by each ant’s fixed day-1 median petiole-to-tag length. Four-frame causal averaging reduces tracking jitter. PCA uses day 1 with equal colony, ant, observed-hour, minute and valid-sample weights. The smallest rank explaining at least 90% of coordinate variance is retained. No per-ant mean posture is removed.",
        "2. **Ant phenotype:** for each sampled clip, measure the mean amplitude of every retained mode and RMS within-clip amplitude derivative (12 Hz), plus signed mean and RMS anterior/lateral velocity. Give minute clips equal weight within each hour. The q25/median/q75 of hourly measurements form an ant profile. Transform velocity tails and positive mode rates, standardize every profile variable, then learn the dominant shared profile PC. A one-dimensional unequal-variance Gaussian mixture groups ants on this common axis. Both PCA stages are learned without spatial labels.",
        "",
        "The one-dimensional ant phenotype is a simplicity choice, distinct from the multi-dimensional posture basis. Failure to find two groups in this representation would not prove that no higher-dimensional behavioral partition exists. RMS rates measure short-timescale fluctuations; they do not recover full phase dynamics, motifs or equations of motion.",
        "",
        "## Sampling and coverage",
        "",
        f"The source contains {len(q)} identities, 57 per colony. Training eligibility is determined from observation availability alone: at least 12 hours for posture, dynamics and velocity, with at least five observed minutes per hour and eight valid posture/velocity windows per clip (seven derivative pairs). {int(q.day1_eligible.sum())} ants qualify: "
        + ", ".join(
            f"{side} {int((q.side.eq(side)&q.day1_eligible).sum())}"
            for side in ["left", "right"]
        )
        + ". Every identity remains in all_ant_coverage.csv. Day-2 availability does not select training ants.",
        "One random 2.5-second clip is sampled per minute. Day 1 spans July 24 10:00–July 25 10:00 JST; day 2 is the next 24 hours. Landmark positions are reconstructed from unfitted cached segment vectors and lengths; all 114 source tracking file signatures were checked. Missing samples are not zeros. Geometry checks, duplicates, camera transitions and mismatched velocity/pose cameras are excluded locally.",
        "",
        "## K selection and validation",
        "",
        "The rule was recorded before this fit: compare K=1–4; require any BIC improvement over K=1, bootstrap median ARI ≥0.8, independently learned even/odd-minute ARI ≥0.6, and every group at least max(4,10% of ants). Select the smaller eligible K within two BIC units of the best, otherwise K=1. This repeats the preceding analysis’s rule for comparability. A BIC difference below two is inconclusive even if the rule returns two groups; class correspondence and evidence for exactly two classes are separate claims.",
        "",
        "| Colony | Inputs | K | Spatial agreement | Day-2 fixed labels | Day-2 phenotype refit ARI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        m = metrics[metrics.side.eq(row.side) & metrics.family.eq(row.family)].iloc[0]
        lines.append(
            f"| {row.side} | {row.family} | {row.selected_k} | "
            + (f"{m.accuracy:.1%}" if row.selected_k > 1 else "unsplit")
            + " | "
            + (
                f"{row.day2_frozen_retention:.1%} | {row.day2_refit_ari:.3f}"
                if row.selected_k > 1
                else "n/a | n/a"
            )
            + " |"
        )
    lines += [
        "",
        "Phenotype bootstrap repeats the ant-profile scaling/PCA/mixture with the landmark basis fixed; separate ant bootstrap checks the posture subspace. Even/odd-minute comparisons learn the landmark PCA independently before profile fitting. Day 2 does not enter training or K selection. Its independent phenotype refit still uses the frozen day-1 posture basis.",
        "",
        "## Limits",
        "",
        "The full-recording spatial maps are an external feature comparison after fitting, but overlap the physical data in time; they are not an independent spatial holdout. Day-2 checks use the same ants. Location-dependent visibility and camera-dependent skeleton errors can still affect posture; body-relative coordinates alone do not eliminate that confounding. Camera and geometric quality checks reduce obvious artifacts but do not establish intrinsic biological ant types. The small, two-colony cohort also limits inference about a unique K.",
        "",
        "The method follows the learn-shape-first principle of [Stephens et al. (2008)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000028). Their eigenworms used body tangent angles; this analysis uses antennal landmark coordinates and does not claim to reproduce their dynamical-system inference.",
        "",
        "## Files",
        "",
        "- [All figures PDF](eigenposture_figures.pdf)",
        "- [Ant assignments and spatial comparison](spatial_comparison.csv)",
        "- [Learned-mode ant measurements](ant_mode_measurements.csv)",
        "- [K selection](k_selection.csv)",
        "- [Rank sensitivity and explicitly diagnostic K=2 fits](rank_sensitivity.csv)",
        "- [Coverage of all identities](all_ant_coverage.csv)",
        "- [Held-out mode interpretation](mode_interpretation.csv)",
        "- [Reconstruction accuracy](heldout_reconstruction.csv)",
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
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    cards = "".join(
        f'<section><a href="figures/{name}.pdf"><img src="figures/{name}.png" alt="{name}"></a><p>{html.escape(caption)}</p></section>'
        for name, caption in gallery
    )
    page = (
        '<!doctype html><html><head><meta charset="utf-8"><title>July 24 antennal eigenpostures</title><style>body{font:17px/1.5 system-ui;max-width:1450px;margin:30px auto;padding:0 24px;background:#f5f7fa;color:#23323e}section{background:white;padding:20px;margin:26px 0;border-radius:8px}img{width:100%}a{color:#196699}.note{padding:18px;background:#fff0d5}</style></head><body><h1>Antennal eigenpostures → physical groups → spatial comparison</h1><p>Learned landmark modes replace predefined shape descriptors.</p><ul>'
        + "".join("<li>" + html.escape(f) + "</li>" for f in findings)
        + f'</ul><p>{rank} learned modes explain {basis["variance_ratio"][:rank].sum():.1%} of posture variance. Straightness is examined only afterward: held-out reconstruction R²={rec.straightness_r2:.3f}.</p><p class="note">'
        + html.escape(diagnostic_text)
        + ' Clustering correspondence and evidence for exactly two classes are different questions. Small BIC improvements are inconclusive. This is exploratory analysis of two colonies, with same-ant day-2 checks and remaining camera/visibility confounds.</p><p><a href="REPORT.md">Full report</a> · <a href="eigenposture_figures.pdf">All figures PDF</a> · <a href="spatial_comparison.csv">Assignments</a> · <a href="reproduction/README.md">Reproduce</a></p>'
        + cards
        + "</body></html>"
    )
    (output / "index.html").write_text(page)
    (output / "REPORT_COMPLETE.json").write_text(
        json.dumps(
            dict(
                created_at=datetime.now(timezone.utc).isoformat(),
                figures=len(gallery),
                rank=rank,
                straightness_heldout_r2=float(rec.straightness_r2),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
    a = p.parse_args()
    build(a.output, a.spatial_reference, a.occupancy_root)


if __name__ == "__main__":
    main()
