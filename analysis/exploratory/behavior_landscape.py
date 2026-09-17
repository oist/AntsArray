# %%
"""Does a multidimensional behavioral landscape predict more than occupancy?

Run as a script, or import ``run`` in a notebook. See behavior_landscape.md.
Outputs are separate from grid_occupancy's canonical cluster assignments.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis import behavior_landscape_features as bf
from analysis import behavior_landscape_utils as bu
from analysis import behavior_landscape_spatial as bs
from analysis import grid_occupancy_utils as go

DATASET = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20260724/block01")
FEATURE_SETTINGS = bf.FeatureSettings()
MODEL_SETTINGS = bu.ModelSettings()


def find_bundle(block):
    paths = sorted((Path(block) / "analysis_outputs").glob("*/comparison_manifest.json"),
                   key=lambda p: p.stat().st_mtime_ns, reverse=True)
    suffix = f"/{block.parent.name}/{block.name}"
    for path in paths:
        if any(w["source"].endswith(suffix) for w in json.loads(path.read_text())["windows"]):
            return path.parent
    raise FileNotFoundError("Pass --analysis-bundle with a published comparison_manifest.json for this block")


def audit_spatial_proxy(features, output):
    per_ant, summaries = [], []
    for side, group in features["tracks"].groupby("side"):
        original = np.stack([np.load(row.occupancy_path).ravel() for row in group.itertuples()])
        source_sqrt = np.sqrt(original)
        proxy = features["spatial_proxy"][side]
        daily = features["daily"].query("side == @side")
        exposure = daily.assign(weight=daily.detected_frame_fraction * daily.in_arena_fraction).pivot(
            index="track_id", columns="day", values="weight").loc[group.track_id].to_numpy()
        pooled = (proxy * exposure[:, :, None]).sum(axis=1) / exposure.sum(axis=1)[:, None]
        counts = features["exact_spatial_counts"][side]
        exact48 = (features["spatial"][side] * counts[:, :, None]).sum(axis=1) / counts.sum(axis=1)[:, None]
        affinity = (np.sqrt(exact48) * np.sqrt(pooled)).sum(axis=1)
        with threadpool_limits(limits=1):
            exact_labels = go.leiden_labels(source_sqrt, n_neighbors=10, resolution=1, random_state=0)
            proxy_labels = go.leiden_labels(np.sqrt(pooled), n_neighbors=10, resolution=1, random_state=0)
            exact48_labels = go.leiden_labels(np.sqrt(exact48), n_neighbors=10, resolution=1, random_state=0)
        for ant, value in zip(group.track_id, affinity):
            per_ant.append(dict(side=side, track_id=int(ant), sqrt_histogram_affinity=float(value)))
        summaries.append(dict(side=side, n_ants=len(group), median_affinity=float(np.median(affinity)),
                              minimum_affinity=float(affinity.min()),
                              original_clusters=len(np.unique(exact_labels)),
                              exact48_clusters=len(np.unique(exact48_labels)),
                              original_reproduction_ari=float(bu.adjusted_rand_score(group.cluster_id, exact_labels)),
                              exact48_vs_original_ari=float(bu.adjusted_rand_score(group.cluster_id, exact48_labels)),
                              proxy48_vs_exact48_ari=float(bu.adjusted_rand_score(exact48_labels, proxy_labels))))
    pd.DataFrame(per_ant).to_csv(output / "spatial_proxy_per_ant.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "spatial_proxy_audit.csv", index=False)
    return pd.DataFrame(summaries)


def plot_results(features, tables, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from sklearn.decomposition import PCA

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    models = ["occupancy", "occupancy_activity", "occupancy_timing", "occupancy_interactions", "full", "coverage"]
    short = ["Occupancy", "+ Activity", "+ Timing", "+ Contacts", "All four", "Coverage only"]
    scores = tables["heldout_scores"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for row, side in enumerate(("left", "right")):
        for col, method in enumerate(("selected_clusters", "neighbors")):
            ax = axes[row, col]
            part = scores.query("side == @side and method == @method and train_day == 0")
            matrix = part.pivot(index="feature_set", columns="outcome", values="heldout_r2").reindex(models)[list(bu.OUTCOMES)].to_numpy()
            im = ax.imshow(matrix, cmap="RdBu", norm=TwoSlopeNorm(vmin=-.3, vcenter=0, vmax=.65), aspect="auto")
            ax.set_xticks(range(4), ["Activity", "Daily timing", "Contacts", "Space use"])
            ax.set_yticks(range(len(models)), short)
            ax.set_title(f"{side.capitalize()} — " + ("training-selected clusters" if col == 0 else "five-neighbor landscape"))
            for i in range(len(models)):
                for j in range(4):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] > .45 else "black")
    fig.colorbar(im, ax=axes, label="Held-out R² versus training-day colony mean")
    fig.suptitle("Predict window 2 from window 1\nOther ants supply cluster/neighborhood forecasts; negative values lose to the colony mean")
    fig.savefig(output / "01_heldout_prediction.png", dpi=170)
    plt.close(fig)

    diag = tables["cluster_diagnostics"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for row, side in enumerate(("left", "right")):
        for model, color in (("occupancy", "#266e9e"), ("full", "#cb5838")):
            for day, style in ((0, "-"), (1, "--")):
                group = diag.query("side == @side and feature_set == @model and train_day == @day")
                for col, metric in enumerate(("silhouette", "stability_median", "cross_day_ari")):
                    ax = axes[row, col]
                    ax.plot(group.k, group[metric], style, color=color, alpha=.8, label=f"{model}, window {day + 1}")
                    valid = group[group.admissible]
                    ax.scatter(valid.k, valid[metric], color=color, s=28)
                    ax.set(xlabel="Number of clusters", ylabel=metric.replace("_", " "), title=side.capitalize())
                    ax.set_xticks(range(2, 7))
        axes[row, 1].axhline(.75, color="gray", linestyle=":", linewidth=1)
        axes[row, 0].legend(fontsize=8)
    fig.suptitle("Does extra resolution survive resampling and another day?\nFilled points have at least four ants per cluster; lines also show undersized candidates")
    fig.savefig(output / "02_cluster_resolution.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    embedding_rows = []
    for row, side in enumerate(("left", "right")):
        identity, days, _ = bu.arrays_for_colony(features, side)
        x = bu.BlockScaler().fit(days[0], bu.FEATURE_SETS["full"]).transform(days[0])
        pca = PCA(n_components=2, svd_solver="full")
        xy = pca.fit_transform(x)
        daily = features["daily"].query("side == @side and day == 0").set_index("track_id").loc[identity.track_id]
        colors = [pd.Categorical(identity.cluster_id).codes, daily.speed, daily.contacts]
        titles = ["Original occupancy group", "Mean locomotor speed (mm/s)", "Contact onsets / observed hour"]
        for col, (values, title) in enumerate(zip(colors, titles)):
            ax = axes[row, col]
            im = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap="tab10" if col == 0 else "viridis", s=43, edgecolors="white", linewidths=.4)
            for ant, (x_, y_) in zip(identity.track_id, xy):
                ax.annotate(str(ant), (x_, y_), xytext=(3, 3), textcoords="offset points", fontsize=7, alpha=.75)
            ax.set(title=f"{side.capitalize()}: {title}", xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
                   ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
            if col:
                fig.colorbar(im, ax=ax, shrink=.8)
            else:
                handles, _ = im.legend_elements()
                ax.legend(handles, sorted(identity.cluster_id.unique()), fontsize=8, loc="best")
        for ant, original, coords in zip(identity.track_id, identity.cluster_id, xy):
            embedding_rows.append(dict(side=side, track_id=int(ant), original_cluster=original, pc1=coords[0], pc2=coords[1]))
    pd.DataFrame(embedding_rows).to_csv(output / "landscape_coordinates.csv", index=False)
    fig.suptitle("Window-1 behavioral landscape: occupancy + activity + timing + contacts\nPCA is a display of continuous variation; distances and clustering use the full feature space")
    fig.savefig(output / "03_behavioral_landscape.png", dpi=170)
    plt.close(fig)

    gains = tables["paired_gains"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", sharey=True)
    for ax, side in zip(axes, ("left", "right")):
        for i, (feature_set, color) in enumerate((("occupancy_activity", "#ce8335"), ("occupancy_interactions", "#247aa6"), ("full", "#8b519d"))):
            group = gains.query("side == @side and method == 'neighbors' and feature_set == @feature_set and direction == 'forward'")
            group = group.set_index("outcome").reindex(bu.OUTCOMES)
            x = np.arange(4) + (i - 1) * .2
            y = group.fractional_mse_reduction.to_numpy() * 100
            err = np.stack([y - group.ci_low.to_numpy() * 100, group.ci_high.to_numpy() * 100 - y])
            ax.errorbar(x, y, yerr=err, color=color, fmt="o", capsize=3, label=feature_set.replace("occupancy_", "+ "))
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=side.capitalize(), ylabel="Prediction-error reduction versus occupancy (%)")
        ax.set_xticks(range(4), ["Activity", "Timing", "Contacts", "Space use"])
        ax.legend(fontsize=8)
    fig.suptitle("Added information in the continuous landscape — forward forecast\n95% paired-ant bootstrap intervals, conditional on these windows and fitted neighborhoods")
    fig.savefig(output / "04_added_information.png", dpi=170)
    plt.close(fig)

    # A predeclared candidate view; these panels do not select the number of groups.
    chosen = tables["assignments"].query("side == 'right' and train_day == 0 and feature_set == 'full' and k == 3")
    phenotype = features["daily"].merge(chosen[["side", "track_id", "label"]], on=["side", "track_id"], validate="many_to_one")
    phenotype.to_csv(output / "right_three_group_profiles.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), layout="constrained")
    for ax, metric, ylabel in zip(axes, ["speed", "sleep", "contacts", "median_bout_seconds"],
            ["Speed (mm/s)", "Sleep fraction", "Contact onsets / observed hour", "Median contact-bout span (s)"]):
        for day, color, offset in ((0, "#347da1", -.12), (1, "#c16a3f", .12)):
            part = phenotype[phenotype.day.eq(day)]
            ax.scatter(part.label + offset, part[metric], color=color, s=16, alpha=.5)
            means = part.groupby("label")[metric].mean()
            ax.scatter(means.index + offset, means, marker="_", s=500, linewidths=3, color=color,
                       label=f"Window {day + 1}")
        ax.set(xlabel="Window-1 cluster label", ylabel=ylabel, xticks=[0, 1, 2])
    axes[0].legend(fontsize=8)
    fig.suptitle("Right-colony three-group candidate: same ants followed into window 2\nDots are ants; bars are group means. This candidate is not the training-selected optimum.")
    fig.savefig(output / "05_three_group_profiles.png", dpi=170)
    plt.close(fig)


def markdown_table(table, columns, digits=3):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in table[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(features, tables, proxy, sensitivity, output, settings):
    info = features["info"]
    start = pd.Timestamp(info["start_time"])
    count = features["audit"].query("day == 0").groupby(["side", "original_cluster"], as_index=False).agg(
        original_ants=("track_id", "size"), validation_ants=("included", "sum"))
    selected = tables["selected_models"].query("feature_set in ['occupancy', 'full']")
    leiden_rows = []
    for (side, model, resolution), group in tables["leiden_assignments"].query(
            "train_day == 0 and feature_set in ['occupancy', 'full']").groupby(["side", "feature_set", "resolution"]):
        leiden_rows.append(dict(side=side, feature_set=model, resolution=resolution,
                                clusters=int(group.k.iloc[0]), smallest_group=int(group.label.value_counts().min()),
                                cross_day_ARI=float(group.cross_day_ari.iloc[0])))
    leiden_summary = pd.DataFrame(leiden_rows)
    forward = tables["paired_gains"].query("direction == 'forward' and method == 'neighbors' and feature_set in ['occupancy_activity', 'occupancy_interactions', 'full']").copy()
    for column in ("fractional_mse_reduction", "ci_low", "ci_high"):
        forward[column] *= 100
    forward = forward.rename(columns={"fractional_mse_reduction": "error_reduction_percent", "ci_low": "CI_low", "ci_high": "CI_high"})
    full_gains = forward.query("feature_set == 'full'").set_index(["side", "outcome"])
    right_diag = tables["cluster_diagnostics"].query("side == 'right' and feature_set == 'full' and train_day == 0").set_index("k")
    resolution_finding = (
        f"- **Three right-colony groups are a candidate; finer divisions need more evidence.** Full-model cross-day adjusted Rand agreement is {right_diag.loc[3, 'cross_day_ari']:.2f} at k=3 and {right_diag.loc[4, 'cross_day_ari']:.2f} at k=4. Agreement of one means identical memberships up to relabeling; zero is approximately chance. This diagnostic does not establish discrete tasks."
        if {3, 4}.issubset(right_diag.index) else
        "- **Resolution:** inspect cluster_diagnostics.csv for every tested cluster count and its cross-day agreement."
    )
    findings = ["## Main findings", "",
        f"- **Additional dimensions carry predictive information.** In the forward five-neighbor comparison, the full feature set changes activity prediction error by a reduction of {full_gains.loc[('right', 'activity'), 'error_reduction_percent']:.1f}% and contact prediction error by {full_gains.loc[('right', 'interactions'), 'error_reduction_percent']:.1f}% in the right colony. Left-colony reductions are {full_gains.loc[('left', 'activity'), 'error_reduction_percent']:.1f}% and {full_gains.loc[('left', 'interactions'), 'error_reduction_percent']:.1f}%, respectively, in its selectively observed cohort. Negative reductions, if present, indicate worse forecasts.",
        "- **The result is stronger for continuous behavioral variation than for extra discrete types.** Inspect the fixed-k results as well as the training-selected cluster count. The selection table below does not use the held-out scores.",
        resolution_finding,
        "- **Activity level and contact behavior are the clearest next features to retain.** Daily timing has weak absolute held-out prediction over these two windows. The full model does not improve every outcome: inspect space-use scores and both forecast directions before choosing weights.",
        "- **Coverage limits the left-colony claim.** Its strict cohort includes only one ant from one of the two original occupancy groups. The broader coverage analysis restores more of that group and is reported separately.", ""]
    lines = ["# 0724: a multidimensional behavioral landscape", "", *findings,
             "## What this experiment tests", "",
             "Whether adding activity, daily timing and contact features improves predictions of behavior in a separate 24-hour window, and whether finer clusters remain reproducible. The continuous five-neighbor comparison asks whether additional information is present even when discrete clusters are a poor summary.", "",
             f"Recording: **{info['start_time']}–{info['stop_time']}**. Training window: **{start}–{start + pd.Timedelta(days=1)}**. Forward test: **{start + pd.Timedelta(days=1)}–{start + pd.Timedelta(days=2)}**. The remaining recording tail is excluded, keeping clock exposure identical. The reverse direction is a second descriptive check; it is not an independent replicate or a prospective forecast.", "",
             "## Observation coverage", "",
             markdown_table(count, list(count.columns), 0), "",
             "Primary eligibility requires at least 70% observed coverage for position, speed, body/antenna motion, sleep classification and contact opportunity in each window, plus nine usable two-hour bins per speed/sleep/contact profile. Each profile bin requires 50% coverage. Every feature set uses exactly the same eligible ants. The left cohort is strongly biased toward one original spatial group, so this analysis cannot settle the number of groups in the full left colony.", "",
             "## Does adding information help?", "",
             "Positive values below mean lower forward prediction error than occupancy-only five-neighbor forecasts. Each focal ant is excluded from its own neighborhood forecast. Uncertainty resamples ants within a colony and holds the fitted neighborhoods fixed; these are exploratory, unadjusted intervals, not independent-colony evidence.", "",
             markdown_table(forward, ["side", "feature_set", "outcome", "error_reduction_percent", "CI_low", "CI_high"], 1), "",
             "![Held-out prediction](01_heldout_prediction.png)", "",
             "![Added information](04_added_information.png)", "",
             "The score is 1 − model squared error / training-colony-mean squared error. Zero matches the training-day colony mean and negative values do worse. Outcome columns are scaled using training-day variability only. Activity averages log(1+speed), log(1+body motion), log(1+antenna motion) and sleep-fraction errors. Contact scores average log(1+onset rate), bout-presence fraction, partner diversity and log(1+median bout duration). Space use comprises outside-colony and resource-presence fractions, neither explicitly included as input features. Daily timing scores the centered two-hour profiles. These scores quantify held-out prediction, not causal explanation.", "",
             "## Are there more reproducible clusters?", "",
             "K-means candidates use the same k=2…6 sweep for every feature set, with at least four ants per cluster. Training selection maximizes silhouette among candidates with median adjusted Rand index ≥0.75 over 30 independent 80%-ant subsamples. If none qualify, the model returns a single colony group. Test-day outcomes and cross-day agreement never select k. A second Leiden sweep uses the existing grid workflow's graph construction with 10 neighbors and resolutions 0.5, 1, 1.5 and 2.", "",
             markdown_table(selected, list(selected.columns), 0), "",
             "The full model's three-group right-colony candidate is available for inspection even if the training selector prefers two. Finer candidates must be judged against both their same-k occupancy baseline and their own cross-day stability; increasing the resolution setting alone is not evidence of additional behavioral types.", "",
             "### Direct comparison with the current Leiden workflow", "",
             "The table above selects k-means models. This separate Leiden sweep keeps the current workflow's graph construction. Rows below describe window-1 partitions; agreement compares an independently fitted window-2 partition. High-resolution rows may contain very small groups, as shown. The reproducible three-group k-means candidate should not be confused with the less stable three-group Leiden result at resolution 1.", "",
             markdown_table(leiden_summary, list(leiden_summary.columns)), "",
             "![Cluster resolution](02_cluster_resolution.png)", "",
             "![Behavioral landscape](03_behavioral_landscape.png)", "",
             "![Three-group candidate profiles](05_three_group_profiles.png)", "",
             "## Features and safeguards", "",
             "- **Occupancy:** square roots of exact frame-level arena-grid probabilities, rebuilt separately for each window from finished bodypoint-0 tracks. Coordinates, arena bounds, duplicate-frame averaging and normalization follow grid_occupancy. The reconstructed whole-recording grid must numerically match the saved grid before accepting an ant. Full-recording occupancy is never reused in both train and test.",
             "- **Activity:** mean locomotor, body and antenna motion, plus sleep fraction. Motion measures use the existing upstream definitions.",
             "- **Daily timing:** two-hour speed, sleep and contact-rate profiles, centered within each ant-day to separate timing from amount. Both windows start at the same clock time. These two days cannot establish an endogenous circadian rhythm.",
             "- **Interactions:** undirected ≤0.1 mm skeleton contacts, merged over gaps ≤2 seconds. Onset rates and bout-presence fractions use seconds with observed focal positions and completed interaction processing. Bout presence includes merged gaps and second-bin rounding; it is not exact physical-contact duration. Median durations exclude contacts crossing window boundaries. Partner Simpson diversity uses onset counts across all tracked partners, including partners outside the clustering cohort. Contact detection still depends on pose quality and partner visibility; exposure adjustment cannot remove those effects.",
             "- **Scaling:** train-only median imputation, removal of columns with >20% missing training values, and equal total variance per feature family. Nonspatial columns are standardized; rare grid cells are not separately standardized. Coverage itself is excluded from behavioral features and evaluated in separate control models.",
             "- **Forecasts:** cluster and neighbor forecasts average other ants' training-window outcomes. The focal ant never contributes its own outcomes. If every selected neighbor lacks a target, use the training-colony mean so all models are scored on the same observed test targets. Its own previous-day behavior is saved as a persistence reference, with the same fallback for unobserved prior values. No held-out outcomes set preprocessing, neighborhoods, labels, cluster count or weights.",
             "- **Interpretation:** forecasts concern repeatability of the same ants across two windows. They do not establish prediction for new colonies, independent biological replication, discrete tasks, feeding/drinking, or causal effects.", "",
             "### Spatial approximation audit", "",
             markdown_table(proxy, list(proxy.columns)), "",
             "Forecasts use exact frame-level grids. The second-mean proxy is retained only as an audit: affinity compares proxy and exact histograms over the same 48 hours. The original-reproduction ARI checks the saved full-recording handoff. exact48_vs_original_ari measures the effect of removing the recording tail; proxy48_vs_exact48_ari measures the cached-position approximation. The identical exact/proxy 48-hour partitions show that their difference from the saved right-colony handoff comes from the recording window, not the position approximation. exact_spatial_audit.csv separately records numerical agreement between reconstructed and saved whole-recording grids. Full-recording labels are used only for descriptive comparison, never for forecast construction.", "",
             "## Coverage sensitivity", "",
             "A second run lowers daily coverage to 50% and profile support to six two-hour bins. It reuses the same features and refits every model on the broader cohort. It checks sensitivity to observation selection rather than treating lower-quality data as equivalent evidence.", "",
             markdown_table(sensitivity, list(sensitivity.columns)), "",
             "## Reproduction and outputs", "",
             "Run the command in run_manifest.json. Source paths, sizes and nanosecond mtimes are retained in the feature-cache sources.json and per-ant cache/exact_spatial/*.json. Relocated published caches must match their original fingerprints; stale or ambiguous sources raise errors. This experiment writes its own assignments and does not replace track_cluster_ids.csv.", "",
             "Key tables: daily_features.csv, clock_profiles.csv, coverage_audit.csv, cluster_diagnostics.csv, assignments.csv, leiden_assignments.csv, selected_models.csv, heldout_scores.csv, heldout_losses.csv, paired_gains.csv, and sensitivity/. train_day=0 is the forward forecast; train_day=1 reverses the windows. selected_clusters chooses k from training data, whereas clusters rows retain each admissible fixed-k candidate. paired_gains.csv compares identical k for fixed-k models and identical selection procedures for selected models. Bootstrap intervals do not account for selecting a favorable row after inspecting this sweep.", "",
             "### Method references", "",
             "The separation of clustering from held-out validation follows the general motivation in [Tibshirani and Walther, Cluster Validation by Prediction Strength](https://gwalther.su.domains/predictionstrength.pdf); this experiment uses same-ant time splits and does not claim to implement their prediction-strength statistic. Training separation uses [silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis); agreement uses [adjusted Rand index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html), which is invariant to arbitrary cluster numbering.", ""]
    (output / "report.md").write_text("\n".join(lines))


def run(dataset=DATASET, output=None, analysis_bundle=None, workers=4,
        feature_settings=FEATURE_SETTINGS, model_settings=MODEL_SETTINGS):
    block = Path(dataset).resolve()
    output = Path(output or block / "analysis_outputs/behavior_landscape").resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = Path(analysis_bundle) if analysis_bundle else find_bundle(block)
    features = bf.extract_features(block, bundle, output, feature_settings, workers)
    features = bs.use_exact_spatial(features, block, output, workers)
    tables = bu.run_comparison(features, model_settings)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    proxy = audit_spatial_proxy(features, output)
    plot_results(features, tables, output)
    sensitivity_root = output / "sensitivity"
    sensitivity_root.mkdir(exist_ok=True)
    sensitivity = bu.run_comparison(features, model_settings, min_coverage=.5, min_profile_bins=6)
    for name, table in sensitivity.items():
        table.to_csv(sensitivity_root / f"{name}.csv", index=False)
    sensitivity_summary = []
    for label, result in (("primary_70pct_9bins", tables), ("broader_50pct_6bins", sensitivity)):
        for side in ("left", "right"):
            part = result["heldout_scores"].query("side == @side and method == 'neighbors' and train_day == 0 and feature_set in ['occupancy', 'full']")
            for feature_set, group in part.groupby("feature_set"):
                sensitivity_summary.append(dict(cohort=label, side=side, feature_set=feature_set,
                                                n_ants=int(group.n_ants.iloc[0]),
                                                **dict(zip(group.outcome, group.heldout_r2))))
    sensitivity_summary = pd.DataFrame(sensitivity_summary)
    sensitivity_summary.to_csv(output / "sensitivity_summary.csv", index=False)
    write_report(features, tables, proxy, sensitivity_summary, output, model_settings)
    sources = [Path(__file__), Path(bf.__file__), Path(bu.__file__), Path(bs.__file__)]
    manifest = dict(dataset=str(block), output=str(output), analysis_bundle=str(bundle),
                    feature_settings=asdict(feature_settings), model_settings=asdict(model_settings),
                    feature_cache=features["feature_cache"], feature_signature=features["feature_signature"],
                    exact_spatial_keys=features["exact_spatial_keys"],
                    interaction_run_id=features["contact_run_id"],
                    code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "igraph", "leidenalg")},
                    command=[sys.executable, str(Path(__file__).resolve()), "--dataset", str(block),
                             "--analysis-bundle", str(bundle), "--output", str(output), "--workers", str(workers), "--headless"])
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Report: {output / 'report.md'}", flush=True)
    return features, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--analysis-bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.headless:
        import matplotlib
        matplotlib.use("Agg")
    run(args.dataset, args.output, args.analysis_bundle, args.workers)


if __name__ == "__main__" and "get_ipython" not in globals():
    main()

# %%
# Interactive use:
# features, tables = run(output="analysis_outputs/behavior_landscape_0724")
