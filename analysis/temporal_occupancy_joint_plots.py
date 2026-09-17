"""Behavior-first figures for joint occupancy clusters; UMAP is not used."""
from __future__ import annotations

import base64
import gzip
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from PIL import Image

from analysis import temporal_occupancy_plots as temporal_plots
from analysis import temporal_occupancy_behavior_figures as behavior_plots

COLORS = ["#4477aa", "#ee6677", "#228833", "#aa3377", "#ccbb44", "#66ccee", "#aa4499", "#888888"]


def color(label):
    return COLORS[int(str(label).split("_J")[-1]) % len(COLORS)]


def timelines(rows, output):
    for side, p in rows.groupby("side"):
        weight = (p.coverage * p.recorded_hours).groupby([p.ant, p.joint_cluster]).sum()
        predominant = {ant: part.droplevel(0).idxmax() for ant, part in weight.groupby(level=0)}
        ants = sorted(p.ant.unique(), key=lambda a: (predominant.get(a, "zzz"), a))
        k = int(p.joint_code.max()) + 1
        bins = np.arange(p.bin.min(), p.bin.max() + 1)
        tick = np.unique(np.r_[0, np.flatnonzero(bins % 6 == 0)])
        dates = pd.to_datetime(bins[tick] * 4 * 3600, unit="s").strftime("%b %d")
        fig, axes = plt.subplots(1, 4, figsize=(23, max(10, .185 * len(ants))), sharey=True)
        specs = [("joint_code", "Joint cluster assignment", ListedColormap(COLORS[:k]), -.5, k - .5),
                 ("colony_percent", "Time inside colony (%)", "YlGnBu", 0, 100),
                 ("mean_speed_mm_s", "Movement speed (mm/s)", "YlOrRd", 0, 2),
                 ("coverage", "Tracking coverage", "Greys", 0, 1)]
        for ax, (metric, title, cmap, lo, hi) in zip(axes, specs):
            wide = p.pivot(index="ant", columns="bin", values=metric).reindex(index=ants, columns=bins)
            im = ax.imshow(wide, aspect="auto", interpolation="none", cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks(tick, dates, rotation=40, ha="right", fontsize=10)
            ax.set_yticks(range(len(ants)), ants, fontsize=6.7)
            ax.set_title(title, fontsize=12, pad=14)
            bar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=.065, fraction=.022)
            if metric == "joint_code":
                bar.set_ticks(range(k), labels=["J" + str(i) for i in range(k)])
                uncertain = p.loc[p.joint_cluster.notna() & ~p.joint_supported.fillna(False).astype(bool)]
                order = {ant: i for i, ant in enumerate(ants)}
                ax.scatter(uncertain.bin - bins[0], uncertain.ant.map(order), s=2, c="black", alpha=.55)
            elif metric == "mean_speed_mm_s":
                bar.set_ticks([0, 1, 2], labels=["0", "1", "≥2"])
            elif metric == "coverage":
                bar.set_ticks([0, .5, 1], labels=["0%", "50%", "100%"])
        fig.suptitle(f"{side.capitalize()}: follow each ant's group, colony presence and movement through time", fontsize=18)
        fig.text(.5, .936, "Includes July 23 block01 + block02, July 24 and the tracked July 29 window. Each row is an ant; each cell is four hours.", ha="center", fontsize=11)
        fig.text(.5, .012, "All tracked ants are retained; rows are sorted by their predominant joint cluster. Calendar gaps remain blank.\n"
                 "Dots mark assignments with limited observation, low separation or weak bootstrap support. Cluster IDs have one meaning across dates within each colony.\n"
                 "Movement colors saturate at 2 mm/s. Low tracking coverage is not inactivity; compare the last two panels.", ha="center", fontsize=10)
        fig.subplots_adjust(top=.90, bottom=.10, wspace=.15)
        temporal_plots.save(fig, output / f"01_{side}_joint_clusters_and_behavior.png")


def cluster_profiles(source, tables, prototypes, output):
    rows = tables["assignments"]
    with np.load(source / "grid_edges.npz") as edges:
        for side, summary in tables["cluster_summary"].groupby("side"):
            summary = summary.sort_values("joint_cluster")
            x, y = edges[side + "_x"], edges[side + "_y"]
            fig, axes = plt.subplots(len(summary), 4, figsize=(15, 4 * len(summary)), squeeze=False)
            maximum = max(np.sqrt(prototypes[label]).max() for label in summary.joint_cluster)
            for i, row in enumerate(summary.itertuples()):
                label = row.joint_cluster
                p = rows.loc[rows.joint_cluster.eq(label)]
                ax = axes[i, 0]
                ax.imshow(np.sqrt(prototypes[label]), origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="magma", vmin=0, vmax=maximum)
                ax.set(title=f"{label}: typical spatial use\n{row.ants} ants, {row.profiles} windows", xlabel="Arena x (mm)", ylabel="Arena y (mm)")
                for ax, metric, title, limits in zip(axes[i, 1:],
                        ["colony_percent", "mean_speed_mm_s", "trip_rate"],
                        ["Time inside colony (%)", "Movement speed (mm/s)", "Trips per observed hour"],
                        [(0, 100), (0, 2.5), None]):
                    values = p[metric].dropna().to_numpy()
                    if len(values):
                        ax.hist(values, bins=25, color=color(label), alpha=.8)
                        ax.axvline(np.median(values), color="black", lw=1.5, ls="--", label=f"Median {np.median(values):.2g}")
                        ax.legend(fontsize=9)
                    ax.set(xlabel=f"{title}\n{len(values)} windows with this measurement", ylabel="Four-hour profiles")
                    if limits is not None:
                        # Keep unusual values visible rather than discard histogram tails.
                        ax.set_xlim(limits[0], max(limits[1], float(values.max()) * 1.03) if len(values) else limits[1])
            fig.suptitle(f"What distinguishes the {side} joint groups?", fontsize=18)
            fig.text(.5, .015, "Maps use observed-hour weighting and a common √ occupancy scale. Histograms show all available measurements; missing trip caches stay missing.\n"
                     "Behavior was not used for clustering. Ants can contribute to multiple groups over time; profiles are not independent animals.", ha="center", fontsize=10)
            fig.tight_layout(rect=(0, .065, 1, .95))
            temporal_plots.save(fig, output / f"03_{side}_what_each_cluster_means.png")


def selection_plot(tables, output):
    for j, (side, part) in enumerate(tables["model_selection"].groupby("side")):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        part = part.sort_values("k")
        for ax, metric, title in [(axes[0], "weighted_silhouette", "Separation in full occupancy features"),
                                  (axes[1], "bootstrap_ari_median", "Stability when whole ants are resampled")]:
            ax.plot(part.k, part[metric], marker="o", color=COLORS[j])
            chosen = part.loc[part.selected]
            ax.scatter(chosen.k, chosen[metric], marker="*", s=230, color=COLORS[j], edgecolors="black", zorder=3)
            ax.set(xlabel="Number of clusters", ylabel="Silhouette" if metric.startswith("weighted") else "Median adjusted Rand", title=title)
        axes[1].plot(part.k, part.bootstrap_ari_p10, color=COLORS[j], ls=":", alpha=.7)
        axes[0].axhline(0, color=".6", lw=.8)
        axes[1].axhline(.8, color=".5", ls="--", lw=.8)
        sensitivity = tables["sensitivity"]
        names = list(sensitivity.comparison.unique())
        selected = sensitivity.loc[sensitivity.side.eq(side)].set_index("comparison").reindex(names)
        axes[2].barh(np.arange(len(names)), selected.adjusted_rand, height=.6, color=COLORS[j])
        axes[2].set(yticks=range(len(names)), yticklabels=names, xlim=(0, 1.02), xlabel="Adjusted Rand vs selected fit", title="Sensitivity to spatial scale and weights")
        fig.suptitle(f"{side.capitalize()} colony — cluster selection and stability", fontsize=15)
        fig.text(.5, .02, "Stars = selected models. Dotted stability lines = bootstrap 10th percentile. Resampling keeps every date for an ant together.\n"
                 "A reproducible split can still describe a continuous population: low silhouette values do not establish discrete task states.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .12, 1, .95))
        temporal_plots.save(fig, output / f"04_{side}_cluster_selection_and_stability.png")


def switch_events(rows):
    events = []
    for ant, part in rows.groupby("ant"):
        part = part.sort_values("bin").reset_index(drop=True)
        for i in range(1, len(part)):
            a, b = part.iloc[i - 1], part.iloc[i]
            if b.bin != a.bin + 1 or (b.first_observed - a.last_observed).total_seconds() > 300:
                continue
            if pd.isna(a.joint_cluster) or pd.isna(b.joint_cluster) or a.joint_cluster == b.joint_cluster:
                continue
            supported = bool(a.joint_supported == True and b.joint_supported == True)
            following = part.iloc[i + 1] if i + 1 < len(part) else None
            persistent = following is not None and following.bin == b.bin + 1 and following.joint_cluster == b.joint_cluster and (following.first_observed - b.last_observed).total_seconds() <= 300
            events.append(dict(ant=ant,side=b.side,start=b.start,from_cluster=a.joint_cluster,to_cluster=b.joint_cluster,
                               before_colony=a.colony_percent,after_colony=b.colony_percent,before_speed=a.mean_speed_mm_s,
                               after_speed=b.mean_speed_mm_s,supported=supported,persists_next_window=persistent))
    return pd.DataFrame(events)


def write_report(tables, events, output):
    selected = tables["model_selection"].loc[tables["model_selection"].selected]
    lines = ["# Joint clustering of ant spatial behavior", "",
             "The former plot 5 used five-neighbor voting against fixed July 23 block02 labels in the square-root grid space. It did not cluster UMAP coordinates. The independent four-hour Leiden fits also used grid features. Those two views answer different questions and are retained in the previous report.", "",
             "This report fits shared clusters across all four-hour maps, separately for each colony, in all 2,689 occupancy dimensions. KMeans uses square-root occupancy probabilities, including the observed fraction outside the arena as one additional bin. Time, ant identity, UMAP coordinates, activity, sleep and trip summaries are not clustering features. Every nonempty map contributes with weight equal to its observed tracking hours. No tracked rows are removed from the output.", "",
             "Later windows are assigned to their nearest joint centroid. These are retrospective clusters defined using all dates, not predictions based only on July 23. The primary labels are J0, J1, etc.; original reference labels remain separate columns. Numeric labels are ordered by median colony occupancy only after fitting for easier reading.", "",
             "## Number of groups and stability", "",
             "| Colony | Chosen K | Silhouette | Median ant-bootstrap ARI | Bootstrap ARI 10th percentile |", "|---|---:|---:|---:|---:|"]
    for r in selected.itertuples():
        lines.append(f"| {r.side} | {r.k} | {r.weighted_silhouette:.3f} | {r.bootstrap_ari_median:.3f} | {r.bootstrap_ari_p10:.3f} |")
    lines += ["", "Selection compares K=2–8 using 24 whole-ant bootstrap replicates. Eligible candidates have median adjusted Rand >=0.8, at least three distinct ants per cluster and at least one observed hour per cluster. The smallest K within 0.02 silhouette of the best eligible score is chosen. Silhouette samples use unweighted neighbor distances, then their mean is weighted by observation exposure. All candidate scores are saved.", "",
              "Two broad groups are reproducible, but their low silhouette scores indicate substantial overlap. Extra subdivisions are less stable. These results support useful descriptive groups, not a claim of sharply separated biological tasks.", "",
              "Membership also depends on spatial scale: after smoothing the maps by 1 mm, adjusted Rand agreement with the primary fit is 0.77 for the left colony and 0.59 for the right; at 2 mm it is 0.70 and 0.55. This sensitivity is a material limitation, especially on the right. Equal profile weighting gives agreements of 0.94 and 0.83. No spatial smoothing is used in the primary fit.", "",
              "Bootstrap assignments are matched using only in-bag ants; per-profile out-of-bag agreement measures how often a profile keeps its group when its ant was absent from the fit. It is a stability statistic, not a posterior probability. Support flags additionally require at least five out-of-bag trials, >=80% agreement, >=10% relative centroid-distance margin, >=40% detection and >=95% recording; these flags never remove displayed data.", "",
              "## What the groups contain", "",
              "| Colony/group | Ants | Four-hour profiles | Median colony % | Median speed mm/s | Median trips/hour |", "|---|---:|---:|---:|---:|---:|"]
    for r in tables["cluster_summary"].itertuples():
        lines.append(f"| {r.joint_cluster} | {r.ants} | {r.profiles} | {r.median_colony_percent:.1f} | {r.median_speed_mm_s:.3f} | {r.median_trip_rate:.2f} |")
    lines += ["", "The behavior summaries above were held out of clustering and describe its result. Low activity and concentrated occupancy can have multiple causes; neither should automatically be called a task transition. Optional trip measurements retain NA where unavailable.", "",
              "## Changes through time", "",
              f"There are {len(events)} assignment changes between adjacent observed four-hour windows across all ants. Of these, {int(events.supported.sum()) if len(events) else 0} have support at both endpoints; {int((events.supported & events.persists_next_window).sum()) if len(events) else 0} also retain the new label in the next available consecutive window. These are events, not independent ants. No change is inferred across a recording gap longer than five minutes.", "",
              "The timeline panels place group membership beside actual colony presence, speed and tracking coverage. A hard label may flip during gradual behavior change, so use the underlying hourly traces and transition close-ups. The original July 23 reference remains available in the parent report.", "",
              "## Method references", "",
              "[KMeans sample weighting and nearest-centroid prediction](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) · [Silhouette definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html). Method choice and selection thresholds are explicit analysis decisions, not guarantees of biological states."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    figures = "".join(f'<h2>{html.escape(p.stem.replace("_", " "))}</h2><a href="{p.name}"><img loading="lazy" src="{p.name}"></a>' for p in sorted(output.glob("*.png")))
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Joint ant occupancy clusters</title><style>body{font:16px system-ui;margin:30px;line-height:1.5;color:#263238}img{width:100%;max-width:1600px}pre{white-space:pre-wrap;max-width:1200px}</style><h1>Ant behavior groups across time</h1><p><a href="interactive.html">Simple interactive ant timelines</a> · <a href="report.md">Methods and findings</a> · <a href="assignments.csv">All assignments</a> · <a href="../index.html">Original reference analysis</a></p><p>These shared clusters were fitted across all dates directly in the full occupancy features. UMAP is not used. Each row in the timeline figures is an ant; each column is four hours.</p>' + figures + '<pre>' + html.escape("\n".join(lines)) + '</pre>')


def explorer(source, output, tables, prototypes):
    from plotly.offline import get_plotlyjs
    rows = tables["assignments"]
    one = pd.read_parquet(source / "profiles_1h.parquet")
    keep = ["profile_key","ant","side","bin","start","stop","center","first_observed","last_observed","joint_cluster","joint_code","joint_supported","coverage","recording_fraction","oob_agreement","centroid_margin","colony_percent","mean_speed_mm_s","trip_rate"]
    with np.load(source / "grid_edges.npz") as edges:
        geometry = {side:[edges[side+"_x"].tolist(),edges[side+"_y"].tolist()] for side in ("left","right")}
    with np.load(source / "maps_4h.npz") as maps:
        raw = {key:np.round(maps[key],8).tolist() for key in maps.files}
    manifest = json.loads((source / "source_run_manifest.json").read_text())
    data = dict(four=temporal_plots.records(rows[keep]),one=temporal_plots.records(one[["ant","bin","start","stop","center","first_observed","last_observed","colony_percent","mean_speed_mm_s","trip_rate","coverage","recording_fraction"]]),
                summary=temporal_plots.records(tables["cluster_summary"]),colors=COLORS,edges=geometry,maps=raw,
                prototypes={key:np.round(value,8).tolist() for key,value in prototypes.items()},
                windows=temporal_plots.records(pd.read_parquet(source / "windows_4h.parquet")),
                light_on=manifest["light_on_hour"],light_off=manifest["light_off_hour"])
    encoded = base64.b64encode(gzip.compress(json.dumps(data,separators=(",",":")).encode())).decode()
    template = Path(__file__).with_name("temporal_occupancy_joint_dashboard.html").read_text()
    (output / "interactive.html").write_text(template.replace("__PLOTLY__",get_plotlyjs()).replace("__DATA__",encoded))


def render(source, output, tables, prototypes):
    timelines(tables["assignments"],output)
    cluster_profiles(source,tables,prototypes,output)
    selection_plot(tables,output)
    # Reuse the straightforward hourly behavior view, with the new group labels.
    four = tables["assignments"].copy()
    four["prediction"] = four.joint_cluster.str.replace("_J","_",regex=False)
    one = pd.read_parquet(source / "profiles_1h.parquet")
    manifest = json.loads((source / "source_run_manifest.json").read_text())
    behavior_plots.transition_closeups(one,four,manifest,output,cluster_kind="Joint")
    events = switch_events(tables["assignments"])
    events.to_csv(output / "switch_events.csv",index=False)
    write_report(tables,events,output)
    explorer(source,output,tables,prototypes)
    for path in output.glob("*.png"):
        with Image.open(path) as im:im.verify()
    print("JOINT_FIGURES_COMPLETE",len(list(output.glob("*.png"))),flush=True)
