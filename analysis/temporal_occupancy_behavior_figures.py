"""Intuitive behavior companions to the fixed-reference four-hour assignments.

Reads saved temporal tables only. No tracking, aggregation or clustering is rerun.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from analysis import temporal_occupancy_plots as plots


def overview(four, reference, windows, side, output):
    """Same reference-ant order and calendar spacing as figure 05."""
    ref = reference.loc[reference.side.eq(side)].sort_values(["original_cluster", "ant"])
    part = four.loc[four.ant.isin(ref.ant)].copy()
    part["cluster_code"] = part.prediction.map(
        lambda value: float(str(value).rsplit("_", 1)[-1]) if pd.notna(value) else np.nan
    )
    bins = np.arange(windows.bin.min(), windows.bin.max() + 1)
    tick = np.unique(np.r_[0, np.flatnonzero(bins % 6 == 0)])
    labels = pd.to_datetime(bins[tick] * 4 * 3600, unit="s").strftime("%b %d")
    fig = plt.figure(figsize=(20, 13))
    grid = fig.add_gridspec(2, 3, height_ratios=[30, 1], hspace=.18, wspace=.22)
    specifications = [
        ("cluster_code", "Which original group?", ListedColormap(plots.COLORS[:2]), -.5, 1.5,
         [0, 1], ["Cluster 0", "Cluster 1"]),
        ("colony_percent", "How much time inside the colony?", "YlGnBu", 0, 100,
         [0, 50, 100], ["0%", "50%", "100%"]),
        ("mean_speed_mm_s", "How much does the ant move?", "YlOrRd", 0, 2,
         [0, 1, 2], ["0", "1", "≥2 mm/s"]),
    ]
    group_sizes = ref.groupby("original_cluster", sort=True).size()
    boundaries = group_sizes.cumsum().iloc[:-1].to_numpy() - .5
    low = part.loc[part.coverage.lt(.4) | part.recording_fraction.lt(.95)]
    row_index = {ant: i for i, ant in enumerate(ref.ant)}
    for column, (metric, title, cmap, lower, upper, ticks, ticklabels) in enumerate(specifications):
        ax = fig.add_subplot(grid[0, column])
        wide = part.pivot(index="ant", columns="bin", values=metric).reindex(index=ref.ant, columns=bins)
        im = ax.imshow(wide, aspect="auto", interpolation="none", cmap=cmap, vmin=lower, vmax=upper)
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xticks(tick, labels, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(np.arange(len(ref)), ref.ant if column == 0 else [""] * len(ref), fontsize=8)
        if column == 0:
            ax.set_ylabel("One row = the same ant throughout", fontsize=11)
        for boundary in boundaries:
            ax.axhline(boundary, color="black", lw=.7)
        finite = low.loc[low[metric].notna()]
        ax.scatter(finite.bin - bins[0], finite.ant.map(row_index), s=3, color="black", alpha=.6)
        bar = fig.colorbar(im, ax=ax, location="top", shrink=.72, fraction=.025, pad=.035)
        bar.set_ticks(ticks, labels=ticklabels)
        recorded = windows.set_index("bin").recording_fraction.reindex(bins).to_numpy()
        clock = fig.add_subplot(grid[1, column])
        clock.imshow(recorded[None, :], aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="none")
        clock.set(yticks=[], xticks=tick, xticklabels=labels)
        clock.tick_params(axis="x", labelrotation=30, labelsize=8)
        clock.set_xlabel("Recording availability: white = gap; black = full 4 h", fontsize=8)
    fig.suptitle(f"{side.capitalize()} colony — follow each ant from left to right", fontsize=19, y=.985)
    fig.text(.5, .948, "Each cell is four hours. Compare the same row across all three panels.", ha="center", fontsize=12)
    fig.text(.5, .014,
             "Same reference ants and row order as plot 5. Black dots flag <40% detection or <95% recording; those observations remain shown.\n"
             "White means unavailable. Cluster numbers are fixed to 0723/block02, not biological task names. Movement colors saturate at 2 mm/s; underlying values are unchanged.",
             ha="center", fontsize=10)
    fig.subplots_adjust(top=.88, bottom=.09)
    path = output / f"01_{side}_assignments_colony_presence_movement.png"
    plots.save(fig, path)
    return path


def transition_closeups(one, four, manifest, output, cluster_kind="Reference"):
    """Hourly behavior, with the actual support of partial recording hours."""
    cases = [
        ("left:036", "2026-07-24 12:00", "2026-07-25 12:00", "July 24: sustained change over several hours"),
        ("right:052", "2026-07-30 18:00", "2026-07-31 18:00", "July 30: transition at the start of the observed window"),
    ]
    paths = []
    for ant, first, last, title in cases:
        fig = plt.figure(figsize=(12, 10))
        grid = fig.add_gridspec(4, 1, height_ratios=[.55, 2.4, 2.1, .65], hspace=.28, wspace=.24)
        axes = []
        start, stop = pd.Timestamp(first), pd.Timestamp(last)
        hourly = one.loc[one.ant.eq(ant) & one.stop.gt(start) & one.start.lt(stop)].sort_values("bin")
        four_hour = four.loc[four.ant.eq(ant) & four.stop.gt(start) & four.start.lt(stop)]
        group = [fig.add_subplot(grid[row, 0]) for row in range(4)]
        axes.append(group)
        for ax in group:
            plots.clock_shading(ax, manifest, start, stop)
            ax.set_xlim(start, stop)
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
            ax.tick_params(axis="x", labelsize=9)
        group[0].set_title(f"{ant} · {title}", fontsize=12, pad=35)
        for row in four_hour.itertuples():
            if pd.isna(row.prediction):
                continue
            lo, hi = max(start, row.first_observed), min(stop, row.last_observed)
            group[0].axvspan(lo, hi, color=plots.color(row.prediction), alpha=.95)
            center = lo + (hi - lo) / 2
            group[0].text(center, .5, row.prediction.rsplit("_", 1)[-1], ha="center", va="center", color="white", fontweight="bold")
        group[0].set(ylim=(0, 1), yticks=[], ylabel="4 h\ncluster")
        for row in hourly.itertuples():
            lo, hi = max(start, row.first_observed), min(stop, row.last_observed)
            mid = lo + (hi - lo) / 2
            width = (hi - lo).total_seconds() / 86400
            if pd.notna(row.colony_percent):
                group[1].bar(mid, row.colony_percent, width=width, color="#168a80", alpha=.75, linewidth=0)
            if pd.notna(row.mean_speed_mm_s):
                group[2].bar(mid, row.mean_speed_mm_s, width=width, color="#dc8739", alpha=.8, linewidth=0)
            group[3].bar(mid, 100 * row.coverage, width=width, color="#68788a", linewidth=0)
        group[1].set(ylim=(0, 105), yticks=[0, 50, 100], ylabel="Time inside colony (%)")
        group[1].axhline(50, color=".5", lw=.6, ls=":")
        group[2].set(ylim=(0, max(2., float(hourly.mean_speed_mm_s.max()) * 1.08)), ylabel="Movement speed (mm/s)")
        group[3].set(ylim=(0, 100), yticks=[0, 100], ylabel="Tracked\n(%)")
        for ax in group[:-1]:
            ax.tick_params(labelbottom=False)
        if ant == "left:036":
            group[1].annotate("Almost always inside", xy=(pd.Timestamp("2026-07-24 16:30"), 100),
                              xytext=(pd.Timestamp("2026-07-24 13:00"), 75), fontsize=10,
                              arrowprops=dict(arrowstyle="->", color=".3"))
            group[1].annotate("Mostly outside", xy=(pd.Timestamp("2026-07-24 22:30"), 1),
                              xytext=(pd.Timestamp("2026-07-25 00:30"), 29), fontsize=10,
                              arrowprops=dict(arrowstyle="->", color=".3"))
        else:
            group[1].annotate("First bar = only 11 min\nof recording", xy=(pd.Timestamp("2026-07-30 21:54"), 100),
                              xytext=(pd.Timestamp("2026-07-31 00:00"), 83), fontsize=10,
                              arrowprops=dict(arrowstyle="->", color=".3"))
        fig.suptitle(f"{ant.split(':')[0].capitalize()} colony — what does a change in cluster assignment look like in actual behavior?", fontsize=18, y=.98)
        fig.legend(handles=[Patch(color=plots.COLORS[0], label=f"{cluster_kind} cluster 0"),
                            Patch(color=plots.COLORS[1], label=f"{cluster_kind} cluster 1"),
                            Patch(color=".92", label="Lights off")], loc="upper center", bbox_to_anchor=(.5, .946), ncol=3, frameon=False)
        fig.text(.5, .02, "Behavior bars summarize one hour; cluster bands summarize four hours. Bar width follows actual recorded time, including the short first hour.\n"
                 "Colony presence is the fraction of observed locations inside the colony. Tracking coverage is relative to recorded time. No smoothing or data exclusion.",
                 ha="center", fontsize=10)
        fig.subplots_adjust(top=.83, bottom=.115)
        path = output / f"02_{ant.split(':')[0]}_transition_closeups.png"
        plots.save(fig, path)
        paths.append(path)
    return paths


def run(source, output):
    output.mkdir(parents=True, exist_ok=True)
    files = ["profiles_4h.parquet", "profiles_1h.parquet", "reference.parquet", "windows_4h.parquet", "source_run_manifest.json"]
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files}
    four = pd.read_parquet(source / files[0])
    one = pd.read_parquet(source / files[1])
    reference = pd.read_parquet(source / files[2])
    windows = pd.read_parquet(source / files[3])
    manifest = json.loads((source / files[4]).read_text())
    figures = [overview(four, reference, windows, side, output) for side in ("left", "right")]
    figures.extend(transition_closeups(one, four, manifest, output))
    from PIL import Image
    for path in figures:
        with Image.open(path) as image:
            image.verify()
    assert hashes == {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files}
    provenance = dict(created=datetime.now().isoformat(),source=str(source),input_sha256=hashes,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      plot_helpers_sha256=hashlib.sha256(Path(plots.__file__).read_bytes()).hexdigest(),
                      reference_ants=len(reference),profiles_4h=len(four),profiles_1h=len(one),
                      figures=[path.name for path in figures],validation="All source tables unchanged; PNGs verified")
    (output / "run_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    text = """<h1>Follow changes in each ant's behavior</h1>
<p><a href="../index.html">Full temporal analysis</a> · <a href="../interactive.html">Interactive ant explorer</a></p>
<p><strong>Reading plot 5:</strong> each row is one ant and each column is a four-hour window.
The left panel assigns that window to an original 0723/block02 cluster using KNN.
The middle panel measures distance from the reference; the right panel shows tracking coverage.
These are fixed reference assignments. The independently fitted four-hour clusters have separate local IDs.</p>
<p>The first two figures keep the same ant order as plot 5 and show cluster assignment beside actual colony presence and movement speed.
Blue versus coral means reference cluster 0 versus 1, not predefined biological tasks.
The close-ups below show whether those label changes accompany changes in observed behavior.</p>"""
    sections = "".join(f'<h2>{html.escape(path.stem.replace("_", " "))}</h2><a href="{path.name}"><img src="{path.name}" loading="lazy"></a>' for path in figures)
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Ant behavior through time</title><style>body{font:16px system-ui;margin:32px;line-height:1.5;color:#263238}p{max-width:1100px}img{width:100%;max-width:1600px}</style>' + text + sections)
    print("BEHAVIOR_FIGURES_COMPLETE", json.dumps(provenance), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.temporal_folder, args.output or args.temporal_folder / "behavior_timelines")


if __name__ == "__main__":
    main()
