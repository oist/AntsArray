"""Grid-occupancy-inspired task investment plots across actual calendar dates."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import long_timescale_tasks as tasks
from analysis import long_timescale_plots as plots

PANELS = (("colony", "food", "water"), ("speed", "body", "antenna"), ("sleep",), ("trip_count",),
          ("trip_rate",), ("trip_duration",), ("trip_investment",))
OVERVIEW = ("colony", "trip_rate", "trip_duration", "trip_investment", "speed", "sleep", "food")


def individual_figures(tables, manifest, output):
    root = Path(output) / "individual_ants"
    root.mkdir(exist_ok=True)
    bins = tables["task_bins"]
    times = plots.time_axis(bins)
    files = {}
    for number, (ant, part) in enumerate(bins.groupby("ant", sort=True), 1):
        wide = part.set_index("timestamp").reindex(times)
        fig, axes = plt.subplots(4, 2, figsize=(14, 12), layout="constrained")
        for ax, metrics in zip(axes.flat, PANELS):
            for i, metric in enumerate(metrics):
                col, _, label, unit = tasks.METRICS[metric]
                ax.plot(times, wide[col], color=plots.COLORS[i], lw=.85, marker=".", markersize=2, label=label)
            ax.set_ylabel(unit)
            ax.legend(fontsize=8, loc="upper left")
            plots.shade_nights(ax, times, manifest)
        ax = axes.flat[-1]
        ax.plot(times, 100 * wide.position_coverage, label="Colony-state coverage", lw=.85)
        ax.plot(times, 100 * wide.trip_coverage, label="Trip position exposure", lw=.85)
        ax.set_ylabel("% of bin observed"); ax.legend(fontsize=8)
        plots.shade_nights(ax, times, manifest)
        available = int(part.loc[part.trip_available, "source_block"].nunique())
        fig.suptitle(f"{ant}: individual behavior across all recordings\n"
                     f"30 min bins; shading = lights off; gaps = unknown; existing trip analyses in {available} blocks/windows")
        name = ant.replace(":", "_") + ".png"
        fig.savefig(root / name, dpi=110, facecolor="white")
        plt.close(fig)
        files[ant] = "individual_ants/" + name
        if number == 1 or number % 25 == 0:
            print(f"Individual ant figures: {number}/{bins.ant.nunique()}", flush=True)
    manifest["individual_figures"] = files


def daily_heatmaps(tables, manifest, output):
    daily = tables["task_daily"].query("phase == 'all'")
    days = pd.date_range(daily.calendar_date.min(), daily.calendar_date.max()).strftime("%Y-%m-%d")
    for side in plots.SIDES:
        ants = sorted(tables["task_bins"].loc[lambda d: d.side.eq(side), "ant"].unique())
        fig, axes = plt.subplots(1, len(OVERVIEW), figsize=(24, 14), layout="constrained")
        for ax, metric in zip(axes, OVERVIEW):
            part = daily.query("side == @side and metric == @metric")
            wide = part.pivot(index="ant", columns="calendar_date", values="value").reindex(index=ants, columns=days)
            label, unit = tasks.METRICS[metric][2:]
            vmax = 100 if unit == "%" else None
            im = ax.imshow(wide, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax)
            ax.set(title=label, xticks=np.arange(len(days)), xticklabels=[d[5:] for d in days],
                   yticks=np.arange(len(ants)), yticklabels=ants if ax is axes[0] else [])
            ax.tick_params(axis="x", rotation=60); ax.tick_params(axis="y", labelsize=7)
            fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.03, pad=.05, label=unit)
        fig.suptitle(f"{side}: daily task profiles of every tracked identity\n"
                     "Same ant order in all panels; all observed data; white = unavailable, including optional trip analyses")
        plots.save(fig, output, f"task_daily_profiles_{side}")


def investment_paths(tables, manifest, output):
    """Extend grid colony-use versus trip-investment plots through all dates."""
    daily = tables["task_daily"].query("phase == 'all'")
    days = sorted(daily.calendar_date.unique())
    for j, side in enumerate(plots.SIDES):
        fig, axes = plt.subplots(2, 1, figsize=(11, 12), squeeze=False, layout="constrained")
        wide = daily[daily.side.eq(side)].pivot(index=["ant", "calendar_date"], columns="metric", values="value").reset_index()
        for i, metric in enumerate(("trip_rate", "trip_duration")):
            ax = axes[i, 0]
            for _, part in wide.groupby("ant"):
                part = part.set_index("calendar_date").reindex(days)
                # The line joins observed endpoints as a trajectory in behavior
                # space; its presence does not imply observations between days.
                ax.plot(part.colony, part[metric], color="#8c97a3", alpha=.2, lw=.6)
            points = ax.scatter(wide.colony, wide[metric], c=pd.to_datetime(wide.calendar_date).map(pd.Timestamp.toordinal),
                                cmap="viridis", s=12, alpha=.7)
            ax.set(title=side, xlabel="Observed time inside colony (%)",
                   ylabel=tasks.METRICS[metric][2] + " (" + tasks.METRICS[metric][3] + ")")
            ticks = [pd.Timestamp(d).toordinal() for d in days]
            cb = fig.colorbar(points, ax=ax, ticks=ticks)
            cb.ax.set_yticklabels([d[5:] for d in days])
        fig.suptitle(f"{side.capitalize()} colony\n" + "Colony occupancy versus trip investment across all observation days\n"
                     "Color = actual calendar date; gray paths = same ant; daily values depend on observed hours")
        plots.save(fig, output, f"task_colony_use_and_trip_investment_{side}")


def matched_changes(tables, manifest, output):
    data = tables["task_standardized"].query("phase == 'all'")
    metrics = [m for m in OVERVIEW if m != "trip_duration"]
    for j, side in enumerate(plots.SIDES):
        fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 22), squeeze=False, layout="constrained")
        for i, metric in enumerate(metrics):
            ax = axes[i, 0]
            part = data.query("metric == @metric and side == @side")
            ants = sorted(tables["task_bins"].loc[lambda d: d.side.eq(side), "ant"].unique())
            wide = part.pivot(index="ant", columns="recording", values="value").reindex(index=ants, columns=manifest["recordings"])
            change = wide.sub(wide.iloc[:, 0], axis=0)
            finite = np.abs(change.to_numpy()[np.isfinite(change.to_numpy())])
            lim = max(float(finite.max()), 1e-9) if finite.size else 1.
            im = ax.imshow(change, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set(title=f"{side}: {tasks.METRICS[metric][2]}", xticks=np.arange(len(wide.columns)),
                   xticklabels=wide.columns, ylabel="Ants in fixed tag order")
            fig.colorbar(im, ax=ax, label="Change (" + tasks.METRICS[metric][3] + ")")
        fig.suptitle(f"{side.capitalize()} colony\n" + "Do individual task profiles change after matching clock time?\n"
                     "Same clock slots in every recording; change from first recording; white = insufficient shared observations")
        plots.save(fig, output, f"task_changes_on_shared_clock_hours_{side}")


def save_figures(tables, manifest, output):
    daily_heatmaps(tables, manifest, output)
    investment_paths(tables, manifest, output)
    matched_changes(tables, manifest, output)
    individual_figures(tables, manifest, output)
