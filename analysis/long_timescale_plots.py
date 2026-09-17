"""Saved figures and an offline browser explorer for longitudinal ant analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

from analysis import grid_occupancy_utils as go
from analysis import long_timescale_utils as lt

SIDES = ("left", "right")
COLORS = ("#187ca5", "#d47830", "#7566b0", "#42906b", "#c55e82", "#686868")


def save(fig, output, name):
    fig.savefig(Path(output) / (name + ".png"), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def time_axis(bins):
    recorded = bins[bins.in_recording_bin]
    step = pd.Timedelta(minutes=float(bins.bin_minutes.iloc[0]))
    return pd.date_range(recorded.timestamp.min(), recorded.timestamp.max(), freq=step)


def shade_nights(ax, times, manifest):
    first, last = pd.Timestamp(times.min()).floor("D"), pd.Timestamp(times.max()).ceil("D")
    for day in pd.date_range(first - pd.Timedelta(days=1), last, freq="D"):
        lo = day + pd.Timedelta(hours=manifest["light_off_hour"])
        hi = day + pd.Timedelta(days=1, hours=manifest["light_on_hour"])
        ax.axvspan(lo, hi, color="#b5bed0", alpha=.17, lw=0)
    ax.set_xlim(times.min(), times.max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))


def calendar_summary(tables, manifest, output, metrics, name):
    long, times = tables["long"], time_axis(tables["bins"])
    for j, side in enumerate(SIDES):
        fig = plt.figure(figsize=(12, 2 + 3 * len(metrics)), layout="constrained")
        grid = fig.add_gridspec(len(metrics) + 1, 1, height_ratios=[1.3] + [3] * len(metrics))
        coverage = fig.add_subplot(grid[0, :])
        windows = sorted(manifest["windows"], key=lambda w: w["start"])
        for row, window in enumerate(windows):
            start, stop = pd.Timestamp(window["start"]), pd.Timestamp(window["stop"])
            coverage.broken_barh([(mdates.date2num(start), (stop-start).total_seconds()/86400)],
                                 (row-.3, .6), color=COLORS[manifest["recordings"].index(Path(window["block"]).parent.name)])
        coverage.set(yticks=range(len(windows)), yticklabels=["/".join(Path(w["block"]).parts[-2:]) for w in windows],
                     xlim=(times.min(), times.max()), title="Finished tracking windows: 0723 includes both blocks")
        coverage.invert_yaxis(); coverage.tick_params(labelsize=8)
        ticks = [times.min(), *pd.date_range(times.min().ceil("D"), times.max().floor("D"), freq="D")]
        coverage.set_xticks(ticks, [t.strftime("%b %d\n%H:%M") for t in ticks])
        handoffs = recording_handoffs(manifest)
        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(grid[i+1, 0])
            part = long[long.metric.eq(metric) & long.side.eq(side) & long.in_recording_bin]
            bytime = part.groupby("timestamp").value
            mean, q25, q75 = bytime.mean().reindex(times), bytime.quantile(.25).reindex(times), bytime.quantile(.75).reindex(times)
            ax.fill_between(times, q25, q75, color=COLORS[j], alpha=.17)
            ax.plot(times, mean, color=COLORS[j], lw=1.4)
            shade_nights(ax, times, manifest)
            ax.set_xticks(ticks, [t.strftime("%b %d\n%H:%M") for t in ticks])
            ax.tick_params(axis="x", labelsize=8)
            for handoff in handoffs:
                ax.axvline(handoff["start"], color="#414b56", ls="--", lw=.8, alpha=.6)
            ax.set(title=f"{side} — {lt.METRICS[metric][2]}", ylabel=lt.METRICS[metric][3])
            ax.grid(alpha=.15)
        note = "; ".join(f"{h['start']:%b %d %H:%M} recording handoff: {h['gap_seconds']:.0f} s" for h in handoffs)
        fig.suptitle(f"{side.capitalize()} colony\n" + "Actual calendar time: equal-ant mean and interquartile range\nShading = lights off; dashed line = " + (note or "recording handoff"))
        save(fig, output, f"{name}_{side}")


def recording_handoffs(manifest):
    """Near-continuous transitions between source date folders, without filling gaps."""
    windows = sorted(manifest["windows"], key=lambda w: w["start"])
    out = []
    for previous, following in zip(windows, windows[1:]):
        stop, start = pd.Timestamp(previous["stop"]), pd.Timestamp(following["start"])
        gap = (start-stop).total_seconds()
        if Path(previous["block"]).parent.name != Path(following["block"]).parent.name and 0 <= gap < manifest["bin_minutes"]*60:
            out.append(dict(stop=stop, start=start, gap_seconds=gap,
                            previous=previous["block"], following=following["block"]))
    return out


def calendar_handoff(tables, manifest, output):
    handoffs = recording_handoffs(manifest)
    if not handoffs:
        return
    handoff = handoffs[0]
    center = handoff["start"]
    times = time_axis(tables["bins"])
    times = times[(times >= center-pd.Timedelta(hours=3)) & (times <= center+pd.Timedelta(hours=3))]
    for j, side in enumerate(SIDES):
        fig, axes = plt.subplots(4, 1, figsize=(11, 12), squeeze=False, layout="constrained")
        for i, metric in enumerate(("speed", "body", "antenna", "sleep")):
            ax = axes[i, 0]
            part = tables["long"].query("side == @side and metric == @metric and in_recording_bin")
            grouped = part.groupby("timestamp").value
            mean = grouped.mean().reindex(times)
            ax.fill_between(times, grouped.quantile(.25).reindex(times), grouped.quantile(.75).reindex(times), color=COLORS[j], alpha=.18)
            ax.plot(times, mean, color=COLORS[j], marker="o", markersize=4)
            ax.axvspan(handoff["stop"], handoff["start"], color="black", alpha=.2, label=f"{handoff['gap_seconds']:.1f} s without tracking")
            ax.set(title=f"{side}: {lt.METRICS[metric][2]}", ylabel=lt.METRICS[metric][3], xlim=(times.min(), times.max()))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
            ax.grid(alpha=.15)
        fig.suptitle(f"{side.capitalize()} colony\n" + f"0723 → 0724: consecutive 30-minute bins at the recording handoff\n"
                     f"Previous tracking ends {handoff['stop']:%b %d %H:%M:%S}; next begins {handoff['start']:%H:%M:%S}; gap {handoff['gap_seconds']:.1f} seconds")
        save(fig, output, f"calendar_activity_and_sleep_handoff_{side}")


def coverage_plot(tables, manifest, output):
    bins, times = tables["bins"], time_axis(tables["bins"])
    for j, side in enumerate(SIDES):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), squeeze=False, layout="constrained")
        for metric in ("speed", "sleep", "outside", "food"):
            part = tables["long"].query("side == @side and metric == @metric and in_recording_bin")
            count = part.groupby("timestamp").value.count().reindex(times)
            axes[0, 0].plot(times, count, label=lt.METRICS[metric][2])
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set(title=f"{side}: contributing ants", ylabel="Ant count")
        shade_nights(axes[0, 0], times, manifest)
        for i, metric in enumerate(("speed", "sleep"), 1):
            part = tables["long"].query("side == @side and metric == @metric and in_recording_bin")
            wide = part.pivot(index="ant", columns="timestamp", values="coverage").reindex(columns=times).sort_index()
            image = axes[i, 0].imshow(wide, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
            ticks = np.linspace(0, len(times)-1, 6).astype(int)
            axes[i, 0].set(xticks=ticks, xticklabels=times[ticks].strftime("%b %d\n%H:%M"),
                           ylabel="Ants (stable tag order)", title=f"{side}: {metric} valid fraction")
            fig.colorbar(image, ax=axes[i, 0], label="Fraction observed")
        fig.suptitle(f"{side.capitalize()} colony\n" + "Observation coverage across the full timeline; gaps are not inactivity")
        save(fig, output, f"observation_coverage_{side}")


def clock_plot(tables, manifest, output):
    profiles = tables["profiles"]
    for j, side in enumerate(SIDES):
        fig, axes = plt.subplots(len(lt.METRICS), 1, figsize=(11, 22), squeeze=False, layout="constrained")
        for i, (metric, (_, _, label, unit)) in enumerate(lt.METRICS.items()):
            ax = axes[i, 0]
            for k, recording in enumerate(manifest["recordings"]):
                part = profiles.query("metric == @metric and side == @side and recording == @recording")
                mean = part.groupby("clock_hour").value.mean().reindex(manifest["clock_hours"])
                ax.plot(mean.index, mean, label=recording, color=COLORS[k % len(COLORS)])
            ax.axvspan(0, manifest["light_on_hour"], color="#b5bed0", alpha=.25)
            ax.axvspan(manifest["light_off_hour"], 24, color="#b5bed0", alpha=.25)
            ax.set(title=f"{side}: {label}", ylabel=unit, xlim=(0, 24), xticks=np.arange(0, 25, 4))
            if i == 0:
                ax.legend(fontsize=8)
        fig.suptitle(f"{side.capitalize()} colony\n" + "All-recording clock profiles: cycles averaged within each ant, then ants equally\nLabels identify recordings; observation dates and coverage differ")
        save(fig, output, f"all_recording_clock_profiles_{side}")


def phase_plot(tables, manifest, output):
    table = tables["phase_summary"]
    dates = pd.date_range(tables["bins"].timestamp.min().floor("D"), tables["bins"].timestamp.max().floor("D"))
    for j, side in enumerate(SIDES):
        fig, axes = plt.subplots(len(lt.METRICS), 1, figsize=(11, 22), squeeze=False, layout="constrained")
        for i, (metric, (_, _, label, unit)) in enumerate(lt.METRICS.items()):
            ax = axes[i, 0]
            for phase, color in (("light", "#ce9232"), ("dark", "#535b9f")):
                data = table.query("metric == @metric and side == @side and phase == @phase").copy()
                data.index = pd.to_datetime(data.calendar_date)
                data = data.reindex(dates)
                ax.plot(dates, data["mean"], "o-", color=color, label=phase)
                ax.fill_between(dates, data.ci_low, data.ci_high, alpha=.15, color=color)
            ax.set(title=f"{side}: {label}", ylabel=unit)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            if i == 0:
                ax.legend()
        fig.suptitle(f"{side.capitalize()} colony\n" + f"Light/dark behavior by actual calendar day; ≥{manifest['settings']['min_phase_hours']:g} h eligible bins per ant/phase\nObserved-hour summaries; bands are ant bootstrap intervals within each colony")
        save(fig, output, f"day_night_over_calendar_days_{side}")


def standardized_plot(tables, manifest, output, *, shared=True):
    table = tables["standardized" if shared else "observed_recording_means"]
    recordings = manifest["recordings"]
    dates = pd.to_datetime([manifest["recording_centers"][r] for r in recordings])
    for j, side in enumerate(SIDES):
        fig, axes = plt.subplots(len(lt.METRICS), 1, figsize=(11, 22), squeeze=False, layout="constrained")
        for i, (metric, (_, _, label, unit)) in enumerate(lt.METRICS.items()):
            ax = axes[i, 0]
            data = table.query("metric == @metric and side == @side and phase == 'all' and included")
            wide = data.pivot(index="ant", columns="recording", values="value").reindex(columns=recordings)
            if len(wide):
                ax.plot(dates, wide.to_numpy().T, color=".5", alpha=.25, lw=.6)
            summary = tables["standardized_summary" if shared else "observed_recording_summary"].query("metric == @metric and side == @side and phase == 'all'").set_index("recording").reindex(recordings)
            ax.plot(dates, summary["mean"], "o-", color=COLORS[j], lw=2)
            ax.fill_between(dates, summary.ci_low, summary.ci_high, color=COLORS[j], alpha=.2)
            ax.set(title=f"{side}: {label}, n={len(wide)}", ylabel=unit, xticks=dates,
                   xticklabels=[f"{r[4:6]}/{r[6:]}\n{d:%b %d}" for r, d in zip(recordings, dates)])
            ax.grid(alpha=.15)
        title = ("Optional comparison: identical clock slots in EVERY recording" if shared else
                 "All tracked ants: means over every observed recording bin; no observation-hour cutoff")
        fig.suptitle(f"{side.capitalize()} colony\n" + title + "\nGray = individual ants; color = mean and ant bootstrap interval; x = actual recording center (days)")
        save(fig, output, ("all_recording_standardized_trajectories" if shared else "all_recording_observed_trajectories") + f"_{side}")


def individual_heatmaps(tables, manifest, output):
    times = time_axis(tables["bins"])
    for metric in ("speed", "sleep", "outside", "food", "water"):
        all_values = tables["long"].query("metric == @metric").value.dropna()
        vmax = 100 if lt.METRICS[metric][3] == "%" else all_values.quantile(.99)
        for side in SIDES:
            fig, ax = plt.subplots(figsize=(18, 7), layout="constrained")
            part = tables["long"].query("metric == @metric and side == @side and in_recording_bin")
            wide = part.pivot(index="ant", columns="timestamp", values="value").reindex(columns=times).sort_index()
            im = ax.imshow(wide, aspect="auto", interpolation="nearest", vmin=0, vmax=max(1e-6, vmax), cmap="magma")
            ticks = np.linspace(0, len(times)-1, 8).astype(int)
            yticks = np.arange(0, len(wide), 4)
            ax.set(xticks=ticks, xticklabels=times[ticks].strftime("%b %d\n%H:%M"),
                   yticks=yticks, yticklabels=wide.index[yticks], title=side)
            fig.colorbar(im, ax=ax, label=f"{lt.METRICS[metric][2]} ({lt.METRICS[metric][3]})")
            fig.suptitle(f"{side.capitalize()} colony — individual behavior through calendar time\nStable ant ordering; recording gaps stay blank")
            save(fig, output, f"individual_timeline_{metric}_{side}")


def drift_plot(tables, manifest, output):
    recordings = manifest["recordings"]
    for j, side in enumerate(SIDES):
        fig, axes = plt.subplots(len(lt.METRICS), 1, figsize=(10, 24), squeeze=False, layout="constrained")
        for i, (metric, (_, _, label, unit)) in enumerate(lt.METRICS.items()):
            data = tables["standardized"].query("metric == @metric and side == @side and phase == 'all' and included")
            wide = data.pivot(index="ant", columns="recording", values="value").reindex(columns=recordings).sort_index()
            change = wide.subtract(wide.iloc[:, 0], axis=0)
            ax = axes[i, 0]
            if len(change):
                limit = max(float(np.nanpercentile(np.abs(change), 95)), 1e-6)
                im = ax.imshow(change, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit, interpolation="nearest")
                fig.colorbar(im, ax=ax, label=unit)
            else:
                ax.text(.5, .5, "Insufficient shared coverage", ha="center", transform=ax.transAxes)
            ax.set(title=f"{side}: {label}", xticks=range(len(recordings)), xticklabels=recordings,
                   ylabel="Ants in stable tag order")
        fig.suptitle(f"{side.capitalize()} colony\n" + "Individual deviations from the first recording, using the same clock slots across ALL recordings")
        save(fig, output, f"individual_behavior_drift_{side}")


def bout_plot(tables, manifest, output):
    bouts = tables["sleep_bouts"]
    # First average per ant so prolific/fragmented sleepers cannot dominate.
    complete = bouts[bouts.complete]
    medians = complete.groupby(["side", "ant", "recording"]).duration_seconds.median().reset_index()
    for j, side in enumerate(SIDES):
        fig, axes = plt.subplots(2, 1, figsize=(11, 10), squeeze=False, layout="constrained")
        dates = pd.to_datetime([manifest["recording_centers"][r] for r in manifest["recordings"]])
        wide = medians[medians.side.eq(side)].pivot(index="ant", columns="recording", values="duration_seconds").reindex(columns=manifest["recordings"])
        axes[0, 0].plot(dates, wide.to_numpy().T, color=".6", alpha=.3, lw=.7)
        axes[0, 0].plot(dates, wide.median(), "o-", color=COLORS[j])
        axes[0, 0].set(title=f"{side}: complete sleep bouts", ylabel="Per-ant median bout length (s)")
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        counts = bouts[bouts.side.eq(side)].groupby("recording").complete.agg(["count", "sum"]).reindex(manifest["recordings"])
        axes[1, 0].bar(manifest["recordings"], counts["sum"], label="Complete", color=COLORS[j])
        axes[1, 0].bar(manifest["recordings"], counts["count"]-counts["sum"], bottom=counts["sum"], label="Censored by gap / boundary", color=".75")
        axes[1, 0].set(ylabel="Number of one-second sleep runs", title=f"{side}: bout completeness")
        axes[1, 0].legend()
        fig.suptitle(f"{side.capitalize()} colony\n" + "Sleep fragmentation diagnostic from existing one-second sleep states\nComplete-bout durations exclude runs touching unknown tracking or recording boundaries; descriptive, coverage-sensitive")
        save(fig, output, f"sleep_bout_diagnostics_{side}")


def spatial_maps(windows, manifest, output):
    """Pool existing maps within ant, then weight ants equally, on identical grids."""
    maps, geometry = {}, {}
    fingerprints = []
    for window in windows:
        recording = window["block"].parent.name
        grid = window["block"] / "stitched/grid_occupancy_histograms_arena"
        tracks = go.load_grid_tracks(grid)
        selected = window["inventory"][lt.IDENTITY]
        tracks = tracks.merge(selected, on=lt.IDENTITY, validate="one_to_one")
        for _, row in tracks.iterrows():
            hist, x, y = go.load_histogram(row)
            key = (recording, row.side, int(row.track_id))
            gkey = row.side
            if gkey in geometry:
                old = geometry[gkey]
                if not np.array_equal(x, old[0]) or not np.array_equal(y, old[1]):
                    raise ValueError("Occupancy grids differ; cannot combine spatial maps")
            geometry[gkey] = (x, y, row, go.load_panorama_regions(go.panorama_regions_path(window["block"])))
            # Histograms are detected-frame fractions. Multiplying by detected
            # count pools distinct source blocks for an ant before normalization.
            meta = json.loads(Path(row.metadata_path).read_text())
            weight = meta["n_detected_frames"]
            maps[key] = maps.get(key, np.zeros_like(hist, dtype=float)) + hist * weight
            for name in ("occupancy_path", "x_edges_path", "y_edges_path"):
                fingerprints.append(lt.rs.fingerprint(Path(row[name])))
    result = []
    for recording in manifest["recordings"]:
        for side in SIDES:
            parts = [hist / hist.sum() for (rec, s, ant), hist in maps.items() if rec == recording and s == side and hist.sum() > 0]
            if not parts:
                continue
            x, y, row, regions = geometry[side]
            density = np.mean(parts, axis=0)
            annotations = []
            for _, region in regions[(regions.side == side) & (regions.region_type != "arena")].iterrows():
                geo = go._region_geometry_in_grid_mm(region, row)
                annotations.append(dict(geo, shape=region["shape"], name=region["name"], kind=region.region_type))
            result.append(dict(recording=recording, side=side, n_ants=len(parts), z=density.tolist(),
                               x=((x[:-1]+x[1:])/2).tolist(), y=((y[:-1]+y[1:])/2).tolist(), annotations=annotations))
    plot_spatial_maps(result, manifest, output)
    return result, fingerprints


def plot_spatial_maps(result, manifest, output):
    for i, side in enumerate(SIDES):
        fig, axes = plt.subplots(1, len(manifest["recordings"]), figsize=(5*len(manifest["recordings"]), 6), squeeze=False, layout="constrained")
        items = [v for v in result if v["side"] == side]
        maximum = max((np.sqrt(v["z"]).max() for v in items), default=1)
        for j, recording in enumerate(manifest["recordings"]):
            ax = axes[0, j]
            item = next((v for v in items if v["recording"] == recording), None)
            if item is None:
                continue
            im = ax.pcolormesh(item["x"], item["y"], np.sqrt(item["z"]), shading="nearest", cmap="magma", vmin=0, vmax=maximum)
            for annotation in item["annotations"]:
                if annotation["shape"] == "rectangle":
                    patch = Rectangle((annotation["x_min"], annotation["y_min"]), annotation["x_max"]-annotation["x_min"], annotation["y_max"]-annotation["y_min"], fill=False, edgecolor="cyan", lw=.8)
                else:
                    patch = Circle((annotation["center_x"], annotation["center_y"]), annotation["radius"], fill=False, edgecolor="cyan", lw=.8)
                ax.add_patch(patch)
            ax.set(title=f"{side}, recording {recording}, n={item['n_ants']}", xlabel="Arena x (mm)", ylabel="Arena y (mm)", aspect="equal")
            ax.invert_yaxis()
            fig.colorbar(im, ax=ax, label="√ probability per grid cell")
        fig.suptitle(f"{side.capitalize()} colony\n" + "All-recording spatial occupancy from existing ant maps; equal ant weighting\nDescriptive whole-window maps, without clock matching; cyan = panorama annotations")
        save(fig, output, f"all_recording_spatial_maps_{side}")


def save_figures(tables, manifest, windows, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    coverage_plot(tables, manifest, output)
    calendar_summary(tables, manifest, output, ("speed", "body", "antenna", "sleep"), "calendar_activity_and_sleep")
    calendar_handoff(tables, manifest, output)
    calendar_summary(tables, manifest, output, ("colony", "outside", "food", "water"), "calendar_spatial_presence")
    clock_plot(tables, manifest, output)
    phase_plot(tables, manifest, output)
    standardized_plot(tables, manifest, output, shared=False)
    standardized_plot(tables, manifest, output)
    individual_heatmaps(tables, manifest, output)
    drift_plot(tables, manifest, output)
    bout_plot(tables, manifest, output)
    spatial, fingerprints = spatial_maps(windows, manifest, output)
    return spatial, fingerprints


def write_dashboard(tables, manifest, spatial, output):
    """Ship Plotly locally and embed tables: the dashboard works from file://."""
    from plotly.offline import get_plotlyjs
    from analysis import long_timescale_tasks as tasks

    output = Path(output)
    (output / "plotly.min.js").write_text(get_plotlyjs())
    # Column-oriented packing removes repeated keys from every ant/time row.
    columns = ["side", "track_id", "ant", "timestamp", "calendar_date", "recording", "phase", "clock_hour", "zt_bin"]
    columns += list(dict.fromkeys(c for spec in lt.METRICS.values() for c in spec[:2]))
    bins = tables["bins"].loc[tables["bins"].in_recording_bin, columns].copy().sort_values("timestamp")
    bins["timestamp"] = bins.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S")
    packed = json.loads(bins.to_json(orient="split", index=False, double_precision=5))
    clean_manifest = {k: v for k, v in manifest.items() if k not in ("sources", "task_analysis")}
    task_columns = columns + ["source_block", "trip_available", "trip_observed_hours", "trip_coverage"]
    task_columns += [spec[0] for spec in tasks.TRIP_METRICS.values()]
    task_bins = tables["task_bins"][task_columns].copy()
    task_bins["timestamp"] = task_bins.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S")
    payload = dict(bins=packed, meta=clean_manifest,
                   trip_diagnostics="trip_duration_diagnostics/index.html" if (output / "trip_duration_diagnostics/index.html").is_file() else None,
                   task_bins=json.loads(task_bins.to_json(orient="split", index=False, double_precision=5)),
                   task_daily=json.loads(tables["task_daily"].to_json(orient="records", double_precision=5)),
                   task_metrics=tasks.METRICS,
                   standardized=json.loads(tables["standardized"].to_json(orient="records", double_precision=5)),
                   observed=json.loads(tables["observed_recording_means"].to_json(orient="records", double_precision=5)),
                   phase_summary=json.loads(tables["phase_summary"].to_json(orient="records", double_precision=5)),
                   spatial=spatial,
                   figures=[p.name for p in sorted(output.glob("*.png"))])
    text = (Path(__file__).with_name("long_timescale_dashboard.html")).read_text()
    embedded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (output / "index.html").write_text(text.replace("__LONG_TIMESCALE_DATA__", embedded))
