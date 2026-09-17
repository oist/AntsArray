"""Paired worker comparisons across recording windows, matched on clock time.

Reads existing clock tables and one-second motion/position contexts. No raw pose
processing, sleep reclassification, or correspondence between cluster numbers.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import adjusted_rand_score

from analysis import arena_grid_utils as arena
from analysis import grid_occupancy_utils as go
from analysis import return_sleep_utils as rs
from analysis import sleep_motion_analysis_utils as sma


@dataclass(frozen=True)
class ComparisonSettings:
    min_bin_coverage: float = 0.5
    min_matched_hours: float = 6.0
    n_bootstrap: int = 2000
    random_state: int = 0

    def validate(self):
        if not 0 < self.min_bin_coverage <= 1 or not 0 < self.min_matched_hours <= 24:
            raise ValueError("Require coverage in (0, 1] and matched hours in (0, 24]")
        if self.n_bootstrap < 100 or int(self.n_bootstrap) != self.n_bootstrap:
            raise ValueError("Use at least 100 bootstrap replicates")


# key: value column, coverage column, display label, units
METRICS = {
    "speed": ("mean_speed_mm_s", "speed_coverage", "Locomotor speed (point 0)", "mm/s"),
    "body": ("body_motion_mm_s", "body_coverage", "Body motion (points 0-3)", "mm/s"),
    "antenna": ("antenna_motion_mm_s", "antenna_coverage", "Antenna motion (points 4-9)", "mm/s"),
    "sleep": ("sleep_percent", "sleep_coverage", "Sleep", "%"),
    "outside": ("outside_percent", "position_coverage", "Outside colony", "%"),
}
IDENTITY = ["side", "track_id"]


def _require(paths, message):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(message + "\n" + "\n".join(missing))


def _unique(table, columns, name):
    if table.duplicated(columns).any():
        raise ValueError(f"Duplicate {name}: {columns}")


def bin_context(context, template, *, bin_frames, n_context_bins):
    """Aggregate existing second means; missing seconds never count as zeros."""
    result = template.copy()
    raw_lo = result.bin_start_frame.to_numpy(float) / bin_frames
    raw_hi = result.bin_stop_frame.to_numpy(float) / bin_frames
    if not (np.allclose(raw_lo, np.rint(raw_lo)) and np.allclose(raw_hi, np.rint(raw_hi))):
        raise ValueError("Clock bins do not align with the cached motion sampling intervals")
    lo = np.clip(np.rint(raw_lo).astype(np.int64), 0, n_context_bins)
    hi = np.clip(np.rint(raw_hi).astype(np.int64), 0, n_context_bins)
    for metric, source in (("body", "body_speed"), ("antenna", "antenna_speed"), ("outside", "inside")):
        values = np.asarray(context[source])
        if values.shape != (n_context_bins,):
            raise ValueError(f"Unexpected global-frame span in cached {source}")
        if source == "inside":
            if not np.isin(values, [-1, 0, 1]).all():
                raise ValueError("Unrecognized position state")
            valid = values >= 0
            values = (values == 0).astype(float) * 100
        else:
            valid = np.isfinite(values) & (values >= 0)
        counts = np.r_[0, np.cumsum(valid, dtype=np.int64)]
        sums = np.r_[0., np.cumsum(np.where(valid, values, 0.), dtype=np.float64)]
        number = counts[hi] - counts[lo]
        value_column, coverage_column, _, _ = METRICS[metric]
        result[value_column] = np.divide(sums[hi] - sums[lo], number,
                                         out=np.full(len(lo), np.nan), where=number > 0)
        result[coverage_column] = number / (raw_hi - raw_lo)
    return result


def load_window(block, cache_root, *, label, force=False, max_workers=4):
    """Fail on absent/stale upstream inputs, and cache only the small joined bins."""
    block, cache_root = Path(block), Path(cache_root)
    grid = block / "stitched" / "grid_occupancy_histograms_arena"
    clock_path = grid / "activity_sleep_clock" / "ant_cycle_time_bins.parquet"
    cluster_path = grid / "track_cluster_ids.csv"
    settings_path = grid / "sleep_motion_analysis" / "return_response" / "settings.json"
    region_path = go.panorama_regions_path(block)
    _require([clock_path, cluster_path, settings_path, region_path],
             f"Missing comparison inputs for {block}. Run grid_occupancy for this block first:")
    meta_paths = go.metadata_paths(grid)
    if not meta_paths:
        raise FileNotFoundError(f"No annotated-arena grid metadata in {grid}")
    metadata = [json.loads(path.read_text()) for path in meta_paths]
    regions = go.load_panorama_regions(region_path)
    bounds = arena.arena_bounds_from_regions(regions)
    for row in metadata:
        if row.get("bounds_source") != "panorama_arena_regions" or row.get("arena_bounds_px") != bounds[row["side"]]:
            raise ValueError(f"Stale arena grids: rerun grid_occupancy for {block}")
    start = min(row["frame_min"] for row in metadata)
    stop = max(row["frame_max"] for row in metadata) + 1
    inventory = pd.DataFrame([dict(side=row["side"], track_id=int(row["track_id"]),
                                   track_name=row["track_name"],
                                   detection_fraction=row["n_detected_frames"] / (stop - start))
                              for row in metadata])
    _unique(inventory, IDENTITY, "finished ant identities")
    clusters = pd.read_csv(cluster_path).rename(columns={"TrackID": "track_id"})
    _unique(clusters, IDENTITY, "cluster identities")
    inventory = inventory.merge(clusters[IDENTITY + ["track_name", "cluster_id"]],
                                on=IDENTITY + ["track_name"], how="left", validate="one_to_one")
    if inventory.cluster_id.notna().sum() != len(clusters):
        raise ValueError("Cluster handoff disagrees with finished track identities")
    inventory["selected"] = inventory.cluster_id.notna()
    bins = pd.read_parquet(clock_path)
    _unique(bins, IDENTITY + ["cycle_index", "zt_bin"], "clock bins")
    expected = set(map(tuple, clusters[IDENTITY].to_numpy()))
    if set(map(tuple, bins[IDENTITY].to_numpy())) != expected:
        raise ValueError("Clock tables and cluster handoff select different ants; rerun grid_occupancy")
    named = bins[IDENTITY + ["track_name", "cluster_id"]].drop_duplicates()
    check = named.merge(clusters, on=IDENTITY + ["track_name", "cluster_id"], how="inner")
    if len(named) != len(clusters) or len(check) != len(clusters):
        raise ValueError("Clock-table cluster assignments or filenames are stale")
    if (bins.bin_minutes.nunique() != 1 or bins.light_on_hour.nunique() != 1
            or bins.light_off_hour.nunique() != 1 or bins.min_bin_coverage.nunique() != 1):
        raise ValueError("Clock tables mix binning parameters")
    source_settings = json.loads(settings_path.read_text())
    settings = rs.ReturnSettings(**source_settings["settings"])
    settings.validate()
    labels = sma.load_sleep_label_tracks(block / "stitched" / "sleep_motion_labels")
    if not labels.fps.eq(settings.fps).all():
        raise ValueError("Incompatible motion and sleep-label frame rates")
    classifier = json.loads(labels.classifier_parameters.iloc[0])
    if classifier != source_settings["sleep_classifier_parameters"]:
        raise ValueError("Sleep classifier changed since the cached analysis")
    n_bins = int(labels.frame_max.max()) // settings.bin_frames + 1
    selected = labels.merge(clusters[IDENTITY + ["track_name"]], on=IDENTITY + ["track_name"], validate="one_to_one")
    if len(selected) != len(clusters):
        raise FileNotFoundError("Missing sleep labels for selected comparison ants")
    sources = [clock_path, cluster_path, settings_path, region_path, *meta_paths]
    tasks = []
    cache_time = clock_path.stat().st_mtime_ns
    speed_parameters = None
    for row in selected.itertuples():
        path, _ = rs.context_cache_paths(row, block, regions, settings, n_bins,
                                         block / "stitched" / "analysis_cache" / "return_sleep")
        _require([path], f"Missing current motion context for {label}; rerun grid_occupancy:")
        speed_meta = block / "stitched" / "speed_vectors" / "per_track" / Path(row.track_name).stem / "speed_metadata.json"
        label_meta = Path(row.metadata_path)
        label_data = json.loads(label_meta.read_text())
        sleep_path = Path(row.state_path)
        _require([speed_meta, sleep_path], "Missing source vector metadata:")
        speed_data = json.loads(speed_meta.read_text())
        speed_path = speed_meta.parent / Path(speed_data["speed_path"]).name
        _require([speed_path], "Missing locomotor speed vector:")
        if speed_data["track_name"] != row.track_name or speed_data["fps"] != settings.fps:
            raise ValueError("Speed metadata identity or frame rate mismatch")
        parameters = {key: speed_data[key] for key in ("fps", "mm_per_px", "bodypoint_filter",
                      "max_interp_gap_frames", "smooth_sigma_frames", "max_speed_mm_s")}
        if parameters["bodypoint_filter"] != 0:
            raise ValueError("This comparison labels locomotor speed as bodypoint 0")
        if speed_parameters is not None and speed_parameters != parameters:
            raise ValueError("Locomotor speed parameters differ between ants")
        speed_parameters = parameters
        upstream = [label_meta, sleep_path, speed_meta, speed_path]
        if any(p.stat().st_mtime_ns > cache_time for p in upstream):
            raise ValueError(f"Clock bins predate updated vectors for {row.track_name}; rerun grid_occupancy")
        if label_data["classifier_parameters"] != classifier:
            raise ValueError("Sleep-label parameters differ between ants")
        sources.extend([path, *upstream])
        part = bins[bins.side.eq(row.side) & bins.track_id.eq(row.track_id)].copy()
        tasks.append((path, part))
    signature = {"version": 1, "start_frame": start, "stop_frame": stop,
                 "n_context_bins": n_bins, "context_bin_frames": settings.bin_frames,
                 "sources": [rs.fingerprint(path) for path in sources]}
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / f"{label}_binned_activity.parquet"
    marker = output.with_suffix(".json")
    if not force and output.is_file() and marker.is_file() and json.loads(marker.read_text()) == signature:
        print(f"Comparison cache hit: {output}", flush=True)
        bins = pd.read_parquet(output)
    else:
        def read(task):
            path, part = task
            with np.load(path) as context:
                return bin_context(context, part, bin_frames=settings.bin_frames, n_context_bins=n_bins)
        parts = []
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futures = [pool.submit(read, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                parts.append(future.result())
                if index == 1 or index % 20 == 0 or index == len(tasks):
                    print(f"{label}: cached motion/position bins {index}/{len(tasks)} ants", flush=True)
        bins = pd.concat(parts, ignore_index=True).sort_values(IDENTITY + ["cycle_index", "zt_bin"])
        bins["full_recording_bin"] = bins.bin_start_frame.ge(start) & bins.bin_stop_frame.le(stop)
        temporary = output.with_suffix(".parquet.tmp")
        bins.to_parquet(temporary, index=False)
        temporary.replace(output)
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(signature, indent=2) + "\n")
        temporary.replace(marker)
    origin = pd.to_datetime(block.parent.name, format="%Y%m%d") + pd.Timedelta(seconds=go.start_time_from_track_table(labels))
    info = dict(label=label, block=str(block), frame_start=start, frame_stop=stop, fps=settings.fps,
                start_time=str(origin + pd.Timedelta(seconds=start / settings.fps)),
                stop_time=str(origin + pd.Timedelta(seconds=stop / settings.fps)),
                bin_minutes=float(bins.bin_minutes.iloc[0]), light_on_hour=float(bins.light_on_hour.iloc[0]),
                light_off_hour=float(bins.light_off_hour.iloc[0]),
                cached_min_coverage=float(bins.min_bin_coverage.iloc[0]),
                classifier=classifier, speed_parameters=speed_parameters, context_settings=asdict(settings),
                regions=regions.to_json(orient="records"), sources=signature["sources"])
    return dict(bins=bins, inventory=inventory, info=info)


def validate_windows(early, late):
    for name in ("fps", "bin_minutes", "light_on_hour", "light_off_hour", "classifier", "speed_parameters", "regions"):
        if early["info"][name] != late["info"][name]:
            raise ValueError(f"Windows have incompatible {name}; reconcile the upstream analyses")
    for name in ("fps", "position_bin_seconds", "min_position_fraction"):
        if early["info"]["context_settings"][name] != late["info"]["context_settings"][name]:
            raise ValueError(f"Incompatible cached context parameter: {name}")
    if early["info"]["block"] == late["info"]["block"]:
        raise ValueError("Choose two different recording blocks")


def fold_profiles(bins, settings):
    settings.validate()
    if settings.min_bin_coverage < float(bins.min_bin_coverage.max()):
        raise ValueError("Requested coverage is below the cached clock-table threshold; rebuild upstream bins first")
    parts = []
    for metric, (column, coverage, _, _) in METRICS.items():
        part = bins[IDENTITY + ["zt_bin", "zt_hour", "clock_hour", "bin_minutes"]].copy()
        valid = bins.full_recording_bin & bins[coverage].ge(settings.min_bin_coverage) & np.isfinite(bins[column])
        part["value"] = bins[column].where(valid)
        part["coverage"] = bins[coverage].where(valid)
        folded = part.groupby(IDENTITY + ["zt_bin", "zt_hour", "clock_hour", "bin_minutes"], as_index=False).agg(
            value=("value", "mean"), coverage=("coverage", "mean"), n_cycles=("value", "count"))
        parts.append(folded.assign(metric=metric))
    return pd.concat(parts, ignore_index=True)


def pair_profiles(early, late, settings):
    """Equal weight to shared clock slots within each ant, then to ants."""
    settings.validate()
    keys = IDENTITY + ["metric", "zt_bin", "zt_hour", "clock_hour", "bin_minutes"]
    for part in (early, late):
        _unique(part, IDENTITY + ["metric", "zt_bin"], "folded profile")
    paired = early.merge(late, on=keys, how="inner", suffixes=("_early", "_late"), validate="one_to_one")
    if paired.empty:
        raise ValueError("No common ant identities and clock bins")
    paired["matched"] = np.isfinite(paired.value_early) & np.isfinite(paired.value_late)
    group = IDENTITY + ["metric"]
    work = paired.assign(early=paired.value_early.where(paired.matched), late=paired.value_late.where(paired.matched),
                         coverage_early=paired.coverage_early.where(paired.matched),
                         coverage_late=paired.coverage_late.where(paired.matched))
    ants = work.groupby(group, as_index=False).agg(
        early=("early", "mean"), late=("late", "mean"), n_matched_bins=("matched", "sum"),
        bin_minutes=("bin_minutes", "first"), mean_coverage_early=("coverage_early", "mean"),
        mean_coverage_late=("coverage_late", "mean"))
    ants["matched_clock_hours"] = ants.n_matched_bins * ants.bin_minutes / 60
    for window in ("early", "late"):
        ants[f"matched_valid_hours_{window}"] = ants.matched_clock_hours * ants[f"mean_coverage_{window}"]
    ants["included"] = ants.matched_clock_hours.ge(settings.min_matched_hours)
    ants["delta"] = ants.late - ants.early
    for label, part in (("early", early), ("late", late)):
        unmatched = part.groupby(group, as_index=False).value.mean().rename(columns={"value": f"unmatched_clock_mean_{label}"})
        ants = ants.merge(unmatched, on=group, how="left", validate="one_to_one")
    ants["exclusion_reason"] = np.where(ants.included, "", "too few shared clock bins")
    return paired, ants


def identity_audit(early, late, changes):
    audit = early.merge(late, on=IDENTITY, how="outer", suffixes=("_early", "_late"), validate="one_to_one")
    for window in ("early", "late"):
        audit[f"selected_{window}"] = audit[f"selected_{window}"].eq(True)
    audit["selected_both"] = audit.selected_early & audit.selected_late
    hours = changes.pivot(index=IDENTITY, columns="metric", values="matched_clock_hours").add_suffix("_matched_clock_hours").reset_index()
    return audit.merge(hours, on=IDENTITY, how="left", validate="one_to_one").sort_values(IDENTITY)


def _bootstrap_rho(a, b):
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    denominator = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    return np.divide((ra * rb).sum(axis=1), denominator,
                     out=np.full(len(ra), np.nan), where=denominator > 0)


def summarize_changes(changes, settings):
    rng = np.random.default_rng(settings.random_state)
    rows = []
    for side in ("left", "right"):
        for metric in METRICS:
            data = changes[changes.side.eq(side) & changes.metric.eq(metric) & changes.included]
            n = len(data)
            a, b = data.early.to_numpy(float), data.late.to_numpy(float)
            ci, rho_ci = [np.nan, np.nan], [np.nan, np.nan]
            rho = float(spearmanr(a, b).statistic) if n > 2 and np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan
            if n >= 5:
                sample = rng.integers(n, size=(settings.n_bootstrap, n))
                ci = np.quantile((b[sample] - a[sample]).mean(axis=1), [.025, .975])
                correlations = _bootstrap_rho(a[sample], b[sample])
                if np.isfinite(correlations).sum() >= settings.n_bootstrap / 2:
                    rho_ci = np.nanquantile(correlations, [.025, .975])
            rows.append(dict(side=side, metric=metric, n_ants=n,
                             early_mean=a.mean() if n else np.nan, late_mean=b.mean() if n else np.nan,
                             mean_delta=(b-a).mean() if n else np.nan,
                             median_delta=np.median(b-a) if n else np.nan,
                             median_abs_delta=np.median(np.abs(b-a)) if n else np.nan,
                             delta_ci_low=ci[0], delta_ci_high=ci[1], spearman_rho=rho,
                             rho_ci_low=rho_ci[0], rho_ci_high=rho_ci[1],
                             median_matched_hours=data.matched_clock_hours.median(),
                             min_matched_hours=settings.min_matched_hours, min_bin_coverage=settings.min_bin_coverage))
    return pd.DataFrame(rows)


def cluster_overlap(audit):
    common = audit[audit.selected_both]
    counts = common.groupby(["side", "cluster_id_early", "cluster_id_late"]).size().rename("n_ants").reset_index()
    counts["fraction_early_cluster"] = counts.n_ants / counts.groupby(["side", "cluster_id_early"]).n_ants.transform("sum")
    scores = []
    for side, part in common.groupby("side"):
        scores.append(dict(side=side, n_ants=len(part), adjusted_rand_index=adjusted_rand_score(part.cluster_id_early, part.cluster_id_late)))
    return counts, pd.DataFrame(scores)


def _clock_axis(ax, light_on, light_off):
    dark_start = (light_off - light_on) % 24
    ax.axvspan(dark_start, 24, color="0.92", zorder=-10)
    ax.set_xlim(0, 24)
    ticks = np.arange(0, 25, 4)
    ax.set_xticks(ticks, [go.format_clock_time((light_on + t)*3600) for t in ticks])
    ax.set_xlabel("Recording clock time")


def _profile_ylabel(metric):
    label, unit = METRICS[metric][2:]
    label = {"speed": "Locomotor speed", "body": "Body motion", "antenna": "Antenna motion"}.get(metric, label)
    return f"{label}\n({unit})"


def plot_coverage(audit, min_hours):
    fig, axes = plt.subplots(1, 2, figsize=(12, 17), layout="constrained")
    for ax, side in zip(axes, ("left", "right")):
        data = audit[audit.side.eq(side)].sort_values("track_id")
        values = data[["detection_fraction_early", "detection_fraction_late"]].to_numpy(float) * 100
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("0.85")
        im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", vmin=0, vmax=100, cmap=cmap)
        for i, row in enumerate(data.itertuples()):
            for j, selected in enumerate((row.selected_early, row.selected_late)):
                if selected:
                    ax.text(j, i, "+", ha="center", va="center", color="white", fontsize=9)
        ax.set_yticks(np.arange(len(data)), data.track_id.astype(str), fontsize=7)
        ax.set_xticks([0, 1], ["Early", "Late"])
        ax.set_ylabel("Ant ID")
        ax.set_title(f"{side.capitalize()}: {int(data.selected_both.sum())} selected in both")
    fig.colorbar(im, ax=list(axes), label="Detected frames / recording-window frames (%)", shrink=.5)
    fig.suptitle(f"Identity and observation audit; + = selected by grid analysis\nPaired analyses additionally require {min_hours:g} h of shared clock bins per metric")
    return fig


def _pick_ids(fig, artist, data, ax):
    annotation = ax.annotate("", (0, 0), xytext=(7, 9), textcoords="offset points",
                             bbox=dict(facecolor="white", edgecolor="0.5", alpha=.95))
    annotation.set_visible(False)
    def pick(event):
        if event.artist is artist and len(event.ind):
            row = data.iloc[int(event.ind[0])]
            annotation.xy = (row.early, row.late)
            annotation.set_text(f"{row.side} T{int(row.track_id)}\nEarly {row.early:.3g}; late {row.late:.3g}\n{row.matched_clock_hours:g} matched h")
            annotation.set_visible(True)
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect("pick_event", pick)


def plot_paired_activity(changes, summary, *, metrics=("speed", "body", "antenna")):
    fig, axes = plt.subplots(len(metrics), 2, figsize=(13, 4 * len(metrics)), squeeze=False, layout="constrained")
    for i, metric in enumerate(metrics):
        pool = changes[changes.metric.eq(metric) & changes.included]
        limit = max(.01, float(pool[["early", "late"]].max().max()) * 1.08) if len(pool) else 1
        for j, side in enumerate(("left", "right")):
            data = pool[pool.side.eq(side)].copy()
            ax = axes[i, j]
            ax.plot([0, limit], [0, limit], "--", color="0.5", lw=1)
            artist = ax.scatter(data.early, data.late, c="#167f91" if side == "left" else "#c15c43", s=28, alpha=.8, picker=5)
            _pick_ids(fig, artist, data, ax)
            # Separate the six labels vertically so crowded low-speed IDs remain readable.
            extremes = data.loc[data.delta.abs().nlargest(6).index].sort_values("late")
            label_y = np.clip(extremes.late.to_numpy() / limit + .015, .025, .95)
            for k in range(1, len(label_y)):
                label_y[k] = max(label_y[k], label_y[k-1] + .045)
            if len(label_y) and label_y[-1] > .95:
                label_y -= label_y[-1] - .95
            for row, y in zip(extremes.itertuples(), label_y):
                ax.annotate(str(row.track_id), (row.early, row.late),
                            xytext=(min(row.early / limit + .015, .95), y), textcoords="axes fraction",
                            fontsize=8, arrowprops=dict(arrowstyle="-", color="0.5", lw=.5),
                            bbox=dict(facecolor="white", edgecolor="none", alpha=.65, pad=.2))
            stat = summary[summary.side.eq(side) & summary.metric.eq(metric)].iloc[0]
            label, unit = METRICS[metric][2:]
            ax.set(xlim=(0, limit), ylim=(0, limit), xlabel=f"Early: {label} ({unit})", ylabel=f"Late: {label} ({unit})",
                   title=f"{side.capitalize()}: n={len(data)}, rank rho={stat.spearman_rho:.2f}; mean change={stat.mean_delta:+.3g}")
            ax.grid(alpha=.15)
    fig.suptitle("Same workers, same clock bins; equal time-bin weight within each ant\nLabels identify the six largest observed changes per panel, not significance tests")
    return fig


def plot_changes_by_id(changes, audit):
    sides = ("left", "right")
    height = max(8, .17 * max(int((audit.side.eq(s) & audit.selected_both).sum()) for s in sides))
    fig, axes = plt.subplots(1, 10, figsize=(17, height), layout="constrained")
    for k, (metric, (_, _, label, unit)) in enumerate(METRICS.items()):
        valid = changes[changes.metric.eq(metric) & changes.included]
        vmax = max(.01, float(valid.delta.abs().quantile(.98))) if len(valid) else 1
        for j, side in enumerate(sides):
            ids = audit.loc[audit.side.eq(side) & audit.selected_both, "track_id"].sort_values()
            values = valid[valid.side.eq(side)].set_index("track_id").delta.reindex(ids)
            ax = axes[j*5+k]
            cmap = plt.get_cmap("RdBu_r").copy()
            cmap.set_bad("0.85")
            im = ax.imshow(np.ma.masked_invalid(values.to_numpy(float)[:, None]), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            if k == 0:
                ax.set_yticks(np.arange(len(ids)), ids.astype(str), fontsize=7)
            else:
                ax.set_yticks([])
            ax.set_title((f"{side.capitalize()}\n" if k == 0 else "\n") + metric.capitalize(), fontsize=10)
            if k == 0:
                ax.set_ylabel("Ant ID (fixed numeric order)")
            fig.colorbar(im, ax=ax, orientation="horizontal", shrink=.9, aspect=8, extend="both",
                         label="pp" if unit == "%" else unit)
    fig.suptitle("Late minus early, matched clock bins\nGray = insufficient paired coverage; each metric has a shared scale across colonies")
    return fig


def plot_clock_changes(paired, changes, info):
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(13, 13), sharex=True, sharey="row", layout="constrained")
    accepted = changes[changes.included][IDENTITY + ["metric"]]
    data = paired[paired.matched].merge(accepted, on=IDENTITY + ["metric"], validate="many_to_one")
    summaries = []
    n_bins = round(1440 / info["bin_minutes"])
    for i, (metric, (_, _, label, unit)) in enumerate(METRICS.items()):
        for j, side in enumerate(("left", "right")):
            part = data[data.side.eq(side) & data.metric.eq(metric)]
            grouped = part.groupby("zt_bin").agg(early=("value_early", "mean"), late=("value_late", "mean"), n_ants=("track_id", "nunique"))
            grouped = grouped.reindex(np.arange(n_bins))
            grouped["zt_hour"] = (grouped.index.to_numpy()+.5) * info["bin_minutes"] / 60
            ax = axes[i, j]
            for window, color in (("early", "#267eaa"), ("late", "#bf503c")):
                ax.plot(grouped.zt_hour, grouped[window].where(grouped.n_ants >= 5), color=color, label=window.capitalize())
            ax.set_ylabel(_profile_ylabel(metric), fontsize=10)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.15)
            _clock_axis(ax, info["light_on_hour"], info["light_off_hour"])
            if i < len(METRICS) - 1:
                ax.set_xlabel("")
            if i == 0:
                ax.set_title(f"{side.capitalize()} colony")
                ax.legend()
            summaries.append(grouped.reset_index().assign(side=side, metric=metric))
    fig.suptitle("Clock-matched activity profiles; the same ants contribute to early and late at each time\nBins with fewer than five paired ants omitted; composition can differ between clock bins")
    return fig, pd.concat(summaries, ignore_index=True)


def plot_cluster_overlap(counts, scores):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, side in zip(axes, ("left", "right")):
        data = counts[counts.side.eq(side)]
        matrix = data.pivot(index="cluster_id_early", columns="cluster_id_late", values="n_ants").fillna(0)
        fractions = matrix.div(matrix.sum(axis=1), axis=0)
        im = ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                ax.text(j, i, str(int(matrix.iloc[i, j])), ha="center", va="center", color="white" if fractions.iloc[i, j] > .6 else "black")
        stat = scores[scores.side.eq(side)].iloc[0]
        ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
        ax.set_yticks(np.arange(len(matrix)), matrix.index)
        ax.set(xlabel="Late cluster", ylabel="Early cluster", title=f"{side.capitalize()}: n={stat.n_ants}, adjusted Rand={stat.adjusted_rand_index:.2f}")
    fig.colorbar(im, ax=list(axes), label="Fraction of early cluster among common IDs")
    fig.suptitle("Spatial-cluster membership overlap (cell labels = ant counts)\nWhole-window occupancy, not clock-matched; cluster numbers are arbitrary, not task labels")
    return fig


def plot_ant_profiles(early, late, *, side, track_id, info):
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(11, 11), sharex=True, layout="constrained")
    n_bins = round(1440 / info["bin_minutes"])
    any_data = False
    for ax, (metric, (_, _, label, unit)) in zip(axes, METRICS.items()):
        for window, frame, color in (("Early", early, "#267eaa"), ("Late", late, "#bf503c")):
            part = frame[frame.side.eq(side) & frame.track_id.eq(track_id) & frame.metric.eq(metric)]
            series = part.set_index("zt_bin").value.reindex(np.arange(n_bins))
            any_data |= series.notna().any()
            ax.plot((np.arange(n_bins)+.5)*info["bin_minutes"]/60, series, ".-", color=color, label=window)
        ax.set_ylabel(_profile_ylabel(metric), fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.15)
        _clock_axis(ax, info["light_on_hour"], info["light_off_hour"])
        if ax is not axes[-1]:
            ax.set_xlabel("")
    axes[0].legend()
    fig.suptitle(f"{side.capitalize()} T{track_id}: early and late activity\nUnobserved or low-coverage bins remain missing")
    if not any_data:
        plt.close(fig)
        raise ValueError(f"No eligible cached profiles for {side} T{track_id}; inspect identity_audit.csv")
    return fig


def write_report(output, early, late, settings, summary, audit, changes, scores, sensitivity, figure_paths):
    output = Path(output)
    manifest = dict(settings=asdict(settings), early=early["info"], late=late["info"],
                    identity_key=IDENTITY, weighting="equal common clock slots per ant; equal ants within colony",
                    full_clock_bins_only=True, uncertainty="descriptive paired-ant bootstrap; no colony-level replication",
                    figures=list(map(str, figure_paths)))
    (output / "settings.json").write_text(json.dumps(manifest, indent=2) + "\n")
    common = audit[audit.selected_both]
    text = ["# Early Versus Late Worker Activity", "",
            f"Early: {early['info']['start_time']} to {early['info']['stop_time']}.",
            f"Late: {late['info']['start_time']} to {late['info']['stop_time']}.", "",
            f"{len(common)} workers selected in both windows, keyed by colony side and tag ID. "
            f"Each metric requires {settings.min_matched_hours:g} hours of shared clock bins with "
            f">={100*settings.min_bin_coverage:g}% valid samples in each window. Only fully recorded bins enter comparisons.", "",
            "## Paired Results", "",
            "A worker's early and late means use exactly the same clock slots. Differences are late minus early. "
            "Intervals resample paired ants, not frames; they describe these colonies, not independent colony replication.", "",
            "| Colony | Metric | Ants | Early | Late | Mean change [95% ant bootstrap] | Rank rho |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        text.append(f"| {row.side} | {METRICS[row.metric][2]} | {row.n_ants} | {row.early_mean:.3g} | {row.late_mean:.3g} | "
                    f"{row.mean_delta:+.3g} [{row.delta_ci_low:+.3g}, {row.delta_ci_high:+.3g}] | {row.spearman_rho:.2f} |")
    text += ["", "Movement units are mm/s; sleep and outside-colony levels are percentages, with changes in percentage points.", "",
             "## Coverage Sensitivity", "",
             "Cells show number of eligible ants and mean late-minus-early change. Cohorts change with the filters; "
             "a small or empty cohort cannot establish robustness. Full estimates and intervals are in coverage_sensitivity.csv.", "",
             "| Colony | Metric | 50%, 3 h | 50%, 6 h | 50%, 9 h | 75%, 3 h | 75%, 6 h |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for side in ("left", "right"):
        for metric in METRICS:
            cells = []
            for coverage, hours in ((.5, 3), (.5, 6), (.5, 9), (.75, 3), (.75, 6)):
                part = sensitivity[sensitivity.side.eq(side) & sensitivity.metric.eq(metric)
                                   & sensitivity.min_bin_coverage.eq(coverage) & sensitivity.min_matched_hours.eq(hours)]
                cells.append(f"n={int(part.n_ants.iloc[0])}; {part.mean_delta.iloc[0]:+.3g}" if len(part) and part.n_ants.iloc[0] else "no eligible ants")
            text.append(f"| {side} | {METRICS[metric][2]} | " + " | ".join(cells) + " |")
    text += ["",
             "## Largest Observed Changes", "", "These are descriptive extremes, not individual significance tests; regression to the mean can affect their ranking.", ""]
    for metric in ("body", "sleep", "outside"):
        text.append(f"### {METRICS[metric][2]}")
        for side in ("left", "right"):
            part = changes[changes.included & changes.side.eq(side) & changes.metric.eq(metric)]
            largest = part.loc[part.delta.abs().nlargest(5).index]
            text.append(f"- {side}: " + "; ".join(f"T{r.track_id}: {r.early:.3g} to {r.late:.3g} ({r.delta:+.3g})" for r in largest.itertuples()))
        text.append("")
    text += ["## Interpretation Limits", "",
             "- Stable rank order and changing absolute activity can coexist. Rank correlation describes persistence; the paired changes describe shifts in level.",
             "- Unequal time-of-day coverage is controlled at the clock-bin level, but each ant/metric can contribute different clock slots. Consult paired_clock_bins.parquet and the coverage sensitivity table.",
             "- Missing detections are not inactivity, disappearance, or death. The analysis assumes a tag ID continues to identify the same physical ant.",
             "- These two windows do not distinguish aging, acclimation, colony-wide conditions, tracking changes, or causal task reassignment. Ants are socially dependent; ant-bootstrap intervals are descriptive.",
             f"- Locomotor speed uses bodypoint 0, with the existing gap/smoothing settings and values above {early['info']['speed_parameters']['max_speed_mm_s']} mm/s excluded. Body and antenna motion average the cached per-frame group percentiles at one-second resolution, not the arithmetic mean of all points; their coverage counts valid seconds, not raw frames. Sleep uses the unchanged motion classifier.",
             "- Matched clock hours describe supported clock slots, not uninterrupted observation. Effective valid hours (slot duration times mean coverage) are also recorded per ant and metric.",
             "- Spatial cluster overlap uses whole-window histograms with unequal clock coverage. Cluster labels are arbitrary and are not named tasks. Do not count unequal cluster numbers as task switches.",
             "- The reference data define the light schedule, not a new measured light log. No missing clocks are filled by interpolation.", "", "## Spatial Membership", ""]
    for row in scores.itertuples():
        text.append(f"- {row.side}: adjusted Rand index {row.adjusted_rand_index:.3f} across {row.n_ants} common IDs.")
    text += ["", "## Files", "", "- [All ant identities and coverage](identity_audit.csv)",
             "- [Per-ant paired changes](ant_changes.csv)", "- [Paired summary](paired_summary.csv)",
             "- [Coverage sensitivity](coverage_sensitivity.csv)", "- [Parameters and input provenance](settings.json)", "",
             "## Figures", ""]
    text.extend(f"- [{path.stem}]({path.name})" for path in figure_paths)
    (output / "comparison_report.md").write_text("\n".join(text) + "\n")
