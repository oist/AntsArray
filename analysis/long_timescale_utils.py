"""Missing-aware longitudinal summaries of existing grid_occupancy caches.

Identities are (colony side, tag ID). Recording labels and actual calendar time
are kept separate. No tracking, pose inference, or sleep classification occurs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import block_activity_utils as ba
from analysis import grid_occupancy_utils as go
from analysis import return_sleep_utils as rs
from analysis import sleep_motion_analysis_utils as sma
from analysis.trip_phenotyping_utils import _run_length_encoding

IDENTITY = ba.IDENTITY
METRICS = {
    **ba.METRICS,
    "colony": ("colony_percent", "position_coverage", "Inside colony", "%"),
    "food": ("food_percent", "resource_coverage", "Food region presence", "%"),
    "water": ("water_percent", "resource_coverage", "Water region presence", "%"),
}


@dataclass(frozen=True)
class Settings:
    min_bin_coverage: float = 0.
    min_phase_hours: float = 0.
    min_shared_hours: float = 6.
    bootstrap: int = 1000
    seed: int = 0

    def validate(self):
        if not 0 <= self.min_bin_coverage <= 1:
            raise ValueError("Coverage must be in [0, 1]")
        if not 0 <= self.min_phase_hours <= 10 or not 0 < self.min_shared_hours <= 24:
            raise ValueError("Invalid required observation hours")
        if self.bootstrap < 100:
            raise ValueError("Use at least 100 ant bootstrap replicates")


def discover_blocks(folder, dates):
    """Discover original blocks/window views, never also add stitched duplicates."""
    blocks, audit = [], []
    for date in dates:
        root = Path(folder) / str(date)
        if not root.is_dir():
            raise FileNotFoundError(root)
        for block in sorted(root.glob("block*")):
            tracked = any((block / "stitched/per_track").glob("*.parquet"))
            audit.append(dict(block=str(block), included=tracked,
                              reason="finished tracks" if tracked else "no finished tracks"))
            if tracked:
                blocks.append(block)
    if not blocks:
        raise FileNotFoundError("No blocks with finished tracking found")
    return blocks, audit


def path_resolver(mappings=()):
    pairs = sorted((item.split("=", 1) for item in mappings), key=lambda p: -len(p[0]))
    def resolve(value):
        value = str(value)
        for old, new in pairs:
            if value == old or value.startswith(old.rstrip("/") + "/"):
                return Path(new + value[len(old):])
        return Path(value)
    return resolve


def load_windows(blocks, output, *, bundle=None, mappings=(), workers=4):
    """Reuse a fingerprint-validated combined bundle, or existing window caches.

    A bundle is useful for published caches whose original signatures refer to
    their flash staging paths. Path remapping permits the same shared files to
    be read through another machine's mount; size and nanosecond mtime must match.
    """
    resolve = path_resolver(mappings)
    windows = []
    if bundle:
        bundle = Path(bundle)
        manifest = json.loads((bundle / "comparison_manifest.json").read_text())
        bins = pd.read_parquet(bundle / "combined_activity_bins.parquet")
        inventory = pd.read_csv(bundle / "combined_identity_inventory.csv")
        entries = {str(resolve(w["source"]).resolve()): w for w in manifest["windows"]}
        for block in blocks:
            entry = entries.get(str(block.resolve()))
            if entry is None:
                raise ValueError(f"Bundle does not contain tracked block {block}")
            info = entry["info"]
            for stamp in info["sources"]:
                path = resolve(stamp["path"])
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (stamp["size"], stamp["mtime_ns"]):
                    raise ValueError(f"Upstream cache changed since bundle creation: {path}")
            regions = go.load_panorama_regions(go.panorama_regions_path(block))
            if regions.to_json(orient="records") != info["regions"]:
                raise ValueError(f"Annotations changed since bundle creation: {block}")
            part = bins[bins.source_block.eq(entry["source"])].copy()
            inv = inventory[inventory.source_block.eq(entry["source"])].copy()
            if part.empty or inv.empty:
                raise ValueError(f"Empty bundle window: {block}")
            windows.append(dict(bins=part, inventory=inv, info=info, block=block))
    else:
        for block in blocks:
            window = ba.load_window(block, Path(output) / "cache", label=block.parent.name + "_" + block.name,
                                    max_workers=workers)
            windows.append(dict(window, block=block))
    for window in windows[1:]:
        ba.validate_windows(windows[0], window)
    # Even non-overlapping ant selections must not hide overlapping recordings.
    intervals = sorted((pd.Timestamp(w["info"]["start_time"]), pd.Timestamp(w["info"]["stop_time"]),
                        str(w["block"])) for w in windows)
    for before, after in zip(intervals, intervals[1:]):
        if after[0] < before[1]:
            raise ValueError(f"Overlapping tracking windows would double-count time: {before[2]}, {after[2]}")
    return windows, resolve


def calendar_bins(bins, info):
    """Preserve global frame offsets, including window views beginning after day 0."""
    out = bins.copy()
    origin = pd.Timestamp(info["start_time"]) - pd.Timedelta(seconds=info["frame_start"] / info["fps"])
    out["timestamp"] = origin + pd.to_timedelta((out.bin_start_frame + out.bin_stop_frame) / (2 * info["fps"]), unit="s")
    out["calendar_date"] = out.timestamp.dt.strftime("%Y-%m-%d")
    out["phase"] = np.where((out.clock_hour >= info["light_on_hour"]) &
                            (out.clock_hour < info["light_off_hour"]), "light", "dark")
    out["ant"] = out.side + ":" + out.track_id.astype(int).astype(str).str.zfill(3)
    out["colony_percent"] = 100 - out.outside_percent
    return out


def interval_sum(values, starts, stops, width):
    lo0, hi0 = np.asarray(starts) / width, np.asarray(stops) / width
    if not np.allclose(lo0, np.rint(lo0)) or not np.allclose(hi0, np.rint(hi0)):
        raise ValueError("Clock bins must align with the existing context sampling")
    lo = np.clip(np.rint(lo0).astype(int), 0, len(values))
    hi = np.clip(np.rint(hi0).astype(int), 0, len(values))
    prefix = np.r_[0., np.cumsum(values, dtype=np.float64)]
    return prefix[hi] - prefix[lo]


def bin_resources(context, hits, bins, fps, width):
    """Exact cached resource frames / detected position frames; union overlaps."""
    out = bins.copy()
    position = interval_sum(context["position_count"], bins.bin_start_frame, bins.bin_stop_frame, width)
    out["resource_coverage"] = position / (bins.bin_stop_frame - bins.bin_start_frame)
    if ((out.resource_coverage < 0) | (out.resource_coverage > 1 + 1e-6)).any():
        raise ValueError("Invalid position-frame counts")
    for region in ("food", "water"):
        frames = np.sort(hits.loc[hits.region_type.eq(region), "frame"].unique())
        counts = np.searchsorted(frames, bins.bin_stop_frame) - np.searchsorted(frames, bins.bin_start_frame)
        if (counts > position + 1e-5).any():
            raise ValueError("Resource hits exceed detected positions; caches are incompatible")
        out[region + "_percent"] = np.divide(100. * counts, position,
                                             out=np.full(len(bins), np.nan), where=position > 0)
    return out


def sleep_bouts(context, info, identity):
    """One-second sleep runs; retain censor flags at gaps and recording edges."""
    dt = info["context_settings"]["position_bin_seconds"]
    first = max(0, int(np.ceil(info["frame_start"] / (dt * info["fps"]))))
    stop = int(np.floor(info["frame_stop"] / (dt * info["fps"])))
    state = np.asarray(context["state"])[first:stop]
    starts, ends, values = _run_length_encoding(state)
    origin = pd.Timestamp(info["start_time"]) - pd.Timedelta(seconds=info["frame_start"] / info["fps"])
    rows = []
    for i in np.flatnonzero(values == 1):
        left = i == 0 or values[i - 1] != 0
        right = i == len(values) - 1 or values[i + 1] != 0
        time = origin + pd.Timedelta(seconds=(first + starts[i]) * dt)
        clock = time.hour + time.minute / 60 + time.second / 3600
        rows.append(dict(**identity, timestamp=time, calendar_date=time.strftime("%Y-%m-%d"),
                         phase="light" if info["light_on_hour"] <= clock < info["light_off_hour"] else "dark",
                         duration_seconds=float((ends[i] - starts[i] + 1) * dt),
                         left_censored=left, right_censored=right, complete=not (left or right)))
    return rows


def all_recording_clock_bins(window, resolve, workers):
    """Bin existing vectors over the entire recording, including partial cycles.

    The upstream clock matrix deliberately prefers complete 24-hour cycles.
    Its ant selection is reusable, but its time-axis subset is unsuitable for
    a calendar timeline. Reuse the same binning implementation and dense cached
    vectors with partial cycles enabled; never reclassify sleep or reread poses.
    """
    info = window["info"]
    original = resolve(info["block"])
    clusters = window["inventory"][IDENTITY + ["track_name", "cluster_id"]].copy()
    clusters["cluster_id"] = clusters.cluster_id.fillna(clusters.side + "_unclustered")
    speeds = go.load_speed_tracks(original / "stitched/speed_vectors")
    sleeps = sma.load_sleep_label_tracks(original / "stitched/sleep_motion_labels", clusters)
    origin = pd.Timestamp(info["start_time"]) - pd.Timedelta(seconds=info["frame_start"] / info["fps"])
    bins, _, audit = sma.compute_activity_sleep_clock_profiles(
        clusters, speeds, sleeps, fps=info["fps"],
        recording_start_frame=info["frame_start"], recording_stop_frame=info["frame_stop"],
        start_clock_seconds=(origin - origin.normalize()).total_seconds(),
        light_on_hour=info["light_on_hour"], light_off_hour=info["light_off_hour"],
        bin_minutes=info["bin_minutes"], min_bin_coverage=0.,
        min_cycles=1, include_partial_cycles=True, recording_date=origin.strftime("%Y%m%d"),
        max_workers=workers,
    )
    if not (audit.speed_status.eq("ok").all() and audit.sleep_status.eq("ok").all()):
        raise FileNotFoundError(f"Missing existing clock vectors for {original}:\n{audit}")
    bins["full_recording_bin"] = bins.bin_start_frame.ge(info["frame_start"]) & bins.bin_stop_frame.le(info["frame_stop"])
    bins["in_recording_bin"] = bins.bin_start_frame.lt(info["frame_stop"]) & bins.bin_stop_frame.gt(info["frame_start"])
    return bins


def extend_window(window, resolve, output, workers):
    """Read current context and resource caches, never rescan tracks."""
    info, block = window["info"], window["block"]
    resource = block / "stitched/grid_occupancy_histograms_arena/panorama_region_analysis/resource_presence_frames.parquet"
    if not resource.is_file() or resource.stat().st_mtime_ns < go.panorama_regions_path(block).stat().st_mtime_ns:
        raise FileNotFoundError(f"Missing/current resource cache required: {resource}; run grid_occupancy first")
    original = resolve(info["block"])
    all_tracks = sma.load_sleep_label_tracks(original / "stitched/sleep_motion_labels")
    regions = go.load_panorama_regions(go.panorama_regions_path(original))
    context_settings = rs.ReturnSettings(**info["context_settings"])
    n_context_bins = int(all_tracks.frame_max.max()) // context_settings.bin_frames + 1
    paths, extra_sources = [], []
    inventory_names = set(window["inventory"].track_name)
    if set(all_tracks.track_name) != inventory_names:
        raise ValueError("Sleep-label and finished-track inventories disagree")
    for row in all_tracks.itertuples():
        if json.loads(row.classifier_parameters) != info["classifier"]:
            raise ValueError(f"Incompatible sleep classifier for {row.track_name}")
        path, _ = rs.context_cache_paths(row, original, regions, context_settings, n_context_bins,
                                        original / "stitched/analysis_cache/return_sleep")
        if not path.is_file():
            raise FileNotFoundError(f"Missing existing context for tracked ant: {path}")
        paths.append(path)
        speed = original / "stitched/speed_vectors/per_track" / Path(row.track_name).stem
        speed_meta = json.loads((speed / "speed_metadata.json").read_text())
        if any(speed_meta[key] != value for key, value in info["speed_parameters"].items()):
            raise ValueError(f"Incompatible speed parameters for {row.track_name}")
        for source in (path, Path(row.metadata_path), Path(row.state_path), speed / "speed_metadata.json", speed / "speed_mm_s.npy"):
            extra_sources.append(rs.fingerprint(source))
    contexts = {}
    for path in paths:
        name = path.stem.rsplit("_", 1)[0] + ".parquet"
        if name in contexts:
            raise ValueError(f"Ambiguous context cache: {name}")
        contexts[name] = path
    missing_resources = window["inventory"].loc[~window["inventory"].selected].copy()
    track_sources = [rs.fingerprint(original / "stitched/per_track" / name) for name in missing_resources.track_name]
    signature = dict(version=3, clock_policy="all_ants_all_recorded_cycles_no_display_coverage_cutoff",
                     info=info, resource=rs.fingerprint(resource), all_ant_sources=extra_sources,
                     added_resource_track_sources=track_sources)
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(output) / "cache" / f"{block.parent.name}_{block.name}_{digest}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    bins_path, bouts_path = cache.with_suffix(".parquet"), cache.with_suffix(".bouts.parquet")
    if bins_path.is_file() and bouts_path.is_file():
        print(f"Long-timescale cache hit: {block}", flush=True)
        return pd.read_parquet(bins_path), pd.read_parquet(bouts_path), signature
    hits = pd.read_parquet(resource, columns=["side", "track_id", "track_name", "frame", "region_type"])
    # The old resource cache was extracted only for the clustered subset. Reuse
    # it there, and extract only previously excluded ants from existing tracks.
    if len(missing_resources):
        resource_cache = cache.with_suffix(".resources.parquet")
        if resource_cache.is_file():
            additional = pd.read_parquet(resource_cache)
        else:
            missing_resources["cluster_id"] = missing_resources.side + "_unclustered"
            missing_resources["leiden_cluster"] = -1
            print(f"{block}: adding resource summaries for {len(missing_resources)} previously excluded ants", flush=True)
            additional = go.extract_resource_presence_frames(missing_resources, regions, original / "stitched/per_track", max_workers=workers)
            additional.to_parquet(resource_cache, index=False)
        hits = pd.concat([hits, additional[hits.columns]], ignore_index=True)
    grouped = {name: table for name, table in hits.groupby("track_name", sort=False)}
    template = hits.iloc[:0]
    bins = all_recording_clock_bins(window, resolve, workers)
    width = round(info["fps"] * info["context_settings"]["position_bin_seconds"])
    def read(part):
        name = part.track_name.iloc[0]
        if name not in contexts:
            raise FileNotFoundError(f"No validated existing context for {name}")
        with np.load(contexts[name]) as context:
            part = ba.bin_context(context, part, bin_frames=width, n_context_bins=len(context["inside"]))
            part = calendar_bins(part, info)
            expanded = bin_resources(context, grouped.get(name, template), part, info["fps"], width)
            identity = dict(side=part.side.iloc[0], track_id=int(part.track_id.iloc[0]),
                            ant=part.ant.iloc[0], recording=block.parent.name, source_block=str(block))
            bouts = sleep_bouts(context, info, identity)
        return expanded, bouts
    parts, events = [], []
    groups = [part for _, part in bins.groupby("track_name", sort=True)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (part, bouts) in enumerate(pool.map(read, groups), 1):
            parts.append(part)
            events.extend(bouts)
            if i == 1 or i % 25 == 0 or i == len(groups):
                print(f"{block.parent.name}/{block.name}: existing caches {i}/{len(groups)} ants", flush=True)
    bins = pd.concat(parts, ignore_index=True).assign(recording=block.parent.name, source_block=str(block))
    columns = ["side", "track_id", "ant", "recording", "source_block", "timestamp", "calendar_date", "phase",
               "duration_seconds", "left_censored", "right_censored", "complete"]
    bouts = pd.DataFrame(events, columns=columns)
    bins.to_parquet(bins_path, index=False)
    bouts.to_parquet(bouts_path, index=False)
    return bins, bouts, signature


def to_long(bins, settings):
    settings.validate()
    if settings.min_bin_coverage < bins.min_bin_coverage.max():
        raise ValueError("Requested coverage is below the threshold used in upstream caches")
    if bins.duplicated(IDENTITY + ["timestamp"]).any():
        # Non-full clock templates can extend outside the recording; only the
        # genuinely recorded parts must be unique.
        recorded = bins[bins.in_recording_bin]
        if recorded.duplicated(IDENTITY + ["timestamp"]).any():
            raise ValueError("Overlapping ant/time bins would double-count recordings")
    columns = IDENTITY + ["ant", "timestamp", "calendar_date", "recording", "source_block", "phase",
                           "clock_hour", "zt_bin", "bin_minutes", "full_recording_bin", "in_recording_bin"]
    parts = []
    for metric, (value, coverage, _, _) in METRICS.items():
        part = bins[columns].copy()
        part["metric"], part["coverage"], part["raw_value"] = metric, bins[coverage], bins[value]
        part["valid"] = bins.in_recording_bin & bins[coverage].ge(settings.min_bin_coverage) & np.isfinite(bins[value])
        part["value"] = bins[value].where(part.valid)
        part["valid_hours"] = (bins.bin_minutes / 60 * bins[coverage]).where(part.valid, 0)
        parts.append(part)
    # Rows outside recorded time are unnecessary; true partial bins remain in
    # the audit but are masked from estimates and displayed as missing.
    return pd.concat(parts, ignore_index=True)


def clock_profiles(long):
    return long[long.in_recording_bin].groupby(IDENTITY + ["ant", "metric", "recording", "phase", "zt_bin", "clock_hour"], as_index=False).agg(
        value=("value", "mean"), cycles=("value", "count"), valid_hours=("valid_hours", "sum"),
        bin_minutes=("bin_minutes", "first"))


def phase_table(long, settings):
    groups = IDENTITY + ["ant", "metric", "calendar_date", "phase"]
    out = long[long.in_recording_bin].groupby(groups, as_index=False).agg(
        value=("value", "mean"), valid_bins=("value", "count"), valid_hours=("valid_hours", "sum"),
        bin_minutes=("bin_minutes", "first"))
    out["covered_clock_hours"] = out.valid_bins * out.bin_minutes / 60
    out["included"] = (out.valid_bins > 0) & (out.covered_clock_hours >= settings.min_phase_hours)
    out["value"] = out.value.where(out.included)
    return out


def standardize_all_recordings(profiles, recordings, settings):
    """Same ant, same clock slots in EVERY recording; equal clock-slot weights.

    Phase estimates independently require the requested shared phase support.
    Ants failing all-recording support remain in the audit with included=False.
    """
    rows, slots = [], []
    for keys, data in profiles.groupby(IDENTITY + ["ant", "metric"], sort=True):
        wide = data.pivot(index=["zt_bin", "clock_hour", "phase"], columns="recording", values="value").reindex(columns=recordings)
        shared = wide.dropna()
        step = float(data.bin_minutes.iloc[0]) / 60
        for phase in ("all", "light", "dark"):
            subset = shared if phase == "all" else shared[shared.index.get_level_values("phase") == phase]
            hours = len(subset) * step
            required = settings.min_shared_hours if phase == "all" else settings.min_phase_hours
            means = subset.mean()
            for recording in recordings:
                rows.append(dict(zip(IDENTITY + ["ant", "metric"], keys), recording=recording, phase=phase,
                                 shared_hours=hours, included=hours > 0 and hours >= required,
                                 value=float(means.get(recording, np.nan)) if hours > 0 and hours >= required else np.nan))
        if len(shared):
            flat = shared.reset_index().melt(id_vars=["zt_bin", "clock_hour", "phase"], value_vars=recordings,
                                             var_name="recording", value_name="value")
            for key, value in zip(IDENTITY + ["ant", "metric"], keys):
                flat[key] = value
            slots.append(flat)
    return pd.DataFrame(rows), pd.concat(slots, ignore_index=True) if slots else pd.DataFrame()


def summarize_ants(table, groups, settings):
    """Equal-weight ant means and bootstrap intervals within each colony."""
    rng = np.random.default_rng(settings.seed)
    rows = []
    for keys, data in table.groupby(groups, sort=True, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = data.value.dropna().to_numpy(float)
        lo = hi = np.nan
        if len(values) >= 5:
            boot = values[rng.integers(len(values), size=(settings.bootstrap, len(values)))].mean(axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
        rows.append(dict(zip(groups, keys), mean=float(np.mean(values)) if len(values) else np.nan,
                         median=float(np.median(values)) if len(values) else np.nan,
                         n_ants=len(values), ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def longitudinal_slopes(standardized, recording_times, settings):
    """Descriptive per-ant linear slopes across all recordings, on actual days."""
    rows = []
    recordings = list(recording_times)
    t = pd.to_datetime(list(recording_times.values()))
    x = np.asarray((t - t.min()).total_seconds()) / 86400
    for keys, part in standardized[standardized.phase.eq("all")].groupby(IDENTITY + ["ant", "metric"]):
        part = part.set_index("recording").reindex(recordings)
        y = part.value.to_numpy(float)
        included = np.isfinite(y).all() and len(y) >= 3 and np.ptp(x) > 0
        slope = float(np.polyfit(x, y, 1)[0]) if included else np.nan
        rows.append(dict(zip(IDENTITY + ["ant", "metric"], keys), included=included, slope_per_day=slope,
                         first_value=y[0], last_value=y[-1], shared_hours=part.shared_hours.min()))
    ants = pd.DataFrame(rows)
    summary = summarize_ants(ants.rename(columns={"slope_per_day": "value"}), ["side", "metric"], settings)
    return ants, summary


def build_tables(windows, resolve, output, settings, workers=4):
    pieces, events, signatures, inventories = [], [], [], []
    for window in windows:
        bins, bouts, signature = extend_window(window, resolve, output, workers)
        pieces.append(bins)
        events.append(bouts)
        signatures.append(signature)
        inv = window["inventory"].copy()
        inv["recording"] = window["block"].parent.name
        inv["source_block"] = str(window["block"])
        inventories.append(inv)
    bins = pd.concat(pieces, ignore_index=True)
    long = to_long(bins, settings)
    profiles = clock_profiles(long)
    recordings = sorted(bins.recording.unique())
    standardized, common = standardize_all_recordings(profiles, recordings, settings)
    times = {}
    for recording in recordings:
        intervals = [(pd.Timestamp(w["info"]["start_time"]), pd.Timestamp(w["info"]["stop_time"]))
                     for w in windows if w["block"].parent.name == recording]
        weights = np.array([(b-a).total_seconds() for a, b in intervals])
        centers = np.array([(a+(b-a)/2).value for a, b in intervals], dtype=float)
        times[recording] = str(pd.Timestamp(int(np.average(centers, weights=weights))))
    phase = phase_table(long, settings)
    observed_parts = []
    for phase_name in ("all", "light", "dark"):
        data = long if phase_name == "all" else long[long.phase.eq(phase_name)]
        means = data[data.in_recording_bin].groupby(IDENTITY + ["ant", "metric", "recording"], as_index=False).agg(
            value=("value", "mean"), valid_bins=("value", "count"), valid_hours=("valid_hours", "sum"))
        means["phase"] = phase_name
        means["included"] = means.valid_bins > 0
        observed_parts.append(means)
    observed = pd.concat(observed_parts, ignore_index=True)
    slopes, slope_summary = longitudinal_slopes(standardized, times, settings)
    tables = dict(bins=bins, long=long, profiles=profiles, standardized=standardized, common_clock_bins=common,
                  phases=phase, sleep_bouts=pd.concat(events, ignore_index=True),
                  inventory=pd.concat(inventories, ignore_index=True), slopes=slopes, slope_summary=slope_summary,
                  observed_recording_means=observed,
                  observed_recording_summary=summarize_ants(observed, ["side", "metric", "recording", "phase"], settings),
                  phase_summary=summarize_ants(phase, ["side", "metric", "calendar_date", "phase"], settings),
                  standardized_summary=summarize_ants(standardized, ["side", "metric", "recording", "phase"], settings))
    for name, table in tables.items():
        table.to_parquet(Path(output) / (name + ".parquet"), index=False)
        if name not in ("bins", "long", "common_clock_bins", "sleep_bouts"):
            table.to_csv(Path(output) / (name + ".csv"), index=False)
    return tables, dict(settings=asdict(settings), recordings=recordings, recording_centers=times,
                        bin_minutes=float(bins.bin_minutes.iloc[0]), clock_hours=sorted(bins.clock_hour.unique().tolist()),
                        windows=[dict(block=str(w["block"]), start=w["info"]["start_time"], stop=w["info"]["stop_time"])
                                 for w in windows],
                        light_on_hour=windows[0]["info"]["light_on_hour"], light_off_hour=windows[0]["info"]["light_off_hour"],
                        metrics=METRICS, sources=signatures)
