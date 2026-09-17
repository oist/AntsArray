"""Individual task-allocation diagnostics assembled from existing analysis caches.

Trip definitions and events come from grid_occupancy's trip_phenotyping_utils.
An ant omitted from that optional upstream analysis has unknown trip metrics,
not zero trips. No raw tracks are scanned and no cross-block trips are invented.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import long_timescale_utils as lt
from analysis import return_sleep_utils as rs
from analysis import trip_phenotyping_utils as trips

TRIP_METRICS = {
    "trip_count": ("trip_count", "trip_coverage", "Completed-trip departures", "trips"),
    "trip_rate": ("trip_rate", "trip_coverage", "Completed-trip rate", "trips/observed h"),
    "trip_duration": ("trip_duration_minutes", "trip_coverage", "Mean completed-trip duration", "min"),
    "trip_investment": ("trip_investment_percent", "trip_coverage", "Observed time in completed trips", "%"),
}
METRICS = {**lt.METRICS, **TRIP_METRICS}
TRIP_SETTINGS = dict(bodypoint=0, position_bin_seconds=1., max_state_gap_seconds=30.,
                     min_state_run_seconds=5., min_colony_anchor_seconds=5.,
                     min_trip_seconds=30., min_trip_coverage=.2, max_path_gap_seconds=3.,
                     trip_entropy_bin_mm=10.)


def bin_trip_events(template, events, positions, observed_seconds, *, fps, dt=1.):
    """Count departures once; allocate observed trip seconds to their own bin.

    A trip straddling midnight contributes one departure, its full duration to
    that departure's bin, and its observed outside time to both calendar days.
    """
    out = template.copy()
    lo, hi = out.bin_start_frame.to_numpy(), out.bin_stop_frame.to_numpy()
    exposure = lt.interval_sum(observed_seconds.astype(float), lo, hi, fps * dt) * dt
    exits = events.sort_values("exit_frame") if len(events) else events
    frames = exits.exit_frame.to_numpy() if len(exits) else np.array([])
    first, last = np.searchsorted(frames, lo), np.searchsorted(frames, hi)
    number = last - first
    sums = np.r_[0., np.cumsum(exits.duration_minutes)] if len(exits) else np.array([0.])
    durations = sums[last] - sums[first]
    pos = np.sort(positions.elapsed_seconds.to_numpy() * fps) if len(positions) else np.array([])
    investment = (np.searchsorted(pos, hi) - np.searchsorted(pos, lo)) * dt
    if (investment > exposure + 1e-6).any():
        raise ValueError("Cached trip seconds exceed observed position exposure")
    out["trip_observed_hours"] = exposure / 3600
    out["trip_coverage"] = exposure * fps / (hi - lo)
    out["trip_count"] = np.where(exposure > 0, number, np.nan)
    out["trip_duration_sum_minutes"] = durations
    out["trip_observed_outside_seconds"] = investment
    out["trip_rate"] = np.divide(number * 3600., exposure, out=np.full(len(out), np.nan), where=exposure > 0)
    out["trip_duration_minutes"] = np.divide(durations, number, out=np.full(len(out), np.nan), where=number > 0)
    out["trip_investment_percent"] = np.divide(investment * 100., exposure, out=np.full(len(out), np.nan), where=exposure > 0)
    return out


def load_window_trips(window, bins, signature, resolve):
    """Reuse the saved optional trip analysis and exact position exposure."""
    info, block = window["info"], window["block"]
    root = block / "stitched/grid_occupancy_histograms_arena/panorama_region_analysis/optional_trip_phenotyping"
    paths = [root / name for name in ("trip_extraction_metadata.json", "completed_trips.parquet",
                                     "completed_trip_positions_1s.parquet", "trip_extraction_diagnostics.csv")]
    out = bins[bins.in_recording_bin].copy()
    for name in [v[0] for v in TRIP_METRICS.values()] + ["trip_coverage", "trip_observed_hours",
                "trip_duration_sum_minutes", "trip_observed_outside_seconds"]:
        out[name] = np.nan
    out["trip_available"] = False
    origin = pd.Timestamp(info["start_time"]) - pd.Timedelta(seconds=info["frame_start"] / info["fps"])
    audit = dict(block=str(block), analyzed_ants=0, tracked_ants=out.ant.nunique(), sources=[],
                 status="optional trip cache unavailable")
    empty = pd.DataFrame(columns=["trip_id", "ant", "recording", "source_block", "exit_timestamp", "return_timestamp"])
    if not all(p.is_file() for p in paths):
        return out, empty, audit
    metadata = json.loads(paths[0].read_text())
    rectangles = trips.colony_rectangles_from_regions(lt.go.load_panorama_regions(lt.go.panorama_regions_path(block)))
    expected = dict(TRIP_SETTINGS, fps=info["fps"], colony_rectangles=rectangles,
                    start_clock_seconds=(origin - origin.normalize()).total_seconds())
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Trip definitions/annotations differ from the grid analysis: {root}")
    events = pd.read_parquet(paths[1])
    positions = pd.read_parquet(paths[2], columns=["track_name", "elapsed_seconds"])
    diagnostics = pd.read_csv(paths[3])
    names = set(metadata["track_names"])
    if names != set(diagnostics.track_name):
        raise ValueError(f"Incomplete optional trip cache: {root}")
    if not names.issubset(set(out.track_name)):
        raise ValueError(f"Trip cache contains unknown tracked identities: {root}")
    track_sources = [rs.fingerprint(resolve(info["block"]) / "stitched/per_track" / name) for name in sorted(names)]
    if any(s["mtime_ns"] > paths[0].stat().st_mtime_ns for s in track_sources):
        raise ValueError(f"Finished tracks are newer than trip cache: {root}")
    contexts = {Path(s["path"]).stem.rsplit("_", 1)[0] + ".parquet": resolve(s["path"])
                for s in signature["all_ant_sources"] if s["path"].endswith(".npz")}
    eg = {name: part for name, part in events.groupby("track_name")}
    pg = {name: part for name, part in positions.groupby("track_name")}
    for name in sorted(names):
        mask = out.track_name.eq(name)
        with np.load(contexts[name]) as context:
            observed = np.asarray(context["position_count"]) > 0
        part = bin_trip_events(out.loc[mask], eg.get(name, events.iloc[:0]), pg.get(name, positions.iloc[:0]),
                               observed, fps=info["fps"])
        out.loc[mask, part.columns] = part
        out.loc[mask, "trip_available"] = True
    if int(out.trip_count.sum()) != len(events):
        raise ValueError(f"Trip departure counts were lost or duplicated: {root}")
    if not (events.groupby("track_name").size().reindex(diagnostics.track_name, fill_value=0).to_numpy()
            == diagnostics.n_completed_trips.to_numpy()).all():
        raise ValueError(f"Trip diagnostics disagree with events: {root}")
    events["ant"] = events.side + ":" + events.track_id.astype(int).astype(str).str.zfill(3)
    events["recording"], events["source_block"] = block.parent.name, str(block)
    events["trip_id"] = block.parent.name + "/" + block.name + "/" + events.trip_id
    events["exit_timestamp"] = origin + pd.to_timedelta(events.exit_frame / info["fps"], unit="s")
    events["return_timestamp"] = origin + pd.to_timedelta(events.return_frame / info["fps"], unit="s")
    audit.update(analyzed_ants=len(names), completed_trips=len(events), status="existing grid trip cache",
                 settings=metadata, sources=[rs.fingerprint(p) for p in paths] + track_sources)
    return out, events, audit


def task_long(bins):
    columns = lt.IDENTITY + ["ant", "timestamp", "calendar_date", "recording", "source_block", "phase",
                             "clock_hour", "zt_bin", "bin_minutes", "in_recording_bin"]
    pieces = []
    for metric, (column, coverage, _, _) in METRICS.items():
        part = bins[columns].copy()
        part["metric"], part["value"] = metric, bins[column]
        part["valid_hours"] = (bins[coverage] * bins.bin_minutes / 60).where(bins[column].notna(), 0.)
        # Daily values use each measure's actual denominator, not a 24 h estimate.
        if metric == "trip_count":
            part["weight"] = np.where(bins[column].notna(), 1., 0.)
        elif metric == "trip_duration":
            part["weight"] = bins.trip_count.where(bins[column].notna(), 0.).fillna(0.)
        else:
            part["weight"] = part.valid_hours
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def daily_profiles(long):
    rows = []
    for phase in ("all", "light", "dark"):
        data = long if phase == "all" else long[long.phase.eq(phase)]
        data = data.copy()
        data["weighted"] = data.value.fillna(0) * data.weight
        part = data.groupby(lt.IDENTITY + ["ant", "calendar_date", "metric"], as_index=False).agg(
            total=("weighted", "sum"), denominator=("weight", "sum"), valid_hours=("valid_hours", "sum"),
            n_bins=("value", "count"))
        part["value"] = part.total / part.denominator.where(part.denominator > 0)
        count = part.metric.eq("trip_count") & part.n_bins.gt(0)
        part.loc[count, "value"] = part.loc[count, "total"]
        part["phase"] = phase
        rows.append(part.drop(columns="total"))
    return pd.concat(rows, ignore_index=True)


def build(tables, manifest, windows, resolve, output, settings):
    pieces, events, sources = [], [], []
    for window, signature in zip(windows, manifest["sources"]):
        part = tables["bins"][tables["bins"].source_block.eq(str(window["block"]))]
        data, event, audit = load_window_trips(window, part, signature, resolve)
        pieces.append(data); events.append(event); sources.append(audit)
        print(f"Task profiles {window['block']}: existing trip analyses for {audit['analyzed_ants']}/{audit['tracked_ants']} ants", flush=True)
    bins = pd.concat(pieces, ignore_index=True)
    long = task_long(bins)
    profiles = lt.clock_profiles(long[long.metric.isin(TRIP_METRICS) & ~long.metric.eq("trip_count")])
    standardized, _ = lt.standardize_all_recordings(profiles, manifest["recordings"], settings)
    standardized = pd.concat([tables["standardized"], standardized], ignore_index=True)
    slopes, summary = lt.longitudinal_slopes(standardized, manifest["recording_centers"], settings)
    extra = dict(task_bins=bins, completed_trips=pd.concat(events, ignore_index=True),
                 task_daily=daily_profiles(long), task_standardized=standardized,
                 task_slopes=slopes, task_slope_summary=summary,
                 trip_availability=pd.DataFrame([{k: v for k, v in s.items() if k not in ("sources", "settings")} for s in sources]))
    for name, data in extra.items():
        data.to_parquet(Path(output) / (name + ".parquet"), index=False)
        if name != "task_bins":
            data.to_csv(Path(output) / (name + ".csv"), index=False)
    tables.update(extra)
    manifest["task_analysis"] = dict(metrics=METRICS, trip_sources=sources,
                                    definitions="Existing grid_occupancy completed trips; unavailable optional analyses remain unknown")
    return tables
