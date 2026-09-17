"""Investigate changing trip duration using existing longitudinal analysis tables.

All completed events are retained in the primary views. Matching and sensitivity
subsets are explicit additional analyses, with event counts and exposure saved.
Run on Deigo with the same environment as long_timescale.py; no tracking or raw
position extraction is performed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis import long_timescale_utils as lt

MEASURES = {"mean": "Mean trip duration (min)", "median": "Median trip duration (min)",
            "p90": "90th percentile (min)", "long10": "Trips lasting ≥10 min (%)"}


def weighted_quantile(values, weights, quantile):
    """Inverse weighted empirical CDF; the convention is explicit for matching."""
    order = np.argsort(values)
    x, w = np.asarray(values)[order], np.asarray(weights, dtype=float)[order]
    if not len(x) or w.sum() <= 0:
        return np.nan
    return float(x[min(np.searchsorted(np.cumsum(w), quantile * w.sum()), len(x)-1)])


def duration_summary(data, weights=None):
    if data.empty:
        return dict(n_trips=0, mean=np.nan, median=np.nan, p90=np.nan, long10=np.nan, coverage=np.nan)
    x = data.duration_minutes.to_numpy(float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    return dict(n_trips=len(x), mean=float(np.average(x, weights=w)),
                median=float(np.median(x)) if weights is None else weighted_quantile(x, w, .5),
                p90=float(np.quantile(x, .9)) if weights is None else weighted_quantile(x, w, .9),
                long10=float(np.average(x >= 10, weights=w) * 100),
                coverage=float(np.average(data.observed_coverage, weights=w)))


def add_calendar(events, manifest):
    out = events.copy()
    times = out.exit_timestamp
    out["calendar_date"] = times.dt.strftime("%Y-%m-%d")
    out["clock_hour"] = times.dt.hour + times.dt.minute / 60 + times.dt.second / 3600
    on, off = manifest["light_on_hour"], manifest["light_off_hour"]
    out["phase"] = np.where(out.clock_hour.ge(on) & out.clock_hour.lt(off), "light", "dark")
    out["cycle_day"] = (times - pd.Timedelta(hours=on)).dt.strftime("%Y-%m-%d")
    # Two-hour strata begin at lights on, so no stratum straddles a phase change
    # in the existing 14 h light / 10 h dark schedule.
    if not np.isclose((off - on) % 2, 0):
        raise ValueError("Two-hour clock strata require phase changes on the stratum grid")
    out["clock_stratum"] = (((out.clock_hour - on) % 24) // 2).astype(int)
    starts = {w["block"]: pd.Timestamp(w["start"]) for w in manifest["windows"]}
    stops = {w["block"]: pd.Timestamp(w["stop"]) for w in manifest["windows"]}
    out["minutes_since_block_start"] = (times - out.source_block.map(starts)).dt.total_seconds() / 60
    out["minutes_until_block_stop"] = (out.source_block.map(stops) - times).dt.total_seconds() / 60
    return out


def phase_coverage(manifest, days):
    rows = []
    for day in days:
        for phase in ("light", "dark"):
            start = pd.Timestamp(day) + pd.Timedelta(hours=manifest["light_on_hour"] if phase == "light" else manifest["light_off_hour"])
            stop = pd.Timestamp(day) + (pd.Timedelta(hours=manifest["light_off_hour"]) if phase == "light" else pd.Timedelta(days=1, hours=manifest["light_on_hour"]))
            intervals = [(max(start, pd.Timestamp(w["start"])), min(stop, pd.Timestamp(w["stop"]))) for w in manifest["windows"]]
            hours = sum(max(0., (b-a).total_seconds() / 3600) for a, b in intervals)
            rows.append(dict(cycle_day=day, phase=phase, recording_hours=hours,
                             phase_hours=(stop-start).total_seconds()/3600))
    return pd.DataFrame(rows)


def observed_summaries(events, bins):
    rows = []
    for phase in ("all", "light", "dark"):
        ev = events if phase == "all" else events[events.phase.eq(phase)]
        obs = bins if phase == "all" else bins[bins.phase.eq(phase)]
        grouped = {key: part for key, part in ev.groupby(["side", "ant", "recording"])}
        for key, part in obs.groupby(["side", "ant", "recording"]):
            data = grouped.get(key, ev.iloc[:0])
            available = bool(part.trip_available.any())
            hours = float(part.trip_observed_hours.sum()) if available else np.nan
            stats = duration_summary(data)
            if not available:
                stats["n_trips"] = np.nan
            rows.append(dict(zip(("side", "ant", "recording"), key), phase=phase,
                             **stats, trip_available=available, observed_hours=hours,
                             trip_rate=len(data)/hours if hours > 0 else np.nan))
    return pd.DataFrame(rows)


def clock_matched(events, bins, recordings):
    """Equal weight per shared clock stratum, then per ant in group summaries.

    A stratum must contain a completed departure in every recording to compare
    its conditional duration distribution. There is no event-count cutoff.
    """
    rows, supports = [], []
    for (side, ant), data in events.groupby(["side", "ant"]):
        if set(data.recording) != set(recordings):
            continue
        counts = data.groupby(["clock_stratum", "recording"]).size().unstack().reindex(columns=recordings)
        shared = counts.dropna().index
        for phase in ("all", "light", "dark"):
            eligible = data if phase == "all" else data[data.phase.eq(phase)]
            slots = sorted(set(eligible.clock_stratum).intersection(shared))
            for recording in recordings:
                part = eligible[eligible.recording.eq(recording) & eligible.clock_stratum.isin(slots)]
                obs = bins[bins.ant.eq(ant) & bins.recording.eq(recording) & bins.clock_stratum.isin(slots)]
                if part.empty:
                    continue
                weights = 1. / part.groupby("clock_stratum").duration_minutes.transform("size")
                stats = duration_summary(part, weights)
                hours = float(obs.trip_observed_hours.sum())
                # Equal-stratum trip rates use exactly the same clock strata.
                rates = part.groupby("clock_stratum").size() / obs.groupby("clock_stratum").trip_observed_hours.sum()
                rows.append(dict(side=side, ant=ant, phase=phase, recording=recording, **stats,
                                 shared_strata=len(slots), observed_hours=hours, trip_rate=float(rates.mean())))
                for slot in slots:
                    supports.append(dict(side=side, ant=ant, phase=phase, recording=recording, clock_stratum=slot,
                                         n_trips=int(part.clock_stratum.eq(slot).sum()),
                                         observed_hours=float(obs.loc[obs.clock_stratum.eq(slot), "trip_observed_hours"].sum())))
    return pd.DataFrame(rows), pd.DataFrame(supports)


def changes(summary, recordings, *, bootstrap=2000):
    rows = []
    rng = np.random.default_rng(0)
    for (side, phase), part in summary.groupby(["side", "phase"]):
        for metric in list(MEASURES) + (["trip_rate"] if "trip_rate" in part else []):
            wide = part.pivot(index="ant", columns="recording", values=metric).reindex(columns=recordings).dropna()
            if wide.empty:
                continue
            first, last = wide.iloc[:, 0].to_numpy(), wide.iloc[:, -1].to_numpy()
            ratio = np.divide(last, first, out=np.full(len(first), np.nan), where=first > 0)
            finite = ratio[np.isfinite(ratio)]
            boot = np.median(finite[rng.integers(len(finite), size=(bootstrap, len(finite)))], axis=1) if len(finite) else np.array([np.nan])
            lo, hi = np.quantile(boot, [.025, .975])
            rows.append(dict(side=side, phase=phase, metric=metric, n_ants=len(wide), increased=int((last > first).sum()),
                             decreased=int((last < first).sum()), median_first=float(np.median(first)), median_last=float(np.median(last)),
                             median_change=float(np.median(last-first)), median_fold=float(np.median(finite)) if len(finite) else np.nan,
                             fold_ci_low=lo, fold_ci_high=hi, n_ratios=len(finite),
                             monotonic_increase=int((np.diff(wide.to_numpy(), axis=1) > 0).all(axis=1).sum())))
    return pd.DataFrame(rows)


def add_sleep_context(events, manifest, resolve, workers):
    """Measure classified sleep during each trip using existing 1 Hz contexts."""
    contexts = {}
    for window, source in zip(manifest["windows"], manifest["sources"]):
        for stamp in source["all_ant_sources"]:
            if stamp["path"].endswith(".npz"):
                path = resolve(stamp["path"])
                if (path.stat().st_size, path.stat().st_mtime_ns) != (stamp["size"], stamp["mtime_ns"]):
                    raise ValueError(f"Context changed since combined analysis: {path}")
                contexts[window["block"], path.stem.rsplit("_", 1)[0] + ".parquet"] = path
    info = {w["block"]: s["info"] for w, s in zip(manifest["windows"], manifest["sources"])}
    def extract(item):
        key, part = item
        cfg = info[key[0]]
        width = cfg["fps"] * cfg["context_settings"]["position_bin_seconds"]
        with np.load(contexts[key]) as context:
            state = context["state"]
            known = lt.interval_sum((state >= 0).astype(float), part.exit_frame, part.return_frame, width)
            asleep = lt.interval_sum((state == 1).astype(float), part.exit_frame, part.return_frame, width)
        part = part.copy()
        part["sleep_percent"] = np.divide(100*asleep, known, out=np.full(len(part), np.nan), where=known > 0)
        part["sleep_coverage"] = known * width / (part.return_frame - part.exit_frame)
        return part
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return pd.concat(pool.map(extract, events.groupby(["source_block", "track_name"], sort=True)), ignore_index=True)


def build_tables(events, bins, manifest, resolve, workers, bootstrap):
    events = add_calendar(events, manifest)
    bins = bins.copy()
    bins["clock_stratum"] = (((bins.clock_hour - manifest["light_on_hour"]) % 24) // 2).astype(int)
    events = add_sleep_context(events, manifest, resolve, workers)
    recordings = manifest["recordings"]
    observed = observed_summaries(events, bins)
    matched, support = clock_matched(events, bins, recordings)
    cycles = []
    for (side, ant, day, phase), part in events.groupby(["side", "ant", "cycle_day", "phase"]):
        cycles.append(dict(side=side, ant=ant, cycle_day=day, phase=phase, **duration_summary(part)))
    cycle_table = pd.DataFrame(cycles)
    availability = phase_coverage(manifest, pd.date_range(events.cycle_day.min(), events.cycle_day.max()).strftime("%Y-%m-%d"))
    # A sensitivity check for short-window censoring: require one hour of
    # recording on either side of departure and examine durations <= 1 hour.
    rules = {"all_completed": np.ones(len(events), dtype=bool),
             "coverage_at_least_95_percent": events.observed_coverage.ge(.95),
             "one_hour_edge_buffer_and_duration_cap": events.minutes_since_block_start.ge(60) & events.minutes_until_block_stop.ge(60) & events.duration_minutes.le(60)}
    sensitivity = []
    for method, mask in rules.items():
        subset = observed_summaries(events[mask], bins)
        delta = changes(subset, recordings, bootstrap=bootstrap)
        delta["method"], delta["retained_trips"] = method, int(np.sum(mask))
        sensitivity.append(delta)
    # Identify complete dark phases contained entirely within the 0724 source
    # (kept separate from the cross-recording comparison).
    within = []
    for recording in recordings:
        windows = [w for w in manifest["windows"] if Path(w["block"]).parent.name == recording]
        complete_days = []
        for day in sorted(events.loc[events.recording.eq(recording), "cycle_day"].unique()):
            start = pd.Timestamp(day) + pd.Timedelta(hours=manifest["light_off_hour"])
            stop = pd.Timestamp(day) + pd.Timedelta(days=1, hours=manifest["light_on_hour"])
            if any(pd.Timestamp(w["start"]) <= start and pd.Timestamp(w["stop"]) >= stop for w in windows):
                complete_days.append(day)
        if len(complete_days) < 2:
            continue
        part = events[events.recording.eq(recording) & events.phase.eq("dark") & events.cycle_day.isin(complete_days)]
        for (side, ant, day), data in part.groupby(["side", "ant", "cycle_day"]):
            within.append(dict(side=side, ant=ant, phase="dark", recording=day, source_recording=recording, **duration_summary(data)))
    within_table = pd.DataFrame(within)
    within_changes = []
    if not within_table.empty:
        for rec, part in within_table.groupby("source_recording"):
            ch = changes(part, sorted(part.recording.unique()), bootstrap=bootstrap)
            ch["source_recording"] = rec
            within_changes.append(ch)
    return dict(events=events, observed=observed, clock_matched=matched, clock_support=support,
                observed_changes=changes(observed, recordings, bootstrap=bootstrap),
                clock_matched_changes=changes(matched, recordings, bootstrap=bootstrap),
                cycle_profiles=cycle_table, phase_coverage=availability,
                sensitivity=pd.concat(sensitivity, ignore_index=True), complete_nights=within_table,
                complete_night_changes=pd.concat(within_changes, ignore_index=True) if within_changes else pd.DataFrame())


def write_report(tables, manifest, output):
    raw = tables["observed_changes"].query("phase == 'all'")
    lines = ["# Are the same ants taking longer trips?", "",
             "The existing colony-use/trip-investment plot shows one ant per calendar-day point. Color is calendar date; it is not elapsed time within a block. The lower panels use arithmetic mean completed-trip duration, which is sensitive to long excursions.", "",
             "## Within-ant evidence across all three recordings", "",
             "This comparison uses ants with completed trips in every recording. It does not impose a minimum trip-count or coverage filter. Each ant receives equal weight.", "",
             "| Colony | Measure | Ants increased | Median first value | Median last value | Median within-ant fold |",
             "|---|---|---:|---:|---:|---:|"]
    for r in raw.itertuples():
        label = MEASURES.get(r.metric, "Trips per observed hour")
        lines.append(f"| {r.side} | {label} | {r.increased}/{r.n_ants} | {r.median_first:.3g} | {r.median_last:.3g} | {r.median_fold:.3g} |")
    lines += ["", "A fold change is calculated within each ant before taking the median; it is not the ratio of the two group medians.", "",
              "## Clock time and partial days", "",
              "The 20260729 tracked window actually spans July 30 21:48:50 to July 31 21:23:38. July 30 therefore contains only about 2 h 11 min of nighttime departures; July 31 mixes night and day. Those calendar-day means are not comparable 24-hour budgets. New cycle plots assign an entire night to the date on which darkness begins, and print the recorded phase hours.", "",
              "Clock-matched comparisons give equal weight to 2-hour clock strata containing completed departures in every recording, separately within each ant and phase. Strata start at lights on (05:30). Observations within each stratum share its weight; busy periods cannot dominate just because they contain more trips. This is a conditional duration comparison, not an estimate for clock strata with no completed trips.", "",
              "| Colony | Phase | Median duration increased | Median within-ant fold | Shared-stratum weighting |",
              "|---|---|---:|---:|---|"]
    for r in tables["clock_matched_changes"].query("metric == 'median'").itertuples():
        lines.append(f"| {r.side} | {r.phase} | {r.increased}/{r.n_ants} | {r.median_fold:.3g} | Equal 2 h strata |")
    lines += ["", "## Change within a continuous recording", ""]
    for r in tables["complete_night_changes"].query("metric == 'median'").itertuples():
        lines.append(f"- {r.source_recording}, {r.side}: median duration increased in {r.increased}/{r.n_ants} ants between the first and last fully recorded nights; median within-ant fold {r.median_fold:.3g}.")
    lines += ["", "## Coverage, trip tails, and interpretation", "",
              "The sensitivity figure separately checks trips with at least 95% observed position coverage, and trips no longer than one hour whose departures are at least one hour from both recording boundaries. These are labeled extra checks; the primary figures retain every saved completed trip. Counts and results for each check are in sensitivity.csv.", "",
              "Only completed colony–outside–colony events exist in these caches. Trips censored by block boundaries or tracking gaps cannot be recovered from the completed-trip tables. The edge check reduces a specific opportunity bias but cannot correct all informative censoring. Recorded position coverage, completed-trip availability, and event counts must be considered alongside duration.", "",
              "Sleep fractions during trips reuse the existing one-second sleep classifier/context. Sleep is divided by known sleep-state seconds, with its coverage retained. Outside time may contain locomotion, resting, resource visits, or other behavior; a longer trip does not by itself identify a new biological task or establish foraging.", "",
              "A rise in means, medians, and long-trip frequency across the same ants supports lengthening of outside excursions. It does not establish a sharp switch into a distinct task. Ant-bootstrap intervals are descriptive within these two colonies; ants are not independent colony replicates. The data do not establish aging effects or behavior during the recording gaps.", "",
              "## Files", "", "Open index.html for the figure guide and individual_trip_explorer.html for interactive per-ant trip events. All source completed events, including their context-derived sleep values, are in events.parquet and events.csv. The run manifest records source fingerprints, clock weighting, and code hashes."]
    (Path(output) / "report.md").write_text("\n".join(lines) + "\n")


def run(analysis_folder, output, workers=4, bootstrap=2000):
    from analysis import long_timescale_trip_plots as plots
    source, output = Path(analysis_folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETE.json").unlink(missing_ok=True)
    manifest = json.loads((source / "run_manifest.json").read_text())
    events = pd.read_parquet(source / "completed_trips.parquet")
    bins = pd.read_parquet(source / "task_bins.parquet")
    mappings = [] if Path("/flash/ReiterU").exists() else ["/flash/ReiterU=/home/sam-reiter/flash", "/bucket/ReiterU=/home/sam-reiter/bucket/ReiterU", "/home/s/samuel-reiter=/home/sam-reiter/saionHome"]
    tables = build_tables(events, bins, manifest, lt.path_resolver(mappings), workers, bootstrap)
    for name, table in tables.items():
        table.to_parquet(output / (name + ".parquet"), index=False)
        table.to_csv(output / (name + ".csv"), index=False)
    assert len(tables["events"]) == len(events) and set(tables["events"].trip_id) == set(events.trip_id)
    print("TRIP_DIAGNOSTIC_TABLES_COMPLETE", flush=True)
    plots.save_figures(tables, manifest, output)
    write_report(tables, manifest, output)
    plots.write_explorer(tables, manifest, output)
    provenance = dict(source=str(source), created=datetime.now().isoformat(), bootstrap=bootstrap,
                      source_fingerprints=[lt.rs.fingerprint(source / name) for name in ("run_manifest.json", "completed_trips.parquet", "task_bins.parquet")],
                      context_sources=[s["all_ant_sources"] for s in manifest["sources"]],
                      code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(plots.__file__))},
                      figures=[p.name for p in sorted(output.glob("*.png"))],
                      clock_stratum_hours=2, equal_weight="clock strata within ant, then ants", all_completed_events_retained=len(events))
    (output / "run_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plots.write_index(output)
    result = dict(events=len(events), figures=len(provenance["figures"]), report=str(output / "index.html"))
    (output / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    print("TRIP_DIAGNOSTICS_COMPLETE", json.dumps(result), flush=True)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.workers < 1 or args.bootstrap < 100:
        parser.error("Require positive workers and at least 100 bootstrap samples")
    import matplotlib
    matplotlib.use("Agg")
    run(args.analysis_folder, args.output, args.workers, args.bootstrap)


if __name__ == "__main__":
    main()
