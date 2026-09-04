"""Optional trip-based phenotyping for putative roaming ants.

This module is deliberately separate from ``grid_occupancy_utils``. The
corresponding final block in ``grid_occupancy.py`` can be deleted (along with
this file) without affecting the main spatial or speed analyses.

A completed trip is a colony -> outside -> colony state sequence.  State is
estimated in one-second bins from bodypoint-0 positions and the current colony
rectangle in ``panorama_regions.csv``.  Short missing stretches are bridged,
but long gaps break a trajectory so that they cannot manufacture a completed
return. The retained summaries treat differences among roaming ants as
continuous instead of imposing a second clustering layer.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def colony_rectangles_from_regions(regions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return the single annotated colony rectangle for each side, in pixels."""
    chosen = regions[(regions["region_type"] == "colony") & (regions["shape"] == "rectangle")]
    rectangles: dict[str, dict[str, float]] = {}
    for side in ("left", "right"):
        side_rows = chosen[chosen["side"] == side]
        if len(side_rows) != 1:
            raise ValueError(f"Expected exactly one rectangular colony region for {side}, found {len(side_rows)}")
        row = side_rows.iloc[0]
        rectangles[side] = {
            "x_min_px": float(row["tracking_x_min_px"]),
            "x_max_px": float(row["tracking_x_max_px"]),
            "y_min_px": float(row["tracking_y_min_px"]),
            "y_max_px": float(row["tracking_y_max_px"]),
            "mm_per_pixel": float(row["mm_per_pixel"]),
            "region_name": str(row["name"]),
        }
    return rectangles


def _run_length_encoding(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(values) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(values) - 1]
    return starts.astype(np.int64), ends.astype(np.int64), values[starts]


def _bridge_short_state_gaps(state: np.ndarray, max_gap_bins: int) -> np.ndarray:
    """Nearest-neighbor fill missing state runs no longer than max_gap_bins."""
    filled = state.copy()
    starts, ends, values = _run_length_encoding(filled)
    for index, (start, end, value) in enumerate(zip(starts, ends, values)):
        if value != -1 or end - start + 1 > max_gap_bins or index == 0 or index == len(values) - 1:
            continue
        left = int(values[index - 1])
        right = int(values[index + 1])
        if left == -1 or right == -1:
            continue
        if left == right:
            filled[start : end + 1] = left
        else:
            midpoint = int(start + (end - start + 1) // 2)
            filled[start:midpoint] = left
            filled[midpoint : end + 1] = right
    return filled


def _remove_short_state_flicker(state: np.ndarray, min_run_bins: int) -> np.ndarray:
    """Remove brief border flicker only when equal known states surround it."""
    smoothed = state.copy()
    for _ in range(3):
        starts, ends, values = _run_length_encoding(smoothed)
        changed = False
        for index in range(1, len(values) - 1):
            length = int(ends[index] - starts[index] + 1)
            if (
                values[index] != -1
                and length < min_run_bins
                and values[index - 1] == values[index + 1]
                and values[index - 1] != -1
            ):
                smoothed[starts[index] : ends[index] + 1] = values[index - 1]
                changed = True
        if not changed:
            break
    return smoothed


def _distance_from_rectangle_px(x: np.ndarray, y: np.ndarray, rectangle: dict[str, float]) -> np.ndarray:
    dx = np.maximum.reduce(
        [
            np.full(len(x), float(rectangle["x_min_px"])) - x,
            x - float(rectangle["x_max_px"]),
            np.zeros(len(x)),
        ]
    )
    dy = np.maximum.reduce(
        [
            np.full(len(y), float(rectangle["y_min_px"])) - y,
            y - float(rectangle["y_max_px"]),
            np.zeros(len(y)),
        ]
    )
    return np.hypot(dx, dy)


def _scan_track_to_position_bins(task: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Stream one parquet and return completed trips plus outside positions."""
    import pyarrow.dataset as ds

    track_path = Path(task["track_path"])
    fps = float(task["fps"])
    position_bin_seconds = float(task["position_bin_seconds"])
    frames_per_bin = max(1, int(round(fps * position_bin_seconds)))
    n_bins = max(1, int(task["frame_max"]) // frames_per_bin + 1)
    rectangle = task["rectangle"]

    observed_count = np.zeros(n_bins, dtype=np.int32)
    inside_count = np.zeros(n_bins, dtype=np.int32)
    outside_count = np.zeros(n_bins, dtype=np.int32)
    outside_x_sum = np.zeros(n_bins, dtype=np.float64)
    outside_y_sum = np.zeros(n_bins, dtype=np.float64)

    dataset = ds.dataset(track_path, format="parquet")
    schema_names = set(dataset.schema.names)
    required = {"Frame", "TrackX", "TrackY"}
    missing = required.difference(schema_names)
    if missing:
        raise ValueError(f"{track_path.name} is missing columns {sorted(missing)}")
    position_filter = ds.field("Bodypoint") == int(task["bodypoint"]) if "Bodypoint" in schema_names else None
    scanner = dataset.scanner(
        columns=["Frame", "TrackX", "TrackY"],
        filter=position_filter,
        batch_size=262_144,
        use_threads=False,
    )
    for batch in scanner.to_batches():
        frame = batch.column(0).to_numpy(zero_copy_only=False)
        x = batch.column(1).to_numpy(zero_copy_only=False)
        y = batch.column(2).to_numpy(zero_copy_only=False)
        valid = np.isfinite(frame) & np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        frame = np.rint(frame[valid]).astype(np.int64, copy=False)
        x = x[valid].astype(np.float64, copy=False)
        y = y[valid].astype(np.float64, copy=False)
        time_bin = frame // frames_per_bin
        valid_bin = (time_bin >= 0) & (time_bin < n_bins)
        if not np.all(valid_bin):
            time_bin = time_bin[valid_bin]
            x = x[valid_bin]
            y = y[valid_bin]
        inside = (
            (x >= float(rectangle["x_min_px"]))
            & (x <= float(rectangle["x_max_px"]))
            & (y >= float(rectangle["y_min_px"]))
            & (y <= float(rectangle["y_max_px"]))
        )
        observed_count += np.bincount(time_bin, minlength=n_bins).astype(np.int32, copy=False)
        inside_count += np.bincount(time_bin, weights=inside, minlength=n_bins).astype(np.int32, copy=False)
        outside = ~inside
        if np.any(outside):
            outside_bin = time_bin[outside]
            outside_count += np.bincount(outside_bin, minlength=n_bins).astype(np.int32, copy=False)
            outside_x_sum += np.bincount(outside_bin, weights=x[outside], minlength=n_bins)
            outside_y_sum += np.bincount(outside_bin, weights=y[outside], minlength=n_bins)

    observed = observed_count > 0
    state = np.full(n_bins, -1, dtype=np.int8)
    # A mixed transition second is inside only when most observations are inside.
    state[observed] = (inside_count[observed] > observed_count[observed] / 2.0).astype(np.int8)
    max_gap_bins = max(0, int(round(float(task["max_state_gap_seconds"]) / position_bin_seconds)))
    min_run_bins = max(1, int(round(float(task["min_state_run_seconds"]) / position_bin_seconds)))
    state = _bridge_short_state_gaps(state, max_gap_bins)
    state = _remove_short_state_flicker(state, min_run_bins)

    outside_x = np.full(n_bins, np.nan, dtype=float)
    outside_y = np.full(n_bins, np.nan, dtype=float)
    has_outside = outside_count > 0
    outside_x[has_outside] = outside_x_sum[has_outside] / outside_count[has_outside]
    outside_y[has_outside] = outside_y_sum[has_outside] / outside_count[has_outside]

    min_trip_bins = max(1, int(np.ceil(float(task["min_trip_seconds"]) / position_bin_seconds)))
    min_anchor_bins = max(1, int(np.ceil(float(task["min_colony_anchor_seconds"]) / position_bin_seconds)))
    max_path_gap_bins = max(1, int(round(float(task["max_path_gap_seconds"]) / position_bin_seconds)))
    mm_per_pixel = float(rectangle["mm_per_pixel"])
    starts, ends, values = _run_length_encoding(state)
    trip_rows: list[dict[str, Any]] = []
    position_rows: list[pd.DataFrame] = []

    for run_index in range(1, len(values) - 1):
        start = int(starts[run_index])
        end = int(ends[run_index])
        duration_bins = end - start + 1
        if values[run_index] != 0 or duration_bins < min_trip_bins:
            continue
        previous_length = int(ends[run_index - 1] - starts[run_index - 1] + 1)
        next_length = int(ends[run_index + 1] - starts[run_index + 1] + 1)
        if values[run_index - 1] != 1 or values[run_index + 1] != 1:
            continue
        if previous_length < min_anchor_bins or next_length < min_anchor_bins:
            continue

        raw_coverage = float(np.mean(observed[start : end + 1]))
        valid_position = has_outside[start : end + 1]
        n_position_bins = int(valid_position.sum())
        if raw_coverage < float(task["min_trip_coverage"]) or n_position_bins < 2:
            continue
        local_bins = np.arange(start, end + 1, dtype=np.int64)[valid_position]
        x_px = outside_x[local_bins]
        y_px = outside_y[local_bins]
        x_mm = x_px * mm_per_pixel
        y_mm = y_px * mm_per_pixel
        distance_mm = _distance_from_rectangle_px(x_px, y_px, rectangle) * mm_per_pixel

        step_gap = np.diff(local_bins)
        step_distance = np.hypot(np.diff(x_mm), np.diff(y_mm))
        accepted_step = step_gap <= max_path_gap_bins
        observed_path_length_mm = float(step_distance[accepted_step].sum())
        farthest_index = int(np.nanargmax(distance_mm))
        max_excursion_mm = float(distance_mm[farthest_index])
        spatial_x_bin = np.floor(x_mm / float(task["trip_entropy_bin_mm"])).astype(int)
        spatial_y_bin = np.floor(y_mm / float(task["trip_entropy_bin_mm"])).astype(int)
        spatial_keys = spatial_x_bin.astype(str) + ":" + spatial_y_bin.astype(str)
        _, spatial_counts = np.unique(spatial_keys, return_counts=True)
        spatial_probability = spatial_counts / spatial_counts.sum()
        spatial_entropy = float(-(spatial_probability * np.log2(spatial_probability)).sum())

        trip_number = len(trip_rows) + 1
        trip_id = f"{task['side']}:{int(task['track_id']):04d}:{trip_number:04d}"
        exit_seconds = start * position_bin_seconds
        return_seconds = (end + 1) * position_bin_seconds
        duration_seconds = duration_bins * position_bin_seconds
        trip_rows.append(
            {
                "trip_id": trip_id,
                "trip_number": trip_number,
                "side": str(task["side"]),
                "track_id": int(task["track_id"]),
                "track_name": str(task["track_name"]),
                "cluster_id": str(task["cluster_id"]),
                "exit_frame": int(round(exit_seconds * fps)),
                "return_frame": int(round(return_seconds * fps)),
                "exit_elapsed_h": exit_seconds / 3600.0,
                "return_elapsed_h": return_seconds / 3600.0,
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_seconds / 60.0,
                "clock_hour_exit": ((float(task["start_clock_seconds"]) + exit_seconds) % 86400.0) / 3600.0,
                "n_observed_seconds": n_position_bins * position_bin_seconds,
                "observed_coverage": raw_coverage,
                "observed_path_length_mm": observed_path_length_mm,
                "max_excursion_mm": max_excursion_mm,
                "median_excursion_mm": float(np.nanmedian(distance_mm)),
                "turnaround_x_mm": float(x_mm[farthest_index]),
                "turnaround_y_mm": float(y_mm[farthest_index]),
                "mean_x_mm": float(np.nanmean(x_mm)),
                "mean_y_mm": float(np.nanmean(y_mm)),
                "span_x_mm": float(np.nanmax(x_mm) - np.nanmin(x_mm)),
                "span_y_mm": float(np.nanmax(y_mm) - np.nanmin(y_mm)),
                "spatial_entropy_bits": spatial_entropy,
                "path_efficiency": min(1.0, 2.0 * max_excursion_mm / observed_path_length_mm)
                if observed_path_length_mm > 0
                else np.nan,
            }
        )
        position_rows.append(
            pd.DataFrame(
                {
                    "trip_id": trip_id,
                    "side": str(task["side"]),
                    "track_id": int(task["track_id"]),
                    "track_name": str(task["track_name"]),
                    "cluster_id": str(task["cluster_id"]),
                    "elapsed_seconds": local_bins * position_bin_seconds,
                    "trip_elapsed_seconds": (local_bins - start) * position_bin_seconds,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "excursion_mm": distance_mm,
                }
            )
        )

    trip_table = pd.DataFrame(trip_rows)
    trip_positions = pd.concat(position_rows, ignore_index=True) if position_rows else pd.DataFrame()
    diagnostic = {
        "side": str(task["side"]),
        "track_id": int(task["track_id"]),
        "track_name": str(task["track_name"]),
        "cluster_id": str(task["cluster_id"]),
        "n_observed_position_bins": int(observed.sum()),
        "observed_position_hours": float(observed.sum() * position_bin_seconds / 3600.0),
        "n_completed_trips": len(trip_rows),
    }
    return trip_table, trip_positions, diagnostic


def load_or_extract_completed_trips(
    tracks: pd.DataFrame,
    regions: pd.DataFrame,
    per_track_root: Path,
    output_root: Path,
    *,
    fps: float,
    start_clock_seconds: float,
    bodypoint: int = 0,
    position_bin_seconds: float = 1.0,
    max_state_gap_seconds: float = 30.0,
    min_state_run_seconds: float = 5.0,
    min_colony_anchor_seconds: float = 5.0,
    min_trip_seconds: float = 10.0,
    min_trip_coverage: float = 0.20,
    max_path_gap_seconds: float = 3.0,
    trip_entropy_bin_mm: float = 10.0,
    max_workers: int = 4,
    recompute: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract and cache completed trips for the selected candidate ants."""
    if tracks.empty:
        raise ValueError("No candidate tracks supplied for trip extraction")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    trip_path = output_root / "completed_trips.parquet"
    position_path = output_root / "completed_trip_positions_1s.parquet"
    diagnostic_path = output_root / "trip_extraction_diagnostics.csv"
    metadata_path = output_root / "trip_extraction_metadata.json"
    rectangles = colony_rectangles_from_regions(regions)
    settings = {
        "fps": float(fps),
        "start_clock_seconds": float(start_clock_seconds),
        "bodypoint": int(bodypoint),
        "position_bin_seconds": float(position_bin_seconds),
        "max_state_gap_seconds": float(max_state_gap_seconds),
        "min_state_run_seconds": float(min_state_run_seconds),
        "min_colony_anchor_seconds": float(min_colony_anchor_seconds),
        "min_trip_seconds": float(min_trip_seconds),
        "min_trip_coverage": float(min_trip_coverage),
        "max_path_gap_seconds": float(max_path_gap_seconds),
        "trip_entropy_bin_mm": float(trip_entropy_bin_mm),
        "track_names": sorted(tracks["track_name"].astype(str).unique().tolist()),
        "colony_rectangles": rectangles,
    }
    cache_current = False
    if not recompute and all(path.is_file() for path in (trip_path, position_path, diagnostic_path, metadata_path)):
        try:
            cache_current = json.loads(metadata_path.read_text()) == settings
        except Exception:
            cache_current = False
    if cache_current:
        print(f"Loaded cached optional trip phenotyping data from {output_root}")
        return pd.read_parquet(trip_path), pd.read_parquet(position_path), pd.read_csv(diagnostic_path)

    required_columns = {"side", "track_id", "track_name", "cluster_id", "frame_max"}
    missing = required_columns.difference(tracks.columns)
    if missing:
        raise ValueError(f"Candidate track table is missing {sorted(missing)}")
    tasks = []
    for row in tracks.drop_duplicates("track_name").itertuples(index=False):
        track_path = Path(per_track_root) / str(row.track_name)
        if not track_path.is_file():
            raise FileNotFoundError(f"Missing raw track for trip extraction: {track_path}")
        task = {
            **settings,
            "track_path": str(track_path),
            "side": str(row.side),
            "track_id": int(row.track_id),
            "track_name": str(row.track_name),
            "cluster_id": str(row.cluster_id),
            "frame_max": int(row.frame_max),
            "rectangle": rectangles[str(row.side)],
        }
        task.pop("track_names", None)
        task.pop("colony_rectangles", None)
        tasks.append(task)

    trip_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {executor.submit(_scan_track_to_position_bins, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            trips, positions, diagnostic = future.result()
            if not trips.empty:
                trip_parts.append(trips)
            if not positions.empty:
                position_parts.append(positions)
            diagnostics.append(diagnostic)
            completed += 1
            if completed == 1 or completed % 8 == 0 or completed == len(tasks):
                print(
                    f"trips: processed {completed}/{len(tasks)} tracks; "
                    f"latest={task['track_name']}; completed trips so far={sum(len(part) for part in trip_parts):,}",
                    flush=True,
                )

    if not trip_parts:
        raise ValueError("No completed colony-outside-colony trips passed the configured filters")
    trip_table = pd.concat(trip_parts, ignore_index=True).sort_values(
        ["side", "track_id", "exit_frame"], kind="mergesort"
    ).reset_index(drop=True)
    trip_positions = pd.concat(position_parts, ignore_index=True).sort_values(
        ["side", "track_id", "elapsed_seconds"], kind="mergesort"
    ).reset_index(drop=True)
    diagnostic_table = pd.DataFrame(diagnostics).sort_values(["side", "track_id"]).reset_index(drop=True)
    trip_table.to_parquet(trip_path, index=False)
    trip_positions.to_parquet(position_path, index=False)
    diagnostic_table.to_csv(diagnostic_path, index=False)
    metadata_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Saved {len(trip_table):,} completed trips and {len(trip_positions):,} outside seconds to {output_root}")
    return trip_table, trip_positions, diagnostic_table


def _trip_summary(tracks: pd.DataFrame, trips: pd.DataFrame, positions: pd.DataFrame, fps: float) -> pd.DataFrame:
    base_columns = ["side", "track_id", "track_name", "cluster_id", "frame_max"]
    base = tracks[base_columns].drop_duplicates("track_name").set_index("track_name")
    grouped = trips.groupby("track_name")
    summary = grouped.agg(
        n_completed_trips=("trip_id", "size"),
        total_trip_minutes=("duration_minutes", "sum"),
        mean_trip_minutes=("duration_minutes", "mean"),
        median_trip_minutes=("duration_minutes", "median"),
        q90_trip_minutes=("duration_minutes", lambda value: value.quantile(0.90)),
        sd_trip_minutes=("duration_minutes", "std"),
        median_path_length_mm=("observed_path_length_mm", "median"),
        median_max_excursion_mm=("max_excursion_mm", "median"),
        q90_max_excursion_mm=("max_excursion_mm", lambda value: value.quantile(0.90)),
        max_excursion_mm=("max_excursion_mm", "max"),
        median_path_efficiency=("path_efficiency", "median"),
        median_spatial_entropy_bits=("spatial_entropy_bits", "median"),
    )
    summary = base.join(summary, how="left")
    summary["n_completed_trips"] = summary["n_completed_trips"].fillna(0).astype(int)
    zero_fill = ["total_trip_minutes"]
    summary[zero_fill] = summary[zero_fill].fillna(0.0)
    recording_days = (summary["frame_max"].astype(float) + 1.0) / float(fps) / 86400.0
    summary["completed_trips_per_day"] = summary["n_completed_trips"] / recording_days
    outside_seconds = grouped["n_observed_seconds"].sum().reindex(summary.index, fill_value=0).astype(float)
    summary["observed_trip_hours"] = outside_seconds / 3600.0
    return summary.reset_index()


def summarize_trip_candidates(
    candidate_tracks: pd.DataFrame,
    trips: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    fps: float,
) -> pd.DataFrame:
    """Summarize completed-trip investment for every candidate ant."""
    return _trip_summary(candidate_tracks, trips, positions, fps).sort_values(
        ["side", "track_id"], kind="mergesort"
    ).reset_index(drop=True)


def _bootstrap_interval(
    values: np.ndarray,
    *,
    statistic: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    sampled = values[rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))]
    if statistic == "median":
        estimates = np.median(sampled, axis=1)
    elif statistic == "mean":
        estimates = np.mean(sampled, axis=1)
    else:
        raise ValueError(f"Unsupported bootstrap statistic: {statistic}")
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)




def compute_trip_investment_confidence(
    trip_summary: pd.DataFrame,
    trips: pd.DataFrame,
    *,
    fps: float,
    n_bootstrap: int = 2_000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Return ant-level trip frequency and duration with two-dimensional 95% intervals.

    The frequency interval is an exact Poisson interval using the ant's full
    recording duration as exposure. The duration interval bootstraps that
    ant's observed trip durations and reports uncertainty around the median.
    """
    from scipy.stats import chi2

    if fps <= 0 or n_bootstrap < 1:
        raise ValueError("fps and n_bootstrap must be positive")
    rng = np.random.default_rng(int(random_state))
    rows: list[dict[str, Any]] = []
    for side, table in trip_summary.groupby("side", sort=True):
        for ant in table.drop_duplicates("track_name").itertuples(index=False):
            ant_trips = trips[trips["track_name"].astype(str) == str(ant.track_name)]
            durations = ant_trips["duration_minutes"].to_numpy(float)
            n_trips = len(durations)
            recording_days = (float(ant.frame_max) + 1.0) / float(fps) / 86400.0
            trips_per_day = n_trips / recording_days
            trips_per_day_low = (
                0.5 * chi2.ppf(0.025, 2 * n_trips) / recording_days if n_trips > 0 else 0.0
            )
            trips_per_day_high = 0.5 * chi2.ppf(0.975, 2 * (n_trips + 1)) / recording_days
            duration_low, duration_high = _bootstrap_interval(
                durations,
                statistic="median",
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            rows.append(
                {
                    "side": side,
                    "track_id": int(ant.track_id),
                    "track_name": str(ant.track_name),
                    "n_completed_trips": n_trips,
                    "recording_days": recording_days,
                    "completed_trips_per_day": trips_per_day,
                    "completed_trips_per_day_ci_low": trips_per_day_low,
                    "completed_trips_per_day_ci_high": trips_per_day_high,
                    "median_trip_minutes": float(np.median(durations)),
                    "median_trip_minutes_ci_low": duration_low,
                    "median_trip_minutes_ci_high": duration_high,
                }
            )
    return pd.DataFrame(rows).sort_values(["side", "track_id"]).reset_index(drop=True)


def compute_ant_time_of_day_percent(
    trip_summary: pd.DataFrame,
    completed_trip_positions: pd.DataFrame,
    resource_presence_frames: pd.DataFrame,
    *,
    fps: float,
    start_clock_seconds: float,
    bin_minutes: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-ant clock-time percentages for trips and resource presence.

    Each ant is normalized separately within each output: its clock bins sum to
    100%. Trip values count one-second outside positions in completed trips;
    resource values count unique frames in any annotated food or water region.
    """
    if fps <= 0 or bin_minutes <= 0 or not np.isclose(1440.0 / bin_minutes, round(1440.0 / bin_minutes)):
        raise ValueError("fps and bin_minutes must be positive, and bin_minutes must divide 24 hours")
    n_bins = int(round(1440.0 / float(bin_minutes)))
    bin_seconds = float(bin_minutes) * 60.0
    eligible = trip_summary[["side", "track_id", "track_name"]].drop_duplicates(
        "track_name"
    ).copy()
    eligible["track_name"] = eligible["track_name"].astype(str)

    trip_positions = completed_trip_positions[
        completed_trip_positions["track_name"].astype(str).isin(eligible["track_name"])
    ][["track_name", "elapsed_seconds"]].copy()
    trip_clock_seconds = (
        float(start_clock_seconds) + trip_positions["elapsed_seconds"].to_numpy(float)
    ) % 86400.0
    trip_positions["tod_bin"] = np.floor(trip_clock_seconds / bin_seconds).astype(int)
    trip_counts = trip_positions.groupby(["track_name", "tod_bin"]).size()

    resource = resource_presence_frames[
        resource_presence_frames["track_name"].astype(str).isin(eligible["track_name"])
    ][["track_name", "frame"]].drop_duplicates(["track_name", "frame"])
    resource_clock_seconds = (
        float(start_clock_seconds) + resource["frame"].to_numpy(float) / float(fps)
    ) % 86400.0
    resource = resource.copy()
    resource["tod_bin"] = np.floor(resource_clock_seconds / bin_seconds).astype(int)
    resource_counts = resource.groupby(["track_name", "tod_bin"]).size()

    def normalized_long(counts: pd.Series, source: str) -> pd.DataFrame:
        matrix = counts.unstack(fill_value=0).reindex(
            index=eligible["track_name"],
            columns=np.arange(n_bins),
            fill_value=0,
        )
        percentages = 100.0 * matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
        long = percentages.rename_axis(index="track_name", columns="tod_bin").stack(
            future_stack=True
        ).rename("percent_of_ant_time").reset_index()
        long["clock_hour"] = (long["tod_bin"].astype(float) + 0.5) * float(bin_minutes) / 60.0
        long["source"] = source
        return long.merge(eligible, on="track_name", how="left", validate="many_to_one")[
            ["source", "side", "track_id", "track_name", "tod_bin", "clock_hour", "percent_of_ant_time"]
        ]

    return normalized_long(trip_counts, "completed_trip"), normalized_long(resource_counts, "resource")


def plot_trip_investment_confidence(investment: pd.DataFrame) -> None:
    """Plot trips/day versus median duration with per-ant horizontal and vertical CIs."""
    sides = [side for side in ("left", "right") if side in set(investment["side"].astype(str))]
    fig, axes = plt.subplots(1, len(sides), figsize=(7.2 * len(sides), 6.4), squeeze=False)
    for column, side in enumerate(sides):
        ax = axes[0, column]
        ants = investment[investment["side"].astype(str) == side].sort_values("track_id")
        x = ants["completed_trips_per_day"].to_numpy(float)
        y = ants["median_trip_minutes"].to_numpy(float)
        xerr = np.vstack(
            [
                x - ants["completed_trips_per_day_ci_low"].to_numpy(float),
                ants["completed_trips_per_day_ci_high"].to_numpy(float) - x,
            ]
        )
        yerr = np.vstack(
            [
                y - ants["median_trip_minutes_ci_low"].to_numpy(float),
                ants["median_trip_minutes_ci_high"].to_numpy(float) - y,
            ]
        )
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            ms=5,
            color="tab:blue" if side == "left" else "tab:orange",
            ecolor="0.55",
            elinewidth=0.8,
            capsize=1.8,
            alpha=0.78,
        )
        for ant in ants.itertuples(index=False):
            ax.annotate(
                str(int(ant.track_id)),
                (ant.completed_trips_per_day, ant.median_trip_minutes),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=6,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Completed trips / recording day")
        ax.set_ylabel("Median time per trip (min)")
        ax.set_title(f"{side} colony ({len(ants)} ants)")
        ax.grid(True, which="both", alpha=0.2)
    fig.suptitle(
        "Trip frequency versus duration; horizontal Poisson and vertical bootstrap 95% confidence intervals"
    )
    fig.tight_layout()
    plt.show()


def plot_ant_time_of_day_heatmap(
    time_table: pd.DataFrame,
    *,
    source: str,
    light_off_hour: float,
    light_on_hour: float,
) -> None:
    """Plot left/right ant-by-clock heatmaps with every row normalized to 100%."""
    chosen = time_table[time_table["source"] == source].copy()
    sides = [side for side in ("left", "right") if side in set(chosen["side"].astype(str))]
    matrices: dict[str, pd.DataFrame] = {}
    for side in sides:
        side_table = chosen[chosen["side"].astype(str) == side]
        matrix = side_table.pivot(index="track_id", columns="clock_hour", values="percent_of_ant_time")
        matrices[side] = matrix.sort_index()
    all_values = np.concatenate([matrix.to_numpy(float).ravel() for matrix in matrices.values()])
    color_high = max(1.0, float(np.nanpercentile(all_values, 99.0)))

    fig, axes = plt.subplots(1, len(sides), figsize=(7.4 * len(sides), 8.2), squeeze=False)
    image = None
    for column, side in enumerate(sides):
        ax = axes[0, column]
        matrix = matrices[side]
        image = ax.imshow(
            matrix.to_numpy(float),
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=0,
            vmax=color_high,
            extent=[0, 24, len(matrix), 0],
        )
        ax.axvline(float(light_on_hour), color="cyan", ls="--", lw=1.0)
        ax.axvline(float(light_off_hour), color="cyan", ls="--", lw=1.0)
        ax.set_xticks(np.arange(0, 25, 3))
        ax.set_xlim(0, 24)
        ax.set_xlabel("Clock time (hour)")
        ax.set_yticks(np.arange(len(matrix)) + 0.5, matrix.index.astype(int), fontsize=6)
        ax.set_ylabel("TrackID")
        ax.set_title(f"{side} colony ({len(matrix)} ants)")
    label = "outside time in completed trips" if source == "completed_trip" else "resource-detected time"
    fig.subplots_adjust(left=0.08, right=0.88, top=0.86, bottom=0.08, wspace=0.24)
    if image is not None:
        colorbar_ax = fig.add_axes([0.91, 0.13, 0.018, 0.68])
        fig.colorbar(image, cax=colorbar_ax, label=f"Percent of ant's {label} per clock bin")
    fig.suptitle(
        f"Each ant's {label} over time of day; every row sums to 100%\n"
        f"cyan lines: lights on {light_on_hour:g}, lights off {light_off_hour:g}"
    )
    plt.show()
