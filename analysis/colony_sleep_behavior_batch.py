#!/usr/bin/env python3
"""
Batch colony sleep/behavior analysis for stitched per-track parquet files.

The interactive notebook-style script in this directory is useful for tuning
parameters. This file exposes the same expensive per-track work as CLI stages
that can be submitted as one Slurm worker job per ant.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd


FPS = 24.0
MM_PER_PX = 0.016
MIN_TRACK_PRESENT_FRAC = 0.40
POSITION_COLUMNS = ("TrackX", "TrackY")
POSITION_COLUMN_FALLBACKS = (
    ("TrackX", "TrackY"),
    ("ArucoX", "ArucoY"),
    ("X", "Y"),
    ("SleapAnchorX", "SleapAnchorY"),
)
BODYPOINT_FOR_XY = 0
MAX_INTERP_GAP_FRAMES = 5
SMOOTH_SIGMA_FRAMES = 2.0
MAX_REASONABLE_SPEED_MM_S: float | None = 5.0
STATIONARY_THRESHOLD_MM_S: float | None = 0.1
MIN_SLEEP_STATIONARY_SECONDS = 10.0
COLONY_BOXES_MM: list[tuple[float, float, float, float]] | None = [
    (-86, -32, -63, -8),
    (93, 149, -63, -8),
]
N_COLONY_BOXES = 2
COLONY_DENSITY_BINS = 220
COLONY_DENSITY_SMOOTH_SIGMA_BINS = 2.0
COLONY_DENSITY_HIGH_QUANTILE = 0.86
COLONY_BOX_PAD_MM = 5.0
COLONY_BOX_CENTRAL_FRACTION = 0.90
N_BEHAVIOR_CLUSTERS = 2
WORKER_INSIDE_COLONY_FRAC_THRESHOLD: float | None = 0.95
RHYTHM_BIN_SECONDS = 10 * 60
SPEED_VECTOR_SAMPLE_PER_TRACK = 100_000
POSITION_SAMPLE_PER_TRACK = 20_000


@dataclass(frozen=True)
class TrackFileSummary:
    path: Path
    track_id: int
    n_observed_frames: int
    min_frame: int
    max_frame: int
    dataset_num_frames: int | None
    present_frac: float | None


@dataclass(frozen=True)
class WorkItem:
    task_index: int
    track_id: int
    track_key: str
    path: Path


SPEED_CACHE_DTYPES = {
    "Frame": "int64",
    "TrackID": "int64",
    "X_px": "float64",
    "Y_px": "float64",
    "n_rows": "int64",
    "Observed": "bool",
    "Interpolated": "bool",
    "X_px_smooth": "float64",
    "Y_px_smooth": "float64",
    "X_mm": "float64",
    "Y_mm": "float64",
    "TimeS": "float64",
    "TimeRelativeS": "float64",
    "DistanceMm": "float64",
    "SpeedMmPerSec": "float64",
    "ValidPosition": "bool",
    "ValidSpeed": "bool",
    "TrackKey": "object",
    "TrackPath": "object",
}


def empty_speed_cache_table() -> pd.DataFrame:
    return pd.DataFrame(
        {col: pd.Series(dtype=dtype) for col, dtype in SPEED_CACHE_DTYPES.items()}
    )


def ensure_cache_dirs(cache_dir: Path) -> None:
    for subdir in (
        cache_dir,
        cache_dir / "speed_tracks",
        cache_dir / "labeled_tracks",
        cache_dir / "outside_labeled_tracks",
        cache_dir / "speed_samples",
        cache_dir / "per_track_sleep_summary",
        cache_dir / "per_track_behavior_summary",
        cache_dir / "per_track_rhythm_bins",
        cache_dir / "vectors",
        cache_dir / "tables",
    ):
        subdir.mkdir(parents=True, exist_ok=True)


def use_cache(path: Path, *, force: bool = False) -> bool:
    return path.exists() and not force


def track_cache_stem(path: Path | str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).stem)


def speed_cache_path(cache_dir: Path, track_path: Path | str) -> Path:
    return cache_dir / "speed_tracks" / f"{track_cache_stem(track_path)}_speed.parquet"


def labeled_cache_path(cache_dir: Path, track_path: Path | str) -> Path:
    return cache_dir / "labeled_tracks" / f"{track_cache_stem(track_path)}_labeled.parquet"


def outside_labeled_cache_path(cache_dir: Path, track_path: Path | str) -> Path:
    return cache_dir / "outside_labeled_tracks" / f"{track_cache_stem(track_path)}_outside_labeled.parquet"


def speed_sample_path(cache_dir: Path, track_path: Path | str) -> Path:
    return cache_dir / "speed_samples" / f"{track_cache_stem(track_path)}_speed_sample.npy"


def sleep_summary_track_path(cache_dir: Path, track_path: Path | str) -> Path:
    return cache_dir / "per_track_sleep_summary" / f"{track_cache_stem(track_path)}_sleep_summary.parquet"


def behavior_summary_track_path(cache_dir: Path, track_path: Path | str) -> Path:
    return cache_dir / "per_track_behavior_summary" / f"{track_cache_stem(track_path)}_behavior_summary.parquet"


def rhythm_bins_track_path(cache_dir: Path, track_path: Path | str) -> Path:
    return cache_dir / "per_track_rhythm_bins" / f"{track_cache_stem(track_path)}_rhythm_bins.parquet"


def table_cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / "tables" / name


def vector_cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / "vectors" / name


def default_worklist_path(cache_dir: Path) -> Path:
    return table_cache_path(cache_dir, "good_track_worklist.tsv")


def track_presence_matches_input_dir(track_presence: pd.DataFrame, per_track_dir: Path) -> bool:
    if "path" not in track_presence.columns:
        return False
    expected_parent = str(per_track_dir.expanduser())
    parents = {
        str(Path(path).expanduser().parent)
        for path in track_presence["path"].dropna().astype(str)
    }
    return parents == {expected_parent}


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv" or path.suffix == ".tsv":
        sep = "\t" if path.suffix == ".tsv" else ","
        df.to_csv(path, index=False, sep=sep)
    else:
        df.to_parquet(path, index=False)


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix == ".csv" or path.suffix == ".tsv":
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        return df if columns is None else df[columns]
    return pd.read_parquet(path, columns=columns)


def sample_series(values: pd.Series, max_n: int, *, random_state: int = 0) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) > max_n:
        v = v.sample(max_n, random_state=random_state)
    return v.to_numpy(float)


def speed_histogram_table(speed_values: Iterable[float], *, bins: int = 12000) -> pd.DataFrame:
    speed = pd.Series(speed_values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if speed.empty:
        return pd.DataFrame(columns=["bin_left", "bin_right", "bin_center", "count"])
    clipped = speed.clip(upper=speed.quantile(0.995))
    counts, edges = np.histogram(clipped.to_numpy(float), bins=int(bins))
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": 0.5 * (edges[:-1] + edges[1:]),
            "count": counts,
        }
    )


def track_id_from_path(path: Path) -> int | None:
    match = re.search(r"TrackID_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def list_track_files(per_track_dir: Path, side_filter: str | None = None) -> list[Path]:
    paths = sorted(per_track_dir.glob("TrackID_*.parquet"))
    if side_filter:
        side = side_filter.lower()
        paths = [p for p in paths if f"_{side}" in p.stem.lower() or p.stem.lower().endswith(side)]
    return paths


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema.names
    except Exception:
        return list(pd.read_parquet(path).columns)


def parquet_num_frames(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(path).metadata.metadata or {}
        raw = metadata.get(b"num_frames")
        return int(raw.decode("utf-8")) if raw is not None else None
    except Exception:
        return None


def position_column_candidates(position_columns: tuple[str, str]) -> list[tuple[str, str]]:
    requested = (str(position_columns[0]), str(position_columns[1]))
    pairs = [requested]
    for pair in POSITION_COLUMN_FALLBACKS:
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def position_pair_for_columns(
    columns: Iterable[str],
    position_columns: tuple[str, str],
) -> tuple[str, str] | None:
    cols = set(columns)
    for x_col, y_col in position_column_candidates(position_columns):
        if {x_col, y_col}.issubset(cols):
            return x_col, y_col
    return None


def position_column_diagnostics(
    tracks: pd.DataFrame,
    *,
    position_columns: tuple[str, str],
) -> pd.DataFrame:
    rows = []
    for row in tracks.itertuples(index=False):
        path = Path(row.path)
        try:
            cols = set(parquet_columns(path))
            pair = position_pair_for_columns(cols, position_columns)
            error = ""
        except Exception as exc:
            cols = set()
            pair = None
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "TrackID": int(row.TrackID),
                "TrackKey": str(row.TrackKey),
                "path": str(path),
                "usable_position_columns": pair is not None,
                "position_x_col": "" if pair is None else pair[0],
                "position_y_col": "" if pair is None else pair[1],
                "available_columns": ",".join(sorted(cols)),
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def summarize_track_file(path: Path) -> TrackFileSummary:
    cols = parquet_columns(path)
    if "Frame" not in cols:
        raise ValueError(f"{path} has no Frame column")
    frame = pd.to_numeric(pd.read_parquet(path, columns=["Frame"])["Frame"], errors="coerce").dropna()
    tid = track_id_from_path(path)
    if frame.empty:
        return TrackFileSummary(path, -1 if tid is None else tid, 0, -1, -1, parquet_num_frames(path), None)
    if tid is None and "TrackID" in cols:
        tid_series = pd.read_parquet(path, columns=["TrackID"])["TrackID"]
        tid_numeric = pd.to_numeric(tid_series, errors="coerce").dropna()
        tid = int(tid_numeric.iloc[0]) if not tid_numeric.empty else -1
    unique_frames = frame.astype(np.int64).nunique()
    return TrackFileSummary(
        path=path,
        track_id=-1 if tid is None else int(tid),
        n_observed_frames=int(unique_frames),
        min_frame=int(frame.min()),
        max_frame=int(frame.max()),
        dataset_num_frames=parquet_num_frames(path),
        present_frac=None,
    )


def infer_dataset_frame_count(summaries: list[TrackFileSummary]) -> int:
    metadata_counts = [s.dataset_num_frames for s in summaries if s.dataset_num_frames is not None]
    if metadata_counts:
        return int(max(metadata_counts))
    if not summaries:
        raise ValueError("No track summaries available")
    return int(max(s.max_frame for s in summaries if s.max_frame >= 0) + 1)


def add_presence_fraction(
    summaries: list[TrackFileSummary],
    dataset_num_frames: int,
) -> pd.DataFrame:
    rows = []
    for s in summaries:
        frac = s.n_observed_frames / dataset_num_frames if dataset_num_frames > 0 else np.nan
        rows.append(
            {
                "TrackID": s.track_id,
                "TrackKey": track_cache_stem(s.path),
                "path": str(s.path),
                "n_observed_frames": s.n_observed_frames,
                "min_frame": s.min_frame,
                "max_frame": s.max_frame,
                "dataset_num_frames": dataset_num_frames,
                "present_frac": frac,
            }
        )
    return pd.DataFrame(rows).sort_values(["TrackID", "TrackKey"]).reset_index(drop=True)


def load_track_position_table(
    path: Path,
    *,
    position_columns: tuple[str, str] = POSITION_COLUMNS,
    bodypoint: int = BODYPOINT_FOR_XY,
) -> pd.DataFrame:
    cols = set(parquet_columns(path))
    if "Frame" not in cols:
        raise ValueError(f"{path.name} missing columns: ['Frame']")

    tried_pairs = position_column_candidates(position_columns)
    available_pairs = [(x, y) for x, y in tried_pairs if {x, y}.issubset(cols)]
    if not available_pairs:
        tried = ", ".join(f"{x}/{y}" for x, y in tried_pairs)
        raise ValueError(f"{path.name} has no usable position columns; tried {tried}")

    fallback_tid = track_id_from_path(path)
    for x_col, y_col in available_pairs:
        read_cols = ["Frame", x_col, y_col]
        if "Bodypoint" in cols:
            read_cols.append("Bodypoint")
        df: pd.DataFrame
        if "Bodypoint" in cols:
            try:
                import pyarrow.compute as pc
                import pyarrow.dataset as ds

                table = ds.dataset(path, format="parquet").to_table(
                    columns=read_cols,
                    filter=pc.field("Bodypoint") == int(bodypoint),
                )
                df = table.to_pandas()
            except Exception:
                df = pd.read_parquet(path, columns=read_cols)
                df = df[pd.to_numeric(df["Bodypoint"], errors="coerce") == int(bodypoint)]
        else:
            df = pd.read_parquet(path, columns=read_cols)
        df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

        df = df.dropna(subset=["Frame", x_col, y_col])
        if df.empty:
            continue

        if (x_col, y_col) != tuple(position_columns):
            print(
                f"{path.name}: using fallback position columns {x_col}/{y_col} "
                f"instead of {position_columns[0]}/{position_columns[1]}",
                flush=True,
            )

        df["Frame"] = df["Frame"].round().astype(np.int64)
        df["X_px"] = df[x_col].astype(float)
        df["Y_px"] = df[y_col].astype(float)
        grouped = df.groupby("Frame", sort=True, as_index=False)
        out = grouped.agg(X_px=("X_px", "mean"), Y_px=("Y_px", "mean"), n_rows=("X_px", "size"))
        out["TrackID"] = -1 if fallback_tid is None else fallback_tid
        return out.sort_values("Frame", kind="mergesort").reset_index(drop=True)

    return pd.DataFrame(columns=["Frame", "TrackID", "X_px", "Y_px"])


def interpolate_short_gaps(
    frame_xy: pd.DataFrame,
    *,
    max_gap_frames: int = 5,
) -> pd.DataFrame:
    if frame_xy.empty:
        return frame_xy.copy()
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be >= 0")

    frame_xy = frame_xy.sort_values("Frame", kind="mergesort").drop_duplicates("Frame").copy()
    frame_min = int(frame_xy["Frame"].min())
    frame_max = int(frame_xy["Frame"].max())
    full = pd.DataFrame({"Frame": np.arange(frame_min, frame_max + 1, dtype=np.int64)})
    out = full.merge(frame_xy, on="Frame", how="left")
    out["Observed"] = out["X_px"].notna() & out["Y_px"].notna()
    out["Interpolated"] = False

    obs_idx = np.flatnonzero(out["Observed"].to_numpy())
    if len(obs_idx) < 2 or max_gap_frames == 0:
        return out

    for left, right in zip(obs_idx[:-1], obs_idx[1:]):
        gap = int(right - left - 1)
        if gap <= 0 or gap > max_gap_frames:
            continue
        x0, y0 = float(out.loc[left, "X_px"]), float(out.loc[left, "Y_px"])
        x1, y1 = float(out.loc[right, "X_px"]), float(out.loc[right, "Y_px"])
        frac = np.arange(1, gap + 1, dtype=float) / float(gap + 1)
        fill_idx = np.arange(left + 1, right)
        out.loc[fill_idx, "X_px"] = x0 + frac * (x1 - x0)
        out.loc[fill_idx, "Y_px"] = y0 + frac * (y1 - y0)
        out.loc[fill_idx, "Interpolated"] = True
    return out


def smooth_by_valid_segments(values: pd.Series, valid: pd.Series, *, sigma_frames: float) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid_arr = valid.to_numpy(bool)
    if not valid_arr.any():
        return out

    try:
        from scipy.ndimage import gaussian_filter1d
    except Exception:
        gaussian_filter1d = None

    idx = np.flatnonzero(valid_arr)
    split_points = np.where(np.diff(idx) > 1)[0] + 1
    for segment in np.split(idx, split_points):
        if len(segment) == 0:
            continue
        segment_values = values.iloc[segment].to_numpy(float)
        if gaussian_filter1d is not None and sigma_frames > 0 and len(segment) >= 3:
            smoothed = gaussian_filter1d(segment_values, sigma=float(sigma_frames), mode="nearest")
        elif sigma_frames > 0 and len(segment) >= 3:
            window = int(max(3, round(6 * sigma_frames + 1)))
            if window % 2 == 0:
                window += 1
            smoothed = (
                pd.Series(segment_values)
                .rolling(window, center=True, min_periods=1)
                .mean()
                .to_numpy(float)
            )
        else:
            smoothed = segment_values
        out.iloc[segment] = smoothed
    return out


def compute_gap_aware_speed(
    frame_xy: pd.DataFrame,
    *,
    fps: float = FPS,
    mm_per_px: float = MM_PER_PX,
    max_interp_gap_frames: int = MAX_INTERP_GAP_FRAMES,
    smooth_sigma_frames: float = SMOOTH_SIGMA_FRAMES,
    max_reasonable_speed_mm_s: float | None = MAX_REASONABLE_SPEED_MM_S,
) -> pd.DataFrame:
    out = interpolate_short_gaps(frame_xy, max_gap_frames=max_interp_gap_frames)
    if out.empty:
        return out

    out["TrackID"] = int(pd.to_numeric(frame_xy["TrackID"], errors="coerce").dropna().iloc[0])
    valid_pos = out["X_px"].notna() & out["Y_px"].notna()
    out["X_px_smooth"] = smooth_by_valid_segments(out["X_px"], valid_pos, sigma_frames=smooth_sigma_frames)
    out["Y_px_smooth"] = smooth_by_valid_segments(out["Y_px"], valid_pos, sigma_frames=smooth_sigma_frames)
    out["X_mm"] = out["X_px_smooth"] * float(mm_per_px)
    out["Y_mm"] = out["Y_px_smooth"] * float(mm_per_px)
    out["TimeS"] = out["Frame"].astype(float) / float(fps)
    out["TimeRelativeS"] = (out["Frame"].astype(float) - float(out["Frame"].min())) / float(fps)

    step_valid = valid_pos & valid_pos.shift(1, fill_value=False)
    dx = out["X_px_smooth"].diff()
    dy = out["Y_px_smooth"].diff()
    distance_px = np.sqrt(dx * dx + dy * dy)
    out["DistanceMm"] = (distance_px * float(mm_per_px)).where(step_valid)
    out["SpeedMmPerSec"] = (out["DistanceMm"] * float(fps)).where(step_valid)

    if max_reasonable_speed_mm_s is not None:
        bad = out["SpeedMmPerSec"] > float(max_reasonable_speed_mm_s)
        out.loc[bad, ["DistanceMm", "SpeedMmPerSec"]] = np.nan

    out["ValidPosition"] = valid_pos
    out["ValidSpeed"] = out["SpeedMmPerSec"].notna()
    return out


def otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float(np.nanmedian(values)) if len(values) else np.nan
    hist, edges = np.histogram(values, bins=128)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return float(centers[:-1][int(np.nanargmax(variance12))])


def estimate_stationary_threshold(speed_mm_s: Iterable[float]) -> float:
    speed = np.asarray(list(speed_mm_s), dtype=float)
    speed = speed[np.isfinite(speed) & (speed >= 0)]
    if len(speed) < 50:
        return float(np.nanpercentile(speed, 35)) if len(speed) else np.nan

    trim_hi = np.nanpercentile(speed, 99.5)
    speed = speed[speed <= trim_hi]
    log_speed = np.log1p(speed).reshape(-1, 1)

    try:
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(n_components=2, random_state=0, covariance_type="full")
        gm.fit(log_speed)
        means = gm.means_.ravel()
        order = np.argsort(means)
        grid = np.linspace(means[order[0]], means[order[1]], 1024).reshape(-1, 1)
        posterior = gm.predict_proba(grid)[:, order]
        crossing = int(np.argmin(np.abs(posterior[:, 0] - posterior[:, 1])))
        return float(np.expm1(grid[crossing, 0]))
    except Exception:
        return float(np.expm1(otsu_threshold(np.log1p(speed))))


def label_sleep_from_speed(
    speed_df: pd.DataFrame,
    *,
    stationary_threshold_mm_s: float,
    min_sleep_stationary_seconds: float = MIN_SLEEP_STATIONARY_SECONDS,
    fps: float = FPS,
) -> pd.DataFrame:
    out = speed_df.copy()
    valid = out["SpeedMmPerSec"].notna()
    stationary = valid & (out["SpeedMmPerSec"] <= float(stationary_threshold_mm_s))
    out["Stationary"] = stationary

    state = stationary.to_numpy(bool)
    valid_arr = valid.to_numpy(bool)
    sleep = np.zeros(len(out), dtype=bool)
    min_frames = int(np.ceil(float(min_sleep_stationary_seconds) * float(fps)))

    start = 0
    while start < len(out):
        if not valid_arr[start] or not state[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(out) and valid_arr[stop] and state[stop]:
            stop += 1
        if stop - start >= min_frames:
            sleep[start:stop] = True
        start = stop

    out["Sleep"] = sleep
    out["Wake"] = valid & ~out["Sleep"]
    return out


def infer_colony_box(
    positions: pd.DataFrame,
    *,
    central_fraction: float = COLONY_BOX_CENTRAL_FRACTION,
) -> tuple[float, float, float, float]:
    if not 0 < central_fraction < 1:
        raise ValueError("central_fraction must be in (0, 1)")
    q_low = (1.0 - central_fraction) / 2.0
    q_high = 1.0 - q_low
    x = pd.to_numeric(positions["X_mm"], errors="coerce")
    y = pd.to_numeric(positions["Y_mm"], errors="coerce")
    return (
        float(x.quantile(q_low)),
        float(x.quantile(q_high)),
        float(y.quantile(q_low)),
        float(y.quantile(q_high)),
    )


def _weighted_kmeans_2d(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    n_clusters: int,
    n_iter: int = 80,
) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=int)
    n_clusters = min(int(n_clusters), len(points))
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)

    centers = [points[int(np.argmax(weights))]]
    while len(centers) < n_clusters:
        dist2 = np.min(
            np.stack([np.sum((points - center) ** 2, axis=1) for center in centers], axis=1),
            axis=1,
        )
        centers.append(points[int(np.argmax(dist2 * weights))])
    centers_arr = np.asarray(centers, dtype=float)

    labels = np.zeros(len(points), dtype=int)
    for _ in range(n_iter):
        dist2 = np.stack([np.sum((points - center) ** 2, axis=1) for center in centers_arr], axis=1)
        new_labels = np.argmin(dist2, axis=1)
        new_centers = centers_arr.copy()
        for k in range(n_clusters):
            mask = new_labels == k
            if mask.any():
                new_centers[k] = np.average(points[mask], axis=0, weights=weights[mask])
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers_arr):
            break
        labels = new_labels
        centers_arr = new_centers
    return labels


def infer_colony_boxes_by_density(
    positions: pd.DataFrame,
    *,
    n_boxes: int = N_COLONY_BOXES,
    bins: int = COLONY_DENSITY_BINS,
    smooth_sigma_bins: float = COLONY_DENSITY_SMOOTH_SIGMA_BINS,
    high_quantile: float = COLONY_DENSITY_HIGH_QUANTILE,
    pad_mm: float = COLONY_BOX_PAD_MM,
    fallback_central_fraction: float = COLONY_BOX_CENTRAL_FRACTION,
) -> tuple[list[tuple[float, float, float, float]], pd.DataFrame]:
    pos = positions[["X_mm", "Y_mm"]].apply(pd.to_numeric, errors="coerce").dropna()
    if pos.empty:
        raise ValueError("No valid positions for colony box inference")

    x = pos["X_mm"].to_numpy(float)
    y = pos["Y_mm"].to_numpy(float)
    hist, xedges, yedges = np.histogram2d(x, y, bins=int(bins))

    density = hist.astype(float)
    if smooth_sigma_bins > 0:
        try:
            from scipy.ndimage import gaussian_filter

            density = gaussian_filter(density, sigma=float(smooth_sigma_bins))
        except Exception:
            pass

    nonzero = density[density > 0]
    if nonzero.size == 0:
        fallback = infer_colony_box(pos, central_fraction=fallback_central_fraction)
        return [fallback], pd.DataFrame(
            [{"ColonyBoxID": 0, "xmin_mm": fallback[0], "xmax_mm": fallback[1], "ymin_mm": fallback[2], "ymax_mm": fallback[3], "source": "fallback_quantile"}]
        )

    threshold = float(np.quantile(nonzero, float(high_quantile)))
    high_mask = density >= threshold
    if int(high_mask.sum()) < max(2, int(n_boxes)):
        fallback = infer_colony_box(pos, central_fraction=fallback_central_fraction)
        return [fallback], pd.DataFrame(
            [{"ColonyBoxID": 0, "xmin_mm": fallback[0], "xmax_mm": fallback[1], "ymin_mm": fallback[2], "ymax_mm": fallback[3], "source": "fallback_quantile"}]
        )

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    ix, iy = np.where(high_mask)
    points = np.column_stack([xcenters[ix], ycenters[iy]])
    weights = density[ix, iy]

    labels = _weighted_kmeans_2d(points, weights, n_clusters=int(n_boxes))
    boxes: list[tuple[float, float, float, float]] = []
    rows = []
    dx = float(np.median(np.diff(xedges))) if len(xedges) > 1 else 0.0
    dy = float(np.median(np.diff(yedges))) if len(yedges) > 1 else 0.0
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        if not mask.any():
            continue
        px = points[mask, 0]
        py = points[mask, 1]
        w = weights[mask]
        xmin = float(px.min() - 0.5 * dx - pad_mm)
        xmax = float(px.max() + 0.5 * dx + pad_mm)
        ymin = float(py.min() - 0.5 * dy - pad_mm)
        ymax = float(py.max() + 0.5 * dy + pad_mm)
        boxes.append((xmin, xmax, ymin, ymax))
        rows.append(
            {
                "raw_cluster": int(label),
                "density_mass": float(w.sum()),
                "high_density_bins": int(mask.sum()),
                "center_x_mm": float(np.average(px, weights=w)),
                "center_y_mm": float(np.average(py, weights=w)),
                "xmin_mm": xmin,
                "xmax_mm": xmax,
                "ymin_mm": ymin,
                "ymax_mm": ymax,
            }
        )

    if len(boxes) < int(n_boxes):
        fallback = infer_colony_box(pos, central_fraction=fallback_central_fraction)
        return [fallback], pd.DataFrame(
            [{"ColonyBoxID": 0, "xmin_mm": fallback[0], "xmax_mm": fallback[1], "ymin_mm": fallback[2], "ymax_mm": fallback[3], "source": "fallback_quantile"}]
        )

    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0] + boxes[i][1], boxes[i][2] + boxes[i][3]))
    boxes = [boxes[i] for i in order]
    diag = pd.DataFrame(rows).iloc[order].reset_index(drop=True)
    diag["ColonyBoxID"] = np.arange(len(diag), dtype=int)
    return boxes, diag


def add_outside_colony_label(
    speed_df: pd.DataFrame,
    *,
    colony_boxes_mm: list[tuple[float, float, float, float]],
) -> pd.DataFrame:
    out = speed_df.copy()
    valid = out["X_mm"].notna() & out["Y_mm"].notna()
    colony_id = np.full(len(out), -1, dtype=int)
    x = out["X_mm"].to_numpy(float)
    y = out["Y_mm"].to_numpy(float)
    for box_id, (xmin, xmax, ymin, ymax) in enumerate(colony_boxes_mm):
        inside = valid.to_numpy(bool) & (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        colony_id[(colony_id < 0) & inside] = int(box_id)
    out["ColonyBoxID"] = colony_id
    out["InsideColony"] = valid & (out["ColonyBoxID"] >= 0)
    out["OutsideColony"] = valid & (out["ColonyBoxID"] < 0)
    return out


def behavior_summary_for_track(
    speed_df: pd.DataFrame,
    *,
    fps: float = FPS,
    track_key: str | None = None,
    track_path: Path | None = None,
    n_colony_boxes: int = N_COLONY_BOXES,
) -> dict[str, float | int | str]:
    tid = int(speed_df["TrackID"].dropna().iloc[0])
    valid_pos = speed_df["ValidPosition"].fillna(False)
    valid_speed = speed_df["ValidSpeed"].fillna(False)
    outside = speed_df.get("OutsideColony", pd.Series(False, index=speed_df.index)).fillna(False)
    inside = speed_df.get("InsideColony", pd.Series(False, index=speed_df.index)).fillna(False)
    sleep = speed_df.get("Sleep", pd.Series(False, index=speed_df.index)).fillna(False)
    colony_id = pd.to_numeric(
        speed_df.get("ColonyBoxID", pd.Series(-1, index=speed_df.index)),
        errors="coerce",
    ).fillna(-1).astype(int)

    outside_arr = outside.to_numpy(bool)
    valid_outside_arr = valid_pos.to_numpy(bool) & outside_arr
    outside_bouts = 0
    max_outside_bout = 0
    i = 0
    while i < len(outside_arr):
        if not valid_outside_arr[i]:
            i += 1
            continue
        j = i + 1
        while j < len(outside_arr) and valid_outside_arr[j]:
            j += 1
        outside_bouts += 1
        max_outside_bout = max(max_outside_bout, j - i)
        i = j

    valid_pos_frames = int(valid_pos.sum())
    valid_speed_frames = int(valid_speed.sum())
    outside_frames = int(valid_outside_arr.sum())
    inside_frames = int((valid_pos & inside).sum())
    sleep_frames = int((valid_speed & sleep).sum())
    duration_frames = int(speed_df["Frame"].max() - speed_df["Frame"].min() + 1)
    colony_counts = {
        int(cid): int(((valid_pos) & (colony_id == int(cid))).sum())
        for cid in sorted(colony_id[colony_id >= 0].unique().tolist())
    }
    primary_colony_box_id = max(colony_counts, key=colony_counts.get) if colony_counts else -1
    primary_colony_frames = colony_counts.get(primary_colony_box_id, 0)

    out: dict[str, float | int | str] = {
        "TrackID": tid,
        "TrackKey": track_key or "",
        "TrackPath": str(track_path) if track_path is not None else "",
        "duration_s": duration_frames / fps,
        "valid_position_s": valid_pos_frames / fps,
        "valid_speed_s": valid_speed_frames / fps,
        "inside_colony_s": inside_frames / fps,
        "inside_colony_frac": inside_frames / valid_pos_frames if valid_pos_frames else np.nan,
        "outside_s": outside_frames / fps,
        "outside_frac": outside_frames / valid_pos_frames if valid_pos_frames else np.nan,
        "outside_bouts": outside_bouts,
        "max_outside_bout_s": max_outside_bout / fps,
        "primary_colony_box_id": primary_colony_box_id,
        "primary_colony_frac": primary_colony_frames / valid_pos_frames if valid_pos_frames else np.nan,
        "sleep_s": sleep_frames / fps,
        "sleep_frac": sleep_frames / valid_speed_frames if valid_speed_frames else np.nan,
        "median_speed_mm_s": float(speed_df["SpeedMmPerSec"].median(skipna=True)),
    }
    for box_id in range(int(n_colony_boxes)):
        frames = colony_counts.get(box_id, 0)
        out[f"colony_box_{box_id}_s"] = frames / fps
        out[f"colony_box_{box_id}_frac"] = frames / valid_pos_frames if valid_pos_frames else np.nan
    return out


def cluster_by_colony_use(
    summary: pd.DataFrame,
    *,
    n_clusters: int = N_BEHAVIOR_CLUSTERS,
    worker_inside_colony_frac_threshold: float | None = WORKER_INSIDE_COLONY_FRAC_THRESHOLD,
) -> pd.DataFrame:
    out = summary.copy()
    if worker_inside_colony_frac_threshold is not None:
        threshold = float(worker_inside_colony_frac_threshold)
        if "inside_colony_frac" in out.columns:
            inside_frac = pd.to_numeric(out["inside_colony_frac"], errors="coerce")
        else:
            inside_frac = 1.0 - pd.to_numeric(out["outside_frac"], errors="coerce")
        is_worker = inside_frac >= threshold
        out["BehaviorCluster"] = np.where(is_worker, 0, 1).astype(int)
        out.loc[inside_frac.isna(), "BehaviorCluster"] = -1
        out["BehaviorLabel"] = np.where(is_worker, "worker", "forager")
        out.loc[inside_frac.isna(), "BehaviorLabel"] = "unclassified"
        out["ColonyUseClusterMethod"] = "inside_colony_frac_threshold"
        out["WorkerInsideColonyFracThreshold"] = threshold
        return out

    features = out[["outside_frac", "max_outside_bout_s", "outside_bouts"]].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    n = min(int(n_clusters), len(features))
    if n <= 1:
        out["BehaviorCluster"] = 0
    else:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans

            z = StandardScaler().fit_transform(features)
            labels = KMeans(n_clusters=n, random_state=0, n_init="auto").fit_predict(z)
        except Exception:
            labels = (
                pd.qcut(
                    out["outside_frac"].rank(method="first"),
                    q=n,
                    labels=False,
                    duplicates="drop",
                )
                .fillna(0)
                .astype(int)
                .to_numpy()
            )
        out["BehaviorCluster"] = labels.astype(int)

    cluster_order = (
        out.groupby("BehaviorCluster")["outside_frac"]
        .mean()
        .sort_values()
        .index.to_list()
    )
    remap = {old: new for new, old in enumerate(cluster_order)}
    out["BehaviorCluster"] = out["BehaviorCluster"].map(remap).astype(int)
    labels = ["mostly_in_colony", "mixed", "outside_prone", "strongly_outside"]
    out["BehaviorLabel"] = out["BehaviorCluster"].map(
        {i: labels[i] if i < len(labels) else f"cluster_{i}" for i in range(len(cluster_order))}
    )
    out["ColonyUseClusterMethod"] = "unsupervised_colony_use"
    out["WorkerInsideColonyFracThreshold"] = np.nan
    return out


def colony_boxes_signature(colony_boxes_mm: list[tuple[float, float, float, float]]) -> str:
    return ";".join(
        ",".join(f"{float(value):.9g}" for value in box)
        for box in colony_boxes_mm
    )


def behavior_cluster_config_table(
    *,
    n_clusters: int,
    worker_inside_colony_frac_threshold: float | None,
) -> pd.DataFrame:
    threshold = (
        np.nan
        if worker_inside_colony_frac_threshold is None
        else float(worker_inside_colony_frac_threshold)
    )
    method = (
        "inside_colony_frac_threshold"
        if worker_inside_colony_frac_threshold is not None
        else "unsupervised_colony_use"
    )
    return pd.DataFrame(
        [
            {
                "cluster_method": method,
                "n_behavior_clusters": int(n_clusters),
                "worker_inside_colony_frac_threshold": threshold,
            }
        ]
    )


def behavior_output_config_table(
    *,
    stationary_threshold_mm_s: float,
    min_sleep_stationary_seconds: float,
    fps: float,
    colony_boxes_mm: list[tuple[float, float, float, float]],
    n_clusters: int,
    worker_inside_colony_frac_threshold: float | None,
) -> pd.DataFrame:
    config = behavior_cluster_config_table(
        n_clusters=n_clusters,
        worker_inside_colony_frac_threshold=worker_inside_colony_frac_threshold,
    )
    config["stationary_threshold_mm_s"] = float(stationary_threshold_mm_s)
    config["min_sleep_stationary_seconds"] = float(min_sleep_stationary_seconds)
    config["fps"] = float(fps)
    config["colony_boxes_signature"] = colony_boxes_signature(colony_boxes_mm)
    return config


def sleep_summary_for_labeled(
    labeled: pd.DataFrame,
    *,
    track_key: str,
    track_path: Path,
) -> pd.DataFrame:
    if labeled.empty:
        return pd.DataFrame()
    tid = int(pd.to_numeric(labeled["TrackID"], errors="coerce").dropna().iloc[0])
    valid = labeled["ValidSpeed"].fillna(False)
    row = {
        "TrackID": tid,
        "TrackKey": track_key,
        "TrackPath": str(track_path),
        "sleep_frac": float(labeled.loc[valid, "Sleep"].mean()) if valid.any() else np.nan,
        "stationary_frac": float(labeled.loc[valid, "Stationary"].mean()) if valid.any() else np.nan,
        "median_speed_mm_s": float(labeled["SpeedMmPerSec"].median(skipna=True)),
        "valid_speed_frames": int(valid.sum()),
    }
    return pd.DataFrame([row])


def sample_valid_positions_from_paths(paths: list[Path], *, sample_per_track: int) -> pd.DataFrame:
    samples = []
    for i, path in enumerate(paths):
        cols = ["TrackID", "Frame", "X_mm", "Y_mm", "ValidPosition"]
        df = read_table(path, columns=cols)
        df = df[df["ValidPosition"].fillna(False)].dropna(subset=["X_mm", "Y_mm"])
        if df.empty:
            continue
        if len(df) > sample_per_track:
            df = df.sample(sample_per_track, random_state=i)
        samples.append(df)
    return pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=["TrackID", "Frame", "X_mm", "Y_mm"])


def rhythm_bins_for_track(
    labeled: pd.DataFrame,
    *,
    track_key: str,
    track_path: Path,
    bin_seconds: float = RHYTHM_BIN_SECONDS,
) -> pd.DataFrame:
    cols = ["TrackID", "Frame", "TimeS", "Sleep", "Stationary", "SpeedMmPerSec", "ValidSpeed"]
    missing = set(cols).difference(labeled.columns)
    if missing or labeled.empty:
        return pd.DataFrame()
    df = labeled[cols].copy()
    df = df[df["ValidSpeed"].fillna(False)]
    if df.empty:
        return pd.DataFrame()
    df["TimeBinS"] = np.floor(df["TimeS"] / float(bin_seconds)) * float(bin_seconds)
    out = (
        df.groupby(["TrackID", "TimeBinS"], as_index=False)
        .agg(
            sleep_sum=("Sleep", "sum"),
            stationary_sum=("Stationary", "sum"),
            speed_median_track=("SpeedMmPerSec", "median"),
            n_frames=("Frame", "size"),
        )
    )
    out["TrackKey"] = track_key
    out["TrackPath"] = str(track_path)
    return out[
        [
            "TrackID",
            "TrackKey",
            "TrackPath",
            "TimeBinS",
            "sleep_sum",
            "stationary_sum",
            "speed_median_track",
            "n_frames",
        ]
    ]


def read_worklist(path: Path) -> list[WorkItem]:
    if not path.exists():
        raise FileNotFoundError(f"Missing worklist: {path}")
    items: list[WorkItem] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 4:
            task_index, track_id, track_key, raw_path = parts
        elif len(parts) == 2:
            track_id, raw_path = parts
            task_index = str(len(items))
            track_key = track_cache_stem(raw_path)
        else:
            raise ValueError(f"Malformed worklist line {line_no}: {line!r}")
        items.append(WorkItem(int(task_index), int(track_id), track_key, Path(raw_path)))
    return items


def select_work_item(worklist: Path, task_id: int | None) -> WorkItem:
    if task_id is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError("Provide --task_id or set SLURM_ARRAY_TASK_ID")
        task_id = int(raw)
    items = read_worklist(worklist)
    if task_id < 0 or task_id >= len(items):
        raise IndexError(f"task_id {task_id} is outside worklist size {len(items)}")
    return items[task_id]


def parse_optional_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    cleaned = str(value).strip()
    if cleaned == "" or cleaned.lower() in {"none", "nan", "auto", "null"}:
        return None
    return float(cleaned)


def parse_colony_boxes(value: str | None) -> list[tuple[float, float, float, float]] | None:
    if value is None:
        return COLONY_BOXES_MM
    cleaned = value.strip()
    if cleaned == "" or cleaned.lower() in {"none", "auto", "infer", "null"}:
        return None
    boxes = []
    for raw_box in cleaned.split(";"):
        raw_box = raw_box.strip()
        if not raw_box:
            continue
        parts = [float(part.strip()) for part in raw_box.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Each colony box must have xmin,xmax,ymin,ymax: {raw_box!r}")
        boxes.append((parts[0], parts[1], parts[2], parts[3]))
    return boxes or None


def common_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "fps": float(args.fps),
        "mm_per_px": float(args.mm_per_px),
        "position_columns": tuple(args.position_columns),
        "bodypoint": int(args.bodypoint_for_xy),
        "max_interp_gap_frames": int(args.max_interp_gap_frames),
        "smooth_sigma_frames": float(args.smooth_sigma_frames),
        "max_reasonable_speed_mm_s": parse_optional_float(args.max_reasonable_speed_mm_s),
        "stationary_threshold_mm_s": parse_optional_float(args.stationary_threshold_mm_s),
        "min_sleep_stationary_seconds": float(args.min_sleep_stationary_seconds),
        "colony_boxes_mm": parse_colony_boxes(args.colony_boxes_mm),
        "n_colony_boxes": int(args.n_colony_boxes),
        "colony_density_bins": int(args.colony_density_bins),
        "colony_density_smooth_sigma_bins": float(args.colony_density_smooth_sigma_bins),
        "colony_density_high_quantile": float(args.colony_density_high_quantile),
        "colony_box_pad_mm": float(args.colony_box_pad_mm),
        "colony_box_central_fraction": float(args.colony_box_central_fraction),
        "n_behavior_clusters": int(args.n_behavior_clusters),
        "worker_inside_colony_frac_threshold": parse_optional_float(args.worker_inside_colony_frac_threshold),
        "rhythm_bin_seconds": float(args.rhythm_bin_seconds),
        "speed_vector_sample_per_track": int(args.speed_vector_sample_per_track),
        "position_sample_per_track": int(args.position_sample_per_track),
        "force": bool(args.force),
    }


def command_prepare(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir
    ensure_cache_dirs(cache_dir)
    track_presence_path = table_cache_path(cache_dir, "track_presence.parquet")
    use_cached_track_presence = use_cache(track_presence_path, force=args.force)
    if use_cached_track_presence:
        track_presence = read_table(track_presence_path)
        if track_presence_matches_input_dir(track_presence, args.per_track_dir):
            print(f"Loaded cached track presence: {track_presence_path}")
        else:
            print(
                f"Ignoring cached track presence with paths from a different input dir: "
                f"{track_presence_path}",
                flush=True,
            )
            use_cached_track_presence = False
    if not use_cached_track_presence:
        track_files = list_track_files(args.per_track_dir, args.side_filter)
        print(f"Found {len(track_files)} per-track parquet files in {args.per_track_dir}")
        if not track_files:
            raise RuntimeError(f"No TrackID_*.parquet files found in {args.per_track_dir}")
        track_summaries = [summarize_track_file(path) for path in track_files]
        dataset_num_frames = args.dataset_num_frames or infer_dataset_frame_count(track_summaries)
        track_presence = add_presence_fraction(track_summaries, dataset_num_frames)
        write_table(track_presence, track_presence_path)
        print(f"Wrote {track_presence_path}")

    good = track_presence[track_presence["present_frac"] >= float(args.min_track_present_frac)].copy()
    good = good.sort_values(["TrackID", "TrackKey"]).reset_index(drop=True)
    if good.empty:
        raise RuntimeError(
            f"No tracks pass min_track_present_frac={float(args.min_track_present_frac):.3g}"
        )

    position_columns = tuple(args.position_columns)
    position_diag = position_column_diagnostics(
        good,
        position_columns=position_columns,  # type: ignore[arg-type]
    )
    position_diag_path = table_cache_path(cache_dir, "worklist_position_columns.parquet")
    write_table(position_diag, position_diag_path)
    position_diag.to_csv(table_cache_path(cache_dir, "worklist_position_columns.csv"), index=False)

    usable = position_diag[position_diag["usable_position_columns"].fillna(False)].copy()
    unusable = len(position_diag) - len(usable)
    if unusable:
        print(
            f"Excluded {unusable} otherwise-good tracks with no usable position columns; "
            f"see {position_diag_path}",
            flush=True,
        )
    usable_paths = set(usable["path"].astype(str))
    good = good[good["path"].astype(str).isin(usable_paths)].reset_index(drop=True)
    if good.empty:
        tried = ", ".join(f"{x}/{y}" for x, y in position_column_candidates(position_columns))  # type: ignore[arg-type]
        raise RuntimeError(f"No good tracks have usable position columns; tried {tried}")

    worklist = args.worklist or default_worklist_path(cache_dir)
    worklist.parent.mkdir(parents=True, exist_ok=True)
    with worklist.open("w") as f:
        for task_index, row in enumerate(good.itertuples(index=False)):
            f.write(f"{task_index}\t{int(row.TrackID)}\t{row.TrackKey}\t{row.path}\n")
    table_cache_path(cache_dir, "worklist_count.txt").write_text(f"{len(good)}\n")
    print(
        f"Good tracks with usable positions: {len(good)} / {len(track_presence)} "
        f"at >= {float(args.min_track_present_frac):.0%} present"
    )
    print(f"Wrote worklist: {worklist}")


def command_speed(args: argparse.Namespace) -> None:
    cfg = common_config_from_args(args)
    cache_dir = args.cache_dir
    ensure_cache_dirs(cache_dir)
    item = select_work_item(args.worklist or default_worklist_path(cache_dir), args.task_id)
    out_path = speed_cache_path(cache_dir, item.path)
    sample_path = speed_sample_path(cache_dir, item.path)
    print(
        f"Speed task {item.task_index}: track={item.path} "
        f"speed_cache={out_path} speed_sample={sample_path}",
        flush=True,
    )
    if not use_cache(out_path, force=bool(cfg["force"])):
        print(f"Speed task {item.task_index}: reading position table", flush=True)
        pos = load_track_position_table(
            item.path,
            position_columns=cfg["position_columns"],  # type: ignore[arg-type]
            bodypoint=int(cfg["bodypoint"]),
        )
        print(f"Speed task {item.task_index}: loaded {len(pos)} position rows", flush=True)
        if pos.empty:
            print(f"Writing empty speed cache for empty position table: {item.path}")
            speed = empty_speed_cache_table()
        else:
            print(f"Speed task {item.task_index}: computing speed table", flush=True)
            speed = compute_gap_aware_speed(
                pos,
                fps=float(cfg["fps"]),
                mm_per_px=float(cfg["mm_per_px"]),
                max_interp_gap_frames=int(cfg["max_interp_gap_frames"]),
                smooth_sigma_frames=float(cfg["smooth_sigma_frames"]),
                max_reasonable_speed_mm_s=cfg["max_reasonable_speed_mm_s"],  # type: ignore[arg-type]
            )
            speed["TrackKey"] = item.track_key
            speed["TrackPath"] = str(item.path)
            print(f"Speed task {item.task_index}: computed {len(speed)} speed rows", flush=True)
        print(f"Speed task {item.task_index}: writing speed cache {out_path}", flush=True)
        write_table(speed, out_path)
        print(f"Speed task {item.task_index}: wrote speed cache {out_path}", flush=True)
    else:
        print(f"Speed task {item.task_index}: using cached speed cache {out_path}", flush=True)
        speed = read_table(out_path, columns=["SpeedMmPerSec"])

    if not use_cache(sample_path, force=bool(cfg["force"])):
        if "SpeedMmPerSec" not in speed.columns:
            speed = read_table(out_path, columns=["SpeedMmPerSec"])
        values = sample_series(
            speed["SpeedMmPerSec"],
            int(cfg["speed_vector_sample_per_track"]),
            random_state=item.task_index,
        )
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Speed task {item.task_index}: writing speed sample {sample_path}", flush=True)
        np.save(sample_path, values)
        print(f"Speed task {item.task_index}: wrote speed sample {sample_path}", flush=True)
    else:
        print(f"Speed task {item.task_index}: using cached speed sample {sample_path}", flush=True)
    print(f"Wrote speed cache for task {item.task_index}: {out_path}")


def load_speed_values_from_samples(cache_dir: Path, worklist: Path, sample_per_track: int) -> np.ndarray:
    parts = []
    for item in read_worklist(worklist):
        sample_path = speed_sample_path(cache_dir, item.path)
        if sample_path.exists():
            parts.append(np.load(sample_path))
            continue
        speed_path = speed_cache_path(cache_dir, item.path)
        if speed_path.exists():
            speed = read_table(speed_path, columns=["SpeedMmPerSec"])
            parts.append(sample_series(speed["SpeedMmPerSec"], sample_per_track, random_state=item.task_index))
    nonempty = [v for v in parts if len(v)]
    return np.concatenate(nonempty) if nonempty else np.empty(0, dtype=float)


def command_threshold(args: argparse.Namespace) -> None:
    cfg = common_config_from_args(args)
    cache_dir = args.cache_dir
    ensure_cache_dirs(cache_dir)
    worklist = args.worklist or default_worklist_path(cache_dir)
    threshold_table_path = table_cache_path(cache_dir, "stationary_threshold.csv")
    speed_vector_path = vector_cache_path(cache_dir, "speed_values_mm_s.npy")
    speed_histogram_path = table_cache_path(cache_dir, "speed_histogram.parquet")
    speed_preview_path = table_cache_path(cache_dir, "speed_preview.parquet")

    speed_values = load_speed_values_from_samples(
        cache_dir,
        worklist,
        int(cfg["speed_vector_sample_per_track"]),
    )
    if len(speed_values):
        np.save(speed_vector_path, speed_values)

    stationary_threshold_mm_s = cfg["stationary_threshold_mm_s"]
    if stationary_threshold_mm_s is None:
        stationary_threshold_mm_s = estimate_stationary_threshold(speed_values)

    write_table(
        pd.DataFrame(
            [
                {
                    "stationary_threshold_mm_s": stationary_threshold_mm_s,
                    "fps": cfg["fps"],
                    "min_sleep_stationary_seconds": cfg["min_sleep_stationary_seconds"],
                    "n_speed_values": len(speed_values),
                }
            ]
        ),
        threshold_table_path,
    )
    write_table(speed_histogram_table(speed_values), speed_histogram_path)

    preview_rows = []
    for item in read_worklist(worklist)[:5]:
        path = speed_cache_path(cache_dir, item.path)
        if path.exists():
            preview_rows.append(read_table(path).head())
    preview = pd.concat(preview_rows, ignore_index=True) if preview_rows else pd.DataFrame()
    write_table(preview, speed_preview_path)
    print(f"Stationary threshold: {float(stationary_threshold_mm_s):.6g} mm/s")


def load_stationary_threshold(cache_dir: Path) -> float:
    threshold_table = read_table(table_cache_path(cache_dir, "stationary_threshold.csv"))
    return float(threshold_table["stationary_threshold_mm_s"].iloc[0])


def command_label(args: argparse.Namespace) -> None:
    cfg = common_config_from_args(args)
    cache_dir = args.cache_dir
    ensure_cache_dirs(cache_dir)
    item = select_work_item(args.worklist or default_worklist_path(cache_dir), args.task_id)
    speed_path = speed_cache_path(cache_dir, item.path)
    if not speed_path.exists():
        raise FileNotFoundError(f"Missing speed cache for {item.path}: {speed_path}")

    out_path = labeled_cache_path(cache_dir, item.path)
    stationary_threshold_mm_s = load_stationary_threshold(cache_dir)
    if not use_cache(out_path, force=bool(cfg["force"])):
        speed = read_table(speed_path)
        labeled = label_sleep_from_speed(
            speed,
            stationary_threshold_mm_s=stationary_threshold_mm_s,
            min_sleep_stationary_seconds=float(cfg["min_sleep_stationary_seconds"]),
            fps=float(cfg["fps"]),
        )
        labeled["TrackKey"] = item.track_key
        labeled["TrackPath"] = str(item.path)
        write_table(labeled, out_path)
    else:
        labeled = read_table(out_path)

    summary = sleep_summary_for_labeled(labeled, track_key=item.track_key, track_path=item.path)
    if not summary.empty:
        write_table(summary, sleep_summary_track_path(cache_dir, item.path))
    print(f"Wrote labeled sleep cache for task {item.task_index}: {out_path}")


def command_colony(args: argparse.Namespace) -> None:
    cfg = common_config_from_args(args)
    cache_dir = args.cache_dir
    ensure_cache_dirs(cache_dir)
    worklist = args.worklist or default_worklist_path(cache_dir)
    colony_box_path = table_cache_path(cache_dir, "colony_boxes.parquet")
    colony_boxes_mm = cfg["colony_boxes_mm"]
    if colony_boxes_mm is not None:
        colony_box_diag = pd.DataFrame(
            [
                {
                    "ColonyBoxID": i,
                    "xmin_mm": box[0],
                    "xmax_mm": box[1],
                    "ymin_mm": box[2],
                    "ymax_mm": box[3],
                    "source": "manual",
                }
                for i, box in enumerate(colony_boxes_mm)  # type: ignore[arg-type]
            ]
        )
    elif use_cache(colony_box_path, force=bool(cfg["force"])):
        colony_box_diag = read_table(colony_box_path)
        colony_boxes_mm = [
            (float(row.xmin_mm), float(row.xmax_mm), float(row.ymin_mm), float(row.ymax_mm))
            for row in colony_box_diag.itertuples(index=False)
        ]
    else:
        labeled_paths = [
            labeled_cache_path(cache_dir, item.path)
            for item in read_worklist(worklist)
            if labeled_cache_path(cache_dir, item.path).exists()
        ]
        if not labeled_paths:
            raise RuntimeError("No labeled track caches are available for colony box inference")
        position_sample_path = table_cache_path(cache_dir, "valid_position_sample.parquet")
        if use_cache(position_sample_path, force=bool(cfg["force"])):
            valid_positions = read_table(position_sample_path)
        else:
            valid_positions = sample_valid_positions_from_paths(
                labeled_paths,
                sample_per_track=int(cfg["position_sample_per_track"]),
            )
            write_table(valid_positions, position_sample_path)
        colony_boxes_mm, colony_box_diag = infer_colony_boxes_by_density(
            valid_positions,
            n_boxes=int(cfg["n_colony_boxes"]),
            bins=int(cfg["colony_density_bins"]),
            smooth_sigma_bins=float(cfg["colony_density_smooth_sigma_bins"]),
            high_quantile=float(cfg["colony_density_high_quantile"]),
            pad_mm=float(cfg["colony_box_pad_mm"]),
            fallback_central_fraction=float(cfg["colony_box_central_fraction"]),
        )
        if "source" not in colony_box_diag.columns:
            colony_box_diag["source"] = "density"

    write_table(colony_box_diag, colony_box_path)
    stationary_threshold_mm_s = load_stationary_threshold(cache_dir)
    config = behavior_output_config_table(
        stationary_threshold_mm_s=stationary_threshold_mm_s,
        min_sleep_stationary_seconds=float(cfg["min_sleep_stationary_seconds"]),
        fps=float(cfg["fps"]),
        colony_boxes_mm=colony_boxes_mm,  # type: ignore[arg-type]
        n_clusters=0,
        worker_inside_colony_frac_threshold=None,
    )
    write_table(config, table_cache_path(cache_dir, "outside_labeled_config.csv"))
    print(f"Wrote colony boxes: {colony_box_path}")


def load_colony_boxes(cache_dir: Path) -> list[tuple[float, float, float, float]]:
    colony_box_diag = read_table(table_cache_path(cache_dir, "colony_boxes.parquet"))
    return [
        (float(row.xmin_mm), float(row.xmax_mm), float(row.ymin_mm), float(row.ymax_mm))
        for row in colony_box_diag.itertuples(index=False)
    ]


def command_outside(args: argparse.Namespace) -> None:
    cfg = common_config_from_args(args)
    cache_dir = args.cache_dir
    ensure_cache_dirs(cache_dir)
    item = select_work_item(args.worklist or default_worklist_path(cache_dir), args.task_id)
    labeled_path = labeled_cache_path(cache_dir, item.path)
    if not labeled_path.exists():
        raise FileNotFoundError(f"Missing labeled cache for {item.path}: {labeled_path}")

    out_path = outside_labeled_cache_path(cache_dir, item.path)
    colony_boxes_mm = load_colony_boxes(cache_dir)
    if not use_cache(out_path, force=bool(cfg["force"])):
        labeled = read_table(labeled_path)
        outside = add_outside_colony_label(labeled, colony_boxes_mm=colony_boxes_mm)
        outside["TrackKey"] = item.track_key
        outside["TrackPath"] = str(item.path)
        write_table(outside, out_path)
    else:
        outside = read_table(out_path)

    if outside.empty:
        print(f"Skipping behavior summary for empty outside-colony cache: {out_path}")
        return

    behavior_row = behavior_summary_for_track(
        outside,
        fps=float(cfg["fps"]),
        track_key=item.track_key,
        track_path=item.path,
        n_colony_boxes=int(cfg["n_colony_boxes"]),
    )
    write_table(pd.DataFrame([behavior_row]), behavior_summary_track_path(cache_dir, item.path))
    rhythm_bins = rhythm_bins_for_track(
        outside,
        track_key=item.track_key,
        track_path=item.path,
        bin_seconds=float(cfg["rhythm_bin_seconds"]),
    )
    if not rhythm_bins.empty:
        write_table(rhythm_bins, rhythm_bins_track_path(cache_dir, item.path))
    print(f"Wrote outside-colony cache for task {item.task_index}: {out_path}")


def concat_existing(paths: list[Path]) -> pd.DataFrame:
    frames = [read_table(path) for path in paths if path.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def command_aggregate(args: argparse.Namespace) -> None:
    cfg = common_config_from_args(args)
    cache_dir = args.cache_dir
    ensure_cache_dirs(cache_dir)
    worklist = args.worklist or default_worklist_path(cache_dir)
    items = read_worklist(worklist)
    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    sleep_summary = concat_existing([sleep_summary_track_path(cache_dir, item.path) for item in items])
    if not sleep_summary.empty:
        sleep_summary = sleep_summary.sort_values(["TrackID", "TrackKey"]).reset_index(drop=True)
    write_table(sleep_summary, table_cache_path(cache_dir, "sleep_summary.parquet"))

    behavior_summary = concat_existing([behavior_summary_track_path(cache_dir, item.path) for item in items])
    if behavior_summary.empty:
        raise RuntimeError("No per-track behavior summaries were produced")
    behavior_summary = cluster_by_colony_use(
        behavior_summary,
        n_clusters=int(cfg["n_behavior_clusters"]),
        worker_inside_colony_frac_threshold=cfg["worker_inside_colony_frac_threshold"],  # type: ignore[arg-type]
    )
    behavior_summary = behavior_summary.sort_values(["TrackID", "TrackKey"]).reset_index(drop=True)
    behavior_summary_path = table_cache_path(cache_dir, "behavior_summary.parquet")
    write_table(behavior_summary, behavior_summary_path)

    colony_boxes_mm = load_colony_boxes(cache_dir)
    stationary_threshold_mm_s = load_stationary_threshold(cache_dir)
    current_config = behavior_output_config_table(
        stationary_threshold_mm_s=stationary_threshold_mm_s,
        min_sleep_stationary_seconds=float(cfg["min_sleep_stationary_seconds"]),
        fps=float(cfg["fps"]),
        colony_boxes_mm=colony_boxes_mm,
        n_clusters=int(cfg["n_behavior_clusters"]),
        worker_inside_colony_frac_threshold=cfg["worker_inside_colony_frac_threshold"],  # type: ignore[arg-type]
    )
    write_table(current_config, table_cache_path(cache_dir, "behavior_summary_config.csv"))
    write_table(current_config, table_cache_path(cache_dir, "behavior_cluster_config.csv"))

    cluster_agg = {
        "n_tracks": ("TrackKey", "nunique"),
        "outside_frac_mean": ("outside_frac", "mean"),
        "outside_s_mean": ("outside_s", "mean"),
        "inside_colony_frac_mean": ("inside_colony_frac", "mean"),
        "sleep_frac_mean": ("sleep_frac", "mean"),
        "sleep_s_mean": ("sleep_s", "mean"),
        "median_speed_mm_s": ("median_speed_mm_s", "median"),
    }
    for col in sorted(c for c in behavior_summary.columns if re.fullmatch(r"colony_box_\d+_frac", c)):
        cluster_agg[f"{col}_mean"] = (col, "mean")
    cluster_sleep_summary = (
        behavior_summary.groupby(["BehaviorCluster", "BehaviorLabel"], as_index=False)
        .agg(**cluster_agg)
        .sort_values("BehaviorCluster")
    )
    write_table(cluster_sleep_summary, table_cache_path(cache_dir, "cluster_sleep_summary.parquet"))
    write_table(current_config, table_cache_path(cache_dir, "cluster_sleep_summary_config.csv"))

    rhythm_bins = concat_existing([rhythm_bins_track_path(cache_dir, item.path) for item in items])
    if not rhythm_bins.empty:
        cluster_cols = behavior_summary[["TrackKey", "TrackID", "BehaviorCluster", "BehaviorLabel"]].copy()
        rhythm_bins = rhythm_bins.merge(
            cluster_cols,
            on=["TrackKey", "TrackID"],
            how="left",
            validate="many_to_one",
        )
        grouped = (
            rhythm_bins.groupby(["BehaviorCluster", "BehaviorLabel", "TimeBinS"], as_index=False)
            .agg(
                sleep_sum=("sleep_sum", "sum"),
                stationary_sum=("stationary_sum", "sum"),
                median_speed_mm_s=("speed_median_track", "median"),
                n_frames=("n_frames", "sum"),
                n_tracks=("TrackKey", "nunique"),
            )
            .sort_values(["BehaviorCluster", "TimeBinS"])
            .reset_index(drop=True)
        )
        grouped["sleep_frac"] = grouped["sleep_sum"] / grouped["n_frames"].replace(0, np.nan)
        grouped["stationary_frac"] = grouped["stationary_sum"] / grouped["n_frames"].replace(0, np.nan)
        sleep_rhythm = grouped[
            [
                "BehaviorCluster",
                "BehaviorLabel",
                "TimeBinS",
                "sleep_frac",
                "stationary_frac",
                "median_speed_mm_s",
                "n_frames",
                "n_tracks",
            ]
        ]
    else:
        sleep_rhythm = pd.DataFrame()
    write_table(sleep_rhythm, table_cache_path(cache_dir, "sleep_rhythm.parquet"))
    write_table(current_config, table_cache_path(cache_dir, "sleep_rhythm_config.csv"))

    if output_dir is not None:
        track_presence_path = table_cache_path(cache_dir, "track_presence.parquet")
        if track_presence_path.exists():
            read_table(track_presence_path).to_csv(output_dir / "track_presence.csv", index=False)
        position_diag_path = table_cache_path(cache_dir, "worklist_position_columns.parquet")
        if position_diag_path.exists():
            read_table(position_diag_path).to_csv(output_dir / "worklist_position_columns.csv", index=False)
        sleep_summary.to_csv(output_dir / "sleep_summary.csv", index=False)
        behavior_summary.to_csv(output_dir / "behavior_summary.csv", index=False)
        cluster_sleep_summary.to_csv(output_dir / "cluster_sleep_summary.csv", index=False)
        sleep_rhythm.to_csv(output_dir / "sleep_rhythm.csv", index=False)
        colony_box_path = table_cache_path(cache_dir, "colony_boxes.parquet")
        if colony_box_path.exists():
            read_table(colony_box_path).to_csv(output_dir / "colony_boxes.csv", index=False)
        manifest = pd.DataFrame(
            {
                "TrackID": [item.track_id for item in items],
                "TrackKey": [item.track_key for item in items],
                "track_path": [str(item.path) for item in items],
                "speed_cache_path": [str(speed_cache_path(cache_dir, item.path)) for item in items],
                "labeled_cache_path": [str(labeled_cache_path(cache_dir, item.path)) for item in items],
                "outside_labeled_cache_path": [str(outside_labeled_cache_path(cache_dir, item.path)) for item in items],
            }
        )
        manifest.to_csv(output_dir / "per_track_cache_manifest.csv", index=False)

    if args.done_file is not None:
        args.done_file.parent.mkdir(parents=True, exist_ok=True)
        args.done_file.write_text(f"completed {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Wrote aggregate behavior outputs under {cache_dir}")


def command_run_local(args: argparse.Namespace) -> None:
    command_prepare(args)
    worklist = args.worklist or default_worklist_path(args.cache_dir)
    items = read_worklist(worklist)
    for item in items:
        args.task_id = item.task_index
        command_speed(args)
    command_threshold(args)
    for item in items:
        args.task_id = item.task_index
        command_label(args)
    command_colony(args)
    for item in items:
        args.task_id = item.task_index
    command_outside(args)
    command_aggregate(args)


def _read_job_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _read_first_job_id(path: Path) -> str:
    ids = _read_job_ids(path)
    return ids[0] if ids else ""


def _file_status(path: Path) -> tuple[bool, int, str]:
    if not path.exists():
        return False, 0, ""
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return True, int(stat.st_size), mtime


def _log_tail(path: Path, max_chars: int = 800) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    with path.open("rb") as fh:
        size = fh.seek(0, os.SEEK_END)
        fh.seek(max(0, size - max_chars))
        text = fh.read().decode("utf-8", errors="replace")
    return " ".join(text.split())


def _log_error_hint(path: Path) -> str:
    tail = _log_tail(path, max_chars=2000).lower()
    if not tail:
        return ""
    if "oom" in tail or "oom_kill" in tail or "killed" in tail:
        return "oom_or_killed"
    if "dependencyneversatisfied" in tail:
        return "dependency_never_satisfied"
    if "no such file or directory" in tail:
        return "missing_file_or_directory"
    if "traceback" in tail:
        return "python_traceback"
    return "see_log_err"


def _stage_cache_paths(cache_dir: Path, item: WorkItem, stage: str) -> tuple[Path, Path | None]:
    if stage == "speed":
        return speed_cache_path(cache_dir, item.path), speed_sample_path(cache_dir, item.path)
    if stage == "label":
        return labeled_cache_path(cache_dir, item.path), sleep_summary_track_path(cache_dir, item.path)
    if stage == "outside":
        return outside_labeled_cache_path(cache_dir, item.path), behavior_summary_track_path(cache_dir, item.path)
    raise ValueError(f"Unknown diagnostic stage: {stage}")


def command_diagnose(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir
    worklist = args.worklist or default_worklist_path(cache_dir)
    jobs_dir = args.jobs_dir
    if jobs_dir is None:
        raise ValueError("--jobs_dir is required for dependency diagnosis")
    logs_dir = args.logs_dir or jobs_dir / "sleep_behavior_logs"
    report_path = args.report_path or jobs_dir / "sleep_behavior_dependency_report.tsv"
    stages = ["speed", "label", "outside"] if args.stage == "all" else [args.stage]
    items = read_worklist(worklist)

    stage_job_files = {
        "speed": jobs_dir / "sleep_behavior_speed_job_ids.txt",
        "label": jobs_dir / "sleep_behavior_label_job_ids.txt",
        "outside": jobs_dir / "sleep_behavior_outside_job_ids.txt",
    }
    stage_dependencies = {
        "speed": "",
        "label": _read_first_job_id(jobs_dir / "sleep_behavior_threshold_job_id.txt"),
        "outside": _read_first_job_id(jobs_dir / "sleep_behavior_colony_job_id.txt"),
    }

    rows = []
    summaries = []
    for stage in stages:
        job_ids = _read_job_ids(stage_job_files[stage])
        stage_ok = 0
        for item in items:
            job_id = job_ids[item.task_index] if item.task_index < len(job_ids) else ""
            primary, secondary = _stage_cache_paths(cache_dir, item, stage)
            required_paths = [primary]
            if stage == "speed" and secondary is not None:
                required_paths.append(secondary)
            missing_required = [str(path) for path in required_paths if not path.exists()]
            if not missing_required and job_id:
                stage_ok += 1

            log_out = logs_dir / f"sleep_{stage}_task{item.task_index}_{job_id}.out" if job_id else Path("")
            log_err = logs_dir / f"sleep_{stage}_task{item.task_index}_{job_id}.err" if job_id else Path("")
            primary_exists, primary_size, primary_mtime = _file_status(primary)
            if secondary is not None:
                secondary_exists, secondary_size, secondary_mtime = _file_status(secondary)
            else:
                secondary_exists, secondary_size, secondary_mtime = False, 0, ""
            log_out_exists, log_out_size, log_out_mtime = _file_status(log_out) if job_id else (False, 0, "")
            log_err_exists, log_err_size, log_err_mtime = _file_status(log_err) if job_id else (False, 0, "")
            log_err_hint = _log_error_hint(log_err) if job_id else ""
            log_err_tail = _log_tail(log_err) if job_id else ""
            status = "ok"
            if not job_id:
                status = "missing_job_id"
            elif missing_required:
                status = "missing_required_outputs"
            elif not log_out_exists and not log_err_exists:
                status = "outputs_present_but_logs_missing"

            rows.append(
                {
                    "stage": stage,
                    "task_id": item.task_index,
                    "job_id": job_id,
                    "depends_on_job_id": stage_dependencies[stage],
                    "track_id": item.track_id,
                    "track_key": item.track_key,
                    "track_path": str(item.path),
                    "status": status,
                    "missing_required_outputs": ";".join(missing_required),
                    "expected_primary": str(primary),
                    "primary_exists": primary_exists,
                    "primary_size_bytes": primary_size,
                    "primary_mtime": primary_mtime,
                    "expected_secondary": str(secondary) if secondary is not None else "",
                    "secondary_exists": secondary_exists,
                    "secondary_size_bytes": secondary_size,
                    "secondary_mtime": secondary_mtime,
                    "log_out": str(log_out) if job_id else "",
                    "log_out_exists": log_out_exists,
                    "log_out_size_bytes": log_out_size,
                    "log_out_mtime": log_out_mtime,
                    "log_err": str(log_err) if job_id else "",
                    "log_err_exists": log_err_exists,
                    "log_err_size_bytes": log_err_size,
                    "log_err_mtime": log_err_mtime,
                    "log_err_hint": log_err_hint,
                    "log_err_tail": log_err_tail,
                }
            )
        summaries.append((stage, stage_ok, len(items), len(job_ids)))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(report_path, sep="\t", index=False)

    print(f"Wrote sleep dependency report: {report_path}")
    for stage, stage_ok, n_items, n_jobs in summaries:
        print(f"{stage}: required outputs complete for {stage_ok}/{n_items} tasks; job ids={n_jobs}")
    missing = report[report["status"].isin(["missing_job_id", "missing_required_outputs"])]
    if missing.empty:
        print("No missing required task outputs found for selected stages.")
        return

    first_missing_stage = None
    for stage in ["speed", "label", "outside"]:
        if stage in stages and not missing[missing["stage"] == stage].empty:
            first_missing_stage = stage
            break
    if first_missing_stage is None:
        first_missing = missing
        print(f"Missing rows: {len(missing)}")
    else:
        first_missing = missing[missing["stage"] == first_missing_stage]
        print(
            f"First incomplete dependency stage: {first_missing_stage} "
            f"({len(first_missing)} missing/blocking rows)"
        )
        downstream_missing = len(missing) - len(first_missing)
        if downstream_missing:
            print(
                f"Downstream missing rows not shown here: {downstream_missing} "
                "(expected until the first incomplete stage clears)"
            )
    hint_counts = first_missing["log_err_hint"].replace("", pd.NA).dropna().value_counts()
    if not hint_counts.empty:
        hints = ", ".join(f"{hint}={count}" for hint, count in hint_counts.items())
        print(f"Error hints in first incomplete stage: {hints}")
    for _, row in first_missing.head(int(args.max_rows)).iterrows():
        print(
            "MISSING "
            f"stage={row['stage']} task={row['task_id']} job={row['job_id']} "
            f"track={row['track_key']} status={row['status']} "
            f"missing={row['missing_required_outputs']} "
            f"log_out={row['log_out']} log_err={row['log_err']} "
            f"hint={row.get('log_err_hint', '')} err_tail={row.get('log_err_tail', '')}"
        )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--worklist", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--mm_per_px", type=float, default=MM_PER_PX)
    parser.add_argument("--min_track_present_frac", type=float, default=MIN_TRACK_PRESENT_FRAC)
    parser.add_argument("--position_columns", nargs=2, default=list(POSITION_COLUMNS))
    parser.add_argument("--bodypoint_for_xy", type=int, default=BODYPOINT_FOR_XY)
    parser.add_argument("--max_interp_gap_frames", type=int, default=MAX_INTERP_GAP_FRAMES)
    parser.add_argument("--smooth_sigma_frames", type=float, default=SMOOTH_SIGMA_FRAMES)
    parser.add_argument("--max_reasonable_speed_mm_s", default=str(MAX_REASONABLE_SPEED_MM_S))
    parser.add_argument("--stationary_threshold_mm_s", default=str(STATIONARY_THRESHOLD_MM_S))
    parser.add_argument("--min_sleep_stationary_seconds", type=float, default=MIN_SLEEP_STATIONARY_SECONDS)
    parser.add_argument(
        "--colony_boxes_mm",
        default=";".join(",".join(str(v) for v in box) for box in COLONY_BOXES_MM or []),
        help="Semicolon-separated xmin,xmax,ymin,ymax boxes, or 'auto' to infer from occupancy.",
    )
    parser.add_argument("--n_colony_boxes", type=int, default=N_COLONY_BOXES)
    parser.add_argument("--colony_density_bins", type=int, default=COLONY_DENSITY_BINS)
    parser.add_argument("--colony_density_smooth_sigma_bins", type=float, default=COLONY_DENSITY_SMOOTH_SIGMA_BINS)
    parser.add_argument("--colony_density_high_quantile", type=float, default=COLONY_DENSITY_HIGH_QUANTILE)
    parser.add_argument("--colony_box_pad_mm", type=float, default=COLONY_BOX_PAD_MM)
    parser.add_argument("--colony_box_central_fraction", type=float, default=COLONY_BOX_CENTRAL_FRACTION)
    parser.add_argument("--n_behavior_clusters", type=int, default=N_BEHAVIOR_CLUSTERS)
    parser.add_argument("--worker_inside_colony_frac_threshold", default=str(WORKER_INSIDE_COLONY_FRAC_THRESHOLD))
    parser.add_argument("--rhythm_bin_seconds", type=float, default=RHYTHM_BIN_SECONDS)
    parser.add_argument("--speed_vector_sample_per_track", type=int, default=SPEED_VECTOR_SAMPLE_PER_TRACK)
    parser.add_argument("--position_sample_per_track", type=int, default=POSITION_SAMPLE_PER_TRACK)
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    add_common_args(prepare)
    prepare.add_argument("--per_track_dir", type=Path, required=True)
    prepare.add_argument("--side_filter", default=None)
    prepare.add_argument("--dataset_num_frames", type=int, default=None)
    prepare.set_defaults(func=command_prepare)

    speed = subparsers.add_parser("speed")
    add_common_args(speed)
    speed.add_argument("--task_id", type=int, default=None)
    speed.set_defaults(func=command_speed)

    threshold = subparsers.add_parser("threshold")
    add_common_args(threshold)
    threshold.set_defaults(func=command_threshold)

    label = subparsers.add_parser("label")
    add_common_args(label)
    label.add_argument("--task_id", type=int, default=None)
    label.set_defaults(func=command_label)

    colony = subparsers.add_parser("colony")
    add_common_args(colony)
    colony.set_defaults(func=command_colony)

    outside = subparsers.add_parser("outside")
    add_common_args(outside)
    outside.add_argument("--task_id", type=int, default=None)
    outside.set_defaults(func=command_outside)

    aggregate = subparsers.add_parser("aggregate")
    add_common_args(aggregate)
    aggregate.add_argument("--output_dir", type=Path, default=None)
    aggregate.add_argument("--done_file", type=Path, default=None)
    aggregate.set_defaults(func=command_aggregate)

    run_local = subparsers.add_parser("run-local")
    add_common_args(run_local)
    run_local.add_argument("--per_track_dir", type=Path, required=True)
    run_local.add_argument("--side_filter", default=None)
    run_local.add_argument("--dataset_num_frames", type=int, default=None)
    run_local.add_argument("--output_dir", type=Path, default=None)
    run_local.add_argument("--done_file", type=Path, default=None)
    run_local.set_defaults(func=command_run_local)

    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--cache_dir", type=Path, required=True)
    diagnose.add_argument("--worklist", type=Path, default=None)
    diagnose.add_argument("--jobs_dir", type=Path, required=True)
    diagnose.add_argument("--logs_dir", type=Path, default=None)
    diagnose.add_argument("--report_path", type=Path, default=None)
    diagnose.add_argument("--stage", choices=["all", "speed", "label", "outside"], default="all")
    diagnose.add_argument("--max_rows", type=int, default=20)
    diagnose.set_defaults(func=command_diagnose)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(
        "[{}] colony_sleep_behavior_batch.py start command={} host={} pid={}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            args.command,
            os.uname().nodename if hasattr(os, "uname") else "unknown",
            os.getpid(),
        ),
        flush=True,
    )
    if hasattr(args, "cache_dir"):
        print(f"cache_dir={args.cache_dir}", flush=True)
    if getattr(args, "worklist", None) is not None:
        print(f"worklist={args.worklist}", flush=True)
    if getattr(args, "task_id", None) is not None:
        print(f"task_id={args.task_id}", flush=True)
    args.func(args)


if __name__ == "__main__":
    main()
