#%%
"""
Interactive colony postprocessing and sleep/behavior analysis.

Open this file in VS Code and run cells with Shift+Enter.
The workflow assumes stitched per-track parquet files from:
    <block>/stitched/per_track/TrackID_*.parquet

Large intermediate outputs are cached under CACHE_DIR. Set
FORCE_RECOMPUTE=True when changing parameters that affect cached outputs.
"""
%matplotlib qt
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    ip = get_ipython()  # type: ignore[name-defined]
    if ip is not None:
        ip.run_line_magic("matplotlib", "qt")
except Exception:
    pass

try:
    display  # type: ignore[name-defined]
except NameError:
    try:
        from IPython.display import display  # type: ignore[assignment]
    except Exception:
        def display(obj: object) -> None:
            print(obj)


#%%
# ------------------------- editable config -------------------------

PER_TRACK_DIR = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/per_track"
)
SIDE_FILTER: str | None = None  # "left", "right", or None
DATASET_NUM_FRAMES: int | None = None  # None = infer from parquet metadata/max frame

FPS = 24.0
MM_PER_PX = 0.016

MIN_TRACK_PRESENT_FRAC = 0.40
POSITION_COLUMNS = ("TrackX", "TrackY")  # Prefer ArUco-driven position.
BODYPOINT_FOR_XY = 0  # Used only if POSITION_COLUMNS=("X", "Y").

MAX_INTERP_GAP_FRAMES = 5
SMOOTH_SIGMA_FRAMES = 2.0
MAX_REASONABLE_SPEED_MM_S: float | None = 5  # Example: 200.0 to wipe extreme jumps.

STATIONARY_THRESHOLD_MM_S: float | None = 0.1  # Set a float to hardcode; None = estimate from speed distribution.
MIN_SLEEP_STATIONARY_SECONDS = 10.0

COLONY_BOXES_MM: list[tuple[float, float, float, float]] | None = [
    (-86, -32, -63, -8),
    (93, 149, -63, -8),
]  # [(xmin, xmax, ymin, ymax), ...]
N_COLONY_BOXES = 2
COLONY_DENSITY_BINS = 220
COLONY_DENSITY_SMOOTH_SIGMA_BINS = 2.0
COLONY_DENSITY_HIGH_QUANTILE = 0.86
COLONY_BOX_PAD_MM = 5.0
COLONY_BOX_CENTRAL_FRACTION = 0.90  # fallback if density detection fails
N_BEHAVIOR_CLUSTERS = 2
WORKER_INSIDE_COLONY_FRAC_THRESHOLD: float | None = .95  # Example: 0.80 labels >=80% inside as worker, else forager.

RHYTHM_BIN_SECONDS = 10 * 60

CACHE_VERSION = ""
CACHE_DIR = PER_TRACK_DIR.parent / "analysis_cache" / CACHE_VERSION
FORCE_RECOMPUTE = False
SPEED_VECTOR_SAMPLE_PER_TRACK = 100_000
POSITION_SAMPLE_PER_TRACK = 20_000

OUTPUT_DIR: Path | None = None


#%%
# ------------------------- cache helpers -------------------------

def ensure_cache_dirs() -> None:
    for subdir in (
        CACHE_DIR,
        CACHE_DIR / "speed_tracks",
        CACHE_DIR / "labeled_tracks",
        CACHE_DIR / "outside_labeled_tracks",
        CACHE_DIR / "vectors",
        CACHE_DIR / "tables",
    ):
        subdir.mkdir(parents=True, exist_ok=True)


def use_cache(path: Path, *, force: bool = FORCE_RECOMPUTE) -> bool:
    return path.exists() and not force


def track_cache_stem(path: Path | str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).stem)


def speed_cache_path(track_path: Path | str) -> Path:
    return CACHE_DIR / "speed_tracks" / f"{track_cache_stem(track_path)}_speed.parquet"


def labeled_cache_path(track_path: Path | str) -> Path:
    return CACHE_DIR / "labeled_tracks" / f"{track_cache_stem(track_path)}_labeled.parquet"


def outside_labeled_cache_path(track_path: Path | str) -> Path:
    return CACHE_DIR / "outside_labeled_tracks" / f"{track_cache_stem(track_path)}_outside_labeled.parquet"


def table_cache_path(name: str) -> Path:
    return CACHE_DIR / "tables" / name


def vector_cache_path(name: str) -> Path:
    return CACHE_DIR / "vectors" / name


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return df if columns is None else df[columns]
    return pd.read_parquet(path, columns=columns)


def sample_series(values: pd.Series, max_n: int, *, random_state: int = 0) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) > max_n:
        v = v.sample(max_n, random_state=random_state)
    return v.to_numpy(float)


def load_cached_good_track_presence() -> pd.DataFrame:
    track_presence = read_table(table_cache_path("track_presence.parquet"))
    return track_presence[track_presence["present_frac"] >= MIN_TRACK_PRESENT_FRAC].copy()


def cache_paths_from_presence(
    presence: pd.DataFrame,
    cache_fn,
    *,
    require_exists: bool = True,
) -> list[Path]:
    paths = [cache_fn(Path(row.path)) for row in presence.itertuples(index=False)]
    if require_exists:
        paths = [path for path in paths if path.exists()]
    return paths


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


def plot_speed_histogram(hist: pd.DataFrame) -> None:
    if hist.empty:
        return
    widths = hist["bin_right"].to_numpy(float) - hist["bin_left"].to_numpy(float)
    plt.bar(
        hist["bin_center"].to_numpy(float),
        hist["count"].to_numpy(float),
        width=widths,
        color="0.25",
        align="center",
    )


def sleep_label_params_match(
    path: Path,
    *,
    stationary_threshold_mm_s: float,
    min_sleep_stationary_seconds: float,
    fps: float,
) -> bool:
    if not use_cache(path):
        return False
    try:
        table = read_table(path)
    except Exception:
        return False
    required = {"stationary_threshold_mm_s", "min_sleep_stationary_seconds", "fps"}
    if table.empty or not required.issubset(table.columns):
        return False
    row = table.iloc[0]
    return (
        np.isclose(float(row["stationary_threshold_mm_s"]), float(stationary_threshold_mm_s), equal_nan=True)
        and np.isclose(float(row["min_sleep_stationary_seconds"]), float(min_sleep_stationary_seconds), equal_nan=True)
        and np.isclose(float(row["fps"]), float(fps), equal_nan=True)
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


def behavior_cluster_config_matches(
    path: Path,
    *,
    n_clusters: int,
    worker_inside_colony_frac_threshold: float | None,
) -> bool:
    if not use_cache(path):
        return False
    try:
        cached = read_table(path)
    except Exception:
        return False
    current = behavior_cluster_config_table(
        n_clusters=n_clusters,
        worker_inside_colony_frac_threshold=worker_inside_colony_frac_threshold,
    )
    required = set(current.columns)
    if cached.empty or not required.issubset(cached.columns):
        return False
    cached_row = cached.iloc[0]
    current_row = current.iloc[0]
    return (
        str(cached_row["cluster_method"]) == str(current_row["cluster_method"])
        and int(cached_row["n_behavior_clusters"]) == int(current_row["n_behavior_clusters"])
        and np.isclose(
            float(cached_row["worker_inside_colony_frac_threshold"]),
            float(current_row["worker_inside_colony_frac_threshold"]),
            equal_nan=True,
        )
    )


def colony_boxes_signature(colony_boxes_mm: list[tuple[float, float, float, float]]) -> str:
    return ";".join(
        ",".join(f"{float(value):.9g}" for value in box)
        for box in colony_boxes_mm
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


def behavior_output_config_matches(
    path: Path,
    *,
    stationary_threshold_mm_s: float,
    min_sleep_stationary_seconds: float,
    fps: float,
    colony_boxes_mm: list[tuple[float, float, float, float]],
    n_clusters: int,
    worker_inside_colony_frac_threshold: float | None,
) -> bool:
    if not use_cache(path):
        return False
    try:
        cached = read_table(path)
    except Exception:
        return False
    current = behavior_output_config_table(
        stationary_threshold_mm_s=stationary_threshold_mm_s,
        min_sleep_stationary_seconds=min_sleep_stationary_seconds,
        fps=fps,
        colony_boxes_mm=colony_boxes_mm,
        n_clusters=n_clusters,
        worker_inside_colony_frac_threshold=worker_inside_colony_frac_threshold,
    )
    required = set(current.columns)
    if cached.empty or not required.issubset(cached.columns):
        return False
    cached_row = cached.iloc[0]
    current_row = current.iloc[0]
    return (
        str(cached_row["cluster_method"]) == str(current_row["cluster_method"])
        and int(cached_row["n_behavior_clusters"]) == int(current_row["n_behavior_clusters"])
        and str(cached_row["colony_boxes_signature"]) == str(current_row["colony_boxes_signature"])
        and np.isclose(
            float(cached_row["worker_inside_colony_frac_threshold"]),
            float(current_row["worker_inside_colony_frac_threshold"]),
            equal_nan=True,
        )
        and np.isclose(
            float(cached_row["stationary_threshold_mm_s"]),
            float(current_row["stationary_threshold_mm_s"]),
            equal_nan=True,
        )
        and np.isclose(
            float(cached_row["min_sleep_stationary_seconds"]),
            float(current_row["min_sleep_stationary_seconds"]),
            equal_nan=True,
        )
        and np.isclose(float(cached_row["fps"]), float(current_row["fps"]), equal_nan=True)
    )


#%%
# ------------------------- parquet and track loading -------------------------

@dataclass(frozen=True)
class TrackFileSummary:
    path: Path
    track_id: int
    n_observed_frames: int
    min_frame: int
    max_frame: int
    dataset_num_frames: int | None
    present_frac: float | None


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


def summarize_track_file(path: Path) -> TrackFileSummary:
    cols = parquet_columns(path)
    if "Frame" not in cols:
        raise ValueError(f"{path} has no Frame column")
    frame = pd.to_numeric(pd.read_parquet(path, columns=["Frame"])["Frame"], errors="coerce").dropna()
    if frame.empty:
        tid = track_id_from_path(path)
        return TrackFileSummary(path, -1 if tid is None else tid, 0, -1, -1, parquet_num_frames(path), None)
    tid = track_id_from_path(path)
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
                "path": str(s.path),
                "n_observed_frames": s.n_observed_frames,
                "min_frame": s.min_frame,
                "max_frame": s.max_frame,
                "dataset_num_frames": dataset_num_frames,
                "present_frac": frac,
            }
        )
    return pd.DataFrame(rows).sort_values("TrackID").reset_index(drop=True)


def load_track_position_table(
    path: Path,
    *,
    position_columns: tuple[str, str] = POSITION_COLUMNS,
    bodypoint: int = BODYPOINT_FOR_XY,
) -> pd.DataFrame:
    cols = parquet_columns(path)
    x_col, y_col = position_columns
    missing = {x_col, y_col, "Frame"}.difference(cols)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    read_cols = ["Frame", x_col, y_col]
    for col in ("TrackID", "Bodypoint", "CameraID", "ArucoCam", "SleapCam", "source_file"):
        if col in cols and col not in read_cols:
            read_cols.append(col)
    df = pd.read_parquet(path, columns=read_cols).copy()
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=["Frame", x_col, y_col])
    if df.empty:
        return pd.DataFrame(columns=["Frame", "TrackID", "X_px", "Y_px"])

    if position_columns == ("X", "Y") and "Bodypoint" in df.columns:
        df = df[pd.to_numeric(df["Bodypoint"], errors="coerce") == int(bodypoint)]

    df["Frame"] = df["Frame"].round().astype(np.int64)
    df["X_px"] = df[x_col].astype(float)
    df["Y_px"] = df[y_col].astype(float)
    grouped = df.groupby("Frame", sort=True, as_index=False)
    out = grouped.agg(X_px=("X_px", "mean"), Y_px=("Y_px", "mean"), n_rows=("X_px", "size"))
    if "TrackID" in df.columns:
        out["TrackID"] = grouped["TrackID"].first()["TrackID"].astype(int)
    else:
        tid = track_id_from_path(path)
        out["TrackID"] = -1 if tid is None else tid
    return out.sort_values("Frame", kind="mergesort").reset_index(drop=True)


#%%
# ------------------------- speed, interpolation, sleep labels -------------------------

def interpolate_short_gaps(
    frame_xy: pd.DataFrame,
    *,
    max_gap_frames: int = 5,
) -> pd.DataFrame:
    """Interpolate X/Y only when the whole missing run is <= max_gap_frames."""
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

    for left, right in zip(obs_idx[:-1], obs_idx[1:], strict=False):
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
    """
    Clean speed calculation:
    1. Dense frame index.
    2. Interpolate only complete gaps up to max_interp_gap_frames.
    3. Smooth X/Y inside valid contiguous segments.
    4. Compute speed only between consecutive valid positions.
    """
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
    """Estimate a stationary/walking threshold from a two-state speed distribution."""
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


#%%
# ------------------------- colony use and clustering -------------------------

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
    """
    Infer colony square interiors from high-density occupancy.

    The low-density wall ring is handled implicitly: only high-density bins are
    clustered into colony interiors, so low-density wall/outside bins do not
    expand the boxes except for the explicit padding.
    """
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


def behavior_summary_for_track(speed_df: pd.DataFrame, *, fps: float = FPS) -> dict[str, float | int]:
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

    out = {
        "TrackID": tid,
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
    for box_id in range(int(N_COLONY_BOXES)):
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


def sleep_summary_from_labeled_paths(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        cols = ["TrackID", "Sleep", "Stationary", "SpeedMmPerSec", "ValidSpeed"]
        df = read_table(path, columns=cols)
        if df.empty:
            continue
        tid = int(pd.to_numeric(df["TrackID"], errors="coerce").dropna().iloc[0])
        valid = df["ValidSpeed"].fillna(False)
        rows.append(
            {
                "TrackID": tid,
                "sleep_frac": float(df.loc[valid, "Sleep"].mean()) if valid.any() else np.nan,
                "stationary_frac": float(df.loc[valid, "Stationary"].mean()) if valid.any() else np.nan,
                "median_speed_mm_s": float(df["SpeedMmPerSec"].median(skipna=True)),
                "valid_speed_frames": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("TrackID").reset_index(drop=True)


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


def behavior_summary_from_labeled_paths(paths: list[Path], *, fps: float = FPS) -> pd.DataFrame:
    rows = []
    for path in paths:
        cols = [
            "TrackID",
            "Frame",
            "ValidPosition",
            "ValidSpeed",
            "InsideColony",
            "OutsideColony",
            "ColonyBoxID",
            "Sleep",
            "SpeedMmPerSec",
        ]
        df = read_table(path, columns=cols)
        if not df.empty:
            rows.append(behavior_summary_for_track(df, fps=fps))
    return pd.DataFrame(rows)


def sleep_rhythm_by_cluster_paths(
    labeled_paths: list[Path],
    cluster_summary: pd.DataFrame,
    *,
    bin_seconds: float = RHYTHM_BIN_SECONDS,
) -> pd.DataFrame:
    cluster_cols = cluster_summary[["TrackID", "BehaviorCluster", "BehaviorLabel"]].copy()
    per_track_bins = []
    for path in labeled_paths:
        cols = ["TrackID", "Frame", "TimeS", "Sleep", "Stationary", "SpeedMmPerSec", "ValidSpeed"]
        df = read_table(path, columns=cols)
        if df.empty:
            continue
        df = df.merge(cluster_cols, on="TrackID", how="left")
        df = df[df["ValidSpeed"].fillna(False)].copy()
        if df.empty:
            continue
        df["TimeBinS"] = np.floor(df["TimeS"] / float(bin_seconds)) * float(bin_seconds)
        per_track_bins.append(
            df.groupby(["BehaviorCluster", "BehaviorLabel", "TimeBinS"], as_index=False)
            .agg(
                sleep_sum=("Sleep", "sum"),
                stationary_sum=("Stationary", "sum"),
                speed_median_track=("SpeedMmPerSec", "median"),
                n_frames=("Frame", "size"),
                n_tracks=("TrackID", "nunique"),
            )
        )
    if not per_track_bins:
        return pd.DataFrame()

    bins = pd.concat(per_track_bins, ignore_index=True)
    out = (
        bins.groupby(["BehaviorCluster", "BehaviorLabel", "TimeBinS"], as_index=False)
        .agg(
            sleep_sum=("sleep_sum", "sum"),
            stationary_sum=("stationary_sum", "sum"),
            median_speed_mm_s=("speed_median_track", "median"),
            n_frames=("n_frames", "sum"),
            n_tracks=("n_tracks", "sum"),
        )
        .sort_values(["BehaviorCluster", "TimeBinS"])
        .reset_index(drop=True)
    )
    out["sleep_frac"] = out["sleep_sum"] / out["n_frames"].replace(0, np.nan)
    out["stationary_frac"] = out["stationary_sum"] / out["n_frames"].replace(0, np.nan)
    return out[
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


#%%
# ------------------------- discover and filter good tracks -------------------------

ensure_cache_dirs()

track_presence_path = table_cache_path("track_presence.parquet")
if use_cache(track_presence_path):
    track_presence = read_table(track_presence_path)
    dataset_num_frames = int(track_presence["dataset_num_frames"].max())
    print(f"Loaded cached track presence: {track_presence_path}")
else:
    track_files = list_track_files(PER_TRACK_DIR, SIDE_FILTER)
    print(f"Found {len(track_files)} per-track parquet files in {PER_TRACK_DIR}")
    track_summaries = [summarize_track_file(path) for path in track_files]
    dataset_num_frames = DATASET_NUM_FRAMES or infer_dataset_frame_count(track_summaries)
    track_presence = add_presence_fraction(track_summaries, dataset_num_frames)
    write_table(track_presence, track_presence_path)
    print(f"Wrote {track_presence_path}")

good_track_presence = track_presence[track_presence["present_frac"] >= MIN_TRACK_PRESENT_FRAC].copy()

print(f"Dataset frames: {dataset_num_frames} ({dataset_num_frames / FPS / 3600:.2f} h at {FPS:g} fps)")
print(f"Good tracks: {len(good_track_presence)} / {len(track_presence)} at >= {MIN_TRACK_PRESENT_FRAC:.0%} present")
display(track_presence.sort_values("present_frac", ascending=False).head(20))

plt.figure(figsize=(7, 4))
plt.hist(track_presence["present_frac"].dropna(), bins=30, color="0.25")
plt.axvline(MIN_TRACK_PRESENT_FRAC, color="crimson", lw=2, label=f"{MIN_TRACK_PRESENT_FRAC:.0%}")
plt.xlabel("fraction of dataset with track present")
plt.ylabel("TrackID count")
plt.legend()
plt.tight_layout()


#%%
# ------------------------- compute speed for good tracks -------------------------

speed_vector_path = vector_cache_path("speed_values_mm_s.npy")
speed_preview_path = table_cache_path("speed_preview.parquet")
speed_histogram_path = table_cache_path("speed_histogram.parquet")

missing_speed_rows = [
    row
    for row in good_track_presence.itertuples(index=False)
    if FORCE_RECOMPUTE or not speed_cache_path(Path(row.path)).exists()
]

for row in missing_speed_rows:
    src_path = Path(row.path)
    out_path = speed_cache_path(src_path)
    pos = load_track_position_table(src_path)
    if pos.empty:
        continue
    speed = compute_gap_aware_speed(
        pos,
        fps=FPS,
        mm_per_px=MM_PER_PX,
        max_interp_gap_frames=MAX_INTERP_GAP_FRAMES,
        smooth_sigma_frames=SMOOTH_SIGMA_FRAMES,
        max_reasonable_speed_mm_s=MAX_REASONABLE_SPEED_MM_S,
    )
    write_table(speed, out_path)

speed_cache_paths = cache_paths_from_presence(good_track_presence, speed_cache_path)

if missing_speed_rows or not (use_cache(speed_vector_path) and use_cache(speed_preview_path)):
    speed_samples = []
    speed_preview_rows = []
    for path in speed_cache_paths:
        speed = read_table(path, columns=["TrackID", "Frame", "SpeedMmPerSec", "ValidSpeed"])
        speed_samples.append(sample_series(speed["SpeedMmPerSec"], SPEED_VECTOR_SAMPLE_PER_TRACK))
        if len(speed_preview_rows) < 5:
            speed_preview_rows.append(read_table(path).head())

    speed_values = np.concatenate([v for v in speed_samples if len(v)]) if speed_samples else np.empty(0, dtype=float)
    np.save(speed_vector_path, speed_values)
    speed_preview = pd.concat(speed_preview_rows, ignore_index=True) if speed_preview_rows else pd.DataFrame()
    write_table(speed_preview, speed_preview_path)
    write_table(speed_histogram_table(speed_values), speed_histogram_path)
    print(f"Saved speed vector for thresholding: {speed_vector_path} ({len(speed_values):,} values)")
else:
    speed_values = np.empty(0, dtype=float)
    speed_preview = read_table(speed_preview_path)
    print(f"Found cached speed vector and loaded cached preview: {speed_vector_path}, {speed_preview_path}")

print(f"Cached speed for {len(speed_cache_paths)} tracks in {CACHE_DIR / 'speed_tracks'}")
display(speed_preview)


#%%
# ------------------------- stationary threshold and sleep/wake labeling -------------------------

if "good_track_presence" not in globals():
    good_track_presence = load_cached_good_track_presence()

speed_vector_path = vector_cache_path("speed_values_mm_s.npy")
speed_histogram_path = table_cache_path("speed_histogram.parquet")
threshold_table_path = table_cache_path("stationary_threshold.csv")
sleep_summary_path = table_cache_path("sleep_summary.parquet")

labeled_expected_paths = [
    labeled_cache_path(Path(row.path))
    for row in good_track_presence.itertuples(index=False)
    if speed_cache_path(Path(row.path)).exists()
]
labeled_cache_paths = [path for path in labeled_expected_paths if path.exists()]
all_labeled_cached = bool(labeled_expected_paths) and len(labeled_cache_paths) == len(labeled_expected_paths)

threshold_n_speed_values = np.nan
if STATIONARY_THRESHOLD_MM_S is not None:
    stationary_threshold_mm_s = float(STATIONARY_THRESHOLD_MM_S)
    sleep_label_params_were_current = sleep_label_params_match(
        threshold_table_path,
        stationary_threshold_mm_s=stationary_threshold_mm_s,
        min_sleep_stationary_seconds=MIN_SLEEP_STATIONARY_SECONDS,
        fps=FPS,
    )
elif use_cache(threshold_table_path):
    threshold_table = read_table(threshold_table_path)
    stationary_threshold_mm_s = float(threshold_table["stationary_threshold_mm_s"].iloc[0])
    if "n_speed_values" in threshold_table.columns:
        threshold_n_speed_values = threshold_table["n_speed_values"].iloc[0]
    sleep_label_params_were_current = sleep_label_params_match(
        threshold_table_path,
        stationary_threshold_mm_s=stationary_threshold_mm_s,
        min_sleep_stationary_seconds=MIN_SLEEP_STATIONARY_SECONDS,
        fps=FPS,
    )
    print(f"Loaded cached stationary threshold: {threshold_table_path}")
else:
    if "speed_values" not in globals() or len(speed_values) == 0:
        if speed_vector_path.exists():
            speed_values = np.load(speed_vector_path)
        else:
            speed_parts = []
            for row in good_track_presence.itertuples(index=False):
                path = speed_cache_path(Path(row.path))
                if path.exists():
                    speed_parts.append(sample_series(read_table(path, columns=["SpeedMmPerSec"])["SpeedMmPerSec"], SPEED_VECTOR_SAMPLE_PER_TRACK))
            speed_values = np.concatenate([v for v in speed_parts if len(v)]) if speed_parts else np.empty(0, dtype=float)
            if len(speed_values):
                np.save(speed_vector_path, speed_values)
    stationary_threshold_mm_s = estimate_stationary_threshold(speed_values)
    threshold_n_speed_values = len(speed_values)
    sleep_label_params_were_current = False

if not sleep_label_params_were_current:
    write_table(
        pd.DataFrame(
            [
                {
                    "stationary_threshold_mm_s": stationary_threshold_mm_s,
                    "fps": FPS,
                    "min_sleep_stationary_seconds": MIN_SLEEP_STATIONARY_SECONDS,
                    "n_speed_values": threshold_n_speed_values,
                }
            ]
        ),
        threshold_table_path,
    )

print(f"Stationary threshold: {stationary_threshold_mm_s:.6g} mm/s")

if use_cache(sleep_summary_path) and all_labeled_cached and sleep_label_params_were_current:
    sleep_summary = read_table(sleep_summary_path)
    print(f"Loaded cached sleep summary: {sleep_summary_path}")
else:
    labeled_cache_paths = []
    for row in good_track_presence.itertuples(index=False):
        src_path = Path(row.path)
        speed_path = speed_cache_path(src_path)
        if not speed_path.exists():
            continue
        out_path = labeled_cache_path(src_path)
        if not sleep_label_params_were_current or not use_cache(out_path):
            speed = read_table(speed_path)
            labeled = label_sleep_from_speed(
                speed,
                stationary_threshold_mm_s=stationary_threshold_mm_s,
                min_sleep_stationary_seconds=MIN_SLEEP_STATIONARY_SECONDS,
                fps=FPS,
            )
            write_table(labeled, out_path)
        labeled_cache_paths.append(out_path)
    sleep_summary = sleep_summary_from_labeled_paths(labeled_cache_paths)
    write_table(sleep_summary, sleep_summary_path)

plt.figure(figsize=(7, 4))
if use_cache(speed_histogram_path):
    speed_histogram = read_table(speed_histogram_path)
else:
    if ("speed_values" not in globals() or len(speed_values) == 0) and speed_vector_path.exists():
        speed_values = np.load(speed_vector_path)
    elif "speed_values" not in globals():
        speed_values = np.empty(0, dtype=float)
    speed_histogram = speed_histogram_table(speed_values)
    write_table(speed_histogram, speed_histogram_path)
plot_speed_histogram(speed_histogram)
plt.axvline(stationary_threshold_mm_s, color="crimson", lw=2, label="stationary threshold")
plt.xlabel("speed (mm/s)")
plt.ylabel("frame count")
plt.legend()
plt.tight_layout()

print(f"Cached labeled sleep/wake tracks: {len(labeled_cache_paths)}")
display(sleep_summary.sort_values("sleep_frac", ascending=False).head(20))


#%%
# ------------------------- infer colony boxes and classify outside-colony time -------------------------

if "good_track_presence" not in globals():
    good_track_presence = load_cached_good_track_presence()

if "labeled_cache_paths" not in globals():
    labeled_cache_paths = cache_paths_from_presence(good_track_presence, labeled_cache_path)

position_sample_path = table_cache_path("valid_position_sample.parquet")
if use_cache(position_sample_path):
    valid_positions = read_table(position_sample_path)
    print(f"Loaded cached valid position sample: {position_sample_path}")
else:
    valid_positions = sample_valid_positions_from_paths(
        labeled_cache_paths,
        sample_per_track=POSITION_SAMPLE_PER_TRACK,
    )
    write_table(valid_positions, position_sample_path)

colony_box_path = table_cache_path("colony_boxes.parquet")
if COLONY_BOXES_MM is not None:
    colony_boxes_mm = list(COLONY_BOXES_MM)
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
            for i, box in enumerate(colony_boxes_mm)
        ]
    )
elif use_cache(colony_box_path):
    colony_box_diag = read_table(colony_box_path)
    colony_boxes_mm = [
        (float(row.xmin_mm), float(row.xmax_mm), float(row.ymin_mm), float(row.ymax_mm))
        for row in colony_box_diag.itertuples(index=False)
    ]
    print(f"Loaded cached colony boxes: {colony_box_path}")
else:
    colony_boxes_mm, colony_box_diag = infer_colony_boxes_by_density(
        valid_positions,
        n_boxes=N_COLONY_BOXES,
        bins=COLONY_DENSITY_BINS,
        smooth_sigma_bins=COLONY_DENSITY_SMOOTH_SIGMA_BINS,
        high_quantile=COLONY_DENSITY_HIGH_QUANTILE,
        pad_mm=COLONY_BOX_PAD_MM,
        fallback_central_fraction=COLONY_BOX_CENTRAL_FRACTION,
    )
    if "source" not in colony_box_diag.columns:
        colony_box_diag["source"] = "density"

print("Colony boxes mm:")
for i, box in enumerate(colony_boxes_mm):
    print(f"  box {i}: xmin={box[0]:.3f}, xmax={box[1]:.3f}, ymin={box[2]:.3f}, ymax={box[3]:.3f}")

write_table(colony_box_diag, colony_box_path)
display(colony_box_diag)

if "stationary_threshold_mm_s" not in globals():
    threshold_table = read_table(table_cache_path("stationary_threshold.csv"))
    stationary_threshold_mm_s = float(threshold_table["stationary_threshold_mm_s"].iloc[0])

current_behavior_output_config = behavior_output_config_table(
    stationary_threshold_mm_s=stationary_threshold_mm_s,
    min_sleep_stationary_seconds=MIN_SLEEP_STATIONARY_SECONDS,
    fps=FPS,
    colony_boxes_mm=colony_boxes_mm,
    n_clusters=N_BEHAVIOR_CLUSTERS,
    worker_inside_colony_frac_threshold=WORKER_INSIDE_COLONY_FRAC_THRESHOLD,
)
outside_labeled_output_config = behavior_output_config_table(
    stationary_threshold_mm_s=stationary_threshold_mm_s,
    min_sleep_stationary_seconds=MIN_SLEEP_STATIONARY_SECONDS,
    fps=FPS,
    colony_boxes_mm=colony_boxes_mm,
    n_clusters=0,
    worker_inside_colony_frac_threshold=None,
)
outside_labeled_config_path = table_cache_path("outside_labeled_config.csv")
outside_labeled_config_current = behavior_output_config_matches(
    outside_labeled_config_path,
    stationary_threshold_mm_s=stationary_threshold_mm_s,
    min_sleep_stationary_seconds=MIN_SLEEP_STATIONARY_SECONDS,
    fps=FPS,
    colony_boxes_mm=colony_boxes_mm,
    n_clusters=0,
    worker_inside_colony_frac_threshold=None,
)

outside_labeled_cache_paths: list[Path] = []
for row in good_track_presence.itertuples(index=False):
    src_path = Path(row.path)
    labeled_path = labeled_cache_path(src_path)
    if not labeled_path.exists():
        continue
    out_path = outside_labeled_cache_path(src_path)
    if not outside_labeled_config_current or not use_cache(out_path):
        labeled = read_table(labeled_path)
        labeled = add_outside_colony_label(labeled, colony_boxes_mm=colony_boxes_mm)
        write_table(labeled, out_path)
    outside_labeled_cache_paths.append(out_path)
if not outside_labeled_config_current:
    write_table(outside_labeled_output_config, outside_labeled_config_path)

behavior_summary_path = table_cache_path("behavior_summary.parquet")
behavior_cluster_config_path = table_cache_path("behavior_cluster_config.csv")
behavior_cluster_config_current = behavior_cluster_config_matches(
    behavior_cluster_config_path,
    n_clusters=N_BEHAVIOR_CLUSTERS,
    worker_inside_colony_frac_threshold=WORKER_INSIDE_COLONY_FRAC_THRESHOLD,
)
behavior_summary_config_path = table_cache_path("behavior_summary_config.csv")
behavior_summary_config_current = behavior_output_config_matches(
    behavior_summary_config_path,
    stationary_threshold_mm_s=stationary_threshold_mm_s,
    min_sleep_stationary_seconds=MIN_SLEEP_STATIONARY_SECONDS,
    fps=FPS,
    colony_boxes_mm=colony_boxes_mm,
    n_clusters=N_BEHAVIOR_CLUSTERS,
    worker_inside_colony_frac_threshold=WORKER_INSIDE_COLONY_FRAC_THRESHOLD,
)
if use_cache(behavior_summary_path) and behavior_summary_config_current:
    behavior_summary = read_table(behavior_summary_path)
    print(f"Loaded cached behavior summary: {behavior_summary_path}")
    missing_cluster_cols = {"BehaviorCluster", "BehaviorLabel"}.difference(behavior_summary.columns)
    if missing_cluster_cols or not behavior_cluster_config_current:
        behavior_summary = cluster_by_colony_use(
            behavior_summary,
            n_clusters=N_BEHAVIOR_CLUSTERS,
            worker_inside_colony_frac_threshold=WORKER_INSIDE_COLONY_FRAC_THRESHOLD,
        )
        write_table(behavior_summary, behavior_summary_path)
        write_table(current_behavior_output_config, behavior_summary_config_path)
        write_table(current_behavior_output_config, behavior_cluster_config_path)
else:
    behavior_summary = behavior_summary_from_labeled_paths(outside_labeled_cache_paths, fps=FPS)
    behavior_summary = cluster_by_colony_use(
        behavior_summary,
        n_clusters=N_BEHAVIOR_CLUSTERS,
        worker_inside_colony_frac_threshold=WORKER_INSIDE_COLONY_FRAC_THRESHOLD,
    )
    write_table(behavior_summary, behavior_summary_path)
    write_table(current_behavior_output_config, behavior_summary_config_path)
    write_table(current_behavior_output_config, behavior_cluster_config_path)

display(behavior_summary.sort_values("outside_frac", ascending=False))

sample_positions = valid_positions.sample(min(len(valid_positions), 200_000), random_state=0) if len(valid_positions) else valid_positions
plt.figure(figsize=(7, 7))
plt.scatter(sample_positions["X_mm"], sample_positions["Y_mm"], s=1, alpha=0.08, color="0.1", rasterized=True)
for i, (xmin, xmax, ymin, ymax) in enumerate(colony_boxes_mm):
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], lw=2, label=f"colony box {i}")
    plt.text(xmin, ymin, str(i), color="crimson", fontsize=12, weight="bold")
plt.gca().set_aspect("equal", adjustable="datalim")
plt.gca().invert_yaxis()
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Inferred colony boxes")
plt.legend()
plt.tight_layout()


#%%
# ------------------------- cluster sleep amount and sleep rhythm -------------------------

if "good_track_presence" not in globals():
    good_track_presence = load_cached_good_track_presence()

if "behavior_summary" not in globals():
    behavior_summary = read_table(table_cache_path("behavior_summary.parquet"))

if "outside_labeled_cache_paths" not in globals():
    outside_labeled_cache_paths = cache_paths_from_presence(good_track_presence, outside_labeled_cache_path)

if "stationary_threshold_mm_s" not in globals():
    threshold_table = read_table(table_cache_path("stationary_threshold.csv"))
    stationary_threshold_mm_s = float(threshold_table["stationary_threshold_mm_s"].iloc[0])

if "colony_boxes_mm" not in globals():
    colony_box_diag = read_table(table_cache_path("colony_boxes.parquet"))
    colony_boxes_mm = [
        (float(row.xmin_mm), float(row.xmax_mm), float(row.ymin_mm), float(row.ymax_mm))
        for row in colony_box_diag.itertuples(index=False)
    ]

if "current_behavior_output_config" not in globals():
    current_behavior_output_config = behavior_output_config_table(
        stationary_threshold_mm_s=stationary_threshold_mm_s,
        min_sleep_stationary_seconds=MIN_SLEEP_STATIONARY_SECONDS,
        fps=FPS,
        colony_boxes_mm=colony_boxes_mm,
        n_clusters=N_BEHAVIOR_CLUSTERS,
        worker_inside_colony_frac_threshold=WORKER_INSIDE_COLONY_FRAC_THRESHOLD,
    )

cluster_sleep_summary_path = table_cache_path("cluster_sleep_summary.parquet")
cluster_sleep_summary_config_path = table_cache_path("cluster_sleep_summary_config.csv")
if (
    use_cache(cluster_sleep_summary_path)
    and behavior_output_config_matches(
        cluster_sleep_summary_config_path,
        stationary_threshold_mm_s=stationary_threshold_mm_s,
        min_sleep_stationary_seconds=MIN_SLEEP_STATIONARY_SECONDS,
        fps=FPS,
        colony_boxes_mm=colony_boxes_mm,
        n_clusters=N_BEHAVIOR_CLUSTERS,
        worker_inside_colony_frac_threshold=WORKER_INSIDE_COLONY_FRAC_THRESHOLD,
    )
):
    cluster_sleep_summary = read_table(cluster_sleep_summary_path)
    print(f"Loaded cached cluster sleep summary: {cluster_sleep_summary_path}")
else:
    cluster_agg = {
        "n_tracks": ("TrackID", "nunique"),
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
    write_table(cluster_sleep_summary, cluster_sleep_summary_path)
    write_table(current_behavior_output_config, cluster_sleep_summary_config_path)
display(cluster_sleep_summary)

sleep_rhythm_path = table_cache_path("sleep_rhythm.parquet")
sleep_rhythm_config_path = table_cache_path("sleep_rhythm_config.csv")
if (
    use_cache(sleep_rhythm_path)
    and behavior_output_config_matches(
        sleep_rhythm_config_path,
        stationary_threshold_mm_s=stationary_threshold_mm_s,
        min_sleep_stationary_seconds=MIN_SLEEP_STATIONARY_SECONDS,
        fps=FPS,
        colony_boxes_mm=colony_boxes_mm,
        n_clusters=N_BEHAVIOR_CLUSTERS,
        worker_inside_colony_frac_threshold=WORKER_INSIDE_COLONY_FRAC_THRESHOLD,
    )
):
    sleep_rhythm = read_table(sleep_rhythm_path)
    print(f"Loaded cached sleep rhythm: {sleep_rhythm_path}")
else:
    sleep_rhythm = sleep_rhythm_by_cluster_paths(
        outside_labeled_cache_paths,
        behavior_summary,
        bin_seconds=RHYTHM_BIN_SECONDS,
    )
    write_table(sleep_rhythm, sleep_rhythm_path)
    write_table(current_behavior_output_config, sleep_rhythm_config_path)
display(sleep_rhythm.head())

plt.figure(figsize=(10, 4))
for label, group in sleep_rhythm.groupby("BehaviorLabel", sort=False):
    plt.plot(group["TimeBinS"] / 3600.0, group["sleep_frac"], marker="o", ms=3, lw=1.5, label=label)
plt.xlabel("dataset time (h)")
plt.ylabel("sleep fraction")
plt.title("Sleep rhythm by colony-use cluster")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8, 4))
for label, group in behavior_summary.groupby("BehaviorLabel", sort=False):
    plt.scatter(group["outside_frac"], group["sleep_frac"], label=label, alpha=0.8)
plt.xlabel("fraction of valid time outside colony box")
plt.ylabel("sleep fraction")
plt.legend()
plt.tight_layout()


#%%
# ------------------------- optional saves -------------------------

if OUTPUT_DIR is not None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    track_presence.to_csv(OUTPUT_DIR / "track_presence.csv", index=False)
    behavior_summary.to_csv(OUTPUT_DIR / "behavior_summary.csv", index=False)
    cluster_sleep_summary.to_csv(OUTPUT_DIR / "cluster_sleep_summary.csv", index=False)
    sleep_rhythm.to_csv(OUTPUT_DIR / "sleep_rhythm.csv", index=False)
    if "colony_box_diag" in globals():
        colony_box_diag.to_csv(OUTPUT_DIR / "colony_boxes.csv", index=False)
    manifest_sources = {
        "speed_cache_path": [str(p) for p in speed_cache_paths] if "speed_cache_paths" in globals() else [],
        "labeled_cache_path": [str(p) for p in labeled_cache_paths] if "labeled_cache_paths" in globals() else [],
        "outside_labeled_cache_path": (
            [str(p) for p in outside_labeled_cache_paths]
            if "outside_labeled_cache_paths" in globals()
            else []
        ),
    }
    max_manifest_len = max((len(v) for v in manifest_sources.values()), default=0)
    manifest = pd.DataFrame(
        {
            key: values + [""] * (max_manifest_len - len(values))
            for key, values in manifest_sources.items()
        }
    )
    manifest.to_csv(OUTPUT_DIR / "per_track_cache_manifest.csv", index=False)
    print(f"Wrote outputs to {OUTPUT_DIR}")

# %%
