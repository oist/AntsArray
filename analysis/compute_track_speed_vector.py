#!/usr/bin/env python3
"""Compute a compact gap-aware speed vector for one stitched track parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value.lower() in {"none", "nan", "off"}:
        return None
    return float(value)


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema.names
    except Exception:
        return list(pd.read_parquet(path).columns)


def load_track_xy(
    path: Path,
    *,
    frame_col: str,
    x_col: str,
    y_col: str,
    bodypoint: int,
) -> pd.DataFrame:
    cols = set(parquet_columns(path))
    missing = {frame_col, x_col, y_col}.difference(cols)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    read_cols = [frame_col, x_col, y_col]
    if "Bodypoint" in cols:
        read_cols.append("Bodypoint")
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

    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[frame_col, x_col, y_col])
    if df.empty:
        raise ValueError(f"{path.name} has no finite {frame_col}/{x_col}/{y_col} rows")

    df["Frame"] = df[frame_col].round().astype(np.int64)
    df["X"] = df[x_col].astype(float)
    df["Y"] = df[y_col].astype(float)

    # TrackX/TrackY are repeated across bodypoint rows in stitched files. Mean
    # collapses exact duplicates and is stable if there is tiny numeric jitter.
    return (
        df.groupby("Frame", sort=True, as_index=False)
        .agg(X=("X", "mean"), Y=("Y", "mean"))
        .sort_values("Frame", kind="mergesort")
        .reset_index(drop=True)
    )


def dense_xy(position: pd.DataFrame) -> tuple[int, int, np.ndarray, np.ndarray]:
    frame = position["Frame"].to_numpy(np.int64)
    frame_min = int(frame.min())
    frame_max = int(frame.max())
    n_frames = frame_max - frame_min + 1
    x = np.full(n_frames, np.nan, dtype=np.float64)
    y = np.full(n_frames, np.nan, dtype=np.float64)
    idx = frame - frame_min
    x[idx] = position["X"].to_numpy(np.float64)
    y[idx] = position["Y"].to_numpy(np.float64)
    return frame_min, frame_max, x, y


def interpolate_short_gaps(x: np.ndarray, y: np.ndarray, max_gap_frames: int) -> tuple[np.ndarray, np.ndarray]:
    if max_gap_frames <= 0:
        return x, y
    out_x = x.copy()
    out_y = y.copy()
    observed = np.isfinite(out_x) & np.isfinite(out_y)
    obs_idx = np.flatnonzero(observed)
    if len(obs_idx) < 2:
        return out_x, out_y

    gaps = np.diff(obs_idx) - 1
    short_gap_pos = np.flatnonzero((gaps > 0) & (gaps <= int(max_gap_frames)))
    for pos in short_gap_pos:
        left = int(obs_idx[pos])
        right = int(obs_idx[pos + 1])
        gap = right - left - 1
        frac = np.arange(1, gap + 1, dtype=np.float64) / float(gap + 1)
        fill = np.arange(left + 1, right)
        out_x[fill] = out_x[left] + frac * (out_x[right] - out_x[left])
        out_y[fill] = out_y[left] + frac * (out_y[right] - out_y[left])
    return out_x, out_y


def moving_average(values: np.ndarray, sigma_frames: float) -> np.ndarray:
    window = int(max(3, round(6 * float(sigma_frames) + 1)))
    if window % 2 == 0:
        window += 1
    return (
        pd.Series(values)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(np.float64)
    )


def smooth_valid_segments(values: np.ndarray, valid: np.ndarray, sigma_frames: float) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    if sigma_frames <= 0 or not valid.any():
        out[valid] = values[valid]
        return out

    try:
        from scipy.ndimage import gaussian_filter1d
    except Exception:
        gaussian_filter1d = None

    valid_idx = np.flatnonzero(valid)
    split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
    for segment in np.split(valid_idx, split_points):
        if len(segment) == 0:
            continue
        segment_values = values[segment]
        if len(segment) < 3:
            out[segment] = segment_values
        elif gaussian_filter1d is not None:
            out[segment] = gaussian_filter1d(segment_values, sigma=float(sigma_frames), mode="nearest")
        else:
            out[segment] = moving_average(segment_values, sigma_frames)
    return out


def speed_vector(
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    fps: float,
    mm_per_px: float,
    max_interp_gap_frames: int,
    smooth_sigma_frames: float,
    max_speed_mm_s: float | None,
) -> np.ndarray:
    x_filled, y_filled = interpolate_short_gaps(x_px, y_px, max_interp_gap_frames)
    valid = np.isfinite(x_filled) & np.isfinite(y_filled)
    x_smooth = smooth_valid_segments(x_filled, valid, smooth_sigma_frames)
    y_smooth = smooth_valid_segments(y_filled, valid, smooth_sigma_frames)

    speed = np.full(len(x_smooth), np.nan, dtype=np.float32)
    if len(speed) < 2:
        return speed

    step_valid = valid[1:] & valid[:-1]
    dx = np.diff(x_smooth)
    dy = np.diff(y_smooth)
    speed_step = np.sqrt(dx * dx + dy * dy) * float(mm_per_px) * float(fps)
    speed_tail = speed[1:]
    speed_tail[step_valid] = speed_step[step_valid].astype(np.float32)

    if max_speed_mm_s is not None:
        speed[speed > float(max_speed_mm_s)] = np.nan
    return speed


def track_id_from_name(path: Path) -> int | None:
    match = re.search(r"TrackID_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=None, help="Input per-track parquet. Defaults to $TRACK_PATH.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to $TASK_OUTPUT_DIR.")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--mm_per_px", type=float, default=0.016)
    parser.add_argument("--frame_col", default="Frame")
    parser.add_argument("--x_col", default="TrackX")
    parser.add_argument("--y_col", default="TrackY")
    parser.add_argument("--bodypoint", type=int, default=0)
    parser.add_argument("--max_interp_gap_frames", type=int, default=5)
    parser.add_argument("--smooth_sigma_frames", type=float, default=2.0)
    parser.add_argument("--max_speed_mm_s", default="5.0", help="Set to none/off to keep all speeds.")
    args = parser.parse_args()

    import os

    track = args.track or Path(os.environ["TRACK_PATH"])
    out_dir = args.out or Path(os.environ["TASK_OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    xy = load_track_xy(
        track,
        frame_col=args.frame_col,
        x_col=args.x_col,
        y_col=args.y_col,
        bodypoint=int(args.bodypoint),
    )
    frame_min, frame_max, x_px, y_px = dense_xy(xy)
    speed = speed_vector(
        x_px,
        y_px,
        fps=float(args.fps),
        mm_per_px=float(args.mm_per_px),
        max_interp_gap_frames=int(args.max_interp_gap_frames),
        smooth_sigma_frames=float(args.smooth_sigma_frames),
        max_speed_mm_s=parse_optional_float(args.max_speed_mm_s),
    )

    speed_path = out_dir / "speed_mm_s.npy"
    np.save(speed_path, speed)

    finite_speed = np.isfinite(speed)
    metadata = {
        "track_path": str(track),
        "track_name": track.name,
        "track_id": track_id_from_name(track),
        "speed_path": str(speed_path),
        "speed_units": "mm/s",
        "frame_min": frame_min,
        "frame_max": frame_max,
        "n_frames": int(len(speed)),
        "n_observed_frames": int(len(xy)),
        "n_valid_speed_frames": int(finite_speed.sum()),
        "fps": float(args.fps),
        "mm_per_px": float(args.mm_per_px),
        "x_col": args.x_col,
        "y_col": args.y_col,
        "bodypoint_filter": int(args.bodypoint),
        "max_interp_gap_frames": int(args.max_interp_gap_frames),
        "smooth_sigma_frames": float(args.smooth_sigma_frames),
        "max_speed_mm_s": parse_optional_float(args.max_speed_mm_s),
        "dtype": str(speed.dtype),
    }
    (out_dir / "speed_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {speed_path} ({len(speed)} frames)")


if __name__ == "__main__":
    main()
