#!/usr/bin/env python3
"""Compute a normalized 2D grid occupancy histogram for one stitched track."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEFAULT_MM_PER_PX = 0.016
DEFAULT_X_SPLIT_PX = 1740.0
DEFAULT_X_MIN_PX = -6884.0
DEFAULT_X_MAX_PX = 10831.0
DEFAULT_Y_MIN_PX = -5280.0
DEFAULT_Y_MAX_PX = 8097.0
DEFAULT_GRID_SIZE_MM = 1.0
DEFAULT_GRID_PAD_MM = 10.0


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

    return (
        df.groupby("Frame", sort=True, as_index=False)
        .agg(X=("X", "mean"), Y=("Y", "mean"))
        .sort_values("Frame", kind="mergesort")
        .reset_index(drop=True)
    )


def track_id_from_name(path: Path) -> int | None:
    match = re.search(r"TrackID_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def side_from_name(path: Path) -> str | None:
    stem = path.stem.lower()
    if stem.endswith("_left") or "_left_" in stem:
        return "left"
    if stem.endswith("_right") or "_right_" in stem:
        return "right"
    return None


def infer_side(path: Path, position: pd.DataFrame, x_split_px: float) -> str:
    side = side_from_name(path)
    if side is not None:
        return side

    median_x = float(np.nanmedian(position["X"].to_numpy(np.float64)))
    return "left" if median_x < float(x_split_px) else "right"


def fixed_width_edges(max_value: float, bin_size: float) -> np.ndarray:
    if max_value <= 0:
        raise ValueError(f"Grid max must be positive, got {max_value}")
    if bin_size <= 0:
        raise ValueError(f"Grid size must be positive, got {bin_size}")
    n_bins = max(1, int(np.ceil(float(max_value) / float(bin_size))))
    return (np.arange(n_bins + 1, dtype=np.float64) * float(bin_size)).astype(np.float32)


def padded_edges(max_value: float, bin_size: float, pad: float) -> np.ndarray:
    if max_value <= 0:
        raise ValueError(f"Grid max must be positive, got {max_value}")
    if bin_size <= 0:
        raise ValueError(f"Grid size must be positive, got {bin_size}")
    if pad < 0:
        raise ValueError(f"Grid pad must be nonnegative, got {pad}")
    start = -float(pad)
    stop = float(max_value) + float(pad)
    n_bins = max(1, int(np.ceil((stop - start) / float(bin_size))))
    return (start + np.arange(n_bins + 1, dtype=np.float64) * float(bin_size)).astype(np.float32)


def side_bounds_px(
    side: str,
    *,
    x_min_px: float,
    x_split_px: float,
    x_max_px: float,
) -> tuple[float, float]:
    if not (x_min_px < x_split_px < x_max_px):
        raise ValueError(
            f"Expected x_min_px < x_split_px < x_max_px, got "
            f"{x_min_px}, {x_split_px}, {x_max_px}"
        )
    if side == "left":
        return float(x_min_px), float(x_split_px)
    if side == "right":
        return float(x_split_px), float(x_max_px)
    raise ValueError(f"side must be left or right, got {side!r}")


def compute_grid_occupancy(
    position: pd.DataFrame,
    *,
    side: str,
    mm_per_px: float,
    grid_size_mm: float,
    x_min_px: float,
    x_split_px: float,
    x_max_px: float,
    y_min_px: float,
    y_max_px: float,
    input_x_is_side_local: bool,
    same_shape_sides: bool,
    grid_pad_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    side_x0_px, side_x1_px = side_bounds_px(
        side,
        x_min_px=float(x_min_px),
        x_split_px=float(x_split_px),
        x_max_px=float(x_max_px),
    )
    if not y_min_px < y_max_px:
        raise ValueError(f"Expected y_min_px < y_max_px, got {y_min_px}, {y_max_px}")

    actual_side_width_px = side_x1_px - side_x0_px
    grid_width_px = actual_side_width_px
    if same_shape_sides:
        grid_width_px = max(float(x_split_px) - float(x_min_px), float(x_max_px) - float(x_split_px))
    side_height_px = float(y_max_px) - float(y_min_px)
    grid_width_mm = grid_width_px * float(mm_per_px)
    grid_height_mm = side_height_px * float(mm_per_px)
    x_edges_mm = padded_edges(grid_width_mm, float(grid_size_mm), float(grid_pad_mm))
    y_edges_mm = padded_edges(grid_height_mm, float(grid_size_mm), float(grid_pad_mm))

    x_px = position["X"].to_numpy(np.float64)
    y_px = position["Y"].to_numpy(np.float64)
    valid = np.isfinite(x_px) & np.isfinite(y_px)
    n_detected = int(valid.sum())
    if n_detected == 0:
        raise ValueError("No detected frames with finite X/Y")

    if input_x_is_side_local:
        x_local_px = x_px
        input_x_origin_px = 0.0
    else:
        x_local_px = x_px - side_x0_px
        input_x_origin_px = side_x0_px
    y_local_px = y_px - float(y_min_px)

    x_mm = x_local_px * float(mm_per_px)
    y_mm = y_local_px * float(mm_per_px)
    in_grid = (
        valid
        & (x_mm >= float(x_edges_mm[0]))
        & (x_mm <= float(x_edges_mm[-1]))
        & (y_mm >= float(y_edges_mm[0]))
        & (y_mm <= float(y_edges_mm[-1]))
    )

    counts, _, _ = np.histogram2d(
        y_mm[in_grid],
        x_mm[in_grid],
        bins=[y_edges_mm.astype(np.float64), x_edges_mm.astype(np.float64)],
    )
    occupancy = (counts / float(n_detected)).astype(np.float32)

    stats: dict[str, int | float] = {
        "n_detected_frames": n_detected,
        "n_in_grid_frames": int(in_grid.sum()),
        "n_out_of_grid_detected_frames": int(n_detected - int(in_grid.sum())),
        "occupancy_sum": float(np.sum(occupancy, dtype=np.float64)),
        "side_x0_px": float(side_x0_px),
        "side_x1_px": float(side_x1_px),
        "actual_side_width_px": float(actual_side_width_px),
        "grid_width_px": float(grid_width_px),
        "grid_height_px": float(side_height_px),
        "grid_width_mm": float(grid_width_mm),
        "grid_height_mm": float(grid_height_mm),
        "grid_pad_mm": float(grid_pad_mm),
        "grid_pad_px": float(grid_pad_mm) / float(mm_per_px),
        "grid_x_min_mm": float(x_edges_mm[0]),
        "grid_x_max_mm": float(x_edges_mm[-1]),
        "grid_y_min_mm": float(y_edges_mm[0]),
        "grid_y_max_mm": float(y_edges_mm[-1]),
        "input_x_origin_px": float(input_x_origin_px),
        "y_origin_px": float(y_min_px),
    }
    return occupancy, x_edges_mm, y_edges_mm, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=None, help="Input per-track parquet. Defaults to $TRACK_PATH.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to $TASK_OUTPUT_DIR.")
    parser.add_argument("--mm_per_px", type=float, default=DEFAULT_MM_PER_PX)
    parser.add_argument("--grid_size_mm", type=float, default=DEFAULT_GRID_SIZE_MM)
    parser.add_argument("--grid_pad_mm", type=float, default=DEFAULT_GRID_PAD_MM)
    parser.add_argument("--x_split_px", type=float, default=DEFAULT_X_SPLIT_PX)
    parser.add_argument("--x_min_px", type=float, default=DEFAULT_X_MIN_PX)
    parser.add_argument("--x_max_px", type=float, default=DEFAULT_X_MAX_PX)
    parser.add_argument("--y_min_px", type=float, default=DEFAULT_Y_MIN_PX)
    parser.add_argument("--y_max_px", type=float, default=DEFAULT_Y_MAX_PX)
    parser.add_argument(
        "--same_shape_sides",
        dest="same_shape_sides",
        action="store_true",
        default=False,
        help="Pad the smaller side to the larger side width so left/right histograms have matching shapes.",
    )
    parser.add_argument(
        "--separate_side_widths",
        dest="same_shape_sides",
        action="store_false",
        help="Use each side's actual width, allowing left/right histogram shapes to differ. This is the default.",
    )
    parser.add_argument(
        "--input_x_is_side_local",
        action="store_true",
        help="Use if right-side TrackX has already been shifted to start at 0.",
    )
    parser.add_argument("--side", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--frame_col", default="Frame")
    parser.add_argument("--x_col", default="TrackX")
    parser.add_argument("--y_col", default="TrackY")
    parser.add_argument("--bodypoint", type=int, default=0)
    parser.add_argument("--fail_if_no_in_grid", action="store_true")
    args = parser.parse_args()

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
    side = infer_side(track, xy, float(args.x_split_px)) if args.side == "auto" else args.side
    x_min_px = float(args.x_min_px)
    x_max_px = float(args.x_max_px)
    y_min_px = float(args.y_min_px)
    y_max_px = float(args.y_max_px)

    occupancy, x_edges_mm, y_edges_mm, stats = compute_grid_occupancy(
        xy,
        side=side,
        mm_per_px=float(args.mm_per_px),
        grid_size_mm=float(args.grid_size_mm),
        x_min_px=x_min_px,
        x_split_px=float(args.x_split_px),
        x_max_px=x_max_px,
        y_min_px=y_min_px,
        y_max_px=y_max_px,
        input_x_is_side_local=bool(args.input_x_is_side_local),
        same_shape_sides=bool(args.same_shape_sides),
        grid_pad_mm=float(args.grid_pad_mm),
    )

    if int(stats["n_in_grid_frames"]) == 0:
        message = (
            f"WARNING: no detected frames for {track.name} fell inside the {side} grid. "
            "Check x/y bounds, mm_per_px, and whether TrackX is already side-local."
        )
        print(message)
        if args.fail_if_no_in_grid:
            raise ValueError(message)

    occupancy_path = out_dir / "grid_occupancy_f4.npy"
    x_edges_path = out_dir / "grid_x_edges_mm.npy"
    y_edges_path = out_dir / "grid_y_edges_mm.npy"
    np.save(occupancy_path, occupancy)
    np.save(x_edges_path, x_edges_mm)
    np.save(y_edges_path, y_edges_mm)

    metadata = {
        "track_path": str(track),
        "track_name": track.name,
        "track_id": track_id_from_name(track),
        "side": side,
        "occupancy_path": str(occupancy_path),
        "x_edges_mm_path": str(x_edges_path),
        "y_edges_mm_path": str(y_edges_path),
        "occupancy_units": "fraction_of_detected_frames",
        "normalization": "bin_count / n_detected_frames",
        "histogram_shape_yx": [int(occupancy.shape[0]), int(occupancy.shape[1])],
        "frame_min": int(xy["Frame"].min()),
        "frame_max": int(xy["Frame"].max()),
        "n_observed_frames": int(len(xy)),
        "mm_per_px": float(args.mm_per_px),
        "grid_size_mm": float(args.grid_size_mm),
        "grid_pad_mm": float(args.grid_pad_mm),
        "bounds_source": "explicit_args",
        "x_split_px": float(args.x_split_px),
        "x_min_px": float(x_min_px),
        "x_max_px": float(x_max_px),
        "y_min_px": float(y_min_px),
        "y_max_px": float(y_max_px),
        "same_shape_sides": bool(args.same_shape_sides),
        "input_x_is_side_local": bool(args.input_x_is_side_local),
        "x_col": args.x_col,
        "y_col": args.y_col,
        "bodypoint_filter": int(args.bodypoint),
        "dtype": str(occupancy.dtype),
        **stats,
    }
    (out_dir / "grid_occupancy_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Wrote {occupancy_path} shape={occupancy.shape} "
        f"sum={float(np.sum(occupancy, dtype=np.float64)):.6f}"
    )


if __name__ == "__main__":
    main()
