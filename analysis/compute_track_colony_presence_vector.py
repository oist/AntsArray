#!/usr/bin/env python3
"""Compute a compact per-frame inside/outside-colony vector for one track."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd


COLONY_BOXES_MM: list[tuple[float, float, float, float]] = [
    (-86, -32, -63, -8),
    (93, 149, -63, -8),
]


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


def colony_presence_vector(
    position: pd.DataFrame,
    *,
    mm_per_px: float,
    colony_boxes_mm: list[tuple[float, float, float, float]],
) -> tuple[int, int, np.ndarray]:
    frame = position["Frame"].to_numpy(np.int64)
    frame_min = int(frame.min())
    frame_max = int(frame.max())
    presence = np.full(frame_max - frame_min + 1, -1, dtype=np.int8)

    x_mm = position["X"].to_numpy(np.float64) * float(mm_per_px)
    y_mm = position["Y"].to_numpy(np.float64) * float(mm_per_px)
    valid = np.isfinite(x_mm) & np.isfinite(y_mm)
    inside_any = np.zeros(len(position), dtype=bool)
    for xmin, xmax, ymin, ymax in colony_boxes_mm:
        inside_any |= valid & (x_mm >= xmin) & (x_mm <= xmax) & (y_mm >= ymin) & (y_mm <= ymax)

    idx = frame - frame_min
    presence[idx[valid]] = inside_any[valid].astype(np.int8)
    return frame_min, frame_max, presence


def parse_colony_boxes(value: str | None) -> list[tuple[float, float, float, float]]:
    if value is None or value.strip() == "":
        return list(COLONY_BOXES_MM)

    boxes = []
    for raw_box in value.split(";"):
        raw_box = raw_box.strip()
        if not raw_box:
            continue
        parts = [float(part.strip()) for part in raw_box.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Each colony box must be xmin,xmax,ymin,ymax: {raw_box!r}")
        xmin, xmax, ymin, ymax = parts
        boxes.append((xmin, xmax, ymin, ymax))
    if not boxes:
        raise ValueError("At least one colony box is required")
    return boxes


def track_id_from_name(path: Path) -> int | None:
    match = re.search(r"TrackID_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=None, help="Input per-track parquet. Defaults to $TRACK_PATH.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to $TASK_OUTPUT_DIR.")
    parser.add_argument("--mm_per_px", type=float, default=0.016)
    parser.add_argument("--frame_col", default="Frame")
    parser.add_argument("--x_col", default="TrackX")
    parser.add_argument("--y_col", default="TrackY")
    parser.add_argument("--bodypoint", type=int, default=0)
    parser.add_argument(
        "--colony_boxes_mm",
        default=None,
        help="Semicolon-separated xmin,xmax,ymin,ymax boxes in mm. Defaults to built-in boxes.",
    )
    args = parser.parse_args()

    track = args.track or Path(os.environ["TRACK_PATH"])
    out_dir = args.out or Path(os.environ["TASK_OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    colony_boxes_mm = parse_colony_boxes(args.colony_boxes_mm)
    xy = load_track_xy(
        track,
        frame_col=args.frame_col,
        x_col=args.x_col,
        y_col=args.y_col,
        bodypoint=int(args.bodypoint),
    )
    frame_min, frame_max, presence = colony_presence_vector(
        xy,
        mm_per_px=float(args.mm_per_px),
        colony_boxes_mm=colony_boxes_mm,
    )

    presence_path = out_dir / "colony_presence_i1.npy"
    np.save(presence_path, presence)

    valid = presence >= 0
    inside = presence == 1
    outside = presence == 0
    metadata = {
        "track_path": str(track),
        "track_name": track.name,
        "track_id": track_id_from_name(track),
        "presence_path": str(presence_path),
        "presence_values": {"missing_xy": -1, "outside_colony": 0, "inside_colony": 1},
        "frame_min": frame_min,
        "frame_max": frame_max,
        "n_frames": int(len(presence)),
        "n_observed_frames": int(len(xy)),
        "n_valid_position_frames": int(valid.sum()),
        "n_inside_colony_frames": int(inside.sum()),
        "n_outside_colony_frames": int(outside.sum()),
        "inside_colony_frac_valid": float(inside.sum() / valid.sum()) if valid.any() else None,
        "mm_per_px": float(args.mm_per_px),
        "x_col": args.x_col,
        "y_col": args.y_col,
        "bodypoint_filter": int(args.bodypoint),
        "colony_boxes_mm": [list(box) for box in colony_boxes_mm],
        "dtype": str(presence.dtype),
    }
    (out_dir / "colony_presence_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {presence_path} ({len(presence)} frames)")


if __name__ == "__main__":
    main()
