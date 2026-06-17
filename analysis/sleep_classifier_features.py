#!/usr/bin/env python3
"""Feature helpers for supervised ant sleep/wake classification."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


ANGLE_PAIRS = {
    "angle_in_left_deg": ((5, 4), (5, 6), True),
    "angle_out_left_deg": ((4, 1), (4, 5), False),
    "angle_in_right_deg": ((8, 7), (8, 9), False),
    "angle_out_right_deg": ((7, 1), (7, 8), True),
}

DISTANCE_PAIRS = [
    (1, 4),
    (1, 7),
    (4, 5),
    (5, 6),
    (7, 8),
    (8, 9),
    (4, 7),
    (5, 8),
]


def track_id_from_name(path: Path | str) -> int | None:
    match = re.search(r"TrackID_(\d+)", Path(path).stem)
    return int(match.group(1)) if match else None


def side_from_name(path: Path | str) -> str | None:
    name = Path(path).stem
    if name.endswith("_left"):
        return "left"
    if name.endswith("_right"):
        return "right"
    return None


def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    ang1 = np.arctan2(v1[:, 1], v1[:, 0])
    ang2 = np.arctan2(v2[:, 1], v2[:, 0])
    ddeg = (np.degrees(ang2 - ang1) + 360.0) % 360.0
    l1 = np.linalg.norm(v1, axis=1)
    l2 = np.linalg.norm(v2, axis=1)
    bad = (l1 == 0) | (l2 == 0) | ~np.isfinite(ang1) | ~np.isfinite(ang2)
    ddeg[bad] = np.nan
    return ddeg


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema.names
    except Exception:
        return list(pd.read_parquet(path).columns)


def read_track_rows(
    track_path: Path,
    *,
    frame_min: int | None = None,
    frame_max: int | None = None,
) -> pd.DataFrame:
    cols = set(parquet_columns(track_path))
    required = {"Frame", "Bodypoint", "X", "Y", "TrackX", "TrackY"}
    missing = required.difference(cols)
    if missing:
        raise ValueError(f"{track_path.name} missing required columns: {sorted(missing)}")

    read_cols = ["Frame", "Bodypoint", "X", "Y", "TrackX", "TrackY"]
    if "TrackID" in cols:
        read_cols.append("TrackID")

    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        filt = None
        if frame_min is not None:
            filt = pc.field("Frame") >= int(frame_min)
        if frame_max is not None:
            right = pc.field("Frame") <= int(frame_max)
            filt = right if filt is None else filt & right
        table = ds.dataset(track_path, format="parquet").to_table(columns=read_cols, filter=filt)
        df = table.to_pandas()
    except Exception:
        df = pd.read_parquet(track_path, columns=read_cols)
        if frame_min is not None:
            df = df[pd.to_numeric(df["Frame"], errors="coerce") >= int(frame_min)]
        if frame_max is not None:
            df = df[pd.to_numeric(df["Frame"], errors="coerce") <= int(frame_max)]

    for col in ["Frame", "Bodypoint"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["X", "Y", "TrackX", "TrackY"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Frame", "Bodypoint"]).copy()
    df["Frame"] = df["Frame"].round().astype(np.int64)
    df["Bodypoint"] = df["Bodypoint"].round().astype(np.int64)
    return df


def pose_wide(track_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if track_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    wide = (
        track_rows.pivot_table(
            index="Frame",
            columns="Bodypoint",
            values=["X", "Y"],
            aggfunc="mean",
            dropna=False,
        )
        .sort_index()
    )
    wide.columns = pd.MultiIndex.from_arrays(
        [wide.columns.get_level_values(0), wide.columns.get_level_values(1).astype(int)]
    )
    anchor = (
        track_rows.groupby("Frame", sort=True, as_index=True)
        .agg(
            track_x_px=("TrackX", "mean"),
            track_y_px=("TrackY", "mean"),
            n_bodypoints_detected=("Bodypoint", "nunique"),
        )
        .sort_index()
    )
    return wide, anchor


def _xy(wide: pd.DataFrame, bodypoint: int) -> tuple[pd.Series, pd.Series]:
    if ("X", bodypoint) in wide.columns:
        x = wide[("X", bodypoint)].astype(float)
    else:
        x = pd.Series(np.nan, index=wide.index, dtype=float)
    if ("Y", bodypoint) in wide.columns:
        y = wide[("Y", bodypoint)].astype(float)
    else:
        y = pd.Series(np.nan, index=wide.index, dtype=float)
    return x, y


def _vector(wide: pd.DataFrame, start: int, stop: int) -> np.ndarray:
    x0, y0 = _xy(wide, start)
    x1, y1 = _xy(wide, stop)
    return np.column_stack([(x1 - x0).to_numpy(np.float64), (y1 - y0).to_numpy(np.float64)])


def posture_features(wide: pd.DataFrame, anchor: pd.DataFrame, *, mm_per_px: float) -> pd.DataFrame:
    out = pd.DataFrame(index=wide.index)
    out["Frame"] = wide.index.to_numpy(np.int64)
    out = out.join(anchor, how="left")

    for name, (first, second, invert) in ANGLE_PAIRS.items():
        angle = angle_between(_vector(wide, *first), _vector(wide, *second))
        if invert:
            angle = 360.0 - angle
        out[name] = angle

    x_values = wide["X"] if "X" in wide.columns.get_level_values(0) else pd.DataFrame(index=wide.index)
    y_values = wide["Y"] if "Y" in wide.columns.get_level_values(0) else pd.DataFrame(index=wide.index)
    out["pose_width_mm"] = (x_values.max(axis=1) - x_values.min(axis=1)).to_numpy(np.float64) * float(mm_per_px)
    out["pose_height_mm"] = (y_values.max(axis=1) - y_values.min(axis=1)).to_numpy(np.float64) * float(mm_per_px)
    out["pose_area_mm2"] = out["pose_width_mm"] * out["pose_height_mm"]

    for left, right in DISTANCE_PAIRS:
        x0, y0 = _xy(wide, left)
        x1, y1 = _xy(wide, right)
        dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) * float(mm_per_px)
        out[f"bp{left}_bp{right}_dist_mm"] = dist.to_numpy(np.float64)

    out["track_x_mm"] = out["track_x_px"] * float(mm_per_px)
    out["track_y_mm"] = out["track_y_px"] * float(mm_per_px)
    return out.reset_index(drop=True)


def find_speed_metadata(speed_root: Path, track_path: Path) -> dict[str, object] | None:
    speed_root = Path(speed_root)
    candidates = [
        speed_root / "per_track" / track_path.stem / "speed_metadata.json",
        speed_root / track_path.stem / "speed_metadata.json",
    ]
    track_id = track_id_from_name(track_path)
    side = side_from_name(track_path)
    if track_id is not None:
        pattern = f"TrackID_{track_id:04d}_*"
        candidates.extend(sorted((speed_root / "per_track").glob(f"{pattern}/speed_metadata.json")))
        candidates.extend(sorted(speed_root.glob(f"{pattern}/speed_metadata.json")))
    for path in candidates:
        if not path.exists():
            continue
        meta = json.loads(path.read_text())
        meta_side = side_from_name(str(meta.get("track_name", path.parent.name)))
        if side is not None and meta_side is not None and meta_side != side:
            continue
        meta["speed_metadata_path"] = str(path)
        return meta
    return None


def load_speed_for_track(speed_root: Path | None, track_path: Path) -> tuple[np.ndarray | None, dict[str, object] | None]:
    if speed_root is None:
        return None, None
    meta = find_speed_metadata(Path(speed_root), track_path)
    if meta is None:
        return None, None
    speed_path = Path(str(meta.get("speed_path", "")))
    if not speed_path.exists():
        speed_path = Path(str(meta["speed_metadata_path"])).parent / "speed_mm_s.npy"
    if not speed_path.exists():
        return None, meta
    return np.load(speed_path, mmap_mode="r"), meta


def _rolling_stats_at_indices(values: np.ndarray, indices: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    window = max(1, int(window))
    half_left = window // 2
    half_right = window - half_left
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    csum2 = np.concatenate([[0.0], np.cumsum(filled * filled)])
    ccount = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    starts = np.maximum(0, indices - half_left)
    stops = np.minimum(len(values), indices + half_right)
    count = ccount[stops] - ccount[starts]
    total = csum[stops] - csum[starts]
    total2 = csum2[stops] - csum2[starts]
    mean = np.full(len(indices), np.nan, dtype=np.float64)
    std = np.full(len(indices), np.nan, dtype=np.float64)
    valid = count > 0
    mean[valid] = total[valid] / count[valid]
    var = np.full(len(indices), np.nan, dtype=np.float64)
    var[valid] = total2[valid] / count[valid] - mean[valid] * mean[valid]
    std[valid] = np.sqrt(np.maximum(var[valid], 0.0))
    frac = count.astype(np.float64) / np.maximum(stops - starts, 1)
    return mean, std, frac


def add_speed_features(
    features: pd.DataFrame,
    speed: np.ndarray | None,
    speed_meta: dict[str, object] | None,
    *,
    fps: float,
    windows_seconds: tuple[float, ...] = (1.0, 5.0, 30.0),
) -> pd.DataFrame:
    out = features.copy()
    if speed is None or speed_meta is None:
        out["speed_mm_s"] = np.nan
        for seconds in windows_seconds:
            suffix = f"{seconds:g}s".replace(".", "p")
            out[f"speed_mean_{suffix}"] = np.nan
            out[f"speed_std_{suffix}"] = np.nan
            out[f"speed_valid_frac_{suffix}"] = np.nan
        return out

    frame_min = int(speed_meta.get("frame_min", 0))
    indices = out["Frame"].to_numpy(np.int64) - frame_min
    valid = (indices >= 0) & (indices < len(speed))
    speed_values = np.full(len(out), np.nan, dtype=np.float64)
    speed_values[valid] = np.asarray(speed[indices[valid]], dtype=np.float64)
    out["speed_mm_s"] = speed_values
    out["speed_log1p_mm_s"] = np.log1p(np.clip(speed_values, 0.0, None))

    for seconds in windows_seconds:
        suffix = f"{seconds:g}s".replace(".", "p")
        window = max(1, int(round(float(seconds) * float(fps))))
        mean = np.full(len(out), np.nan, dtype=np.float64)
        std = np.full(len(out), np.nan, dtype=np.float64)
        frac = np.full(len(out), np.nan, dtype=np.float64)
        if valid.any():
            m, s, f = _rolling_stats_at_indices(np.asarray(speed, dtype=np.float64), indices[valid], window)
            mean[valid] = m
            std[valid] = s
            frac[valid] = f
        out[f"speed_mean_{suffix}"] = mean
        out[f"speed_std_{suffix}"] = std
        out[f"speed_valid_frac_{suffix}"] = frac
    return out


def extract_track_features(
    track_path: Path,
    *,
    speed_root: Path | None = None,
    fps: float = 24.0,
    mm_per_px: float = 0.016,
    frame_min: int | None = None,
    frame_max: int | None = None,
    speed_windows_seconds: tuple[float, ...] = (1.0, 5.0, 30.0),
) -> pd.DataFrame:
    track_path = Path(track_path)
    rows = read_track_rows(track_path, frame_min=frame_min, frame_max=frame_max)
    wide, anchor = pose_wide(rows)
    if wide.empty:
        return pd.DataFrame()

    features = posture_features(wide, anchor, mm_per_px=mm_per_px)
    speed, speed_meta = load_speed_for_track(speed_root, track_path) if speed_root is not None else (None, None)
    features = add_speed_features(features, speed, speed_meta, fps=fps, windows_seconds=speed_windows_seconds)
    features["track_name"] = track_path.name
    features["track_path"] = str(track_path)
    features["track_id"] = track_id_from_name(track_path)
    features["side"] = side_from_name(track_path)
    return features.sort_values("Frame", kind="mergesort").reset_index(drop=True)


def default_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "Frame",
        "track_name",
        "track_path",
        "track_id",
        "side",
        "label",
        "label_value",
    }
    return [
        col
        for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]

