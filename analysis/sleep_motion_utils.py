#!/usr/bin/env python3
"""Reusable all-bodypoint motion caches for sleep-state tuning."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd


SPEED_FILENAME = "bodypoint_speed_mm_s.npy"
FRAME_GAP_FILENAME = "bodypoint_frame_gap_u1.npy"
BODYPOINT_IDS_FILENAME = "bodypoint_ids.npy"
FRAMES_FILENAME = "frames.npy"
METADATA_FILENAME = "sleep_motion_metadata.json"
BODY_BODYPOINT_IDS = (0, 1, 2, 3)
ANTENNA_BODYPOINT_IDS = (4, 5, 6, 7, 8, 9)


def track_id_from_name(path: Path) -> int | None:
    match = re.search(r"TrackID_(\d+)", Path(path).stem)
    return int(match.group(1)) if match else None


def side_from_name(path: Path) -> str | None:
    match = re.search(r"(?:^|_)(left|right)(?:_|\.|$)", Path(path).name, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def cache_dir_for_track(root: Path, track_path: Path) -> Path:
    root = Path(root)
    parent = root if root.name == "per_track" else root / "per_track"
    return parent / Path(track_path).stem


def is_sleep_motion_cache(path: Path) -> bool:
    path = Path(path)
    return all(
        (path / name).is_file()
        for name in (
            SPEED_FILENAME,
            FRAME_GAP_FILENAME,
            BODYPOINT_IDS_FILENAME,
            FRAMES_FILENAME,
            METADATA_FILENAME,
        )
    )


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as stream:
            np.save(stream, values)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_pose_rows(track_path: Path) -> pd.DataFrame:
    track_path = Path(track_path)
    wanted = ["Frame", "Bodypoint", "X", "Y"]
    rows = pd.read_parquet(track_path, columns=wanted)
    missing = set(wanted).difference(rows.columns)
    if missing:
        raise ValueError(f"{track_path.name} is missing required columns: {sorted(missing)}")
    for col in wanted:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.dropna(subset=wanted).copy()
    if rows.empty:
        raise ValueError(f"{track_path.name} has no finite pose rows")
    rows["Frame"] = np.rint(rows["Frame"]).astype(np.int64)
    rows["Bodypoint"] = np.rint(rows["Bodypoint"]).astype(np.int64)
    return rows


def compute_sleep_motion_cache(
    track_path: Path,
    out_dir: Path,
    *,
    fps: float = 24.0,
    mm_per_px: float = 0.016,
    cache_max_gap_frames: int = 120,
) -> dict[str, object]:
    """Compute a dense frame-by-bodypoint speed cache for one track.

    Speeds are assigned to the later sample in each bodypoint pair and divided
    by the actual frame gap. The gap matrix allows downstream code to apply a
    stricter maximum gap without reading the pose parquet again.
    """

    track_path = Path(track_path).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = float(fps)
    mm_per_px = float(mm_per_px)
    cache_max_gap_frames = int(cache_max_gap_frames)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not math.isfinite(mm_per_px) or mm_per_px <= 0:
        raise ValueError(f"mm_per_px must be positive, got {mm_per_px}")
    if cache_max_gap_frames < 1 or cache_max_gap_frames > np.iinfo(np.uint8).max:
        raise ValueError("cache_max_gap_frames must be between 1 and 255")

    rows = read_pose_rows(track_path)
    n_input_rows = int(len(rows))
    rows = (
        rows.sort_values(["Bodypoint", "Frame"], kind="mergesort")
        .drop_duplicates(["Bodypoint", "Frame"], keep="last")
        .reset_index(drop=True)
    )
    n_duplicate_rows = n_input_rows - int(len(rows))
    frame_min = int(rows["Frame"].min())
    frame_max = int(rows["Frame"].max())
    n_frames = frame_max - frame_min + 1
    cache_frames = np.sort(rows["Frame"].unique()).astype(np.int64, copy=False)
    n_cached_frames = int(len(cache_frames))
    bodypoint_ids = np.sort(rows["Bodypoint"].unique()).astype(np.int16, copy=False)
    n_bodypoints = int(len(bodypoint_ids))
    if n_bodypoints == 0:
        raise ValueError(f"{track_path.name} has no bodypoints")

    speed_path = out_dir / SPEED_FILENAME
    gap_path = out_dir / FRAME_GAP_FILENAME
    speed_temp = speed_path.with_name(f".{speed_path.name}.{os.getpid()}.tmp")
    gap_temp = gap_path.with_name(f".{gap_path.name}.{os.getpid()}.tmp")
    speed_matrix = None
    gap_matrix = None
    valid_by_bodypoint: dict[str, int] = {}
    try:
        speed_matrix = np.lib.format.open_memmap(
            speed_temp,
            mode="w+",
            dtype=np.float32,
            shape=(n_cached_frames, n_bodypoints),
        )
        speed_matrix[:] = np.nan
        gap_matrix = np.lib.format.open_memmap(
            gap_temp,
            mode="w+",
            dtype=np.uint8,
            shape=(n_cached_frames, n_bodypoints),
        )
        gap_matrix[:] = 0

        bodypoint_columns = {int(bodypoint): column for column, bodypoint in enumerate(bodypoint_ids)}
        for bodypoint, group in rows.groupby("Bodypoint", sort=False):
            column = bodypoint_columns[int(bodypoint)]
            frames = group["Frame"].to_numpy(np.int64, copy=False)
            x = group["X"].to_numpy(np.float64, copy=False)
            y = group["Y"].to_numpy(np.float64, copy=False)
            if len(frames) < 2:
                valid_by_bodypoint[str(int(bodypoint))] = 0
                continue
            frame_gap = np.diff(frames)
            valid = (frame_gap > 0) & (frame_gap <= cache_max_gap_frames)
            if not valid.any():
                valid_by_bodypoint[str(int(bodypoint))] = 0
                continue
            later_frames = frames[1:][valid]
            dense_index = np.searchsorted(cache_frames, later_frames)
            dx = np.diff(x)[valid]
            dy = np.diff(y)[valid]
            gaps = frame_gap[valid]
            speed = np.hypot(dx, dy) * mm_per_px * fps / gaps
            finite = np.isfinite(speed)
            dense_index = dense_index[finite]
            gaps = gaps[finite]
            speed = speed[finite]
            speed_matrix[dense_index, column] = speed.astype(np.float32, copy=False)
            gap_matrix[dense_index, column] = gaps.astype(np.uint8, copy=False)
            valid_by_bodypoint[str(int(bodypoint))] = int(len(speed))

        speed_matrix.flush()
        gap_matrix.flush()
        del speed_matrix
        del gap_matrix
        speed_matrix = None
        gap_matrix = None
        os.replace(speed_temp, speed_path)
        os.replace(gap_temp, gap_path)
    finally:
        if speed_matrix is not None:
            del speed_matrix
        if gap_matrix is not None:
            del gap_matrix
        speed_temp.unlink(missing_ok=True)
        gap_temp.unlink(missing_ok=True)

    _atomic_save_array(out_dir / BODYPOINT_IDS_FILENAME, bodypoint_ids)
    _atomic_save_array(out_dir / FRAMES_FILENAME, cache_frames)
    source_stat = track_path.stat()
    n_valid_speed_values = int(sum(valid_by_bodypoint.values()))
    metadata: dict[str, object] = {
        "format_version": 2,
        "track_path": str(track_path),
        "track_name": track_path.name,
        "track_id": track_id_from_name(track_path),
        "side": side_from_name(track_path),
        "frame_min": frame_min,
        "frame_max": frame_max,
        "n_frames": n_frames,
        "n_cached_frames": n_cached_frames,
        "bodypoint_ids": [int(value) for value in bodypoint_ids],
        "n_bodypoints": n_bodypoints,
        "n_input_pose_rows": n_input_rows,
        "n_duplicate_pose_rows_removed": n_duplicate_rows,
        "n_valid_speed_values": n_valid_speed_values,
        "valid_speed_values_by_bodypoint": valid_by_bodypoint,
        "fps": fps,
        "mm_per_px": mm_per_px,
        "cache_max_gap_frames": cache_max_gap_frames,
        "speed_units": "mm/s",
        "speed_shape": [n_cached_frames, n_bodypoints],
        "speed_dtype": "float32",
        "frame_gap_dtype": "uint8",
        "missing_speed_value": "NaN",
        "missing_frame_gap_value": 0,
        "speed_assignment": "later_frame",
        "frame_indexing": FRAMES_FILENAME,
        "source_size_bytes": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "files": {
            "bodypoint_speed_mm_s": SPEED_FILENAME,
            "bodypoint_frame_gap": FRAME_GAP_FILENAME,
            "bodypoint_ids": BODYPOINT_IDS_FILENAME,
            "frames": FRAMES_FILENAME,
        },
    }
    _atomic_write_json(out_dir / METADATA_FILENAME, metadata)
    return metadata


@dataclass
class SleepMotionCache:
    root: Path
    metadata: dict[str, object]
    speed_mm_s: np.ndarray
    frame_gap: np.ndarray
    bodypoint_ids: np.ndarray
    frames: np.ndarray

    @property
    def frame_min(self) -> int:
        return int(self.metadata["frame_min"])

    @property
    def frame_max(self) -> int:
        return int(self.metadata["frame_max"])

    @property
    def fps(self) -> float:
        return float(self.metadata["fps"])

    @property
    def cache_max_gap_frames(self) -> int:
        return int(self.metadata["cache_max_gap_frames"])


def load_sleep_motion_cache(path: Path, *, mmap_mode: str | None = "r") -> SleepMotionCache:
    path = Path(path)
    metadata = json.loads((path / METADATA_FILENAME).read_text())
    speed = np.load(path / SPEED_FILENAME, mmap_mode=mmap_mode)
    frame_gap = np.load(path / FRAME_GAP_FILENAME, mmap_mode=mmap_mode)
    bodypoint_ids = np.load(path / BODYPOINT_IDS_FILENAME, mmap_mode=mmap_mode)
    frames = np.load(path / FRAMES_FILENAME, mmap_mode=mmap_mode)
    expected_shape = tuple(int(value) for value in metadata["speed_shape"])
    if speed.shape != expected_shape or frame_gap.shape != expected_shape:
        raise ValueError(
            f"Sleep-motion cache shape mismatch under {path}: "
            f"metadata={expected_shape}, speed={speed.shape}, gaps={frame_gap.shape}"
        )
    if speed.ndim != 2 or len(bodypoint_ids) != speed.shape[1] or len(frames) != speed.shape[0]:
        raise ValueError(f"Invalid sleep-motion cache under {path}")
    return SleepMotionCache(
        root=path,
        metadata=metadata,
        speed_mm_s=speed,
        frame_gap=frame_gap,
        bodypoint_ids=bodypoint_ids,
        frames=frames,
    )


def _aggregate_cached_rows(
    cache: SleepMotionCache,
    start_row: int,
    stop_row: int,
    *,
    bodypoint_percentile: float,
    min_valid_bodypoint_fraction: float,
    max_gap_frames: int,
    max_bodypoint_speed_mm_s: float | None,
) -> np.ndarray:
    speed = np.asarray(cache.speed_mm_s[start_row:stop_row], dtype=np.float32)
    gaps = np.asarray(cache.frame_gap[start_row:stop_row], dtype=np.uint8)
    valid = np.isfinite(speed) & (gaps > 0) & (gaps <= max_gap_frames)
    if max_bodypoint_speed_mm_s is not None and math.isfinite(float(max_bodypoint_speed_mm_s)):
        valid &= speed <= float(max_bodypoint_speed_mm_s)
    work = np.where(valid, speed, np.nan)
    output = np.full(len(work), np.nan, dtype=np.float32)
    min_valid_bodypoints = max(1, int(math.ceil(speed.shape[1] * min_valid_bodypoint_fraction)))
    has_value = valid.sum(axis=1) >= min_valid_bodypoints
    if has_value.any():
        output[has_value] = _finite_row_percentile(
            work[has_value],
            bodypoint_percentile,
        )
    return output


def _finite_row_percentile(values: np.ndarray, percentile: float) -> np.ndarray:
    """Match NumPy's linear nanpercentile without its slow per-row dispatch."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    finite = np.isfinite(values)
    count = finite.sum(axis=1).astype(np.int64, copy=False)
    output = np.full(len(values), np.nan, dtype=np.float32)
    has_value = count > 0
    if not has_value.any():
        return output

    ordered = np.sort(np.where(finite, values, np.inf), axis=1)
    rank = (count[has_value] - 1).astype(np.float64) * (float(percentile) / 100.0)
    lower = np.floor(rank).astype(np.int64)
    upper = np.ceil(rank).astype(np.int64)
    fraction = (rank - lower).astype(np.float32)
    rows = np.flatnonzero(has_value)
    lower_value = ordered[rows, lower]
    upper_value = ordered[rows, upper]
    output[has_value] = lower_value + (upper_value - lower_value) * fraction
    return output


def aggregate_motion_slice(
    cache: SleepMotionCache,
    start_index: int,
    stop_index: int,
    *,
    bodypoint_percentile: float = 95.0,
    min_valid_bodypoint_fraction: float = 0.6,
    max_gap_frames: int = 5,
    max_bodypoint_speed_mm_s: float | None = 20.0,
) -> np.ndarray:
    start_index = max(0, int(start_index))
    stop_index = min(int(stop_index), int(cache.metadata["n_frames"]))
    if stop_index <= start_index:
        return np.empty(0, dtype=np.float32)
    max_gap_frames = int(max_gap_frames)
    if max_gap_frames < 1:
        raise ValueError("max_gap_frames must be at least 1")
    if max_gap_frames > cache.cache_max_gap_frames:
        raise ValueError(
            f"Requested max_gap_frames={max_gap_frames}, but cache only contains gaps up to "
            f"{cache.cache_max_gap_frames}"
        )
    percentile = float(bodypoint_percentile)
    if not 0 <= percentile <= 100:
        raise ValueError("bodypoint_percentile must be between 0 and 100")
    min_valid_bodypoint_fraction = float(min_valid_bodypoint_fraction)
    if not 0 <= min_valid_bodypoint_fraction <= 1:
        raise ValueError("min_valid_bodypoint_fraction must be between 0 and 1")

    start_frame = cache.frame_min + start_index
    stop_frame = cache.frame_min + stop_index
    start_row = int(np.searchsorted(cache.frames, start_frame, side="left"))
    stop_row = int(np.searchsorted(cache.frames, stop_frame, side="left"))
    output = np.full(stop_index - start_index, np.nan, dtype=np.float32)
    if stop_row > start_row:
        row_motion = _aggregate_cached_rows(
            cache,
            start_row,
            stop_row,
            bodypoint_percentile=percentile,
            min_valid_bodypoint_fraction=min_valid_bodypoint_fraction,
            max_gap_frames=max_gap_frames,
            max_bodypoint_speed_mm_s=max_bodypoint_speed_mm_s,
        )
        output_index = np.asarray(cache.frames[start_row:stop_row], dtype=np.int64) - start_frame
        output[output_index] = row_motion
    return output


def aggregate_bodypoint_group_slice(
    cache: SleepMotionCache,
    start_index: int,
    stop_index: int,
    *,
    bodypoint_ids: tuple[int, ...] | list[int] | np.ndarray,
    bodypoint_percentile: float = 75.0,
    min_valid_bodypoint_fraction: float = 0.5,
    max_gap_frames: int = 5,
    max_bodypoint_speed_mm_s: float | None = 20.0,
) -> np.ndarray:
    """Aggregate a selected anatomical group without rereading pose tracks."""

    start_index = max(0, int(start_index))
    stop_index = min(int(stop_index), int(cache.metadata["n_frames"]))
    if stop_index <= start_index:
        return np.empty(0, dtype=np.float32)
    max_gap_frames = int(max_gap_frames)
    if max_gap_frames < 1:
        raise ValueError("max_gap_frames must be at least 1")
    if max_gap_frames > cache.cache_max_gap_frames:
        raise ValueError(
            f"Requested max_gap_frames={max_gap_frames}, but cache only contains gaps up to "
            f"{cache.cache_max_gap_frames}"
        )
    percentile = float(bodypoint_percentile)
    if not 0 <= percentile <= 100:
        raise ValueError("bodypoint_percentile must be between 0 and 100")
    min_valid_bodypoint_fraction = float(min_valid_bodypoint_fraction)
    if not 0 <= min_valid_bodypoint_fraction <= 1:
        raise ValueError("min_valid_bodypoint_fraction must be between 0 and 1")

    requested_ids = {int(value) for value in bodypoint_ids}
    columns = np.asarray(
        [column for column, value in enumerate(cache.bodypoint_ids) if int(value) in requested_ids],
        dtype=np.int64,
    )
    if columns.size == 0:
        return np.full(stop_index - start_index, np.nan, dtype=np.float32)

    start_frame = cache.frame_min + start_index
    stop_frame = cache.frame_min + stop_index
    start_row = int(np.searchsorted(cache.frames, start_frame, side="left"))
    stop_row = int(np.searchsorted(cache.frames, stop_frame, side="left"))
    output = np.full(stop_index - start_index, np.nan, dtype=np.float32)
    if stop_row <= start_row:
        return output

    speed = np.asarray(cache.speed_mm_s[start_row:stop_row, columns], dtype=np.float32)
    gaps = np.asarray(cache.frame_gap[start_row:stop_row, columns], dtype=np.uint8)
    valid = np.isfinite(speed) & (gaps > 0) & (gaps <= max_gap_frames)
    if max_bodypoint_speed_mm_s is not None and math.isfinite(float(max_bodypoint_speed_mm_s)):
        valid &= speed <= float(max_bodypoint_speed_mm_s)
    work = np.where(valid, speed, np.nan)
    row_motion = np.full(len(work), np.nan, dtype=np.float32)
    min_valid_bodypoints = max(1, int(math.ceil(columns.size * min_valid_bodypoint_fraction)))
    has_value = valid.sum(axis=1) >= min_valid_bodypoints
    if has_value.any():
        row_motion[has_value] = _finite_row_percentile(work[has_value], percentile)
    output_index = np.asarray(cache.frames[start_row:stop_row], dtype=np.int64) - start_frame
    output[output_index] = row_motion
    return output


def aggregate_motion_vector(
    cache: SleepMotionCache,
    *,
    bodypoint_percentile: float = 95.0,
    min_valid_bodypoint_fraction: float = 0.6,
    max_gap_frames: int = 5,
    max_bodypoint_speed_mm_s: float | None = 20.0,
    chunk_frames: int = 100_000,
) -> np.ndarray:
    n_frames = int(cache.metadata["n_frames"])
    output = np.full(n_frames, np.nan, dtype=np.float32)
    chunk_frames = max(1, int(chunk_frames))
    for start in range(0, n_frames, chunk_frames):
        stop = min(n_frames, start + chunk_frames)
        output[start:stop] = aggregate_motion_slice(
            cache,
            start,
            stop,
            bodypoint_percentile=bodypoint_percentile,
            min_valid_bodypoint_fraction=min_valid_bodypoint_fraction,
            max_gap_frames=max_gap_frames,
            max_bodypoint_speed_mm_s=max_bodypoint_speed_mm_s,
        )
    return output


def classify_trailing_quiet(
    motion_mm_s: np.ndarray,
    *,
    fps: float,
    speed_threshold_mm_s: float = 0.1,
    window_seconds: float = 60.0,
    quiet_fraction_threshold: float = 0.8,
    min_valid_fraction: float = 0.2,
) -> dict[str, np.ndarray]:
    """Apply the GUI's tunable trailing-window sleep rule to a motion vector."""

    motion = np.asarray(motion_mm_s, dtype=np.float32)
    n_frames = int(len(motion))
    fps = float(fps)
    speed_threshold_mm_s = float(speed_threshold_mm_s)
    window_seconds = float(window_seconds)
    quiet_fraction_threshold = float(quiet_fraction_threshold)
    min_valid_fraction = float(min_valid_fraction)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if not math.isfinite(speed_threshold_mm_s) or speed_threshold_mm_s < 0:
        raise ValueError("speed_threshold_mm_s must be nonnegative")
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not 0 <= quiet_fraction_threshold <= 1:
        raise ValueError("quiet_fraction_threshold must be between 0 and 1")
    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be between 0 and 1")
    window_frames = max(1, int(round(window_seconds * fps)))
    finite = np.isfinite(motion)
    quiet = finite & (motion <= speed_threshold_mm_s)
    speed_values = np.where(finite, motion, 0.0).astype(np.float64, copy=False)

    valid_sum = np.concatenate(([0], np.cumsum(finite, dtype=np.int64)))
    quiet_sum = np.concatenate(([0], np.cumsum(quiet, dtype=np.int64)))
    speed_sum = np.concatenate(([0.0], np.cumsum(speed_values, dtype=np.float64)))
    stop = np.arange(1, n_frames + 1, dtype=np.int64)
    start = np.maximum(0, stop - window_frames)
    denominator = stop - start
    valid_count = valid_sum[stop] - valid_sum[start]
    quiet_count = quiet_sum[stop] - quiet_sum[start]
    window_speed_sum = speed_sum[stop] - speed_sum[start]

    valid_fraction = (valid_count / denominator).astype(np.float32)
    quiet_fraction = np.full(n_frames, np.nan, dtype=np.float32)
    mean_speed = np.full(n_frames, np.nan, dtype=np.float32)
    has_valid = valid_count > 0
    quiet_fraction[has_valid] = (quiet_count[has_valid] / valid_count[has_valid]).astype(np.float32)
    mean_speed[has_valid] = (window_speed_sum[has_valid] / valid_count[has_valid]).astype(np.float32)

    state = np.full(n_frames, -1, dtype=np.int8)
    classifiable = has_valid & (valid_fraction >= min_valid_fraction)
    state[classifiable] = 0
    state[classifiable & (quiet_fraction >= quiet_fraction_threshold)] = 1
    return {
        "state": state,
        "quiet_fraction": quiet_fraction,
        "valid_fraction": valid_fraction,
        "mean_speed_mm_s": mean_speed,
    }


def classify_body_antenna_quiet(
    body_motion_mm_s: np.ndarray,
    antenna_motion_mm_s: np.ndarray,
    *,
    fps: float,
    body_speed_threshold_mm_s: float = 0.5,
    antenna_speed_threshold_mm_s: float = 0.7,
    window_seconds: float = 10.0,
    quiet_fraction_threshold: float = 0.9,
    min_valid_fraction: float = 0.5,
    require_full_window: bool = True,
) -> dict[str, np.ndarray]:
    """Classify sleep from strict body stillness and permissive antenna motion.

    A frame supplies evidence only when both anatomical groups are valid. It is
    low-motion when the body and antenna group metrics are below their separate
    thresholds. Sleep requires enough valid evidence and enough low-motion
    frames in the trailing window; the quiet-fraction rule tolerates occasional
    movement. State values are -1 unknown, 0 wake, and 1 sleep.
    """

    body = np.asarray(body_motion_mm_s, dtype=np.float32)
    antenna = np.asarray(antenna_motion_mm_s, dtype=np.float32)
    if body.shape != antenna.shape or body.ndim != 1:
        raise ValueError("body and antenna motion must be one-dimensional arrays with equal shape")
    fps = float(fps)
    body_threshold = float(body_speed_threshold_mm_s)
    antenna_threshold = float(antenna_speed_threshold_mm_s)
    window_seconds = float(window_seconds)
    quiet_fraction_threshold = float(quiet_fraction_threshold)
    min_valid_fraction = float(min_valid_fraction)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if not math.isfinite(body_threshold) or body_threshold < 0:
        raise ValueError("body_speed_threshold_mm_s must be nonnegative")
    if not math.isfinite(antenna_threshold) or antenna_threshold < 0:
        raise ValueError("antenna_speed_threshold_mm_s must be nonnegative")
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not 0 <= quiet_fraction_threshold <= 1:
        raise ValueError("quiet_fraction_threshold must be between 0 and 1")
    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be between 0 and 1")

    n_frames = int(len(body))
    window_frames = max(1, int(round(window_seconds * fps)))
    valid = np.isfinite(body) & np.isfinite(antenna)
    quiet = valid & (body <= body_threshold) & (antenna <= antenna_threshold)
    body_values = np.where(valid, body, 0.0).astype(np.float64, copy=False)
    antenna_values = np.where(valid, antenna, 0.0).astype(np.float64, copy=False)

    valid_sum = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    quiet_sum = np.concatenate(([0], np.cumsum(quiet, dtype=np.int64)))
    body_sum = np.concatenate(([0.0], np.cumsum(body_values, dtype=np.float64)))
    antenna_sum = np.concatenate(([0.0], np.cumsum(antenna_values, dtype=np.float64)))
    stop = np.arange(1, n_frames + 1, dtype=np.int64)
    start = np.maximum(0, stop - window_frames)
    denominator = stop - start
    valid_count = valid_sum[stop] - valid_sum[start]
    quiet_count = quiet_sum[stop] - quiet_sum[start]

    valid_fraction = (valid_count / denominator).astype(np.float32)
    quiet_fraction = np.full(n_frames, np.nan, dtype=np.float32)
    mean_body = np.full(n_frames, np.nan, dtype=np.float32)
    mean_antenna = np.full(n_frames, np.nan, dtype=np.float32)
    has_valid = valid_count > 0
    quiet_fraction[has_valid] = (quiet_count[has_valid] / valid_count[has_valid]).astype(np.float32)
    mean_body[has_valid] = ((body_sum[stop] - body_sum[start])[has_valid] / valid_count[has_valid]).astype(
        np.float32
    )
    mean_antenna[has_valid] = (
        (antenna_sum[stop] - antenna_sum[start])[has_valid] / valid_count[has_valid]
    ).astype(np.float32)

    state = np.full(n_frames, -1, dtype=np.int8)
    classifiable = has_valid & (valid_fraction >= min_valid_fraction)
    if require_full_window:
        classifiable &= denominator >= window_frames
    state[classifiable] = 0
    state[classifiable & (quiet_fraction >= quiet_fraction_threshold)] = 1
    return {
        "state": state,
        "quiet_fraction": quiet_fraction,
        "valid_fraction": valid_fraction,
        "mean_body_speed_mm_s": mean_body,
        "mean_antenna_speed_mm_s": mean_antenna,
    }
