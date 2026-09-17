#!/usr/bin/env python3
"""Classify cached bodypoint motion into threshold-based sleep labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd


repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analysis.sleep_motion_utils import (  # noqa: E402
    ANTENNA_BODYPOINT_IDS,
    BODY_BODYPOINT_IDS,
    METADATA_FILENAME as SOURCE_METADATA_FILENAME,
    aggregate_bodypoint_group_slice,
    classify_body_antenna_quiet,
    is_sleep_motion_cache,
    load_sleep_motion_cache,
)


STATE_FILENAME = "sleep_state_i1.npy"
QUIET_FRACTION_FILENAME = "quiet_fraction_f2.npy"
VALID_FRACTION_FILENAME = "valid_fraction_f2.npy"
BOUTS_FILENAME = "sleep_bouts.parquet"
LABEL_METADATA_FILENAME = "sleep_motion_label_metadata.json"
SUMMARY_FILENAME = "sleep_motion_label_summary.parquet"
SUMMARY_METADATA_FILENAME = "sleep_motion_label_summary.json"
COMPLETE_FILENAME = "sleep_motion_labels_complete.ok"


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as stream:
            np.save(stream, values)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_write_parquet(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.parquet")
    try:
        table.to_parquet(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _classifier_parameters(args: argparse.Namespace) -> dict[str, object]:
    return {
        "body_speed_threshold_mm_s": float(args.body_threshold),
        "antenna_speed_threshold_mm_s": float(args.antenna_threshold),
        "window_seconds": float(args.history_seconds),
        "quiet_fraction_threshold": float(args.quiet_fraction),
        "min_valid_frame_fraction": float(args.min_valid_frame_fraction),
        "require_full_window": True,
        "body_bodypoint_ids": list(BODY_BODYPOINT_IDS),
        "antenna_bodypoint_ids": list(ANTENNA_BODYPOINT_IDS),
        "body_percentile": float(args.body_percentile),
        "antenna_percentile": float(args.antenna_percentile),
        "min_valid_bodypoint_fraction": float(args.min_valid_body_fraction),
        "min_valid_antenna_fraction": float(args.min_valid_antenna_fraction),
        "max_gap_frames": int(args.max_gap_frames),
        "max_bodypoint_speed_mm_s": float(args.max_speed),
    }


def _validate_parameters(parameters: dict[str, object]) -> None:
    for name in ("window_seconds", "max_gap_frames"):
        value = float(parameters[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "body_speed_threshold_mm_s",
        "antenna_speed_threshold_mm_s",
        "max_bodypoint_speed_mm_s",
    ):
        value = float(parameters[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in (
        "quiet_fraction_threshold",
        "min_valid_frame_fraction",
        "min_valid_bodypoint_fraction",
        "min_valid_antenna_fraction",
    ):
        if not 0 <= float(parameters[name]) <= 1:
            raise ValueError(f"{name} must be between zero and one")
    for name in ("body_percentile", "antenna_percentile"):
        if not 0 <= float(parameters[name]) <= 100:
            raise ValueError(f"{name} must be between zero and 100")


def _sleep_bouts(
    state: np.ndarray,
    quiet_fraction: np.ndarray,
    valid_fraction: np.ndarray,
    *,
    frame_min: int,
    fps: float,
) -> pd.DataFrame:
    is_sleep = np.asarray(state) == 1
    transitions = np.diff(np.pad(is_sleep.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    rows: list[dict[str, object]] = []
    for start, stop in zip(starts, stops, strict=True):
        quiet = quiet_fraction[start:stop]
        valid = valid_fraction[start:stop]
        rows.append(
            {
                "frame_start": int(frame_min + start),
                "frame_end": int(frame_min + stop - 1),
                "n_frames": int(stop - start),
                "duration_seconds": float((stop - start) / fps),
                "mean_quiet_fraction": float(np.nanmean(quiet)),
                "min_quiet_fraction": float(np.nanmin(quiet)),
                "mean_valid_fraction": float(np.nanmean(valid)),
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "frame_start",
            "frame_end",
            "n_frames",
            "duration_seconds",
            "mean_quiet_fraction",
            "min_quiet_fraction",
            "mean_valid_fraction",
        ],
    )


def _existing_summary(
    out_dir: Path,
    source_metadata_path: Path,
    parameters: dict[str, object],
) -> dict[str, object] | None:
    metadata_path = out_dir / LABEL_METADATA_FILENAME
    required = (
        out_dir / STATE_FILENAME,
        out_dir / QUIET_FRACTION_FILENAME,
        out_dir / VALID_FRACTION_FILENAME,
        out_dir / BOUTS_FILENAME,
        metadata_path,
    )
    if not all(path.is_file() for path in required):
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError):
        return None
    source_stat = source_metadata_path.stat()
    if metadata.get("classifier_parameters") != parameters:
        return None
    if metadata.get("source_metadata_size_bytes") != int(source_stat.st_size):
        return None
    if metadata.get("source_metadata_mtime_ns") != int(source_stat.st_mtime_ns):
        return None
    summary = metadata.get("summary")
    if not isinstance(summary, dict):
        return None
    sleep_fraction = summary.get("sleep_fraction_classified")
    if isinstance(sleep_fraction, float) and not math.isfinite(sleep_fraction):
        return None
    return dict(summary)


def _classify_track(
    cache_dir: Path,
    out_dir: Path,
    parameters: dict[str, object],
) -> dict[str, object]:
    started = time.perf_counter()
    cache = load_sleep_motion_cache(cache_dir)
    n_frames = int(cache.metadata["n_frames"])
    common = {
        "max_gap_frames": int(parameters["max_gap_frames"]),
        "max_bodypoint_speed_mm_s": float(parameters["max_bodypoint_speed_mm_s"]),
    }
    body = aggregate_bodypoint_group_slice(
        cache,
        0,
        n_frames,
        bodypoint_ids=BODY_BODYPOINT_IDS,
        bodypoint_percentile=float(parameters["body_percentile"]),
        min_valid_bodypoint_fraction=float(parameters["min_valid_bodypoint_fraction"]),
        **common,
    )
    antenna = aggregate_bodypoint_group_slice(
        cache,
        0,
        n_frames,
        bodypoint_ids=ANTENNA_BODYPOINT_IDS,
        bodypoint_percentile=float(parameters["antenna_percentile"]),
        min_valid_bodypoint_fraction=float(parameters["min_valid_antenna_fraction"]),
        **common,
    )
    classified = classify_body_antenna_quiet(
        body,
        antenna,
        fps=cache.fps,
        body_speed_threshold_mm_s=float(parameters["body_speed_threshold_mm_s"]),
        antenna_speed_threshold_mm_s=float(parameters["antenna_speed_threshold_mm_s"]),
        window_seconds=float(parameters["window_seconds"]),
        quiet_fraction_threshold=float(parameters["quiet_fraction_threshold"]),
        min_valid_fraction=float(parameters["min_valid_frame_fraction"]),
        require_full_window=True,
    )
    state = classified["state"]
    quiet_fraction = classified["quiet_fraction"]
    valid_fraction = classified["valid_fraction"]
    bouts = _sleep_bouts(
        state,
        quiet_fraction,
        valid_fraction,
        frame_min=cache.frame_min,
        fps=cache.fps,
    )
    n_sleep = int(np.count_nonzero(state == 1))
    n_wake = int(np.count_nonzero(state == 0))
    n_unknown = int(np.count_nonzero(state == -1))
    n_classified = n_sleep + n_wake
    summary: dict[str, object] = {
        "track_name": str(cache.metadata.get("track_name", cache_dir.name)),
        "track_id": cache.metadata.get("track_id"),
        "side": cache.metadata.get("side"),
        "frame_min": cache.frame_min,
        "frame_max": cache.frame_max,
        "n_frames": n_frames,
        "n_cached_frames": int(cache.metadata["n_cached_frames"]),
        "n_sleep_frames": n_sleep,
        "n_wake_frames": n_wake,
        "n_unknown_frames": n_unknown,
        "sleep_fraction_classified": float(n_sleep / n_classified) if n_classified else None,
        "n_sleep_bouts": int(len(bouts)),
        "elapsed_seconds": float(time.perf_counter() - started),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save_array(out_dir / STATE_FILENAME, state.astype(np.int8, copy=False))
    _atomic_save_array(out_dir / QUIET_FRACTION_FILENAME, quiet_fraction.astype(np.float16))
    _atomic_save_array(out_dir / VALID_FRACTION_FILENAME, valid_fraction.astype(np.float16))
    _atomic_write_parquet(out_dir / BOUTS_FILENAME, bouts)
    source_metadata_path = cache_dir / SOURCE_METADATA_FILENAME
    source_stat = source_metadata_path.stat()
    metadata: dict[str, object] = {
        "format_version": 1,
        "classifier_type": "trailing_window_body_antenna_quiet",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sleep_motion_cache": str(cache_dir.resolve()),
        "source_metadata_size_bytes": int(source_stat.st_size),
        "source_metadata_mtime_ns": int(source_stat.st_mtime_ns),
        "classifier_parameters": parameters,
        "frame_min": cache.frame_min,
        "frame_max": cache.frame_max,
        "fps": cache.fps,
        "state_values": {"unknown": -1, "wake": 0, "sleep": 1},
        "array_indexing": "array index 0 corresponds to frame_min",
        "files": {
            "sleep_state": STATE_FILENAME,
            "quiet_fraction": QUIET_FRACTION_FILENAME,
            "valid_fraction": VALID_FRACTION_FILENAME,
            "sleep_bouts": BOUTS_FILENAME,
        },
        "summary": summary,
    }
    _atomic_write_json(out_dir / LABEL_METADATA_FILENAME, metadata)
    return summary


def _sort_key(cache_dir: Path) -> tuple[str, int, str]:
    try:
        metadata = json.loads((cache_dir / SOURCE_METADATA_FILENAME).read_text())
    except (OSError, ValueError):
        return ("", -1, cache_dir.name)
    track_id = metadata.get("track_id")
    return (
        str(metadata.get("side") or ""),
        int(track_id) if track_id is not None else -1,
        cache_dir.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--sleep-motion-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--body-threshold", type=float, default=0.5)
    parser.add_argument("--antenna-threshold", type=float, default=0.7)
    parser.add_argument("--history-seconds", type=float, default=10.0)
    parser.add_argument("--quiet-fraction", type=float, default=0.9)
    parser.add_argument("--min-valid-frame-fraction", type=float, default=0.5)
    parser.add_argument("--body-percentile", type=float, default=75.0)
    parser.add_argument("--antenna-percentile", type=float, default=75.0)
    parser.add_argument("--min-valid-body-fraction", type=float, default=0.75)
    parser.add_argument("--min-valid-antenna-fraction", type=float, default=0.5)
    parser.add_argument("--max-gap-frames", type=int, default=5)
    parser.add_argument("--max-speed", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    block_dir = args.block_dir.expanduser().resolve()
    sleep_motion_root = (
        args.sleep_motion_root.expanduser().resolve()
        if args.sleep_motion_root is not None
        else block_dir / "stitched" / "sleep_motion"
    )
    out_root = (
        args.out_root.expanduser().resolve()
        if args.out_root is not None
        else block_dir / "stitched" / "sleep_motion_labels"
    )
    source_per_track = sleep_motion_root / "per_track"
    cache_dirs = sorted(
        (path for path in source_per_track.iterdir() if path.is_dir() and is_sleep_motion_cache(path)),
        key=_sort_key,
    )
    if args.limit is not None:
        cache_dirs = cache_dirs[: max(0, int(args.limit))]
    if not cache_dirs:
        raise FileNotFoundError(f"No sleep-motion caches found under {source_per_track}")

    parameters = _classifier_parameters(args)
    _validate_parameters(parameters)
    if args.force and out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "per_track").mkdir(parents=True, exist_ok=True)
    (out_root / COMPLETE_FILENAME).unlink(missing_ok=True)

    started = time.perf_counter()
    summaries: list[dict[str, object]] = []
    skipped = 0
    total = len(cache_dirs)
    for index, cache_dir in enumerate(cache_dirs, start=1):
        out_dir = out_root / "per_track" / cache_dir.name
        source_metadata_path = cache_dir / SOURCE_METADATA_FILENAME
        summary = None if args.force else _existing_summary(out_dir, source_metadata_path, parameters)
        if summary is None:
            summary = _classify_track(cache_dir, out_dir, parameters)
            action = "classified"
        else:
            skipped += 1
            action = "cached"
        summaries.append(summary)
        print(
            f"[{index:03d}/{total:03d}] {action:10s} {cache_dir.name} "
            f"({float(summary['elapsed_seconds']):.2f}s)",
            flush=True,
        )

    summary_table = pd.DataFrame.from_records(summaries)
    _atomic_write_parquet(out_root / SUMMARY_FILENAME, summary_table)
    n_frames = int(summary_table["n_frames"].sum())
    n_sleep = int(summary_table["n_sleep_frames"].sum())
    n_wake = int(summary_table["n_wake_frames"].sum())
    n_unknown = int(summary_table["n_unknown_frames"].sum())
    n_classified = n_sleep + n_wake
    run_summary: dict[str, object] = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "block_dir": str(block_dir),
        "sleep_motion_root": str(sleep_motion_root),
        "out_root": str(out_root),
        "classifier_parameters": parameters,
        "n_tracks": int(len(summary_table)),
        "n_tracks_reused": skipped,
        "n_frames": n_frames,
        "n_sleep_frames": n_sleep,
        "n_wake_frames": n_wake,
        "n_unknown_frames": n_unknown,
        "sleep_fraction_classified": float(n_sleep / n_classified) if n_classified else None,
        "elapsed_seconds": float(time.perf_counter() - started),
        "files": {"track_summary": SUMMARY_FILENAME},
    }
    _atomic_write_json(out_root / SUMMARY_METADATA_FILENAME, run_summary)
    (out_root / COMPLETE_FILENAME).write_text(
        f"complete {run_summary['generated_at_utc']} tracks={len(summary_table)}\n"
    )
    sleep_fraction_text = (
        f"{run_summary['sleep_fraction_classified']:.4f}"
        if run_summary["sleep_fraction_classified"] is not None
        else "n/a"
    )
    print(
        f"Complete: {len(summary_table)} tracks, {n_frames:,} frames, "
        f"sleep fraction among classified frames={sleep_fraction_text}, "
        f"elapsed={run_summary['elapsed_seconds']:.1f}s\n{out_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
