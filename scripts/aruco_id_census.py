#!/usr/bin/env python3
"""Summarize seen and missing ArUco IDs from one or more detection outputs.

This is meant for quick colony ID checks after running ``run_aruco.py`` on a
short recording. Absence from a short sample is only "not observed", so use a
minimum-frame threshold and combine a few clips when deciding which physical
tags are safe to reuse.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


DEFAULT_PATTERNS = ("*_aruco_tracks*.h5", "*_aruco_detections*.h5", "*_aruco_detections*.csv")


def parse_id_spec(spec: str) -> list[int]:
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending ID range: {part}")
            ids.update(range(start, end + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def format_id_runs(ids: Iterable[int]) -> str:
    ids = sorted(set(int(i) for i in ids))
    if not ids:
        return "-"

    runs: list[str] = []
    start = prev = ids[0]
    for value in ids[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    runs.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(runs)


def expand_inputs(paths: list[str], patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for pattern in patterns:
                files.extend(p for p in path.rglob(pattern) if p.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)

    # Preserve deterministic order and remove duplicates.
    return sorted(set(p.resolve() for p in files))


def counts_from_raw_h5(path: Path, confidence_threshold: float) -> tuple[Counter[int], int] | None:
    with h5py.File(path, "r") as h5f:
        if "aruco_confidences" not in h5f:
            return None
        conf = h5f["aruco_confidences"][...]

    if conf.ndim != 2:
        raise ValueError(f"{path}: expected aruco_confidences shape (frames, ids), got {conf.shape}")

    frame_count = int(conf.shape[0])
    detected = conf > confidence_threshold
    per_id = detected.sum(axis=0)
    counts = Counter({int(marker_id): int(n) for marker_id, n in enumerate(per_id) if n > 0})
    return counts, frame_count


def counts_from_dataframe(df: pd.DataFrame, path: Path) -> tuple[Counter[int], int]:
    if "Instance" not in df.columns:
        raise ValueError(f"{path}: missing required column 'Instance'")

    counts = Counter(int(v) for v in df["Instance"].dropna().astype(int))
    if "Frame" in df.columns and not df.empty:
        frame_count = int(df["Frame"].nunique())
    else:
        frame_count = 0
    return counts, frame_count


def counts_from_table_h5(path: Path) -> tuple[Counter[int], int] | None:
    try:
        df = pd.read_hdf(path, key="detections")
    except (KeyError, ValueError, OSError):
        return None
    return counts_from_dataframe(df, path)


def counts_from_csv(path: Path) -> tuple[Counter[int], int]:
    df = pd.read_csv(path)
    return counts_from_dataframe(df, path)


def load_counts(path: Path, confidence_threshold: float) -> tuple[Counter[int], int]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return counts_from_csv(path)

    if suffix in {".h5", ".hdf5"}:
        raw = counts_from_raw_h5(path, confidence_threshold)
        if raw is not None:
            return raw
        table = counts_from_table_h5(path)
        if table is not None:
            return table

    raise ValueError(f"{path}: unsupported file type or unrecognized ArUco output")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List observed and missing ArUco IDs from short-recording outputs."
    )
    parser.add_argument("inputs", nargs="+", help="ArUco output files or directories to scan.")
    parser.add_argument(
        "--ids",
        default="0-99",
        help="Expected ID set, e.g. '0-99' for A100 or '0-99,120,131' (default: 0-99).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=3,
        help="Minimum detected frames required to count an ID as present (default: 3).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Raw H5 confidence threshold for counting a frame (default: >0).",
    )
    parser.add_argument(
        "--patterns",
        default=",".join(DEFAULT_PATTERNS),
        help="Comma-separated glob patterns used when an input is a directory.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV report path with ID, frame_count, and status.",
    )
    args = parser.parse_args()

    expected_ids = parse_id_spec(args.ids)
    expected_set = set(expected_ids)
    patterns = tuple(p.strip() for p in args.patterns.split(",") if p.strip())
    files = expand_inputs(args.inputs, patterns)

    total_counts: Counter[int] = Counter()
    total_frames = 0
    per_file: list[tuple[Path, int, int]] = []

    for path in files:
        counts, frame_count = load_counts(path, args.confidence_threshold)
        total_counts.update(counts)
        total_frames += frame_count
        per_file.append((path, frame_count, len(counts)))

    present = {marker_id for marker_id, n in total_counts.items() if n >= args.min_frames}
    weak = {marker_id for marker_id, n in total_counts.items() if 0 < n < args.min_frames}
    missing = expected_set - present
    unexpected = set(total_counts) - expected_set

    rows = []
    for marker_id in expected_ids:
        n = int(total_counts.get(marker_id, 0))
        status = "present" if n >= args.min_frames else "weak" if n > 0 else "missing"
        rows.append({"id": marker_id, "detected_frames": n, "status": status})

    report = pd.DataFrame(rows)
    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.csv_out, index=False)

    print("ArUco ID census")
    print(f"  Files scanned: {len(files)}")
    print(f"  Frames represented: {total_frames}")
    print(f"  Expected IDs: {format_id_runs(expected_ids)}")
    print(f"  Present threshold: >= {args.min_frames} detected frames")
    print()
    print(f"Present ({len(present & expected_set)}): {format_id_runs(present & expected_set)}")
    print(f"Weak ({len(weak & expected_set)}): {format_id_runs(weak & expected_set)}")
    print(f"Missing / not observed ({len(missing)}): {format_id_runs(missing)}")
    if unexpected:
        print(f"Unexpected IDs ({len(unexpected)}): {format_id_runs(unexpected)}")
    if args.csv_out is not None:
        print(f"CSV report: {args.csv_out}")

    if per_file:
        print()
        print("Per-file summary:")
        for path, frame_count, n_ids in per_file:
            print(f"  {path}: frames={frame_count}, ids_seen={n_ids}")


if __name__ == "__main__":
    main()
