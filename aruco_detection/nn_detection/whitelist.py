"""Session whitelist utilities for ArUco ID filtering.

Restricts decoded marker IDs to a known set of valid IDs for a given
experiment session.  This reduces wrong-ID false positives by rejecting
decodes that land on IDs not actually present in the arena.

Three ways to obtain a whitelist:
1. Load from a JSON file  (``load_whitelist``)
2. Auto-discover from an existing ArUco CSV  (``discover_whitelist``)
3. Auto-discover from an ArUco H5 tracks file  (``discover_whitelist_from_h5``)

Usage:
    from aruco_detection.nn_detection.whitelist import (
        load_whitelist,
        discover_whitelist,
    )

    wl = load_whitelist("session_ids.json")
    # or
    wl = discover_whitelist("cam04_aruco.csv", min_occurrences=10)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_whitelist(path: str | Path) -> set[int]:
    """Load a whitelist from a JSON file.

    Accepted formats::

        [3, 17, 25, ...]                 # plain list
        {"valid_ids": [3, 17, 25, ...]}   # dict with "valid_ids" key
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return {int(x) for x in data}
    if isinstance(data, dict) and "valid_ids" in data:
        return {int(x) for x in data["valid_ids"]}
    raise ValueError(
        f"Unrecognised whitelist format in {path}. "
        "Expected a JSON list or {{\"valid_ids\": [...]}}."
    )


def save_whitelist(ids: set[int], path: str | Path) -> None:
    """Save a whitelist to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"valid_ids": sorted(ids)}, f, indent=2)


def discover_whitelist(
    aruco_csv_path: str | Path,
    min_occurrences: int = 10,
    max_ids: int = 200,
) -> set[int]:
    """Auto-discover valid marker IDs from an ArUco detection CSV.

    The CSV is expected to have columns: Frame, Instance, X, Y, Confidence
    (the standard output of ``run_aruco.py``).

    Parameters
    ----------
    aruco_csv_path : path
        Path to the ArUco CSV file.
    min_occurrences : int
        An ID must appear in at least this many rows to be whitelisted.
    max_ids : int
        Safety cap — if more IDs pass the threshold, keep only the
        *max_ids* most frequent ones.
    """
    df = pd.read_csv(aruco_csv_path)
    counts = df["Instance"].value_counts()
    valid = counts[counts >= min_occurrences]
    if len(valid) > max_ids:
        valid = valid.head(max_ids)
    return set(valid.index.astype(int))


def discover_whitelist_from_h5(
    h5_path: str | Path,
    min_frames: int = 10,
    max_ids: int = 200,
) -> set[int]:
    """Auto-discover valid marker IDs from an ArUco H5 tracks file.

    The H5 file is expected to contain an ``aruco_tracks`` dataset with
    shape ``(num_frames, 1000, 2)``.  An ID is valid if it has nonzero
    coordinates in at least *min_frames* frames.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        tracks = f["aruco_tracks"][:]  # (num_frames, 1000, 2)

    # Nonzero in either X or Y → detected
    detected = np.any(tracks != 0, axis=2)  # (num_frames, 1000)
    frame_counts = detected.sum(axis=0)  # (1000,)

    valid_mask = frame_counts >= min_frames
    ids = np.where(valid_mask)[0]

    if len(ids) > max_ids:
        # Keep most frequent
        order = np.argsort(-frame_counts[ids])
        ids = ids[order[:max_ids]]

    return set(int(x) for x in ids)
