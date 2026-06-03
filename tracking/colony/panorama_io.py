#!/usr/bin/env python3
"""
Shared I/O helpers for colony panorama-stage tracking files.

Historically, map_combine wrote SLEAP panorama PKLs as raw pandas DataFrames and
ArUco panorama PKLs as {"detections": DataFrame, "num_frames": int}. These
helpers accept both the legacy format and a normalized dict payload.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

SIDES = ("left", "right")
KEY_RE = re.compile(r"^(.+_chunk[0-9]{3})", re.IGNORECASE)
ARUCO_INPUT_RE = re.compile(
    r"""
    ^cam(?P<cam>\d+)
    _cam\d+
    _\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}
    _(?P<chunk>\d{3})
    (?P<suffix>_aruco_tracks_?|_aruco_detections)
    \.(?:h5|hdf5)$
    """,
    re.VERBOSE,
)
ARUCO_SUFFIXES = ("_aruco_tracks_", "_aruco_tracks", "_aruco_detections")


def extract_key(filename: str) -> str | None:
    m = KEY_RE.match(filename)
    return m.group(1) if m else None


def infer_side(filename: str) -> str | None:
    fn = filename.lower()
    for side in SIDES:
        if f"_x_{side}" in fn or f"_{side}" in fn:
            return side
    return None


def pick_one(matches: list[Path], label: str) -> Path | None:
    """Pick lexicographically first to match batch discovery behavior."""
    if not matches:
        return None
    matches_sorted = sorted(matches, key=lambda p: p.name)
    if len(matches_sorted) > 1:
        logging.warning(
            "%s has %d matching files; using: %s",
            label,
            len(matches_sorted),
            matches_sorted[0].name,
        )
    return matches_sorted[0]


def is_aruco_input_file(path: Path) -> bool:
    return path.is_file() and ARUCO_INPUT_RE.match(path.name) is not None


def aruco_base_stem(path: Path) -> str:
    for suffix in ARUCO_SUFFIXES:
        if path.stem.endswith(suffix):
            return path.stem[: -len(suffix)]
    raise ValueError(f"Could not derive ArUco base stem from {path.name}")


def matching_sleap_h5_candidates(aruco_path: Path) -> tuple[Path, Path]:
    base = aruco_base_stem(aruco_path)
    return (
        aruco_path.with_name(f"{base}_sleap_data.h5"),
        aruco_path.with_name(f"{base}_sleap_data.hdf5"),
    )


def find_aruco_input_files(data_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(data_dir.glob("**/*"))
        if "global" not in path.name and is_aruco_input_file(path)
    ]


def validate_aruco_inputs_have_sleap_h5(data_dir: Path) -> list[Path]:
    aruco_files = find_aruco_input_files(data_dir)
    missing: list[tuple[Path, tuple[Path, Path]]] = []

    for aruco_path in aruco_files:
        candidates = matching_sleap_h5_candidates(aruco_path)
        if not any(candidate.is_file() for candidate in candidates):
            missing.append((aruco_path, candidates))

    if missing:
        lines = [
            "Every ArUco H5 must have a matching SLEAP H5 before the colony pipeline can run.",
            f"Checked {len(aruco_files)} ArUco files in {data_dir}; missing {len(missing)} SLEAP files.",
        ]
        for aruco_path, candidates in missing[:20]:
            expected = " or ".join(str(candidate) for candidate in candidates)
            lines.append(f"- {aruco_path} -> expected {expected}")
        if len(missing) > 20:
            lines.append(f"- ... {len(missing) - 20} more missing matches")
        raise FileNotFoundError("\n".join(lines))

    return aruco_files


def unwrap_panorama_payload(path: Path, *, detector: str) -> tuple[pd.DataFrame, int | None]:
    payload: Any = pd.read_pickle(path)

    if isinstance(payload, pd.DataFrame):
        return payload, None

    if not isinstance(payload, dict) or "detections" not in payload:
        raise TypeError(
            f"{detector} PKL did not contain a DataFrame or expected dict payload: "
            f"{path} (type={type(payload)})"
        )

    det = payload["detections"]
    if not isinstance(det, pd.DataFrame):
        raise TypeError(
            f"{detector} payload['detections'] is not a DataFrame: "
            f"{path} (type={type(det)})"
        )

    num_frames_raw = payload.get("num_frames")
    num_frames = None if num_frames_raw is None else int(num_frames_raw)
    return det, num_frames


def load_sleap_pkl(path: Path) -> pd.DataFrame:
    det, _num_frames = unwrap_panorama_payload(path, detector="SLEAP")
    return det


def load_aruco_pkl(path: Path) -> tuple[pd.DataFrame, int]:
    det, num_frames = unwrap_panorama_payload(path, detector="ARUCO")
    if num_frames is None or num_frames <= 0:
        raise ValueError(f"ARUCO payload missing positive num_frames: {path}")
    return det, int(num_frames)


def make_panorama_payload(
    detections: pd.DataFrame,
    *,
    detector: str,
    num_frames: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "detections": detections,
        "detector": detector,
    }
    if num_frames is not None:
        payload["num_frames"] = int(num_frames)
    return payload
