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
SLEAP_INPUT_RE = re.compile(
    r"""
    ^cam(?P<cam>\d+)
    _cam\d+
    _\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}
    _(?P<chunk>\d{3})
    _sleap_data
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
        for path in sorted(data_dir.glob("*"))
        if "global" not in path.name and is_aruco_input_file(path)
    ]


def find_sleap_input_files(data_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(data_dir.glob("*"))
        if "global" not in path.name and path.is_file() and SLEAP_INPUT_RE.match(path.name) is not None
    ]


def discover_complete_input_chunks(data_dir: Path) -> tuple[list[str], dict[str, object]]:
    """Return contiguous complete chunks from chunk000 until the first incomplete chunk.

    A chunk is complete when every expected camera has both ArUco and SLEAP
    H5/HDF5 inputs. The expected camera set comes from the union of ArUco and
    SLEAP files in chunk000, so chunk000 is rejected if a camera has only SLEAP
    or only ArUco. Later chunks may be incomplete or absent; they are ignored
    once the first gap/incomplete chunk is reached.
    """
    aruco_files_by_chunk: dict[str, dict[int, Path]] = {}
    sleap_files_by_chunk: dict[str, dict[int, Path]] = {}
    duplicate_aruco = 0
    for path in find_aruco_input_files(data_dir):
        match = ARUCO_INPUT_RE.match(path.name)
        if match is None:
            continue
        chunk = match.group("chunk")
        cam = int(match.group("cam"))
        existing = aruco_files_by_chunk.setdefault(chunk, {}).get(cam)
        if existing is not None:
            duplicate_aruco += 1
            if "_aruco_detections" in existing.name:
                continue
            if "_aruco_detections" not in path.name:
                continue
        aruco_files_by_chunk[chunk][cam] = path

    for path in find_sleap_input_files(data_dir):
        match = SLEAP_INPUT_RE.match(path.name)
        if match is None:
            continue
        chunk = match.group("chunk")
        cam = int(match.group("cam"))
        sleap_files_by_chunk.setdefault(chunk, {})[cam] = path

    if not aruco_files_by_chunk and not sleap_files_by_chunk:
        return [], {
            "reason": "no ArUco/SLEAP H5 files found",
            "chunks_seen": 0,
            "reference_cameras": [],
            "duplicate_aruco_files": duplicate_aruco,
        }

    reference_cameras = set(aruco_files_by_chunk.get("000", {}).keys()) | set(
        sleap_files_by_chunk.get("000", {}).keys()
    )
    if not reference_cameras:
        return [], {
            "reason": "chunk000 missing",
            "chunks_seen": len(set(aruco_files_by_chunk) | set(sleap_files_by_chunk)),
            "reference_cameras": [],
            "duplicate_aruco_files": duplicate_aruco,
        }

    complete_chunks: list[str] = []
    first_incomplete: dict[str, object] | None = None
    chunk_idx = 0
    while True:
        chunk = f"{chunk_idx:03d}"
        cam_files = aruco_files_by_chunk.get(chunk)
        sleap_files = sleap_files_by_chunk.get(chunk)
        if cam_files is None and sleap_files is None:
            first_incomplete = {
                "chunk": chunk,
                "reason": "missing chunk",
            }
            break

        cam_files = cam_files or {}
        sleap_files = sleap_files or {}
        aruco_cams = set(cam_files)
        sleap_cams = set(sleap_files)
        observed_cams = aruco_cams | sleap_cams
        missing_cams = sorted(reference_cameras - observed_cams)
        extra_cams = sorted(observed_cams - reference_cameras)
        missing_aruco_cams = sorted(reference_cameras - aruco_cams)
        missing_sleap_cams = sorted(reference_cameras - sleap_cams)

        if missing_cams or extra_cams or missing_aruco_cams or missing_sleap_cams:
            first_incomplete = {
                "chunk": chunk,
                "reason": "incomplete chunk",
                "missing_cameras": missing_cams,
                "extra_cameras": extra_cams,
                "missing_aruco_cameras": missing_aruco_cams,
                "missing_sleap_cameras": missing_sleap_cams,
            }
            break

        complete_chunks.append(chunk)
        chunk_idx += 1

    summary: dict[str, object] = {
        "chunks_seen": len(set(aruco_files_by_chunk) | set(sleap_files_by_chunk)),
        "reference_cameras": sorted(reference_cameras),
        "reference_camera_count": len(reference_cameras),
        "complete_chunk_count": len(complete_chunks),
        "duplicate_aruco_files": duplicate_aruco,
        "first_incomplete": first_incomplete,
    }
    return complete_chunks, summary


def validate_aruco_inputs_have_sleap_h5(data_dir: Path) -> list[Path]:
    complete_chunks, summary = discover_complete_input_chunks(data_dir)
    if not complete_chunks:
        lines = [
            "No complete ArUco/SLEAP chunk sequence is available for the colony pipeline.",
            f"Checked {summary.get('chunks_seen', 0)} ArUco chunks in {data_dir}.",
            f"Reason: {summary.get('reason') or summary.get('first_incomplete')}",
        ]
        raise FileNotFoundError("\n".join(lines))

    chunk_set = set(complete_chunks)
    return [
        path
        for path in find_aruco_input_files(data_dir)
        if (match := ARUCO_INPUT_RE.match(path.name)) is not None
        and match.group("chunk") in chunk_set
    ]


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
