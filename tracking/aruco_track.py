#!/usr/bin/env python3
"""Batch‑process SLEAP + ArUco detections.

The script scans an *input_folder* for sub‑directories named ``chunk000``, ``chunk001`` …
etc.  For every chunk it combines SLEAP and ArUco detections (``tracking_utils.get_complete_tracks``)
for both the *left* and *right* camera views, writing the resulting dataframes to identically
named sub‑directories created under *output_path* (e.g. ``output_path/chunk000/chunk000_left.pkl``).

Usage
-----
$ python combine_tracks_chunks.py \
      --input_folder /path/to/input \
      --output_path  /path/to/output
"""
import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add the parent directory of this script to the path so that
# ``tracking.tracking_utils`` can be imported when the script lives inside the repo.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tracking.tracking_utils import get_complete_tracks  # noqa: E402


CHUNK_GLOB = "chunk*"  # pattern that identifies a chunk directory
SIDES = ("left", "right")


def find_unique(pattern: str) -> Path | None:
    """Return the single Path matching *pattern*, or *None* if zero/ambiguous."""
    matches = glob.glob(pattern)
    if len(matches) == 1:
        return Path(matches[0])
    if len(matches) > 1:
        logging.warning("Multiple files match %s → skipping", pattern)
    return None


def process_chunk(chunk_dir: Path, output_root: Path) -> None:
    """Process one chunk directory, writing outputs to *output_root/chunk_dir.name*."""
    out_dir = output_root / chunk_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for side in SIDES:
        sleap_pat = chunk_dir / f"*sleap_panorama_x_{side}*.pkl"
        aruco_pat = chunk_dir / f"*aruco_panorama_x_{side}*.pkl"

        sleap_file = find_unique(str(sleap_pat))
        aruco_file = find_unique(str(aruco_pat))
        if sleap_file is None or aruco_file is None:
            logging.warning("%s: missing %s view files — skipping", chunk_dir.name, side)
            continue

        # Load detections
        sleap_det = pd.read_pickle(sleap_file).dropna()
        aruco_det = pd.read_pickle(aruco_file)

        # Combine to complete tracks and save
        out_pkl = out_dir / f"{chunk_dir.name}_{side}.pkl"
        get_complete_tracks(
            output_path=str(out_pkl),
            aruco_detection=aruco_det,
            sleap_detection=sleap_det,
        )
        logging.info("Wrote %s", out_pkl)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine SLEAP and ArUco detections into complete tracks for each chunk "
            "sub‑folder found in the input folder."
        )
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        type=Path,
        help="Directory containing chunk000, chunk001, … sub‑folders.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=Path,
        help="Root directory where output chunk folders will be created.",
    )
    args = parser.parse_args()

    if not args.input_folder.is_dir():
        parser.error(f"input_folder '{args.input_folder}' is not a directory")

    chunk_dirs = sorted(p for p in args.input_folder.glob(CHUNK_GLOB) if p.is_dir())
    if not chunk_dirs:
        logging.error("No chunk directories matching '%s' found in %s", CHUNK_GLOB, args.input_folder)
        sys.exit(1)

    for chunk_dir in chunk_dirs:
        process_chunk(chunk_dir, args.output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
