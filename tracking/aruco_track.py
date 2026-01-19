#!/usr/bin/env python3
"""Batch-process SLEAP + ArUco detections in a single directory.

The script scans *input_folder* directly (no chunk subfolders). For each dataset
(prefix) found, it combines SLEAP and ArUco detections
(`tracking_utils.get_complete_tracks`) for both the left and right camera views,
writing outputs directly into *output_path* (no chunk subfolders).

Expected filenames (flexible as long as they contain these substrings):
  - SLEAP:  *sleap_panorama_x_left*.pkl   and  *sleap_panorama_x_right*.pkl
  - ArUco:  *aruco_panorama_x_left*.pkl   and  *aruco_panorama_x_right*.pkl

If multiple datasets exist in the same folder, they should differ by a prefix
that appears before "_left"/"_right" in the filename, e.g.:
  sessionA_sleap_panorama_x_left.pkl
  sessionA_aruco_panorama_x_left.pkl
  sessionB_sleap_panorama_x_left.pkl
  ...

Outputs:
  output_path/<prefix>_left.parquet
  output_path/<prefix>_right.parquet

Usage
-----
$ python combine_tracks_flat.py \
      --input_folder /path/to/input \
      --output_path  /path/to/output
"""
import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

# Add the parent directory of this script to the path so that
# ``tracking.tracking_utils`` can be imported when the script lives inside the repo.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tracking.tracking_utils import get_complete_tracks  # noqa: E402


SIDES = ("left", "right")


def _dataset_prefix(path: Path) -> str:
    """
    Extract a dataset prefix by stripping the side token and known detector tokens.

    Example:
      sessionA_sleap_panorama_x_left.pkl  -> sessionA
      chunk000_aruco_panorama_x_right.pkl -> chunk000
    """
    stem = path.stem  # filename without extension
    # Remove trailing _left/_right if present
    stem = re.sub(r"_(left|right)$", "", stem)
    # Remove known middle tokens if present
    stem = stem.replace("_sleap_panorama_x", "")
    stem = stem.replace("_aruco_panorama_x", "")
    return stem


def _pick_one(files: list[Path], label: str) -> Path | None:
    """
    Pick a single file from a list. If multiple, pick the newest by mtime.
    """
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    logging.warning(
        "%s has %d matching files; using newest: %s",
        label,
        len(files_sorted),
        files_sorted[0].name,
    )
    return files_sorted[0]


def _index_files(input_dir: Path) -> dict[str, dict[str, dict[str, Path]]]:
    """
    Build an index:
      index[prefix][side]["sleap"|"aruco"] = Path

    This version *expects* two files per side (one sleap, one aruco) and does not
    confuse them with duplicates.
    """
    index: dict[str, dict[str, dict[str, Path]]] = {}

    # We scan all PKLs once, but we classify them by detector token first.
    all_pkls = list(input_dir.glob("*.pkl"))

    for side in SIDES:
        # Collect detector-specific matches for this side
        sleap_matches = [
            p for p in all_pkls
            if "sleap_panorama_x" in p.name.lower() and f"_{side}" in p.name.lower()
        ]
        aruco_matches = [
            p for p in all_pkls
            if "aruco_panorama_x" in p.name.lower() and f"_{side}" in p.name.lower()
        ]

        # Group by prefix so multiple datasets can coexist in one folder
        # (prefix extraction removes detector tokens and trailing _left/_right)
        by_prefix_sleap: dict[str, list[Path]] = {}
        by_prefix_aruco: dict[str, list[Path]] = {}

        for p in sleap_matches:
            by_prefix_sleap.setdefault(_dataset_prefix(p), []).append(p)
        for p in aruco_matches:
            by_prefix_aruco.setdefault(_dataset_prefix(p), []).append(p)

        prefixes = set(by_prefix_sleap) | set(by_prefix_aruco)
        for prefix in prefixes:
            sleap_file = _pick_one(
                by_prefix_sleap.get(prefix, []),
                label=f"{prefix} side={side} sleap",
            )
            aruco_file = _pick_one(
                by_prefix_aruco.get(prefix, []),
                label=f"{prefix} side={side} aruco",
            )

            index.setdefault(prefix, {}).setdefault(side, {})
            if sleap_file is not None:
                index[prefix][side]["sleap"] = sleap_file
            if aruco_file is not None:
                index[prefix][side]["aruco"] = aruco_file

    return index


def process_dataset(prefix: str, files: dict[str, dict[str, Path]], output_dir: Path) -> None:
    """Process one dataset prefix across left/right, writing directly to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for side in SIDES:
        side_files = files.get(side, {})
        sleap_file = side_files.get("sleap")
        aruco_file = side_files.get("aruco")

        if sleap_file is None or aruco_file is None:
            logging.warning(
                "prefix=%s side=%s: missing file(s) (sleap=%s, aruco=%s) — skipping",
                prefix,
                side,
                sleap_file.name if sleap_file else None,
                aruco_file.name if aruco_file else None,
            )
            continue

        # Load detections
        sleap_det = pd.read_pickle(sleap_file).dropna()
        aruco_det = pd.read_pickle(aruco_file)

        out_parquet = output_dir / f"{prefix}_{side}.parquet"
        get_complete_tracks(
            output_path=str(out_parquet),
            aruco_detection=aruco_det,
            sleap_detection=sleap_det,
        )
        logging.info("Wrote %s", out_parquet)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine SLEAP and ArUco detections into complete tracks by scanning PKL files "
            "directly in the input folder (no chunk subfolders)."
        )
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        type=Path,
        help="Directory containing SLEAP and ArUco PKL files (no chunk* subfolders).",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=Path,
        help="Directory where output parquet files will be written directly.",
    )
    args = parser.parse_args()

    if not args.input_folder.is_dir():
        parser.error(f"input_folder '{args.input_folder}' is not a directory")

    index = _index_files(args.input_folder)
    if not index:
        logging.error(
            "No matching PKL files found in %s. Expected '*sleap_panorama_x_<side>*.pkl' and "
            "'*aruco_panorama_x_<side>*.pkl' for side in {left,right}.",
            args.input_folder,
        )
        sys.exit(1)

    for prefix, files in sorted(index.items()):
        process_dataset(prefix, files, args.output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
