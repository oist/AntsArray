#!/usr/bin/env python3
import argparse
import glob
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tracking.tracking_utils import get_complete_tracks  # noqa: E402

SIDES = ("left", "right")
TS_RE = re.compile(r"^\d{8}-\d{6}$")          # 20251211-104517
CHUNK_RE = re.compile(r"^chunk\d+$", re.I)    # chunk000, chunk12, etc.


def find_unique(pattern: str) -> Path | None:
    matches = glob.glob(pattern)
    if len(matches) == 1:
        return Path(matches[0])
    if len(matches) > 1:
        logging.warning("Multiple files match %s → skipping", pattern)
    return None


def _dataset_id_from_chunk_dir(chunk_dir: Path) -> str:
    """
    Derive datasetID from the nearest parent directory named like a timestamp (YYYYMMDD-HHMMSS).
    Falls back to chunk_dir.parent.name if no timestamp parent exists.
    """
    for p in [chunk_dir] + list(chunk_dir.parents):
        if TS_RE.match(p.name):
            return p.name
    return chunk_dir.parent.name


def _chunk_id_from_chunk_dir(chunk_dir: Path) -> str:
    """
    Use chunk_dir.name if it looks like chunk###; otherwise use chunk_dir.name as-is.
    """
    name = chunk_dir.name
    if CHUNK_RE.match(name):
        return name
    return name


def process_chunk(chunk_dir: Path, output_root: Path) -> None:
    """
    Writes flat outputs into output_root with naming:
      <datasetID>-<chunkID>_<groupSuffix>.parquet

    groupSuffix here is simply "left" or "right".
    """
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_id = _dataset_id_from_chunk_dir(chunk_dir)
    chunk_id = _chunk_id_from_chunk_dir(chunk_dir)

    for side in SIDES:
        sleap_file = find_unique(str(chunk_dir / f"*sleap_panorama_x_{side}*.pkl"))
        aruco_file = find_unique(str(chunk_dir / f"*aruco_panorama_x_{side}*.pkl"))
        if sleap_file is None or aruco_file is None:
            logging.warning("%s: missing %s view files — skipping", chunk_dir.name, side)
            continue

        sleap_det = pd.read_pickle(sleap_file).dropna()
        aruco_det = pd.read_pickle(aruco_file)

        out_parquet = output_root / f"{dataset_id}-{chunk_id}_{side}.parquet"
        get_complete_tracks(
            output_path=str(out_parquet),
            aruco_detection=aruco_det,
            sleap_detection=sleap_det,
        )
        logging.info("Wrote %s", out_parquet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_dir", required=True, type=Path, help="Directory for one chunk (contains pkl files).")
    parser.add_argument("--output_path", required=True, type=Path, help="Flat output directory.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process_chunk(args.chunk_dir, args.output_path)


if __name__ == "__main__":
    main()
