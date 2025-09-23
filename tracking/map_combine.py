#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map ArUco & SLEAP detections from many camera chunks into a common panorama **per chunk**.

For every temporal chunk (e.g. "000", "001" …) this script merges detections
across all cameras and writes the results into a dedicated directory:

    <OUTDIR>/chunkXYZ/

Within each chunk directory four pickle files are produced:

    <EXP>_chunkXYZ_aruco_panorama_x_lt1000.pkl   # X < 1000
    <EXP>_chunkXYZ_aruco_panorama_x_ge1000.pkl   # X ≥ 1000
    <EXP>_chunkXYZ_sleap_panorama_x_lt1000.pkl   # X < 1000
    <EXP>_chunkXYZ_sleap_panorama_x_ge1000.pkl   # X ≥ 1000

The threshold (default 1000) can be adjusted by editing the constant
``X_THRESHOLD`` below.
"""

from __future__ import annotations
import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

# -----------------------------------------------------------------------------#
#                                CONFIGURATION                                 #
# -----------------------------------------------------------------------------#

CONFIG = dict(
    # Homographies exported by refine_homographies.py → contains array "H"
    hmats_npz="/home/sam-reiter/bucket/ReiterU/sam/ant_tracking/frame_1/initial_H_mats.npz",
    data_dir="/home/sam-reiter/bucket/ReiterU/Ants/basler/20250321_2_test/data",
    output_dir="/home/sam-reiter/bucket/ReiterU/Ants/basler/20250321_2_test/",
)

# Split threshold along X‑axis in panorama coordinate system
X_THRESHOLD: float = 1000.0

# -----------------------------------------------------------------------------#
#                                   HELPERS                                    #
# -----------------------------------------------------------------------------#

def load_homographies(npz_file: Path | str) -> List[np.ndarray]:
    """Return list of homography matrices from ``.npz`` file (key ``H``)."""

    npz_file = Path(npz_file)
    if npz_file.suffix != ".npz":
        raise ValueError("Homography file must be .npz (got %s)" % npz_file)

    data = np.load(npz_file)
    if "H" not in data:
        raise KeyError(".npz file lacks key 'H' containing homography stack")

    H_stack = data["H"]
    if H_stack.ndim != 3 or H_stack.shape[1:] != (3, 3):
        raise ValueError("'H' in .npz must have shape (n_cam, 3, 3)")

    return [H_stack[i] for i in range(H_stack.shape[0])]


def apply_homography(xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project points ``xy`` (N×2) using homography ``H`` (3×3)."""
    pts = np.hstack([xy, np.ones((xy.shape[0], 1))])
    proj = pts @ H.T
    return proj[:, :2] / proj[:, [2]]


def split_and_write(df: pd.DataFrame, chunk_dir: Path, base_name: str) -> None:
    """Split dataframe at ``X_THRESHOLD`` and write two pickle files."""
    left = df[df["X"] < X_THRESHOLD]
    right = df[df["X"] >= X_THRESHOLD]

    if not left.empty:
        left_file = chunk_dir / f"{base_name}_x_left{int(X_THRESHOLD)}.pkl"
        left.to_pickle(left_file)
    if not right.empty:
        right_file = chunk_dir / f"{base_name}_x_right{int(X_THRESHOLD)}.pkl"
        right.to_pickle(right_file)

# -----------------------------------------------------------------------------#
#                      FILE DISCOVERY & GROUPING BY CHUNK                      #
# -----------------------------------------------------------------------------#


def group_files_by_chunk(
    root: Path,
    pattern: str,
    ignore_substr: str = "global",
) -> Dict[str, Dict[int, Path]]:
    """Walk *root* and return ``{chunk: {cam: file}}``.

    *pattern* must contain two capture groups:
      1) camera index, 2) chunk index.
    """
    groups: Dict[str, Dict[int, Path]] = defaultdict(dict)
    for path in root.glob("**/*"):
        if path.is_dir() or ignore_substr in path.name:
            continue
        m = re.search(pattern, path.name)
        if not m:
            continue
        cam = int(m.group(1))
        chunk = m.group(2)
        groups[chunk][cam] = path

    # ensure deterministic ordering
    return {ck: dict(sorted(cams.items())) for ck, cams in sorted(groups.items(), key=lambda x: int(x[0]))}

# -----------------------------------------------------------------------------#
#                         CHUNK‑WISE COMBINATION ROUTINES                      #
# -----------------------------------------------------------------------------#

def process_aruco_chunks(hmats: List[np.ndarray], aruco_dir: Path, out_dir: Path, exp: str):
    patt = r"cam(\d{2})_cam\d+_[A-Za-z0-9_-]+_(\d{3})_aruco\.csv$"
    chunk_map = group_files_by_chunk(aruco_dir, patt)

    for chunk, cam_files in tqdm(chunk_map.items(), desc="ArUco chunks"):
        dfs = []
        for cam_idx, file in cam_files.items():
            df = pd.read_csv(file)
            if df.empty:
                continue
            xy = df[["X", "Y"]].to_numpy(float)
            df[["X", "Y"]] = apply_homography(xy, hmats[cam_idx - 1])
            df["Cam"] = cam_idx - 1
            dfs.append(df)

        if dfs:
            chunk_dir = out_dir / f"chunk{chunk}"
            chunk_dir.mkdir(parents=True, exist_ok=True)

            panorama_df = pd.concat(dfs, ignore_index=True)
            base = f"{exp}_chunk{chunk}_aruco_panorama"
            split_and_write(panorama_df, chunk_dir, base)
            print(f"ArUco panorama (chunk {chunk}) → {chunk_dir}")
        else:
            print(f"No ArUco data in chunk {chunk} — skipped.")


def process_sleap_chunks(hmats: List[np.ndarray], sleap_dir: Path, out_dir: Path, exp: str):
    patt = r"cam(\d{2})_cam\d+_[A-Za-z0-9_-]+_(\d{3})_sleap_data\.h5$"
    chunk_map = group_files_by_chunk(sleap_dir, patt)

    for chunk, cam_files in tqdm(chunk_map.items(), desc="SLEAP chunks"):
        dfs = []
        for cam_idx, file in cam_files.items():
            with h5py.File(file, "r") as f:
                df = pd.DataFrame({
                    "Frame": f["Frame"][:],
                    "Instance": f["Instance"][:],
                    "Bodypoint": f["Bodypoint"][:],
                    "X": f["X"][:],
                    "Y": f["Y"][:],
                })
            if df.empty:
                continue
            xy = df[["X", "Y"]].to_numpy(float)
            df[["X", "Y"]] = apply_homography(xy, hmats[cam_idx - 1])
            df["Cam"] = cam_idx - 1
            dfs.append(df)

        if dfs:
            chunk_dir = out_dir / f"chunk{chunk}"
            chunk_dir.mkdir(parents=True, exist_ok=True)

            panorama_df = pd.concat(dfs, ignore_index=True)
            base = f"{exp}_chunk{chunk}_sleap_panorama"
            split_and_write(panorama_df, chunk_dir, base)
            print(f"SLEAP panorama (chunk {chunk}) → {chunk_dir}")
        else:
            print(f"No SLEAP data in chunk {chunk} — skipped.")

# -----------------------------------------------------------------------------#
#                                   CLI                                        #
# -----------------------------------------------------------------------------#

def infer_experiment_name(data_dir: Path) -> str:
    for file in data_dir.iterdir():
        m = re.match(r"cam\d+_cam\d+_(.+?)_\d{3}(?:_aruco)?\.csv$", file.name)
        if m:
            return m.group(1)
    raise RuntimeError("Could not infer experiment name from files in data_dir.")


def main() -> None:
    p = argparse.ArgumentParser(description="Map & concatenate ArUco/SLEAP per chunk (requires .npz homographies)")
    p.add_argument("--hmats", default=CONFIG["hmats_npz"], help=".npz file with homography stack (key 'H')")
    p.add_argument("--data_dir", default=CONFIG["data_dir"], help="directory with per-camera per-chunk detection files")
    p.add_argument("--outdir", default=CONFIG["output_dir"], help="where to write output pickles")
    args = p.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hmats = load_homographies(args.hmats)

    data_dir = Path(args.data_dir)
    exp = infer_experiment_name(data_dir)

    process_aruco_chunks(hmats, data_dir, out_dir, exp)
    process_sleap_chunks(hmats, data_dir, out_dir, exp)


if __name__ == "__main__":
    main()
