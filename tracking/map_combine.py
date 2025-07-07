#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map ArUco & SLEAP detections from many camera chunks into a common panorama.

  1. Load bundle-adjustment parameters  →  homography for each camera.
  2. For every chunk ('*_000.pkl', '*_001.csv', …) of each camera:
       • load detections
       • project (x, y) with the relevant H
       • append into global table
  3. Save two pickle files:
       <EXP>_aruco_panorama.pkl, <EXP>_sleap_panorama.pkl
Author : Sam (2024-12-21) • cleaned & simplified July 2025
"""

from __future__ import annotations
import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py


# -----------------------------------------------------------------------------#
#                                CONFIGURATION                                 #
# -----------------------------------------------------------------------------#

CONFIG = dict(
    hmats_mat="/home/sam-reiter/bucket/ReiterU/sam/ant_tracking/bundle_adjustment_paras.mat",
    data_dir="/home/sam-reiter/bucket/ReiterU/Ants/basler/20250321_2_test/data",
    output_dir="/home/sam-reiter/bucket/ReiterU/Ants/basler/20250321_2_test/",
)

# -----------------------------------------------------------------------------#
#                                   HELPERS                                    #
# -----------------------------------------------------------------------------#


def load_homographies(
    mat_file: Path | str,
    n_cam: int = 25,
    reference_cam: int = 12,  # same ref as in original script
) -> List[np.ndarray]:
    """
    Return H_list[i] that maps camera-i image coords → panorama reference.

    The MATLAB `.mat` file contains an array `paras` per original code.
    The `.npz` file contains a pre-computed array `H` from refine_homographies.py.
    """
    mat_file = Path(mat_file)
    
    if mat_file.suffix == ".npz":
        # Load pre-computed homographies from refine_homographies.py
        data = np.load(mat_file)
        H_stack = data["H"]
        return [H_stack[i] for i in range(H_stack.shape[0])]
    
    # Original MATLAB .mat file loading
    paras = np.squeeze(scipy.io.loadmat(mat_file)["paras"])
    # H_pair[0][i] maps cam-i → cam-0
    H_pair = [[np.eye(3) if i == j else None for j in range(n_cam)] for i in range(n_cam)]
    for i in range(1, n_cam):
        p = paras[4 * (i - 1) : 4 * i]
        S = np.array([[p[0], p[1], p[2]], [p[1], p[0], p[3]], [0, 0, 1]])
        H_pair[0][i] = S
        H_pair[i][0] = np.linalg.inv(S)

    # derive every pair i↔j via cam-0
    for i in range(1, n_cam):
        for j in range(i + 1, n_cam):
            H_pair[i][j] = H_pair[0][j] @ H_pair[i][0]
            H_pair[j][i] = np.linalg.inv(H_pair[i][j])

    # Panorama reference is cam `reference_cam`
    return [H_pair[reference_cam][i] for i in range(n_cam)]


def apply_homography(xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project `xy` (N×2) with homography H."""
    pts = np.hstack([xy, np.ones((xy.shape[0], 1))])
    proj = pts @ H.T
    return proj[:, :2] / proj[:, [2]]


# -----------------------------------------------------------------------------#
#                            ARUCO / SLEAP COMBINERS                           #
# -----------------------------------------------------------------------------#


def collect_chunk_files(
    root: Path,
    pattern: str,
    ignore_substr: str = "global",
) -> Dict[str, Dict[int, List[Path]]]:
    """
    Walk `root` and group files by chunk‐index & camera‐index.

    pattern – regex with two capturing groups:
              1) camera index, 2) chunk index (e.g. 000, 001 …)

    Returns ``{chunk: {cam: [paths,…]}}``.
    """
    groups: Dict[str, Dict[int, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in root.glob("**/*"):
        if path.is_dir() or ignore_substr in path.name:
            continue
        m = re.search(pattern, path.name)
        if not m:
            continue
        # Use the first camera number (first capture group) as the camera index
        cam, chunk = int(m.group(1)), m.group(3)
        groups[cam][chunk].append(path)
    
    # Sort cameras in numerical order and chunks within each camera
    sorted_groups = {}
    for cam in sorted(groups.keys()):
        sorted_groups[cam] = dict(sorted(groups[cam].items(), key=lambda x: int(x[0])))
    return sorted_groups


def combine_aruco(
    hmats: List[np.ndarray], aruco_dir: Path, out_file: Path
) -> None:
    """
    Concatenate all `*_aruco.csv` files (per-camera, per-chunk) into one panorama pickle.
    """
    patt = r"cam(\d{2})_cam(\d{1,2})_.*_(\d{3})_aruco.csv$"
    chunk_map = collect_chunk_files(aruco_dir, patt)
    dfs = []
    for cam_idx, chunks in tqdm(chunk_map.items(), desc="ArUco cameras"):
        frame_offset = 0  # Track cumulative frame offset for this camera
        for chunk, files in chunks.items():
            file = files[0]  # one csv per cam per chunk
            df = pd.read_csv(file)
            if df.empty:
                continue
            # Adjust frame numbers to be continuous across chunks
            df['Frame'] = df['Frame'] + frame_offset
            # Update frame offset for next chunk
            if not df.empty:
                frame_offset = df['Frame'].max() + 1
            
            xy = df[["X", "Y"]].to_numpy(float)
            # Use the first camera number (cam_idx) directly as the camera index
            df[["X", "Y"]] = apply_homography(xy, hmats[cam_idx-1]) #cam_idx-1 is the index of the camera in the hmats list
            df["Cam"] = cam_idx-1
            dfs.append(df)
            
    if not dfs:
        print("No ArUco data found.")
        return
    pd.concat(dfs, ignore_index=True).to_pickle(out_file)
    print(f"ArUco panorama written → {out_file}")


def combine_sleap(
    hmats: List[np.ndarray], sleap_dir: Path, out_file: Path
) -> None:
    """
    Concatenate all SLEAP `.h5` files (per-camera, per-chunk) into one panorama pickle.
    """
    patt = r"cam(\d{2})_cam(\d{1,2})_.*_(\d{3})\_sleap_data.h5$"
    chunk_map = collect_chunk_files(sleap_dir, patt)
    
    dfs = []
    for cam_idx, chunks in tqdm(chunk_map.items(), desc="SLEAP cameras"):
        frame_offset = 0  # Track cumulative frame offset for this camera
        for chunk, files in chunks.items():
            file = files[0]  # one h5 per cam per chunk
            # Load SLEAP HDF5 data as structured in sleap2h5.py
            with h5py.File(file, 'r') as f:
                df_dict = {
                    'Frame': f['Frame'][:],
                    'Instance': f['Instance'][:],
                    'Bodypoint': f['Bodypoint'][:],
                    'X': f['X'][:],
                    'Y': f['Y'][:]
                }
                df = pd.DataFrame(df_dict)
            if df.empty:
                continue
            # Adjust frame numbers to be continuous across chunks
            df['Frame'] = df['Frame'] + frame_offset
            # Update frame offset for next chunk
            if not df.empty:
                frame_offset = df['Frame'].max() + 1
            
            # Extract x, y coordinates from SLEAP format
            xy = df[["X", "Y"]].to_numpy(float)
            # Use the first camera number (cam_idx) directly as the camera index
            df[["X", "Y"]] = apply_homography(xy, hmats[cam_idx-1])
            df["Cam"] = cam_idx-1
            dfs.append(df)
    
    if not dfs:
        print("No SLEAP data found.")
        return
    pd.concat(dfs, ignore_index=True).to_pickle(out_file)
    print(f"SLEAP panorama written → {out_file}")


def plot_panorama(aruco_pkl: str | Path, out_png: str | Path | None = None, max_points: int | None = 100_000):
    """
    Plot mapped ArUco centres in the panorama frame.
    """
  
    df_aru = pd.read_pickle(aruco_pkl)
    if max_points:
        df_aru = df_aru.sample(min(len(df_aru), max_points), random_state=0)

    plt.figure(figsize=(10, 8))
    plt.scatter(df_aru["X"], df_aru["Y"], s=4, c="tab:blue", alpha=0.7, label="ArUco")
    plt.xlabel("panorama X (px)")
    plt.ylabel("panorama Y (px)")
    plt.title("Mapped ArUco detections")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=300)
        print(f"Panorama figure saved → {out_png}")
        plt.close()
    else:
        plt.show()


# -----------------------------------------------------------------------------#
#                                   CLI                                        #
# -----------------------------------------------------------------------------#

def infer_experiment_name(data_dir: Path) -> str:
    # Look for a file like camXX_camYY_<EXP>_001_aruco.csv or camXX_camYY_<EXP>_001.csv
    for file in data_dir.iterdir():
        m = re.match(r"cam\d+_cam\d+_(.+)_\d{3}(?:_aruco)?\.csv$", file.name)
        if m:
            return m.group(1)
    raise RuntimeError("Could not infer experiment name from files in data_dir.")

def main() -> None:
    p = argparse.ArgumentParser(description="Map & concatenate ArUco chunks and plot panorama")
    p.add_argument("--hmats", default=CONFIG["hmats_mat"])
    p.add_argument("--refined_hmats", 
                   help="Optional refined homography file (.npz) from refine_homographies.py")
    p.add_argument("--data_dir", default=CONFIG["data_dir"])
    p.add_argument("--outdir", default=CONFIG["output_dir"])
    args = p.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use refined homographies if provided, otherwise use original
    hmats_file = args.refined_hmats if args.refined_hmats else args.hmats
    hmats = load_homographies(hmats_file)
    
    data_dir = Path(args.data_dir)
    exp_name = infer_experiment_name(data_dir)

    combine_aruco(
        hmats,
        data_dir,
        out_dir / f"{exp_name}_aruco_panorama.pkl",
    )

    combine_sleap(
        hmats,
        data_dir,
        out_dir / f"{exp_name}_sleap_panorama.pkl",
    )

    plot_panorama( out_dir / f"{exp_name}_aruco_panorama.pkl", out_png=out_dir / f"{exp_name}_panorama.png")


if __name__ == "__main__":
    main()
