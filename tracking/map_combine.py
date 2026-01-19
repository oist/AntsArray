#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map ArUco & SLEAP detections from many camera chunks into a common panorama per chunk.

Simplifications
---------------
- ArUco inputs are ALWAYS H5/HDF5 (no CSV support).
- num_frames is ALWAYS read from H5 dataset "aruco_tracks" (shape[0]) and is never None.
- Outputs are written into ONE FLAT output directory (no chunk subfolders).

Updated policy (as requested)
-----------------------------
- If num_frames differs across cameras within a chunk:
    * emit a WARNING
    * use the LARGEST num_frames across all cameras as the chunk's num_frames

Outputs
-------
ArUco panorama outputs are pickles containing a dict:
  {"detections": <pd.DataFrame>, "num_frames": <int>}

SLEAP panorama outputs remain DataFrame pickles (unchanged).

Downstream reading example (ArUco):
  payload = pd.read_pickle(aruco_pkl)
  aruco_df = payload["detections"]
  num_frames = payload["num_frames"]
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------#
#                                CONFIGURATION                                 #
# -----------------------------------------------------------------------------#

CONFIG = dict(
    hmats_npz="/home/sam-reiter/bucket/ReiterU/Ants/basler/2025_Sep_no_pertubation/calibration_dataset/set0_patterns_elevated_by_2mm/initial_H_mats.npz",
    data_dir="/home/sam-reiter/bucket/ReiterU/Ants/basler/20251117_2_stim/data/",
    output_dir="/home/sam-reiter/bucket/ReiterU/Ants/basler/20251117_2_stim/",
)

X_THRESHOLD: float = 1740.0

# -----------------------------------------------------------------------------#
#                                   HELPERS                                    #
# -----------------------------------------------------------------------------#


def load_homographies(npz_file: Path | str) -> List[np.ndarray]:
    npz_file = Path(npz_file)
    if npz_file.suffix != ".npz":
        raise ValueError(f"Homography file must be .npz (got {npz_file})")

    data = np.load(npz_file)
    if "H" not in data:
        raise KeyError(".npz file lacks key 'H' containing homography stack")

    H_stack = data["H"]
    if H_stack.ndim != 3 or H_stack.shape[1:] != (3, 3):
        raise ValueError("'H' in .npz must have shape (n_cam, 3, 3)")

    return [H_stack[i] for i in range(H_stack.shape[0])]


def apply_homography(xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.hstack([xy, np.ones((xy.shape[0], 1))])
    proj = pts @ H.T
    return proj[:, :2] / proj[:, [2]]


def split_and_write_flat(df: pd.DataFrame, out_dir: Path, base_name: str) -> None:
    """
    Write DF pickles (SLEAP outputs), split into left/right, into a flat out_dir.
    """
    left = df[df["X"] < X_THRESHOLD]
    right = df[df["X"] >= X_THRESHOLD]

    if not left.empty:
        left_file = out_dir / f"{base_name}_x_left{int(X_THRESHOLD)}.pkl"
        left.to_pickle(left_file)
    if not right.empty:
        right_file = out_dir / f"{base_name}_x_right{int(X_THRESHOLD)}.pkl"
        right.to_pickle(right_file)


def split_and_write_with_num_frames_flat(
    df: pd.DataFrame,
    out_dir: Path,
    base_name: str,
    num_frames: int,
) -> None:
    """
    Write dict payload with detections + num_frames (ArUco outputs),
    split into left/right, into a flat out_dir.
    """
    left = df[df["X"] < X_THRESHOLD]
    right = df[df["X"] >= X_THRESHOLD]

    if not left.empty:
        left_file = out_dir / f"{base_name}_x_left{int(X_THRESHOLD)}.pkl"
        pd.to_pickle({"detections": left, "num_frames": int(num_frames)}, left_file)

    if not right.empty:
        right_file = out_dir / f"{base_name}_x_right{int(X_THRESHOLD)}.pkl"
        pd.to_pickle({"detections": right, "num_frames": int(num_frames)}, right_file)


def aruco_h5_to_long_df_full(
    f: h5py.File,
    ds_name: str = "aruco_tracks",
    frame_offset: int = 0,
) -> pd.DataFrame:
    arr = f[ds_name][...]
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"Expected (frames, instances, 2); got {arr.shape}")

    valid = np.isfinite(arr).all(axis=2)
    valid &= ~((arr[..., 0] == 0.0) & (arr[..., 1] == 0.0))

    fr_idx, inst_idx = np.nonzero(valid)
    if fr_idx.size == 0:
        return pd.DataFrame(columns=["Frame", "Instance", "X", "Y"])

    return pd.DataFrame(
        {
            "Frame": (fr_idx + frame_offset).astype(np.int32),
            "Instance": inst_idx.astype(np.int16),
            "X": arr[fr_idx, inst_idx, 0].astype(np.float64),
            "Y": arr[fr_idx, inst_idx, 1].astype(np.float64),
        }
    )


def _filter_instances_by_frame_fraction(
    df: pd.DataFrame,
    *,
    min_instance_frame_frac: float,
) -> pd.DataFrame:
    """
    Keep only instances that appear in at least `min_instance_frame_frac` of frames.
    Uses unique frame counts per Instance (robust to duplicates).
    """
    if not (0.0 <= min_instance_frame_frac <= 1.0):
        raise ValueError("--min_instance_frame_frac must be in [0, 1].")

    if df.empty or min_instance_frame_frac <= 0.0:
        return df

    total_frames = df["Frame"].nunique()
    if total_frames == 0:
        return df

    min_frames = int(np.ceil(min_instance_frame_frac * total_frames))
    if min_frames <= 1:
        return df

    instance_frame_counts = df.groupby("Instance")["Frame"].nunique()
    keep_instances = instance_frame_counts.index[instance_frame_counts >= min_frames]
    return df[df["Instance"].isin(keep_instances)].copy()


def _load_aruco_h5_to_df_and_num_frames(
    file: Path,
    *,
    frame_offset: int = 0,
    min_instance_frame_frac: float = 0.05,
    ds_name: str = "aruco_tracks",
) -> Tuple[pd.DataFrame, int]:
    """
    ArUco inputs are ALWAYS H5/HDF5.
    Returns (df, num_frames) where num_frames is f[ds_name].shape[0].
    """
    suf = file.suffix.lower()
    if suf not in {".h5", ".hdf5"}:
        raise ValueError(f"Expected .h5/.hdf5 ArUco input, got {file}")

    with h5py.File(file, "r") as f:
        num_frames = int(f[ds_name].shape[0])
        df = aruco_h5_to_long_df_full(f, ds_name=ds_name, frame_offset=frame_offset)

    df = _filter_instances_by_frame_fraction(df, min_instance_frame_frac=min_instance_frame_frac)
    return df, num_frames


def group_files_by_chunk(
    root: Path,
    pattern,
    ignore_substr: str = "global",
) -> Dict[str, Dict[int, Path]]:
    groups: Dict[str, Dict[int, Path]] = defaultdict(dict)
    is_compiled = hasattr(pattern, "search")

    for path in root.glob("**/*"):
        if path.is_dir() or ignore_substr in path.name:
            continue
        m = pattern.search(path.name) if is_compiled else re.search(pattern, path.name)
        if not m:
            continue
        cam = int(m.group(1))
        chunk = m.group(2)
        groups[chunk][cam] = path

    return {
        ck: dict(sorted(cams.items()))
        for ck, cams in sorted(groups.items(), key=lambda x: int(x[0]))
    }


def process_aruco_chunks(
    hmats: List[np.ndarray],
    aruco_dir: Path,
    out_dir: Path,
    exp: str,
    *,
    min_instance_frame_frac: float,
) -> None:
    """
    Reads per-camera H5 files, homographies into panorama coordinates, concatenates per chunk,
    and writes left/right pickles WITH num_frames into flat out_dir.
    """
    patt = re.compile(
        r"""
        ^cam(?P<cam>\d+)
        _cam\d+
        _\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}
        _(?P<chunk>\d{3})
        (?:_aruco_tracks_)?
        (?:_aruco_detections)?
        \.(?:h5|hdf5)$
        """,
        re.VERBOSE,
    )

    chunk_map: Dict[int, Dict[int, Path]] = group_files_by_chunk(aruco_dir, patt)

    for chunk, cam_files in tqdm(chunk_map.items(), desc="ArUco chunks"):
        dfs: List[pd.DataFrame] = []
        num_frames_vals: List[int] = []

        for cam_idx, file in cam_files.items():
            df, n_frames = _load_aruco_h5_to_df_and_num_frames(
                file, min_instance_frame_frac=min_instance_frame_frac
            )
            num_frames_vals.append(int(n_frames))

            if df.empty:
                continue

            xy = df[["X", "Y"]].to_numpy(float)
            df[["X", "Y"]] = apply_homography(xy, hmats[cam_idx - 1])  # zero-based
            df["Cam"] = cam_idx - 1  # zero-based
            dfs.append(df)

        if not dfs:
            logging.info("No ArUco detections in chunk %s — skipped.", chunk)
            continue

        uniq = sorted(set(num_frames_vals))
        if len(uniq) != 1:
            logging.warning(
                "Chunk %s: inconsistent num_frames across cams: %s. Using max=%d.",
                chunk,
                uniq,
                max(num_frames_vals),
            )
        num_frames = max(num_frames_vals)

        panorama_df = pd.concat(dfs, ignore_index=True)
        base = f"{exp}_chunk{chunk}_aruco_panorama"

        split_and_write_with_num_frames_flat(panorama_df, out_dir, base, num_frames=num_frames)
        logging.info("ArUco panorama (chunk %s) → %s (num_frames=%s)", chunk, out_dir, num_frames)


def process_sleap_chunks(hmats: List[np.ndarray], sleap_dir: Path, out_dir: Path, exp: str) -> None:
    patt = re.compile(
        r"^cam(\d+)_cam\d+_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_(\d{3})(?:_sleap_data)?\.csv$"
    )

    chunk_map = group_files_by_chunk(sleap_dir, patt)

    for chunk, cam_files in tqdm(chunk_map.items(), desc="SLEAP chunks"):
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
            panorama_df = pd.concat(dfs, ignore_index=True)
            base = f"{exp}_chunk{chunk}_sleap_panorama"
            split_and_write_flat(panorama_df, out_dir, base)
            logging.info("SLEAP panorama (chunk %s) → %s", chunk, out_dir)
        else:
            logging.info("No SLEAP data in chunk %s — skipped.", chunk)


def infer_experiment_name(data_dir: Path) -> str:
    pattern = re.compile(
        r"""
        (\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})
        _(\d{3})
        """,
        re.VERBOSE,
    )

    for file in data_dir.iterdir():
        if not file.is_file():
            continue

        m = pattern.search(file.name)
        if m:
            yyyy, mm, dd, HH, MM, SS, _ = m.groups()
            return f"{yyyy}{mm}{dd}_{HH}{MM}{SS}"

    raise RuntimeError("Could not infer experiment name from files in data_dir.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Map & concatenate ArUco/SLEAP per chunk (requires .npz homographies)"
    )

    p.add_argument("--hmats", default=CONFIG["hmats_npz"], help=".npz file with homography stack (key 'H')")
    p.add_argument("--data_dir", default=CONFIG["data_dir"], help="directory with per-camera per-chunk detection files")
    p.add_argument("--outdir", default=CONFIG["output_dir"], help="flat output directory to write pickles")

    p.add_argument("--mode", choices=("aruco", "sleap", "both"), default="both")
    p.add_argument(
        "--min_instance_frame_frac",
        type=float,
        default=0.05,
        help="Drop ArUco Instances that appear in fewer than this fraction of frames (0..1). Default: 0.05",
    )

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hmats = load_homographies(args.hmats)

    data_dir = Path(args.data_dir)
    exp = infer_experiment_name(data_dir)

    if args.mode in ("aruco", "both"):
        process_aruco_chunks(
            hmats,
            data_dir,
            out_dir,
            exp,
            min_instance_frame_frac=args.min_instance_frame_frac,
        )

    if args.mode in ("sleap", "both"):
        process_sleap_chunks(hmats, data_dir, out_dir, exp)


if __name__ == "__main__":
    main()
