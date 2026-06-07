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
import gc
import logging
import os
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
    hmats_npz="/bucket/ReiterU/Ants/basler/2025_Sep_no_pertubation/calibration_dataset/set0_patterns_elevated_by_2mm/initial_H_mats.npz",
    data_dir="/bucket/ReiterU/Ants/basler/20251117_2_stim/data/",
    output_dir="/bucket/ReiterU/Ants/basler/20251117_2_stim/",
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


def _split_output_paths(out_dir: Path, base_name: str) -> tuple[Path, Path]:
    return (
        out_dir / f"{base_name}_x_left{int(X_THRESHOLD)}.pkl",
        out_dir / f"{base_name}_x_right{int(X_THRESHOLD)}.pkl",
    )


def _both_split_outputs_exist(out_dir: Path, base_name: str) -> bool:
    left_file, right_file = _split_output_paths(out_dir, base_name)
    return left_file.exists() and right_file.exists()


def _atomic_pickle(obj, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pd.to_pickle(obj, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def split_and_write_flat(
    df: pd.DataFrame,
    out_dir: Path,
    base_name: str,
    *,
    skip_existing: bool = False,
    anchor_bodypoint: int = 0,
) -> None:
    """
    Write DF pickles (SLEAP outputs), split into left/right, into a flat out_dir.
    """
    left, right = split_sleap_by_anchor(df, anchor_bodypoint=anchor_bodypoint)
    left_file, right_file = _split_output_paths(out_dir, base_name)

    if not left.empty:
        if skip_existing and left_file.exists():
            logging.info("Skipping existing %s", left_file)
        else:
            _atomic_pickle(left, left_file)
    if not right.empty:
        if skip_existing and right_file.exists():
            logging.info("Skipping existing %s", right_file)
        else:
            _atomic_pickle(right, right_file)


def split_sleap_by_anchor(
    df: pd.DataFrame,
    *,
    anchor_bodypoint: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a SLEAP DataFrame into left/right while keeping each skeleton together.

    This avoids a full DataFrame merge on ["Frame", "Instance"], which is too
    memory-heavy for large colony chunks. Side is assigned from the anchor
    bodypoint when present; rows without an anchor fall back to their own X.
    """
    if df.empty:
        return df.copy(), df.copy()

    required = {"Frame", "Instance", "Bodypoint", "X"}
    if not required.issubset(df.columns):
        return df[df["X"] < X_THRESHOLD].copy(), df[df["X"] >= X_THRESHOLD].copy()

    frame = df["Frame"].to_numpy(dtype=np.int64, copy=False)
    inst = df["Instance"].to_numpy(dtype=np.int64, copy=False)
    bodypoint = df["Bodypoint"].to_numpy(dtype=np.int64, copy=False)
    x = df["X"].to_numpy(dtype=np.float64, copy=False)

    side_x = x.copy()
    anchor_mask = bodypoint == int(anchor_bodypoint)
    if np.any(anchor_mask):
        stride = max(int(inst.max()) + 1, 1)
        keys = frame * stride + inst
        anchor_keys = keys[anchor_mask]
        anchor_x = x[anchor_mask]
        finite = np.isfinite(anchor_x)
        anchor_keys = anchor_keys[finite]
        anchor_x = anchor_x[finite]

        if anchor_keys.size:
            order = np.argsort(anchor_keys)
            sorted_keys = anchor_keys[order]
            sorted_x = anchor_x[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
            unique_keys = sorted_keys[starts]
            counts = np.diff(np.r_[starts, sorted_keys.size])
            anchor_mean_x = np.add.reduceat(sorted_x, starts) / counts

            pos = np.searchsorted(unique_keys, keys)
            valid_pos = pos < unique_keys.size
            matched = np.zeros(keys.shape, dtype=bool)
            matched[valid_pos] = unique_keys[pos[valid_pos]] == keys[valid_pos]
            side_x[matched] = anchor_mean_x[pos[matched]]

    left_mask = side_x < X_THRESHOLD
    return df.loc[left_mask].copy(), df.loc[~left_mask].copy()


def split_and_write_with_num_frames_flat(
    df: pd.DataFrame,
    out_dir: Path,
    base_name: str,
    num_frames: int,
    *,
    skip_existing: bool = False,
) -> None:
    """
    Write dict payload with detections + num_frames (ArUco outputs),
    split into left/right, into a flat out_dir.
    """
    left = df[df["X"] < X_THRESHOLD]
    right = df[df["X"] >= X_THRESHOLD]
    left_file, right_file = _split_output_paths(out_dir, base_name)

    if not left.empty:
        if skip_existing and left_file.exists():
            logging.info("Skipping existing %s", left_file)
        else:
            _atomic_pickle({"detections": left, "num_frames": int(num_frames)}, left_file)

    if not right.empty:
        if skip_existing and right_file.exists():
            logging.info("Skipping existing %s", right_file)
        else:
            _atomic_pickle({"detections": right, "num_frames": int(num_frames)}, right_file)

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


def _matching_aruco_tracks_file(file: Path) -> Path | None:
    stem = file.stem
    if stem.endswith("_aruco_tracks_") or stem.endswith("_aruco_tracks"):
        return file
    if not stem.endswith("_aruco_detections"):
        return None

    base = stem[: -len("_aruco_detections")]
    for suffix in ("_aruco_tracks.h5", "_aruco_tracks_.h5", "_aruco_tracks.hdf5", "_aruco_tracks_.hdf5"):
        candidate = file.with_name(f"{base}{suffix}")
        if candidate.is_file():
            return candidate
    return None


def _load_aruco_detections_h5_to_df_and_num_frames(
    file: Path,
    *,
    min_instance_frame_frac: float = 0.05,
    ds_name: str = "aruco_tracks",
) -> Tuple[pd.DataFrame, int]:
    df = pd.read_hdf(file, key="detections")
    required = ["Frame", "Instance", "X", "Y"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{file} missing ArUco detection columns: {missing}")

    df = df.copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    df[["Frame", "Instance"]] = df[["Frame", "Instance"]].astype(int)

    tracks_file = _matching_aruco_tracks_file(file)
    if tracks_file is None:
        num_frames = int(df["Frame"].max() + 1) if not df.empty else 0
        logging.warning(
            "No matching dense ArUco tracks file for %s; using max(Frame)+1=%d for num_frames.",
            file.name,
            num_frames,
        )
    else:
        with h5py.File(tracks_file, "r") as f:
            num_frames = int(f[ds_name].shape[0])

    df = _filter_instances_by_frame_fraction(df, min_instance_frame_frac=min_instance_frame_frac)
    return df, num_frames


def _load_aruco_input_to_df_and_num_frames(
    file: Path,
    *,
    min_instance_frame_frac: float = 0.05,
) -> Tuple[pd.DataFrame, int]:
    if file.stem.endswith("_aruco_detections"):
        return _load_aruco_detections_h5_to_df_and_num_frames(
            file,
            min_instance_frame_frac=min_instance_frame_frac,
        )
    return _load_aruco_h5_to_df_and_num_frames(
        file,
        min_instance_frame_frac=min_instance_frame_frac,
    )


def group_files_by_chunk(
    root: Path,
    pattern,
    ignore_substr: str = "global",
) -> Dict[str, Dict[int, Path]]:
    groups: Dict[str, Dict[int, Path]] = defaultdict(dict)
    is_compiled = hasattr(pattern, "search")

    for path in sorted(root.glob("*")):
        if path.is_dir() or ignore_substr in path.name:
            continue
        m = pattern.search(path.name) if is_compiled else re.search(pattern, path.name)
        if not m:
            continue
        cam = int(m.group(1))
        chunk = m.group(2)
        existing = groups[chunk].get(cam)
        if existing is not None:
            is_aruco_conflict = "_aruco_" in existing.name or "_aruco_" in path.name
            if is_aruco_conflict:
                if "_aruco_detections" in existing.name:
                    continue
                if "_aruco_detections" not in path.name:
                    continue
            elif existing.suffix.lower() in {".h5", ".hdf5"}:
                continue
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
    chunks: set[str] | None = None,
    skip_existing: bool = False,
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
        (?:_aruco_tracks_?)?
        (?:_aruco_detections)?
        \.(?:h5|hdf5)$
        """,
        re.VERBOSE,
    )

    chunk_map: Dict[int, Dict[int, Path]] = group_files_by_chunk(aruco_dir, patt)
    if chunks is not None:
        chunk_map = {chunk: files for chunk, files in chunk_map.items() if chunk in chunks}

    for chunk, cam_files in tqdm(chunk_map.items(), desc="ArUco chunks"):
        base = f"{exp}_chunk{chunk}_aruco_panorama"
        if skip_existing and _both_split_outputs_exist(out_dir, base):
            logging.info("ArUco panorama chunk %s already exists; skipped.", chunk)
            continue

        dfs: List[pd.DataFrame] = []
        num_frames_vals: List[int] = []

        for cam_idx, file in cam_files.items():
            df, n_frames = _load_aruco_input_to_df_and_num_frames(
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

        split_and_write_with_num_frames_flat(
            panorama_df,
            out_dir,
            base,
            num_frames=num_frames,
            skip_existing=skip_existing,
        )
        logging.info("ArUco panorama (chunk %s) → %s (num_frames=%s)", chunk, out_dir, num_frames)


def _normalize_sleap_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Frame", "Instance", "Bodypoint", "X", "Y"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"SLEAP data missing columns: {missing}")

    out = df.copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Score_node" in out.columns:
        out["Score_node"] = pd.to_numeric(out["Score_node"], errors="coerce")
    out = out.dropna(subset=required)
    out[["Frame", "Instance", "Bodypoint"]] = out[["Frame", "Instance", "Bodypoint"]].astype(int)
    return out


def _load_sleap_file(file: Path) -> pd.DataFrame:
    suffix = file.suffix.lower()
    if suffix == ".csv":
        return _normalize_sleap_df(pd.read_csv(file))

    if suffix not in {".h5", ".hdf5"}:
        raise ValueError(f"Expected SLEAP .h5/.hdf5/.csv input, got {file}")

    try:
        return _normalize_sleap_df(pd.read_hdf(file, key="sleap_data"))
    except Exception:
        pass

    with h5py.File(file, "r") as f:
        if "sleap_data" in f:
            arr = f["sleap_data"][:]
            if getattr(arr.dtype, "names", None):
                return _normalize_sleap_df(pd.DataFrame.from_records(arr))

        required = ["Frame", "Instance", "Bodypoint", "X", "Y"]
        missing = [name for name in required if name not in f]
        if missing:
            raise ValueError(f"{file} missing datasets: {missing}")
        data = {name: np.squeeze(f[name][:]) for name in required}
        if "Score_node" in f:
            data["Score_node"] = np.squeeze(f["Score_node"][:])
        return _normalize_sleap_df(pd.DataFrame(data))


def process_sleap_chunks(
    hmats: List[np.ndarray],
    sleap_dir: Path,
    out_dir: Path,
    exp: str,
    *,
    chunks: set[str] | None = None,
    skip_existing: bool = False,
) -> None:
    patt = re.compile(
        r"^cam(\d+)_cam\d+_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_(\d{3})(?:_sleap_data)?\.(?:h5|hdf5|csv)$"
    )

    chunk_map = group_files_by_chunk(sleap_dir, patt)
    if chunks is not None:
        chunk_map = {chunk: files for chunk, files in chunk_map.items() if chunk in chunks}

    for chunk, cam_files in tqdm(chunk_map.items(), desc="SLEAP chunks"):
        base = f"{exp}_chunk{chunk}_sleap_panorama"
        if skip_existing and _both_split_outputs_exist(out_dir, base):
            logging.info("SLEAP panorama chunk %s already exists; skipped.", chunk)
            continue

        left_parts: list[pd.DataFrame] = []
        right_parts: list[pd.DataFrame] = []
        for cam_idx, file in cam_files.items():
            df = _load_sleap_file(file)
            if df.empty:
                continue
            xy = df[["X", "Y"]].to_numpy(float)
            df[["X", "Y"]] = apply_homography(xy, hmats[cam_idx - 1])
            del xy
            df["Cam"] = cam_idx - 1
            left, right = split_sleap_by_anchor(df)
            if not left.empty:
                left_parts.append(left)
            if not right.empty:
                right_parts.append(right)
            del df, left, right
            gc.collect()

        if left_parts or right_parts:
            left_file, right_file = _split_output_paths(out_dir, base)
            if left_parts:
                if skip_existing and left_file.exists():
                    logging.info("Skipping existing %s", left_file)
                else:
                    left_out = (
                        left_parts[0]
                        if len(left_parts) == 1
                        else pd.concat(left_parts, ignore_index=True, copy=False)
                    )
                    _atomic_pickle(left_out, left_file)
                    del left_out
                left_parts.clear()
            if right_parts:
                if skip_existing and right_file.exists():
                    logging.info("Skipping existing %s", right_file)
                else:
                    right_out = (
                        right_parts[0]
                        if len(right_parts) == 1
                        else pd.concat(right_parts, ignore_index=True, copy=False)
                    )
                    _atomic_pickle(right_out, right_file)
                    del right_out
                right_parts.clear()
            gc.collect()
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
        "--chunk",
        action="append",
        help="Only process this chunk number, e.g. 000. May be passed more than once.",
    )
    p.add_argument(
        "--min_instance_frame_frac",
        type=float,
        default=0.05,
        help="Drop ArUco Instances that appear in fewer than this fraction of frames (0..1). Default: 0.05",
    )
    p.add_argument("--skip_existing", action="store_true", help="Do not overwrite existing panorama PKLs.")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hmats = load_homographies(args.hmats)

    data_dir = Path(args.data_dir)
    exp = infer_experiment_name(data_dir)
    chunks = None
    if args.chunk:
        chunks = {str(chunk).zfill(3) for chunk in args.chunk}

    if args.mode in ("aruco", "both"):
        process_aruco_chunks(
            hmats,
            data_dir,
            out_dir,
            exp,
            min_instance_frame_frac=args.min_instance_frame_frac,
            chunks=chunks,
            skip_existing=args.skip_existing,
        )

    if args.mode in ("sleap", "both"):
        process_sleap_chunks(hmats, data_dir, out_dir, exp, chunks=chunks, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
