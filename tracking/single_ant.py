#!/usr/bin/env python3
"""
Combine SLEAP CSVs into per-arena track files across ALL chunks, using ArUco HDF5
to define TRUE chunk lengths and prevent frame drift.

Folder layout
-------------
Given --input_folder ROOT, we iterate over immediate subfolders whose names match
YYYYMMDD-HHMMSS:

  ROOT/20251211-104517/data/
    - camX_###_sleap_data.csv
    - camX_###_aruco_tracks_.h5

Segmentation images are searched under:
  ROOT/seg_arena/
and can have any name after the initial camXX, e.g.
  cam05_cam4_2025-12-18-16-35-12.png

Key correctness guarantee
-------------------------
Frame offsets across chunks are computed from the ArUco HDF5:
  f["aruco_tracks"].shape[0]  == true number of frames in that video chunk

This is independent of SLEAP detection sparsity and cannot drift.

Diagnostics
-----------
For each (run, camera), prints a per-chunk table:
  - chunk id
  - true video frame count
  - min / max / unique SLEAP frames
  - global frame span assigned to that chunk

Outputs
-------
Flat Parquet files in --out_dir:
  <timestamp>_<cam>_<arena>.parquet
  <timestamp>_<cam>_<arena>_track-<Track>.parquet

Segmentation fallback
---------------------
If no seg file is found for a camera, assume a single arena ("Arena00") covering
all pixels, and assign all detections to that arena.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Regex / naming
# -----------------------------------------------------------------------------
CAM_RE = re.compile(r"(?:^|_)cam(\d+)(?:_|$)")
CHUNK_RE = re.compile(r"_(\d{3})(?:_sleap_data)?\.csv$")
TS_RE = re.compile(r"^\d{8}-\d{6}$")  # YYYYMMDD-HHMMSS


# -----------------------------------------------------------------------------
# CSV discovery
# -----------------------------------------------------------------------------
def parse_cam_and_chunk(csv_path: Path) -> Tuple[str, int]:
    name = csv_path.name

    m_cam = CAM_RE.search(name)
    if not m_cam:
        raise ValueError(f"Could not extract cam number from filename: {name}")
    cam_id = f"cam{m_cam.group(1)}"

    m_chunk = CHUNK_RE.search(name)
    if not m_chunk:
        raise ValueError(f"Could not extract chunk id from filename: {name}")
    chunk_id = int(m_chunk.group(1))

    return cam_id, chunk_id


def list_csvs_grouped_by_cam(sleap_dir: Path) -> Dict[str, List[Path]]:
    cam_map: Dict[str, List[Path]] = {}
    for p in sleap_dir.glob("*.csv"):
        try:
            cam_id, _ = parse_cam_and_chunk(p)
        except ValueError:
            continue
        cam_map.setdefault(cam_id, []).append(p)

    for cam_id, files in cam_map.items():
        cam_map[cam_id] = sorted(files, key=lambda fp: parse_cam_and_chunk(fp)[1])

    return dict(sorted(cam_map.items()))


# -----------------------------------------------------------------------------
# ArUco frame counts (ground truth)
# -----------------------------------------------------------------------------
def aruco_n_frames(h5_path: Path, ds_name: str = "aruco_tracks") -> int:
    with h5py.File(h5_path, "r") as f:
        return int(f[ds_name].shape[0])


# -----------------------------------------------------------------------------
# Segmentation helpers
# -----------------------------------------------------------------------------
def read_seg_bgr(seg_path: Path) -> np.ndarray:
    seg = cv2.imread(str(seg_path), cv2.IMREAD_COLOR)
    if seg is None:
        raise FileNotFoundError(f"Could not read seg image: {seg_path}")
    return seg


def unique_nonzero_colors(seg_bgr: np.ndarray) -> List[Tuple[int, int, int]]:
    flat = seg_bgr.reshape(-1, 3)
    colors = np.unique(flat, axis=0)
    return sorted(
        [(int(b), int(g), int(r)) for b, g, r in colors if (b, g, r) != (0, 0, 0)]
    )


def build_color_to_label(colors: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int, int], str]:
    return {c: f"Arena{i:02d}" for i, c in enumerate(colors)}


def sample_label_at_xy(
    seg: np.ndarray,
    x: float,
    y: float,
    color_to_label: Dict[Tuple[int, int, int], str],
) -> str:
    h, w = seg.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return "UNKNOWN"
    b, g, r = map(int, seg[yi, xi])
    if (b, g, r) == (0, 0, 0):
        return "BACKGROUND"
    return color_to_label.get((b, g, r), "UNKNOWN")


def assign_arenas(
    df: pd.DataFrame,
    seg: np.ndarray,
    color_to_label: Dict[Tuple[int, int, int], str],
    xcol: str,
    ycol: str,
) -> pd.DataFrame:
    df = df.dropna(subset=[xcol, ycol]).copy()
    labels = [
        sample_label_at_xy(seg, float(x), float(y), color_to_label)
        for x, y in df[[xcol, ycol]].to_numpy(float)
    ]
    df["ArenaLabel"] = labels
    return df


def find_seg_file_for_cam(seg_dir: Path, cam_id: str) -> Optional[Path]:
    """
    Find a segmentation image for a camera in seg_dir.

    Requirements:
      - seg_dir is ROOT/seg_arena (ROOT is --input_folder)
      - filenames start with camXX, but can have any suffix, e.g.
          cam05_cam4_2025-12-18-16-35-12.png

    Selection rule:
      - If multiple matches exist, pick the most recently modified file.
      - If none exist, return None.
    """
    if not seg_dir.exists() or not seg_dir.is_dir():
        return None

    matches = list(seg_dir.glob(f"{cam_id}*.png"))
    if not matches:
        return None

    return max(matches, key=lambda p: p.stat().st_mtime)


def get_segmentation_or_default(
    seg_path: Optional[Path],
) -> Tuple[Optional[np.ndarray], Dict[Tuple[int, int, int], str], List[str]]:
    """
    Returns (seg_image_or_None, color_to_label, arenas).

    If seg_path is None, does not exist, OR contains no nonzero colors, fall back to a single arena:
      arenas = ["Arena00"]
      seg = None
      color_to_label = {}
    """
    if seg_path is None or not seg_path.exists():
        return None, {}, ["Arena00"]

    seg = read_seg_bgr(seg_path)
    colors = unique_nonzero_colors(seg)

    if len(colors) == 0:
        return None, {}, ["Arena00"]

    color_to_label = build_color_to_label(colors)
    arenas = sorted(color_to_label.values())
    return seg, color_to_label, arenas


def assign_arenas_with_fallback(
    df: pd.DataFrame,
    seg: Optional[np.ndarray],
    color_to_label: Dict[Tuple[int, int, int], str],
    xcol: str,
    ycol: str,
    fallback_label: str = "Arena00",
) -> pd.DataFrame:
    """
    If seg is None (or color_to_label is empty), assign every row to fallback_label.
    Otherwise, sample seg colors at (x, y) to assign ArenaLabel.

    Note: preserves existing behavior of dropping rows missing X/Y.
    """
    df = df.dropna(subset=[xcol, ycol]).copy()

    if seg is None or not color_to_label:
        df["ArenaLabel"] = fallback_label
        return df

    labels = [
        sample_label_at_xy(seg, float(x), float(y), color_to_label)
        for x, y in df[[xcol, ycol]].to_numpy(float)
    ]
    df["ArenaLabel"] = labels
    return df


# -----------------------------------------------------------------------------
# Frame offsetting using ArUco + diagnostics
# -----------------------------------------------------------------------------
def apply_frame_offsets_using_aruco(
    dfs: List[pd.DataFrame],
    chunk_ids: List[int],
    chunk_frame_counts: List[int],
    frame_col: str,
    label: str,
) -> List[pd.DataFrame]:
    if not (len(dfs) == len(chunk_ids) == len(chunk_frame_counts)):
        raise ValueError("dfs, chunk_ids, chunk_frame_counts must have same length")

    offset = 0
    out: List[pd.DataFrame] = []

    print("\n" + "=" * 100)
    print(f"FRAME OFFSET DIAGNOSTICS {label}".strip())
    print("=" * 100)
    print(
        "chunk | video_frames | sleap_min | sleap_max | sleap_unique | global_start | global_end"
    )
    print("-" * 100)

    for df, cid, n_frames in zip(dfs, chunk_ids, chunk_frame_counts):
        if df.empty or frame_col not in df.columns:
            print(
                f"{cid:>5} | {n_frames:>12} |"
                f" {'NA':>9} | {'NA':>9} | {'0':>12} |"
                f" {offset:>12} | {offset + n_frames - 1:>10}"
            )
            offset += n_frames
            out.append(df)
            continue

        d = df.copy()
        fr = pd.to_numeric(d[frame_col], errors="coerce").dropna().astype("int64")

        sleap_min = int(fr.min()) if not fr.empty else None
        sleap_max = int(fr.max()) if not fr.empty else None
        sleap_unique = int(fr.nunique())

        d.loc[fr.index, frame_col] = fr + offset

        global_start = offset
        global_end = offset + n_frames - 1

        print(
            f"{cid:>5} | {n_frames:>12} |"
            f" {str(sleap_min):>9} | {str(sleap_max):>9} | {sleap_unique:>12} |"
            f" {global_start:>12} | {global_end:>10}"
        )

        out.append(d)
        offset += n_frames

    print("=" * 100)
    print("video_frames = authoritative chunk length from ArUco HDF5")
    print("global_*     = assigned frame span (no overlap, no drift)")
    print("=" * 100 + "\n")

    return out


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def safe_label(s: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_") or "NA"


def write_outputs_parquet(
    df: pd.DataFrame,
    out_dir: Path,
    timestamp: str,
    cam_id: str,
    arenas: List[str],
    track_col: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for arena in arenas:
        g = df[df["ArenaLabel"] == arena]
        if g.empty:
            continue
        g.drop(columns=["ArenaLabel"], errors="ignore").to_parquet(
            out_dir / f"{timestamp}_{cam_id}_{arena}.parquet",
            index=False,
        )

    if track_col in df.columns:
        for (arena, track), g in df.groupby(["ArenaLabel", track_col], dropna=False):
            if arena not in arenas or g.empty:
                continue
            g.drop(columns=["ArenaLabel"], errors="ignore").to_parquet(
                out_dir / f"{timestamp}_{cam_id}_{arena}_track-{safe_label(track)}.parquet",
                index=False,
            )


# -----------------------------------------------------------------------------
# Per-run processing
# -----------------------------------------------------------------------------
def process_run(
    run_dir: Path,
    input_folder: Path,
    out_dir: Path,
    xcol: str,
    ycol: str,
    track_col: str,
    frame_col: str,
) -> None:
    data_dir = run_dir / "data"
    sleap_dir = data_dir

    # Segmentation is global to ROOT (input_folder), not inside each run.
    seg_dir = input_folder / "arena_seg"
    
    cam_map = list_csvs_grouped_by_cam(sleap_dir)
    if not cam_map:
        return

    timestamp = run_dir.name  # already validated

    for cam_id, files in cam_map.items():
        seg_path = find_seg_file_for_cam(seg_dir, cam_id)

        if seg_path is None:
            print(f"[SEG] run={timestamp} cam={cam_id}: NO seg found → using Arena00 fallback")
        else:
            print(f"[SEG] run={timestamp} cam={cam_id}: using seg {seg_path.resolve()}")


        seg, color_to_label, arenas = get_segmentation_or_default(seg_path)

        dfs: List[pd.DataFrame] = []
        chunk_ids: List[int] = []
        chunk_frame_counts: List[int] = []

        for csv_path in files:
            cam, cid = parse_cam_and_chunk(csv_path)
            h5_path = csv_path.with_name(
                csv_path.name.replace("_sleap_data.csv", "_aruco_tracks_.h5")
            )
            if not h5_path.exists():
                raise FileNotFoundError(f"Missing ArUco HDF5: {h5_path}")

            dfs.append(pd.read_csv(csv_path))
            chunk_ids.append(cid)
            chunk_frame_counts.append(aruco_n_frames(h5_path))

        dfs = apply_frame_offsets_using_aruco(
            dfs,
            chunk_ids,
            chunk_frame_counts,
            frame_col=frame_col,
            label=f"(run={timestamp} cam={cam_id})",
        )

        parts = []
        for df in dfs:
            if not df.empty:
                parts.append(assign_arenas_with_fallback(df, seg, color_to_label, xcol, ycol))

        if not parts:
            continue

        cam_df = pd.concat(parts, ignore_index=True)

        write_outputs_parquet(
            cam_df,
            out_dir,
            timestamp,
            cam_id,
            arenas,
            track_col,
        )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def run_on_timestamp_subfolders(
    input_folder: Path,
    out_dir: Path,
    xcol: str,
    ycol: str,
    track_col: str,
    frame_col: str,
) -> None:
    runs = [p for p in input_folder.iterdir() if p.is_dir() and TS_RE.match(p.name)]

    if not runs:
        raise FileNotFoundError(f"No timestamp-named run folders under {input_folder}")

    for run_dir in tqdm(sorted(runs), desc="Runs"):
        process_run(run_dir, input_folder, out_dir, xcol, ycol, track_col, frame_col)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_folder", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--xcol", type=str, default="X")
    ap.add_argument("--ycol", type=str, default="Y")
    ap.add_argument("--track_col", type=str, default="Track")
    ap.add_argument("--frame_col", type=str, default="Frame")
    args = ap.parse_args()

    run_on_timestamp_subfolders(
        input_folder=args.input_folder,
        out_dir=args.out_dir,
        xcol=args.xcol,
        ycol=args.ycol,
        track_col=args.track_col,
        frame_col=args.frame_col,
    )
