#!/usr/bin/env python3
"""
Combine SLEAP CSVs into per-arena track files across ALL chunks (no exp arg).

Behavior
--------
- Discovers SLEAP CSVs in --sleap_dir using the chunk token in the filename.
- For each CSV, loads the matching per-camera segmentation PNG (by name).
- Builds arena labels from *all non-zero colors* in that seg image.
- Assigns each SLEAP row (X,Y) -> ArenaLabel by sampling seg pixel at (X,Y).
- Accumulates across ALL chunks and writes:
    out_dir/
      per_arena/
        Arena00.csv, Arena01.csv, ...
      per_arena_tracks/
        Arena00_track-<Track>.csv, ...

Notes
-----
- No transforms/homographies.
- “Non-zero color” means any channel is non-zero (BGR != (0,0,0)).
- If Track column is missing, it will still write per-arena files (no per-track split).
- Seg images must be in the same pixel coordinate system as X,Y.

Install
-------
pip install numpy pandas opencv-python tqdm
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Naming / discovery (deduped)
# -----------------------------
import re
from pathlib import Path
from typing import Dict, List, Tuple


CAM_RE = re.compile(r"(?:^|_)cam(\d+)(?:_|$)")
CHUNK_RE = re.compile(r"_(\d{3})(?:_sleap_data)?\.csv$")


def parse_cam_and_chunk(csv_path: Path) -> Tuple[str, int]:
    """
    Returns:
      cam_id: e.g. 'cam1' or 'cam02' (preserves zero-padding)
      chunk_id: int from the trailing _### token
    """
    name = csv_path.name

    m_cam = CAM_RE.search(name)
    if not m_cam:
        raise ValueError(f"Could not extract cam number from filename: {name}")
    cam_id = f"cam{m_cam.group(1)}"

    m_chunk = CHUNK_RE.search(name)
    if not m_chunk:
        raise ValueError(f"Could not extract 3-digit chunk from filename: {name}")
    chunk_id = int(m_chunk.group(1))

    return cam_id, chunk_id


def list_csvs_grouped_by_cam(sleap_dir: Path) -> Dict[str, List[Path]]:
    """
    Returns {cam_id: [csv_paths_sorted_by_chunk]}.
    Only includes files that contain both camN and a trailing _###(optional _sleap_data).csv chunk token.
    """
    cam_map: Dict[str, List[Path]] = {}
    for p in sleap_dir.glob("*.csv"):
        try:
            cam_id, _chunk = parse_cam_and_chunk(p)
        except ValueError:
            continue
        cam_map.setdefault(cam_id, []).append(p)

    for cam_id, files in cam_map.items():
        cam_map[cam_id] = sorted(files, key=lambda fp: parse_cam_and_chunk(fp)[1])

    return dict(sorted(cam_map.items()))


def seg_path_for_sleap_csv(csv_path: Path, seg_dir: Path) -> Path:
    """
    Seg image is chosen by camera only:
      cam1_*_001.csv  -> seg_dir/cam1_seg.png
      cam02_*_123.csv -> seg_dir/cam02_seg.png
    """
    cam_id, _chunk = parse_cam_and_chunk(csv_path)
    return seg_dir / f"{cam_id}_seg.png"


# -----------------------------
# Seg / color labeling
# -----------------------------
def read_seg_bgr(seg_path: Path) -> np.ndarray:
    seg = cv2.imread(str(seg_path), cv2.IMREAD_COLOR)  # BGR
    if seg is None:
        raise FileNotFoundError(f"Could not read seg image: {seg_path}")
    return seg


def unique_nonzero_colors(seg_bgr: np.ndarray) -> List[Tuple[int, int, int]]:
    flat = seg_bgr.reshape(-1, 3)
    colors = np.unique(flat, axis=0)
    out: List[Tuple[int, int, int]] = []
    for c in colors:
        b, g, r = int(c[0]), int(c[1]), int(c[2])
        if (b, g, r) != (0, 0, 0):
            out.append((b, g, r))
    out.sort()  # deterministic
    return out


def build_color_to_label(colors_bgr: List[Tuple[int, int, int]], prefix: str = "Arena") -> Dict[Tuple[int, int, int], str]:
    return {c: f"{prefix}{i:02d}" for i, c in enumerate(colors_bgr)}


def sample_label_at_xy(
    seg_bgr: np.ndarray,
    x: float,
    y: float,
    color_to_label: Dict[Tuple[int, int, int], str],
    unknown_label: str = "UNKNOWN",
    background_label: str = "BACKGROUND",
) -> str:
    h, w = seg_bgr.shape[:2]
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return unknown_label
    b, g, r = map(int, seg_bgr[yi, xi])
    if (b, g, r) == (0, 0, 0):
        return background_label
    return color_to_label.get((b, g, r), unknown_label)


def assign_arenas(
    df: pd.DataFrame,
    seg_bgr: np.ndarray,
    color_to_label: Dict[Tuple[int, int, int], str],
    xcol: str = "X",
    ycol: str = "Y",
    out_col: str = "ArenaLabel",
) -> pd.DataFrame:
    if xcol not in df.columns or ycol not in df.columns:
        raise ValueError(f"Missing required columns {xcol},{ycol}. Found: {list(df.columns)}")
    df = df.dropna(subset=[xcol, ycol]).copy() #drop rows with NaN X or Y
    xy = df[[xcol, ycol]].to_numpy(float)
    labels = [sample_label_at_xy(seg_bgr, float(x), float(y), color_to_label) for x, y in xy]
    out = df.copy()
    out[out_col] = labels
    return out


# -----------------------------
# Output helpers
# -----------------------------
def safe_label(s: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_") or "NA"


def cam_id_from_csv(csv_path: Path) -> str:
    m = _CAM_PATT.search(csv_path.name)
    if not m:
        raise ValueError(f"Could not extract cam number from filename: {csv_path.name}")
    return f"cam{m.group(1)}"  # preserves zero-padding


def write_cam_arena_outputs(
    df_assigned: pd.DataFrame,
    out_dir: Path,
    cam_id: str,
    color_to_label: Dict[Tuple[int, int, int], str],
    arena_col: str = "ArenaLabel",
    track_col: str = "Track",
    drop_cols: Optional[List[str]] = None,
) -> None:
    """
    Writes per-(cam,arena) and per-(cam,arena,track) CSVs, dropping annotation columns.
    File names encode cam + arena id.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cam_dir = out_dir / cam_id
    per_cam_dir.mkdir(exist_ok=True)

    if drop_cols is None:
        drop_cols = [arena_col, "SegImage", "SourceCSV"]  # safe even if absent

    # Determine stable arena id mapping (Arena00, Arena01, ...) from this camera's palette
    # (Already deterministic in build_color_to_label, but we keep this explicit)
    known_arenas = sorted(set(color_to_label.values()))

    # Write per arena (all tracks)
    for arena_label in known_arenas:
        g = df_assigned[df_assigned[arena_col] == arena_label]
        if g.empty:
            continue
        out = g.drop(columns=[c for c in drop_cols if c in g.columns], errors="ignore")
        out.to_csv(per_cam_dir / f"{cam_id}_{arena_label}.csv", index=False)

    # Write per arena per track (if Track exists)
    if track_col in df_assigned.columns:
        per_track_dir = per_cam_dir / "tracks"
        per_track_dir.mkdir(exist_ok=True)

        for (arena_label, track), g in df_assigned.groupby([arena_col, track_col], sort=True, dropna=False):
            if arena_label not in known_arenas or g.empty:
                continue
            out = g.drop(columns=[c for c in drop_cols if c in g.columns], errors="ignore")
            out.to_csv(
                per_track_dir / f"{cam_id}_{arena_label}_track-{safe_label(track)}.csv",
                index=False,
            )


def apply_frame_offsets_in_place(
    dfs_in_chunk_order: List[pd.DataFrame],
    frame_col: str = "Frame",
) -> List[pd.DataFrame]:
    """
    For a single camera, given chunk-ordered dfs:
      - chunk0 frames remain as-is
      - chunk1 frames += (max_frame(chunk0) + 1)
      - chunk2 frames += (max_frame(chunk0..1) + 1)
    Works even if frames are missing/NaN or df is empty.
    """
    offset = 0
    out: List[pd.DataFrame] = []

    for df in dfs_in_chunk_order:
        if df.empty:
            out.append(df)
            continue

        if frame_col not in df.columns:
            raise ValueError(f"Expected frame column '{frame_col}' in SLEAP CSV. Columns: {list(df.columns)}")

        d = df.copy()

        # Coerce frame to numeric; drop NaNs later only if you want.
        fr = pd.to_numeric(d[frame_col], errors="coerce")

        # Apply offset to finite frames only
        mask = fr.notna()
        d.loc[mask, frame_col] = fr.loc[mask].astype(np.int64) + offset

        # Update offset using max finite frame AFTER shift
        if mask.any():
            offset = int(d.loc[mask, frame_col].max()) + 1

        out.append(d)

    return out


# -----------------------------
# Main combine
# -----------------------------
def combine_all_chunks_into_per_arena_tracks(
    sleap_dir: Path,
    seg_dir: Path,
    out_dir: Path,
    xcol: str = "X",
    ycol: str = "Y",
    track_col: str = "Track",
    frame_col: str = "Frame",
) -> None:
    cam_map = list_csvs_grouped_by_cam(sleap_dir)
    if not cam_map:
        raise FileNotFoundError(f"No SLEAP CSVs with cam + _### chunk token found in {sleap_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for cam_id, files in tqdm(cam_map.items(), desc="Cameras"):
        # Load seg once per camera
        seg_path = seg_dir / f"{cam_id}_seg.png"
        seg_bgr = read_seg_bgr(seg_path)
        colors = unique_nonzero_colors(seg_bgr)
        color_to_label = build_color_to_label(colors, prefix="Arena")

        # Read all chunks for this camera in order
        dfs = []
        for p in files:
            df = pd.read_csv(p)
            if df.empty:
                dfs.append(df)
                continue
            dfs.append(df)

        # Adjust frames across chunks
        dfs = apply_frame_offsets_in_place(dfs, frame_col=frame_col)

        # Assign arenas and concatenate for this camera
        assigned = []
        for df in tqdm(dfs, desc=f"{cam_id} chunks", leave=False):
            if df.empty:
                continue
            df_assigned = assign_arenas(df, seg_bgr, color_to_label, xcol=xcol, ycol=ycol, out_col="ArenaLabel")
            if not df_assigned.empty:
                assigned.append(df_assigned)

        if not assigned:
            continue

        cam_df = pd.concat(assigned, ignore_index=True)

        # Write once per cam: per arena and per arena/track
        write_cam_arena_outputs(
            df_assigned=cam_df,
            out_dir=out_dir,
            cam_id=cam_id,
            color_to_label=color_to_label,
            arena_col="ArenaLabel",
            track_col=track_col,
            drop_cols=["ArenaLabel"],  # keep output compact
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sleap_dir", type=Path, required=True)
    ap.add_argument("--seg_dir", type=Path, required=True, help="Directory containing per-camera seg PNGs")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--xcol", type=str, default="X")
    ap.add_argument("--ycol", type=str, default="Y")
    ap.add_argument("--track_col", type=str, default="Track")
    args = ap.parse_args()

    combine_all_chunks_into_per_arena_tracks(
        sleap_dir=args.sleap_dir,
        seg_dir=args.seg_dir,
        out_dir=args.out_dir,
        xcol=args.xcol,
        ycol=args.ycol,
        track_col=args.track_col,
    )
