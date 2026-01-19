#!/usr/bin/env python3
"""
Stitch per-TrackID parquet files across chunks.

CHANGE:
- If TrackID is missing or invalid, it is set to 0.
- FIX: robust to missing TrackID column in parquet (pyarrow-safe).
- CHANGE: ALL outputs are written into ONE directory: <out_dir>/per_track
          (no per-suffix subfolders).
- NEW: Frame stitching is time-faithful across files using FPS and file timestamps.
       If filenames include YYYYMMDD-HHMMSS (e.g. 20251218-163512_cam02_Arena00.parquet),
       global frames incorporate real gaps between files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


# -----------------------
# parsing / sorting helpers
# -----------------------
CHUNK_KEY_RE = re.compile(r"^chunk(\d+)$")
TS_KEY_RE = re.compile(r"^\d{8}-\d{6}$")
TS_ANYWHERE_RE = re.compile(r"(\d{8}-\d{6})")  # first occurrence anywhere in stem
CHUNK_TOKEN_IN_SUFFIX_RE = re.compile(r"(?:^|_)(chunk\d+)(?:_|$)")


def safe_label(s: str) -> str:
    s = s.strip()
    if not s:
        return "NO_SUFFIX"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "NO_SUFFIX"


@dataclass(frozen=True)
class ParsedName:
    iter_key: str
    group_suffix: str
    raw_name: str


def parse_parquet_name(fp: Path) -> ParsedName:
    stem = fp.stem
    if "_" in stem:
        iter_key, rest = stem.split("_", 1)
        group_suffix = rest
    else:
        iter_key = stem
        group_suffix = ""

    # NEW: collapse chunked suffixes into one logical group
    group_suffix = normalize_group_suffix(group_suffix)

    return ParsedName(iter_key=iter_key, group_suffix=group_suffix, raw_name=fp.name)



def iter_key_sort_value(iter_key: str) -> Tuple[int, object]:
    m = CHUNK_KEY_RE.match(iter_key)
    if m:
        return (0, int(m.group(1)))
    if TS_KEY_RE.match(iter_key):
        return (1, datetime.strptime(iter_key, "%Y%m%d-%H%M%S"))
    return (2, iter_key)


def parse_start_datetime_from_filename(fp: Path) -> Optional[datetime]:
    """
    Extract the first YYYYMMDD-HHMMSS occurrence from the filename stem and parse it.
    Returns None if no timestamp is present.
    """
    m = TS_ANYWHERE_RE.search(fp.stem)
    if not m:
        return None
    ts = m.group(1)
    try:
        return datetime.strptime(ts, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


# -----------------------
# TrackID handling
# -----------------------
def ensure_trackid(df: pd.DataFrame, track_col: str) -> pd.DataFrame:
    if track_col not in df.columns:
        df[track_col] = 0
        return df

    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    df[track_col] = (
        pd.to_numeric(df[track_col], errors="coerce")
        .fillna(0)
        .astype(np.int64)
    )
    return df



def parquet_columns(fp: Path) -> set[str]:
    return set(pq.ParquetFile(fp).schema_arrow.names)


# -----------------------
# main stitching logic
# -----------------------
def stitch_group(
    files_sorted: List[Path],
    group_suffix: str,
    per_track_dir: Path,
    columns: List[str],
    fps: float = 24.0,
    frame_col: str = "Frame",
    track_col: str = "TrackID",
    engine: str = "pyarrow",
    compression: str = "zstd",
) -> None:
    if not files_sorted:
        return

    suffix_tag = safe_label(group_suffix)

    # ---- initialize TrackIDs from FIRST file
    first_cols = parquet_columns(files_sorted[0])
    first_read = [c for c in (columns + [track_col]) if c in first_cols]
    first = pd.read_parquet(files_sorted[0], columns=first_read or None, engine=engine)
    first = ensure_trackid(first, track_col)
    track_ids = sorted(first[track_col].unique().tolist())

    parts_by_track: Dict[int, List[pd.DataFrame]] = {tid: [] for tid in track_ids}

    # ---- precompute start times and reference start time (per group)
    start_dt_by_fp: Dict[Path, Optional[datetime]] = {
        fp: parse_start_datetime_from_filename(fp) for fp in files_sorted
    }
    dts = [dt for dt in start_dt_by_fp.values() if dt is not None]
    ref_dt: Optional[datetime] = min(dts) if dts else None

    # ---- fallback running offset for files with no timestamps
    running_frame_offset = 0

    pbar = tqdm(files_sorted, desc=f"Chunks [{suffix_tag}]", unit="file", leave=False)

    for fp in pbar:
        cols_in_file = parquet_columns(fp)
        if frame_col not in cols_in_file:
            continue

        # Read frame column to determine local span and allow fallback running offset updates.
        frame_df = pd.read_parquet(fp, columns=[frame_col], engine=engine)
        frame_df[frame_col] = pd.to_numeric(frame_df[frame_col], errors="coerce")
        frame_df = frame_df.dropna(subset=[frame_col])
        if frame_df.empty:
            continue

        local_min = int(frame_df[frame_col].min())
        local_max = int(frame_df[frame_col].max())
        local_len = (local_max - local_min) + 1

        # Determine time-faithful offset (if timestamp available); else fall back.
        dt = start_dt_by_fp.get(fp)
        if (ref_dt is not None) and (dt is not None):
            # Absolute frame index for the start of this file relative to the group's first timestamp.
            # Round for stability when fps is non-integer or time deltas are not exact multiples.
            frame_offset = int(round((dt - ref_dt).total_seconds() * float(fps)))
        else:
            frame_offset = running_frame_offset

        requested = list(dict.fromkeys(list(columns) + [track_col]))
        existing = [c for c in requested if c in cols_in_file]
        df = pd.read_parquet(fp, columns=existing, engine=engine)
        if df.empty:
            if (ref_dt is None) or (dt is None):
                running_frame_offset += local_len
            continue

        df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
        df = df.dropna(subset=[frame_col])
        if df.empty:
            if (ref_dt is None) or (dt is None):
                running_frame_offset += local_len
            continue
        
        df = ensure_trackid(df, track_col)

        # Re-base local frames to start at 0, then add the chosen offset.
        df[frame_col] = (df[frame_col].astype(np.int64) - local_min) + frame_offset

        df[track_col] = df[track_col].astype(int)
        df["source_file"] = fp.name

        for tid, g in df.groupby(track_col, sort=False):
            if tid in parts_by_track:
                parts_by_track[tid].append(g)

        # Only advance running offset for timestamp-less mode.
        if (ref_dt is None) or (dt is None):
            running_frame_offset += local_len

    # ---- write outputs (ALL into one directory)
    for tid, parts in parts_by_track.items():
        if not parts:
            continue
        out = pd.concat(parts, ignore_index=True)
        out_path = per_track_dir / f"TrackID_{tid:04d}_all_{suffix_tag}.parquet"
        out.to_parquet(out_path, index=False, engine=engine, compression=compression)


def normalize_group_suffix(group_suffix: str) -> str:
    """
    If group_suffix contains a chunk token like chunk000, remove it so that all
    chunks for the same logical recording group together.

    Example:
      "121514_chunk000_left" -> "121514_left"
      "foo_chunk12_bar"      -> "foo_bar"
    """
    if "chunk" not in group_suffix:
        return group_suffix

    # Replace the chunk token with a single underscore separator, then clean up.
    s = CHUNK_TOKEN_IN_SUFFIX_RE.sub("_", group_suffix)
    s = re.sub(r"__+", "_", s).strip("_")
    return s



def main(input_dir: Path, out_dir: Path, columns: List[str], fps: float, string: str) -> None:
    files = sorted(Path(input_dir).glob(f"*{string}"))
    if not files:
        raise RuntimeError("No parquet files found")

    # SINGLE output directory
    per_track_dir = Path(out_dir) / "per_track"
    per_track_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, List[Path]] = {}
    parsed: Dict[Path, ParsedName] = {}
    
    for fp in files:
        pn = parse_parquet_name(fp)
        parsed[fp] = pn
        groups.setdefault(pn.group_suffix, []).append(fp)

    for group_suffix, fps_list in groups.items():
        fps_sorted = sorted(fps_list, key=lambda p: iter_key_sort_value(parsed[p].iter_key))
        stitch_group(
            fps_sorted,
            group_suffix,
            per_track_dir,
            columns,
            fps=fps,
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--string", type=str, default='.parquet')
    ap.add_argument(
        "--columns",
        nargs="+",
        default=["Frame", "TrackID", "X", "Y", "Bodypoint"],
    )
    args = ap.parse_args()

    main(args.input_dir, args.out_dir, args.columns, fps=args.fps, string=args.string)
