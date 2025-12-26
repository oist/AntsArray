#!/usr/bin/env python3
"""
Stitch per-TrackID parquet files across chunks.

CHANGE:
- If TrackID is missing or invalid, it is set to 0.
- FIX: robust to missing TrackID column in parquet (pyarrow-safe).
- CHANGE: ALL outputs are written into ONE directory: <out_dir>/per_track
          (no per-suffix subfolders).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


# -----------------------
# parsing / sorting helpers
# -----------------------
CHUNK_KEY_RE = re.compile(r"^chunk(\d+)$")
TS_KEY_RE = re.compile(r"^\d{8}-\d{6}$")


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
    return ParsedName(iter_key=iter_key, group_suffix=group_suffix, raw_name=fp.name)


def iter_key_sort_value(iter_key: str) -> Tuple[int, object]:
    m = CHUNK_KEY_RE.match(iter_key)
    if m:
        return (0, int(m.group(1)))
    if TS_KEY_RE.match(iter_key):
        return (1, datetime.strptime(iter_key, "%Y%m%d-%H%M%S"))
    return (2, iter_key)


# -----------------------
# TrackID handling
# -----------------------
def ensure_trackid(df: pd.DataFrame, track_col: str) -> pd.DataFrame:
    if track_col not in df.columns:
        df[track_col] = 0
    else:
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
    frame_offset = 0

    pbar = tqdm(files_sorted, desc=f"Chunks [{suffix_tag}]", unit="file", leave=False)

    for fp in pbar:
        cols_in_file = parquet_columns(fp)
        if frame_col not in cols_in_file:
            continue

        frame_df = pd.read_parquet(fp, columns=[frame_col], engine=engine)
        frame_df[frame_col] = pd.to_numeric(frame_df[frame_col], errors="coerce")
        if frame_df[frame_col].dropna().empty:
            continue

        chunk_len = int(frame_df[frame_col].max()) + 1

        requested = list(dict.fromkeys(list(columns) + [track_col]))
        existing = [c for c in requested if c in cols_in_file]
        df = pd.read_parquet(fp, columns=existing, engine=engine)
        if df.empty:
            frame_offset += chunk_len
            continue

        df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
        df = df.dropna(subset=[frame_col])
        if df.empty:
            frame_offset += chunk_len
            continue

        df = ensure_trackid(df, track_col)
        df[frame_col] = df[frame_col].astype(np.int64) + frame_offset
        df[track_col] = df[track_col].astype(int)
        df["source_file"] = fp.name

        for tid, g in df.groupby(track_col, sort=False):
            if tid in parts_by_track:
                parts_by_track[tid].append(g)

        frame_offset += chunk_len

    # ---- write outputs (ALL into one directory)
    for tid, parts in parts_by_track.items():
        if not parts:
            continue
        out = pd.concat(parts, ignore_index=True)
        out_path = per_track_dir / f"TrackID_{tid:04d}_all_{suffix_tag}.parquet"
        out.to_parquet(out_path, index=False, engine=engine, compression=compression)


def main(input_dir: Path, out_dir: Path, columns: List[str]) -> None:
    files = sorted(Path(input_dir).glob("*.parquet"))
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

    for group_suffix, fps in groups.items():
        fps_sorted = sorted(fps, key=lambda p: iter_key_sort_value(parsed[p].iter_key))
        stitch_group(
            fps_sorted,
            group_suffix,
            per_track_dir,
            columns,
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--columns",
        nargs="+",
        default=["Frame", "TrackID", "X", "Y", "Bodypoint"],
    )
    args = ap.parse_args()

    main(args.input_dir, args.out_dir, args.columns)
