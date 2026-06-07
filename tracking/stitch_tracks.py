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
from tqdm import tqdm


# -----------------------
# parsing / sorting helpers
# -----------------------
CHUNK_KEY_RE = re.compile(r"^chunk(\d+)$")
TS_KEY_RE = re.compile(r"^\d{8}-\d{6}$")
TS_ANYWHERE_RE = re.compile(r"(\d{8}-\d{6})")  # first occurrence anywhere in stem
CHUNK_TOKEN_IN_SUFFIX_RE = re.compile(r"(?:^|_)(chunk\d+)(?:_|$)")
DEFAULT_COLUMNS = [
    "Frame",
    "TrackID",
    "X",
    "Y",
    "Bodypoint",
    "TrackX",
    "TrackY",
    "ArucoX",
    "ArucoY",
    "SleapAnchorX",
    "SleapAnchorY",
]


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
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(fp).schema_arrow.names)


def parquet_num_frames(fp: Path) -> Optional[int]:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(fp).schema_arrow.metadata or {}
    raw = metadata.get(b"num_frames")
    if raw is None:
        return None
    try:
        value = int(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def write_parquet_with_num_frames(
    df: pd.DataFrame,
    out_path: Path,
    *,
    num_frames: int,
    engine: str,
    compression: str,
) -> None:
    if engine != "pyarrow":
        df.to_parquet(out_path, index=False, engine=engine, compression=compression)
        return

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    md = dict(table.schema.metadata or {})
    md[b"num_frames"] = str(int(num_frames)).encode("utf-8")
    table = table.replace_schema_metadata(md)
    pq.write_table(table, str(out_path), compression=compression)


# -----------------------
# trajectory PNG rendering
# -----------------------
def draw_text_with_outline(
    image: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int],
    *,
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(
        image,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def write_track_png(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    frame_col: str,
    x_col: str,
    y_col: str,
    width: int = 1200,
    height: int = 900,
) -> None:
    import cv2

    png_x_col = x_col
    png_y_col = y_col
    if not {frame_col, png_x_col, png_y_col}.issubset(df.columns):
        if (
            x_col == "TrackX"
            and y_col == "TrackY"
            and {frame_col, "X", "Y"}.issubset(df.columns)
        ):
            png_x_col = "X"
            png_y_col = "Y"
        else:
            return

    required = {frame_col, png_x_col, png_y_col}
    if not required.issubset(df.columns):
        return

    d = df[[frame_col, png_x_col, png_y_col]].copy()
    d[frame_col] = pd.to_numeric(d[frame_col], errors="coerce")
    d[png_x_col] = pd.to_numeric(d[png_x_col], errors="coerce")
    d[png_y_col] = pd.to_numeric(d[png_y_col], errors="coerce")
    d = d.dropna(subset=[frame_col, png_x_col, png_y_col]).sort_values(frame_col)
    if d.empty:
        return

    # Multiple bodypoints can share the same frame. Collapse to one XY per frame
    # so the time-colored trace represents the track path, not skeleton density.
    d = d.groupby(frame_col, as_index=False)[[png_x_col, png_y_col]].mean()

    frames = d[frame_col].to_numpy(np.int64)
    xs = d[png_x_col].to_numpy(np.float64)
    ys = d[png_y_col].to_numpy(np.float64)

    image = np.full((height, width, 3), 20, dtype=np.uint8)
    left, top = 72, 56
    right, bottom = width - 38, height - 92
    plot_w = max(1, right - left)
    plot_h = max(1, bottom - top)

    x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
    if abs(x_max - x_min) < 1e-6:
        x_min -= 1.0
        x_max += 1.0
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0

    x_span = x_max - x_min
    y_span = y_max - y_min
    pad = 0.04 * max(x_span, y_span)
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad
    x_span = x_max - x_min
    y_span = y_max - y_min

    data_scale = min(plot_w / max(1e-6, x_span), plot_h / max(1e-6, y_span))
    draw_w = x_span * data_scale
    draw_h = y_span * data_scale
    x_origin = left + (plot_w - draw_w) / 2.0
    y_origin = top + (plot_h - draw_h) / 2.0

    cv2.rectangle(image, (left, top), (right, bottom), (80, 80, 80), 1)
    draw_text_with_outline(image, title[:110], (left, 32), (235, 235, 235), scale=0.7, thickness=1)
    draw_text_with_outline(image, "X", (left, height - 30), (230, 230, 230), scale=0.6, thickness=1)
    draw_text_with_outline(image, "Y", (18, top + 18), (230, 230, 230), scale=0.6, thickness=1)

    xp = np.clip(
        np.round(x_origin + (xs - x_min) * data_scale).astype(np.int32),
        left,
        right,
    )
    yp = np.clip(
        np.round(y_origin + (ys - y_min) * data_scale).astype(np.int32),
        top,
        bottom,
    )

    if len(frames) == 1:
        colors = np.array([[0, 255, 255]], dtype=np.uint8)
    else:
        ramp = np.linspace(0, 255, len(frames), dtype=np.uint8).reshape(-1, 1)
        colors = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO).reshape(-1, 3)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            xpi = np.clip(xp + dx, 0, width - 1)
            ypi = np.clip(yp + dy, 0, height - 1)
            image[ypi, xpi] = colors

    legend = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_TURBO)
    legend = cv2.resize(legend, (180, 14), interpolation=cv2.INTER_LINEAR)
    legend_x = right - legend.shape[1]
    legend_y = bottom + 24
    image[legend_y : legend_y + legend.shape[0], legend_x : legend_x + legend.shape[1]] = legend
    draw_text_with_outline(image, "early", (legend_x, legend_y - 6), (220, 220, 220), scale=0.45)
    draw_text_with_outline(image, "late", (legend_x + legend.shape[1] - 34, legend_y - 6), (220, 220, 220), scale=0.45)
    draw_text_with_outline(image, str(int(frames[0])), (legend_x, legend_y + 34), (220, 220, 220), scale=0.45)
    draw_text_with_outline(image, str(int(frames[-1])), (legend_x + legend.shape[1] - 52, legend_y + 34), (220, 220, 220), scale=0.45)
    draw_text_with_outline(
        image,
        f"points={len(frames)}  x=[{x_min:.1f},{x_max:.1f}]  y=[{y_min:.1f},{y_max:.1f}]",
        (left, bottom + 32),
        (220, 220, 220),
        scale=0.48,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


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
    write_track_pngs: bool = False,
    png_dir: Optional[Path] = None,
    x_col: str = "TrackX",
    y_col: str = "TrackY",
    png_width: int = 1200,
    png_height: int = 900,
    skip_existing: bool = False,
    track_ids_filter: Optional[set[int]] = None,
) -> None:
    if not files_sorted:
        return

    suffix_tag = safe_label(group_suffix)

    # ---- precompute start times and reference start time (per group)
    start_dt_by_fp: Dict[Path, Optional[datetime]] = {
        fp: parse_start_datetime_from_filename(fp) for fp in files_sorted
    }
    dts = [dt for dt in start_dt_by_fp.values() if dt is not None]
    ref_dt: Optional[datetime] = min(dts) if dts else None

    # ---- fallback running offset for files with no timestamps
    running_frame_offset = 0
    requested = list(dict.fromkeys(list(columns) + [track_col]))
    file_infos: List[dict] = []
    track_ids: set[int] = set()

    pbar = tqdm(files_sorted, desc=f"Index [{suffix_tag}]", unit="file", leave=False)

    for fp in pbar:
        cols_in_file = parquet_columns(fp)
        if frame_col not in cols_in_file:
            continue

        num_frames_meta = parquet_num_frames(fp)

        frame_df = pd.read_parquet(fp, columns=[frame_col], engine=engine)
        frame_df[frame_col] = pd.to_numeric(frame_df[frame_col], errors="coerce")
        frame_df = frame_df.dropna(subset=[frame_col])
        if frame_df.empty and num_frames_meta is None:
            continue

        if num_frames_meta is not None:
            local_len = int(num_frames_meta)
        else:
            local_max = int(frame_df[frame_col].max())
            local_len = local_max + 1

        # Determine time-faithful offset (if timestamp available); else fall back.
        dt = start_dt_by_fp.get(fp)
        if (ref_dt is not None) and (dt is not None):
            # Absolute frame index for the start of this file relative to the group's first timestamp.
            # Round for stability when fps is non-integer or time deltas are not exact multiples.
            frame_offset = int(round((dt - ref_dt).total_seconds() * float(fps)))
        else:
            frame_offset = running_frame_offset

        existing = [c for c in requested if c in cols_in_file]
        if track_col in cols_in_file:
            tid_df = pd.read_parquet(fp, columns=[track_col], engine=engine)
            tid_series = pd.to_numeric(tid_df[track_col], errors="coerce").dropna()
            track_ids.update(int(tid) for tid in tid_series.unique())
        else:
            track_ids.add(0)

        file_infos.append(
            {
                "path": fp,
                "existing": existing,
                "local_len": local_len,
                "frame_offset": frame_offset,
                "dt": dt,
            }
        )

        # Only advance running offset for timestamp-less mode.
        if (ref_dt is None) or (dt is None):
            running_frame_offset += local_len

    if track_ids_filter is not None:
        track_ids &= track_ids_filter

    # ---- write outputs (ALL into one directory), one TrackID at a time.
    for tid in tqdm(sorted(track_ids), desc=f"Tracks [{suffix_tag}]", unit="track", leave=False):
        out_path = per_track_dir / f"TrackID_{tid:04d}_all_{suffix_tag}.parquet"
        target_dir = png_dir if png_dir is not None else (per_track_dir.parent / "track_pngs")
        png_path = target_dir / f"TrackID_{tid:04d}_all_{suffix_tag}.png"

        if skip_existing and out_path.exists():
            if not write_track_pngs or png_path.exists():
                continue
            out = pd.read_parquet(out_path, engine=engine)
            write_track_png(
                out,
                png_path,
                title=f"TrackID {tid:04d} {suffix_tag}",
                frame_col=frame_col,
                x_col=x_col,
                y_col=y_col,
                width=png_width,
                height=png_height,
            )
            continue

        parts: List[pd.DataFrame] = []
        for info in file_infos:
            df = pd.read_parquet(info["path"], columns=info["existing"], engine=engine)
            if df.empty:
                continue

            df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
            df = df.dropna(subset=[frame_col])
            if df.empty:
                continue

            df = ensure_trackid(df, track_col)
            df[track_col] = df[track_col].astype(int)
            df = df[df[track_col] == int(tid)]
            if df.empty:
                continue

            df[frame_col] = df[frame_col].astype(np.int64) + int(info["frame_offset"])
            df["source_file"] = info["path"].name
            parts.append(df)

        if not parts:
            continue

        out = pd.concat(parts, ignore_index=True)
        group_num_frames = max(int(info["frame_offset"]) + int(info["local_len"]) for info in file_infos)
        write_parquet_with_num_frames(
            out,
            out_path,
            num_frames=group_num_frames,
            engine=engine,
            compression=compression,
        )
        if write_track_pngs:
            if skip_existing and png_path.exists():
                continue
            write_track_png(
                out,
                png_path,
                title=f"TrackID {tid:04d} {suffix_tag}",
                frame_col=frame_col,
                x_col=x_col,
                y_col=y_col,
                width=png_width,
                height=png_height,
            )


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



def main(
    input_dir: Path,
    out_dir: Path,
    columns: List[str],
    fps: float,
    string: str,
    frame_col: str = "Frame",
    track_col: str = "TrackID",
    write_track_pngs: bool = False,
    png_dir: Optional[Path] = None,
    x_col: str = "TrackX",
    y_col: str = "TrackY",
    png_width: int = 1200,
    png_height: int = 900,
    skip_existing: bool = False,
    track_ids_filter: Optional[set[int]] = None,
) -> None:
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
            frame_col=frame_col,
            track_col=track_col,
            write_track_pngs=write_track_pngs,
            png_dir=png_dir,
            x_col=x_col,
            y_col=y_col,
            png_width=png_width,
            png_height=png_height,
            skip_existing=skip_existing,
            track_ids_filter=track_ids_filter,
        )


def write_pngs_from_existing(
    input_dir: Path,
    out_dir: Path,
    *,
    string: str,
    frame_col: str,
    x_col: str,
    y_col: str,
    png_dir: Optional[Path],
    png_width: int,
    png_height: int,
    skip_existing: bool,
    engine: str = "pyarrow",
) -> None:
    files = sorted(Path(input_dir).glob(f"*{string}"))
    if not files:
        raise RuntimeError(f"No parquet files found in {input_dir}")

    target_dir = png_dir if png_dir is not None else (Path(out_dir) / "track_pngs")
    target_dir.mkdir(parents=True, exist_ok=True)

    for fp in tqdm(files, desc="Track PNGs", unit="file"):
        png_path = target_dir / f"{fp.stem}.png"
        if skip_existing and png_path.exists():
            continue
        df = pd.read_parquet(fp, engine=engine)
        write_track_png(
            df,
            png_path,
            title=fp.stem,
            frame_col=frame_col,
            x_col=x_col,
            y_col=y_col,
            width=png_width,
            height=png_height,
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--string", type=str, default='.parquet')
    ap.add_argument("--frame_col", type=str, default="Frame")
    ap.add_argument("--track_col", type=str, default="TrackID")
    ap.add_argument(
        "--no_track_pngs",
        action="store_true",
        help="Do not write time-colored trajectory PNGs during stitching.",
    )
    ap.add_argument(
        "--pngs_from_existing",
        action="store_true",
        help="Do not stitch. Read existing per-track parquet files from --input_dir and write PNGs only.",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Do not overwrite existing parquet or PNG outputs.",
    )
    ap.add_argument(
        "--track_png_dir",
        type=Path,
        default=None,
        help="Directory for time-colored track PNGs. Default: <out_dir>/track_pngs",
    )
    ap.add_argument("--x_col", type=str, default="TrackX")
    ap.add_argument("--y_col", type=str, default="TrackY")
    ap.add_argument("--track_png_width", type=int, default=1200)
    ap.add_argument("--track_png_height", type=int, default=900)
    ap.add_argument(
        "--track_id",
        action="append",
        type=int,
        default=None,
        help="Only stitch this TrackID. May be passed more than once.",
    )
    ap.add_argument(
        "--columns",
        nargs="+",
        default=DEFAULT_COLUMNS,
    )
    args = ap.parse_args()

    if args.pngs_from_existing:
        write_pngs_from_existing(
            args.input_dir,
            args.out_dir,
            string=args.string,
            frame_col=args.frame_col,
            x_col=args.x_col,
            y_col=args.y_col,
            png_dir=args.track_png_dir,
            png_width=args.track_png_width,
            png_height=args.track_png_height,
            skip_existing=args.skip_existing,
        )
        raise SystemExit(0)

    main(
        args.input_dir,
        args.out_dir,
        args.columns,
        fps=args.fps,
        string=args.string,
        frame_col=args.frame_col,
        track_col=args.track_col,
        write_track_pngs=not args.no_track_pngs,
        png_dir=args.track_png_dir,
        x_col=args.x_col,
        y_col=args.y_col,
        png_width=args.track_png_width,
        png_height=args.track_png_height,
        skip_existing=args.skip_existing,
        track_ids_filter=None if args.track_id is None else set(args.track_id),
    )
