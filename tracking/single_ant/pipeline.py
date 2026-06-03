#!/usr/bin/env python3
"""
Run the single-ant ArUco-aligned merge, then stitch outputs across runs.

Stage 1: tracking/single_ant/build_tracks.py logic (per-run, per-camera, per-arena parquet).
Stage 2: tracking/stitch_tracks.py logic (per-track parquet and PNGs across runs).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow namespace-package style imports from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tracking.single_ant.build_tracks import run_on_timestamp_subfolders  # type: ignore
from tracking.stitch_tracks import main as stitch_tracks  # type: ignore


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Run single_ant merge, then stitch outputs across runs."
    )
    ap.add_argument("--input_folder", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--xcol", type=str, default="X")
    ap.add_argument("--ycol", type=str, default="Y")
    ap.add_argument("--track_col", type=str, default="Track")
    ap.add_argument("--frame_col", type=str, default="Frame")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument(
        "--detection_source",
        choices=["auto", "sleap", "aruco"],
        default="auto",
        help="Detection input mode for stage 1: auto (prefer SLEAP, fallback ArUco), sleap only, or aruco only.",
    )
    ap.add_argument(
        "--write_track_files",
        action="store_true",
        help="Also write per-track files in stage 1 (*_track-<id>.parquet). Off by default for single-ant datasets.",
    )
    ap.add_argument(
        "--stitch_columns",
        nargs="+",
        default=None,
        help="Columns to carry into stitched outputs.",
    )
    ap.add_argument(
        "--no_track_pngs",
        action="store_true",
        help="Do not write time-colored trajectory PNGs during stitching.",
    )
    ap.add_argument(
        "--track_png_dir",
        type=Path,
        default=None,
        help="Directory for track PNGs. Default: <out_dir>/track_pngs",
    )
    ap.add_argument("--track_png_width", type=int, default=1200)
    ap.add_argument("--track_png_height", type=int, default=900)
    ap.add_argument("--skip_existing", action="store_true", help="Do not overwrite existing stitched parquet or PNG files.")
    args = ap.parse_args()

    run_on_timestamp_subfolders(
        input_folder=args.input_folder,
        out_dir=args.out_dir,
        xcol=args.xcol,
        ycol=args.ycol,
        track_col=args.track_col,
        frame_col=args.frame_col,
        detection_source=args.detection_source,
        write_track_files=args.write_track_files,
    )

    stitch_columns = args.stitch_columns
    if stitch_columns is None:
        stitch_columns = [args.frame_col, args.track_col, "X", "Y", "Bodypoint"]

    stitch_tracks(
        args.out_dir,
        args.out_dir,
        stitch_columns,
        fps=args.fps,
        string=".parquet",
        frame_col=args.frame_col,
        track_col=args.track_col,
        write_track_pngs=not args.no_track_pngs,
        png_dir=args.track_png_dir,
        x_col="X",
        y_col="Y",
        png_width=args.track_png_width,
        png_height=args.track_png_height,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
