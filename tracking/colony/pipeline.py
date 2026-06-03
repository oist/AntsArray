#!/usr/bin/env python3
"""
Full colony map/combine/stitch pipeline.

This script is intentionally only orchestration:
- map_combine.py owns panorama mapping from per-camera detections.
- combine_batch.py owns pairing panorama PKLs and running local/SLURM combine jobs.
- stitch_tracks.py owns stitching chunk parquet files into per-track outputs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tracking.colony.combine_batch import discover_jobs, parse_sides, run_local, submit_slurm  # noqa: E402
from tracking.colony.panorama_io import validate_aruco_inputs_have_sleap_h5  # noqa: E402
from tracking.stitch_tracks import main as stitch_tracks  # noqa: E402
from tracking.stitch_tracks import write_pngs_from_existing  # noqa: E402


def run_mapping(
    *,
    hmats_path: Path,
    data_dir: Path,
    panorama_dir: Path,
    map_mode: str,
    min_instance_frame_frac: float,
) -> None:
    from tracking.colony.map_combine import (
        infer_experiment_name,
        load_homographies,
        process_aruco_chunks,
        process_sleap_chunks,
    )

    hmats = load_homographies(hmats_path)
    exp = infer_experiment_name(data_dir)
    panorama_dir.mkdir(parents=True, exist_ok=True)

    if map_mode in ("aruco", "both"):
        process_aruco_chunks(
            hmats,
            data_dir,
            panorama_dir,
            exp,
            min_instance_frame_frac=min_instance_frame_frac,
        )

    if map_mode in ("sleap", "both"):
        process_sleap_chunks(hmats, data_dir, panorama_dir, exp)


def run_combine(
    *,
    panorama_dir: Path,
    tracks_dir: Path,
    side: str,
    runner: str,
    logs_dir: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    job_name: str,
    conda_env: str,
    max_distance: float,
    lost_track_max_frames: int,
    lost_track_max_distance: float | None,
    lost_track_aruco_max_distance: float | None,
) -> None:
    tracks_dir.mkdir(parents=True, exist_ok=True)
    jobs = discover_jobs(panorama_dir, parse_sides(side))
    if not jobs:
        raise RuntimeError(f"No complete ArUco/SLEAP chunk jobs found in {panorama_dir}")

    logging.info("Prepared %d chunk-side combine jobs", len(jobs))
    if runner == "local":
        run_local(
            jobs,
            tracks_dir,
            max_distance=max_distance,
            lost_track_max_frames=lost_track_max_frames,
            lost_track_max_distance=lost_track_max_distance,
            lost_track_aruco_max_distance=lost_track_aruco_max_distance,
        )
        return

    submit_slurm(
        jobs,
        tracks_dir,
        logs_dir=logs_dir,
        partition=partition,
        cpus=cpus,
        mem=mem,
        time_limit=time_limit,
        job_name=job_name,
        conda_env=conda_env,
        max_distance=max_distance,
        lost_track_max_frames=lost_track_max_frames,
        lost_track_max_distance=lost_track_max_distance,
        lost_track_aruco_max_distance=lost_track_aruco_max_distance,
    )


def run_stitching(
    *,
    tracks_dir: Path,
    stitched_dir: Path,
    fps: float,
    columns: list[str],
    frame_col: str,
    track_col: str,
    write_track_pngs: bool,
    track_png_dir: Path | None,
    x_col: str,
    y_col: str,
    track_png_width: int,
    track_png_height: int,
    skip_existing: bool,
    pngs_from_existing: bool,
) -> None:
    stitched_dir.mkdir(parents=True, exist_ok=True)
    if pngs_from_existing:
        input_dir = stitched_dir / "per_track"
        write_pngs_from_existing(
            input_dir,
            stitched_dir,
            string=".parquet",
            frame_col=frame_col,
            x_col=x_col,
            y_col=y_col,
            png_dir=track_png_dir,
            png_width=track_png_width,
            png_height=track_png_height,
            skip_existing=skip_existing,
        )
        return

    stitch_tracks(
        tracks_dir,
        stitched_dir,
        columns,
        fps=fps,
        string=".parquet",
        frame_col=frame_col,
        track_col=track_col,
        write_track_pngs=write_track_pngs,
        png_dir=track_png_dir,
        x_col=x_col,
        y_col=y_col,
        png_width=track_png_width,
        png_height=track_png_height,
        skip_existing=skip_existing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full colony map/combine/stitch pipeline.")
    parser.add_argument("--hmats", type=Path, required=True, help=".npz file with homography stack key 'H'.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory with per-camera chunk files.")
    parser.add_argument("--work_dir", type=Path, required=True, help="Pipeline output directory.")
    parser.add_argument("--panorama_dir", type=Path, default=None, help="Default: <work_dir>/panorama_pkls")
    parser.add_argument("--tracks_dir", type=Path, default=None, help="Default: <work_dir>/tracks")
    parser.add_argument("--stitched_dir", type=Path, default=None, help="Default: <work_dir>/stitched")

    parser.add_argument("--skip_map", action="store_true")
    parser.add_argument("--skip_combine", action="store_true")
    parser.add_argument("--skip_stitch", action="store_true")
    parser.add_argument("--map_mode", choices=("aruco", "sleap", "both"), default="both")
    parser.add_argument("--min_instance_frame_frac", type=float, default=0.05)

    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--combine_runner", choices=("local", "slurm"), default="local")
    parser.add_argument("--logs_dir", type=Path, default=None, help="Default: <work_dir>/logs")
    parser.add_argument("--partition", default="compute")
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--time", default="0-24:00:00")
    parser.add_argument("--job_name", default="combine_tracks")
    parser.add_argument("--conda_env", default="aruco_env")
    parser.add_argument("--max_distance", type=float, default=90.0)
    parser.add_argument("--lost_track_max_frames", type=int, default=120)
    parser.add_argument("--lost_track_max_distance", type=float, default=None)
    parser.add_argument("--lost_track_aruco_max_distance", type=float, default=None)

    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--frame_col", default="Frame")
    parser.add_argument("--track_col", default="TrackID")
    parser.add_argument("--x_col", default="X")
    parser.add_argument("--y_col", default="Y")
    parser.add_argument("--columns", nargs="+", default=["Frame", "TrackID", "X", "Y", "Bodypoint"])
    parser.add_argument("--no_track_pngs", action="store_true")
    parser.add_argument("--track_png_dir", type=Path, default=None)
    parser.add_argument("--track_png_width", type=int, default=1200)
    parser.add_argument("--track_png_height", type=int, default=900)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--pngs_from_existing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    work_dir = args.work_dir
    panorama_dir = args.panorama_dir or work_dir / "panorama_pkls"
    tracks_dir = args.tracks_dir or work_dir / "tracks"
    stitched_dir = args.stitched_dir or work_dir / "stitched"
    logs_dir = args.logs_dir or work_dir / "logs"

    aruco_files = validate_aruco_inputs_have_sleap_h5(args.data_dir)
    logging.info("Preflight passed: %d ArUco H5 files have matching SLEAP H5 files.", len(aruco_files))

    if not args.skip_map:
        logging.info("Stage 1/3: mapping detections into panorama PKLs")
        run_mapping(
            hmats_path=args.hmats,
            data_dir=args.data_dir,
            panorama_dir=panorama_dir,
            map_mode=args.map_mode,
            min_instance_frame_frac=args.min_instance_frame_frac,
        )

    if not args.skip_combine:
        logging.info("Stage 2/3: combining ArUco and SLEAP panorama PKLs")
        run_combine(
            panorama_dir=panorama_dir,
            tracks_dir=tracks_dir,
            side=args.side,
            runner=args.combine_runner,
            logs_dir=logs_dir,
            partition=args.partition,
            cpus=args.cpus,
            mem=args.mem,
            time_limit=args.time,
            job_name=args.job_name,
            conda_env=args.conda_env,
            max_distance=args.max_distance,
            lost_track_max_frames=args.lost_track_max_frames,
            lost_track_max_distance=args.lost_track_max_distance,
            lost_track_aruco_max_distance=args.lost_track_aruco_max_distance,
        )
        if args.combine_runner == "slurm" and not args.skip_stitch:
            logging.info("SLURM combine jobs were submitted asynchronously; skipping stitching for this run.")
            return

    if not args.skip_stitch:
        logging.info("Stage 3/3: stitching chunk tracks")
        run_stitching(
            tracks_dir=tracks_dir,
            stitched_dir=stitched_dir,
            fps=args.fps,
            columns=args.columns,
            frame_col=args.frame_col,
            track_col=args.track_col,
            write_track_pngs=not args.no_track_pngs,
            track_png_dir=args.track_png_dir,
            x_col=args.x_col,
            y_col=args.y_col,
            track_png_width=args.track_png_width,
            track_png_height=args.track_png_height,
            skip_existing=args.skip_existing,
            pngs_from_existing=args.pngs_from_existing,
        )


if __name__ == "__main__":
    main()
