# AntsArray - Claude Code Context

## Project Overview
Multi-camera ant tracking pipeline: 25 synchronized 4K cameras -> individual ant posture tracking on a single global coordinate system.

**Core technologies:** SLEAP (pose estimation), OpenCV ArUco DICT_4X4_1000 (ant ID), homography-based camera alignment.

## Pipeline Stages
1. **Split** — ffmpeg segments raw video into 1-hour chunks
2. **Encode** — re-encode chunks for consistent codec
3. **ArUco detection** (`run_aruco.py`) — OpenCV marker detection per chunk (Deigo cluster, CPU)
4. **SLEAP inference** — pose estimation per chunk (Saion cluster, GPU)
5. **Format conversion** (`sleap2csv.py`, `sleap2h5.py`) — SLEAP .slp -> CSV/H5
6. **Camera alignment** (`tracking/map_combine.py`) — homography projection, left/right split at X=1740px
7. **Track merging** (`tracking/tracking_utils.py`) — nearest-neighbor matching of ArUco IDs + SLEAP poses (threshold: 50px)
8. **Cross-chunk stitching** (`tracking/single_ant_over_chunks.py`) — timestamp-based frame continuity
9. **Analysis** (`analysis/`) — speed, sleep classification, colony behavior

## Key Directories
- `aruco_detection/` — ArUco marker detection and NN-based inference
- `tracking/` — core tracking logic, homography alignment, per-ant stitching
- `analysis/` — behavioral analysis modules (speed, sleep, colony)
- `camera_cal/` — calibration pattern generation and homography computation
- `JOBS_DIR_P/` — SLURM job logs

## Infrastructure
- **Deigo cluster** — CPU jobs (encoding, ArUco detection, tracking)
- **Saion cluster** — GPU jobs (SLEAP inference)
- **Orchestrator:** `transcode_sleap_aruco.sh` coordinates both clusters via sentinel files (.ok)
- **Dependency management:** `uv` (see `pyproject.toml`)

## Data Formats
- ArUco: H5 `(num_frames, 1000, 2)` + CSV (Frame, Instance, X, Y, Confidence)
- SLEAP: CSV (Frame, Instance, Bodypoint, X, Y, Score_node)
- Tracks: Parquet files with TrackID column
- Camera alignment: NPZ with H (n_cam, 3, 3) homography stack

## Conventions
- Shell scripts use SLURM array jobs for parallelism
- Sentinel files (`encode.ok`, `aruco.ok`, `sleap.ok`) signal stage completion
- Video chunks named `{BASE}_raw_###.avi` (raw) / `{BASE}_###.avi` (encoded)
- Python >= 3.9

## Known Issues
- Sleep classification angle measure unreliable; speed-only method preferred
- Gap handling between recording chunks needs timestamp-based frame padding
- `num_frames` field not fully propagated in single-ant pipeline
