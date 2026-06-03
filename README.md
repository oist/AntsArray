# AntsArray

Tracking pipeline for combining ArUco and SLEAP detections into ant tracks.

## Setup

```bash
uv sync
```

Run commands from the repository root unless noted otherwise.

## Which Pipeline?

- **Colony pipeline**: use for chunked multi-camera files named like `camXX_..._NNN...`.
- **Single-ant pipeline**: use for timestamp run folders like `YYYYMMDD-HHMMSS/` plus arena masks.

ArUco detection is upstream of both pipelines. The examples below assume ArUco H5 outputs already exist, except for the optional detection example.

## Repository Layout

```text
run_aruco.py                  # one-video ArUco detection entry point
tracking/
  colony/                     # colony map, combine, and full-pipeline scripts
  single_ant/                 # single-ant arena pipeline and per-run builder
  stitch_tracks.py            # shared stitcher and trajectory PNG writer
  core/                       # shared tracking utilities
  gui/                        # interactive curation tools
  viewers/                    # lightweight playback/viewer scripts
camera_cal/
  similarity_panorama.py      # initial panorama/calibration image generation
  refinement/                 # optional homography refinement scripts
```

## Full Colony Pipeline

Use this for colony-scale, chunked multi-camera data. It maps cameras into panorama coordinates, combines ArUco + SLEAP detections into chunk tracks, then stitches chunks into per-track files.

```bash
python tracking/colony/pipeline.py \
  --hmats /path/to/initial_H_mats.npz \
  --data_dir /path/to/chunk_files \
  --work_dir /path/to/colony_work \
  --map_mode both \
  --side both \
  --combine_runner local \
  --fps 24
```

Default outputs under `--work_dir`:

- `panorama_pkls/`: mapped left/right ArUco and SLEAP panorama PKLs.
- `tracks/`: per-chunk, per-side parquet files.
- `stitched/per_track/`: stitched per-track parquet files.
- `stitched/track_pngs/`: time-colored trajectory PNGs, written by default.

Cluster combine jobs:

```bash
python tracking/colony/pipeline.py \
  --hmats /path/to/initial_H_mats.npz \
  --data_dir /path/to/chunk_files \
  --work_dir /path/to/colony_work \
  --map_mode both \
  --side both \
  --combine_runner slurm \
  --fps 24
```

SLURM combine jobs are submitted asynchronously. After they finish, run stitching only:

```bash
python tracking/colony/pipeline.py \
  --hmats /path/to/initial_H_mats.npz \
  --data_dir /path/to/chunk_files \
  --work_dir /path/to/colony_work \
  --skip_map \
  --skip_combine \
  --fps 24
```

Colony notes:

- ArUco inputs are H5/HDF5 files containing `aruco_tracks`.
- Fastest mapping path uses `*_aruco_detections.h5` for coordinates and sibling `*_aruco_tracks.h5` only for accurate `num_frames`.
- Fastest SLEAP mapping path uses `*_sleap_data.h5`.
- CSV detection files are fallback inputs and do not need to be kept for the pipeline when the matching H5 exists.
- If cameras disagree on chunk frame count, the mapper warns and uses the largest count.
- `tracking/colony/combine_batch.py` is the batch driver for local or SLURM combine jobs.

## Full Single-Ant Pipeline

Expected layout:

```text
ROOT/
  arena_seg/
    cam01_*.png
    cam02_*.png
  20251211-104517/
    data/
      cam1_..._000_sleap_data.csv     # optional
      cam1_..._000_aruco_tracks_.h5
      ...
  20251211-121530/
    data/
      ...
```

Run SLEAP-preferred with ArUco fallback:

```bash
python tracking/single_ant/pipeline.py \
  --input_folder /path/to/ROOT \
  --out_dir /path/to/output \
  --detection_source auto
```

Run ArUco-only single-ant tracking:

```bash
python tracking/single_ant/pipeline.py \
  --input_folder /path/to/ROOT \
  --out_dir /path/to/output \
  --detection_source aruco
```

Single-ant outputs:

- Stage-1 run/camera/arena parquets are written directly to `--out_dir`.
- Stitched tracks are written to `--out_dir/per_track/`.
- Time-colored trajectory PNGs are written to `--out_dir/track_pngs/` by default.

Single-ant notes:

- Preferred mask folder is `ROOT/arena_seg`; fallback is `ROOT/seg_arena`.
- Mask filenames can be camera-specific (`camXX*.png`) or generic (`*arena_seg*.png`).
- In `--detection_source aruco`, marker ID is ignored and output uses one continuous track (`Track=0`).
- Frame offsets across chunks are based on ArUco H5 frame counts, not sparse detection frames.

## Optional ArUco Detection

For one video:

```bash
python run_aruco.py \
  --video-file /path/to/video.avi \
  --output-path /path/to/output \
  --output-format h5
```

## Manual Colony Stages

Map all cameras into panorama coordinates:

```bash
python tracking/colony/map_combine.py \
  --hmats /path/to/initial_H_mats.npz \
  --data_dir /path/to/chunk_files \
  --outdir /path/to/panorama_pkls \
  --mode both
```

Combine one chunk/side:

```bash
python tracking/colony/combine_one_chunk.py \
  --input_file /path/to/20251118_121514_chunk000_aruco_panorama_x_left1740.pkl \
  --output_path /path/to/tracks
```

Combine all chunks locally:

```bash
python tracking/colony/combine_batch.py \
  --input_folder /path/to/panorama_pkls \
  --output_path /path/to/tracks \
  --side both \
  --runner local
```

Combine all chunks with SLURM:

```bash
python tracking/colony/combine_batch.py \
  --input_folder /path/to/panorama_pkls \
  --output_path /path/to/tracks \
  --side both \
  --runner slurm
```

Stitch chunk parquets:

```bash
python tracking/stitch_tracks.py \
  --input_dir /path/to/tracks \
  --out_dir /path/to/stitched \
  --fps 24 \
  --track_col TrackID
```

This writes one PNG per stitched track to `<out_dir>/track_pngs` by default. Each plot uses the same early-to-late TURBO color ramp style as the ArUco curation GUI trajectory panel.

If stitched per-track parquets already exist and you only want PNGs, do not restitch:

```bash
python tracking/stitch_tracks.py \
  --input_dir /path/to/stitched/per_track \
  --out_dir /path/to/stitched \
  --pngs_from_existing \
  --skip_existing
```

## Main Entry Points

| Script | Purpose |
|---|---|
| `tracking/colony/pipeline.py` | Full colony map/combine/stitch pipeline. |
| `tracking/single_ant/pipeline.py` | Full single-ant arena-based pipeline. |
| `run_aruco.py` | ArUco detection for one video. |
| `tracking/colony/map_combine.py` | Manual colony panorama mapping stage. |
| `tracking/colony/combine_batch.py` | Manual colony batch combine stage. |
| `tracking/colony/combine_one_chunk.py` | Manual one chunk/side combine stage. |
| `tracking/stitch_tracks.py` | Shared stitcher and trajectory PNG writer. |
| `tracking/gui/aruco_curation.py` | Interactive ArUco detection curation GUI. |

## Active Notes

- Sleep classification should currently rely more on speed than angle in problematic recordings.
- Long-recording stitching should continue to account for real timestamp gaps between files.
- Validate variable chunk-length behavior when syncing new datasets.
