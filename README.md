# AntsArray

Tracking pipeline for mapping multi-camera ArUco/SLEAP detections into panorama coordinates, combining them into ant tracks, and stitching those tracks across chunks and blocks.

Run commands from the repository root unless noted otherwise.

## Environment

Local development:

```bash
uv sync
```

On deigo, the colony pipeline is expected to run with:

```text
/bucket/ReiterU/sam/miniforge3/envs/aruco_env/bin/python
```

Compute nodes may not be able to write to `/bucket`, so the cluster wrapper writes intermediate and final outputs to `/flash/ReiterU/ant_tmp/$USER/...` and then transfers completed outputs back to `/bucket` from a login node.

## Main Colony Pipeline

Use `tracking/colony/submit_blocks_pipeline.sh` for production colony runs. It handles blocks, chunks, SLURM dependencies, stitching, PNG generation, and final transfer.

Edit the configuration block at the top of:

```text
tracking/colony/submit_blocks_pipeline.sh
```

The most important paths are:

```bash
BLOCKS_ROOT="/bucket/ReiterU/Ants/basler/20260515"
HMATS="/bucket/ReiterU/Ants/basler/cameraArray_calib/.../initial_H_mats.npz"
OUTPUT_ROOT="/flash/ReiterU/ant_tmp/${USER}/colony_pipeline/20260515"
SUBMIT_ROOT="${OUTPUT_ROOT}/jobs"
PYTHON_BIN="/bucket/ReiterU/sam/miniforge3/envs/aruco_env/bin/python"
```

Then submit:

```bash
cd ~/AntsArray/tracking/colony
bash submit_blocks_pipeline.sh
```

The default is `SKIP_EXISTING=1`, so existing panorama PKLs, chunk tracks, stitched parquets, and PNGs are not overwritten. To regenerate a stage, delete the affected outputs or edit `SKIP_EXISTING=0` in the script before submitting.

## Expected Input Layout

Each block should look like:

```text
BLOCKS_ROOT/
  block01/
    data/
      cam01_cam0_YYYY-MM-DD-HH-MM-SS_000_aruco_tracks.h5
      cam01_cam0_YYYY-MM-DD-HH-MM-SS_000_sleap_data.h5
      cam02_cam1_YYYY-MM-DD-HH-MM-SS_000_aruco_tracks.h5
      cam02_cam1_YYYY-MM-DD-HH-MM-SS_000_sleap_data.h5
      ...
  block02/
    data/
      ...
```

Preflight requires every ArUco H5 to have a matching SLEAP H5 before the block runs.

## Pipeline Stages

### 0. Preflight

The wrapper scans every `block*/data` folder, validates ArUco/SLEAP H5 pairs, and discovers chunk numbers. Missing SLEAP files fail early instead of creating partial track outputs.

### 1. Panorama PKL Creation

For each block and chunk, the wrapper submits one SLURM map job. Each map job runs:

```text
tracking/colony/map_combine.py
```

This stage:

- reads per-camera ArUco H5 and SLEAP H5 files;
- applies `initial_H_mats.npz` homographies;
- writes mapped left/right panorama PKLs;
- stores ArUco `num_frames` metadata from dense `aruco_tracks`;
- keeps SLEAP skeletons together by assigning side from anchor bodypoint `0`.

Outputs:

```text
OUTPUT_ROOT/block03/panorama_pkls/
  20260518_085922_chunk012_aruco_panorama_x_left1740.pkl
  20260518_085922_chunk012_aruco_panorama_x_right1740.pkl
  20260518_085922_chunk012_sleap_panorama_x_left1740.pkl
  20260518_085922_chunk012_sleap_panorama_x_right1740.pkl
```

### 2. Per-Chunk Tracking

After all map jobs for a block succeed, a small fan-out job submits one tracking SLURM job per chunk/side. Each worker runs:

```text
tracking/colony/combine_one_chunk.py
```

This stage:

- loads one matching ArUco panorama PKL and SLEAP panorama PKL;
- combines detections with `tracking/core/tracking_utils.py`;
- uses ArUco IDs as `TrackID`;
- requires ArUco detections to be near a SLEAP anchor before they can affect tracking;
- prevents recent same-ID respawns from jumping far from the last known position;
- streams parquet output to avoid OOM.

Outputs:

```text
OUTPUT_ROOT/block03/tracks/
  20260518_085922_chunk012_left.parquet
  20260518_085922_chunk012_right.parquet
```

### 3. Stitch Chunks Within Each Block

After all chunk tracking jobs for a block succeed, the wrapper submits a block stitch fan-out. It first detects `TrackID`s in the block's chunk parquets, then submits one stitch job per `TrackID`.

Each stitch worker runs:

```text
tracking/stitch_tracks.py --track_id <ID>
```

This stage:

- stitches all chunks in a block for one `TrackID`;
- uses parquet `num_frames` metadata so chunk offsets preserve missing frames;
- writes one per-track parquet;
- writes a trajectory PNG by default.

Outputs:

```text
OUTPUT_ROOT/block03/stitched/
  per_track/
    TrackID_0017_all_085922_right.parquet
  track_pngs/
    TrackID_0017_all_085922_right.png
```

### 4. Stitch Continuous Blocks

After all per-block stitch jobs succeed, the wrapper creates a continuous stitching input folder. Existing per-block per-track files are symlinked. Missing track/block combinations get empty placeholder parquet files with `num_frames` metadata so time offsets still include blocks where that track was absent.

Then the wrapper submits one continuous stitch job per `TrackID`.

Outputs:

```text
OUTPUT_ROOT/continuous_stitched/
  per_track/
  track_pngs/
```

### 5. Transfer Back To Bucket

When continuous stitching finishes successfully, a login-side transfer watcher copies outputs from flash back to bucket with `rsync --ignore-existing`.

Default transfer destinations:

```text
/bucket/.../block03/panorama_pkls
/bucket/.../block03/tracks
/bucket/.../block03/stitched
/bucket/.../continuous_stitched
```

Flash files are kept by default. Set `DELETE_FLASH_AFTER_TRANSFER=1` or pass `--delete_flash_after_transfer` to remove transferred flash contents after successful `rsync`.

## Manual Colony Commands

Use manual commands for debugging one block, one chunk, or one stage.

Map one chunk:

```bash
python tracking/colony/map_combine.py \
  --hmats /path/to/initial_H_mats.npz \
  --data_dir /path/to/block03/data \
  --outdir /path/to/block03/panorama_pkls \
  --mode both \
  --chunk 012
```

Track one chunk/side:

```bash
python tracking/colony/combine_one_chunk.py \
  --input_file /path/to/panorama_pkls/20260518_085922_chunk012_aruco_panorama_x_right1740.pkl \
  --output_path /path/to/tracks \
  --max_distance 90.0 \
  --lost_track_max_frames 120
```

Track all chunks locally:

```bash
python tracking/colony/combine_batch.py \
  --input_folder /path/to/panorama_pkls \
  --output_path /path/to/tracks \
  --side both \
  --runner local \
  --skip_existing
```

Track all chunks with SLURM:

```bash
python tracking/colony/combine_batch.py \
  --input_folder /path/to/panorama_pkls \
  --output_path /path/to/tracks \
  --side both \
  --runner slurm \
  --python_bin /bucket/ReiterU/sam/miniforge3/envs/aruco_env/bin/python \
  --logs_dir /flash/ReiterU/ant_tmp/$USER/colony_pipeline/jobs/block03/logs \
  --job_ids_file /flash/ReiterU/ant_tmp/$USER/colony_pipeline/jobs/block03/tracking_job_ids.txt \
  --skip_existing
```

Stitch chunk tracks:

```bash
python tracking/stitch_tracks.py \
  --input_dir /path/to/tracks \
  --out_dir /path/to/stitched \
  --fps 24 \
  --track_col TrackID
```

Write PNGs from already stitched tracks:

```bash
python tracking/stitch_tracks.py \
  --input_dir /path/to/stitched/per_track \
  --out_dir /path/to/stitched \
  --pngs_from_existing \
  --skip_existing
```

## Single-Block Local Orchestrator

`tracking/colony/pipeline.py` is a local/single-block orchestrator for map, combine, and stitch. It is useful for development or non-cluster runs.

```bash
python tracking/colony/pipeline.py \
  --hmats /path/to/initial_H_mats.npz \
  --data_dir /path/to/block03/data \
  --work_dir /path/to/block03_work \
  --map_mode both \
  --side both \
  --combine_runner local \
  --fps 24 \
  --skip_existing
```

If `--combine_runner slurm` is used, combine jobs are submitted asynchronously and stitching is skipped in that run. Run a stitch-only command after those jobs finish.

## Tracking Notes

- `TrackID` comes from ArUco `Instance`.
- Dense ArUco H5 slot index is treated as `Instance`.
- ArUco detections are filtered to those near a SLEAP anchor before tracking decisions use them.
- Existing tracks are updated by same-ID ArUco/SLEAP matches first, then by isolated SLEAP continuity, then by filtered ArUco-only anchor keep-alive.
- Recent same-ID respawns must be spatially consistent with the previous position.
- After `lost_track_max_frames`, an ID can be treated as a fresh acquisition again.
- Duplicate same-ID ArUco detections in one frame are handled by choosing the candidate closest to the previous track position for keep-alive.

## Main Entry Points

| Script | Purpose |
|---|---|
| `tracking/colony/submit_blocks_pipeline.sh` | Production block/chunk SLURM pipeline. |
| `tracking/colony/pipeline.py` | Single-block local or asynchronous combine orchestrator. |
| `tracking/colony/map_combine.py` | Panorama mapping for ArUco and SLEAP files. |
| `tracking/colony/combine_one_chunk.py` | One chunk/side tracking worker. |
| `tracking/colony/combine_batch.py` | Batch tracker launcher for local or SLURM workers. |
| `tracking/stitch_tracks.py` | Chunk/block/continuous stitcher and trajectory PNG writer. |
| `run_aruco.py` | ArUco detection for one video. |

## Optional ArUco Detection

For one video:

```bash
python run_aruco.py \
  --video-file /path/to/video.avi \
  --output-path /path/to/output \
  --output-format h5
```
