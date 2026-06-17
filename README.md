# AntsArray

Tracking and analysis pipeline for mapping multi-camera ArUco/SLEAP detections into panorama coordinates, combining detections into ant tracks, stitching those tracks across chunks, and running block-level colony analyses.

Run commands from the repository root unless noted otherwise.

## Environment

Local development:

```bash
uv sync
```

On deigo, the colony pipeline is expected to run with the `aruco_env` conda environment:

```text
/bucket/ReiterU/sam/miniforge3/envs/aruco_env/bin/python
```

Compute nodes may not be able to write to `/bucket`, so cluster jobs write outputs to flash first, usually under:

```text
/flash/ReiterU/ant_tmp/$USER/colony_pipeline/<date>/
```

Login-side transfer watchers copy completed outputs back to the matching bucket dataset folders.

## Production Block Pipeline

Use `tracking/colony/submit_blocks_pipeline.sh` for production colony tracking and chunk-level interactions.

Typical block run:

```bash
cd ~/AntsArray/tracking/colony
bash submit_blocks_pipeline.sh --block_glob block02
```

The default layout is flat inside each block:

```text
OUTPUT_ROOT/
  block02/
    panorama_pkls/
    tracks/
    stitched/
    interactions/
  jobs/
    block02/
      scripts/
      logs/
        submit/
        tracking_workers/
        stitching/
        interaction_workers/
      state/
```

The data folders under `block02/` are copied back to the corresponding bucket block:

```text
/bucket/.../<date>/block02/panorama_pkls
/bucket/.../<date>/block02/tracks
/bucket/.../<date>/block02/stitched
/bucket/.../<date>/block02/interactions
```

`jobs/block02/` contains generated sbatch scripts, worker logs, job ID files, completion markers, and transfer manifests. It is not copied as analysis data.

The default is `SKIP_EXISTING=1`. Existing outputs are not overwritten. Use `--force_recompute` to recompute existing flash outputs.

## Expected Input Layout

Each block should contain a `data/` directory with paired ArUco and SLEAP H5 files:

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

Preflight requires complete ArUco/SLEAP pairs before a block can run.

## Fanout Approach

The pipeline uses small submitter jobs to create many independent Slurm workers, then joins them with dependency markers:

1. The block map job creates all panorama PKLs for the block.
2. A tracking submitter job runs after mapping and submits one worker per chunk/side.
3. A stitch job runs after all tracking workers finish and writes block-level per-track outputs.
4. An interaction submitter job runs after all tracking workers finish and submits one worker per chunk/side interaction parquet.
5. A login-side transfer watcher waits for fresh stitch and interaction completion markers, then rsyncs the block output folders back to bucket.

The same pattern is used for generic per-track analysis with `scripts/per_track_slurm_fanout.sh`: one worker per `TrackID_*.parquet`, one completion marker after all workers, then a login-side transfer watcher.

## Pipeline Stages

### 1. Panorama PKLs

`tracking/colony/pipeline.py` calls `tracking/colony/map_combine.py` to map per-camera ArUco and SLEAP detections through `initial_H_mats.npz`.

Outputs:

```text
block02/panorama_pkls/
  20260515_142047_chunk000_aruco_panorama_x_left1740.pkl
  20260515_142047_chunk000_aruco_panorama_x_right1740.pkl
  20260515_142047_chunk000_sleap_panorama_x_left1740.pkl
  20260515_142047_chunk000_sleap_panorama_x_right1740.pkl
```

### 2. Per-Chunk Tracking

`tracking/colony/combine_batch.py` discovers complete ArUco/SLEAP panorama pairs and submits `tracking/colony/combine_one_chunk.py` workers. Each worker combines one chunk/side into a parquet file.

Outputs:

```text
block02/tracks/
  20260515_142047_chunk000_left.parquet
  20260515_142047_chunk000_right.parquet
```

Tracking parquet columns include SLEAP bodypoint `X/Y`, tracking anchor `TrackX/TrackY`, matched tag `ArucoX/ArucoY`, and SLEAP anchor `SleapAnchorX/SleapAnchorY`.

If all expected chunk outputs already exist and `--skip_existing` is active, no tracking workers are submitted; downstream stitch and interaction stages still run from the existing track files.

### 3. Block Stitching

The dependent stitch job runs `tracking/colony/pipeline.py --skip_map --skip_combine`, which calls `tracking/stitch_tracks.py` over the block's chunk parquet files.

Outputs:

```text
block02/stitched/
  per_track/
    TrackID_0017_all_142047_left.parquet
  track_pngs/
    TrackID_0017_all_142047_left.png
```

Stitched parquets preserve time gaps using frame counts and chunk timing metadata.

### 4. Chunk Interaction Analysis

`tracking/colony/interaction_batch.py` submits one worker per chunk/side track parquet. Each worker runs `tracking/colony/interaction_one_chunk.py`.

For each frame, the worker first finds ant pairs within an interaction radius using `TrackX/TrackY`. It then records a directed interaction when an antenna bodypoint from one ant is within the micro-interaction distance of any bodypoint on the other ant.

Outputs are flat and intentionally minimal:

```text
block02/interactions/
  20260515_142047_chunk000_left.parquet
  20260515_142047_chunk000_right.parquet
```

Each interaction parquet contains only:

```text
Frame, antenna_track_id, body_track_id
```

The completion marker lives in `jobs/block02/state/interactions_complete_block02.ok`, not in the interaction output folder.

### 5. Transfer Back To Bucket

The transfer watcher is started from the login shell. It waits for fresh completion markers and then runs `rsync` from flash to bucket for:

```text
panorama_pkls/
tracks/
stitched/
interactions/
```

Flash files are kept by default. Pass `--delete_flash_after_transfer` to delete transferred flash output contents after successful rsync.

## Generic Per-Track Fanout

Use `scripts/per_track_slurm_fanout.sh` for expensive analyses that can run independently for every stitched ant track.

The wrapper expects an input folder like:

```text
block02/stitched/per_track/
  TrackID_0000_all_142047_left.parquet
  TrackID_0001_all_142047_left.parquet
  ...
```

It creates one Slurm worker per matching track file. Each worker receives:

```text
TRACK_PATH
TRACK_NAME
TRACK_STEM
TRACK_ID
TRACK_INDEX
TASK_OUTPUT_DIR
RUN_OUTPUT_DIR
PER_TRACK_DIR
JOBS_DIR
```

For an operation named `speed_vectors`, task outputs are written under:

```text
block02/stitched/speed_vectors/
  per_track/
    TrackID_0000_all_142047_left/
      speed_mm_s.npy
      speed_metadata.json
      _SUCCESS
  jobs/
    workers/
    logs/
```

The wrapper then copies the operation output folder back to:

```text
/bucket/.../<date>/<block>/stitched/<operation_output_name>/
```

Saved convenience commands are in:

```bash
bash analysis/commands.sh /flash/ReiterU/ant_tmp/$USER/colony_pipeline/20260515/block02/stitched/per_track
```

That script submits the current standard per-track operations.

## Analysis Steps

### Speed Vectors

Operation script:

```text
analysis/compute_track_speed_vector.py
```

This loads only track `TrackX/TrackY`, builds a dense frame vector, interpolates short gaps, smooths valid segments, converts to mm/s, and saves only the compact speed vector plus metadata.

Per-track output:

```text
speed_mm_s.npy
speed_metadata.json
```

The metadata includes `frame_min`, `frame_max`, `n_frames`, and `n_observed_frames`; interactive scripts use `n_observed_frames / n_frames` to filter good tracks.

### Sleep Crop Label GUI

The sleep label GUI works on small per-ant crop videos, not the original full camera videos. The intended workflow is:

1. Export or receive crop videos around single tracked ants.
2. Label sleep/wake by eye in `analysis/sleep_label_gui.py`.
3. Use the saved label vectors as supervised training data for an automatic classifier.

This is better than manually tuning antenna-angle thresholds. If antenna angles or other pose features are useful, the model can learn that from the labels.

#### Fresh Setup For A Labeler

These commands are written for someone who only needs to label attached crop videos. They do not need to install the full tracking pipeline.

Open Terminal on macOS/Linux, or PowerShell on Windows. Copy and paste one code block at a time.

Install `git` if it is not already installed:

```bash
# macOS
xcode-select --install

# Ubuntu/Linux
sudo apt update
sudo apt install -y git
```

```powershell
# Windows PowerShell
winget install --id Git.Git -e
```

Install `uv`, which will handle the Python environment:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen Terminal/PowerShell after installing `git` or `uv`.

Download the repository into any folder you want to work from:

```bash
# macOS/Linux
cd "<FOLDER_WHERE_YOU_WANT_THE_CODE>"
git clone https://github.com/oist/AntsArray.git
cd AntsArray
git pull
```

```powershell
# Windows PowerShell
cd "<FOLDER_WHERE_YOU_WANT_THE_CODE>"
git clone https://github.com/oist/AntsArray.git
cd AntsArray
git pull
```

Put the attached `.mp4` crop videos in one folder. Set `CROP_VIDEO_FOLDER` to that folder path:

```bash
# macOS/Linux
CROP_VIDEO_FOLDER="<PATH_TO_FOLDER_WITH_CROP_VIDEOS>"
```

```powershell
# Windows PowerShell
$CROP_VIDEO_FOLDER = "<PATH_TO_FOLDER_WITH_CROP_VIDEOS>"
```

Check that the GUI dependencies can be loaded:

```bash
uv run --no-project --with numpy --with pandas --with pyarrow --with opencv-python --with pillow python -c "import cv2, numpy, pandas, PIL, pyarrow; print('Setup OK')"
```

Run the GUI on a folder of crop videos:

```bash
# macOS/Linux, from inside the AntsArray folder
uv run --no-project --with numpy --with pandas --with pyarrow --with opencv-python --with pillow python analysis/sleep_label_gui.py --video_dir "$CROP_VIDEO_FOLDER"
```

```powershell
# Windows PowerShell, from inside the AntsArray folder
uv run --no-project --with numpy --with pandas --with pyarrow --with opencv-python --with pillow python analysis\sleep_label_gui.py --video_dir "$CROP_VIDEO_FOLDER"
```

To update the GUI later:

```bash
# macOS/Linux
cd "<PATH_TO_ANTSARRAY>"
git pull
```

```powershell
# Windows PowerShell
cd "<PATH_TO_ANTSARRAY>"
git pull
```

To run it again after setup, use the same GUI command. `uv` caches the packages, so later runs should start faster.

When finished, send back the whole `label_vectors/` folder that appears next to the crop videos. Do not rename the videos after labeling, because label filenames are based on video filenames.

#### Export Crop Videos Yourself

If you have the full block data and need to make the crop videos first, export one small crop video per visible ant over a 10-minute window:

```bash
python analysis/export_sleep_crop_videos.py \
  --video "<PATH_TO_CAMERA_VIDEO>" \
  --start_time 0 \
  --crop_size_px 360 \
  --video_backend h264
```

The crop exporter infers `tracks/` from the video block by default, maps track positions to camera frames, and writes a folder under:

```text
block02/stitched/sleep_crop_videos/
```

#### Run The GUI

Open a folder of crop videos in the Tk GUI:

```bash
uv run --no-project --with numpy --with pandas --with pyarrow --with opencv-python --with pillow python analysis/sleep_label_gui.py \
  --video_dir "<CROP_VIDEO_FOLDER>"
```

To label one crop video directly:

```bash
uv run --no-project --with numpy --with pandas --with pyarrow --with opencv-python --with pillow python analysis/sleep_label_gui.py \
  --video "<CROP_VIDEO.mp4>"
```

Optional arguments:

- `--labels_dir /path/to/label_vectors`: write labels somewhere other than the crop video's `label_vectors/` folder.
- `--start_frame 1234`: open the first video at a specific frame.

The GUI saves per-frame label vectors beside the crop videos by default:

```text
sleep_crop_videos/<window>/label_vectors/
  <crop_video>_labels.npy
  <crop_video>_labels.parquet
  <crop_video>_metadata.json
```

Label values are `-1` for unlabeled, `0` for wake, and `1` for sleep. The bar under the seek slider shows unlabeled frames in gray, wake in blue, and sleep in orange. Clicking the bar seeks to that part of the video.

Basic labeling workflow:

- Press `s` or the `Sleep` button to start labeling sleep from the current frame.
- Press `w`, `n`, or the `Wake` button to start labeling wake from the current frame.
- While a label is active, normal playback paints each frame the playhead reaches. Seeking to a different frame resets the active label anchor, so skipped frames are not filled.
- Press the same label again, `e`, or `End` to stop the active label at the current frame.
- Press `[` to set a range start, then press `s` or `w` at another frame to label that whole interval once.
- Press `c` or `Clear` to clear the current frame, or to clear the selected interval after setting a range start.
- Use `Save` or `ctrl+s` to save manually. Labels are also saved when changing videos or closing the GUI.

Useful hotkeys:

- space: play/pause;
- left/right: previous/next frame;
- `a`/`d`: jump backward/forward 1 second;
- `A`/`D`: play backward/forward at 2x, then faster up to 30x on repeated presses;
- `s`: start/switch/end `sleep`;
- `w` or `n`: start/switch/end `wake`;
- `c`: clear the current frame or selected range;
- `e`: end the active label at the current frame;
- `[`: set range start for a one-shot interval label;
- `,`/`.`: previous/next crop video;
- `ctrl+s`: save.

### Supervised Sleep Classifier

Train the random-forest classifier:

```bash
/home/sam-reiter/miniforge3/envs/ants/bin/python analysis/sleep_classifier.py train \
  --labels /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/sleep_labels/*_sleep_labels.parquet \
  --per_track_dir /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/per_track \
  --speed_root /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/speed_vectors \
  --out /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/sleep_classifier
```

The model is intentionally simple and interpretable: a balanced random forest trained from speed summaries, posture angles, bodypoint distances, and pose bounding-box cues. Outputs include `sleep_random_forest.joblib`, `feature_importance.csv`, `training_report.txt`, and the labeled `training_features.parquet`.

Apply the classifier to tracks:

```bash
/home/sam-reiter/miniforge3/envs/ants/bin/python analysis/sleep_classifier.py predict \
  --model /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/sleep_classifier/sleep_random_forest.joblib \
  --per_track_dir /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/per_track \
  --speed_root /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/speed_vectors \
  --out /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/sleep_predictions
```

### Colony Presence Vectors

Operation script:

```text
analysis/compute_track_colony_presence_vector.py
```

This computes a compact per-frame vector using the default colony boxes in millimeters:

```text
(-86, -32, -63, -8)
(93, 149, -63, -8)
```

Per-track output:

```text
colony_presence_i1.npy
colony_presence_metadata.json
```

Vector values are `-1` for missing position, `0` for outside colony, and `1` for inside colony.

### Grid Occupancy Histograms

Operation script:

```text
analysis/compute_track_grid_occupancy.py
```

This bins each track into a 2D grid for the left or right colony and normalizes each histogram by the number of detected frames for that ant. Left and right colony histograms are kept separate because the sides do not line up exactly.

Per-track output:

```text
grid_occupancy_f4.npy
grid_x_edges_mm.npy
grid_y_edges_mm.npy
grid_occupancy_metadata.json
```

## Interactive Analysis Scripts

The interactive scripts are plain Python files with `# %%` cells. Open them in VS Code and run cells in the Jupyter interactive window. They try to enable `%matplotlib qt` for interactive figures.

### `analysis/colony_speed.py`

Loads bucket `stitched/speed_vectors`, filters tracks with `n_observed_frames / n_frames > 0.40`, attaches `colony_presence_vectors`, and plots:

- left/right colony-average speed time series;
- light on/off shading from absolute time parsed from track metadata;
- individual ant speed traces;
- speed plus colony in/out for a selected ant;
- all-ant speed images;
- all-ant speed images ordered by fraction of valid frames spent in the colony;
- quiet-period images using the speed threshold.

Helper functions live in `analysis/colony_speed_utils.py`.

### `analysis/grid_occupancy.py`

Loads bucket `stitched/grid_occupancy_histograms`, attaches speed metadata for detection-rate filtering, and plots:

- one ant's 2D occupancy histogram;
- UMAP + Leiden clustering separately for left and right colonies;
- `track_cluster_ids.csv` mapping `TrackID` to cluster ID;
- UMAP cluster plots;
- cluster mean occupancy histograms;
- example ant histograms from each cluster;
- cluster-average speed time series with light on/off shading;
- quiet-period images sorted by occupancy cluster.

Helper functions live in `analysis/grid_occupancy_utils.py`.

### `analysis/cluster_time_of_day_occupancy.py`

Loads `grid_occupancy.py` cluster assignments and recomputes time-of-day occupancy locally from the per-track stitched parquets. Times are relative to light on at 5:30 AM by default.

It can cache binned cluster occupancy into:

```text
stitched/grid_occupancy_histograms/time_of_day_cluster_occupancy/
```

and plots all clusters as tiled occupancy maps with optional additional time binning and smoothing.

### `analysis/ant_interaction_lightweight_test.py`

Local debugging script for tuning interaction radii and antenna bodypoint behavior on one chunk. It can generate labeled SLEAP skeleton debug images with interaction radii marked. Production interaction extraction is in `tracking/colony/interaction_batch.py` and `tracking/colony/interaction_one_chunk.py`.

### `analysis/interaction_analysis.py`

Loads bucket `interactions/` chunk parquet files with the matching `tracks/` chunk parquet and `stitched/grid_occupancy_histograms/track_cluster_ids.csv`. Use `CHUNKS = "all"` to run every chunk for the selected side, `CHUNKS = "000"` for a single chunk, or `MAX_CHUNKS` while tuning.

Expensive intermediate tables are cached under:

```text
stitched/analysis_cache/interaction_analysis/
```

Set `FORCE_REBUILD_CACHE = True` in the script to recompute them.

It plots:

- directed cluster-pair interaction counts;
- antenna/source interaction locations by occupancy cluster;
- body/receiver interaction locations by occupancy cluster;
- interaction counts by time since light on, averaged over light-cycle days by default;
- spatial interaction heatmaps tiled by cluster and time-of-day bin, averaged over light-cycle days by default;
- immobile-bout tests that threshold speed, count interactions involving each immobile ant, and correlate interaction count with time to mobility or bout length.
- wake-prediction regressions that compare elapsed immobility time, cumulative interaction count, and their weighted combination as predictors of waking from immobility.

Helper functions live in `analysis/interaction_analysis_utils.py`.

## Manual Commands

Track all chunks locally:

```bash
python tracking/colony/combine_batch.py \
  --input_folder /path/to/panorama_pkls \
  --output_path /path/to/tracks \
  --side both \
  --runner local \
  --skip_existing
```

Track all chunks with Slurm:

```bash
python tracking/colony/combine_batch.py \
  --input_folder /path/to/panorama_pkls \
  --output_path /path/to/tracks \
  --side both \
  --runner slurm \
  --python_bin /bucket/ReiterU/sam/miniforge3/envs/aruco_env/bin/python \
  --logs_dir /path/to/jobs/block02/logs/tracking_workers \
  --job_ids_file /path/to/jobs/block02/state/tracking_job_ids_block02.txt \
  --skip_existing
```

Run one chunk interaction locally:

```bash
python tracking/colony/interaction_one_chunk.py \
  --chunk_file /path/to/tracks/20260515_142047_chunk000_left.parquet \
  --output_path /path/to/interactions \
  --max_frames none \
  --frame_batch_size 3000 \
  --progress_every_frames 500 \
  --skip_existing
```

Run all chunk interactions through Slurm:

```bash
python tracking/colony/interaction_batch.py \
  --input_folder /path/to/tracks \
  --output_path /path/to/interactions \
  --side both \
  --runner slurm \
  --logs_dir /path/to/jobs/block02/logs/interaction_workers \
  --job_ids_file /path/to/jobs/block02/state/interaction_job_ids_block02.txt \
  --complete_job_id_file /path/to/jobs/block02/state/interaction_complete_job_id_block02.txt \
  --complete_marker_path /path/to/jobs/block02/state/interactions_complete_block02.ok \
  --skip_existing
```

Run one per-track operation through the generic fanout:

```bash
bash scripts/per_track_slurm_fanout.sh \
  --per_track_dir /flash/ReiterU/ant_tmp/$USER/colony_pipeline/20260515/block02/stitched/per_track \
  --operation_script analysis/compute_track_speed_vector.py \
  --operation_name speed_vector \
  --output_name speed_vectors
```

## Visual Tracking Debugger

Use `tracking/gui/multicam_tracking_viewer.py` to play synchronized camera videos or image sequences with stitched tracking overlays projected from panorama coordinates back into each camera.

```bash
python tracking/gui/multicam_tracking_viewer.py \
  --hmats /path/to/initial_H_mats.npz \
  --video_dir /path/to/block02 \
  --cameras 3,4,8,9 \
  --tracks /path/to/stitched/per_track/TrackID_0017_all_142047_left.parquet \
  --track_ids 17 \
  --start_frame 2418500 \
  --trail 24
```

For current tracking outputs, the main overlay follows `TrackX/TrackY`; toggles can also show raw ArUco coordinates, SLEAP anchor coordinates, and unchanged SLEAP skeleton bodypoints.

## Tracking Notes

- `TrackID` comes from ArUco `Instance`.
- Dense ArUco H5 slot index is treated as `Instance`.
- ArUco detections are filtered to those near a SLEAP anchor before tracking decisions use them.
- Existing tracks are updated by same-ID ArUco/SLEAP matches first using ArUco position, then by isolated SLEAP continuity using SLEAP position, then by filtered ArUco-only anchor keep-alive.
- Recent same-ID respawns must be spatially consistent with the previous position. By default, a consecutive-frame ArUco update is limited by `max_distance`; ArUco reacquisition grows by `lost_track_max_distance` per missed frame, defaulting to `max_distance`.
- SLEAP-only continuity is always capped at `max_distance`; it does not grow with missed frames.
- After `lost_track_max_frames`, an ID can be treated as a fresh acquisition again.
- Duplicate same-ID ArUco detections in one frame are handled by choosing the candidate closest to the previous track position for keep-alive.

## Main Entry Points

| Script | Purpose |
|---|---|
| `tracking/colony/submit_blocks_pipeline.sh` | Production block/chunk Slurm pipeline. |
| `tracking/colony/pipeline.py` | Single-block map/combine/stitch orchestrator. |
| `tracking/colony/map_combine.py` | Panorama mapping for ArUco and SLEAP files. |
| `tracking/colony/combine_one_chunk.py` | One chunk/side tracking worker. |
| `tracking/colony/combine_batch.py` | Batch tracking launcher for local or Slurm workers. |
| `tracking/colony/interaction_one_chunk.py` | One chunk/side directed interaction worker. |
| `tracking/colony/interaction_batch.py` | Batch interaction launcher for local or Slurm workers. |
| `scripts/per_track_slurm_fanout.sh` | Generic one-job-per-track fanout wrapper. |
| `analysis/compute_track_speed_vector.py` | Per-track speed vector operation. |
| `analysis/compute_track_colony_presence_vector.py` | Per-track colony in/out vector operation. |
| `analysis/compute_track_grid_occupancy.py` | Per-track normalized grid occupancy operation. |
| `analysis/sleep_label_gui.py` | Tk GUI for crop-video per-frame sleep/wake label vectors. |
| `analysis/export_sleep_crop_videos.py` | Export per-ant crop videos for fast sleep/wake labeling. |
| `analysis/sleep_classifier.py` | Train/apply random-forest supervised sleep classifier. |
| `analysis/sleep_classifier_features.py` | Shared posture and velocity feature extraction helpers. |
| `analysis/colony_speed.py` | VS Code/Jupyter interactive speed and colony-presence plots. |
| `analysis/grid_occupancy.py` | VS Code/Jupyter interactive grid occupancy clustering plots. |
| `analysis/cluster_time_of_day_occupancy.py` | Local time-of-day occupancy analysis by cluster. |
| `analysis/interaction_analysis.py` | VS Code/Jupyter interactive interaction spatial and time-of-day plots. |
| `tracking/stitch_tracks.py` | Chunk/block stitcher and trajectory PNG writer. |
| `run_aruco.py` | ArUco detection for one video. |

## Optional ArUco Detection

For one video:

```bash
python run_aruco.py \
  --video-file /path/to/video.avi \
  --output-path /path/to/output \
  --output-format h5
```
