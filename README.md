# AntsArray

Tracking and analysis pipeline for mapping multi-camera ArUco/SLEAP detections into panorama coordinates, combining detections into ant tracks, stitching those tracks across chunks, and running block-level colony analyses.

Run commands from the repository root unless noted otherwise.

## Environment

Local development:

```bash
uv sync
```

For interactive plotting dependencies, use `uv sync --extra interactive`.
The dependency versions are recorded in `uv.lock`; use `uv sync --locked` (with
the same extras) when reproducing a checkout. Run the synthetic regression suite
without recording data or Slurm access with:

```bash
uv sync --locked --extra interactive --extra test
MPLBACKEND=Agg uv run --locked --extra interactive --extra test python -m pytest -q
```

The supported behavioral workflow is
[`analysis/eigenposture_interactive.py`](analysis/eigenposture_interactive.py).
Its helpers, preprocessing, contact geometry, sleep-motion processing, and
block-combination tools remain in the checkout. Superseded analysis scripts and
dashboards have been removed; their committed versions remain in Git history.
See [analysis workflows](analysis/README.md)
and [block combination](tracking/colony/block_combination.md) for entry points.

Matching a local run also requires the same external inputs and settings:
recordings/detections or finished tracks, calibration homographies, panorama
annotations, and any trained SLEAP or sleep-classifier model. These are dataset
assets, not bundled source files. Pass the dataset/model paths explicitly (for
example `ANTS_DATASET_ROOT` and `--sleep_model`/`SLEEP_MODEL`) and retain the run
metadata alongside the data. Slurm, cluster modules, and mounted storage remain
required for cluster launchers. Generated figures, caches, logs, and run outputs
under `analysis_outputs/` are excluded from Git.

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

Use `tracking/colony/submit_blocks_pipeline.sh` for production colony tracking, per-track analyses, and chunk-level interactions.

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
    analysis_outputs/eigenposture/
  jobs/
    block02/
      scripts/
      logs/
        submit/
        tracking_workers/
        stitching/
        per_track_analysis_submit/
        interaction_workers/
      state/
```

The data folders under `block02/` are copied back to the corresponding bucket block:

```text
/bucket/.../<date>/block02/panorama_pkls
/bucket/.../<date>/block02/tracks
/bucket/.../<date>/block02/stitched
/bucket/.../<date>/block02/interactions
/bucket/.../<date>/block02/analysis_outputs/eigenposture
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
4. A per-track analysis submitter runs after stitching and submits posture interpolation, speed, sleep-motion, and grid-occupancy workers. Legacy mode also runs colony-presence and sleep-label workers.
5. An interaction submitter job runs after all tracking workers finish and submits one worker per chunk/side interaction parquet.
6. The standard `eigenposture_interactive.py` analysis waits for stitched tracks,
   speed/motion/grid caches, and complete skeleton contacts. A Slurm job builds
   the full-recording model packet and runs the interactive cells headlessly.
7. A login-side transfer watcher waits for fresh stitch, per-track cache,
   interaction, and eigenposture completion markers, then rsyncs outputs to bucket.

The same pattern is used for generic per-track analysis with `scripts/per_track_slurm_fanout.sh`: one worker per `TrackID_*.parquet`, one completion marker after all workers, then a login-side transfer watcher.

## Pipeline Stages

### 1. Panorama PKLs

`tracking/colony/pipeline.py` calls `tracking/colony/map_combine.py` to map per-camera ArUco and SLEAP detections through `initial_H_mats.npz`.

Both the pipeline and standalone `map_combine.py` read the left/right split from the block's `panorama_regions.csv`,
falling back to the date folder's annotations. It uses the midpoint between the
two full-arena rectangles in raw tracking pixels. In
`tracking/gui/panorama_region_annotator.py`, the **Left arena** and **Right arena**
buttons draw rectangles with fixed `arena_left` and `arena_right` labels. The GUI
previews the tracking split as a red line and saves that line in the annotated
panorama. Redrawing a side replaces its arena rectangle and can be undone.
Colony/nest rectangles are excluded: their positions within the arenas do not
define the divider. Missing arena annotations stop automatic mapping with an
actionable error. Older `arena`, `arenaL`/`arenaR`, and underscored arena labels
remain readable. If neither file exists, mapping searches earlier dated recording
folders under the same dataset root, including their `block*/panorama_regions.csv`
files. It selects the newest recording date strictly before the current date;
within that date, it selects the most recently modified annotation file. Recording
folder names may be `YYYYMMDD` or start with `YYYYMMDD_` / `YYYYMMDD-`.
The selected source and split are logged. Future dates are excluded. If no earlier
file exists, or the selected annotations are invalid, mapping stops with a clear
error; `--x_threshold` provides an explicit override, which is also logged.
Rebuild panorama PKLs when changing the split.

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

To combine finished **consecutive recording blocks**, run
`tracking/colony/combine_blocks.py DATA_FOLDER` on Deigo. It fans out by ant,
preserves recording-time gaps and colony identity, combines compatible analysis
caches, and publishes `DATA_FOLDER/continous_stitched` after validation.
See [block combination](tracking/colony/block_combination.md) for usage, cache
handling, provenance, and retry instructions.

Outputs:

```text
block02/stitched/
  per_track/
    TrackID_0017_all_142047_left.parquet
  track_pngs/
    TrackID_0017_all_142047_left.png
```

Stitched parquets preserve time gaps using frame counts and chunk timing metadata.

### 4. Per-Track Analysis

After stitching, `scripts/per_track_slurm_fanout.sh` prepares interpolated posture,
speed, motion, and grid caches over `stitched/per_track/`. The optional legacy mode
(`--no_eigenposture`) also produces colony-presence and old sleep labels:

```text
block02/stitched/posture_interpolated/
block02/stitched/colony_presence_vectors/
block02/stitched/speed_vectors/
block02/stitched/sleep_motion/
block02/stitched/sleep_motion_labels/
block02/stitched/grid_occupancy_histograms/
block02/stitched/sleep_predictions/
```

Disable this stage with `--no_per_track_analysis`, or disable only sleep prediction with `--no_sleep_predictions`. Override the sleep classifier model with `--sleep_model /path/to/sleep_random_forest.joblib`; leaving it empty uses the production default in `analysis/compute_track_sleep_predictions.py`.

Motion-based sleep labels run after the body/antenna motion caches finish and
are included in the completion markers required for publication. They are
separate from the optional trained-classifier predictions. Occupancy grids
default to 0.25 mm bins; `--grid_size_mm` overrides the bin size.

`analysis/compute_track_posture_interpolation.py` linearly fills bounded gaps of
at most one second (24 missing frames at 24 fps) in SLEAP bodypoints 1–9. Every
intervening frame must retain a matched tag anchor and tracked position in the
same SLEAP camera and recording. It never fills a missing identity, a whole-pose
loss, an unbounded gap, or the tag trajectory. Existing finite coordinates and
frame/timing metadata are preserved. The pipeline passes its `--fps` setting.

Each worker writes a separate full parquet under
`posture_interpolated/per_track/<track_stem>/<track_name>.parquet`, with
`PostureInterpolated` and `PostureGapFrames` columns. Its
`posture_interpolation_metadata.json` records source size/mtime, parameters,
code hash, per-node gap counts, complete-frame counts, and elapsed time.
Detector confidence is never invented for added nodes. The block transfer waits
for `posture_interpolation_complete.ok`. This operation recomputes after each
stitch so a stale per-track success marker cannot silently reuse old posture.
Original `stitched/per_track/` files and existing analysis inputs are unchanged;
using interpolated posture in a report requires an explicit downstream selection.

To run just this new fan-out on a previously tracked block (on the login node):

```bash
bash scripts/per_track_slurm_fanout.sh \
  --per_track_dir /bucket/ReiterU/Ants/basler/20260928/block01-sync/stitched/per_track \
  --operation_script analysis/compute_track_posture_interpolation.py \
  --operation_args '--fps 24 --max_gap_seconds 1.0'
```

The standard downstream report is `analysis/eigenposture_interactive.py`, run
by `analysis/eigenposture_pipeline.py` after these inputs and interactions finish.
It writes the model packet and PNG/PDF report to
`analysis_outputs/eigenposture/` in the original block. It learns states from
both colonies; the current script fits ant classes and downstream comparisons
for the right colony. Annotations fall back to the latest earlier recording.
The standard report derives putative sleep from the discovered states. The
legacy motion-threshold and trained-classifier sleep jobs run only when
`--no_eigenposture` selects the older downstream workflow.

Use `--eigenposture_python_bin /path/to/python` for an environment with
`analysis/eigenposture_leiden_requirements.txt` installed. Its default is
`~/.venvs/ants-eigenposture/bin/python`; dependencies are checked before jobs are
submitted. The tracking Python can remain the unit tracking environment.
Report resources default to 4 CPUs, 128 GB, and 2 days, adjustable with
`--eigenposture_cpus`, `--eigenposture_mem`, and `--eigenposture_time`.
`--no_eigenposture` opts out of the report. `--no_analysis` and
`--no_per_track_analysis` also disable it; a report requires interactions.
The lightweight login watcher log is `jobs/<block>/logs/eigenposture/watcher.log`.

### 5. Chunk Interaction Analysis

`tracking/colony/interaction_batch.py` submits one worker per chunk/side track parquet. Each worker runs `tracking/colony/interaction_one_chunk.py`.

For each frame, the worker measures the minimum distance between all finished
skeleton segments and nodes, recording each unordered ant pair once when that
distance is at most 0.1 mm by default. The worker's
`--micro_interaction_distance_mm` overrides the threshold. A spatial index
finds candidates without a track-center cutoff.

Outputs are flat and intentionally minimal:

```text
block02/interactions/
  20260515_142047_chunk000_left.parquet
  20260515_142047_chunk000_right.parquet
```

Each interaction parquet contains only:

```text
Frame, ant_a, ant_b, distance_mm
```

Each parquet has a `.metadata.json` sidecar recording the geometry, parameters,
and source-file identity. Existing outputs are reused only when that metadata
matches. Older directed antenna/body interaction caches are incompatible with
this skeleton-contact definition.

The completion marker lives in `jobs/block02/state/interactions_complete_block02.ok`, not in the interaction output folder.

### 6. Transfer Back To Bucket

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

New speed caches reject values above **20 mm/s** as missing (`NaN`). Set
`--speed_max_mm_s` on `tracking/colony/submit_blocks_pipeline.sh` or
`scripts/analysis_after_stitch.sh` to override this cutoff; `none` disables it.
Both wrappers also accept `SPEED_MAX_MM_S` as an environment default.
The standalone calculator uses `--max_speed_mm_s` and the same 20 mm/s default.
The selected cutoff is recorded in `speed_metadata.json`. Existing caches keep
the threshold they were generated with; changing this default does not rebuild them.

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

To start at one crop video directly while still allowing previous/next navigation through the other crop videos in the same folder:

```bash
uv run --no-project --with numpy --with pandas --with pyarrow --with opencv-python --with pillow python analysis/sleep_label_gui.py \
  --video "<CROP_VIDEO.mp4>"
```

Optional arguments:

- `--labels_dir /path/to/label_vectors`: write labels somewhere other than the crop video's `label_vectors/` folder.
- `--start_frame 1234`: open the first video at a specific frame.
- `--single_video`: with `--video`, load only that file instead of the surrounding folder.

The GUI saves per-frame label vectors beside the crop videos by default:

```text
sleep_crop_videos/<window>/label_vectors/
  <crop_video>_labels.npy
  <crop_video>_label_text.npy
  <crop_video>_labels.parquet
  <crop_video>_metadata.json
```

The GUI now stores text labels per frame in `<crop_video>_label_text.npy` and in the `label` column of `<crop_video>_labels.parquet`. The legacy `<crop_video>_labels.npy` is still written for sleep/wake compatibility, with values `-1` for unlabeled or non-sleep/wake custom labels, `0` for wake, and `1` for sleep. The bar under the seek slider shows unlabeled frames in gray, wake in blue, sleep in orange, and custom labels in deterministic additional colors. Clicking the bar seeks to that part of the video.

Basic labeling workflow:

- Press `s` or the `Sleep` button to start labeling sleep from the current frame.
- Press `w`, `n`, or the `Wake` button to start labeling wake from the current frame.
- Type another state in the `Label` text box, then press Enter or `Apply` to start labeling that text label.
- While a label is active, normal playback paints each frame the playhead reaches. Seeking to a different frame resets the active label anchor, so skipped frames are not filled.
- Press the same label again, `e`, or `End` to stop the active label at the current frame.
- Press `[` to set a range start, then press `s`, `w`, or apply the text label at another frame to label that whole interval once.
- Press `c` or `Clear` to start clearing as `unlabeled`; playback will keep clearing frames until you press `End` or choose another label. After setting a range start, `Clear` clears that selected interval once.
- Use `Save` or `ctrl+s` to save manually. Labels are also saved when changing videos or closing the GUI.

Useful hotkeys:

- space: play/pause;
- left/right: previous/next frame;
- `a`/`d`: jump backward/forward 1 second;
- `A`/`D`: play backward/forward at 2x, then faster up to 30x on repeated presses;
- `s`: start/switch/end `sleep`;
- `w` or `n`: start/switch/end `wake`;
- Enter in the `Label` box: start/switch/end the typed label;
- `c`: start/switch to clearing as `unlabeled`, or clear the selected range;
- `e`: end the active label at the current frame;
- `[`: set range start for a one-shot interval label;
- `,`/`.`: previous/next crop video;
- `ctrl+s`: save.

### Supervised Sleep Classifier

Train the random-forest classifier:

```bash
/home/sam-reiter/miniforge3/envs/ants/bin/python analysis/sleep_classifier.py train \
  --labels_dir /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/sleep_crop_videos/random_100_600s_min75pct_480px_20260622_161247/label_vectors \
  --per_track_dir /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/per_track \
  --speed_root /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/speed_vectors \
  --out /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/sleep_crop_videos/random_100_600s_min75pct_480px_20260622_161247/sleep_wake_speed_only_classifier \
  --feature_set speed_only \
  --n_estimators 100 \
  --max_depth 8
```

The fast production fanout model is intentionally small: a balanced random forest trained from the existing `speed_vectors` output using `speed_mm_s`, `speed_log1p_mm_s`, and 1/5/30 second rolling speed mean/std/valid-fraction features. Use `--feature_set posture_motion` if you want the slower posture-bodypoint model. Outputs include `sleep_random_forest.joblib`, `feature_importance.csv`, `training_report.txt`, `training_predictions.parquet`, and summary PNGs.

Apply the classifier to tracks:

```bash
/home/sam-reiter/miniforge3/envs/ants/bin/python analysis/sleep_classifier.py predict \
  --model /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/sleep_crop_videos/random_100_600s_min75pct_480px_20260622_161247/sleep_wake_speed_only_classifier/sleep_random_forest.joblib \
  --per_track_dir /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/per_track \
  --speed_root /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/speed_vectors \
  --out /home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/sleep_predictions
```

For cluster-scale fanout, use the per-track operation script. It writes one folder per ant track with `sleep_predictions.parquet`, dense `sleep_probability_f4.npy`, `wake_probability_f4.npy`, `predicted_sleep_i1.npy`, `sleep_bouts.parquet`, and `sleep_prediction_metadata.json`.

```bash
bash scripts/per_track_slurm_fanout.sh \
  --per_track_dir /bucket/ReiterU/Ants/basler/20260515/block02/stitched/per_track \
  --operation_script analysis/compute_track_sleep_predictions.py \
  --run_workdir /home/s/samuel-reiter/AntsArray \
  --conda_bin /bucket/ReiterU/sam/miniforge3/bin/conda \
  --conda_env aruco_env \
  --skip_existing
```

With a bucket `--per_track_dir`, the wrapper writes work to `/flash/ReiterU/ant_tmp/$USER/colony_pipeline/<date>/<block>/stitched/sleep_predictions` and copies results back to `/bucket/ReiterU/Ants/basler/<date>/<block>/stitched/sleep_predictions`. The operation defaults to the production speed-only model and the sibling `stitched/speed_vectors` folder. If worker Python lacks `joblib`/`sklearn`, it automatically uses the model's `*_portable.npz` sidecar. Override paths with `--operation_args "--model MODEL --speed_root SPEED_ROOT"` if needed.

The block pipeline now submits this sleep-prediction fanout after speed vectors by default. Set `SLEEP_MODEL` in `tracking/colony/submit_blocks_pipeline.sh` or pass `--sleep_model MODEL` to override the classifier for that run.

If `speed_vectors` already exist, `analysis/commands.sh` can submit the same sleep-prediction fanout with:

```bash
RUN_SLEEP_PREDICTIONS=1 \
bash analysis/commands.sh /flash/ReiterU/ant_tmp/$USER/colony_pipeline/20260515/block02/stitched/per_track
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

## Behavioral Analysis

Open [`analysis/eigenposture_interactive.py`](analysis/eigenposture_interactive.py)
and run its `# %%` cells in order. This is the retained behavioral analysis:
body-aligned posture PCA, behavioral states, ant classes from state proportions,
and downstream spatial, speed, sleep, trip, and return-contact comparisons.
[`analysis/eigenposture_task_report.py`](analysis/eigenposture_task_report.py)
runs the same cells headlessly and produces the PNG/PDF report. See the
[workflow guide](analysis/README.md) and
[methods and parameter choices](analysis/eigenposture_interactive.md).

`analysis/grid_occupancy.py` prepares the independent spatial reference labels
and QC plots used by that report. Its former speed, sleep, trip, rhythm, and
contact-analysis cells have been removed. Shared utility modules remain where
required by the supported report, preprocessing, or tracking review tools.

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
| `tracking/colony/interaction_one_chunk.py` | One chunk/side undirected skeleton-contact worker. |
| `tracking/colony/interaction_batch.py` | Batch interaction launcher for local or Slurm workers. |
| `scripts/per_track_slurm_fanout.sh` | Generic one-job-per-track fanout wrapper. |
| `analysis/compute_track_speed_vector.py` | Per-track speed vector operation. |
| `analysis/compute_track_colony_presence_vector.py` | Per-track colony in/out vector operation. |
| `analysis/compute_track_grid_occupancy.py` | Per-track normalized grid occupancy operation. |
| `analysis/compute_track_sleep_predictions.py` | Per-track sleep/wake classifier prediction operation. |
| `analysis/sleep_label_gui.py` | Tk GUI for crop-video per-frame sleep/wake label vectors. |
| `analysis/export_sleep_crop_videos.py` | Export per-ant crop videos for fast sleep/wake labeling. |
| `analysis/sleep_classifier.py` | Train/apply random-forest supervised sleep classifier. |
| `analysis/sleep_classifier_features.py` | Shared posture and velocity feature extraction helpers. |
| `analysis/eigenposture_interactive.py` | Supported interactive behavioral analysis. |
| `analysis/eigenposture_task_report.py` | Headless PNG/PDF report from the same analysis cells. |
| `analysis/grid_occupancy.py` | Prepare independent spatial reference labels and QC plots. |
| `tracking/colony/combine_blocks.py` | Combine tracked blocks and their preprocessing caches. |
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
