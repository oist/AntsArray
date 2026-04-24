# AntsArray

Tracking and analysis pipeline for combining ArUco + SLEAP detections into ant tracks.

## Setup

This project can use `uv` for dependency management:

```bash
uv sync
```

## Pipelines At A Glance

There are two main workflows in this repo.

1. **Colony / chunked multi-camera data**
2. **Single-ant / arena-segmented data across runs**

If you are not sure where to start, use this rule:

- If your files are per-camera chunk outputs (`camXX_..._chunkNNN...`), use the **colony** flow.
- If your input is timestamp folders (`YYYYMMDD-HHMMSS`) plus arena masks, use the **single-ant** flow.

## Script Map (What Each Entry Point Does)

| Script | Purpose | Typical Input | Main Output |
|---|---|---|---|
| `run_aruco.py` | Detect ArUco markers in video | One video file | `*_aruco_tracks.h5` (+ optional CSV/H5 detections) |
| `tracking/map_combine.py` | Homography-map per-camera ArUco/SLEAP chunk data into panorama coordinates | Folder with per-camera chunk files + homography `.npz` | Flat left/right panorama `.pkl` files per chunk |
| `tracking/combine_tracks_one_chunk.py` | Merge ArUco + SLEAP panorama PKLs for one chunk/side into track parquet | One representative chunk `.pkl` | `<dataset>_chunkNNN_<side>.parquet` |
| `tracking/submit_chunks.sh` | SLURM wrapper to run `combine_tracks_one_chunk.py` over all chunk keys | Folder of left/right panorama `.pkl` files | Many per-chunk parquet files |
| `tracking/aruco_track.py` | Flat-folder merger by dataset prefix (non-chunk-subfolder mode) | Folder with `*sleap_panorama_x_left/right*.pkl` and `*aruco_panorama_x_left/right*.pkl` | `<prefix>_left.parquet`, `<prefix>_right.parquet` |
| `analysis/single_ant_over_chunks.py` | Stitch many parquet files into per-track continuous outputs | Folder of chunk/run parquet files | `out_dir/per_track/TrackID_....parquet` |
| `tracking/single_ant.py` | Build per-run, per-camera, per-arena parquet files using ArUco true frame counts (SLEAP or ArUco-only detections) | Root with timestamp subfolders + `arena_seg` masks | `<timestamp>_<cam>_<arena>.parquet` (+ per-track variants) |
| `tracking/single_ant_pipeline.py` | Run `single_ant.py` then stitch with `single_ant_over_chunks.py` in one command | Same as `single_ant.py` | Per-run parquet + stitched `per_track` parquet |
| `get_arena_seg` | Save first frame of each `.avi` for manual arena segmentation | Folder of `.avi` files | `.png` frames to annotate |

## Colony Pipeline (Recommended Order)

### 1) ArUco detection

```bash
python run_aruco.py \
  --video-file /path/to/video.avi \
  --output-path /path/to/output \
  --output-format h5
```

### 2) Map all cameras into panorama coordinates

```bash
python tracking/map_combine.py \
  --hmats /path/to/initial_H_mats.npz \
  --data_dir /path/to/chunk_files \
  --outdir /path/to/panorama_pkls \
  --mode both
```

Notes:
- ArUco inputs are expected as H5/HDF5.
- `num_frames` is taken from ArUco H5 and carried in ArUco panorama payloads.
- If per-camera `num_frames` differ within a chunk, the script warns and uses the largest value.

### 3) Merge panorama detections into tracking parquet

Single chunk test:

```bash
python tracking/combine_tracks_one_chunk.py \
  --input_file /path/to/20251118_121514_chunk000_aruco_panorama_x_left1740.pkl \
  --output_path /path/to/tracks
```

Batch (SLURM):

```bash
bash tracking/submit_chunks.sh \
  --input_folder /path/to/panorama_pkls \
  --output_path /path/to/tracks \
  --side both
```

### 4) Stitch chunks into per-track files

```bash
python analysis/single_ant_over_chunks.py \
  --input_dir /path/to/tracks \
  --out_dir /path/to/stitched \
  --fps 24
```

## Single-Ant Pipeline (Arena-Based)

Expected root layout:

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

Mask lookup:
- Preferred folder: `ROOT/arena_seg`
- Fallback folder: `ROOT/seg_arena`
- Filenames can be camera-specific (`camXX*.png`) or generic (`*arena_seg*.png`)

Run full single-ant flow in one command:

```bash
python tracking/single_ant_pipeline.py \
  --input_folder /path/to/ROOT \
  --out_dir /path/to/output \
  --detection_source auto
```

`--detection_source` options:
- `auto` (default): use SLEAP if available per chunk; otherwise use ArUco XY directly.
- `sleap`: require SLEAP CSVs.
- `aruco`: ignore SLEAP CSVs and track from ArUco data only.

In `aruco` mode, `single_ant.py` assumes one ant:
- ArUco marker ID is ignored for final track identity.
- Output uses one continuous track (`track=0`), reducing split files from ID flicker.
- If a frame has multiple detections, the selected XY is the one closest to the previous selected frame.

Or run stages manually:

```bash
python tracking/single_ant.py --input_folder /path/to/ROOT --out_dir /path/to/output --detection_source aruco
python analysis/single_ant_over_chunks.py --input_dir /path/to/output --out_dir /path/to/output --fps 24
```

## Known Gaps / Active Notes

- Sleep classification should currently rely more on speed than angle in problematic recordings.
- Long-recording stitching should continue to account for real timestamp gaps between files.
- Validate and monitor variable chunk-length behavior whenever syncing new datasets.
