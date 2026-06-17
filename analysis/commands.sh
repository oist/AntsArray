#!/usr/bin/env bash
set -euo pipefail

# Edit this for a different dataset/block.
PER_TRACK_DIR="${1:-/flash/ReiterU/ant_tmp/samuel-reiter/colony_pipeline/20260515/block02/stitched/per_track}"

REPO_DIR="/home/s/samuel-reiter/AntsArray"
CONDA_BIN="/bucket/ReiterU/sam/miniforge3/bin/conda"
CONDA_ENV="aruco_env"
BUCKET_DATA_ROOT="/bucket/ReiterU/Ants/basler"

# Full command: colony presence vectors.
bash scripts/per_track_slurm_fanout.sh \
  --per_track_dir "${PER_TRACK_DIR}" \
  --operation_script analysis/compute_track_colony_presence_vector.py \
  --operation_name colony_presence \
  --output_name colony_presence_vectors \
  --run_workdir "${REPO_DIR}" \
  --conda_bin "${CONDA_BIN}" \
  --conda_env "${CONDA_ENV}" \
  --bucket_data_root "${BUCKET_DATA_ROOT}"

# Full command: speed vectors.
bash scripts/per_track_slurm_fanout.sh \
  --per_track_dir "${PER_TRACK_DIR}" \
  --operation_script analysis/compute_track_speed_vector.py \
  --operation_name speed_vector \
  --output_name speed_vectors \
  --run_workdir "${REPO_DIR}" \
  --conda_bin "${CONDA_BIN}" \
  --conda_env "${CONDA_ENV}" \
  --bucket_data_root "${BUCKET_DATA_ROOT}"

# Full command: 10 mm grid occupancy histograms.
bash scripts/per_track_slurm_fanout.sh \
  --per_track_dir "${PER_TRACK_DIR}" \
  --operation_script analysis/compute_track_grid_occupancy.py \
  --operation_name grid_occupancy \
  --output_name grid_occupancy_histograms \
  --run_workdir "${REPO_DIR}" \
  --conda_bin "${CONDA_BIN}" \
  --conda_env "${CONDA_ENV}" \
  --bucket_data_root "${BUCKET_DATA_ROOT}"
