#!/usr/bin/env bash
set -euo pipefail

# Edit this for a different dataset/block.
PER_TRACK_DIR="${1:-/flash/ReiterU/ant_tmp/samuel-reiter/colony_pipeline/20260515/block02/stitched/per_track}"

REPO_DIR="/home/s/samuel-reiter/AntsArray"
CONDA_BIN="/bucket/ReiterU/sam/miniforge3/bin/conda"
CONDA_ENV="aruco_env"
BUCKET_DATA_ROOT="/bucket/ReiterU/Ants/basler"
SLEEP_MODEL="${SLEEP_MODEL:-}"
SLEEP_SPEED_ROOT="${SLEEP_SPEED_ROOT:-$(dirname "${PER_TRACK_DIR}")/speed_vectors}"
SLEEP_CONDA_ENV="${SLEEP_CONDA_ENV:-${CONDA_ENV}}"
RUN_SLEEP_PREDICTIONS="${RUN_SLEEP_PREDICTIONS:-0}"
SLEEP_SKIP_EXISTING="${SLEEP_SKIP_EXISTING:-auto}"

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

if [[ -n "${SLEEP_MODEL}" || "${RUN_SLEEP_PREDICTIONS}" == "1" ]]; then
  if [[ -n "${SLEEP_MODEL}" && ! -f "${SLEEP_MODEL}" ]]; then
    echo "ERROR: SLEEP_MODEL does not exist: ${SLEEP_MODEL}" >&2
    exit 2
  fi
  speed_metadata_example="$(find "${SLEEP_SPEED_ROOT}/per_track" -maxdepth 2 -name speed_metadata.json -print 2>/dev/null | head -n 1 || true)"
  if [[ -z "${speed_metadata_example}" ]]; then
    echo "ERROR: SLEEP_SPEED_ROOT has no per-track speed metadata yet: ${SLEEP_SPEED_ROOT}" >&2
    echo "       Run the speed_vector fanout first, then rerun the sleep prediction fanout." >&2
    exit 2
  fi
  sleep_operation_args=(--speed_root "${SLEEP_SPEED_ROOT}")
  if [[ -n "${SLEEP_MODEL}" ]]; then
    sleep_operation_args=(--model "${SLEEP_MODEL}" "${sleep_operation_args[@]}")
  fi
  sleep_operation_args_text="$(printf ' %q' "${sleep_operation_args[@]}")"
  sleep_operation_args_text="${sleep_operation_args_text# }"

  sleep_skip_existing_arg=()
  case "${SLEEP_SKIP_EXISTING}" in
    auto|"")
      if [[ -z "${SLEEP_MODEL}" ]]; then
        sleep_skip_existing_arg=(--skip_existing)
      fi
      ;;
    1|true|TRUE|yes|YES)
      sleep_skip_existing_arg=(--skip_existing)
      ;;
    0|false|FALSE|no|NO)
      ;;
    *)
      echo "ERROR: SLEEP_SKIP_EXISTING must be auto, 1, or 0; got ${SLEEP_SKIP_EXISTING}" >&2
      exit 2
      ;;
  esac

  # Full command: sleep/wake predictions from a trained classifier.
  bash scripts/per_track_slurm_fanout.sh \
    --per_track_dir "${PER_TRACK_DIR}" \
    --operation_script analysis/compute_track_sleep_predictions.py \
    --operation_name sleep_prediction \
    --output_name sleep_predictions \
    --operation_args "${sleep_operation_args_text}" \
    --run_workdir "${REPO_DIR}" \
    --conda_bin "${CONDA_BIN}" \
    --conda_env "${SLEEP_CONDA_ENV}" \
    --bucket_data_root "${BUCKET_DATA_ROOT}" \
    "${sleep_skip_existing_arg[@]}"
fi
