#!/usr/bin/env bash
# Submit colony tracking for every block directory under a root.
#
# For each block:
#   1. submit one SLURM job to create panorama PKLs
#   2. submit a dependent SLURM job that fans out per-chunk/side tracking jobs
#   3. submit a dependent stitch job that writes stitched/per_track outputs
#   4. submit a dependent SLURM job that fans out per-chunk interaction jobs
#   5. transfer completed flash outputs back to bucket
#
# Expected block layout:
#   <blocks_root>/<block>/data/
#
# Edit the CONFIGURATION block below, then run:
#   bash tracking/colony/submit_blocks_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIPELINE_PY="$SCRIPT_DIR/pipeline.py"
COMBINE_BATCH_PY="$SCRIPT_DIR/combine_batch.py"
INTERACTION_BATCH_PY="$SCRIPT_DIR/interaction_batch.py"
RUN_USER="${USER:-${LOGNAME:-$(id -un)}}"
HOME_DIR="${HOME:-/home/sam-reiter}"
if [[ -d "${HOME_DIR}/bucket/ReiterU" ]]; then
  DEFAULT_BUCKET_ROOT="${HOME_DIR}/bucket"
else
  DEFAULT_BUCKET_ROOT="/bucket"
fi
if [[ -d "${HOME_DIR}/flash/ant_tmp" ]]; then
  DEFAULT_FLASH_TMP="${HOME_DIR}/flash/ant_tmp"
else
  DEFAULT_FLASH_TMP="/flash/ReiterU/ant_tmp"
fi
# Conda kept only as a dead fallback: when PYTHON_BIN is a real path,
# combine_batch.py/interaction_batch.py never invoke conda (python_cmd = python_bin).
DEFAULT_CONDA_ENV="aruco_env"
DEFAULT_CONDA_BIN="${DEFAULT_BUCKET_ROOT}/ReiterU/sam/miniforge3/bin/conda"
# Conda-free default: the self-contained unit ant_tracking venv python. Combined with
# the PYTHONNOUSERSITE=1 the generated sbatch scripts already export, this venv is
# authoritative in every job (orchestrator + per-chunk workers). Override with
# --python_bin (e.g. a versioned path like .../ant_tracking/2026.07/venv/bin/python).
DEFAULT_PYTHON_BIN="/apps/unit/ReiterU/ant_tracking/venv/bin/python"

# ----------------------------- CONFIGURATION -----------------------------
# Main input paths. These are the only paths you usually need to edit.
BLOCKS_ROOT="${DEFAULT_BUCKET_ROOT}/ReiterU/Ants/basler/20260515"
HMATS="${DEFAULT_BUCKET_ROOT}/ReiterU/Ants/basler/cameraArray_calib/20260414_calibration_dataset/set0_patterns_elevated_by_2mm/frame0/initial_H_mats.npz"

# Compute nodes cannot write to /bucket. All generated outputs go here.
OUTPUT_ROOT="${DEFAULT_FLASH_TMP}/${RUN_USER}/colony_pipeline/20260515"

# Sbatch scripts and submit-side SLURM logs go here.
SUBMIT_ROOT="${OUTPUT_ROOT}/jobs"

# Which block folders under BLOCKS_ROOT to submit.
BLOCK_GLOB="block*"

# Output layout under OUTPUT_ROOT. Leave WORK_NAME empty for the existing flat
# block layout:
#   OUTPUT_ROOT/block02/panorama_pkls
#   OUTPUT_ROOT/block02/tracks
#   OUTPUT_ROOT/block02/stitched
WORK_NAME=""
LOGS_NAME="logs"

# Cluster/runtime setup.
CONDA_ENV="${DEFAULT_CONDA_ENV}"
CONDA_BIN="${DEFAULT_CONDA_BIN}"
PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
PARTITION="compute"
SBATCH_BIN="sbatch"
# If your cluster needs a module or profile command before sbatch exists, put it here.
# Example: SLURM_SETUP='source /etc/profile.d/modules.sh && module load slurm'
SLURM_SETUP=""

# Pipeline behavior.
MAP_MODE="both"
SIDE="both"
SKIP_EXISTING=1

# Panorama mapping job resources.
MAP_CPUS=8
MAP_MEM="64G"
MAP_TIME="0-12:00:00"

# Small dependent job that submits the per-chunk tracking jobs after mapping.
TRACK_SUBMIT_CPUS=1
TRACK_SUBMIT_MEM="4G"
TRACK_SUBMIT_TIME="0-01:00:00"

# Per-chunk tracking job resources.
TRACK_CPUS=32
TRACK_MEM="32G"
TRACK_TIME="0-24:00:00"

# Tracking parameters.
MAX_DISTANCE="90.0"
LOST_TRACK_MAX_FRAMES="120"
LOST_TRACK_MAX_DISTANCE=""
LOST_TRACK_ARUCO_MAX_DISTANCE=""

# Per-block stitch job resources. This creates stitched/per_track outputs.
STITCH_CPUS=8
STITCH_MEM="32G"
STITCH_TIME="0-08:00:00"
FPS="24.0"
WRITE_TRACK_PNGS=1

# Interaction jobs run after all chunk tracking jobs for a block complete.
RUN_INTERACTIONS=1
INTERACTION_OUTPUT_NAME="interactions"
INTERACTION_SUBMIT_CPUS=1
INTERACTION_SUBMIT_MEM="4G"
INTERACTION_SUBMIT_TIME="0-01:00:00"
INTERACTION_CPUS=4
INTERACTION_MEM="16G"
INTERACTION_TIME="0-12:00:00"
INTERACTION_MM_PER_PX="0.016"
INTERACTION_RADIUS_MM="8.0"
INTERACTION_MICRO_DISTANCE_MM="1.0"
INTERACTION_FRAME_BATCH_SIZE="3000"
INTERACTION_PROGRESS_EVERY_FRAMES="500"
INTERACTION_MAX_FRAMES="none"

# Login-side transfer back to /bucket after SLURM work completes.
TRANSFER_TO_BUCKET=1
TRANSFER_POLL_SECONDS=120
DELETE_FLASH_AFTER_TRANSFER=0

# Set to 1 to write sbatch scripts without submitting jobs.
DRY_RUN=0
# -------------------------------------------------------------------------

blocks_root="$BLOCKS_ROOT"
hmats="$HMATS"
output_root="$OUTPUT_ROOT"
submit_root="$SUBMIT_ROOT"
submit_root_overridden=0
block_glob="$BLOCK_GLOB"
work_name="$WORK_NAME"
logs_name="$LOGS_NAME"
conda_env="$CONDA_ENV"
conda_bin="$CONDA_BIN"
python_bin="$PYTHON_BIN"
partition="$PARTITION"
sbatch_bin="$SBATCH_BIN"
slurm_setup="$SLURM_SETUP"
map_mode="$MAP_MODE"
side="$SIDE"
skip_existing="$SKIP_EXISTING"
map_cpus="$MAP_CPUS"
map_mem="$MAP_MEM"
map_time="$MAP_TIME"
track_submit_cpus="$TRACK_SUBMIT_CPUS"
track_submit_mem="$TRACK_SUBMIT_MEM"
track_submit_time="$TRACK_SUBMIT_TIME"
track_cpus="$TRACK_CPUS"
track_mem="$TRACK_MEM"
track_time="$TRACK_TIME"
max_distance="$MAX_DISTANCE"
lost_track_max_frames="$LOST_TRACK_MAX_FRAMES"
lost_track_max_distance="$LOST_TRACK_MAX_DISTANCE"
lost_track_aruco_max_distance="$LOST_TRACK_ARUCO_MAX_DISTANCE"
stitch_cpus="$STITCH_CPUS"
stitch_mem="$STITCH_MEM"
stitch_time="$STITCH_TIME"
fps="$FPS"
write_track_pngs="$WRITE_TRACK_PNGS"
run_interactions="$RUN_INTERACTIONS"
interaction_output_name="$INTERACTION_OUTPUT_NAME"
interaction_submit_cpus="$INTERACTION_SUBMIT_CPUS"
interaction_submit_mem="$INTERACTION_SUBMIT_MEM"
interaction_submit_time="$INTERACTION_SUBMIT_TIME"
interaction_cpus="$INTERACTION_CPUS"
interaction_mem="$INTERACTION_MEM"
interaction_time="$INTERACTION_TIME"
interaction_mm_per_px="$INTERACTION_MM_PER_PX"
interaction_radius_mm="$INTERACTION_RADIUS_MM"
interaction_micro_distance_mm="$INTERACTION_MICRO_DISTANCE_MM"
interaction_frame_batch_size="$INTERACTION_FRAME_BATCH_SIZE"
interaction_progress_every_frames="$INTERACTION_PROGRESS_EVERY_FRAMES"
interaction_max_frames="$INTERACTION_MAX_FRAMES"
transfer_to_bucket="$TRANSFER_TO_BUCKET"
transfer_poll_seconds="$TRANSFER_POLL_SECONDS"
delete_flash_after_transfer="$DELETE_FLASH_AFTER_TRANSFER"
dry_run="$DRY_RUN"

usage() {
  cat <<EOF
Submit colony tracking for every block directory under a root.

For each block:
  1. submit one SLURM job to create panorama PKLs
  2. submit a dependent SLURM job that fans out per-chunk/side tracking jobs
  3. submit a dependent stitch job that writes stitched/per_track outputs
  4. submit a dependent SLURM job that fans out per-chunk interaction jobs
  5. transfer completed flash outputs back to bucket

Expected block layout:
  <blocks_root>/<block>/data/

Edit the CONFIGURATION block at the top of this script, then run:
  bash tracking/colony/submit_blocks_pipeline.sh

Optional:
  --blocks_root PATH        Override BLOCKS_ROOT.
  --hmats PATH              Override HMATS.
  --output_root PATH        Override OUTPUT_ROOT. Compute-node outputs go here.
  --submit_root PATH        Override SUBMIT_ROOT. Sbatch scripts and submit logs go here.
  --block_glob PATTERN      Block directory glob under --blocks_root. Default: block*
  --work_name NAME          Optional work subdirectory under OUTPUT_ROOT/<block>. Default: flat block dir.
  --logs_name NAME          Log directory name under SUBMIT_ROOT/<block>. Default: logs
  --conda_env NAME          Conda environment name kept for fallback. Default: aruco_env
  --conda_bin PATH          Conda executable used inside sbatch jobs.
  --python_bin PATH         Python executable used inside sbatch jobs.
  --partition NAME          SLURM partition. Default: compute
  --sbatch_bin PATH         sbatch command/path. Default: sbatch
  --slurm_setup CMD         Command to run before sbatch, e.g. module load slurm.
  --map_mode MODE           aruco, sleap, or both. Default: both
  --side SIDE               left, right, or both. Default: both
  --skip_existing           Do not overwrite existing outputs. Default: on.
  --force_recompute         Overwrite/recompute existing flash outputs.
  --map_cpus N              CPUs for panorama jobs. Default: 8
  --map_mem MEM             Memory for panorama jobs. Default: 64G
  --map_time TIME           Time for panorama jobs. Default: 0-12:00:00
  --track_cpus N            CPUs for each chunk tracking job. Default: 32
  --track_mem MEM           Memory for each chunk tracking job. Default: 32G
  --track_time TIME         Time for each chunk tracking job. Default: 0-24:00:00
  --track_submit_cpus N     CPUs for dependent fan-out submitter job. Default: 1
  --track_submit_mem MEM    Memory for dependent fan-out submitter job. Default: 4G
  --track_submit_time TIME  Time for dependent fan-out submitter job. Default: 0-01:00:00
  --max_distance FLOAT      Tracking max distance. Default: 90.0
  --lost_track_max_frames N Tracking lost-frame limit. Default: 120
  --lost_track_max_distance FLOAT
  --lost_track_aruco_max_distance FLOAT
  --stitch_cpus N           CPUs for block stitch job. Default: 8
  --stitch_mem MEM          Memory for block stitch job. Default: 32G
  --stitch_time TIME        Time for block stitch job. Default: 0-08:00:00
  --fps FLOAT               FPS used for stitched global frame offsets. Default: 24.0
  --write_track_pngs        Write trajectory PNGs during stitching. Default: on
  --no_track_pngs           Do not write trajectory PNGs during stitching.
  --no_interactions         Do not submit interaction jobs after tracking.
  --interaction_cpus N      CPUs for each chunk interaction job. Default: 4
  --interaction_mem MEM     Memory for each chunk interaction job. Default: 16G
  --interaction_time TIME   Time for each chunk interaction job. Default: 0-12:00:00
  --interaction_submit_cpus N
  --interaction_submit_mem MEM
  --interaction_submit_time TIME
  --interaction_radius_mm FLOAT
  --interaction_micro_distance_mm FLOAT
  --interaction_frame_batch_size N
  --interaction_progress_every_frames N
  --interaction_max_frames N|none
  --transfer_to_bucket      Start login-side transfer watcher after submission. Default: on
  --no_transfer_to_bucket   Do not start the login-side transfer watcher.
  --transfer_poll_seconds N Poll interval for transfer watcher. Default: 120
  --delete_flash_after_transfer
                            Delete transferred flash files after successful rsync. Default: off
  --dry_run                 Write scripts but do not submit.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --blocks_root) blocks_root="$2"; shift 2 ;;
    --hmats) hmats="$2"; shift 2 ;;
    --output_root) output_root="$2"; shift 2 ;;
    --submit_root) submit_root="$2"; submit_root_overridden=1; shift 2 ;;
    --block_glob) block_glob="$2"; shift 2 ;;
    --work_name) work_name="$2"; shift 2 ;;
    --logs_name) logs_name="$2"; shift 2 ;;
    --conda_env) conda_env="$2"; shift 2 ;;
    --conda_bin) conda_bin="$2"; shift 2 ;;
    --python_bin) python_bin="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --sbatch_bin) sbatch_bin="$2"; shift 2 ;;
    --slurm_setup) slurm_setup="$2"; shift 2 ;;
    --map_mode) map_mode="$2"; shift 2 ;;
    --side) side="$2"; shift 2 ;;
    --skip_existing) skip_existing=1; shift ;;
    --force_recompute) skip_existing=0; shift ;;
    --map_cpus) map_cpus="$2"; shift 2 ;;
    --map_mem) map_mem="$2"; shift 2 ;;
    --map_time) map_time="$2"; shift 2 ;;
    --track_cpus) track_cpus="$2"; shift 2 ;;
    --track_mem) track_mem="$2"; shift 2 ;;
    --track_time) track_time="$2"; shift 2 ;;
    --track_submit_cpus) track_submit_cpus="$2"; shift 2 ;;
    --track_submit_mem) track_submit_mem="$2"; shift 2 ;;
    --track_submit_time) track_submit_time="$2"; shift 2 ;;
    --max_distance) max_distance="$2"; shift 2 ;;
    --lost_track_max_frames) lost_track_max_frames="$2"; shift 2 ;;
    --lost_track_max_distance) lost_track_max_distance="$2"; shift 2 ;;
    --lost_track_aruco_max_distance) lost_track_aruco_max_distance="$2"; shift 2 ;;
    --stitch_cpus) stitch_cpus="$2"; shift 2 ;;
    --stitch_mem) stitch_mem="$2"; shift 2 ;;
    --stitch_time) stitch_time="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --write_track_pngs) write_track_pngs=1; shift ;;
    --no_track_pngs) write_track_pngs=0; shift ;;
    --no_interactions) run_interactions=0; shift ;;
    --interaction_cpus) interaction_cpus="$2"; shift 2 ;;
    --interaction_mem) interaction_mem="$2"; shift 2 ;;
    --interaction_time) interaction_time="$2"; shift 2 ;;
    --interaction_submit_cpus) interaction_submit_cpus="$2"; shift 2 ;;
    --interaction_submit_mem) interaction_submit_mem="$2"; shift 2 ;;
    --interaction_submit_time) interaction_submit_time="$2"; shift 2 ;;
    --interaction_radius_mm) interaction_radius_mm="$2"; shift 2 ;;
    --interaction_micro_distance_mm) interaction_micro_distance_mm="$2"; shift 2 ;;
    --interaction_frame_batch_size) interaction_frame_batch_size="$2"; shift 2 ;;
    --interaction_progress_every_frames) interaction_progress_every_frames="$2"; shift 2 ;;
    --interaction_max_frames) interaction_max_frames="$2"; shift 2 ;;
    --transfer_to_bucket) transfer_to_bucket=1; shift ;;
    --no_transfer_to_bucket) transfer_to_bucket=0; shift ;;
    --transfer_poll_seconds) transfer_poll_seconds="$2"; shift 2 ;;
    --delete_flash_after_transfer) delete_flash_after_transfer=1; shift ;;
    --dry_run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d "$blocks_root" ]]; then
  echo "ERROR: blocks root does not exist: $blocks_root" >&2
  exit 2
fi
if [[ ! -f "$hmats" ]]; then
  echo "ERROR: homography file does not exist: $hmats" >&2
  exit 2
fi
if [[ "$submit_root_overridden" -eq 0 ]]; then
  submit_root="$output_root/jobs"
fi
missing_flash_mount=""
for flash_mount in /flash/ant_tmp /flash/ReiterU/ant_tmp; do
  if [[ "$submit_root" == "$flash_mount"* || "$output_root" == "$flash_mount"* ]]; then
    if [[ ! -d "$flash_mount" ]]; then
      missing_flash_mount="$flash_mount"
    fi
  fi
done
if [[ -n "$missing_flash_mount" ]]; then
  echo "ERROR: $missing_flash_mount does not exist on this host; cannot write job files to flash." >&2
  echo "On deigo, /flash/ReiterU/ant_tmp appears to be the Reiter writable flash scratch." >&2
  exit 2
fi
mkdir -p "$submit_root"
if [[ -n "$slurm_setup" ]]; then
  eval "$slurm_setup"
fi
if [[ "$dry_run" -eq 0 ]] && ! command -v "$sbatch_bin" >/dev/null 2>&1; then
  echo "ERROR: sbatch command not found: $sbatch_bin" >&2
  echo "Edit SBATCH_BIN or SLURM_SETUP at the top of $0." >&2
  exit 2
fi
if [[ "$dry_run" -eq 0 && "$transfer_to_bucket" -eq 1 ]] && ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: transfer watcher requires rsync, but rsync was not found." >&2
  exit 2
fi

shopt -s nullglob
blocks=( "$blocks_root"/$block_glob )
if [[ ${#blocks[@]} -eq 0 ]]; then
  echo "ERROR: no block directories matched: $blocks_root/$block_glob" >&2
  exit 2
fi

submitted=0
for block_dir in "${blocks[@]}"; do
  [[ -d "$block_dir" ]] || continue

  data_dir="$block_dir/data"
  if [[ ! -d "$data_dir" ]]; then
    echo "Skipping $(basename "$block_dir"): no data/ directory" >&2
    continue
  fi

  block_name="$(basename "$block_dir")"
  if [[ -n "$work_name" ]]; then
    work_dir="$output_root/$block_name/$work_name"
  else
    work_dir="$output_root/$block_name"
  fi
  panorama_dir="$work_dir/panorama_pkls"
  tracks_dir="$work_dir/tracks"
  stitched_dir="$work_dir/stitched"
  interaction_dir="$output_root/$block_name/$interaction_output_name"
  bucket_panorama_dir="$block_dir/panorama_pkls"
  bucket_tracks_dir="$block_dir/tracks"
  bucket_stitched_dir="$block_dir/stitched"
  bucket_interaction_dir="$block_dir/$interaction_output_name"
  submit_block_dir="$submit_root/$block_name"
  script_dir="$submit_block_dir/scripts"
  logs_dir="$submit_block_dir/$logs_name"
  state_dir="$submit_block_dir/state"
  submit_logs_dir="$logs_dir/submit"
  chunk_logs_dir="$logs_dir/tracking_workers"
  stitch_logs_dir="$logs_dir/stitching"
  interaction_logs_dir="$logs_dir/interaction_workers"
  tracking_job_ids_file="$state_dir/tracking_job_ids_${block_name}.txt"
  tracking_complete_script="$script_dir/tracking_complete_${block_name}.sbatch"
  tracking_complete_done_file="$state_dir/tracking_complete_${block_name}.ok"
  tracking_complete_job_id_file="$state_dir/tracking_complete_job_id_${block_name}.txt"
  stitch_script="$script_dir/stitch_${block_name}.sbatch"
  stitch_done_file="$state_dir/stitch_${block_name}.ok"
  stitch_job_id_file="$state_dir/stitch_job_id_${block_name}.txt"
  interaction_job_ids_file="$state_dir/interaction_job_ids_${block_name}.txt"
  interaction_complete_job_id_file="$state_dir/interaction_complete_job_id_${block_name}.txt"
  interaction_complete_done_file="$state_dir/interactions_complete_${block_name}.ok"
  interaction_transfer_job_id_file="$state_dir/interaction_transfer_job_id_${block_name}.txt"
  transfer_manifest="$state_dir/transfer_manifest_${block_name}.tsv"
  transfer_script="$script_dir/transfer_to_bucket_${block_name}.sh"
  transfer_log="$logs_dir/transfer_to_bucket_${block_name}.log"
  transfer_lock_dir="$state_dir/transfer_to_bucket_${block_name}.lock"
  transfer_done_file="$state_dir/transfer_to_bucket_${block_name}.ok"
  interaction_done_file_for_transfer=""
  if [[ "$run_interactions" -eq 1 ]]; then
    interaction_done_file_for_transfer="$interaction_complete_done_file"
  fi
  mkdir -p "$script_dir" "$submit_logs_dir" "$state_dir"
  : > "$transfer_manifest"
  printf '%s\t%s\n' "$panorama_dir" "$bucket_panorama_dir" >> "$transfer_manifest"
  printf '%s\t%s\n' "$tracks_dir" "$bucket_tracks_dir" >> "$transfer_manifest"
  printf '%s\t%s\n' "$stitched_dir" "$bucket_stitched_dir" >> "$transfer_manifest"
  if [[ "$run_interactions" -eq 1 ]]; then
    printf '%s\t%s\n' "$interaction_dir" "$bucket_interaction_dir" >> "$transfer_manifest"
  fi

  map_script="$script_dir/map_${block_name}.sbatch"
  track_submit_script="$script_dir/submit_tracking_${block_name}.sbatch"
  interaction_submit_script="$script_dir/submit_interactions_${block_name}.sbatch"
  skip_existing_arg=""
  if [[ "$skip_existing" -eq 1 ]]; then
    skip_existing_arg=" \\
  --skip_existing"
  fi

  cat > "$map_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J map_${block_name}
#SBATCH -p ${partition}
#SBATCH -c ${map_cpus}
#SBATCH --mem=${map_mem}
#SBATCH -t ${map_time}
#SBATCH -o ${submit_logs_dir}/map_${block_name}_%j.out
#SBATCH -e ${submit_logs_dir}/map_${block_name}_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
mkdir -p "${panorama_dir}" "${tracks_dir}" "${logs_dir}"
"${python_bin}" "${PIPELINE_PY}" \\
  --hmats "${hmats}" \\
  --data_dir "${data_dir}" \\
  --work_dir "${work_dir}" \\
  --panorama_dir "${panorama_dir}" \\
  --tracks_dir "${tracks_dir}" \\
  --map_mode "${map_mode}" \\
  --skip_combine \\
  --skip_stitch${skip_existing_arg}
EOF
  chmod 755 "$map_script"

  extra_tracking_text=""
  if [[ -n "$lost_track_max_distance" ]]; then
    extra_tracking_text+=" \\
  --lost_track_max_distance \"${lost_track_max_distance}\""
  fi
  if [[ -n "$lost_track_aruco_max_distance" ]]; then
    extra_tracking_text+=" \\
  --lost_track_aruco_max_distance \"${lost_track_aruco_max_distance}\""
  fi
  stitch_no_pngs_arg=""
  if [[ "$write_track_pngs" -eq 0 ]]; then
    stitch_no_pngs_arg=" \\
  --no_track_pngs"
  fi

  cat > "$tracking_complete_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J track_done_${block_name}
#SBATCH -p ${partition}
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 0-00:10:00
#SBATCH -o ${submit_logs_dir}/tracking_complete_${block_name}_%j.out
#SBATCH -e ${submit_logs_dir}/tracking_complete_${block_name}_%j.err

set -euo pipefail
printf 'completed %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" > "${tracking_complete_done_file}"
EOF
  chmod 755 "$tracking_complete_script"

  cat > "$stitch_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J stitch_${block_name}
#SBATCH -p ${partition}
#SBATCH -c ${stitch_cpus}
#SBATCH --mem=${stitch_mem}
#SBATCH -t ${stitch_time}
#SBATCH -o ${submit_logs_dir}/stitch_${block_name}_%j.out
#SBATCH -e ${submit_logs_dir}/stitch_${block_name}_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
mkdir -p "${stitched_dir}" "${stitch_logs_dir}"
"${python_bin}" "${PIPELINE_PY}" \\
  --hmats "${hmats}" \\
  --data_dir "${data_dir}" \\
  --work_dir "${work_dir}" \\
  --panorama_dir "${panorama_dir}" \\
  --tracks_dir "${tracks_dir}" \\
  --stitched_dir "${stitched_dir}" \\
  --logs_dir "${stitch_logs_dir}" \\
  --skip_map \\
  --skip_combine \\
  --side "${side}" \\
  --fps "${fps}"${stitch_no_pngs_arg}${skip_existing_arg}
printf 'completed %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" > "${stitch_done_file}"
EOF
  chmod 755 "$stitch_script"

  cat > "$track_submit_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J submit_track_${block_name}
#SBATCH -p ${partition}
#SBATCH -c ${track_submit_cpus}
#SBATCH --mem=${track_submit_mem}
#SBATCH -t ${track_submit_time}
#SBATCH -o ${submit_logs_dir}/submit_tracking_${block_name}_%j.out
#SBATCH -e ${submit_logs_dir}/submit_tracking_${block_name}_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
mkdir -p "${tracks_dir}" "${chunk_logs_dir}"
rm -f "${tracking_job_ids_file}" "${tracking_complete_done_file}" "${tracking_complete_job_id_file}" "${stitch_done_file}" "${stitch_job_id_file}"
"${python_bin}" "${COMBINE_BATCH_PY}" \\
  --input_folder "${panorama_dir}" \\
  --output_path "${tracks_dir}" \\
  --side "${side}" \\
  --runner slurm \\
  --logs_dir "${chunk_logs_dir}" \\
  --partition "${partition}" \\
  --cpus "${track_cpus}" \\
  --mem "${track_mem}" \\
  --time "${track_time}" \\
  --job_name "track_${block_name}" \\
  --conda_env "${conda_env}" \\
  --conda_bin "${conda_bin}" \\
  --python_bin "${python_bin}" \\
  --sbatch_bin "${sbatch_bin}" \\
  --job_ids_file "${tracking_job_ids_file}" \\
  --max_distance "${max_distance}" \\
  --lost_track_max_frames "${lost_track_max_frames}"${extra_tracking_text}${skip_existing_arg}

tracking_dependency_args=()
if [[ -s "${tracking_job_ids_file}" ]]; then
  mapfile -t tracking_job_ids < "${tracking_job_ids_file}"
  if [[ "\${#tracking_job_ids[@]}" -gt 0 ]]; then
    tracking_dependency="\$(IFS=:; echo "\${tracking_job_ids[*]}")"
    tracking_dependency_args=(--dependency=afterok:"\${tracking_dependency}")
    echo "Tracking worker dependency: \${tracking_dependency}"
  fi
else
  existing_track_count="\$(find "${tracks_dir}" -maxdepth 1 -type f -name '*.parquet' | wc -l | tr -d '[:space:]')"
  echo "No new tracking jobs were submitted; existing_track_count=\${existing_track_count} tracks_dir=${tracks_dir}"
  if [[ -z "\${existing_track_count}" || "\${existing_track_count}" -eq 0 ]]; then
    echo "ERROR: tracking job IDs file is empty and no existing track parquet files were found in ${tracks_dir}" >&2
    exit 3
  fi
fi

tracking_complete_job_id="\$("${sbatch_bin}" --parsable "\${tracking_dependency_args[@]}" "${tracking_complete_script}")"
echo "\${tracking_complete_job_id}" > "${tracking_complete_job_id_file}"
echo "Submitted tracking completion marker: \${tracking_complete_job_id}"
stitch_job_id="\$("${sbatch_bin}" --parsable "\${tracking_dependency_args[@]}" "${stitch_script}")"
echo "\${stitch_job_id}" > "${stitch_job_id_file}"
echo "Submitted stitch job: \${stitch_job_id}"
if [[ "${run_interactions}" -eq 1 ]]; then
  interaction_submit_job_id="\$("${sbatch_bin}" --parsable "\${tracking_dependency_args[@]}" "${interaction_submit_script}")"
  echo "Submitted interaction fan-out submitter: \${interaction_submit_job_id}"
fi
EOF
  chmod 755 "$track_submit_script"

  cat > "$interaction_submit_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J submit_inter_${block_name}
#SBATCH -p ${partition}
#SBATCH -c ${interaction_submit_cpus}
#SBATCH --mem=${interaction_submit_mem}
#SBATCH -t ${interaction_submit_time}
#SBATCH -o ${submit_logs_dir}/submit_interactions_${block_name}_%j.out
#SBATCH -e ${submit_logs_dir}/submit_interactions_${block_name}_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
mkdir -p "${interaction_dir}" "${interaction_logs_dir}"
rm -f "${interaction_job_ids_file}" "${interaction_complete_job_id_file}" "${interaction_complete_done_file}" "${interaction_transfer_job_id_file}"
"${python_bin}" "${INTERACTION_BATCH_PY}" \\
  --input_folder "${tracks_dir}" \\
  --output_path "${interaction_dir}" \\
  --side "${side}" \\
  --runner slurm \\
  --logs_dir "${interaction_logs_dir}" \\
  --partition "${partition}" \\
  --cpus "${interaction_cpus}" \\
  --mem "${interaction_mem}" \\
  --time "${interaction_time}" \\
  --job_name "inter_${block_name}" \\
  --conda_env "${conda_env}" \\
  --conda_bin "${conda_bin}" \\
  --python_bin "${python_bin}" \\
  --sbatch_bin "${sbatch_bin}" \\
  --job_ids_file "${interaction_job_ids_file}" \\
  --complete_job_id_file "${interaction_complete_job_id_file}" \\
  --complete_marker_path "${interaction_complete_done_file}" \\
  --mm_per_px "${interaction_mm_per_px}" \\
  --interaction_radius_mm "${interaction_radius_mm}" \\
  --micro_interaction_distance_mm "${interaction_micro_distance_mm}" \\
  --frame_batch_size "${interaction_frame_batch_size}" \\
  --progress_every_frames "${interaction_progress_every_frames}" \\
  --max_frames "${interaction_max_frames}"${skip_existing_arg}
EOF
  chmod 755 "$interaction_submit_script"

  cat > "$transfer_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

stitch_done_file="${stitch_done_file}"
interaction_done_file="${interaction_done_file_for_transfer}"
manifest="${transfer_manifest}"
transfer_log="${transfer_log}"
poll_seconds="${transfer_poll_seconds}"
delete_flash_after_transfer="${delete_flash_after_transfer}"
lock_dir="${transfer_lock_dir}"
transfer_done_file="${transfer_done_file}"
run_epoch="$(date +%s)"

log() {
  printf '[%s] %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" "\$*" | tee -a "\${transfer_log}"
}

file_mtime() {
  local path="\$1"
  stat -c %Y "\${path}" 2>/dev/null || printf '0\n'
}

fmt_elapsed() {
  local s="\$1"
  printf '%dh%02dm' "\$(( s / 3600 ))" "\$(( (s % 3600) / 60 ))"
}

acquire_lock() {
  if mkdir "\${lock_dir}" 2>/dev/null; then
    printf '%s\n' "\$\$" > "\${lock_dir}/pid"
    hostname > "\${lock_dir}/host"
    trap 'rm -rf "\${lock_dir}"' EXIT
    return 0
  fi
  local old_pid=""
  local old_host=""
  [[ -s "\${lock_dir}/pid" ]] && old_pid="\$(cat "\${lock_dir}/pid" 2>/dev/null || true)"
  [[ -s "\${lock_dir}/host" ]] && old_host="\$(cat "\${lock_dir}/host" 2>/dev/null || true)"
  if [[ "\${old_host}" == "\$(hostname)" && -n "\${old_pid}" ]] && kill -0 "\${old_pid}" 2>/dev/null; then
    log "Another transfer watcher is already running as PID \${old_pid}; exiting."
    exit 0
  fi
  log "Removing stale transfer lock: \${lock_dir}"
  rm -rf "\${lock_dir}"
  mkdir "\${lock_dir}"
  printf '%s\n' "\$\$" > "\${lock_dir}/pid"
  hostname > "\${lock_dir}/host"
  trap 'rm -rf "\${lock_dir}"' EXIT
}

log "Transfer watcher start"
log "stitch_done_file=\${stitch_done_file}"
log "interaction_done_file=\${interaction_done_file}"
log "manifest=\${manifest}"
log "run_epoch=\${run_epoch}"
if [[ -s "\${manifest}" ]]; then
  while IFS=\$'\t' read -r src dst; do
    [[ -n "\${src}" && -n "\${dst}" ]] || continue
    log "manifest_entry \${src} -> \${dst}"
  done < "\${manifest}"
fi

acquire_lock
wait_for_fresh_file() {
  local path="\$1"
  local label="\$2"
  if [[ -z "\${path}" ]]; then
    return 0
  fi
  log "Waiting for \${label}: \${path}"
  while true; do
    if [[ -s "\${path}" ]]; then
      mtime="\$(file_mtime "\${path}")"
      if [[ "\${mtime}" =~ ^[0-9]+$ ]] && (( mtime >= run_epoch )); then
        log "Fresh \${label} marker found: \${path} mtime=\${mtime}"
        return 0
      fi
      log "Ignoring stale \${label} marker: \${path} mtime=\${mtime}"
    fi
    log "still waiting for \${label} (elapsed \$(fmt_elapsed \$(( \$(date +%s) - run_epoch )))); next check in \${poll_seconds}s"
    sleep "\${poll_seconds}"
  done
}

wait_for_fresh_file "\${stitch_done_file}" "stitch"
wait_for_fresh_file "\${interaction_done_file}" "interaction"

if [[ ! -s "\${manifest}" ]]; then
  log "ERROR: missing transfer manifest: \${manifest}"
  exit 2
fi

while IFS=\$'\t' read -r src dst; do
  [[ -n "\${src}" && -n "\${dst}" ]] || continue
  if [[ ! -d "\${src}" ]]; then
    log "WARNING: source directory missing, skipping: \${src}"
    continue
  fi
  mkdir -p "\${dst}"
  log "rsync \${src}/ -> \${dst}/"
  rsync -a --partial --protect-args "\${src}/" "\${dst}/" >> "\${transfer_log}" 2>&1
  log "rsync complete: \${src}/ -> \${dst}/"
  if [[ "\${delete_flash_after_transfer}" -eq 1 ]]; then
    log "Deleting transferred source contents: \${src}"
    find "\${src}" -mindepth 1 -delete
  fi
done < "\${manifest}"

printf 'completed %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" > "\${transfer_done_file}"
log "Transfer to bucket complete. marker=\${transfer_done_file}"
EOF
  chmod 755 "$transfer_script"

  if [[ "$dry_run" -eq 1 ]]; then
    echo "[dry-run] map script: $map_script"
    echo "[dry-run] tracking submitter script: $track_submit_script"
    if [[ "$run_interactions" -eq 1 ]]; then
      echo "[dry-run] interaction submitter script: $interaction_submit_script"
    fi
    if [[ "$transfer_to_bucket" -eq 1 ]]; then
      echo "[dry-run] transfer watcher script: $transfer_script"
    fi
    submitted=$((submitted + 1))
    continue
  fi

  map_job_id="$("$sbatch_bin" --parsable "$map_script")"
  track_submit_job_id="$("$sbatch_bin" --parsable --dependency=afterok:"$map_job_id" "$track_submit_script")"
  if [[ "$run_interactions" -eq 1 ]]; then
    echo "Submitted $block_name: map job $map_job_id, tracking fan-out job $track_submit_job_id; interaction fan-out will be scheduled after tracking workers"
  else
    echo "Submitted $block_name: map job $map_job_id, tracking fan-out job $track_submit_job_id"
  fi
  if [[ "$transfer_to_bucket" -eq 1 ]]; then
    nohup "$transfer_script" >/dev/null 2>&1 &
    transfer_pid="$!"
    echo "Started transfer watcher for $block_name PID $transfer_pid; log: $transfer_log"
  fi
  submitted=$((submitted + 1))
done

if [[ "$submitted" -eq 0 ]]; then
  echo "ERROR: no block jobs were submitted." >&2
  exit 2
fi
