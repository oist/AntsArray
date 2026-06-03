#!/usr/bin/env bash
# Submit colony tracking for every block directory under a root.
#
# For each block:
#   1. submit one SLURM job to create panorama PKLs
#   2. submit a dependent SLURM job that fans out per-chunk/side tracking jobs
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

# ----------------------------- CONFIGURATION -----------------------------
# Main input paths. These are the only paths you usually need to edit.
BLOCKS_ROOT="/bucket/ReiterU/Ants/basler/20260515"
HMATS="/bucket/ReiterU/Ants/basler/cameraArray_calib/20260414_calibration_dataset/set0_patterns_elevated_by_2mm/frame0/initial_H_mats.npz"

# Compute nodes cannot write to /bucket. All generated outputs go here.
OUTPUT_ROOT="/flash/ReiterU/ant_tmp/20260515"

# Sbatch scripts and submit-side SLURM logs go here.
SUBMIT_ROOT="${OUTPUT_ROOT}/jobs"

# Which block folders under BLOCKS_ROOT to submit.
BLOCK_GLOB="block*"

# Output layout under OUTPUT_ROOT.
WORK_NAME="colony_work"
LOGS_NAME="slurm_colony_pipeline"

# Cluster/runtime setup.
CONDA_ENV="ants"
PARTITION="compute"
SBATCH_BIN="sbatch"
# If your cluster needs a module or profile command before sbatch exists, put it here.
# Example: SLURM_SETUP='source /etc/profile.d/modules.sh && module load slurm'
SLURM_SETUP=""

# Pipeline behavior.
MAP_MODE="both"
SIDE="both"
SKIP_EXISTING=0

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
dry_run="$DRY_RUN"

usage() {
  cat <<EOF
Submit colony tracking for every block directory under a root.

For each block:
  1. submit one SLURM job to create panorama PKLs
  2. submit a dependent SLURM job that fans out per-chunk/side tracking jobs

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
  --work_name NAME          Work directory name under OUTPUT_ROOT/<block>. Default: colony_work
  --logs_name NAME          Logs/scripts directory inside each work dir. Default: slurm_colony_pipeline
  --conda_env NAME          Conda environment. Default: ants
  --partition NAME          SLURM partition. Default: compute
  --sbatch_bin PATH         sbatch command/path. Default: sbatch
  --slurm_setup CMD         Command to run before sbatch, e.g. module load slurm.
  --map_mode MODE           aruco, sleap, or both. Default: both
  --side SIDE               left, right, or both. Default: both
  --skip_existing           Do not overwrite existing panorama PKLs, chunk parquets, or stitched outputs.
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
    --partition) partition="$2"; shift 2 ;;
    --sbatch_bin) sbatch_bin="$2"; shift 2 ;;
    --slurm_setup) slurm_setup="$2"; shift 2 ;;
    --map_mode) map_mode="$2"; shift 2 ;;
    --side) side="$2"; shift 2 ;;
    --skip_existing) skip_existing=1; shift ;;
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
  work_dir="$output_root/$block_name/$work_name"
  panorama_dir="$work_dir/panorama_pkls"
  tracks_dir="$work_dir/tracks"
  logs_dir="$work_dir/$logs_name"
  submit_block_dir="$submit_root/$block_name"
  script_dir="$submit_block_dir/scripts"
  submit_logs_dir="$submit_block_dir/logs"
  chunk_logs_dir="$logs_dir/chunk_tracking"
  mkdir -p "$script_dir" "$submit_logs_dir"

  map_script="$script_dir/map_${block_name}.sbatch"
  track_submit_script="$script_dir/submit_tracking_${block_name}.sbatch"
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
source ~/.bashrc
conda activate "${conda_env}"

cd "${REPO_ROOT}"
mkdir -p "${panorama_dir}" "${tracks_dir}" "${logs_dir}"
python "${PIPELINE_PY}" \\
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
source ~/.bashrc
conda activate "${conda_env}"

cd "${REPO_ROOT}"
mkdir -p "${tracks_dir}" "${chunk_logs_dir}"
python "${COMBINE_BATCH_PY}" \\
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
  --max_distance "${max_distance}" \\
  --lost_track_max_frames "${lost_track_max_frames}"${extra_tracking_text}${skip_existing_arg}
EOF
  chmod 755 "$track_submit_script"

  if [[ "$dry_run" -eq 1 ]]; then
    echo "[dry-run] map script: $map_script"
    echo "[dry-run] tracking submitter script: $track_submit_script"
    submitted=$((submitted + 1))
    continue
  fi

  map_job_id="$("$sbatch_bin" --parsable "$map_script")"
  track_submit_job_id="$("$sbatch_bin" --parsable --dependency=afterok:"$map_job_id" "$track_submit_script")"
  echo "Submitted $block_name: map job $map_job_id, tracking fan-out job $track_submit_job_id"
  submitted=$((submitted + 1))
done

if [[ "$submitted" -eq 0 ]]; then
  echo "ERROR: no block jobs were submitted." >&2
  exit 2
fi
