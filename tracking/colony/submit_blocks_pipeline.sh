#!/usr/bin/env bash
# Submit colony tracking for every block directory under a root.
#
# For each block:
#   1. submit one SLURM job per chunk to create ArUco/SLEAP panorama PKLs
#   2. submit a dependent SLURM job that fans out per-chunk/side tracking jobs
#   3. stitch per-block and continuous per-ant tracks
#   4. run colony sleep/behavior analysis as per-ant Slurm fan-out jobs
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
MAP_COMBINE_PY="$SCRIPT_DIR/map_combine.py"
COMBINE_BATCH_PY="$SCRIPT_DIR/combine_batch.py"
STITCH_TRACKS_PY="$REPO_ROOT/tracking/stitch_tracks.py"
SLEEP_BEHAVIOR_PY="$REPO_ROOT/analysis/colony_sleep_behavior_batch.py"
RUN_USER="${USER:-${LOGNAME:-$(id -un)}}"

# ----------------------------- CONFIGURATION -----------------------------
# Main input paths. These are the only paths you usually need to edit.
BLOCKS_ROOT="/bucket/ReiterU/Ants/basler/20260515"
HMATS="/bucket/ReiterU/Ants/basler/cameraArray_calib/20260414_calibration_dataset/set0_patterns_elevated_by_2mm/frame0/initial_H_mats.npz"

# Compute nodes cannot write to /bucket. All generated outputs go here.
OUTPUT_ROOT="/flash/ReiterU/ant_tmp/${RUN_USER}/colony_pipeline/20260515"

# Sbatch scripts and all SLURM logs go here.
SUBMIT_ROOT="${OUTPUT_ROOT}/jobs"

# Which block folders under BLOCKS_ROOT to submit.
BLOCK_GLOB="block*"

# Output layout under OUTPUT_ROOT. Leave WORK_NAME empty for a flat block layout:
#   OUTPUT_ROOT/block03/panorama_pkls
#   OUTPUT_ROOT/block03/tracks
WORK_NAME=""

# Cluster/runtime setup.
CONDA_ENV="aruco_env"
CONDA_BIN="/bucket/ReiterU/sam/miniforge3/bin/conda"
PYTHON_BIN="/bucket/ReiterU/sam/miniforge3/envs/aruco_env/bin/python"
PARTITION="compute"
SBATCH_BIN="sbatch"
SQUEUE_BIN="squeue"
SACCT_BIN="sacct"
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
MAP_MIN_INSTANCE_FRAME_FRAC="0.25"

# Small dependent job that submits the per-chunk tracking jobs after mapping.
TRACK_SUBMIT_CPUS=1
TRACK_SUBMIT_MEM="4G"
TRACK_SUBMIT_TIME="0-01:00:00"

# Per-chunk tracking job resources.
TRACK_CPUS=32
TRACK_MEM="32G"
TRACK_TIME="0-24:00:00"

# Tracking parameters.
MAX_DISTANCE="100.0"
LOST_TRACK_MAX_FRAMES="120"
LOST_TRACK_MAX_DISTANCE="100.0"
LOST_TRACK_ARUCO_MAX_DISTANCE=""

# Per-block stitch job resources.
STITCH_CPUS=8
STITCH_MEM="32G"
STITCH_TIME="0-08:00:00"

# Final continuous stitch job resources.
FINAL_STITCH_CPUS=8
FINAL_STITCH_MEM="32G"
FINAL_STITCH_TIME="0-08:00:00"

# Stitching parameters.
FPS="24.0"
WRITE_TRACK_PNGS=1

# Sleep/behavior analysis runs after continuous stitching. Expensive per-ant
# stages are submitted as one Slurm worker job per ant, mirroring tracking.
RUN_SLEEP_ANALYSIS=1
SLEEP_ANALYSIS_NAME="colony_sleep_behavior"
SLEEP_SIDE_FILTER=""
SLEEP_MIN_TRACK_PRESENT_FRAC="0.40"
SLEEP_MM_PER_PX="0.016"
SLEEP_STATIONARY_THRESHOLD_MM_S="0.1"
SLEEP_MIN_SLEEP_STATIONARY_SECONDS="10.0"
SLEEP_MAX_REASONABLE_SPEED_MM_S="5.0"
SLEEP_COLONY_BOXES_MM="-86,-32,-63,-8;93,149,-63,-8"
SLEEP_WORKER_INSIDE_COLONY_FRAC_THRESHOLD="0.95"
SLEEP_SUBMIT_CPUS=2
SLEEP_SUBMIT_MEM="8G"
SLEEP_SUBMIT_TIME="0-02:00:00"
SLEEP_ANT_CPUS=4
SLEEP_ANT_MEM="16G"
SLEEP_ANT_TIME="0-12:00:00"
SLEEP_AGG_CPUS=8
SLEEP_AGG_MEM="32G"
SLEEP_AGG_TIME="0-08:00:00"
SLEEP_FORCE_RECOMPUTE=0

# Login-side transfer back to /bucket after all SLURM work completes.
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
conda_env="$CONDA_ENV"
conda_bin="$CONDA_BIN"
python_bin="$PYTHON_BIN"
partition="$PARTITION"
sbatch_bin="$SBATCH_BIN"
squeue_bin="$SQUEUE_BIN"
sacct_bin="$SACCT_BIN"
slurm_setup="$SLURM_SETUP"
map_mode="$MAP_MODE"
side="$SIDE"
skip_existing="$SKIP_EXISTING"
map_cpus="$MAP_CPUS"
map_mem="$MAP_MEM"
map_time="$MAP_TIME"
map_min_instance_frame_frac="$MAP_MIN_INSTANCE_FRAME_FRAC"
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
final_stitch_cpus="$FINAL_STITCH_CPUS"
final_stitch_mem="$FINAL_STITCH_MEM"
final_stitch_time="$FINAL_STITCH_TIME"
fps="$FPS"
write_track_pngs="$WRITE_TRACK_PNGS"
run_sleep_analysis="$RUN_SLEEP_ANALYSIS"
sleep_analysis_name="$SLEEP_ANALYSIS_NAME"
sleep_side_filter="$SLEEP_SIDE_FILTER"
sleep_min_track_present_frac="$SLEEP_MIN_TRACK_PRESENT_FRAC"
sleep_mm_per_px="$SLEEP_MM_PER_PX"
sleep_stationary_threshold_mm_s="$SLEEP_STATIONARY_THRESHOLD_MM_S"
sleep_min_sleep_stationary_seconds="$SLEEP_MIN_SLEEP_STATIONARY_SECONDS"
sleep_max_reasonable_speed_mm_s="$SLEEP_MAX_REASONABLE_SPEED_MM_S"
sleep_colony_boxes_mm="$SLEEP_COLONY_BOXES_MM"
sleep_worker_inside_colony_frac_threshold="$SLEEP_WORKER_INSIDE_COLONY_FRAC_THRESHOLD"
sleep_submit_cpus="$SLEEP_SUBMIT_CPUS"
sleep_submit_mem="$SLEEP_SUBMIT_MEM"
sleep_submit_time="$SLEEP_SUBMIT_TIME"
sleep_ant_cpus="$SLEEP_ANT_CPUS"
sleep_ant_mem="$SLEEP_ANT_MEM"
sleep_ant_time="$SLEEP_ANT_TIME"
sleep_agg_cpus="$SLEEP_AGG_CPUS"
sleep_agg_mem="$SLEEP_AGG_MEM"
sleep_agg_time="$SLEEP_AGG_TIME"
sleep_force_recompute="$SLEEP_FORCE_RECOMPUTE"
transfer_to_bucket="$TRANSFER_TO_BUCKET"
transfer_poll_seconds="$TRANSFER_POLL_SECONDS"
delete_flash_after_transfer="$DELETE_FLASH_AFTER_TRANSFER"
dry_run="$DRY_RUN"

usage() {
  cat <<EOF
Submit colony tracking for every block directory under a root.

For each block:
  1. submit one SLURM job per chunk to create ArUco/SLEAP panorama PKLs
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
  --work_name NAME          Optional work directory name under OUTPUT_ROOT/<block>. Default: flat block dir
  --conda_env NAME          Conda environment name kept for fallback. Default: aruco_env
  --conda_bin PATH          Conda executable used inside sbatch jobs.
  --python_bin PATH         Python executable used inside sbatch jobs.
  --partition NAME          SLURM partition. Default: compute
  --sbatch_bin PATH         sbatch command/path. Default: sbatch
  --squeue_bin PATH         squeue command/path. Default: squeue
  --sacct_bin PATH          sacct command/path. Default: sacct
  --slurm_setup CMD         Command to run before sbatch, e.g. module load slurm.
  --map_mode MODE           aruco, sleap, or both. Default: both
  --side SIDE               left, right, or both. Default: both
  --skip_existing           Do not overwrite existing panorama PKLs, chunk parquets, or stitched outputs. Default: on
  --map_cpus N              CPUs for panorama jobs. Default: 8
  --map_mem MEM             Memory for panorama jobs. Default: 64G
  --map_time TIME           Time for panorama jobs. Default: 0-12:00:00
  --map_min_instance_frame_frac FLOAT
                            ArUco ID frame-fraction filter after all cameras in a chunk are merged.
  --track_cpus N            CPUs for each chunk tracking job. Default: 32
  --track_mem MEM           Memory for each chunk tracking job. Default: 32G
  --track_time TIME         Time for each chunk tracking job. Default: 0-24:00:00
  --track_submit_cpus N     CPUs for dependent fan-out submitter job. Default: 1
  --track_submit_mem MEM    Memory for dependent fan-out submitter job. Default: 4G
  --track_submit_time TIME  Time for dependent fan-out submitter job. Default: 0-01:00:00
  --max_distance FLOAT      Tracking max distance. Default: 100.0
  --lost_track_max_frames N Tracking lost-frame limit. Default: 120
  --lost_track_max_distance FLOAT
  --lost_track_aruco_max_distance FLOAT
  --stitch_cpus N          CPUs for each block stitch job. Default: 8
  --stitch_mem MEM         Memory for each block stitch job. Default: 32G
  --stitch_time TIME       Time for each block stitch job. Default: 0-08:00:00
  --final_stitch_cpus N    CPUs for final continuous stitch job. Default: 8
  --final_stitch_mem MEM   Memory for final continuous stitch job. Default: 32G
  --final_stitch_time TIME Time for final continuous stitch job. Default: 0-08:00:00
  --fps FLOAT              FPS used to convert chunk/batch offsets. Default: 24.0
  --write_track_pngs       Write trajectory PNGs during stitching. Default: on
  --run_sleep_analysis     Run post-stitch colony sleep/behavior analysis. Default: on
  --no_sleep_analysis      Skip post-stitch colony sleep/behavior analysis.
  --sleep_colony_boxes_mm BOXES
                            Semicolon-separated xmin,xmax,ymin,ymax boxes, or auto.
  --sleep_stationary_threshold_mm_s FLOAT|auto
                            Stationary speed threshold. Default: 0.1
  --sleep_force_recompute  Recompute sleep-analysis cache files even if present.
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
    --logs_name) echo "ERROR: --logs_name was removed; all SLURM logs now go under --submit_root/<block>/logs." >&2; exit 2 ;;
    --conda_env) conda_env="$2"; shift 2 ;;
    --conda_bin) conda_bin="$2"; shift 2 ;;
    --python_bin) python_bin="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --sbatch_bin) sbatch_bin="$2"; shift 2 ;;
    --squeue_bin) squeue_bin="$2"; shift 2 ;;
    --sacct_bin) sacct_bin="$2"; shift 2 ;;
    --slurm_setup) slurm_setup="$2"; shift 2 ;;
    --map_mode) map_mode="$2"; shift 2 ;;
    --side) side="$2"; shift 2 ;;
    --skip_existing) skip_existing=1; shift ;;
    --map_cpus) map_cpus="$2"; shift 2 ;;
    --map_mem) map_mem="$2"; shift 2 ;;
    --map_time) map_time="$2"; shift 2 ;;
    --map_min_instance_frame_frac) map_min_instance_frame_frac="$2"; shift 2 ;;
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
    --final_stitch_cpus) final_stitch_cpus="$2"; shift 2 ;;
    --final_stitch_mem) final_stitch_mem="$2"; shift 2 ;;
    --final_stitch_time) final_stitch_time="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --write_track_pngs) write_track_pngs=1; shift ;;
    --run_sleep_analysis) run_sleep_analysis=1; shift ;;
    --no_sleep_analysis) run_sleep_analysis=0; shift ;;
    --sleep_analysis_name) sleep_analysis_name="$2"; shift 2 ;;
    --sleep_side_filter) sleep_side_filter="$2"; shift 2 ;;
    --sleep_min_track_present_frac) sleep_min_track_present_frac="$2"; shift 2 ;;
    --sleep_mm_per_px) sleep_mm_per_px="$2"; shift 2 ;;
    --sleep_stationary_threshold_mm_s) sleep_stationary_threshold_mm_s="$2"; shift 2 ;;
    --sleep_min_sleep_stationary_seconds) sleep_min_sleep_stationary_seconds="$2"; shift 2 ;;
    --sleep_max_reasonable_speed_mm_s) sleep_max_reasonable_speed_mm_s="$2"; shift 2 ;;
    --sleep_colony_boxes_mm) sleep_colony_boxes_mm="$2"; shift 2 ;;
    --sleep_worker_inside_colony_frac_threshold) sleep_worker_inside_colony_frac_threshold="$2"; shift 2 ;;
    --sleep_submit_cpus) sleep_submit_cpus="$2"; shift 2 ;;
    --sleep_submit_mem) sleep_submit_mem="$2"; shift 2 ;;
    --sleep_submit_time) sleep_submit_time="$2"; shift 2 ;;
    --sleep_ant_cpus) sleep_ant_cpus="$2"; shift 2 ;;
    --sleep_ant_mem) sleep_ant_mem="$2"; shift 2 ;;
    --sleep_ant_time) sleep_ant_time="$2"; shift 2 ;;
    --sleep_agg_cpus) sleep_agg_cpus="$2"; shift 2 ;;
    --sleep_agg_mem) sleep_agg_mem="$2"; shift 2 ;;
    --sleep_agg_time) sleep_agg_time="$2"; shift 2 ;;
    --sleep_force_recompute) sleep_force_recompute=1; shift ;;
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
if [[ "$run_sleep_analysis" -eq 1 && ! -f "$SLEEP_BEHAVIOR_PY" ]]; then
  echo "ERROR: sleep/behavior analysis script does not exist: $SLEEP_BEHAVIOR_PY" >&2
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
if [[ "$dry_run" -eq 0 && "$transfer_to_bucket" -eq 1 ]]; then
  for transfer_cmd in rsync; do
    if ! command -v "$transfer_cmd" >/dev/null 2>&1; then
      echo "ERROR: transfer watcher requires command not found: $transfer_cmd" >&2
      exit 2
    fi
  done
fi

shopt -s nullglob
blocks=( "$blocks_root"/$block_glob )
if [[ ${#blocks[@]} -eq 0 ]]; then
  echo "ERROR: no block directories matched: $blocks_root/$block_glob" >&2
  exit 2
fi

submitted=0
track_submit_job_ids=()
submitted_block_names=()
submitted_work_dirs=()
block_stitch_id_files=()
continuous_manifest="$submit_root/continuous_block_stitch_id_files.txt"
transfer_manifest="$submit_root/transfer_to_bucket_manifest.tsv"
: > "$continuous_manifest"
: > "$transfer_manifest"
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
  bucket_panorama_dir="$block_dir/panorama_pkls"
  bucket_tracks_dir="$block_dir/tracks"
  bucket_stitched_dir="$block_dir/stitched"
  submit_block_dir="$submit_root/$block_name"
  script_dir="$submit_block_dir/scripts"
  submit_logs_dir="$submit_block_dir/logs"
  chunk_logs_dir="$submit_logs_dir"
  tracking_job_ids_file="$submit_block_dir/tracking_job_ids.txt"
  block_stitch_job_id_file="$submit_block_dir/block_stitch_job_id.txt"
  block_stitch_submit_done_file="$submit_block_dir/block_stitch_submit_complete.ok"
  mkdir -p "$script_dir" "$submit_logs_dir"
  printf '%s\t%s\n' "$panorama_dir" "$bucket_panorama_dir" >> "$transfer_manifest"
  printf '%s\t%s\n' "$tracks_dir" "$bucket_tracks_dir" >> "$transfer_manifest"
  printf '%s\t%s\n' "$stitched_dir" "$bucket_stitched_dir" >> "$transfer_manifest"

  echo "Validating ArUco/SLEAP H5 pairs and discovering chunks for $block_name..."
  mapfile -t chunks < <(PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" - "${data_dir}" <<'PY'
import sys
from pathlib import Path

from tracking.colony.panorama_io import discover_complete_input_chunks

data_dir = Path(sys.argv[1])
chunks, summary = discover_complete_input_chunks(data_dir)
if not chunks:
    print(
        f"No complete ArUco/SLEAP chunks can be processed in {data_dir}; summary={summary}",
        file=sys.stderr,
    )
    raise SystemExit(0)
print(
    "Complete ArUco/SLEAP chunks available: "
    f"{len(chunks)} contiguous chunks (chunk{chunks[0]}..chunk{chunks[-1]}), "
    f"{summary.get('reference_camera_count', 0)} cameras. "
    f"First incomplete: {summary.get('first_incomplete')}",
    file=sys.stderr,
)
for chunk in chunks:
    print(chunk)
PY
  )
  if [[ ${#chunks[@]} -eq 0 ]]; then
    echo "Skipping ${block_name}: no complete ArUco/SLEAP chunks found in ${data_dir}" >&2
    continue
  fi
  echo "${block_name}: ${#chunks[@]} contiguous complete chunks will be processed (chunk${chunks[0]}..chunk${chunks[-1]})."
  complete_chunks_literal=""
  for chunk in "${chunks[@]}"; do
    complete_chunks_literal+=" \"${chunk}\""
  done

  track_submit_script="$script_dir/submit_tracking_${block_name}.sbatch"
  stitch_script="$script_dir/stitch_${block_name}.sbatch"
  stitch_track_ids_file="$submit_block_dir/stitch_track_ids.txt"
  stitch_worker_dir="$script_dir/stitch_${block_name}_track_jobs"
  skip_existing_arg=""
  if [[ "$skip_existing" -eq 1 ]]; then
    skip_existing_arg=" \\
  --skip_existing"
  fi

  map_job_ids=()
  for chunk in "${chunks[@]}"; do
    map_script="$script_dir/map_${block_name}_chunk${chunk}.sbatch"

    cat > "$map_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J map_${block_name}_${chunk}
#SBATCH -p ${partition}
#SBATCH -c ${map_cpus}
#SBATCH --mem=${map_mem}
#SBATCH -t ${map_time}
#SBATCH -o ${submit_logs_dir}/map_${block_name}_chunk${chunk}_%j.out
#SBATCH -e ${submit_logs_dir}/map_${block_name}_chunk${chunk}_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
mkdir -p "${panorama_dir}" "${tracks_dir}"
"${python_bin}" "${MAP_COMBINE_PY}" \\
  --hmats "${hmats}" \\
  --data_dir "${data_dir}" \\
  --outdir "${panorama_dir}" \\
  --mode "${map_mode}" \\
  --chunk "${chunk}" \\
  --min_instance_frame_frac "${map_min_instance_frame_frac}"${skip_existing_arg}
EOF
    chmod 755 "$map_script"
  done

  extra_tracking_text=""
  if [[ -n "$lost_track_max_distance" ]]; then
    extra_tracking_text+=" \\
  --lost_track_max_distance \"${lost_track_max_distance}\""
  fi
  if [[ -n "$lost_track_aruco_max_distance" ]]; then
    extra_tracking_text+=" \\
  --lost_track_aruco_max_distance \"${lost_track_aruco_max_distance}\""
  fi
  stitch_no_pngs_arg="--no_track_pngs"
  if [[ "$write_track_pngs" -eq 1 ]]; then
    stitch_no_pngs_arg=""
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
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
mkdir -p "${tracks_dir}" "${chunk_logs_dir}"
rm -f "${tracking_job_ids_file}" "${block_stitch_job_id_file}" "${block_stitch_submit_done_file}" "${stitch_track_ids_file}"
complete_chunks=(${complete_chunks_literal})
combine_chunk_args=()
for chunk in "\${complete_chunks[@]}"; do
  combine_chunk_args+=( --chunk "\${chunk}" )
done
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
  --job_ids_file "${tracking_job_ids_file}" \\
  --max_distance "${max_distance}" \\
  --lost_track_max_frames "${lost_track_max_frames}" \\
  "\${combine_chunk_args[@]}"${extra_tracking_text}${skip_existing_arg}

if [[ ! -s "${tracking_job_ids_file}" ]]; then
  echo "ERROR: no tracking job IDs were written by combine_batch.py" >&2
  exit 2
fi
tracking_dependency="\$(paste -sd: "${tracking_job_ids_file}")"
stitch_job_id="\$("${sbatch_bin}" --parsable --dependency=afterok:"\${tracking_dependency}" "${stitch_script}")"
echo "Submitted ${block_name} stitch submitter job \${stitch_job_id} after tracking jobs \${tracking_dependency}"
EOF
  chmod 755 "$track_submit_script"

cat > "$stitch_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J submit_stitch_${block_name}
#SBATCH -p ${partition}
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 0-01:00:00
#SBATCH -o ${submit_logs_dir}/submit_stitch_${block_name}_%j.out
#SBATCH -e ${submit_logs_dir}/submit_stitch_${block_name}_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
mkdir -p "${stitched_dir}" "${stitch_worker_dir}"
rm -f "${block_stitch_job_id_file}" "${block_stitch_submit_done_file}" "${stitch_track_ids_file}"

"${python_bin}" - "${tracks_dir}" "${stitch_track_ids_file}" "${stitched_dir}/per_track" "${skip_existing}" <<'PY'
import sys
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

tracks_dir = Path(sys.argv[1])
out_file = Path(sys.argv[2])
per_track_dir = Path(sys.argv[3])
skip_existing = sys.argv[4] == "1"
chunk_token_re = re.compile(r"(?:^|_)(chunk\d+)(?:_|$)")


def safe_label(s: str) -> str:
    s = s.strip()
    if not s:
        return "NO_SUFFIX"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "NO_SUFFIX"


def group_suffix(fp: Path) -> str:
    stem = fp.stem
    if "_" in stem:
        _, suffix = stem.split("_", 1)
    else:
        suffix = ""
    if "chunk" in suffix:
        suffix = chunk_token_re.sub("_", suffix)
        suffix = re.sub(r"__+", "_", suffix).strip("_")
    return suffix


def output_exists(track_id: int, suffix: str) -> bool:
    out_path = per_track_dir / f"TrackID_{track_id:04d}_all_{safe_label(suffix)}.parquet"
    return out_path.exists()


track_groups = defaultdict(set)
for fp in sorted(tracks_dir.glob("*.parquet")):
    suffix = group_suffix(fp)
    cols = set(pq.ParquetFile(fp).schema_arrow.names)
    if "TrackID" not in cols:
        track_groups[0].add(suffix)
        continue
    s = pd.read_parquet(fp, columns=["TrackID"])["TrackID"]
    s = pd.to_numeric(s, errors="coerce").dropna()
    for x in s.unique():
        track_groups[int(x)].add(suffix)
track_ids = [
    tid
    for tid, suffixes in sorted(track_groups.items())
    if (not skip_existing) or any(not output_exists(tid, suffix) for suffix in suffixes)
]
out_file.write_text("\\n".join(str(x) for x in sorted(track_ids)) + ("\\n" if track_ids else ""))
if skip_existing and track_groups and not track_ids:
    print("All per-block stitched TrackID outputs already exist.", file=sys.stderr)
PY

if [[ ! -s "${stitch_track_ids_file}" ]]; then
  if [[ "${skip_existing}" -eq 1 ]]; then
    echo "No ${block_name} stitch work remains; all expected outputs already exist."
    : > "${block_stitch_job_id_file}"
    printf 'submitted %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" > "${block_stitch_submit_done_file}"
    exit 0
  fi
  echo "ERROR: no TrackIDs found in ${tracks_dir}" >&2
  exit 2
fi

while IFS= read -r tid; do
  [[ -n "\${tid}" ]] || continue
  worker="${stitch_worker_dir}/stitch_${block_name}_track\${tid}.sbatch"
  cat > "\${worker}" <<WORKER
#!/usr/bin/env bash
#SBATCH -J stitch_${block_name}_\${tid}
#SBATCH -p ${partition}
#SBATCH -c ${stitch_cpus}
#SBATCH --mem=${stitch_mem}
#SBATCH -t ${stitch_time}
#SBATCH -o ${submit_logs_dir}/stitch_${block_name}_track\${tid}_%j.out
#SBATCH -e ${submit_logs_dir}/stitch_${block_name}_track\${tid}_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
stitch_args=(
  --input_dir "${tracks_dir}"
  --out_dir "${stitched_dir}"
  --fps "${fps}"
  --track_id "\${tid}"
)
complete_chunks=(${complete_chunks_literal})
for chunk in "\\\${complete_chunks[@]}"; do
  stitch_args+=( --chunk "\\\${chunk}" )
done
if [[ -n "${stitch_no_pngs_arg}" ]]; then
  stitch_args+=( "${stitch_no_pngs_arg}" )
fi
if [[ "${skip_existing}" -eq 1 ]]; then
  stitch_args+=( --skip_existing )
fi
"${python_bin}" "${STITCH_TRACKS_PY}" "\\\${stitch_args[@]}"
WORKER
  chmod 755 "\${worker}"
  job_id="\$("${sbatch_bin}" --parsable "\${worker}")"
  echo "\${job_id}" >> "${block_stitch_job_id_file}"
  echo "Submitted ${block_name} TrackID \${tid} stitch job \${job_id}"
done < "${stitch_track_ids_file}"
printf 'submitted %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" > "${block_stitch_submit_done_file}"
EOF
  chmod 755 "$stitch_script"

  if [[ "$dry_run" -eq 1 ]]; then
    for chunk in "${chunks[@]}"; do
      echo "[dry-run] map script: $script_dir/map_${block_name}_chunk${chunk}.sbatch"
    done
    echo "[dry-run] tracking submitter script: $track_submit_script"
    echo "[dry-run] block stitch script: $stitch_script"
    printf '%s\t%s\t%s\n' "$block_stitch_job_id_file" "$stitched_dir" "$block_stitch_submit_done_file" >> "$continuous_manifest"
    block_stitch_id_files+=( "$block_stitch_job_id_file" )
    submitted_block_names+=( "$block_name" )
    submitted_work_dirs+=( "$work_dir" )
    submitted=$((submitted + 1))
    continue
  fi

  for chunk in "${chunks[@]}"; do
    map_script="$script_dir/map_${block_name}_chunk${chunk}.sbatch"
    map_job_id="$("$sbatch_bin" --parsable "$map_script")"
    map_job_ids+=( "$map_job_id" )
    echo "Submitted $block_name chunk $chunk: map job $map_job_id"
  done
  map_dependency="$(IFS=:; echo "${map_job_ids[*]}")"
  track_submit_job_id="$("$sbatch_bin" --parsable --dependency=afterok:"$map_dependency" "$track_submit_script")"
  echo "Submitted $block_name: tracking fan-out job $track_submit_job_id after map jobs $map_dependency"
  track_submit_job_ids+=( "$track_submit_job_id" )
  printf '%s\t%s\t%s\n' "$block_stitch_job_id_file" "$stitched_dir" "$block_stitch_submit_done_file" >> "$continuous_manifest"
  block_stitch_id_files+=( "$block_stitch_job_id_file" )
  submitted_block_names+=( "$block_name" )
  submitted_work_dirs+=( "$work_dir" )
  submitted=$((submitted + 1))
done

if [[ "$submitted" -eq 0 ]]; then
  echo "ERROR: no block jobs were submitted." >&2
  exit 2
fi

if [[ "$run_sleep_analysis" -eq 1 && ${#submitted_work_dirs[@]} -ne 1 ]]; then
  echo "ERROR: block-scoped sleep analysis requires exactly one submitted block." >&2
  echo "Submit blocks one at a time for per-block sleep outputs, or use --no_sleep_analysis for multi-block tracking/continuous stitching." >&2
  exit 2
fi

continuous_script="$submit_root/stitch_continuous_batches.sbatch"
continuous_submitter_script="$submit_root/submit_stitch_continuous_batches.sbatch"
continuous_submitter_job_id_file="$submit_root/continuous_stitch_submitter_job_id.txt"
continuous_job_id_file="$submit_root/continuous_stitch_job_id.txt"
continuous_done_file="$submit_root/continuous_stitch_complete.ok"
continuous_track_ids_file="$submit_root/continuous_stitch_track_ids.txt"
continuous_worker_dir="$submit_root/continuous_stitch_track_jobs"
continuous_input_dir="$output_root/continuous_input"
continuous_out_dir="$output_root/continuous_stitched"
bucket_continuous_out_dir="$blocks_root/continuous_stitched"
continuous_logs_dir="$submit_root/logs"
mkdir -p "$continuous_logs_dir"
printf '%s\t%s\n' "$continuous_out_dir" "$bucket_continuous_out_dir" >> "$transfer_manifest"

sleep_analysis_dir="$output_root/$sleep_analysis_name"
sleep_input_per_track_dir="$continuous_out_dir/per_track"
if [[ "$run_sleep_analysis" -eq 1 ]]; then
  sleep_analysis_dir="${submitted_work_dirs[0]}/stitched"
  sleep_input_per_track_dir="$sleep_analysis_dir/per_track"
fi
sleep_cache_dir="$sleep_analysis_dir/analysis_cache"
sleep_output_dir="$sleep_analysis_dir/outputs"
sleep_worklist="$sleep_cache_dir/tables/good_track_worklist.tsv"
sleep_done_file="$sleep_analysis_dir/sleep_behavior_complete.ok"
sleep_job_id_file="$submit_root/sleep_behavior_job_id.txt"
sleep_submitter_job_id_file="$submit_root/sleep_behavior_submitter_job_id.txt"
sleep_logs_dir="$submit_root/sleep_behavior_logs"
sleep_submitter_script="$submit_root/submit_sleep_behavior.sbatch"
sleep_threshold_script="$submit_root/sleep_behavior_threshold.sbatch"
sleep_colony_script="$submit_root/sleep_behavior_colony.sbatch"
sleep_aggregate_script="$submit_root/sleep_behavior_aggregate.sbatch"
sleep_speed_worker_dir="$submit_root/sleep_behavior_speed_jobs"
sleep_label_worker_dir="$submit_root/sleep_behavior_label_jobs"
sleep_outside_worker_dir="$submit_root/sleep_behavior_outside_jobs"
sleep_speed_job_ids_file="$submit_root/sleep_behavior_speed_job_ids.txt"
sleep_label_job_ids_file="$submit_root/sleep_behavior_label_job_ids.txt"
sleep_outside_job_ids_file="$submit_root/sleep_behavior_outside_job_ids.txt"
sleep_threshold_job_id_file="$submit_root/sleep_behavior_threshold_job_id.txt"
sleep_colony_job_id_file="$submit_root/sleep_behavior_colony_job_id.txt"
sleep_dependency_report_file="$submit_root/sleep_behavior_dependency_report.tsv"
if [[ "$run_sleep_analysis" -eq 1 ]]; then
  mkdir -p "$sleep_logs_dir"
fi

cat > "$continuous_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J stitch_continuous
#SBATCH -p ${partition}
#SBATCH -c ${final_stitch_cpus}
#SBATCH --mem=${final_stitch_mem}
#SBATCH -t ${final_stitch_time}
#SBATCH -o ${continuous_logs_dir}/stitch_continuous_%j.out
#SBATCH -e ${continuous_logs_dir}/stitch_continuous_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

log() {
  printf '[%s] %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" "\$*" >&2
}

submit_sbatch() {
  log "Submitting: \$*"
  local output status
  set +e
  if command -v timeout >/dev/null 2>&1; then
    output="\$(timeout 300 "\$@" 2>&1)"
    status=\$?
  else
    output="\$("\$@" 2>&1)"
    status=\$?
  fi
  set -e
  if [[ "\${status}" -ne 0 ]]; then
    log "ERROR: sbatch command failed with status \${status}"
    printf '%s\n' "\${output}" >&2
    exit "\${status}"
  fi
  log "sbatch output: \${output}"
  printf '%s\n' "\${output}" | tail -n 1
}

cd "${REPO_ROOT}"
log "Starting continuous stitch job on host \$(hostname); SLURM_JOB_ID=\${SLURM_JOB_ID:-none}"
log "PWD=\$(pwd)"
log "python_bin=${python_bin}"
log "stitch_script=${STITCH_TRACKS_PY}"
log "continuous_input=${continuous_input_dir}"
log "continuous_out=${continuous_out_dir}"
mkdir -p "${continuous_input_dir}" "${continuous_out_dir}"
find "${continuous_input_dir}" -maxdepth 1 \( -type l -o -type f \) -delete

log "Building continuous input symlinks from ${continuous_manifest}"
"${python_bin}" - "${continuous_manifest}" "${output_root}" "${continuous_input_dir}" <<'PY'
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

manifest = Path(sys.argv[1])
_output_root = Path(sys.argv[2])
continuous_input = Path(sys.argv[3])

side_re = re.compile(r"_(left|right)$")
track_re = re.compile(r"^(TrackID_\\d+)_")
columns = [
    "Frame",
    "TrackID",
    "Bodypoint",
    "X",
    "Y",
    "TrackX",
    "TrackY",
    "ArucoX",
    "ArucoY",
    "SleapAnchorX",
    "SleapAnchorY",
    "CameraID",
    "ArucoCam",
    "SleapCam",
    "source_file",
]
schema = pa.schema(
    [
        ("Frame", pa.int64()),
        ("TrackID", pa.int64()),
        ("Bodypoint", pa.int64()),
        ("X", pa.float64()),
        ("Y", pa.float64()),
        ("TrackX", pa.float64()),
        ("TrackY", pa.float64()),
        ("ArucoX", pa.float64()),
        ("ArucoY", pa.float64()),
        ("SleapAnchorX", pa.float64()),
        ("SleapAnchorY", pa.float64()),
        ("CameraID", pa.int64()),
        ("ArucoCam", pa.int64()),
        ("SleapCam", pa.int64()),
        ("source_file", pa.string()),
    ]
)


def num_frames(path: Path) -> int | None:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    raw = metadata.get(b"num_frames")
    if raw is None:
        return None
    try:
        value = int(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def write_empty_placeholder(path: Path, duration: int) -> None:
    table = pa.Table.from_pandas(pd.DataFrame(columns=columns), schema=schema, preserve_index=False)
    md = dict(table.schema.metadata or {})
    md[b"num_frames"] = str(int(duration)).encode("utf-8")
    table = table.replace_schema_metadata(md)
    pq.write_table(table, path)

blocks = []
wanted = set()
for line in manifest.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split("\t")
    id_file = Path(parts[0])
    stitched_dir = Path(parts[1]) if len(parts) > 1 else id_file.parent.parent / "stitched"
    block_name = id_file.parent.name
    per_track = stitched_dir / "per_track"
    if not per_track.is_dir():
        raise FileNotFoundError(f"Missing per-block stitched directory: {per_track}")
    existing = {}
    duration = None
    for src in sorted(per_track.glob("TrackID_*.parquet")):
        stem = src.stem
        track_match = track_re.match(stem)
        side_match = side_re.search(stem)
        if not track_match or not side_match:
            continue
        key = (track_match.group(1), side_match.group(1))
        existing[key] = src
        wanted.add(key)
        duration = duration or num_frames(src)
    if duration is None:
        raise RuntimeError(f"Could not infer block duration from {per_track}")
    blocks.append((block_name, existing, duration))

for block_name, existing, duration in blocks:
    for track_id, side in sorted(wanted):
        dst = continuous_input / f"{block_name}_{track_id}_{side}.parquet"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        src = existing.get((track_id, side))
        if src is None:
            write_empty_placeholder(dst, duration)
        else:
            dst.symlink_to(src)
PY

mkdir -p "${continuous_worker_dir}"
rm -f "${continuous_track_ids_file}" "${continuous_done_file}"
log "Continuous input count: \$(find "${continuous_input_dir}" -maxdepth 1 \( -type l -o -type f \) | wc -l)"
log "Building continuous stitch worklist: ${continuous_track_ids_file}"
"${python_bin}" - "${continuous_input_dir}" "${continuous_track_ids_file}" "${continuous_out_dir}/per_track" "${skip_existing}" <<'PY'
import sys
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

input_dir = Path(sys.argv[1])
out_file = Path(sys.argv[2])
per_track_dir = Path(sys.argv[3])
skip_existing = sys.argv[4] == "1"
chunk_token_re = re.compile(r"(?:^|_)(chunk\d+)(?:_|$)")


def safe_label(s: str) -> str:
    s = s.strip()
    if not s:
        return "NO_SUFFIX"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "NO_SUFFIX"


def group_suffix(fp: Path) -> str:
    stem = fp.stem
    if "_" in stem:
        _, suffix = stem.split("_", 1)
    else:
        suffix = ""
    if "chunk" in suffix:
        suffix = chunk_token_re.sub("_", suffix)
        suffix = re.sub(r"__+", "_", suffix).strip("_")
    return suffix


def output_exists(track_id: int, suffix: str) -> bool:
    out_path = per_track_dir / f"TrackID_{track_id:04d}_all_{safe_label(suffix)}.parquet"
    return out_path.exists()


track_groups = defaultdict(set)
for fp in sorted(input_dir.glob("*.parquet")):
    suffix = group_suffix(fp)
    cols = set(pq.ParquetFile(fp).schema_arrow.names)
    if "TrackID" not in cols:
        track_groups[0].add(suffix)
        continue
    s = pd.read_parquet(fp, columns=["TrackID"])["TrackID"]
    s = pd.to_numeric(s, errors="coerce").dropna()
    for x in s.unique():
        track_groups[int(x)].add(suffix)
track_ids = [
    tid
    for tid, suffixes in sorted(track_groups.items())
    if (not skip_existing) or any(not output_exists(tid, suffix) for suffix in suffixes)
]
out_file.write_text("\\n".join(str(x) for x in sorted(track_ids)) + ("\\n" if track_ids else ""))
if skip_existing and track_groups and not track_ids:
    print("All continuous stitched TrackID outputs already exist.", file=sys.stderr)
PY

if [[ ! -s "${continuous_track_ids_file}" ]]; then
  if [[ "${skip_existing}" -eq 1 ]]; then
    log "No continuous stitch work remains; all expected outputs already exist."
    printf 'completed %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" > "${continuous_done_file}"
    log "Wrote continuous done marker: ${continuous_done_file}"
    exit 0
  fi
  echo "ERROR: no TrackIDs found in ${continuous_input_dir}" >&2
  exit 2
fi
log "Continuous stitch task count: \$(wc -l < "${continuous_track_ids_file}" | tr -d '[:space:]')"

worker_job_ids=()
while IFS= read -r tid; do
  [[ -n "\${tid}" ]] || continue
  worker="${continuous_worker_dir}/stitch_continuous_track\${tid}.sbatch"
  cat > "\${worker}" <<WORKER
#!/usr/bin/env bash
#SBATCH -J stitch_cont_\${tid}
#SBATCH -p ${partition}
#SBATCH -c ${final_stitch_cpus}
#SBATCH --mem=${final_stitch_mem}
#SBATCH -t ${final_stitch_time}
#SBATCH -o ${continuous_logs_dir}/stitch_continuous_track\${tid}_%j.out
#SBATCH -e ${continuous_logs_dir}/stitch_continuous_track\${tid}_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
stitch_args=(
  --input_dir "${continuous_input_dir}"
  --out_dir "${continuous_out_dir}"
  --fps "${fps}"
  --track_id "\${tid}"
)
if [[ -n "${stitch_no_pngs_arg}" ]]; then
  stitch_args+=( "${stitch_no_pngs_arg}" )
fi
if [[ "${skip_existing}" -eq 1 ]]; then
  stitch_args+=( --skip_existing )
fi
"${python_bin}" "${STITCH_TRACKS_PY}" "\\\${stitch_args[@]}"
WORKER
  chmod 755 "\${worker}"
  job_id="\$(submit_sbatch "${sbatch_bin}" --parsable "\${worker}")"
  worker_job_ids+=( "\${job_id}" )
  echo "Submitted continuous TrackID \${tid} stitch job \${job_id}"
done < "${continuous_track_ids_file}"

worker_dependency="\$(IFS=:; echo "\${worker_job_ids[*]}")"
marker="${continuous_worker_dir}/continuous_stitch_done.sbatch"
cat > "\${marker}" <<MARKER
#!/usr/bin/env bash
#SBATCH -J stitch_cont_done
#SBATCH -p ${partition}
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 0-00:10:00
#SBATCH -o ${continuous_logs_dir}/stitch_continuous_done_%j.out
#SBATCH -e ${continuous_logs_dir}/stitch_continuous_done_%j.err

set -euo pipefail
printf 'completed %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" > "${continuous_done_file}"
MARKER
chmod 755 "\${marker}"
marker_job_id="\$(submit_sbatch "${sbatch_bin}" --parsable --dependency=afterok:"\${worker_dependency}" "\${marker}")"
echo "Submitted continuous stitch completion marker \${marker_job_id} after jobs \${worker_dependency}"
EOF
chmod 755 "$continuous_script"

cat > "$continuous_submitter_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J submit_stitch_cont
#SBATCH -p ${partition}
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 0-01:00:00
#SBATCH -o ${continuous_logs_dir}/submit_stitch_continuous_%j.out
#SBATCH -e ${continuous_logs_dir}/submit_stitch_continuous_%j.err

set -euo pipefail

log() {
  printf '[%s] %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" "\$*" >&2
}

submit_sbatch() {
  log "Submitting: \$*"
  local output status
  set +e
  if command -v timeout >/dev/null 2>&1; then
    output="\$(timeout 300 "\$@" 2>&1)"
    status=\$?
  else
    output="\$("\$@" 2>&1)"
    status=\$?
  fi
  set -e
  if [[ "\${status}" -ne 0 ]]; then
    log "ERROR: sbatch command failed with status \${status}"
    printf '%s\n' "\${output}" >&2
    exit "\${status}"
  fi
  log "sbatch output: \${output}"
  printf '%s\n' "\${output}" | tail -n 1
}

log "Starting continuous submitter on host \$(hostname); SLURM_JOB_ID=\${SLURM_JOB_ID:-none}"
log "PATH=\${PATH}"
log "sbatch path=\$(command -v "${sbatch_bin}" || true)"
log "manifest=${continuous_manifest}"

if [[ ! -s "${continuous_manifest}" ]]; then
  echo "ERROR: missing continuous stitch manifest: ${continuous_manifest}" >&2
  exit 2
fi
log "manifest line count: \$(wc -l < "${continuous_manifest}" | tr -d '[:space:]')"

while IFS=\$'\t' read -r id_file _stitched_dir submit_done_file _rest; do
  [[ -n "\${id_file}" ]] || continue
  if [[ -n "\${submit_done_file:-}" ]]; then
    log "Waiting for block stitch submit marker: \${submit_done_file}"
    while [[ ! -s "\${submit_done_file}" ]]; do
      sleep 60
    done
    log "Found block stitch submit marker: \$(ls -l "\${submit_done_file}")"
  fi
  log "Waiting for block stitch job id file: \${id_file}"
  while [[ ! -e "\${id_file}" ]]; do
    sleep 60
  done
  log "Found block stitch job id file: \$(ls -l "\${id_file}")"
done < "${continuous_manifest}"

rm -f "${continuous_job_id_file}" "${continuous_done_file}" "${continuous_track_ids_file}"
stitch_dependency="\$(
  while IFS=\$'\t' read -r id_file _rest; do
    if [[ -n "\${id_file}" && -s "\${id_file}" ]]; then
      cat "\${id_file}"
    fi
  done < "${continuous_manifest}" | awk 'NF' | paste -sd:
)"
log "stitch_dependency='\${stitch_dependency}'"
if [[ -n "\${stitch_dependency}" ]]; then
  continuous_job_id="\$(submit_sbatch "${sbatch_bin}" --parsable --dependency=afterok:"\${stitch_dependency}" "${continuous_script}")"
else
  log "No block stitch dependencies remain; submitting continuous stitch directly."
  continuous_job_id="\$(submit_sbatch "${sbatch_bin}" --parsable "${continuous_script}")"
fi
echo "\${continuous_job_id}" > "${continuous_job_id_file}"
log "Wrote continuous stitch job id file: ${continuous_job_id_file}"
echo "Submitted continuous stitch job \${continuous_job_id} after block stitch jobs \${stitch_dependency}"
EOF
chmod 755 "$continuous_submitter_script"

sleep_force_arg=""
if [[ "$sleep_force_recompute" -eq 1 ]]; then
  sleep_force_arg=" \\
  --force"
fi

if [[ "$run_sleep_analysis" -eq 1 ]]; then
cat > "$sleep_threshold_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J sleep_threshold
#SBATCH -p ${partition}
#SBATCH -c ${sleep_agg_cpus}
#SBATCH --mem=${sleep_agg_mem}
#SBATCH -t ${sleep_agg_time}
#SBATCH -o ${sleep_logs_dir}/sleep_threshold_%j.out
#SBATCH -e ${sleep_logs_dir}/sleep_threshold_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Starting sleep threshold on host \$(hostname); SLURM_JOB_ID=\${SLURM_JOB_ID:-none}" >&2
cd "${REPO_ROOT}"
"${python_bin}" "${SLEEP_BEHAVIOR_PY}" threshold \\
  --cache_dir "${sleep_cache_dir}" \\
  --worklist "${sleep_worklist}" \\
  --fps "${fps}" \\
  --mm_per_px "${sleep_mm_per_px}" \\
  --min_track_present_frac "${sleep_min_track_present_frac}" \\
  --max_reasonable_speed_mm_s "${sleep_max_reasonable_speed_mm_s}" \\
  --stationary_threshold_mm_s "${sleep_stationary_threshold_mm_s}" \\
  --min_sleep_stationary_seconds "${sleep_min_sleep_stationary_seconds}" \\
  --colony_boxes_mm="${sleep_colony_boxes_mm}" \\
  --worker_inside_colony_frac_threshold "${sleep_worker_inside_colony_frac_threshold}"${sleep_force_arg}
EOF
chmod 755 "$sleep_threshold_script"

cat > "$sleep_colony_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J sleep_colony
#SBATCH -p ${partition}
#SBATCH -c ${sleep_agg_cpus}
#SBATCH --mem=${sleep_agg_mem}
#SBATCH -t ${sleep_agg_time}
#SBATCH -o ${sleep_logs_dir}/sleep_colony_%j.out
#SBATCH -e ${sleep_logs_dir}/sleep_colony_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Starting sleep colony on host \$(hostname); SLURM_JOB_ID=\${SLURM_JOB_ID:-none}" >&2
cd "${REPO_ROOT}"
"${python_bin}" "${SLEEP_BEHAVIOR_PY}" colony \\
  --cache_dir "${sleep_cache_dir}" \\
  --worklist "${sleep_worklist}" \\
  --fps "${fps}" \\
  --mm_per_px "${sleep_mm_per_px}" \\
  --min_track_present_frac "${sleep_min_track_present_frac}" \\
  --max_reasonable_speed_mm_s "${sleep_max_reasonable_speed_mm_s}" \\
  --stationary_threshold_mm_s "${sleep_stationary_threshold_mm_s}" \\
  --min_sleep_stationary_seconds "${sleep_min_sleep_stationary_seconds}" \\
  --colony_boxes_mm="${sleep_colony_boxes_mm}" \\
  --worker_inside_colony_frac_threshold "${sleep_worker_inside_colony_frac_threshold}"${sleep_force_arg}
EOF
chmod 755 "$sleep_colony_script"

cat > "$sleep_aggregate_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J sleep_aggregate
#SBATCH -p ${partition}
#SBATCH -c ${sleep_agg_cpus}
#SBATCH --mem=${sleep_agg_mem}
#SBATCH -t ${sleep_agg_time}
#SBATCH -o ${sleep_logs_dir}/sleep_aggregate_%j.out
#SBATCH -e ${sleep_logs_dir}/sleep_aggregate_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Starting sleep aggregate on host \$(hostname); SLURM_JOB_ID=\${SLURM_JOB_ID:-none}" >&2
cd "${REPO_ROOT}"
"${python_bin}" "${SLEEP_BEHAVIOR_PY}" aggregate \\
  --cache_dir "${sleep_cache_dir}" \\
  --worklist "${sleep_worklist}" \\
  --output_dir "${sleep_output_dir}" \\
  --done_file "${sleep_done_file}" \\
  --fps "${fps}" \\
  --mm_per_px "${sleep_mm_per_px}" \\
  --min_track_present_frac "${sleep_min_track_present_frac}" \\
  --max_reasonable_speed_mm_s "${sleep_max_reasonable_speed_mm_s}" \\
  --stationary_threshold_mm_s "${sleep_stationary_threshold_mm_s}" \\
  --min_sleep_stationary_seconds "${sleep_min_sleep_stationary_seconds}" \\
  --colony_boxes_mm="${sleep_colony_boxes_mm}" \\
  --worker_inside_colony_frac_threshold "${sleep_worker_inside_colony_frac_threshold}"${sleep_force_arg}
EOF
chmod 755 "$sleep_aggregate_script"

cat > "$sleep_submitter_script" <<EOF
#!/usr/bin/env bash
#SBATCH -J submit_sleep_behavior
#SBATCH -p ${partition}
#SBATCH -c ${sleep_submit_cpus}
#SBATCH --mem=${sleep_submit_mem}
#SBATCH -t ${sleep_submit_time}
#SBATCH -o ${sleep_logs_dir}/submit_sleep_behavior_%j.out
#SBATCH -e ${sleep_logs_dir}/submit_sleep_behavior_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

log() {
  printf '[%s] %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" "\$*" >&2
}

submit_sbatch() {
  log "Submitting: \$*"
  local output status
  set +e
  if command -v timeout >/dev/null 2>&1; then
    output="\$(timeout 300 "\$@" 2>&1)"
    status=\$?
  else
    output="\$("\$@" 2>&1)"
    status=\$?
  fi
  set -e
  if [[ "\${status}" -ne 0 ]]; then
    log "ERROR: sbatch command failed with status \${status}"
    printf '%s\n' "\${output}" >&2
    exit "\${status}"
  fi
  log "sbatch output: \${output}"
  printf '%s\n' "\${output}" | tail -n 1
}

cd "${REPO_ROOT}"
log "Starting sleep behavior submitter on host \$(hostname); SLURM_JOB_ID=\${SLURM_JOB_ID:-none}"
log "PWD=\$(pwd)"
log "python_bin=${python_bin}"
log "sleep_script=${SLEEP_BEHAVIOR_PY}"
log "cache_dir=${sleep_cache_dir}"
log "output_dir=${sleep_output_dir}"
log "continuous_done_file=${continuous_done_file}"
log "continuous_job_id_file=${continuous_job_id_file}"
mkdir -p "${sleep_cache_dir}" "${sleep_output_dir}" "${sleep_logs_dir}" \\
  "${sleep_speed_worker_dir}" "${sleep_label_worker_dir}" "${sleep_outside_worker_dir}"
rm -f "${sleep_done_file}" "${sleep_job_id_file}" \\
  "${sleep_speed_job_ids_file}" "${sleep_label_job_ids_file}" "${sleep_outside_job_ids_file}" \\
  "${sleep_threshold_job_id_file}" "${sleep_colony_job_id_file}" "${sleep_dependency_report_file}"
find "${sleep_speed_worker_dir}" "${sleep_label_worker_dir}" "${sleep_outside_worker_dir}" \\
  -maxdepth 1 -type f -name '*.sbatch' -delete

log "Waiting for continuous stitch marker: ${continuous_done_file}"
waited_seconds=0
while [[ ! -s "${continuous_done_file}" ]]; do
  if (( waited_seconds % 300 == 0 )); then
    if [[ -e "${continuous_job_id_file}" ]]; then
      log "Still waiting; continuous job id file: \$(cat "${continuous_job_id_file}" 2>/dev/null || true)"
    else
      log "Still waiting; continuous job id file is missing: ${continuous_job_id_file}"
    fi
  fi
  sleep 60
  waited_seconds=\$((waited_seconds + 60))
done
log "Found continuous stitch marker: \$(ls -l "${continuous_done_file}")"
log "Sleep input per-track dir: ${sleep_input_per_track_dir}"
log "Sleep input per-track file count: \$(find "${sleep_input_per_track_dir}" -maxdepth 1 -type f -name 'TrackID_*.parquet' 2>/dev/null | wc -l)"

prepare_args=(
  prepare
  --per_track_dir "${sleep_input_per_track_dir}"
  --cache_dir "${sleep_cache_dir}"
  --worklist "${sleep_worklist}"
  --fps "${fps}"
  --mm_per_px "${sleep_mm_per_px}"
  --min_track_present_frac "${sleep_min_track_present_frac}"
  --max_reasonable_speed_mm_s "${sleep_max_reasonable_speed_mm_s}"
  --stationary_threshold_mm_s "${sleep_stationary_threshold_mm_s}"
  --min_sleep_stationary_seconds "${sleep_min_sleep_stationary_seconds}"
  --colony_boxes_mm="${sleep_colony_boxes_mm}"
  --worker_inside_colony_frac_threshold "${sleep_worker_inside_colony_frac_threshold}"
)
if [[ -n "${sleep_side_filter}" ]]; then
  prepare_args+=( --side_filter "${sleep_side_filter}" )
fi
if [[ "${sleep_force_recompute}" -eq 1 ]]; then
  prepare_args+=( --force )
fi
log "Running sleep prepare: ${python_bin} ${SLEEP_BEHAVIOR_PY} \${prepare_args[*]}"
"${python_bin}" "${SLEEP_BEHAVIOR_PY}" "\${prepare_args[@]}"

n_tasks="\$(wc -l < "${sleep_worklist}" | tr -d '[:space:]')"
if [[ -z "\${n_tasks}" || "\${n_tasks}" -le 0 ]]; then
  echo "ERROR: no sleep-analysis work items were written to ${sleep_worklist}" >&2
  exit 2
fi
log "Sleep worklist task count: \${n_tasks}"
log "Sleep worklist preview:"
sed -n '1,5p' "${sleep_worklist}" >&2 || true
upper="\$((n_tasks - 1))"

for task_id in \$(seq 0 "\${upper}"); do
  worker="${sleep_speed_worker_dir}/sleep_speed_task\${task_id}.sbatch"
  cat > "\${worker}" <<WORKER
#!/usr/bin/env bash
#SBATCH -J sleep_sp_\${task_id}
#SBATCH -p ${partition}
#SBATCH -c ${sleep_ant_cpus}
#SBATCH --mem=${sleep_ant_mem}
#SBATCH -t ${sleep_ant_time}
#SBATCH -o ${sleep_logs_dir}/sleep_speed_task\${task_id}_%j.out
#SBATCH -e ${sleep_logs_dir}/sleep_speed_task\${task_id}_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
echo "Starting sleep speed task \${task_id}" >&2
"${python_bin}" "${SLEEP_BEHAVIOR_PY}" speed \\
  --cache_dir "${sleep_cache_dir}" \\
  --worklist "${sleep_worklist}" \\
  --fps "${fps}" \\
  --mm_per_px "${sleep_mm_per_px}" \\
  --min_track_present_frac "${sleep_min_track_present_frac}" \\
  --max_reasonable_speed_mm_s "${sleep_max_reasonable_speed_mm_s}" \\
  --stationary_threshold_mm_s "${sleep_stationary_threshold_mm_s}" \\
  --min_sleep_stationary_seconds "${sleep_min_sleep_stationary_seconds}" \\
  --colony_boxes_mm="${sleep_colony_boxes_mm}" \\
  --worker_inside_colony_frac_threshold "${sleep_worker_inside_colony_frac_threshold}" \\
  --task_id "\${task_id}"${sleep_force_arg}
WORKER
  chmod 755 "\${worker}"
  job_id="\$(submit_sbatch "${sbatch_bin}" --parsable "\${worker}")"
  echo "\${job_id}" >> "${sleep_speed_job_ids_file}"
  echo "Submitted sleep speed task \${task_id} job \${job_id}"
done
speed_dependency="\$(paste -sd: "${sleep_speed_job_ids_file}")"
log "speed_dependency=\${speed_dependency}"
threshold_id="\$(submit_sbatch "${sbatch_bin}" --parsable --dependency=afterok:\${speed_dependency} "${sleep_threshold_script}")"
echo "\${threshold_id}" > "${sleep_threshold_job_id_file}"
echo "Submitted sleep threshold job \${threshold_id} after speed jobs \${speed_dependency}"

for task_id in \$(seq 0 "\${upper}"); do
  worker="${sleep_label_worker_dir}/sleep_label_task\${task_id}.sbatch"
  cat > "\${worker}" <<WORKER
#!/usr/bin/env bash
#SBATCH -J sleep_lb_\${task_id}
#SBATCH -p ${partition}
#SBATCH -c ${sleep_ant_cpus}
#SBATCH --mem=${sleep_ant_mem}
#SBATCH -t ${sleep_ant_time}
#SBATCH -o ${sleep_logs_dir}/sleep_label_task\${task_id}_%j.out
#SBATCH -e ${sleep_logs_dir}/sleep_label_task\${task_id}_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
echo "Starting sleep label task \${task_id}" >&2
"${python_bin}" "${SLEEP_BEHAVIOR_PY}" label \\
  --cache_dir "${sleep_cache_dir}" \\
  --worklist "${sleep_worklist}" \\
  --fps "${fps}" \\
  --mm_per_px "${sleep_mm_per_px}" \\
  --min_track_present_frac "${sleep_min_track_present_frac}" \\
  --max_reasonable_speed_mm_s "${sleep_max_reasonable_speed_mm_s}" \\
  --stationary_threshold_mm_s "${sleep_stationary_threshold_mm_s}" \\
  --min_sleep_stationary_seconds "${sleep_min_sleep_stationary_seconds}" \\
  --colony_boxes_mm="${sleep_colony_boxes_mm}" \\
  --worker_inside_colony_frac_threshold "${sleep_worker_inside_colony_frac_threshold}" \\
  --task_id "\${task_id}"${sleep_force_arg}
WORKER
  chmod 755 "\${worker}"
  job_id="\$(submit_sbatch "${sbatch_bin}" --parsable --dependency=afterok:\${threshold_id} "\${worker}")"
  echo "\${job_id}" >> "${sleep_label_job_ids_file}"
  echo "Submitted sleep label task \${task_id} job \${job_id} after threshold \${threshold_id}"
done
label_dependency="\$(paste -sd: "${sleep_label_job_ids_file}")"
log "label_dependency=\${label_dependency}"
colony_id="\$(submit_sbatch "${sbatch_bin}" --parsable --dependency=afterok:\${label_dependency} "${sleep_colony_script}")"
echo "\${colony_id}" > "${sleep_colony_job_id_file}"
echo "Submitted sleep colony job \${colony_id} after label jobs \${label_dependency}"

for task_id in \$(seq 0 "\${upper}"); do
  worker="${sleep_outside_worker_dir}/sleep_outside_task\${task_id}.sbatch"
  cat > "\${worker}" <<WORKER
#!/usr/bin/env bash
#SBATCH -J sleep_out_\${task_id}
#SBATCH -p ${partition}
#SBATCH -c ${sleep_ant_cpus}
#SBATCH --mem=${sleep_ant_mem}
#SBATCH -t ${sleep_ant_time}
#SBATCH -o ${sleep_logs_dir}/sleep_outside_task\${task_id}_%j.out
#SBATCH -e ${sleep_logs_dir}/sleep_outside_task\${task_id}_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
echo "Starting sleep outside task \${task_id}" >&2
"${python_bin}" "${SLEEP_BEHAVIOR_PY}" outside \\
  --cache_dir "${sleep_cache_dir}" \\
  --worklist "${sleep_worklist}" \\
  --fps "${fps}" \\
  --mm_per_px "${sleep_mm_per_px}" \\
  --min_track_present_frac "${sleep_min_track_present_frac}" \\
  --max_reasonable_speed_mm_s "${sleep_max_reasonable_speed_mm_s}" \\
  --stationary_threshold_mm_s "${sleep_stationary_threshold_mm_s}" \\
  --min_sleep_stationary_seconds "${sleep_min_sleep_stationary_seconds}" \\
  --colony_boxes_mm="${sleep_colony_boxes_mm}" \\
  --worker_inside_colony_frac_threshold "${sleep_worker_inside_colony_frac_threshold}" \\
  --task_id "\${task_id}"${sleep_force_arg}
WORKER
  chmod 755 "\${worker}"
  job_id="\$(submit_sbatch "${sbatch_bin}" --parsable --dependency=afterok:\${colony_id} "\${worker}")"
  echo "\${job_id}" >> "${sleep_outside_job_ids_file}"
  echo "Submitted sleep outside task \${task_id} job \${job_id} after colony \${colony_id}"
done
outside_dependency="\$(paste -sd: "${sleep_outside_job_ids_file}")"
log "outside_dependency=\${outside_dependency}"
aggregate_id="\$(submit_sbatch "${sbatch_bin}" --parsable --dependency=afterok:\${outside_dependency} "${sleep_aggregate_script}")"
echo "\${aggregate_id}" > "${sleep_job_id_file}"
log "Wrote sleep final job id file: ${sleep_job_id_file}"
if "${python_bin}" "${SLEEP_BEHAVIOR_PY}" diagnose \\
  --cache_dir "${sleep_cache_dir}" \\
  --worklist "${sleep_worklist}" \\
  --jobs_dir "${submit_root}" \\
  --logs_dir "${sleep_logs_dir}" \\
  --report_path "${sleep_dependency_report_file}" \\
  --stage all; then
  log "Wrote sleep dependency report: ${sleep_dependency_report_file}"
else
  log "WARNING: sleep dependency report command failed"
fi
log "To refresh dependency status later: ${python_bin} ${SLEEP_BEHAVIOR_PY} diagnose --cache_dir ${sleep_cache_dir} --worklist ${sleep_worklist} --jobs_dir ${submit_root} --logs_dir ${sleep_logs_dir} --report_path ${sleep_dependency_report_file} --stage all"
echo "Submitted sleep aggregate job \${aggregate_id} after outside jobs \${outside_dependency}"
EOF
chmod 755 "$sleep_submitter_script"
fi

postprocess_job_id_file="$continuous_job_id_file"
postprocess_done_file="$continuous_done_file"
postprocess_label="Continuous stitch"
if [[ "$run_sleep_analysis" -eq 1 ]]; then
  postprocess_job_id_file="$sleep_job_id_file"
  postprocess_done_file="$sleep_done_file"
  postprocess_label="Sleep behavior analysis"
fi

transfer_script="$submit_root/transfer_to_bucket_when_done.sh"
transfer_log="$submit_root/transfer_to_bucket.log"
cat > "$transfer_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

postprocess_job_id_file="${postprocess_job_id_file}"
postprocess_done_file="${postprocess_done_file}"
postprocess_label="${postprocess_label}"
transfer_manifest="${transfer_manifest}"
transfer_log="${transfer_log}"
poll_seconds="${transfer_poll_seconds}"
delete_flash_after_transfer="${delete_flash_after_transfer}"
lock_dir="${submit_root}/transfer_to_bucket.lock"

log() {
  printf '[%s] %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" "\$*" | tee -a "\${transfer_log}"
}

file_mtime() {
  local path="\$1"
  stat -c %Y "\${path}" 2>/dev/null || printf '0\n'
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
  if [[ -s "\${lock_dir}/pid" ]]; then
    old_pid="\$(cat "\${lock_dir}/pid" 2>/dev/null || true)"
  fi
  if [[ -s "\${lock_dir}/host" ]]; then
    old_host="\$(cat "\${lock_dir}/host" 2>/dev/null || true)"
  fi
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

wait_for_file() {
  local path="\$1"
  log "Waiting for \${path}"
  while [[ ! -s "\${path}" ]]; do
    sleep "\${poll_seconds}"
  done
}

wait_for_fresh_file() {
  local path="\$1"
  local min_mtime="\$2"
  local stale_logged=0
  log "Waiting for \${path} with mtime >= \${min_mtime}"
  while true; do
    if [[ -s "\${path}" ]]; then
      local mtime
      mtime="\$(file_mtime "\${path}")"
      if [[ "\${mtime}" =~ ^[0-9]+$ ]] && (( mtime >= min_mtime )); then
        log "Fresh completion marker found: \${path} mtime=\${mtime}"
        return 0
      fi
      if [[ "\${stale_logged}" -eq 0 ]]; then
        log "Ignoring stale completion marker: \${path} mtime=\${mtime}, job_id_file_mtime=\${min_mtime}"
        stale_logged=1
      fi
    fi
    sleep "\${poll_seconds}"
  done
}

log "Transfer watcher start: label=\${postprocess_label}"
log "job_id_file=\${postprocess_job_id_file}"
log "done_file=\${postprocess_done_file}"
log "manifest=\${transfer_manifest}"
if [[ -s "\${transfer_manifest}" ]]; then
  while IFS=\$'\t' read -r src dst; do
    [[ -n "\${src}" && -n "\${dst}" ]] || continue
    log "manifest_entry \${src} -> \${dst}"
  done < "\${transfer_manifest}"
fi

wait_for_file "\${postprocess_job_id_file}"
postprocess_job_id_file_mtime="\$(file_mtime "\${postprocess_job_id_file}")"
acquire_lock
postprocess_job_id="\$(head -n 1 "\${postprocess_job_id_file}")"
log "\${postprocess_label} final job is \${postprocess_job_id}; job_id_file_mtime=\${postprocess_job_id_file_mtime}"
wait_for_fresh_file "\${postprocess_done_file}" "\${postprocess_job_id_file_mtime}"
log "\${postprocess_label} success marker found; starting transfer to bucket."

if [[ ! -s "\${transfer_manifest}" ]]; then
  log "ERROR: missing transfer manifest: \${transfer_manifest}"
  exit 2
fi

while IFS=\$'\t' read -r src dst; do
  [[ -n "\${src}" && -n "\${dst}" ]] || continue
  if [[ ! -d "\${src}" ]]; then
    log "Skipping missing source directory: \${src}"
    continue
  fi
  mkdir -p "\${dst}"
  log "rsync \${src}/ -> \${dst}/"
  set +e
  rsync -a --partial --protect-args "\${src}/" "\${dst}/" >> "\${transfer_log}" 2>&1
  status="\$?"
  set -e
  if [[ "\${status}" -ne 0 ]]; then
    log "ERROR: rsync failed with status \${status}: \${src}/ -> \${dst}/"
    exit "\${status}"
  fi
  log "rsync complete: \${src}/ -> \${dst}/"
  if [[ "\${delete_flash_after_transfer}" -eq 1 ]]; then
    log "Deleting transferred source contents: \${src}"
    find "\${src}" -mindepth 1 -delete
  fi
done < "\${transfer_manifest}"

log "Transfer to bucket complete."
EOF
chmod 755 "$transfer_script"

if [[ "$dry_run" -eq 1 ]]; then
  echo "[dry-run] continuous stitch submitter script: $continuous_submitter_script"
  echo "[dry-run] continuous stitch script: $continuous_script"
  if [[ "$run_sleep_analysis" -eq 1 ]]; then
    echo "[dry-run] sleep behavior submitter script: $sleep_submitter_script"
    echo "[dry-run] sleep behavior speed worker dir: $sleep_speed_worker_dir"
    echo "[dry-run] sleep behavior threshold script: $sleep_threshold_script"
    echo "[dry-run] sleep behavior label worker dir: $sleep_label_worker_dir"
    echo "[dry-run] sleep behavior colony script: $sleep_colony_script"
    echo "[dry-run] sleep behavior outside worker dir: $sleep_outside_worker_dir"
    echo "[dry-run] sleep behavior aggregate script: $sleep_aggregate_script"
  fi
  echo "[dry-run] transfer watcher script: $transfer_script"
elif [[ ${#track_submit_job_ids[@]} -gt 0 ]]; then
  submit_dependency="$(IFS=:; echo "${track_submit_job_ids[*]}")"
  rm -f "$continuous_submitter_job_id_file" "$sleep_submitter_job_id_file"
  continuous_submitter_job_id="$("$sbatch_bin" --parsable --dependency=afterok:"$submit_dependency" "$continuous_submitter_script")"
  echo "$continuous_submitter_job_id" > "$continuous_submitter_job_id_file"
  echo "Submitted continuous stitch submitter job $continuous_submitter_job_id after tracking fan-out jobs $submit_dependency"
  if [[ "$run_sleep_analysis" -eq 1 ]]; then
    sleep_submitter_job_id="$("$sbatch_bin" --parsable --dependency=afterok:"$continuous_submitter_job_id" "$sleep_submitter_script")"
    echo "$sleep_submitter_job_id" > "$sleep_submitter_job_id_file"
    echo "Submitted sleep behavior submitter job $sleep_submitter_job_id after continuous submitter job $continuous_submitter_job_id"
  fi
  if [[ "$transfer_to_bucket" -eq 1 ]]; then
    nohup "$transfer_script" >/dev/null 2>&1 &
    transfer_pid="$!"
    echo "Started login-side transfer watcher PID $transfer_pid; log: $transfer_log"
  fi
fi
