#!/usr/bin/env bash
# Submit colony tracking for every block directory under a root.
#
# For each block:
#   1. submit one SLURM job per chunk to create ArUco/SLEAP panorama PKLs
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
MAP_COMBINE_PY="$SCRIPT_DIR/map_combine.py"
COMBINE_BATCH_PY="$SCRIPT_DIR/combine_batch.py"
STITCH_TRACKS_PY="$REPO_ROOT/tracking/stitch_tracks.py"
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
MAP_MIN_INSTANCE_FRAME_FRAC="0.05"

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
  --stitch_cpus N          CPUs for each block stitch job. Default: 8
  --stitch_mem MEM         Memory for each block stitch job. Default: 32G
  --stitch_time TIME       Time for each block stitch job. Default: 0-08:00:00
  --final_stitch_cpus N    CPUs for final continuous stitch job. Default: 8
  --final_stitch_mem MEM   Memory for final continuous stitch job. Default: 32G
  --final_stitch_time TIME Time for final continuous stitch job. Default: 0-08:00:00
  --fps FLOAT              FPS used to convert chunk/batch offsets. Default: 24.0
  --write_track_pngs       Write trajectory PNGs during stitching. Default: on
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
  mkdir -p "$script_dir" "$submit_logs_dir"
  printf '%s\t%s\n' "$panorama_dir" "$bucket_panorama_dir" >> "$transfer_manifest"
  printf '%s\t%s\n' "$tracks_dir" "$bucket_tracks_dir" >> "$transfer_manifest"
  printf '%s\t%s\n' "$stitched_dir" "$bucket_stitched_dir" >> "$transfer_manifest"

  echo "Validating ArUco/SLEAP H5 pairs and discovering chunks for $block_name..."
  mapfile -t chunks < <(PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" - "${data_dir}" <<'PY'
import sys
from pathlib import Path

from tracking.colony.panorama_io import ARUCO_INPUT_RE, validate_aruco_inputs_have_sleap_h5

data_dir = Path(sys.argv[1])
aruco_files = validate_aruco_inputs_have_sleap_h5(data_dir)
chunks = sorted({ARUCO_INPUT_RE.match(path.name).group("chunk") for path in aruco_files})
for chunk in chunks:
    print(chunk)
PY
  )
  if [[ ${#chunks[@]} -eq 0 ]]; then
    echo "Skipping ${block_name}: no ArUco chunk files found in ${data_dir}" >&2
    continue
  fi

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
rm -f "${tracking_job_ids_file}" "${block_stitch_job_id_file}" "${stitch_track_ids_file}"
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
  --lost_track_max_frames "${lost_track_max_frames}"${extra_tracking_text}${skip_existing_arg}

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
rm -f "${block_stitch_job_id_file}" "${stitch_track_ids_file}"

"${python_bin}" - "${tracks_dir}" "${stitch_track_ids_file}" <<'PY'
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

tracks_dir = Path(sys.argv[1])
out_file = Path(sys.argv[2])
track_ids = set()
for fp in sorted(tracks_dir.glob("*.parquet")):
    cols = set(pq.ParquetFile(fp).schema_arrow.names)
    if "TrackID" not in cols:
        track_ids.add(0)
        continue
    s = pd.read_parquet(fp, columns=["TrackID"])["TrackID"]
    s = pd.to_numeric(s, errors="coerce").dropna()
    track_ids.update(int(x) for x in s.unique())
out_file.write_text("\\n".join(str(x) for x in sorted(track_ids)) + ("\\n" if track_ids else ""))
PY

if [[ ! -s "${stitch_track_ids_file}" ]]; then
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
EOF
  chmod 755 "$stitch_script"

  if [[ "$dry_run" -eq 1 ]]; then
    for chunk in "${chunks[@]}"; do
      echo "[dry-run] map script: $script_dir/map_${block_name}_chunk${chunk}.sbatch"
    done
    echo "[dry-run] tracking submitter script: $track_submit_script"
    echo "[dry-run] block stitch script: $stitch_script"
    printf '%s\t%s\n' "$block_stitch_job_id_file" "$stitched_dir" >> "$continuous_manifest"
    block_stitch_id_files+=( "$block_stitch_job_id_file" )
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
  printf '%s\t%s\n' "$block_stitch_job_id_file" "$stitched_dir" >> "$continuous_manifest"
  block_stitch_id_files+=( "$block_stitch_job_id_file" )
  submitted=$((submitted + 1))
done

if [[ "$submitted" -eq 0 ]]; then
  echo "ERROR: no block jobs were submitted." >&2
  exit 2
fi

continuous_script="$submit_root/stitch_continuous_batches.sbatch"
continuous_submitter_script="$submit_root/submit_stitch_continuous_batches.sbatch"
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

cd "${REPO_ROOT}"
mkdir -p "${continuous_input_dir}" "${continuous_out_dir}"
find "${continuous_input_dir}" -maxdepth 1 \( -type l -o -type f \) -delete

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
columns = ["Frame", "TrackID", "Bodypoint", "X", "Y", "source_file"]
schema = pa.schema(
    [
        ("Frame", pa.int64()),
        ("TrackID", pa.int64()),
        ("Bodypoint", pa.int64()),
        ("X", pa.float64()),
        ("Y", pa.float64()),
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
"${python_bin}" - "${continuous_input_dir}" "${continuous_track_ids_file}" <<'PY'
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

input_dir = Path(sys.argv[1])
out_file = Path(sys.argv[2])
track_ids = set()
for fp in sorted(input_dir.glob("*.parquet")):
    cols = set(pq.ParquetFile(fp).schema_arrow.names)
    if "TrackID" not in cols:
        track_ids.add(0)
        continue
    s = pd.read_parquet(fp, columns=["TrackID"])["TrackID"]
    s = pd.to_numeric(s, errors="coerce").dropna()
    track_ids.update(int(x) for x in s.unique())
out_file.write_text("\\n".join(str(x) for x in sorted(track_ids)) + ("\\n" if track_ids else ""))
PY

if [[ ! -s "${continuous_track_ids_file}" ]]; then
  echo "ERROR: no TrackIDs found in ${continuous_input_dir}" >&2
  exit 2
fi

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
  job_id="\$("${sbatch_bin}" --parsable "\${worker}")"
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
marker_job_id="\$("${sbatch_bin}" --parsable --dependency=afterok:"\${worker_dependency}" "\${marker}")"
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

if [[ ! -s "${continuous_manifest}" ]]; then
  echo "ERROR: missing continuous stitch manifest: ${continuous_manifest}" >&2
  exit 2
fi

while IFS= read -r id_file; do
  [[ -n "\${id_file}" ]] || continue
  id_file="\${id_file%%\$'\t'*}"
  echo "Waiting for block stitch job id file: \${id_file}"
  while [[ ! -s "\${id_file}" ]]; do
    sleep 60
  done
done < "${continuous_manifest}"

rm -f "${continuous_job_id_file}" "${continuous_done_file}" "${continuous_track_ids_file}"
stitch_dependency="\$(while IFS= read -r id_file; do [[ -n "\${id_file}" ]] && id_file="\${id_file%%\$'\t'*}" && cat "\${id_file}"; done < "${continuous_manifest}" | paste -sd:)"
continuous_job_id="\$("${sbatch_bin}" --parsable --dependency=afterok:"\${stitch_dependency}" "${continuous_script}")"
echo "\${continuous_job_id}" > "${continuous_job_id_file}"
echo "Submitted continuous stitch job \${continuous_job_id} after block stitch jobs \${stitch_dependency}"
EOF
chmod 755 "$continuous_submitter_script"

transfer_script="$submit_root/transfer_to_bucket_when_done.sh"
transfer_log="$submit_root/transfer_to_bucket.log"
cat > "$transfer_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

continuous_job_id_file="${continuous_job_id_file}"
continuous_done_file="${continuous_done_file}"
transfer_manifest="${transfer_manifest}"
transfer_log="${transfer_log}"
poll_seconds="${transfer_poll_seconds}"
delete_flash_after_transfer="${delete_flash_after_transfer}"

log() {
  printf '[%s] %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" "\$*" | tee -a "\${transfer_log}"
}

wait_for_file() {
  local path="\$1"
  log "Waiting for \${path}"
  while [[ ! -s "\${path}" ]]; do
    sleep "\${poll_seconds}"
  done
}

wait_for_file "\${continuous_job_id_file}"
continuous_job_id="\$(head -n 1 "\${continuous_job_id_file}")"
log "Continuous stitch job is \${continuous_job_id}"
wait_for_file "\${continuous_done_file}"
log "Continuous stitch success marker found; starting transfer to bucket."

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
  rsync -a --ignore-existing --partial --protect-args "\${src}/" "\${dst}/"
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
  echo "[dry-run] transfer watcher script: $transfer_script"
elif [[ ${#track_submit_job_ids[@]} -gt 0 ]]; then
  submit_dependency="$(IFS=:; echo "${track_submit_job_ids[*]}")"
  continuous_submitter_job_id="$("$sbatch_bin" --parsable --dependency=afterok:"$submit_dependency" "$continuous_submitter_script")"
  echo "Submitted continuous stitch submitter job $continuous_submitter_job_id after tracking fan-out jobs $submit_dependency"
  if [[ "$transfer_to_bucket" -eq 1 ]]; then
    nohup "$transfer_script" >> "$transfer_log" 2>&1 &
    transfer_pid="$!"
    echo "Started login-side transfer watcher PID $transfer_pid; log: $transfer_log"
  fi
fi
