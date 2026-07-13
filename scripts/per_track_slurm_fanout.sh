#!/usr/bin/env bash
# Fan out a generic per-track operation over stitched per-track parquet files.
#
# The operation command is run once per TrackID_*.parquet file in its own
# Slurm job. Each worker gets these environment variables:
#   TRACK_PATH, TRACK_NAME, TRACK_STEM, TRACK_ID, TRACK_INDEX
#   TASK_OUTPUT_DIR, RUN_OUTPUT_DIR, PER_TRACK_DIR, JOBS_DIR
#
# Simple example for the standard colony layout:
#   bash scripts/per_track_slurm_fanout.sh \
#     --per_track_dir /flash/ReiterU/ant_tmp/$USER/colony_pipeline/20260515/block02/stitched/per_track \
#     --operation_script analysis/compute_track_colony_presence_vector.py

set -euo pipefail

PER_TRACK_DIR=""
FLASH_OUTPUT_DIR=""
BUCKET_OUTPUT_DIR=""
JOBS_DIR=""
OPERATION_NAME=""
OUTPUT_NAME=""
OPERATION_CMD=""
OPERATION_SCRIPT=""
OPERATION_ARGS=""
WORKER_SETUP=""
RUN_WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACK_GLOB="TrackID_*.parquet"
DONE_MARKER="_SUCCESS"
# Conda-free by default: workers run under the unit ant_tracking venv. Conda is
# kept only as an opt-in fallback (pass --conda_env NAME --conda_bin PATH).
CONDA_ENV=""
CONDA_BIN="/bucket/ReiterU/sam/miniforge3/bin/conda"
WORKER_PYTHON_BIN="/apps/unit/ReiterU/ant_tracking/venv/bin/python"
BUCKET_DATA_ROOT="/bucket/ReiterU/Ants/basler"
RUN_USER="${USER:-${LOGNAME:-unknown}}"
FLASH_PIPELINE_ROOT="/flash/ReiterU/ant_tmp/${RUN_USER}/colony_pipeline"

PARTITION="compute"
SBATCH_BIN="sbatch"
PYTHON_BIN="python3"
SBATCH_DEPENDENCY=""
CPUS=4
MEM="16G"
TIME_LIMIT="0-12:00:00"
POLL_SECONDS=120
TRANSFER_TO_BUCKET=1
DELETE_FLASH_AFTER_TRANSFER=0
SKIP_EXISTING=0
DRY_RUN=0
SLURM_SETUP=""

usage() {
  cat <<EOF
Run one Slurm job per per-track parquet file, then copy results back to bucket.

Required:
  --per_track_dir PATH       Folder containing per-track parquet files.
  --operation_script PATH    Python script called as: python SCRIPT --track "\$TRACK_PATH" --out "\$TASK_OUTPUT_DIR"
                            OR use --operation_cmd CMD for a custom worker command.

Useful options:
  --operation_name NAME      Job prefix. Default: operation-specific known name or operation script stem.
  --operation_args ARGS      Extra args appended after --track/--out when using --operation_script.
  --output_name NAME         Output folder under stitched/. Default: operation-specific known folder or operation_name.
  --flash_output_dir PATH    Flash output root. Default: <per_track_dir>/../<output_name>
  --bucket_output_dir PATH   Bucket destination. Default: ${BUCKET_DATA_ROOT}/<date>/<block>/stitched/<output_name>
  --bucket_data_root PATH    Bucket dataset root. Default: ${BUCKET_DATA_ROOT}
  --flash_pipeline_root PATH Flash colony_pipeline root used when --per_track_dir is a bucket path.
                            Default: ${FLASH_PIPELINE_ROOT}
  --jobs_dir PATH            Sbatch/log directory. Default: <flash_output_dir>/jobs
  --worker_setup CMD         Shell setup run inside each worker before CMD.
  --conda_env NAME           Conda env to activate inside each worker. Default: ${CONDA_ENV}
  --conda_bin PATH           Conda executable used for --conda_env. Default: ${CONDA_BIN}
  --no_conda                 Do not auto-activate conda inside workers.
  --worker_python_bin PATH   Python executable whose bin dir is prepended to PATH inside each worker.
  --run_workdir PATH         Worker working directory. Default: ${RUN_WORKDIR}
  --track_glob GLOB          Track file glob. Default: TrackID_*.parquet
  --skip_existing            Skip tracks with <TASK_OUTPUT_DIR>/${DONE_MARKER}.
  --done_marker NAME         Per-track success marker. Default: ${DONE_MARKER}
  --partition NAME           Slurm partition. Default: compute
  --cpus N                   CPUs per worker. Default: 4
  --mem MEM                  Memory per worker. Default: 16G
  --time TIME                Time limit per worker. Default: 0-12:00:00
  --python_bin PATH          Python used to build the worklist. Default: python3
  --sbatch_bin PATH          sbatch executable. Default: sbatch
  --dependency SPEC          Slurm dependency for worker jobs, e.g. afterok:12345.
  --slurm_setup CMD          Shell setup run before sbatch commands.
  --no_transfer_to_bucket    Do not start the rsync watcher.
  --transfer_poll_seconds N  Poll interval for transfer watcher. Default: 120
  --delete_flash_after_transfer
                            Delete flash output contents after successful rsync.
  --dry_run                  Write scripts/worklist without submitting jobs.
  -h, --help                 Show this help.

Operation command environment:
  TRACK_PATH                 Full input parquet path.
  TRACK_NAME                 Input filename.
  TRACK_STEM                 Input filename without suffix.
  TRACK_ID                   Numeric TrackID parsed from filename, if present.
  TRACK_INDEX                Zero-based task index.
  TASK_OUTPUT_DIR            Per-track output directory under flash_output_dir/per_track.
  RUN_OUTPUT_DIR             Flash output root.
  PER_TRACK_DIR              Input per-track folder.
  JOBS_DIR                   Generated jobs/logs folder.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --per_track_dir) PER_TRACK_DIR="$2"; shift 2 ;;
    --flash_output_dir) FLASH_OUTPUT_DIR="$2"; shift 2 ;;
    --bucket_output_dir) BUCKET_OUTPUT_DIR="$2"; shift 2 ;;
    --jobs_dir) JOBS_DIR="$2"; shift 2 ;;
    --operation_name) OPERATION_NAME="$2"; shift 2 ;;
    --output_name) OUTPUT_NAME="$2"; shift 2 ;;
    --operation_cmd) OPERATION_CMD="$2"; shift 2 ;;
    --operation_script) OPERATION_SCRIPT="$2"; shift 2 ;;
    --operation_args) OPERATION_ARGS="$2"; shift 2 ;;
    --worker_setup) WORKER_SETUP="$2"; shift 2 ;;
    --conda_env) CONDA_ENV="$2"; shift 2 ;;
    --conda_bin) CONDA_BIN="$2"; shift 2 ;;
    --no_conda) CONDA_ENV=""; shift ;;
    --worker_python_bin) WORKER_PYTHON_BIN="$2"; shift 2 ;;
    --bucket_data_root) BUCKET_DATA_ROOT="$2"; shift 2 ;;
    --flash_pipeline_root) FLASH_PIPELINE_ROOT="$2"; shift 2 ;;
    --run_workdir) RUN_WORKDIR="$2"; shift 2 ;;
    --track_glob) TRACK_GLOB="$2"; shift 2 ;;
    --done_marker) DONE_MARKER="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --python_bin) PYTHON_BIN="$2"; shift 2 ;;
    --sbatch_bin) SBATCH_BIN="$2"; shift 2 ;;
    --dependency|--sbatch_dependency) SBATCH_DEPENDENCY="$2"; shift 2 ;;
    --slurm_setup) SLURM_SETUP="$2"; shift 2 ;;
    --transfer_poll_seconds) POLL_SECONDS="$2"; shift 2 ;;
    --skip_existing) SKIP_EXISTING=1; shift ;;
    --no_transfer_to_bucket) TRANSFER_TO_BUCKET=0; shift ;;
    --delete_flash_after_transfer) DELETE_FLASH_AFTER_TRANSFER=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$PER_TRACK_DIR" ]]; then
  echo "ERROR: --per_track_dir is required." >&2
  usage >&2
  exit 2
fi
if [[ ! -d "$PER_TRACK_DIR" ]]; then
  echo "ERROR: per-track directory does not exist: $PER_TRACK_DIR" >&2
  exit 2
fi

if [[ -z "$OPERATION_CMD" && -z "$OPERATION_SCRIPT" ]]; then
  echo "ERROR: provide --operation_script or --operation_cmd." >&2
  usage >&2
  exit 2
fi
operation_script_stem=""
default_operation_name=""
default_output_name=""
if [[ -n "$OPERATION_SCRIPT" ]]; then
  operation_script_stem="$(basename "$OPERATION_SCRIPT")"
  operation_script_stem="${operation_script_stem%.*}"
  case "$operation_script_stem" in
    compute_track_speed_vector)
      default_operation_name="speed_vector"
      default_output_name="speed_vectors"
      ;;
    compute_track_colony_presence_vector)
      default_operation_name="colony_presence"
      default_output_name="colony_presence_vectors"
      ;;
    compute_track_grid_occupancy)
      default_operation_name="grid_occupancy"
      default_output_name="grid_occupancy_histograms"
      ;;
    compute_track_sleep_predictions)
      default_operation_name="sleep_prediction"
      default_output_name="sleep_predictions"
      ;;
    *)
      default_operation_name="$operation_script_stem"
      default_output_name="$operation_script_stem"
      ;;
  esac
fi
if [[ -z "$OPERATION_NAME" ]]; then
  OPERATION_NAME="${default_operation_name:-${OUTPUT_NAME:-per_track_operation}}"
fi
if [[ -z "$OUTPUT_NAME" ]]; then
  OUTPUT_NAME="${default_output_name:-$OPERATION_NAME}"
fi
if [[ -z "$OPERATION_CMD" ]]; then
  operation_script_q="$(printf '%q' "$OPERATION_SCRIPT")"
  OPERATION_ARGS="${OPERATION_ARGS//$'\n'/ }"
  OPERATION_CMD="python ${operation_script_q} --track \"\$TRACK_PATH\" --out \"\$TASK_OUTPUT_DIR\""
  if [[ -n "$OPERATION_ARGS" ]]; then
    OPERATION_CMD+=" ${OPERATION_ARGS}"
  fi
fi
dataset_date=""
block_name=""
inferred_bucket_data_root="$BUCKET_DATA_ROOT"
per_track_layout=""
if [[ "$PER_TRACK_DIR" =~ /colony_pipeline/([^/]+)/(block[^/]+)/stitched/per_track/?$ ]]; then
  dataset_date="${BASH_REMATCH[1]}"
  block_name="${BASH_REMATCH[2]}"
  per_track_layout="flash"
elif [[ "$PER_TRACK_DIR" =~ ^(.*/Ants/basler)/([^/]+)/(block[^/]+)/stitched/per_track/?$ ]]; then
  inferred_bucket_data_root="${BASH_REMATCH[1]}"
  dataset_date="${BASH_REMATCH[2]}"
  block_name="${BASH_REMATCH[3]}"
  per_track_layout="bucket"
fi
if [[ -z "$FLASH_OUTPUT_DIR" ]]; then
  if [[ "$per_track_layout" == "bucket" ]]; then
    FLASH_OUTPUT_DIR="${FLASH_PIPELINE_ROOT}/${dataset_date}/${block_name}/stitched/${OUTPUT_NAME}"
  else
    stitched_dir="$(dirname "$PER_TRACK_DIR")"
    FLASH_OUTPUT_DIR="${stitched_dir}/${OUTPUT_NAME}"
  fi
fi
if [[ -z "$BUCKET_OUTPUT_DIR" ]]; then
  if [[ -n "$dataset_date" && -n "$block_name" ]]; then
    BUCKET_OUTPUT_DIR="${inferred_bucket_data_root}/${dataset_date}/${block_name}/stitched/${OUTPUT_NAME}"
  else
    echo "ERROR: could not infer bucket output from --per_track_dir: $PER_TRACK_DIR" >&2
    echo "       Expected .../colony_pipeline/<date>/<block>/stitched/per_track or .../Ants/basler/<date>/<block>/stitched/per_track; pass --bucket_output_dir to override." >&2
    exit 2
  fi
fi
if [[ -z "$JOBS_DIR" ]]; then
  JOBS_DIR="$FLASH_OUTPUT_DIR/jobs"
fi
OPERATION_CMD="${OPERATION_CMD//$'\n'/ }"
WORKER_SETUP="${WORKER_SETUP//$'\n'/ }"

AUTO_WORKER_SETUP=""
if [[ -n "$CONDA_ENV" ]]; then
  conda_bin_q="$(printf '%q' "$CONDA_BIN")"
  conda_env_q="$(printf '%q' "$CONDA_ENV")"
  AUTO_WORKER_SETUP="conda_bin=${conda_bin_q}; conda_base=\$(dirname \"\$(dirname \"\${conda_bin}\")\"); if [[ -f \"\${conda_base}/etc/profile.d/conda.sh\" ]]; then source \"\${conda_base}/etc/profile.d/conda.sh\"; else eval \"\$(\"\${conda_bin}\" shell.bash hook)\"; fi; conda activate ${conda_env_q}; export PATH=\"\${CONDA_PREFIX}/bin:\${PATH}\"; hash -r"
fi
if [[ -n "$WORKER_PYTHON_BIN" ]]; then
  worker_python_bin_q="$(printf '%q' "$WORKER_PYTHON_BIN")"
  if [[ -n "$AUTO_WORKER_SETUP" ]]; then
    AUTO_WORKER_SETUP+="; "
  fi
  AUTO_WORKER_SETUP+="worker_python_bin=${worker_python_bin_q}; export PATH=\"\$(dirname \"\${worker_python_bin}\"):\${PATH}\"; hash -r"
fi
if [[ -n "$AUTO_WORKER_SETUP" && -n "$WORKER_SETUP" ]]; then
  WORKER_SETUP="${AUTO_WORKER_SETUP}; ${WORKER_SETUP}"
elif [[ -n "$AUTO_WORKER_SETUP" ]]; then
  WORKER_SETUP="$AUTO_WORKER_SETUP"
fi

if [[ -n "$SLURM_SETUP" ]]; then
  eval "$SLURM_SETUP"
fi

safe_name="$(printf '%s' "$OPERATION_NAME" | tr -c 'A-Za-z0-9_-' '_' | sed 's/^_*//; s/_*$//')"
if [[ -z "$safe_name" ]]; then
  safe_name="per_track_operation"
fi
job_prefix="${safe_name:0:16}"

WORKER_DIR="$JOBS_DIR/workers"
LOGS_DIR="$JOBS_DIR/logs"
WORKLIST="$JOBS_DIR/${safe_name}_worklist.tsv"
JOB_IDS_FILE="$JOBS_DIR/${safe_name}_job_ids.tsv"
FINAL_SCRIPT="$JOBS_DIR/${safe_name}_complete.sbatch"
FINAL_JOB_ID_FILE="$JOBS_DIR/${safe_name}_complete_job_id.txt"
DONE_FILE="$FLASH_OUTPUT_DIR/${safe_name}_complete.ok"
TRANSFER_SCRIPT="$JOBS_DIR/${safe_name}_transfer_when_done.sh"
TRANSFER_LOG="$JOBS_DIR/${safe_name}_transfer.log"
LOCK_DIR="$JOBS_DIR/${safe_name}_transfer.lock"
TRANSFER_DONE_BASENAME="${safe_name}_transfer_complete.ok"
RUN_EPOCH="$(date +%s)"

mkdir -p "$FLASH_OUTPUT_DIR" "$JOBS_DIR" "$WORKER_DIR" "$LOGS_DIR"
rm -f "$WORKLIST" "$JOB_IDS_FILE" "$FINAL_JOB_ID_FILE" "$DONE_FILE"
find "$WORKER_DIR" -maxdepth 1 -type f -name "${safe_name}_task*.sbatch" -delete

"$PYTHON_BIN" - "$PER_TRACK_DIR" "$FLASH_OUTPUT_DIR" "$WORKLIST" "$TRACK_GLOB" "$DONE_MARKER" "$SKIP_EXISTING" <<'PY'
from pathlib import Path
import re
import sys

per_track_dir = Path(sys.argv[1])
output_root = Path(sys.argv[2])
worklist = Path(sys.argv[3])
track_glob = sys.argv[4]
done_marker = sys.argv[5]
skip_existing = sys.argv[6] == "1"

rows = []
for path in sorted(per_track_dir.glob(track_glob)):
    if not path.is_file():
        continue
    stem = path.stem
    match = re.search(r"TrackID_(\d+)", stem)
    track_id = match.group(1) if match else ""
    task_output_dir = output_root / "per_track" / stem
    if skip_existing and (task_output_dir / done_marker).exists():
        continue
    rows.append((path, path.name, stem, track_id, task_output_dir))

worklist.parent.mkdir(parents=True, exist_ok=True)
with worklist.open("w", encoding="utf-8") as fh:
    for index, row in enumerate(rows):
        fh.write("\t".join([str(index), *(str(value) for value in row)]) + "\n")
PY

task_count="$(wc -l < "$WORKLIST" | tr -d '[:space:]')"
if [[ -z "$task_count" || "$task_count" -eq 0 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] no per-track work to submit. Worklist: $WORKLIST"
    exit 0
  fi
  printf 'completed %s no work\n' "$(date '+%Y-%m-%d %H:%M:%S')" > "$DONE_FILE"
  echo "No per-track work to submit. Wrote ${DONE_FILE}"
  if [[ "$TRANSFER_TO_BUCKET" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$BUCKET_OUTPUT_DIR"
    rsync -a --partial --protect-args "$FLASH_OUTPUT_DIR/" "$BUCKET_OUTPUT_DIR/"
    echo "Copied existing output to bucket: $BUCKET_OUTPUT_DIR"
  fi
  exit 0
fi

operation_cmd_quoted="$(printf '%q' "$OPERATION_CMD")"
worker_setup_quoted="$(printf '%q' "$WORKER_SETUP")"
run_workdir_quoted="$(printf '%q' "$RUN_WORKDIR")"

submit_sbatch() {
  local output status
  set +e
  output="$("$@" 2>&1)"
  status="$?"
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "ERROR: sbatch command failed with status ${status}" >&2
    printf '%s\n' "$output" >&2
    exit "$status"
  fi
  printf '%s\n' "$output" | tail -n 1
}

worker_job_ids=()
worker_dependency_args=()
if [[ -n "$SBATCH_DEPENDENCY" ]]; then
  worker_dependency_args=(--dependency="$SBATCH_DEPENDENCY")
fi
while IFS=$'\t' read -r task_index track_path track_name track_stem track_id task_output_dir; do
  worker="$WORKER_DIR/${safe_name}_task${task_index}.sbatch"
  task_index_q="$(printf '%q' "$task_index")"
  track_path_q="$(printf '%q' "$track_path")"
  track_name_q="$(printf '%q' "$track_name")"
  track_stem_q="$(printf '%q' "$track_stem")"
  track_id_q="$(printf '%q' "$track_id")"
  task_output_dir_q="$(printf '%q' "$task_output_dir")"
  flash_output_dir_q="$(printf '%q' "$FLASH_OUTPUT_DIR")"
  per_track_dir_q="$(printf '%q' "$PER_TRACK_DIR")"
  jobs_dir_q="$(printf '%q' "$JOBS_DIR")"
  done_marker_q="$(printf '%q' "$DONE_MARKER")"
  cat > "$worker" <<WORKER
#!/usr/bin/env bash
#SBATCH -J ${job_prefix}_${task_index}
#SBATCH -p ${PARTITION}
#SBATCH -c ${CPUS}
#SBATCH --mem=${MEM}
#SBATCH -t ${TIME_LIMIT}
#SBATCH -o ${LOGS_DIR}/${safe_name}_task${task_index}_%j.out
#SBATCH -e ${LOGS_DIR}/${safe_name}_task${task_index}_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

TRACK_INDEX=${task_index_q}
TRACK_PATH=${track_path_q}
TRACK_NAME=${track_name_q}
TRACK_STEM=${track_stem_q}
TRACK_ID=${track_id_q}
TASK_OUTPUT_DIR=${task_output_dir_q}
RUN_OUTPUT_DIR=${flash_output_dir_q}
PER_TRACK_DIR=${per_track_dir_q}
JOBS_DIR=${jobs_dir_q}
DONE_MARKER=${done_marker_q}
operation_cmd=${operation_cmd_quoted}
worker_setup=${worker_setup_quoted}
run_workdir=${run_workdir_quoted}

export TRACK_INDEX TRACK_PATH TRACK_NAME TRACK_STEM TRACK_ID TASK_OUTPUT_DIR
export RUN_OUTPUT_DIR PER_TRACK_DIR JOBS_DIR DONE_MARKER

mkdir -p "\${TASK_OUTPUT_DIR}"
cd "\${run_workdir}"
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Starting ${safe_name} task \${TRACK_INDEX}: \${TRACK_PATH}" >&2
if [[ -n "\${worker_setup}" ]]; then
  eval "\${worker_setup}"
fi
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Worker environment diagnostics:" >&2
echo "  PATH=\${PATH}" >&2
echo "  python=\$(command -v python || true)" >&2
python --version >&2 || true
python - <<'PY' >&2 || true
import os
import sys
print(f"  sys.executable={sys.executable}")
print(f"  sys.prefix={sys.prefix}")
print(f"  CONDA_PREFIX={os.environ.get('CONDA_PREFIX', '')}")
PY
bash -lc "\${operation_cmd}"
printf 'completed %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" > "\${TASK_OUTPUT_DIR}/\${DONE_MARKER}"
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Completed ${safe_name} task \${TRACK_INDEX}" >&2
WORKER
  chmod 755 "$worker"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] worker script: $worker"
    continue
  fi
  job_id="$(submit_sbatch "$SBATCH_BIN" --parsable "${worker_dependency_args[@]}" "$worker")"
  worker_job_ids+=( "$job_id" )
  printf '%s\t%s\t%s\t%s\n' "$task_index" "$job_id" "$track_path" "$task_output_dir" >> "$JOB_IDS_FILE"
  echo "Submitted ${safe_name} task ${task_index}: job ${job_id}"
done < "$WORKLIST"

cat > "$FINAL_SCRIPT" <<MARKER
#!/usr/bin/env bash
#SBATCH -J ${job_prefix}_done
#SBATCH -p ${PARTITION}
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 0-00:10:00
#SBATCH -o ${LOGS_DIR}/${safe_name}_complete_%j.out
#SBATCH -e ${LOGS_DIR}/${safe_name}_complete_%j.err

set -euo pipefail
printf 'completed %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" > "${DONE_FILE}"
MARKER
chmod 755 "$FINAL_SCRIPT"

cat > "$TRANSFER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail

done_file="${DONE_FILE}"
src_dir="${FLASH_OUTPUT_DIR}"
dst_dir="${BUCKET_OUTPUT_DIR}"
transfer_log="${TRANSFER_LOG}"
transfer_done_basename="${TRANSFER_DONE_BASENAME}"
poll_seconds="${POLL_SECONDS}"
delete_flash_after_transfer="${DELETE_FLASH_AFTER_TRANSFER}"
lock_dir="${LOCK_DIR}"
run_epoch="${RUN_EPOCH}"

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
log "done_file=\${done_file}"
log "source=\${src_dir}"
log "destination=\${dst_dir}"
log "run_epoch=\${run_epoch}"
acquire_lock
while true; do
  if [[ -s "\${done_file}" ]]; then
    mtime="\$(file_mtime "\${done_file}")"
    if [[ "\${mtime}" =~ ^[0-9]+$ ]] && (( mtime >= run_epoch )); then
      log "Fresh completion marker found: \${done_file} mtime=\${mtime}"
      break
    fi
    log "Ignoring stale completion marker: \${done_file} mtime=\${mtime}"
  fi
  sleep "\${poll_seconds}"
done

mkdir -p "\${dst_dir}"
log "rsync \${src_dir}/ -> \${dst_dir}/"
rsync -a --partial --protect-args "\${src_dir}/" "\${dst_dir}/" >> "\${transfer_log}" 2>&1
log "rsync complete: \${src_dir}/ -> \${dst_dir}/"
dst_done_file="\${dst_dir}/\${transfer_done_basename}"
printf 'completed %s\nsource=%s\ndestination=%s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" "\${src_dir}" "\${dst_dir}" > "\${dst_done_file}"
log "TRANSFER_TO_BUCKET_COMPLETE operation=${safe_name} destination=\${dst_dir} marker=\${dst_done_file}"
if [[ "\${delete_flash_after_transfer}" -eq 1 ]]; then
  log "Deleting transferred source contents: \${src_dir}"
  find "\${src_dir}" -mindepth 1 -delete
fi
log "Transfer to bucket complete."
EOF
chmod 755 "$TRANSFER_SCRIPT"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] worklist: $WORKLIST"
  echo "[dry-run] final marker script: $FINAL_SCRIPT"
  echo "[dry-run] transfer watcher: $TRANSFER_SCRIPT"
  exit 0
fi

dependency="$(IFS=:; echo "${worker_job_ids[*]}")"
final_job_id="$(submit_sbatch "$SBATCH_BIN" --parsable --dependency=afterok:"$dependency" "$FINAL_SCRIPT")"
echo "$final_job_id" > "$FINAL_JOB_ID_FILE"
echo "Submitted completion marker job ${final_job_id} after ${#worker_job_ids[@]} worker jobs."

if [[ "$TRANSFER_TO_BUCKET" -eq 1 ]]; then
  nohup "$TRANSFER_SCRIPT" >/dev/null 2>&1 &
  transfer_pid="$!"
  echo "Started transfer watcher PID ${transfer_pid}; log: ${TRANSFER_LOG}"
  echo "Transfer completion line: grep TRANSFER_TO_BUCKET_COMPLETE ${TRANSFER_LOG}"
  echo "Bucket transfer marker: ${BUCKET_OUTPUT_DIR}/${TRANSFER_DONE_BASENAME}"
fi
