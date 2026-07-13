#!/bin/bash -l
# pipeline.sh — entry point for the chunk-ordered aruco + sleap pipeline.
#
# Run from a deigo login. Builds a manifest of grid videos, renders the
# per-stage sbatch templates, and submits the stage DAG. Each downstream
# template sources $JOBS_ROOT/pipeline.env for all configuration.
#
# Usage:
#   bash detection_pipeline/pipeline.sh --dir <experiment_dir> \
#        --sleap-model-centroid <dir> --sleap-model-instance <dir> [options]
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"
TEMPLATES_DIR="$SCRIPT_DIR/templates"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

usage() {
	cat <<'EOT'
Usage: bash pipeline.sh --dir <experiment_dir> [options]

Required:
  --dir PATH                        Experiment directory with grid*.{mkv,mp4,avi}
  --sleap-model-centroid PATH       SLEAP centroid model directory
  --sleap-model-instance PATH       SLEAP centered-instance model directory

Aruco:
  --aruco-dict {A|B|PATH}           A=custom_4x4_A100, B=custom_4x4_B300, or full .npz path
                                    default: A
  --aruco-params "FLAGS"            Extra run_aruco.py detector parameter flags
                                    from aruco_curation.py parameter tests.

Chunking:
  --chunk-sec N                     Chunk duration in seconds. default: 7200 (2h)
  --chunk-ext {mkv|mp4|avi}         Output chunk container. default: mkv

SLEAP runtime:
  --sleap-runtime {tensorrt|onnx|pytorch}  default: tensorrt
  --skip-trt-export                 Use 'sleap-nn track' fallback (raw model dirs)
  --saion-partition NAME            default: largegpu. short-a100 = 4x GPUs (2h wall).
                                    Picking the partition auto-sizes the four knobs
                                    below; you only set them to hold resources back.
  --sleap-module NAME               saion module name. default: sleap-nn/0.2.0
  --sleap-batch-size N              TRT inference batch; must be <= engine max profile batch. default: 8
  --sleap-cpus N                    cpus per sleap task. default: auto = cpu_cap/concurrency
  --sleap-mem SIZE                  mem per sleap task.  default: auto = mem_cap/concurrency
  --sleap-wall D-HH                 per-task walltime.   default: auto = partition wall
                                                         (largegpu 0-12, short-a100 0-2)

Concurrency:
  --aruco-concurrency N             default: 16
  --sleap-concurrency N             default: auto = partition GPU cap (largegpu 8, short-a100 32)
  --datacp-concurrency N            default: 4

Batching (minimize submitted job count):
  --batch-size N                    chunks per array task. default: 1
                                    (one chunk/task; avoids per-task walltime
                                    timeouts on the slow aruco leg. Pass a larger
                                    N to pack chunks, or "" to auto-size:
                                    auto = ceil(total_chunks / --max-array-tasks))
  --max-array-tasks N               cap when auto-sizing batch (--batch-size ""). default: 500

Phase isolation (for testing):
  --only-chunk                      Stop after chunking
  --only-aruco                      Skip sleap branch
  --only-sleap                      Skip aruco branch (still chunks)
  --only-backup                     Build manifest + submit ONLY the raw-video
                                    backup job (no chunk/aruco/sleap). Handy for
                                    testing the backup or re-backing-up a block.

Roots:
  --jobs-root PATH                  default: /flash/ReiterU/$USER/jobs/<exp>
  --flash-root PATH                 default: /flash/ReiterU/$USER/<exp>

Backup:
  --no-backup                       Do not submit the automatic raw-video backup
  --backup-root PATH                default: /bucket/<unit>/Backup/<collection>
                                    (collection = exp path minus date/block,
                                    e.g. Ants_basler)
  --backup-archive NAME             archive filename under --backup-root;
                                    default: <date>_<block>_raw_videos.zip
  --backup-owner TEXT               description file name: line. default: $USER
  --backup-project TEXT             description file project: line
  --backup-partition NAME           default: datacp

Permissions:
  --group NAME                      Group owner for shared bucket outputs;
                                    chgrp + setgid on dirs the pipeline creates.
                                    default: reiteruni

Tracking (optional auto-trigger after detection completes):
  --run-tracking                    After aruco+sleap outputs land in the bucket,
                                    auto-submit colony tracking for this block.
                                    Requires --tracking-hmats. default: off
  --no-run-tracking                 Explicitly disable the tracking auto-trigger.
  --tracking-hmats PATH             Homography .npz (key 'H'). Required with
                                    --run-tracking.
  --tracking-submit PATH            tracking/colony/submit_blocks_pipeline.sh path.
                                    default: <repo>/tracking/colony/submit_blocks_pipeline.sh
  --tracking-python-bin PATH        Conda-free python for tracking jobs; overrides the
                                    submit script's DEFAULT_PYTHON_BIN unit venv.
  --tracking-output-root PATH       Flash output root for tracking.
                                    default: /flash/ReiterU/$USER/colony_pipeline/<date>
  --tracking-args "FLAGS"           Extra space-separated flags passed verbatim to the
                                    tracking submit script.
  --tracking-poll-secs N            Bucket poll interval. default: 300
  --tracking-timeout N              Deadline (s) to wait for detection outputs.
                                    default: 172800 (48h)

Other:
  -h, --help                        Show this help
EOT
	exit 1
}

# Defaults
DIR=""
ARUCO_DICT="A"
ARUCO_EXTRA_ARGS=""
CHUNK_SEC=7200
CHUNK_EXT=mkv
SLEAP_MODEL_CENTROID=""
SLEAP_MODEL_INSTANCE=""
SLEAP_RUNTIME=tensorrt
SKIP_TRT_EXPORT=0
SAION_PARTITION=largegpu
SLEAP_MODULE="sleap-nn/0.2.0"
# TRT per-frame inference batch. Must be <= the exported engine's max optimization
# profile batch. Full-res Simple_skeleton engines max out at 8 (batch>=16 fails to
# build with a Myelin int32 overflow), so 8 is the safe default; raise only if the
# engine was exported with a larger max batch.
SLEAP_BATCH_SIZE=8
ARUCO_CONCURRENCY=100   # compute assoc cap=2000 cpu; at -c 16 that's ~125 concurrent max, so 100 leaves headroom for the bridge (also on compute)
# Sleap GPU concurrency + per-task resources. Empty = auto-derived from the
# selected --saion-partition after arg parsing (see saion_caps below):
#   concurrency -> partition per-user GPU cap (largegpu=8, short-a100=32)
#   cpus / mem  -> cpu_cap / mem_cap divided by concurrency (saturates the caps)
#   wall        -> partition default wall (largegpu=0-12, short-a100=0-2)
# Set any of these explicitly only to hold resources back for other jobs.
SLEAP_CONCURRENCY=""
SLEAP_CPUS=""
SLEAP_MEM=""
SLEAP_WALL=""
DATACP_CONCURRENCY=4
BATCH_SIZE=1         # default: one chunk per array task (set "" to auto-size under MAX_ARRAY_TASKS)
MAX_ARRAY_TASKS=500
OUTPUT_GROUP=reiteruni   # group owner for shared bucket outputs (chgrp + setgid on created dirs)
ONLY_CHUNK=0
ONLY_ARUCO=0
ONLY_SLEAP=0
ONLY_BACKUP=0
JOBS_ROOT=""
FLASH_ROOT=""
RUN_BACKUP=1
BACKUP_ROOT=""
BACKUP_ARCHIVE=""
BACKUP_OWNER="${USER:-unknown}"
BACKUP_PROJECT=""
BACKUP_PARTITION=datacp
# Tracking auto-trigger (optional, off by default). RUN_TRACKING gates a login-side
# poller (templates/track_trigger.sh) that submits colony tracking for this block once
# detection outputs are all in the bucket. See tracking/colony/submit_blocks_pipeline.sh.
RUN_TRACKING=0
TRACKING_HMATS=""
TRACKING_SUBMIT=""
TRACKING_PYTHON_BIN=""
TRACKING_OUTPUT_ROOT=""
TRACKING_EXTRA_ARGS=""
TRACKING_POLL_SECS=300
TRACKING_TIMEOUT=172800

while [[ $# -gt 0 ]]; do
	case "$1" in
		--dir) DIR="$2"; shift 2 ;;
		--aruco-dict) ARUCO_DICT="$2"; shift 2 ;;
		--aruco-params) ARUCO_EXTRA_ARGS="$2"; shift 2 ;;
		--chunk-sec) CHUNK_SEC="$2"; shift 2 ;;
		--chunk-ext) CHUNK_EXT="$2"; shift 2 ;;
		--sleap-model-centroid) SLEAP_MODEL_CENTROID="$2"; shift 2 ;;
		--sleap-model-instance) SLEAP_MODEL_INSTANCE="$2"; shift 2 ;;
		--sleap-runtime) SLEAP_RUNTIME="$2"; shift 2 ;;
		--skip-trt-export) SKIP_TRT_EXPORT=1; shift ;;
		--saion-partition) SAION_PARTITION="$2"; shift 2 ;;
		--sleap-module) SLEAP_MODULE="$2"; shift 2 ;;
		--sleap-batch-size) SLEAP_BATCH_SIZE="$2"; shift 2 ;;
		--aruco-concurrency) ARUCO_CONCURRENCY="$2"; shift 2 ;;
		--sleap-concurrency) SLEAP_CONCURRENCY="$2"; shift 2 ;;
		--sleap-cpus) SLEAP_CPUS="$2"; shift 2 ;;
		--sleap-mem) SLEAP_MEM="$2"; shift 2 ;;
		--sleap-wall) SLEAP_WALL="$2"; shift 2 ;;
		--datacp-concurrency) DATACP_CONCURRENCY="$2"; shift 2 ;;
		--batch-size) BATCH_SIZE="$2"; shift 2 ;;
		--max-array-tasks) MAX_ARRAY_TASKS="$2"; shift 2 ;;
		--group) OUTPUT_GROUP="$2"; shift 2 ;;
		--only-chunk) ONLY_CHUNK=1; shift ;;
		--only-aruco) ONLY_ARUCO=1; shift ;;
		--only-sleap) ONLY_SLEAP=1; shift ;;
		--only-backup) ONLY_BACKUP=1; shift ;;
		--jobs-root) JOBS_ROOT="$2"; shift 2 ;;
		--flash-root) FLASH_ROOT="$2"; shift 2 ;;
		--no-backup) RUN_BACKUP=0; shift ;;
		--backup-root) BACKUP_ROOT="$2"; shift 2 ;;
		--backup-archive) BACKUP_ARCHIVE="$2"; shift 2 ;;
		--backup-owner) BACKUP_OWNER="$2"; shift 2 ;;
		--backup-project) BACKUP_PROJECT="$2"; shift 2 ;;
		--backup-partition) BACKUP_PARTITION="$2"; shift 2 ;;
		--run-tracking) RUN_TRACKING=1; shift ;;
		--no-run-tracking) RUN_TRACKING=0; shift ;;
		--tracking-hmats) TRACKING_HMATS="$2"; shift 2 ;;
		--tracking-submit) TRACKING_SUBMIT="$2"; shift 2 ;;
		--tracking-python-bin) TRACKING_PYTHON_BIN="$2"; shift 2 ;;
		--tracking-output-root) TRACKING_OUTPUT_ROOT="$2"; shift 2 ;;
		--tracking-args) TRACKING_EXTRA_ARGS="$2"; shift 2 ;;
		--tracking-poll-secs) TRACKING_POLL_SECS="$2"; shift 2 ;;
		--tracking-timeout) TRACKING_TIMEOUT="$2"; shift 2 ;;
		-h|--help) usage ;;
		*) echo "[ERR] unknown arg: $1" >&2; usage ;;
	esac
done

[[ -d "$DIR" ]] || { echo "[ERR] --dir is required and must exist" >&2; usage; }
DIR=$(readlink -f "$DIR")
EXP_NAME=$(basename "$DIR")
# Namespace scratch (/flash jobs + /work) by <date>_<block> so same-named blocks
# from different dates (20260707/block01 vs 20260713/block01) don't collide and
# overwrite each other's rendered templates / pipeline.env / chunks.
if [[ "$EXP_NAME" =~ ^block[0-9] ]]; then
	EXP_NAME="$(basename "$(dirname "$DIR")")_$EXP_NAME"
fi

BACKUP_UNIT_ROOT=""
BACKUP_REL_DIR=""
BACKUP_ARCHIVE_PATH=""
BACKUP_DESC_PATH=""
if (( RUN_BACKUP == 1 )); then
	if [[ "$DIR" != /bucket/*/* ]]; then
		echo "[ERR] automatic backup requires --dir under /bucket/<unit>/...; pass --no-backup to skip" >&2
		exit 2
	fi
	bucket_tail="${DIR#/bucket/}"
	unit_name="${bucket_tail%%/*}"
	BACKUP_UNIT_ROOT="/bucket/$unit_name"
	BACKUP_REL_DIR="${DIR#$BACKUP_UNIT_ROOT/}"

	# Group archives under Backup/<collection>/ rather than one flat directory.
	# Split the experiment's relative path into a collection (every component
	# except the last two, e.g. Ants/basler) and a per-block tail (the last two,
	# e.g. 20260520/block02): the collection becomes a subfolder under Backup/
	# and the tail becomes the archive filename. Paths shallower than three
	# components fall back to the old flat Backup/ naming.
	sanitize_token() { printf '%s' "${1//\//_}" | sed -e 's/[^A-Za-z0-9._-]/_/g' -e 's/___*/_/g' -e 's/^_//' -e 's/_$//'; }
	IFS='/' read -ra _rel_parts <<< "$BACKUP_REL_DIR"
	if (( ${#_rel_parts[@]} >= 3 )); then
		backup_collection="${BACKUP_REL_DIR%/*/*}"
		backup_tail="${BACKUP_REL_DIR#"$backup_collection"/}"
	else
		backup_collection=""
		backup_tail="$BACKUP_REL_DIR"
	fi
	collection_token=$(sanitize_token "$backup_collection")
	BACKUP_ROOT="${BACKUP_ROOT:-$BACKUP_UNIT_ROOT/Backup${collection_token:+/$collection_token}}"
	if [[ -z "$BACKUP_ARCHIVE" ]]; then
		BACKUP_ARCHIVE="$(sanitize_token "$backup_tail")_raw_videos.zip"
	fi
	[[ "$BACKUP_ARCHIVE" != */* ]] || { echo "[ERR] --backup-archive must be a filename, not a path" >&2; exit 2; }
	[[ "$BACKUP_ARCHIVE" == *.zip ]] || BACKUP_ARCHIVE="${BACKUP_ARCHIVE}.zip"
	BACKUP_PROJECT="${BACKUP_PROJECT:-AntsArray raw videos: $BACKUP_REL_DIR}"
	BACKUP_ARCHIVE_PATH="$BACKUP_ROOT/$BACKUP_ARCHIVE"
	BACKUP_DESC_PATH="${BACKUP_ARCHIVE_PATH%.zip}.txt"
fi

# Validate model dirs unless we're only chunking or only doing aruco
if (( ONLY_CHUNK != 1 && ONLY_ARUCO != 1 && ONLY_BACKUP != 1 )); then
	[[ -d "$SLEAP_MODEL_CENTROID" ]] || { echo "[ERR] --sleap-model-centroid must exist (dir)" >&2; exit 2; }
	[[ -d "$SLEAP_MODEL_INSTANCE" ]] || { echo "[ERR] --sleap-model-instance must exist (dir)" >&2; exit 2; }
fi

# --- Auto-size sleap GPU resources from the selected partition ----------------
# Per-user association caps differ by partition, and the cpu/mem-per-GPU ratio
# is NOT constant (largegpu: 16 cpu + 128 GB per GPU; short-a100: 8 cpu + 64 GB
# per GPU). Derive concurrency/cpus/mem/wall from the partition unless the user
# set them explicitly (empty = unset). Override any one to hold resources back.
saion_caps() {
	# echo: <gpu_cap> <cpu_cap> <mem_cap_GB> <default_wall>
	case "$1" in
		largegpu)   echo "8 128 1024 0-12" ;;
		short-a100) echo "32 256 2048 0-2"  ;;
		gpu-a100)   echo "8 128 1024 0-8"   ;;
		*)          echo "" ;;
	esac
}
read -r _GPU_CAP _CPU_CAP _MEM_CAP _DEF_WALL <<<"$(saion_caps "$SAION_PARTITION")" || true
if [[ -n "$_GPU_CAP" ]]; then
	: "${SLEAP_CONCURRENCY:=$_GPU_CAP}"
	if (( SLEAP_CONCURRENCY > _GPU_CAP )); then
		echo "[WARN] --sleap-concurrency $SLEAP_CONCURRENCY exceeds $SAION_PARTITION per-user GPU cap $_GPU_CAP; extra tasks will pend (AssocGrpGRES)" >&2
	fi
	if [[ -z "$SLEAP_CPUS" ]]; then
		SLEAP_CPUS=$(( _CPU_CAP / SLEAP_CONCURRENCY ))
		(( SLEAP_CPUS >= 1 )) || SLEAP_CPUS=1
	fi
	if [[ -z "$SLEAP_MEM" ]]; then
		SLEAP_MEM="$(( _MEM_CAP / SLEAP_CONCURRENCY ))G"
	fi
	: "${SLEAP_WALL:=$_DEF_WALL}"
else
	# Unknown partition: fall back to the legacy largegpu-shaped defaults.
	: "${SLEAP_CONCURRENCY:=8}"
	: "${SLEAP_CPUS:=16}"
	: "${SLEAP_MEM:=128G}"
	: "${SLEAP_WALL:=0-12}"
fi
echo "[INFO] sleap: partition=$SAION_PARTITION concurrency=$SLEAP_CONCURRENCY per-task '-c $SLEAP_CPUS --mem=$SLEAP_MEM -t $SLEAP_WALL'"

# Resolve aruco dict
ARUCO_DICT_ROOT="/bucket/ReiterU/Ants/aruco_dicts"
case "$ARUCO_DICT" in
	A) ARUCO_DICT_PATH=$(ls -1 "$ARUCO_DICT_ROOT"/custom_4x4_A100_d4_*.npz 2>/dev/null | sort -r | head -1) ;;
	B) ARUCO_DICT_PATH=$(ls -1 "$ARUCO_DICT_ROOT"/custom_4x4_B300_d4_*.npz 2>/dev/null | sort -r | head -1) ;;
	*) ARUCO_DICT_PATH="$ARUCO_DICT" ;;
esac
if (( ONLY_SLEAP != 1 && ONLY_BACKUP != 1 )); then
	[[ -f "$ARUCO_DICT_PATH" ]] || { echo "[ERR] aruco dict npz not found: $ARUCO_DICT_PATH" >&2; exit 2; }
fi

case "$SLEAP_RUNTIME" in
	tensorrt|onnx|pytorch) ;;
	*) echo "[ERR] --sleap-runtime must be tensorrt|onnx|pytorch" >&2; exit 2 ;;
esac
case "$CHUNK_EXT" in
	mkv|mp4|avi) ;;
	*) echo "[ERR] --chunk-ext must be mkv|mp4|avi" >&2; exit 2 ;;
esac

# Roots
JOBS_ROOT="${JOBS_ROOT:-/flash/ReiterU/$USER/jobs/$EXP_NAME}"
FLASH_ROOT="${FLASH_ROOT:-/flash/ReiterU/$USER/$EXP_NAME}"
SAION_WORK_ROOT="/work/ReiterU/$USER/$EXP_NAME"
DATA_DIR="$DIR/data"
HPC_LOGS_DIR="$DIR/hpc_logs"
ENV_FILE="$JOBS_ROOT/pipeline.env"

mkdir -p "$JOBS_ROOT" "$FLASH_ROOT" "$DATA_DIR" "$HPC_LOGS_DIR"
source "$LIB_DIR/perms.sh"
source "$LIB_DIR/hosts.sh"   # sbatch_retry: survive transient slurmctld socket timeouts
ensure_group_perms "$JOBS_ROOT" "$FLASH_ROOT" "$DATA_DIR" "$HPC_LOGS_DIR"
# Preflight: warn (don't fail) if the experiment dir isn't group-shared. It may
# hold other users' files we can't chgrp, so this is advisory only.
check_group_perms "$DIR" || true

# --- Tracking auto-trigger: validate + derive defaults ------------------------
if (( RUN_TRACKING == 1 )); then
	if (( ONLY_CHUNK == 1 || ONLY_ARUCO == 1 || ONLY_SLEAP == 1 || ONLY_BACKUP == 1 )); then
		echo "[ERR] --run-tracking needs a full aruco+sleap run; drop the --only-* flag(s)" >&2
		exit 2
	fi
	: "${TRACKING_SUBMIT:=$(cd "$SCRIPT_DIR/.." && pwd)/tracking/colony/submit_blocks_pipeline.sh}"
	[[ -f "$TRACKING_HMATS" ]] || { echo "[ERR] --run-tracking requires --tracking-hmats <existing .npz>" >&2; exit 2; }
	[[ -f "$TRACKING_SUBMIT" ]] || { echo "[ERR] tracking submit script not found: $TRACKING_SUBMIT" >&2; exit 2; }
	: "${TRACKING_OUTPUT_ROOT:=/flash/ReiterU/$USER/colony_pipeline/$(basename "$(dirname "$DIR")")}"
	echo "[INFO] tracking auto-trigger ON: submit=$TRACKING_SUBMIT hmats=$TRACKING_HMATS output_root=$TRACKING_OUTPUT_ROOT"
fi

cat > "$ENV_FILE" <<EOF
# Auto-generated by pipeline.sh at $(date)
export EXP_NAME="$EXP_NAME"
export EXP_DIR="$DIR"
export DATA_DIR="$DATA_DIR"
export HPC_LOGS_DIR="$HPC_LOGS_DIR"
export LOG_SHIP_INTERVAL="${LOG_SHIP_INTERVAL:-300}"
export FLASH_ROOT="$FLASH_ROOT"
export JOBS_ROOT="$JOBS_ROOT"
export ENV_FILE="$ENV_FILE"
export SAION_WORK_ROOT="$SAION_WORK_ROOT"
export TEMPLATES_DIR="$TEMPLATES_DIR"
export LIB_DIR="$LIB_DIR"
export SCRIPTS_DIR="$SCRIPTS_DIR"
export CHUNK_SEC="$CHUNK_SEC"
export CHUNK_EXT="$CHUNK_EXT"
export ARUCO_DICT_PATH="$ARUCO_DICT_PATH"
export ARUCO_EXTRA_ARGS="$ARUCO_EXTRA_ARGS"
export ARUCO_CONCURRENCY="$ARUCO_CONCURRENCY"
export SLEAP_CONCURRENCY="$SLEAP_CONCURRENCY"
export SLEAP_CPUS="$SLEAP_CPUS"
export SLEAP_MEM="$SLEAP_MEM"
export SLEAP_WALL="$SLEAP_WALL"
export DATACP_CONCURRENCY="$DATACP_CONCURRENCY"
export BATCH_SIZE="$BATCH_SIZE"
export MAX_ARRAY_TASKS="$MAX_ARRAY_TASKS"
export OUTPUT_GROUP="$OUTPUT_GROUP"
export SAION_PARTITION="$SAION_PARTITION"
export SLEAP_MODULE="$SLEAP_MODULE"
export SLEAP_BATCH_SIZE="$SLEAP_BATCH_SIZE"
export SLEAP_MODEL_CENTROID="$SLEAP_MODEL_CENTROID"
export SLEAP_MODEL_INSTANCE="$SLEAP_MODEL_INSTANCE"
export SLEAP_RUNTIME="$SLEAP_RUNTIME"
export SKIP_TRT_EXPORT="$SKIP_TRT_EXPORT"
export ONLY_ARUCO="$ONLY_ARUCO"
export ONLY_SLEAP="$ONLY_SLEAP"
export RUN_BACKUP="$RUN_BACKUP"
export BACKUP_UNIT_ROOT="$BACKUP_UNIT_ROOT"
export BACKUP_REL_DIR="$BACKUP_REL_DIR"
export BACKUP_ROOT="$BACKUP_ROOT"
export BACKUP_ARCHIVE_PATH="$BACKUP_ARCHIVE_PATH"
export BACKUP_DESC_PATH="$BACKUP_DESC_PATH"
export BACKUP_OWNER="$BACKUP_OWNER"
export BACKUP_PROJECT="$BACKUP_PROJECT"
export BACKUP_PARTITION="$BACKUP_PARTITION"
export RUN_TRACKING="$RUN_TRACKING"
export TRACKING_HMATS="$TRACKING_HMATS"
export TRACKING_SUBMIT="$TRACKING_SUBMIT"
export TRACKING_PYTHON_BIN="$TRACKING_PYTHON_BIN"
export TRACKING_OUTPUT_ROOT="$TRACKING_OUTPUT_ROOT"
export TRACKING_EXTRA_ARGS="$TRACKING_EXTRA_ARGS"
export TRACKING_POLL_SECS="$TRACKING_POLL_SECS"
export TRACKING_TIMEOUT="$TRACKING_TIMEOUT"
EOF
echo "[INFO] env file: $ENV_FILE"

# Build manifest
MANIFEST="$JOBS_ROOT/manifest.csv"
echo "[INFO] building manifest -> $MANIFEST"
MANIFEST_EXTRA=()
(( ONLY_BACKUP == 1 )) && MANIFEST_EXTRA+=(--no-probe)   # backup needs only source_path; skip ffprobe
python3 "$LIB_DIR/manifest.py" --dir "$DIR" --out "$MANIFEST" --chunk-sec "$CHUNK_SEC" "${MANIFEST_EXTRA[@]}"

N_VIDEOS=$(($(wc -l < "$MANIFEST") - 1))
if (( N_VIDEOS <= 0 )); then
	echo "[ERR] manifest has no grid videos" >&2
	exit 2
fi
echo "[INFO] $N_VIDEOS grid videos discovered"

# Render every template once (single placeholder: __JOBS_ROOT__)
echo "[INFO] rendering templates -> $JOBS_ROOT/"
for t in chunk.sbatch chunk_finalize.sbatch backup.sbatch aruco_array.sbatch aruco_datacp.sbatch bridge.sbatch cleanup.sbatch; do
	sed "s#__JOBS_ROOT__#$JOBS_ROOT#g" "$TEMPLATES_DIR/$t" > "$JOBS_ROOT/$t"
	chmod +x "$JOBS_ROOT/$t"
done

# --only-backup: build manifest + submit ONLY the raw-video backup job.
if (( ONLY_BACKUP )); then
	(( RUN_BACKUP == 1 )) || { echo "[ERR] --only-backup conflicts with --no-backup" >&2; exit 2; }
	JID_BACKUP=$(sbatch_retry backup --partition="$BACKUP_PARTITION" "$JOBS_ROOT/backup.sbatch")
	echo "  backup        $JID_BACKUP"
	echo "$JID_BACKUP" > "$JOBS_ROOT/jid_backup.txt"
	echo "[INFO] --only-backup: submitted backup job only -> ${BACKUP_ARCHIVE_PATH:-(disabled)}"
	exit 0
fi

# Submit chunk array
CHUNK_UPPER=$(( N_VIDEOS - 1 ))
echo "[INFO] sbatch chunk_array=0-${CHUNK_UPPER}"
JID_CHUNK=$(sbatch_retry chunk --array=0-${CHUNK_UPPER} "$JOBS_ROOT/chunk.sbatch")
echo "  chunk         $JID_CHUNK"
echo "$JID_CHUNK" > "$JOBS_ROOT/jid_chunk.txt"

if (( ONLY_CHUNK )); then
	echo "[INFO] --only-chunk: stopping after chunk submission"
	exit 0
fi

# Submit chunk_finalize (builds worklist + submits aruco / bridge / cleanup)
JID_CHUNK_FIN=$(sbatch_retry chunk_fin --dependency=afterok:$JID_CHUNK "$JOBS_ROOT/chunk_finalize.sbatch")
echo "  chunk_finalize $JID_CHUNK_FIN (dep: $JID_CHUNK)"
echo "$JID_CHUNK_FIN" > "$JOBS_ROOT/jid_chunk_fin.txt"

# Optional: launch the login-side tracking auto-trigger (nohup poller). It waits for
# detection outputs to appear in the bucket, then submits colony tracking for this
# block. Mirrors the tracking transfer watcher's login-nohup pattern; survives logout.
if (( RUN_TRACKING == 1 )); then
	mkdir -p "$HPC_LOGS_DIR/pipeline"
	sed "s#__JOBS_ROOT__#$JOBS_ROOT#g" "$TEMPLATES_DIR/track_trigger.sh" > "$JOBS_ROOT/track_trigger.sh"
	chmod +x "$JOBS_ROOT/track_trigger.sh"
	nohup "$JOBS_ROOT/track_trigger.sh" >> "$HPC_LOGS_DIR/pipeline/track_trigger.log" 2>&1 &
	echo "  track_trigger  PID $! (nohup; log: $HPC_LOGS_DIR/pipeline/track_trigger.log)"
fi

cat <<EOF

[INFO] pipeline submitted for $EXP_NAME

Monitor:
  squeue -u \$USER
  ls $JOBS_ROOT/

Job ids:
  $JOBS_ROOT/jid_*.txt

Stage outputs:
  flash chunks      $FLASH_ROOT/<vname>/<vname>_NNN.$CHUNK_EXT
  aruco staging     $FLASH_ROOT/aruco/<vname>/
  bucket outputs    $DATA_DIR/
  bucket backup     ${BACKUP_ARCHIVE_PATH:-disabled}
EOF
