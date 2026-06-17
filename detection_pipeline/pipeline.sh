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
  --saion-partition NAME            default: largegpu
  --sleap-module NAME               saion module name. default: sleap-nn/0.2.0
  --sleap-batch-size N              TRT inference batch; must be <= engine max profile batch. default: 8

Concurrency:
  --aruco-concurrency N             default: 16
  --sleap-concurrency N             default: 8
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
  --backup-root PATH                default: /bucket/<unit>/Backup
  --backup-archive NAME             archive filename under --backup-root;
                                    default: <relative_exp_path>_raw_videos.zip
  --backup-owner TEXT               description file name: line. default: $USER
  --backup-project TEXT             description file project: line
  --backup-partition NAME           default: datacp

Permissions:
  --group NAME                      Group owner for shared bucket outputs;
                                    chgrp + setgid on dirs the pipeline creates.
                                    default: reiteruni

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
SLEAP_CONCURRENCY=8     # bounded by saion largegpu having only 4 A100 nodes anyway
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
		-h|--help) usage ;;
		*) echo "[ERR] unknown arg: $1" >&2; usage ;;
	esac
done

[[ -d "$DIR" ]] || { echo "[ERR] --dir is required and must exist" >&2; usage; }
DIR=$(readlink -f "$DIR")
EXP_NAME=$(basename "$DIR")

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
	BACKUP_ROOT="${BACKUP_ROOT:-$BACKUP_UNIT_ROOT/Backup}"
	if [[ -z "$BACKUP_ARCHIVE" ]]; then
		safe_rel="${BACKUP_REL_DIR//\//_}"
		safe_rel="${safe_rel// /_}"
		safe_rel=$(printf '%s' "$safe_rel" | sed -e 's/[^A-Za-z0-9._-]/_/g' -e 's/___*/_/g' -e 's/^_//' -e 's/_$//')
		BACKUP_ARCHIVE="${safe_rel}_raw_videos.zip"
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
