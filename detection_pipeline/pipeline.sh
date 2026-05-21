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

Chunking:
  --chunk-sec N                     Chunk duration in seconds. default: 7200 (2h)
  --chunk-ext {mkv|mp4|avi}         Output chunk container. default: mkv

SLEAP runtime:
  --sleap-runtime {tensorrt|onnx|pytorch}  default: tensorrt
  --skip-trt-export                 Use 'sleap-nn track' fallback (raw model dirs)
  --saion-partition NAME            default: largegpu
  --sleap-module NAME               saion module name. default: sleap-nn/0.2.0

Concurrency:
  --aruco-concurrency N             default: 16
  --sleap-concurrency N             default: 8
  --datacp-concurrency N            default: 4

Batching (minimize submitted job count):
  --batch-size N                    chunks per array task. default: auto
                                    auto = ceil(total_chunks / --max-array-tasks)
  --max-array-tasks N               cap when auto-sizing batch. default: 500

Phase isolation (for testing):
  --only-chunk                      Stop after chunking
  --only-aruco                      Skip sleap branch
  --only-sleap                      Skip aruco branch (still chunks)

Roots:
  --jobs-root PATH                  default: /flash/ReiterU/$USER/jobs/<exp>
  --flash-root PATH                 default: /flash/ReiterU/$USER/<exp>

Other:
  -h, --help                        Show this help
EOT
	exit 1
}

# Defaults
DIR=""
ARUCO_DICT="A"
CHUNK_SEC=7200
CHUNK_EXT=mkv
SLEAP_MODEL_CENTROID=""
SLEAP_MODEL_INSTANCE=""
SLEAP_RUNTIME=tensorrt
SKIP_TRT_EXPORT=0
SAION_PARTITION=largegpu
SLEAP_MODULE="sleap-nn/0.2.0"
ARUCO_CONCURRENCY=400   # under deigo compute cap (cpu=2000/4=500); leaves ~20% headroom
SLEAP_CONCURRENCY=8     # bounded by saion largegpu having only 4 A100 nodes anyway
DATACP_CONCURRENCY=4
BATCH_SIZE=""        # "" => auto
MAX_ARRAY_TASKS=500
ONLY_CHUNK=0
ONLY_ARUCO=0
ONLY_SLEAP=0
JOBS_ROOT=""
FLASH_ROOT=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--dir) DIR="$2"; shift 2 ;;
		--aruco-dict) ARUCO_DICT="$2"; shift 2 ;;
		--chunk-sec) CHUNK_SEC="$2"; shift 2 ;;
		--chunk-ext) CHUNK_EXT="$2"; shift 2 ;;
		--sleap-model-centroid) SLEAP_MODEL_CENTROID="$2"; shift 2 ;;
		--sleap-model-instance) SLEAP_MODEL_INSTANCE="$2"; shift 2 ;;
		--sleap-runtime) SLEAP_RUNTIME="$2"; shift 2 ;;
		--skip-trt-export) SKIP_TRT_EXPORT=1; shift ;;
		--saion-partition) SAION_PARTITION="$2"; shift 2 ;;
		--sleap-module) SLEAP_MODULE="$2"; shift 2 ;;
		--aruco-concurrency) ARUCO_CONCURRENCY="$2"; shift 2 ;;
		--sleap-concurrency) SLEAP_CONCURRENCY="$2"; shift 2 ;;
		--datacp-concurrency) DATACP_CONCURRENCY="$2"; shift 2 ;;
		--batch-size) BATCH_SIZE="$2"; shift 2 ;;
		--max-array-tasks) MAX_ARRAY_TASKS="$2"; shift 2 ;;
		--only-chunk) ONLY_CHUNK=1; shift ;;
		--only-aruco) ONLY_ARUCO=1; shift ;;
		--only-sleap) ONLY_SLEAP=1; shift ;;
		--jobs-root) JOBS_ROOT="$2"; shift 2 ;;
		--flash-root) FLASH_ROOT="$2"; shift 2 ;;
		-h|--help) usage ;;
		*) echo "[ERR] unknown arg: $1" >&2; usage ;;
	esac
done

[[ -d "$DIR" ]] || { echo "[ERR] --dir is required and must exist" >&2; usage; }
DIR=$(readlink -f "$DIR")
EXP_NAME=$(basename "$DIR")

# Validate model dirs unless we're only chunking or only doing aruco
if (( ONLY_CHUNK != 1 && ONLY_ARUCO != 1 )); then
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
if (( ONLY_SLEAP != 1 )); then
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
ENV_FILE="$JOBS_ROOT/pipeline.env"

mkdir -p "$JOBS_ROOT" "$FLASH_ROOT" "$DATA_DIR"
chmod 2775 "$JOBS_ROOT" "$FLASH_ROOT" 2>/dev/null || true

cat > "$ENV_FILE" <<EOF
# Auto-generated by pipeline.sh at $(date)
export EXP_NAME="$EXP_NAME"
export DATA_DIR="$DATA_DIR"
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
export ARUCO_CONCURRENCY="$ARUCO_CONCURRENCY"
export SLEAP_CONCURRENCY="$SLEAP_CONCURRENCY"
export DATACP_CONCURRENCY="$DATACP_CONCURRENCY"
export BATCH_SIZE="$BATCH_SIZE"
export MAX_ARRAY_TASKS="$MAX_ARRAY_TASKS"
export SAION_PARTITION="$SAION_PARTITION"
export SLEAP_MODULE="$SLEAP_MODULE"
export SLEAP_MODEL_CENTROID="$SLEAP_MODEL_CENTROID"
export SLEAP_MODEL_INSTANCE="$SLEAP_MODEL_INSTANCE"
export SLEAP_RUNTIME="$SLEAP_RUNTIME"
export SKIP_TRT_EXPORT="$SKIP_TRT_EXPORT"
export ONLY_ARUCO="$ONLY_ARUCO"
export ONLY_SLEAP="$ONLY_SLEAP"
EOF
echo "[INFO] env file: $ENV_FILE"

# Build manifest
MANIFEST="$JOBS_ROOT/manifest.csv"
echo "[INFO] building manifest -> $MANIFEST"
python3 "$LIB_DIR/manifest.py" --dir "$DIR" --out "$MANIFEST" --chunk-sec "$CHUNK_SEC"

N_VIDEOS=$(($(wc -l < "$MANIFEST") - 1))
if (( N_VIDEOS <= 0 )); then
	echo "[ERR] manifest has no grid videos" >&2
	exit 2
fi
echo "[INFO] $N_VIDEOS grid videos discovered"

# Render every template once (single placeholder: __JOBS_ROOT__)
echo "[INFO] rendering templates -> $JOBS_ROOT/"
for t in chunk.sbatch chunk_finalize.sbatch aruco_array.sbatch aruco_datacp.sbatch bridge.sbatch cleanup.sbatch; do
	sed "s#__JOBS_ROOT__#$JOBS_ROOT#g" "$TEMPLATES_DIR/$t" > "$JOBS_ROOT/$t"
	chmod +x "$JOBS_ROOT/$t"
done

# Submit chunk array
CHUNK_UPPER=$(( N_VIDEOS - 1 ))
echo "[INFO] sbatch chunk_array=0-${CHUNK_UPPER}"
JID_CHUNK=$(sbatch --parsable --array=0-${CHUNK_UPPER} "$JOBS_ROOT/chunk.sbatch")
echo "  chunk         $JID_CHUNK"
echo "$JID_CHUNK" > "$JOBS_ROOT/jid_chunk.txt"

if (( ONLY_CHUNK )); then
	echo "[INFO] --only-chunk: stopping after chunk submission"
	exit 0
fi

# Submit chunk_finalize (builds worklist + submits aruco / bridge / cleanup)
JID_CHUNK_FIN=$(sbatch --parsable --dependency=afterok:$JID_CHUNK "$JOBS_ROOT/chunk_finalize.sbatch")
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
EOF
