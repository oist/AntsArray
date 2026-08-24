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
  --chunk-range A-B                 Process ONLY chunk indices A..B (inclusive) --
                                    one "wave" of a long block instead of all of
                                    it. Chunk indices are stable for a given
                                    --chunk-sec, so a window always names the same
                                    span of wall-clock; run waves back to back to
                                    keep the array under the partition's submit
                                    cap and to bound how much /flash is held at
                                    once. Each wave is recorded in
                                    <exp>/data/PIPELINE_STATE.json.
                                    default: empty = the whole block.
                                    NOTE: --chunk-sec must be a whole number of
                                    GOPs for waves; chunk.sbatch verifies the
                                    first chunk of every seeked wave and fails
                                    rather than emit misaligned chunks.

Processing contract:
  --new-processing-run              Archive the block's existing
                                    data/PIPELINE_STATE.json and start a fresh
                                    contract. Needed only to deliberately
                                    re-process a block under different settings
                                    (new chunking, new models). Existing outputs
                                    in data/ are NOT deleted -- move them aside
                                    yourself first, or the two runs' outputs will
                                    be interleaved under identical filenames.
  --aruco-force-recompute           Recompute every ArUco chunk in the wave, even
                                    ones already complete on the bucket. Mirrors
                                    --sleap-force-recompute; use it when changing
                                    detector parameters.

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

Chunk prefetch (saion-side staging ahead of the GPU array):
  --prefetch-partition NAME         saion partition that stages chunks from the
                                    /deigo_flash cross-mount onto /work before
                                    the SLEAP task needs them, so the ~1.3 GB
                                    copy stops eating a scarce GPU allocation.
                                    One prefetch task per SLEAP task, paired with
                                    --dependency=aftercorr, so each chunk is
                                    handed over as soon as IT is staged instead
                                    of waiting for the whole staging array.
                                    saion caps resources per partition and has no
                                    submit-count limit, so this spends none of
                                    largegpu's / short-a100's GPU quota.
                                    'off' = every SLEAP task copies its own chunk
                                    (the behaviour before this existed).
                                    default: gpu
  --prefetch-concurrency N          array %N cap for the prefetch. Keep it >= the
                                    sleap concurrency so staging stays ahead.
                                    default: 32
  --prefetch-cpus N                 cpus per prefetch task. default: 2
  --prefetch-mem SIZE               mem per prefetch task.  default: 4G
  --prefetch-wall D-HH              walltime per prefetch task. default: 0-2

Waves:
  --force-submit                    Submit even when the concurrent-wave guard
                                    objects. The guard refuses two live runs that
                                    share a --jobs-root (they would race on the
                                    same pipeline.env and worklist), and warns
                                    when the requested --chunk-range overlaps a
                                    range already claimed in PIPELINE_STATE.json.

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

SLEAP bucket skip:
  --sleap-force-recompute           Recompute every chunk's SLEAP output, even
                                    ones already complete on the bucket. Default
                                    is a bucket-aware skip: a re-run only redoes
                                    chunks whose .slp/_sleap_data.h5 are missing
                                    or were produced under a different chunking
                                    (expected_frames verified). USE THIS when
                                    re-running to apply a NEW model -- otherwise
                                    the existing outputs are skipped and kept.

Phase isolation (for testing):
  --only-chunk                      Stop after chunking
  --only-aruco                      Skip sleap branch
  --only-sleap                      Skip aruco branch (still chunks)
  --only-backup                     Build manifest + submit ONLY the raw-video
                                    backup job (no chunk/aruco/sleap). Handy for
                                    testing the backup or re-backing-up a block.

Roots:
  --jobs-root PATH                  default: /flash/ReiterU/$USER/jobs/<exp>
                                    ... plus /wave_<A>-<B> when --chunk-range is
                                    given, so overlapping waves of one block never
                                    share pipeline.env or a worklist. An explicit
                                    path is used verbatim (no wave suffix).
  --flash-root PATH                 default: /flash/ReiterU/$USER/<exp>
                                    Block-level on purpose: a chunk filename
                                    carries its index, so waves cannot collide,
                                    and cleanup frees only its own wave's chunks.

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

Notifications:
  --notify-email ADDR               Email ADDR when detection completes (both
                                    aruco+sleap on bucket; sent by the login-side
                                    poller) and on any job failure (Slurm
                                    --mail-type=FAIL on every submitted job;
                                    arrays notify once per array). Forwarded to
                                    the tracking submit script with --run-tracking.
                                    default: $NOTIFY_EMAIL env, else off

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
CHUNK_RANGE=""            # "A-B" inclusive; empty = the whole block (one wave)
WAVE_SLUG=""              # derived from CHUNK_RANGE; namespaces this wave's control dirs
FORCE_SUBMIT=0            # override the concurrent-wave guard
NEW_PROCESSING_RUN=0      # archive the block's contract and start a fresh one
ARUCO_FORCE_RECOMPUTE=0   # redo every aruco chunk, ignoring bucket-complete ones
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
# Chunk staging ahead of the GPU array; "off" disables it and every sleap task
# copies its own chunk, as before. See --prefetch-partition in the usage text.
PREFETCH_PARTITION=gpu
PREFETCH_CONCURRENCY=32
PREFETCH_CPUS=2
PREFETCH_MEM=4G
PREFETCH_WALL=0-2
DATACP_CONCURRENCY=4
BATCH_SIZE=1         # default: one chunk per array task (set "" to auto-size under MAX_ARRAY_TASKS)
MAX_ARRAY_TASKS=500
OUTPUT_GROUP=reiteruni   # group owner for shared bucket outputs (chgrp + setgid on created dirs)
SLEAP_FORCE_RECOMPUTE=0   # default: bucket-aware skip in bridge; set 1 to redo every chunk
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
# Email notifications (empty = off): Slurm FAIL mail on every job + a
# detection-complete email from the login-side poller (track_trigger.sh).
NOTIFY_EMAIL="${NOTIFY_EMAIL:-}"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--dir) DIR="$2"; shift 2 ;;
		--aruco-dict) ARUCO_DICT="$2"; shift 2 ;;
		--aruco-params) ARUCO_EXTRA_ARGS="$2"; shift 2 ;;
		--chunk-sec) CHUNK_SEC="$2"; shift 2 ;;
		--chunk-range) CHUNK_RANGE="$2"; shift 2 ;;
		--force-submit) FORCE_SUBMIT=1; shift ;;
		--new-processing-run) NEW_PROCESSING_RUN=1; shift ;;
		--aruco-force-recompute) ARUCO_FORCE_RECOMPUTE=1; shift ;;
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
		--prefetch-partition) PREFETCH_PARTITION="$2"; shift 2 ;;
		--prefetch-concurrency) PREFETCH_CONCURRENCY="$2"; shift 2 ;;
		--prefetch-cpus) PREFETCH_CPUS="$2"; shift 2 ;;
		--prefetch-mem) PREFETCH_MEM="$2"; shift 2 ;;
		--prefetch-wall) PREFETCH_WALL="$2"; shift 2 ;;
		--datacp-concurrency) DATACP_CONCURRENCY="$2"; shift 2 ;;
		--batch-size) BATCH_SIZE="$2"; shift 2 ;;
		--max-array-tasks) MAX_ARRAY_TASKS="$2"; shift 2 ;;
		--group) OUTPUT_GROUP="$2"; shift 2 ;;
		--sleap-force-recompute) SLEAP_FORCE_RECOMPUTE=1; shift ;;
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
		--notify-email) NOTIFY_EMAIL="$2"; shift 2 ;;
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
if [[ -n "$CHUNK_RANGE" ]]; then
	[[ "$CHUNK_RANGE" =~ ^[0-9]+-[0-9]+$ ]] || {
		echo "[ERR] --chunk-range must be 'A-B' with 0 <= A <= B, got '$CHUNK_RANGE'" >&2; exit 2; }
	(( ${CHUNK_RANGE%%-*} <= ${CHUNK_RANGE##*-} )) || {
		echo "[ERR] --chunk-range start is past its end: '$CHUNK_RANGE'" >&2; exit 2; }
	echo "[INFO] wave run: chunk indices ${CHUNK_RANGE} only (chunk_sec=${CHUNK_SEC}s -> "\
	     "$(( ${CHUNK_RANGE%%-*} * CHUNK_SEC ))s..$(( (${CHUNK_RANGE##*-} + 1) * CHUNK_SEC ))s into each video)"
	# Namespaces every MUTABLE control file this wave owns, on both clusters:
	# /flash jobs dir (pipeline.env, worklist, jid_*.txt, rendered templates) and
	# saion's $SAION_WORK_ROOT/jobs (worklist + rendered arrays). Those files are
	# re-read AT RUN TIME by jobs that are already queued -- prefetch, predict,
	# datacp and the verify gate all index the worklist by row -- so a second wave
	# writing them under the block name would hand a running array a different
	# wave's rows. Every task would still report COMPLETED; the chunks would just
	# be the wrong ones. The chunk/output files themselves stay block-level: their
	# names carry the chunk index, so waves cannot collide there, and sharing them
	# is what lets prefetch skip an already-staged chunk.
	WAVE_SLUG="wave_${CHUNK_RANGE}"
fi

# Auto-select the SLEAP inference path from the model dir contents: sleap-nn
# checkpoints (best.ckpt) can be TRT/ONNX-exported; legacy TF models
# (best_model.h5 only, no best.ckpt) cannot, so fall back to 'sleap-nn track'
# automatically. Skips if the user already forced --skip-trt-export or pytorch.
if (( SKIP_TRT_EXPORT == 0 )) && [[ "$SLEAP_RUNTIME" != "pytorch" ]]; then
	_legacy_model=0
	[[ -n "$SLEAP_MODEL_CENTROID" && -d "$SLEAP_MODEL_CENTROID" && ! -f "$SLEAP_MODEL_CENTROID/best.ckpt" ]] && _legacy_model=1
	[[ -n "$SLEAP_MODEL_INSTANCE"  && -d "$SLEAP_MODEL_INSTANCE"  && ! -f "$SLEAP_MODEL_INSTANCE/best.ckpt"  ]] && _legacy_model=1
	if (( _legacy_model == 1 )); then
		echo "[INFO] SLEAP model dir has no best.ckpt (legacy TF model); auto-enabling --skip-trt-export ('sleap-nn track' fallback)" >&2
		SKIP_TRT_EXPORT=1
	fi
fi

# Roots. Only the jobs dir is wave-scoped -- see the WAVE_SLUG comment above.
# An explicit --jobs-root is honoured verbatim: the caller has already chosen the
# namespace, and silently appending to it would surprise a rescue run.
JOBS_ROOT="${JOBS_ROOT:-/flash/ReiterU/$USER/jobs/$EXP_NAME${WAVE_SLUG:+/$WAVE_SLUG}}"
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

# --- Concurrent-wave guard ----------------------------------------------------
# Wave scoping (WAVE_SLUG) makes DIFFERENT ranges safe to overlap. What it cannot
# make safe is two live runs that land in the SAME jobs dir: the second would
# rewrite pipeline.env and aruco_worklist.txt under the first one's running array,
# which reads them by row at task start. So refuse that outright, and warn on a
# range some earlier wave already claimed (usually a typo, occasionally a
# deliberate recompute). Both are overridable with --force-submit.
#
# Jobs are identified by the jid_*.txt files the previous run left in that dir,
# not by job name: every deigo template uses a generic -J (bridge, cleanup,
# chunk_fin), so squeue alone cannot tell this block's jobs from another's. Each
# cluster's queue is fetched once -- a per-jid `squeue -j` on a finished id exits
# non-zero and would spend ssh_retry's whole backoff ladder on every file.
guard_concurrent_wave() {
	local live_deigo live_saion q_deigo q_saion f base jid
	[[ -d "$JOBS_ROOT" ]] || return 0
	shopt -s nullglob
	local jid_files=( "$JOBS_ROOT"/jid_*.txt )
	shopt -u nullglob
	(( ${#jid_files[@]} )) || return 0

	q_deigo=$(squeue -h -u "$USER" -o '%i' 2>/dev/null || true)
	# Best-effort: no saion login (or a network blip) must not block a submission,
	# but say so, because the saion half of the check then did not happen.
	q_saion=$(ssh -x -oBatchMode=yes -oStrictHostKeyChecking=no \
		-oConnectTimeout=15 saion "squeue -h -u \$USER -o '%i'" 2>/dev/null) || {
		echo "[WARN] could not reach saion to check for a live sleap array; guard covers deigo only" >&2
		q_saion=""
	}

	live_deigo=""; live_saion=""
	for f in "${jid_files[@]}"; do
		base=$(basename "$f")
		jid=$(tr -d '[:space:]' < "$f")
		[[ -n "$jid" ]] || continue
		# Array tasks show as <jid>_<task>, so anchor on the id plus a boundary.
		local pat='^'"$jid"'(_|$)'
		if [[ "$base" == jid_saion_* ]]; then
			if grep -qE "$pat" <<<"$q_saion"; then live_saion+=" $base=$jid"; fi
		else
			if grep -qE "$pat" <<<"$q_deigo"; then live_deigo+=" $base=$jid"; fi
		fi
	done

	if [[ -n "$live_deigo$live_saion" ]]; then
		echo "[ERR] a run is still live in this jobs dir: $JOBS_ROOT" >&2
		[[ -n "$live_deigo" ]] && echo "        deigo:$live_deigo" >&2
		[[ -n "$live_saion" ]] && echo "        saion:$live_saion" >&2
		echo "      Submitting now would rewrite its pipeline.env and worklist while its" >&2
		echo "      array is still reading them, silently pairing tasks to the wrong chunks." >&2
		if [[ -z "$CHUNK_RANGE" ]]; then
			echo "      Give this run its own --chunk-range (waves get their own jobs dir)," >&2
			echo "      or its own --jobs-root, or wait for the run above to finish." >&2
		else
			echo "      This is the same wave ($WAVE_SLUG) as the live run. Pick a different" >&2
			echo "      --chunk-range, or wait for it to finish." >&2
		fi
		return 1
	fi
	return 0
}

if ! guard_concurrent_wave; then
	if (( FORCE_SUBMIT == 1 )); then
		echo "[WARN] --force-submit: proceeding into a live jobs dir anyway" >&2
	else
		echo "[ERR] refusing to submit; pass --force-submit to override" >&2
		exit 2
	fi
fi

# Overlap against the block ledger. Advisory only: a deliberate recompute of an
# already-processed window is legitimate (that is what --sleap-force-recompute
# and --new-processing-run are for), and the ledger records intent rather than
# completion, so this must not be able to block a rescue run.
if [[ -d "$DATA_DIR" ]]; then
	_overlap=$(python3 "$LIB_DIR/pipeline_state.py" check-range \
		--data-dir "$DATA_DIR" --range "$CHUNK_RANGE" 2>&1) || true
	if [[ -n "$_overlap" ]]; then
		echo "[WARN] this range overlaps a wave already recorded in PIPELINE_STATE.json:" >&2
		echo "$_overlap" | sed 's/^/       /' >&2
		echo "       Re-running it recomputes those chunks and overwrites their outputs." >&2
	fi
fi

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
	if [[ -n "$CHUNK_RANGE" ]]; then
		# The poller gates on the BLOCK's declared total, not this wave's worklist
		# (track_trigger.sh), so it simply waits until every wave has landed --
		# tracking cannot run on part of a block in any case, because map_combine
		# drops absent cameras silently rather than shortening its output.
		# Harmless on any single wave; the thing to avoid is one poller per wave.
		echo "[INFO] --run-tracking waits for ALL $CHUNK_RANGE-style waves to finish"
		echo "       (it gates on the block's declared chunk total, not this wave)."
		echo "       Pass it on ONE wave only -- a second poller would submit tracking twice."
	fi
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
export CHUNK_RANGE="$CHUNK_RANGE"
export WAVE_SLUG="$WAVE_SLUG"
export ARUCO_FORCE_RECOMPUTE="$ARUCO_FORCE_RECOMPUTE"
export ARUCO_DICT_PATH="$ARUCO_DICT_PATH"
export ARUCO_EXTRA_ARGS="$ARUCO_EXTRA_ARGS"
export ARUCO_CONCURRENCY="$ARUCO_CONCURRENCY"
export SLEAP_CONCURRENCY="$SLEAP_CONCURRENCY"
export SLEAP_CPUS="$SLEAP_CPUS"
export SLEAP_MEM="$SLEAP_MEM"
export SLEAP_WALL="$SLEAP_WALL"
export PREFETCH_PARTITION="$PREFETCH_PARTITION"
export PREFETCH_CONCURRENCY="$PREFETCH_CONCURRENCY"
export PREFETCH_CPUS="$PREFETCH_CPUS"
export PREFETCH_MEM="$PREFETCH_MEM"
export PREFETCH_WALL="$PREFETCH_WALL"
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
export SLEAP_FORCE_RECOMPUTE="$SLEAP_FORCE_RECOMPUTE"
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
export NOTIFY_EMAIL="$NOTIFY_EMAIL"
EOF
echo "[INFO] env file: $ENV_FILE"

# Slurm-native failure mail on every job submitted here (chunk_finalize adds the
# same args to the stages it fans out; arrays notify once per array, not per task).
MAIL_ARGS=()
[[ -n "$NOTIFY_EMAIL" ]] && MAIL_ARGS=(--mail-type=FAIL --mail-user="$NOTIFY_EMAIL")

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

# --- Processing contract ------------------------------------------------------
# Declare (first run) or verify (every later run) how this block is processed.
# A chunk's filename encodes its INDEX, not its settings, so re-running a block
# under a different --chunk-sec or a different model silently overwrites part of
# data/ with content that no longer lines up with the part it did not overwrite.
# That is unrecoverable after the fact -- every file stays individually valid --
# so a disagreement stops the run here rather than at the first output.
# See lib/pipeline_state.py. Skipped for --only-backup, which produces no data/.
if (( ONLY_BACKUP != 1 )); then
	STATE_LEGS="aruco,sleap"
	(( ONLY_ARUCO == 1 )) && STATE_LEGS="aruco"
	(( ONLY_SLEAP == 1 )) && STATE_LEGS="sleap"

	# Only pass keys this run actually exercises: an --only-sleap run supplies no
	# aruco settings, and absence means "not exercised", never "clear it".
	STATE_SET=(--set "chunk_sec=$CHUNK_SEC" --set "chunk_ext=$CHUNK_EXT")
	if (( ONLY_SLEAP != 1 )); then
		STATE_SET+=(--set "aruco_dict=$ARUCO_DICT_PATH"
		            --set "aruco_params=$ARUCO_EXTRA_ARGS"
		            --set "aruco_script=$(basename "${ARUCO_SCRIPT:-run_aruco_mp.py}")")
	fi
	if (( ONLY_ARUCO != 1 )); then
		STATE_SET+=(--set "sleap_model_centroid=$SLEAP_MODEL_CENTROID"
		            --set "sleap_model_instance=$SLEAP_MODEL_INSTANCE"
		            --set "sleap_module=$SLEAP_MODULE"
		            --set "sleap_runtime=$SLEAP_RUNTIME"
		            --set "saion_partition=$SAION_PARTITION")
	fi
	NEW_RUN_ARG=()
	(( NEW_PROCESSING_RUN == 1 )) && NEW_RUN_ARG=(--new-run)

	if ! python3 "$LIB_DIR/pipeline_state.py" sync \
			--data-dir "$DATA_DIR" \
			--manifest "$MANIFEST" \
			--block-dir "$DIR" \
			--legs "$STATE_LEGS" \
			"${STATE_SET[@]}" "${NEW_RUN_ARG[@]}"; then
		echo "[ERR] refusing to submit: this run conflicts with the block's recorded" >&2
		echo "      processing contract (see the diff above). Nothing was submitted." >&2
		exit 2
	fi
fi

# Render every template once (single placeholder: __JOBS_ROOT__)
echo "[INFO] rendering templates -> $JOBS_ROOT/"
for t in chunk.sbatch chunk_finalize.sbatch backup.sbatch aruco_array.sbatch aruco_datacp.sbatch bridge.sbatch cleanup.sbatch; do
	sed "s#__JOBS_ROOT__#$JOBS_ROOT#g" "$TEMPLATES_DIR/$t" > "$JOBS_ROOT/$t"
	chmod +x "$JOBS_ROOT/$t"
done

# --only-backup: build manifest + submit ONLY the raw-video backup job.
if (( ONLY_BACKUP )); then
	(( RUN_BACKUP == 1 )) || { echo "[ERR] --only-backup conflicts with --no-backup" >&2; exit 2; }
	JID_BACKUP=$(sbatch_retry backup --partition="$BACKUP_PARTITION" "${MAIL_ARGS[@]}" "$JOBS_ROOT/backup.sbatch")
	echo "  backup        $JID_BACKUP"
	echo "$JID_BACKUP" > "$JOBS_ROOT/jid_backup.txt"
	echo "[INFO] --only-backup: submitted backup job only -> ${BACKUP_ARCHIVE_PATH:-(disabled)}"
	exit 0
fi

# Submit chunk array
CHUNK_UPPER=$(( N_VIDEOS - 1 ))
echo "[INFO] sbatch chunk_array=0-${CHUNK_UPPER}"
JID_CHUNK=$(sbatch_retry chunk --array=0-${CHUNK_UPPER} "${MAIL_ARGS[@]}" "$JOBS_ROOT/chunk.sbatch")
echo "  chunk         $JID_CHUNK"
echo "$JID_CHUNK" > "$JOBS_ROOT/jid_chunk.txt"

if (( ONLY_CHUNK )); then
	echo "[INFO] --only-chunk: stopping after chunk submission"
	exit 0
fi

# Submit chunk_finalize (builds worklist + submits aruco / bridge / cleanup)
JID_CHUNK_FIN=$(sbatch_retry chunk_fin --dependency=afterok:$JID_CHUNK "${MAIL_ARGS[@]}" "$JOBS_ROOT/chunk_finalize.sbatch")
echo "  chunk_finalize $JID_CHUNK_FIN (dep: $JID_CHUNK)"
echo "$JID_CHUNK_FIN" > "$JOBS_ROOT/jid_chunk_fin.txt"

# Optional: launch the login-side tracking auto-trigger (nohup poller). It waits for
# detection outputs to appear in the bucket, then emails NOTIFY_EMAIL (detection
# complete) and/or submits colony tracking for this block. Mirrors the tracking
# transfer watcher's login-nohup pattern; survives logout. The notify-only launch
# is restricted to full runs: an --only-* run never completes both modalities, so
# the poller would just sit until its 48h deadline and mail a spurious timeout.
# A --chunk-range wave is the same situation for the same reason -- the poller
# gates on the block's DECLARED total, which one wave cannot reach -- so exclude
# it too. Slurm's own --mail-type=FAIL is unaffected: that rides on each
# submitted job via MAIL_ARGS, not on this poller, so failure mail still arrives.
# With --run-tracking the poller IS wanted on a wave: waiting for the whole block
# is precisely its job.
if (( RUN_TRACKING == 1 )) || { [[ -n "$NOTIFY_EMAIL" ]] && [[ -z "$CHUNK_RANGE" ]] \
		&& (( ONLY_ARUCO == 0 && ONLY_SLEAP == 0 )); }; then
	mkdir -p "$HPC_LOGS_DIR/pipeline"
	sed "s#__JOBS_ROOT__#$JOBS_ROOT#g" "$TEMPLATES_DIR/track_trigger.sh" > "$JOBS_ROOT/track_trigger.sh"
	chmod +x "$JOBS_ROOT/track_trigger.sh"
	nohup "$JOBS_ROOT/track_trigger.sh" >> "$HPC_LOGS_DIR/pipeline/track_trigger.log" 2>&1 &
	echo "  track_trigger  PID $! (nohup; tracking=$RUN_TRACKING notify=${NOTIFY_EMAIL:-off}; log: $HPC_LOGS_DIR/pipeline/track_trigger.log)"
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
