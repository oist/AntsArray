#!/bin/bash -l
# export_sleap_trt.sh — one-shot TRT/ONNX export of a SLEAP model set on saion.
#
# Invoked from saion-login by bridge.sbatch via ssh. If not already on a GPU
# node, re-execs itself under srun on largegpu and blocks until export is done.
set -eo pipefail

usage() {
	cat >&2 <<EOT
Usage: $0 --centroid <dir> --instance <dir> --out <dir>
          [--runtime tensorrt|onnx|both] [--partition NAME]

TRT engines are GPU-architecture-specific. Build on the same partition the
predict jobs will run on, otherwise the engine will refuse to load.

Defaults:
  --runtime    tensorrt
  --partition  largegpu   (A100 SM80)
EOT
	exit 1
}

CENTROID=""
INSTANCE=""
OUT=""
RUNTIME="tensorrt"
PARTITION="largegpu"
SLEAP_MODULE="${SLEAP_MODULE:-sleap-nn/0.2.0}"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--centroid)  CENTROID="$2"; shift 2 ;;
		--instance)  INSTANCE="$2"; shift 2 ;;
		--out)       OUT="$2"; shift 2 ;;
		--runtime)   RUNTIME="$2"; shift 2 ;;
		--partition) PARTITION="$2"; shift 2 ;;
		-h|--help)   usage ;;
		*) echo "[ERR] unknown arg: $1" >&2; usage ;;
	esac
done

[[ -d "$CENTROID" ]] || { echo "[ERR] --centroid not a dir: $CENTROID" >&2; exit 2; }
[[ -d "$INSTANCE" ]] || { echo "[ERR] --instance not a dir: $INSTANCE" >&2; exit 2; }
[[ -n "$OUT" ]]      || usage

mkdir -p "$OUT"

case "$RUNTIME" in
	tensorrt) marker="$OUT/model.trt" ;;
	onnx)     marker="$OUT/model.onnx" ;;
	both)     marker="$OUT/model.trt" ;;
	*) echo "[ERR] unknown --runtime $RUNTIME (tensorrt|onnx|both)" >&2; exit 2 ;;
esac

if [[ -f "$marker" ]]; then
	echo "[OK] export already at $OUT (marker $marker)"
	exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
	echo "[INFO] re-execing under srun on partition=$PARTITION"
	exec srun -p "$PARTITION" -c 8 --mem=64G --gres=gpu:1 -t 0-2 \
		bash "$(readlink -f "$0")" \
		--centroid "$CENTROID" --instance "$INSTANCE" --out "$OUT" \
		--runtime "$RUNTIME" --partition "$PARTITION"
fi

source ~/.bashrc
module load "$SLEAP_MODULE"

echo "[INFO] sleap-nn export"
echo "  centroid: $CENTROID"
echo "  instance: $INSTANCE"
echo "  out:      $OUT"
echo "  runtime:  $RUNTIME"

case "$RUNTIME" in
	tensorrt) fmt=tensorrt ;;
	onnx)     fmt=onnx ;;
	both)     fmt=both ;;
esac

MAX_BATCH="${MAX_BATCH:-8}"    # override with MAX_BATCH=N before invoking
# Note: sleap-nn's TRT export bakes a 2x h/w margin into the max profile, so
# the engine's activation memory scales as ~4x_resolution * batch_size.
# At 4024x3036 / fp16 with default 2.1 GB workspace, MAX_BATCH=8 builds reliably;
# MAX_BATCH=16 intermittently fails autotuning ("insufficient memory ... no tactics
# to implement ... GridSample") on contended nodes. Keep MAX_BATCH >= SLEAP_BATCH_SIZE
# (the inference batch, default 8); raise both together only if 8 is too slow.
sleap-nn export "$CENTROID" "$INSTANCE" \
	-o "$OUT" \
	-f "$fmt" \
	--precision fp16 \
	--max-batch-size "$MAX_BATCH" \
	--device cuda

echo "[OK] exported to $OUT"
ls -lh "$OUT"
