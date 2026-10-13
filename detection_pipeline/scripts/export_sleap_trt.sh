#!/bin/bash -l
# export_sleap_trt.sh — one-shot TRT/ONNX export of a SLEAP model set on saion.
#
# Invoked from saion-login by bridge.sbatch via ssh. If not already on a GPU
# node, re-execs itself under srun on largegpu and blocks until export is done.
set -eo pipefail

usage() {
	cat >&2 <<EOT
Usage: $0 --centroid <dir> --instance <dir> --out <dir>
          [--runtime tensorrt|onnx|both] [--partition NAME] [--max-instances N]
          [--anchor-fill]

TRT engines are GPU-architecture-specific. Build on the same partition the
predict jobs will run on, otherwise the engine will refuse to load.

--max-instances is the most animals the engine keeps per frame. It is baked into
the exported centroid graph, so it can only be changed by exporting again; an
existing export with a different cap is refused, never reused.

--anchor-fill exports through sleap_nn_export_anchor_fill.py: where the instance
model leaves the anchor (tag) node below threshold, the engine puts the centroid
there (Score_node stays 0) instead of NaN, and anchor_fill.json records it. An
existing export with the other setting is refused, never reused.

Defaults:
  --runtime        tensorrt
  --partition      largegpu   (A100 SM80)
  --max-instances  \$MAX_INSTANCES or 128
EOT
	exit 1
}

CENTROID=""
INSTANCE=""
OUT=""
RUNTIME="tensorrt"
PARTITION="largegpu"
SLEAP_MODULE="${SLEAP_MODULE:-sleap-nn/0.2.0}"
# sleap-nn's own default is 20, which silently truncated every colony nest camera.
# Cap sweep on 20260928 cam09 (densest nest camera, 2026-10-05): 48 and 64 still hit
# the cap on 98-100 % of frames; 96 finds 70 ants/frame (max 82) with no frame at the
# cap and the same detections as 128, 31 % faster. So 96.
MAX_INSTANCES="${MAX_INSTANCES:-96}"
ANCHOR_FILL=0
SELF_DIR="$(dirname "$(readlink -f "$0")")"
# The centered-instance model runs on up to cap x batch crops at once, and TensorRT cannot
# build above a crop budget that depends on the model's crop size (192 at 640, 140 at 768;
# lib/engine_crops.sh). So the batch follows the cap and the crop unless MAX_BATCH is set.
source "$SELF_DIR/../lib/engine_crops.sh"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--centroid)  CENTROID="$2"; shift 2 ;;
		--instance)  INSTANCE="$2"; shift 2 ;;
		--out)       OUT="$2"; shift 2 ;;
		--runtime)   RUNTIME="$2"; shift 2 ;;
		--partition) PARTITION="$2"; shift 2 ;;
		--max-instances) MAX_INSTANCES="$2"; shift 2 ;;
		--anchor-fill) ANCHOR_FILL=1; shift ;;
		-h|--help)   usage ;;
		*) echo "[ERR] unknown arg: $1" >&2; usage ;;
	esac
done

[[ -d "$CENTROID" ]] || { echo "[ERR] --centroid not a dir: $CENTROID" >&2; exit 2; }
[[ -d "$INSTANCE" ]] || { echo "[ERR] --instance not a dir: $INSTANCE" >&2; exit 2; }
[[ -n "$OUT" ]]      || usage
CROP="$(instance_crop_size "$INSTANCE")" || {
	echo "[ERR] no fixed crop_size in $INSTANCE/training_config.yaml; the engine's crop budget depends on it" >&2; exit 2; }
MAX_ENGINE_CROPS="$(max_engine_crops "$CROP")" || {
	echo "[ERR] crop $CROP has no measured engine limit; measure which cap x batch builds at this crop" \
	     "and add it to lib/engine_crops.sh" >&2; exit 2; }
[[ "$MAX_INSTANCES" =~ ^[1-9][0-9]*$ ]] || {
	echo "[ERR] --max-instances must be a positive integer, got '$MAX_INSTANCES'" >&2; exit 2; }
(( MAX_INSTANCES <= MAX_ENGINE_CROPS )) || {
	echo "[ERR] --max-instances $MAX_INSTANCES exceeds the $MAX_ENGINE_CROPS-crop engine limit even at batch 1" >&2; exit 2; }
if [[ -z "${MAX_BATCH:-}" ]]; then
	MAX_BATCH=$(( MAX_ENGINE_CROPS / MAX_INSTANCES ))
	(( MAX_BATCH > 8 )) && MAX_BATCH=8
fi
[[ "$MAX_BATCH" =~ ^[1-9][0-9]*$ ]] || {
	echo "[ERR] MAX_BATCH must be a positive integer, got '$MAX_BATCH'" >&2; exit 2; }
(( MAX_INSTANCES * MAX_BATCH <= MAX_ENGINE_CROPS )) || {
	echo "[ERR] max_instances $MAX_INSTANCES x batch $MAX_BATCH = $(( MAX_INSTANCES * MAX_BATCH )) crops;" \
	     "TensorRT cannot build above $MAX_ENGINE_CROPS -- lower MAX_BATCH" >&2; exit 2; }

# A field the export was built with, from its metadata (empty if none is recorded).
exported_meta() {
	local f
	for f in "$1/model.trt.metadata.json" "$1/export_metadata.json"; do
		[[ -f "$f" ]] || continue
		sed -n "s/.*\"$2\": *\\([0-9][0-9]*\\).*/\\1/p" "$f" | head -1
		return 0
	done
}

case "$RUNTIME" in
	tensorrt) marker="$OUT/model.trt" ;;
	onnx)     marker="$OUT/model.onnx" ;;
	both)     marker="$OUT/model.trt" ;;
	*) echo "[ERR] unknown --runtime $RUNTIME (tensorrt|onnx|both)" >&2; exit 2 ;;
esac

FILL_RECORD="$OUT/anchor_fill.json"
fill_args=()
(( ANCHOR_FILL == 0 )) || fill_args=(--anchor-fill)

if [[ -f "$marker" ]]; then
	have_k="$(exported_meta "$OUT" max_instances)"
	have_b="$(exported_meta "$OUT" max_batch_size)"
	have_fill=0
	[[ -f "$FILL_RECORD" ]] && have_fill=1
	if [[ "$have_k" == "$MAX_INSTANCES" && "$have_b" == "$MAX_BATCH" && "$have_fill" == "$ANCHOR_FILL" ]]; then
		echo "[OK] export already at $OUT (marker $marker, max_instances $have_k, max_batch_size $have_b," \
		     "anchor fill $have_fill)"
		exit 0
	fi
	echo "[ERR] $OUT already holds an export with max_instances=${have_k:-unrecorded}," \
	     "max_batch_size=${have_b:-unrecorded}, anchor fill $have_fill;" \
	     "asked for $MAX_INSTANCES / $MAX_BATCH / anchor fill $ANCHOR_FILL." >&2
	echo "      Export to another --out; the pipeline's engine cache key carries all three" \
	     "(__k<N>b<B>__, __afill__)." >&2
	exit 2
fi

mkdir -p "$OUT"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
	echo "[INFO] re-execing under srun on partition=$PARTITION"
	MAX_BATCH="$MAX_BATCH" exec srun -p "$PARTITION" -c 8 --mem=64G --gres=gpu:1 -t 0-2 \
		bash "$(readlink -f "$0")" \
		--centroid "$CENTROID" --instance "$INSTANCE" --out "$OUT" \
		--runtime "$RUNTIME" --partition "$PARTITION" --max-instances "$MAX_INSTANCES" "${fill_args[@]}"
fi

# Batch jobs need the site modules, not interactive conda/mamba hooks. Sourcing
# ~/.bashrc under set -e can exit before SLEAP loads when a user hook is missing.
if ! type module >/dev/null 2>&1; then
	source /etc/profile
fi
export PYTHONNOUSERSITE=1
module use /apps/unit/ReiterU/.modulefiles
module load "$SLEAP_MODULE"

echo "[INFO] sleap-nn export"
echo "  centroid: $CENTROID"
echo "  instance: $INSTANCE"
echo "  out:      $OUT"
echo "  runtime:  $RUNTIME"
echo "  max_instances: $MAX_INSTANCES"
echo "  max_batch_size: $MAX_BATCH"
echo "  anchor fill: $ANCHOR_FILL"

case "$RUNTIME" in
	tensorrt) fmt=tensorrt ;;
	onnx)     fmt=onnx ;;
	both)     fmt=both ;;
esac

# MAX_BATCH (set above from the cap) is the engine's max profile batch: the inference
# batch (SLEAP_BATCH_SIZE) must not exceed it. Speed barely depends on it -- per frame
# ~9.6 ms + 0.63 ms x cap on A100, the same for sparse and dense cameras, because the
# engine runs every crop slot whether or not an animal fills it.
export_args=("$CENTROID" "$INSTANCE" -o "$OUT" -f "$fmt" --precision fp16
             --max-batch-size "$MAX_BATCH" --max-instances "$MAX_INSTANCES" --device cuda)
if (( ANCHOR_FILL == 1 )); then
	# The sleap-nn module's own interpreter, so the patch loads the same sleap_nn.
	sleap_py="${SLEAP_NN_PYTHON:-$(dirname "$(readlink -f "$(command -v sleap-nn)")")/python}"
	"$sleap_py" "$SELF_DIR/sleap_nn_export_anchor_fill.py" export "${export_args[@]}"
	[[ -f "$FILL_RECORD" ]] || {
		echo "[ERR] the anchor-fill export wrote no $FILL_RECORD" >&2; exit 3; }
else
	sleap-nn export "${export_args[@]}"
fi

have="$(exported_meta "$OUT" max_instances)"
if [[ "$have" != "$MAX_INSTANCES" ]]; then
	echo "[ERR] export metadata records max_instances=${have:-nothing}, asked for $MAX_INSTANCES" >&2
	exit 3
fi
echo "[OK] exported to $OUT (max_instances $have, anchor fill $ANCHOR_FILL)"
ls -lh "$OUT"
