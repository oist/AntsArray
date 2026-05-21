#!/bin/bash -l
# saion largegpu user cap: cpu=128, gres/gpu=8, mem=1T, MaxWall=12h
# With 8 concurrent tasks (GPU-capped), each task = 16 cpus, 128 GB, 1 GPU.
#SBATCH -t 0-12
#SBATCH -c 16
#SBATCH --partition=__SAION_PARTITION__
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -J sleap
#SBATCH -o __REMOTE_JOBS__/sleap_%A_%a.out
#SBATCH -e __REMOTE_JOBS__/sleap_%A_%a.err
set -eo pipefail

source ~/.bashrc
module load __SLEAP_MODULE__

# Home is shared deigo<->saion, so the rendered deigo repo path also works on saion.
source "__HOSTS_LIB__"

REMOTE_JOBS="__REMOTE_JOBS__"
REMOTE_INPUT="__REMOTE_INPUT__"
REMOTE_OUTPUT="__REMOTE_OUTPUT__"
EXPORT_DIR="__EXPORT_DIR__"
SLEAP_RUNTIME="__SLEAP_RUNTIME__"
CHUNK_EXT="__CHUNK_EXT__"
SLEAP_MODEL_CENTROID="__SLEAP_MODEL_CENTROID__"
SLEAP_MODEL_INSTANCE="__SLEAP_MODEL_INSTANCE__"
SKIP_TRT_EXPORT=__SKIP_TRT_EXPORT__
DEIGO_FLASH_SAION_PREFIX="__DEIGO_FLASH_SAION_PREFIX__"   # /deigo_flash/.../<exp> mount on saion
DATA_DIR="__DATA_DIR__"                      # bucket path; reached via "saion" alias (login has write)
SLEAP_BATCH_SIZE="${SLEAP_BATCH_SIZE:-16}"   # per-frame inference batch (engine max from export)
BATCH_SIZE=__BATCH_SIZE__                    # chunks per array task

WORKLIST="$REMOTE_JOBS/aruco_worklist.txt"

start_idx=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
end_idx=$(( start_idx + BATCH_SIZE ))

mkdir -p "$REMOTE_INPUT" "$REMOTE_OUTPUT"

for (( row_idx=start_idx; row_idx<end_idx; row_idx++ )); do
	row=$(sed -n "$((row_idx + 1))p" "$WORKLIST")
	[[ -n "$row" ]] || break
	vname=$(printf '%s' "$row" | cut -f1)
	chunk=$(printf '%s' "$row" | cut -f2)

	# Source: deigo's /flash visible on saion compute as /deigo_flash (read-only).
	# We copy to /work so saion owns an isolated copy (no risk if deigo cleanup
	# fires while sleap is still running).
	src_remote="$DEIGO_FLASH_SAION_PREFIX/$vname/${vname}_${chunk}.${CHUNK_EXT}"
	input="$REMOTE_INPUT/${vname}_${chunk}.${CHUNK_EXT}"
	out_slp="$REMOTE_OUTPUT/${vname}_${chunk}.slp"

	if [[ -f "$out_slp" ]]; then
		echo "[SKIP] $out_slp already exists"
		continue
	fi

	# Self-fetch if not already on /work
	if [[ ! -s "$input" ]]; then
		if [[ ! -s "$src_remote" ]]; then
			echo "[ERR] source missing on /deigo_flash: $src_remote" >&2
			continue
		fi
		echo "[$(date)] cp $src_remote -> $input"
		if ! cp "$src_remote" "$input"; then
			echo "[ERR] cp failed for ${vname}_${chunk}" >&2
			continue
		fi
	fi

	echo "[$(date)] sleap on ${vname}_${chunk} (runtime=$SLEAP_RUNTIME, skip_trt=$SKIP_TRT_EXPORT)"

	if (( SKIP_TRT_EXPORT == 0 )) && [[ "$SLEAP_RUNTIME" != "pytorch" ]]; then
		# Exported-model path: ONNX or TensorRT
		sleap-nn predict "$EXPORT_DIR" "$input" \
			-o "$out_slp" \
			--runtime "$SLEAP_RUNTIME" \
			--batch-size "$SLEAP_BATCH_SIZE" \
			--device cuda
	else
		# Fallback: legacy PyTorch path via raw model dirs
		sleap-nn track \
			-i "$input" \
			-m "$SLEAP_MODEL_CENTROID" \
			-m "$SLEAP_MODEL_INSTANCE" \
			-o "$out_slp" \
			-d cuda \
			--batch_size "$SLEAP_BATCH_SIZE" \
			--no_empty_frames
	fi

	echo "[OK] ${vname}_${chunk}"

	# Inline upload: stream SLP to bucket via ssh saion (saion compute can't
	# write /bucket directly). sleap_datacp at end-of-run is the safety net.
	echo "[$(date)] uploading ${vname}_${chunk}.slp -> bucket"
	rsync_retry -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw \
		"$out_slp" "saion:$DATA_DIR/" \
		|| echo "[WARN] inline upload of ${vname}_${chunk} failed; sleap_datacp end-of-run will retry" >&2
done
