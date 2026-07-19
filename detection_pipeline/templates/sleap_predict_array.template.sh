#!/bin/bash -l
# Per-task resources are rendered by bridge.sbatch from pipeline.sh knobs so one
# template serves both A100 partitions (same SM80 hardware, gpu23-26):
#   largegpu   (8 GPU cap):  -c 16 --mem=128G -t 0-12  -> 8 concurrent saturate cpu/mem/gpu
#   short-a100 (32 GPU cap): -c 8  --mem=64G  -t 0-2   -> 32 concurrent; one chunk/task
#                                                         (BATCH_SIZE=1) fits well inside 2h;
#                                                         a run preempted past the 1h
#                                                         non-preemptible window is REQUEUEd
#SBATCH -t __SLEAP_WALL__
#SBATCH -c __SLEAP_CPUS__
#SBATCH --partition=__SAION_PARTITION__
#SBATCH --mem=__SLEAP_MEM__
#SBATCH --gres=gpu:1
#SBATCH -J sleap
#SBATCH -o __REMOTE_JOBS__/sleap_%A_%a.out
#SBATCH -e __REMOTE_JOBS__/sleap_%A_%a.err
# Deliver SIGTERM ~60s before the walltime SIGKILL so the log-stream trap can flush.
#SBATCH --signal=TERM@60
set -eo pipefail

source ~/.bashrc
module load __SLEAP_MODULE__

# Home is shared deigo<->saion, so the rendered deigo repo path also works on saion.
source "__HOSTS_LIB__"
source "__SHIP_LIB__"

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
OUTPUT_GROUP="__OUTPUT_GROUP__"              # group owner for shared bucket outputs
SLEAP_BATCH_SIZE="${SLEAP_BATCH_SIZE:-8}"   # per-frame inference batch (must be <= engine max profile batch)
BATCH_SIZE=__BATCH_SIZE__                    # chunks per array task
SCRIPTS_DIR="__SCRIPTS_DIR__"                # pipeline scripts dir (sleap2h5.py, sleap2csv.py)

WORKLIST="$REMOTE_JOBS/aruco_worklist.txt"

# Layer 1: stream this task's Slurm log to bucket (via the saion login, which has
# bucket write -- compute nodes do not) while it runs, so a walltime/scancel/node
# death still leaves a diagnosable log under hpc_logs/sleap/. saion_cleanup also
# archives logs before rm, but streaming does not depend on any downstream job.
HPC_LOGS_DIR="__HPC_LOGS_DIR__"
_LA="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}"; _la="${SLURM_ARRAY_TASK_ID:-0}"
log_stream_start "saion" "$HPC_LOGS_DIR/sleap" \
	"$REMOTE_JOBS/sleap_${_LA}_${_la}.status" \
	"$REMOTE_JOBS/sleap_${_LA}_${_la}.out" \
	"$REMOTE_JOBS/sleap_${_LA}_${_la}.err"
install_log_traps

start_idx=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
end_idx=$(( start_idx + BATCH_SIZE ))

mkdir -p "$REMOTE_INPUT" "$REMOTE_OUTPUT"

for (( row_idx=start_idx; row_idx<end_idx; row_idx++ )); do
	row=$(sed -n "$((row_idx + 1))p" "$WORKLIST")
	[[ -n "$row" ]] || break
	vname=$(printf '%s' "$row" | cut -f1)
	chunk=$(printf '%s' "$row" | cut -f2)
	# Column 3 (expected_frames) caps inference so sleap-nn 0.2 doesn't walk past
	# the actual decodable end on _000 chunks (ffmpeg -c copy propagates the source
	# video's full frame count into the first segment's container metadata).
	# Empty (old 2-col worklist) = no cap.
	n_frames=$(printf '%s' "$row" | cut -f3)

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

	_t0=$SECONDS
	if (( SKIP_TRT_EXPORT == 0 )) && [[ "$SLEAP_RUNTIME" != "pytorch" ]]; then
		# Exported-model path: ONNX or TensorRT
		sleap-nn predict "$EXPORT_DIR" "$input" \
			-o "$out_slp" \
			--runtime "$SLEAP_RUNTIME" \
			--batch-size "$SLEAP_BATCH_SIZE" \
			${n_frames:+--n-frames "$n_frames"} \
			--device cuda
	else
		# Fallback: legacy PyTorch path via raw model dirs.
		# Note: sleap-nn track does not accept --n-frames; on _000 chunks with
		# inflated container metadata this path will fail with the same
		# IndexError. Workaround there is to ffmpeg-trim the chunk first.
		sleap-nn track \
			-i "$input" \
			-m "$SLEAP_MODEL_CENTROID" \
			-m "$SLEAP_MODEL_INSTANCE" \
			-o "$out_slp" \
			-d cuda \
			--batch_size "$SLEAP_BATCH_SIZE" \
			--no_empty_frames
	fi
	_dt=$(( SECONDS - _t0 ))

	# Layer 4: greppable per-chunk throughput line -> durable fps history in bucket.
	# n_frames is worklist col 3 (expected_frames); empty on legacy 2-col worklists.
	if [[ "${n_frames:-}" =~ ^[0-9]+$ ]] && (( _dt > 0 )); then
		awk -v f="$n_frames" -v s="$_dt" -v c="${vname}_${chunk}" \
			'BEGIN{ printf "[FPS] %s frames=%d elapsed=%ds fps=%.2f\n", c, f, s, f/s }'
	else
		echo "[FPS] ${vname}_${chunk} frames=${n_frames:-NA} elapsed=${_dt}s fps=NA"
	fi

	echo "[OK] ${vname}_${chunk}"

	# Post-process (slp -> h5, then rsync .slp + .h5 to bucket) runs in the
	# BACKGROUND so the next chunk's GPU work overlaps with this chunk's CPU
	# work + upload. With ~30 fps inference and ~few-min post-processing, at
	# most ~1 background job is in flight at a time. The end-of-loop `wait`
	# below ensures none are orphaned when the task exits. UV_TOOL_DIR is set
	# by `module load sleap-nn/...`; pandas + h5py are in that venv.
	# sleap_datacp end-of-run remains the safety net for anything missed here.
	(
		out_h5="$REMOTE_OUTPUT/${vname}_${chunk}_sleap_data.h5"
		echo "[$(date)] [bg] slp -> h5 ${vname}_${chunk}"
		"$UV_TOOL_DIR/sleap-nn/bin/python" "$SCRIPTS_DIR/sleap2h5.py" "$out_slp" "$REMOTE_OUTPUT" \
			|| echo "[WARN] sleap2h5 failed for ${vname}_${chunk}; .slp will still upload" >&2

		upload_files=( "$out_slp" )
		[[ -s "$out_h5" ]] && upload_files+=( "$out_h5" )
		echo "[$(date)] [bg] uploading ${vname}_${chunk} (.slp + .h5) -> bucket"
		rsync_retry -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw --chown=:"$OUTPUT_GROUP" \
			"${upload_files[@]}" "saion:$DATA_DIR/" \
			|| echo "[WARN] inline upload of ${vname}_${chunk} failed; sleap_datacp end-of-run will retry" >&2
	) &
done
# Wait for any backgrounded post-processing (slp2h5 + rsync) before exiting.
wait
echo "[$(date)] all background uploads finished; task done"
