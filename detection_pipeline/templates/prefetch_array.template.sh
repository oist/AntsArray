#!/bin/bash -l
# prefetch_array — stage chunks onto saion /work ahead of the SLEAP array.
#
# Why this exists
# ---------------
# SLEAP tasks self-fetch their own chunk from the /deigo_flash cross-mount at
# task start (sleap_predict_array.template.sh). That copy is ~1.3 GB and happens
# INSIDE a scarce GPU allocation -- on short-a100 the whole task budget is 2 h.
# When GPU allocation is tight the GPUs are the queue bottleneck while the
# CPU-ish partitions (gpu, intel, test-gpu) sit idle, so the copy can be moved
# off the GPU entirely and overlapped with the GPU queue wait.
#
# Task IDs pair 1:1 with the SLEAP array
# --------------------------------------
# bridge submits the SLEAP array with `--dependency=aftercorr:<this job>`, which
# starts SLEAP task N only after prefetch task N. Both arrays are sized from the
# SAME filtered worklist with the SAME BATCH_SIZE and both index it by row, so
# task N handles the same rows in both. Reading a different worklist here would
# silently pair the wrong chunks -- the filtered list is the one bridge uploads
# to $REMOTE_JOBS/aruco_worklist.txt, and it is what both arrays read.
#
# This task ALWAYS exits 0
# ------------------------
# aftercorr is a per-task afterok: a non-zero exit here CANCELS the corresponding
# SLEAP task, and that chunk would then be missing from a block that otherwise
# looks finished. Not hypothetical -- an unsatisfiable afterok is exactly what
# cancelled sleap_datacp and stranded 73 GB on saion /work in the 20260810 wave-2
# run. So a failed copy is logged and swallowed: SLEAP still carries its own
# self-fetch, and the only cost of a miss is that one task paying for its own
# copy, which is precisely the behaviour we had before this job existed.
#SBATCH -t __PREFETCH_WALL__
#SBATCH -c __PREFETCH_CPUS__
#SBATCH --partition=__PREFETCH_PARTITION__
#SBATCH --mem=__PREFETCH_MEM__
#SBATCH -J prefetch
#SBATCH -o __REMOTE_JOBS__/prefetch_%A_%a.out
#SBATCH -e __REMOTE_JOBS__/prefetch_%A_%a.err
set -uo pipefail          # NOT -e: a failed copy must never fail the task
shopt -s nullglob

REMOTE_JOBS="__REMOTE_JOBS__"
REMOTE_INPUT="__REMOTE_INPUT__"
DEIGO_FLASH_SAION_PREFIX="__DEIGO_FLASH_SAION_PREFIX__"
CHUNK_EXT="__CHUNK_EXT__"
BATCH_SIZE="__BATCH_SIZE__"
WORKLIST="$REMOTE_JOBS/aruco_worklist.txt"

if [[ ! -r "$WORKLIST" ]]; then
	echo "[WARN] no readable worklist at $WORKLIST; nothing to prefetch" >&2
	exit 0
fi
mkdir -p "$REMOTE_INPUT" || true

start_idx=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
end_idx=$(( start_idx + BATCH_SIZE ))

copied=0; skipped=0; failed=0
for (( row_idx=start_idx; row_idx<end_idx; row_idx++ )); do
	row=$(sed -n "$((row_idx + 1))p" "$WORKLIST")
	[[ -n "$row" ]] || break          # ran off the end of the worklist
	vname=$(printf '%s' "$row" | cut -f1)
	chunk=$(printf '%s' "$row" | cut -f2)
	[[ -n "$vname" && -n "$chunk" ]] || continue

	src="$DEIGO_FLASH_SAION_PREFIX/$vname/${vname}_${chunk}.${CHUNK_EXT}"
	dst="$REMOTE_INPUT/${vname}_${chunk}.${CHUNK_EXT}"
	tmp="$dst.part.$$"

	if [[ ! -s "$src" ]]; then
		echo "[WARN] source missing on /deigo_flash: $src (sleap will retry)" >&2
		failed=$((failed+1))
		continue
	fi

	# Size-equality, not mere presence: a chunk left half-written by a killed
	# task is non-empty, and `-s` alone would call it done.
	src_sz=$(stat -c%s "$src" 2>/dev/null || echo 0)
	dst_sz=$(stat -c%s "$dst" 2>/dev/null || echo -1)
	if [[ "$src_sz" == "$dst_sz" ]]; then
		echo "[SKIP] ${vname}_${chunk} already staged ($dst_sz bytes)"
		skipped=$((skipped+1))
		continue
	fi

	# Copy to a temp name, then rename into place. aftercorr already keeps the
	# SLEAP task off this chunk until we finish, but a requeue of THIS task can
	# still leave a partial file behind, and rename is atomic within /work.
	_t0=$SECONDS
	if cp "$src" "$tmp" && mv -f "$tmp" "$dst"; then
		echo "[OK] ${vname}_${chunk} staged in $((SECONDS-_t0))s ($src_sz bytes)"
		copied=$((copied+1))
	else
		echo "[WARN] copy failed for ${vname}_${chunk}; sleap will self-fetch" >&2
		rm -f "$tmp"
		failed=$((failed+1))
	fi
done

echo "[DONE] prefetch task $SLURM_ARRAY_TASK_ID: copied=$copied skipped=$skipped failed=$failed"
# Always 0 -- see the header. A miss costs one task its own copy, never a chunk.
exit 0
