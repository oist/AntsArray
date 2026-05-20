#!/bin/bash -l
#SBATCH -t 0-12
#SBATCH -c 4
#SBATCH --partition=compute
#SBATCH --mem=8G
#SBATCH -J aruco
#SBATCH -o __JOBS_ROOT__/aruco_%A_%a.out
#SBATCH -e __JOBS_ROOT__/aruco_%A_%a.err
set -eo pipefail
shopt -s nullglob

JOBS_ROOT="__JOBS_ROOT__"
source "$JOBS_ROOT/pipeline.env"
source "$LIB_DIR/hosts.sh"

module load opencv/4.9.0

ARUCO_SCRIPT="${ARUCO_SCRIPT:-$HOME/run_aruco.py}"
[[ -f "$ARUCO_SCRIPT" ]] || { echo "[ERR] missing $ARUCO_SCRIPT" >&2; exit 2; }

WORKLIST="$JOBS_ROOT/aruco_worklist.txt"
BATCH_SIZE="${BATCH_SIZE:-1}"

start_idx=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
end_idx=$(( start_idx + BATCH_SIZE ))

for (( row_idx=start_idx; row_idx<end_idx; row_idx++ )); do
	row=$(sed -n "$((row_idx + 1))p" "$WORKLIST")
	[[ -n "$row" ]] || break  # ran off the end of the worklist
	vname=$(printf '%s' "$row" | cut -f1)
	chunk=$(printf '%s' "$row" | cut -f2)

	flash_dir="$FLASH_ROOT/$vname"
	aruco_dir="$FLASH_ROOT/aruco/$vname"
	mkdir -p "$aruco_dir"
	chmod 2775 "$aruco_dir"

	chunk_file="$flash_dir/${vname}_${chunk}.${CHUNK_EXT}"
	if [[ ! -s "$chunk_file" ]]; then
		echo "[ERR] missing chunk: $chunk_file" >&2
		continue
	fi

	out_h5="$aruco_dir/${vname}_${chunk}_aruco_tracks_.h5"
	bucket_h5="$DATA_DIR/${vname}_${chunk}_aruco_tracks_.h5"
	sentinel="$aruco_dir/${vname}_${chunk}.aruco.ok"

	if [[ -s "$out_h5" && -f "$sentinel" ]]; then
		echo "[SKIP] aruco already done for ${vname}_${chunk}"
		# Upload to bucket if not already there (catches earlier inline-upload misses)
		if [[ ! -f "$bucket_h5" ]]; then
			rsync_retry -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw \
				"$out_h5" "deigo:$DATA_DIR/" \
				|| echo "[WARN] inline upload of ${vname}_${chunk} failed; aruco_datacp end-of-run will retry" >&2
		fi
		continue
	fi

	echo "[$(date)] aruco on ${vname}_${chunk} (dict: $ARUCO_DICT_PATH)"
	python3 "$ARUCO_SCRIPT" \
		--video-file "$chunk_file" \
		--output-path "$aruco_dir/" \
		--custom-dict "$ARUCO_DICT_PATH" \
		--output-format h5

	touch "$sentinel"
	echo "[OK] ${vname}_${chunk}"

	# Inline upload: stream this h5 to bucket via deigo login (compute can't
	# write /bucket directly). aruco_datacp at end-of-run is the safety net.
	echo "[$(date)] uploading $(basename "$out_h5") -> bucket"
	rsync_retry -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw \
		"$out_h5" "deigo:$DATA_DIR/" \
		|| echo "[WARN] inline upload of ${vname}_${chunk} failed; aruco_datacp end-of-run will retry" >&2
done
