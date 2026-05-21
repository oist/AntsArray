#!/bin/bash -l
#SBATCH -t 0-4
#SBATCH -c 2
#SBATCH --partition=datacp
#SBATCH --mem=4G
#SBATCH -J aruco_dc
#SBATCH -o __JOBS_ROOT__/aruco_datacp_%j.out
#SBATCH -e __JOBS_ROOT__/aruco_datacp_%j.err
set -eo pipefail
umask 0002

JOBS_ROOT="__JOBS_ROOT__"
source "$JOBS_ROOT/pipeline.env"

# Single-job datacp: scan the whole worklist and rsync every aruco h5 that
# exists on /flash to $DATA_DIR. Submitting as one job (not an array) keeps
# us under deigo's AssocGrpSubmitJobsLimit when there are many chunks.

WORKLIST="$JOBS_ROOT/aruco_worklist.txt"

mkdir -p "$DATA_DIR"
chmod 2775 "$DATA_DIR" 2>/dev/null || true

uploaded=0
missing=0
echo "[$(date)] starting aruco datacp -> $DATA_DIR/"

while IFS=$'\t' read -r vname chunk; do
	for suffix in aruco_tracks aruco_detections; do
		src="$FLASH_ROOT/aruco/$vname/${vname}_${chunk}_${suffix}.h5"
		if [[ ! -s "$src" ]]; then
			missing=$((missing+1))
			continue
		fi
		rsync -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw "$src" "$DATA_DIR/"
		uploaded=$((uploaded+1))
	done
done < "$WORKLIST"

echo "[$(date)] done: uploaded=$uploaded missing=$missing total_worklist=$(wc -l < "$WORKLIST")"
