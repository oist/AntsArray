#!/bin/bash -l
#SBATCH -J tag_roster_map
#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 0-02
# One chunk of a block's ArUco detections -> panorama, split left/right, with NO frequency filter
# (--min_instance_frame_frac 0) so rarely seen tags stay visible. Submitted by tag_roster.sh, which
# passes BLOCK HMATS PANO MAP_COMBINE PY N_CAMS [X_THRESHOLD] via --export; array index = chunk.
set -euo pipefail
: "${BLOCK:?}" "${HMATS:?}" "${PANO:?}" "${MAP_COMBINE:?}" "${PY:?}" "${N_CAMS:?}"
CHUNK=$(printf %03d "${SLURM_ARRAY_TASK_ID:?run as an array}")
echo "[$(date '+%F %T')] chunk $CHUNK of $BLOCK on $(hostname)"
n=$( (ls "$BLOCK"/data/cam*_"${CHUNK}"_aruco_detections.h5 2>/dev/null || true) | wc -l)
(( n == N_CAMS )) || { echo "[ERR] chunk $CHUNK has $n of $N_CAMS camera ArUco files"; exit 3; }
split=()
[[ -n "${X_THRESHOLD:-}" ]] && split=(--x_threshold "$X_THRESHOLD")
"$PY" "$MAP_COMBINE" --mode aruco --chunk "$CHUNK" --hmats "$HMATS" --data_dir "$BLOCK/data" \
	--outdir "$PANO" --min_instance_frame_frac 0 "${split[@]}"
echo "[$(date '+%F %T')] done chunk $CHUNK"
