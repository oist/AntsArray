#!/bin/bash -l
#SBATCH -t 0-1
#SBATCH -c 1
#SBATCH --partition=short
#SBATCH --mem=4G
#SBATCH -J chunk_fin
#SBATCH -o __JOBS_ROOT__/chunk_finalize_%j.out
#SBATCH -e __JOBS_ROOT__/chunk_finalize_%j.err
set -eo pipefail

JOBS_ROOT="__JOBS_ROOT__"
source "$JOBS_ROOT/pipeline.env"

WORKLIST="$JOBS_ROOT/aruco_worklist.txt"
echo "[INFO] building worklist -> $WORKLIST"
python3 "$LIB_DIR/worklist.py" \
	--manifest "$JOBS_ROOT/manifest.csv" \
	--out "$WORKLIST"

N=$(wc -l < "$WORKLIST")
if (( N == 0 )); then
	echo "[ERR] empty worklist" >&2
	exit 1
fi

# Resolve batch size (auto = keep array under MAX_ARRAY_TASKS).
if [[ -z "${BATCH_SIZE:-}" ]]; then
	if (( N > MAX_ARRAY_TASKS )); then
		BATCH_SIZE=$(( (N + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS ))
	else
		BATCH_SIZE=1
	fi
fi
(( BATCH_SIZE >= 1 )) || BATCH_SIZE=1

ARRAY_COUNT=$(( (N + BATCH_SIZE - 1) / BATCH_SIZE ))
UPPER=$(( ARRAY_COUNT - 1 ))

# Persist resolved batch size so array tasks see the same value.
echo "export BATCH_SIZE=$BATCH_SIZE" >> "$JOBS_ROOT/pipeline.env"
echo "[INFO] worklist=$N chunks, batch_size=$BATCH_SIZE -> array 0-$UPPER ($ARRAY_COUNT tasks)"

JID_ARUCO=""
JID_ARUCO_DC=""
JID_BRIDGE=""

# --- ArUco branch ---
if (( ONLY_SLEAP != 1 )); then
	echo "[INFO] sbatch aruco_array=0-${UPPER}%${ARUCO_CONCURRENCY}"
	JID_ARUCO=$(sbatch --parsable \
		--array=0-${UPPER}%${ARUCO_CONCURRENCY} \
		"$JOBS_ROOT/aruco_array.sbatch")
	echo "  aruco         $JID_ARUCO"
	echo "$JID_ARUCO" > "$JOBS_ROOT/jid_aruco.txt"

	echo "[INFO] sbatch aruco_datacp (single job, afterok:$JID_ARUCO)"
	JID_ARUCO_DC=$(sbatch --parsable \
		--dependency=afterok:$JID_ARUCO \
		"$JOBS_ROOT/aruco_datacp.sbatch")
	echo "  aruco_datacp  $JID_ARUCO_DC"
	echo "$JID_ARUCO_DC" > "$JOBS_ROOT/jid_aruco_datacp.txt"
fi

# --- Bridge / SLEAP branch ---
if (( ONLY_ARUCO != 1 )); then
	echo "[INFO] sbatch bridge"
	JID_BRIDGE=$(sbatch --parsable "$JOBS_ROOT/bridge.sbatch")
	echo "  bridge        $JID_BRIDGE"
	echo "$JID_BRIDGE" > "$JOBS_ROOT/jid_bridge.txt"
fi

# --- Cleanup ---
# afterany on both legs so a saion outage can't permanently wedge /flash reclamation.
CLEANUP_DEP=""
[[ -n "$JID_ARUCO_DC" ]] && CLEANUP_DEP="afterany:$JID_ARUCO_DC"
[[ -n "$JID_BRIDGE" ]] && CLEANUP_DEP="${CLEANUP_DEP:+$CLEANUP_DEP,}afterany:$JID_BRIDGE"

if [[ -n "$CLEANUP_DEP" ]]; then
	JID_CLEAN=$(sbatch --parsable --dependency="$CLEANUP_DEP" "$JOBS_ROOT/cleanup.sbatch")
else
	JID_CLEAN=$(sbatch --parsable "$JOBS_ROOT/cleanup.sbatch")
fi
echo "  cleanup       $JID_CLEAN (dep: $CLEANUP_DEP)"
echo "$JID_CLEAN" > "$JOBS_ROOT/jid_cleanup.txt"

echo "[INFO] all stages submitted"
