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
source "$LIB_DIR/hosts.sh"   # sbatch_retry: a transient slurmctld timeout must not abort the DAG

# Slurm-native failure mail on every fanned-out stage (arrays notify once per array).
MAIL_ARGS=()
[[ -n "${NOTIFY_EMAIL:-}" ]] && MAIL_ARGS=(--mail-type=FAIL --mail-user="$NOTIFY_EMAIL")

# The wave's canonical worklist: every chunk this run is responsible for.
# CHUNK_RANGE (empty = the whole block) selects one window of chunk indices; a
# chunk's index is a pure function of (video, CHUNK_SEC), so the same window
# always names the same span of wall-clock. cleanup, bridge and track_trigger all
# read THIS file -- the ArUco leg gets its own already-done-filtered copy below.
WORKLIST="$JOBS_ROOT/aruco_worklist.txt"
echo "[INFO] building worklist -> $WORKLIST${CHUNK_RANGE:+ (chunk-range $CHUNK_RANGE)}"
python3 "$LIB_DIR/worklist.py" \
	--manifest "$JOBS_ROOT/manifest.csv" \
	--chunk-sec "$CHUNK_SEC" \
	${CHUNK_RANGE:+--chunk-range "$CHUNK_RANGE"} \
	--out "$WORKLIST"

N_WAVE=$(wc -l < "$WORKLIST")
if (( N_WAVE == 0 )); then
	echo "[ERR] empty worklist" >&2
	exit 1
fi

# Resolve batch size (auto = keep array under MAX_ARRAY_TASKS). Sized from the
# full wave rather than the filtered ArUco list, because BATCH_SIZE is shared
# with the saion sleap array, whose own filtered list can be longer.
if [[ -z "${BATCH_SIZE:-}" ]]; then
	if (( N_WAVE > MAX_ARRAY_TASKS )); then
		BATCH_SIZE=$(( (N_WAVE + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS ))
	else
		BATCH_SIZE=1
	fi
fi
(( BATCH_SIZE >= 1 )) || BATCH_SIZE=1

# Persist resolved batch size so array tasks see the same value. Drop any earlier
# BATCH_SIZE line first: this used to append blindly, so re-running the same wave
# left two definitions and the file grew a line each time. Harmless while the last
# one written happens to be the right one, but it turns the env file into a log
# rather than a record, and hand-editing the wrong line then silently does
# nothing. Safe to rewrite here because every consumer sources this file at task
# start and none has been submitted yet -- the sbatch calls are all below.
sed -i '/^export BATCH_SIZE=/d' "$JOBS_ROOT/pipeline.env"
echo "export BATCH_SIZE=$BATCH_SIZE" >> "$JOBS_ROOT/pipeline.env"

# --- ArUco leg: skip chunks already complete on the bucket --------------------
# Mirrors what bridge.sbatch does for SLEAP. Without it a re-run recomputes every
# ArUco chunk once cleanup has freed /flash, because aruco_array.sbatch's own
# skip can only see flash-local output.
ARUCO_WORKLIST="$JOBS_ROOT/aruco_worklist.run.txt"
if [[ "${ARUCO_FORCE_RECOMPUTE:-0}" == "1" ]]; then
	echo "[INFO] ARUCO_FORCE_RECOMPUTE=1: running every chunk in the wave"
	cp "$WORKLIST" "$ARUCO_WORKLIST"
elif ! python3 "$SCRIPTS_DIR/filter_done_aruco.py" \
		--worklist "$WORKLIST" --data-dir "$DATA_DIR" --out "$ARUCO_WORKLIST"; then
	echo "[WARN] aruco bucket-skip filter failed; running every chunk in the wave" >&2
	cp "$WORKLIST" "$ARUCO_WORKLIST"
fi
echo "export ARUCO_RUN_WORKLIST=\"$ARUCO_WORKLIST\"" >> "$JOBS_ROOT/pipeline.env"

N=$(wc -l < "$ARUCO_WORKLIST")
ARRAY_COUNT=$(( (N + BATCH_SIZE - 1) / BATCH_SIZE ))
UPPER=$(( ARRAY_COUNT - 1 ))
echo "[INFO] wave=$N_WAVE chunks, aruco to run=$N, batch_size=$BATCH_SIZE -> array 0-$UPPER ($ARRAY_COUNT tasks)"

JID_ARUCO=""
JID_ARUCO_DC=""
JID_BRIDGE=""
JID_BACKUP=""

# --- Bucket Backup branch ---
if (( RUN_BACKUP == 1 )); then
	echo "[INFO] sbatch backup (single job, stable archive: $BACKUP_ARCHIVE_PATH)"
	JID_BACKUP=$(sbatch_retry backup \
		--partition="$BACKUP_PARTITION" \
		"${MAIL_ARGS[@]}" \
		"$JOBS_ROOT/backup.sbatch")
	echo "  backup        $JID_BACKUP"
	echo "$JID_BACKUP" > "$JOBS_ROOT/jid_backup.txt"
fi

# --- ArUco branch ---
if (( ONLY_SLEAP != 1 )) && (( N == 0 )); then
	echo "[INFO] every ArUco chunk in this wave is already complete on the bucket; "\
	     "skipping the aruco array and its datacp"
fi
if (( ONLY_SLEAP != 1 )) && (( N > 0 )); then
	echo "[INFO] sbatch aruco_array=0-${UPPER}%${ARUCO_CONCURRENCY}"
	JID_ARUCO=$(sbatch_retry aruco \
		--array=0-${UPPER}%${ARUCO_CONCURRENCY} \
		"${MAIL_ARGS[@]}" \
		"$JOBS_ROOT/aruco_array.sbatch")
	echo "  aruco         $JID_ARUCO"
	echo "$JID_ARUCO" > "$JOBS_ROOT/jid_aruco.txt"

	echo "[INFO] sbatch aruco_datacp (single job, afterok:$JID_ARUCO)"
	JID_ARUCO_DC=$(sbatch_retry aruco_dc \
		--dependency=afterok:$JID_ARUCO \
		"${MAIL_ARGS[@]}" \
		"$JOBS_ROOT/aruco_datacp.sbatch")
	echo "  aruco_datacp  $JID_ARUCO_DC"
	echo "$JID_ARUCO_DC" > "$JOBS_ROOT/jid_aruco_datacp.txt"
fi

# --- Bridge / SLEAP branch ---
if (( ONLY_ARUCO != 1 )); then
	echo "[INFO] sbatch bridge"
	JID_BRIDGE=$(sbatch_retry bridge "${MAIL_ARGS[@]}" "$JOBS_ROOT/bridge.sbatch")
	echo "  bridge        $JID_BRIDGE"
	echo "$JID_BRIDGE" > "$JOBS_ROOT/jid_bridge.txt"
fi

# --- Cleanup ---
# afterany on both legs so a saion outage can't permanently wedge /flash reclamation.
CLEANUP_DEP=""
[[ -n "$JID_ARUCO_DC" ]] && CLEANUP_DEP="afterany:$JID_ARUCO_DC"
[[ -n "$JID_BRIDGE" ]] && CLEANUP_DEP="${CLEANUP_DEP:+$CLEANUP_DEP,}afterany:$JID_BRIDGE"

if [[ -n "$CLEANUP_DEP" ]]; then
	JID_CLEAN=$(sbatch_retry cleanup --dependency="$CLEANUP_DEP" "${MAIL_ARGS[@]}" "$JOBS_ROOT/cleanup.sbatch")
else
	JID_CLEAN=$(sbatch_retry cleanup "${MAIL_ARGS[@]}" "$JOBS_ROOT/cleanup.sbatch")
fi
echo "  cleanup       $JID_CLEAN (dep: $CLEANUP_DEP)"
echo "$JID_CLEAN" > "$JOBS_ROOT/jid_cleanup.txt"

# --- Record the wave in the block's processing ledger -------------------------
# Intent only: which chunk window was claimed, by which jobs, when. Completion is
# never written from here -- coverage is derived from what is actually in data/,
# so a killed job or a lost poller cannot leave the ledger claiming work that
# does not exist (see lib/pipeline_state.py).
#
# Via ssh to the deigo login because this job runs on a compute partition, and
# compute nodes cannot write /bucket -- the same reason the upload stages ssh
# back to the login alias.
_wave_jids="chunk_finalize=${SLURM_JOB_ID:-?}${JID_ARUCO:+,aruco=$JID_ARUCO}${JID_BRIDGE:+,bridge=$JID_BRIDGE}"
if ssh_retry deigo "python3 '$LIB_DIR/pipeline_state.py' add-wave \
		--data-dir '$DATA_DIR' \
		--range '${CHUNK_RANGE:-}' \
		--rows '$N_WAVE' \
		--batch-size '$BATCH_SIZE' \
		--jids '$_wave_jids'" >/dev/null; then
	echo "[INFO] wave recorded in $DATA_DIR/PIPELINE_STATE.json"
else
	# Not fatal: the ledger is bookkeeping, and the outputs themselves remain the
	# source of truth. But an unrecorded wave leaves a gap in the coverage report,
	# so say so loudly rather than letting it pass unnoticed.
	echo "[WARN] could not record this wave in $DATA_DIR/PIPELINE_STATE.json." >&2
	echo "       Detection is unaffected. To record it by hand from a login node:" >&2
	echo "       python3 $LIB_DIR/pipeline_state.py add-wave --data-dir '$DATA_DIR' \\" >&2
	echo "           --range '${CHUNK_RANGE:-}' --rows $N_WAVE --batch-size $BATCH_SIZE" >&2
fi

echo "[INFO] all stages submitted"
