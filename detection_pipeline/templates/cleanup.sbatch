#!/bin/bash -l
# short assoc caps wall at 2h; -t 0-6 was rejected (AssocMaxWallDurationPerJobLimit).
#SBATCH -t 0-2
#SBATCH -c 1
#SBATCH --partition=short
#SBATCH --mem=2G
#SBATCH -J cleanup
#SBATCH -o __JOBS_ROOT__/cleanup_%j.out
#SBATCH -e __JOBS_ROOT__/cleanup_%j.err
set -eo pipefail

JOBS_ROOT="__JOBS_ROOT__"
source "$JOBS_ROOT/pipeline.env"
source "$LIB_DIR/hosts.sh"

# Layers 2+3 (deigo side): persist deigo job logs + the aruco sacct post-mortem to
# bucket before ANY scratch reclamation. Runs first so even the timeout-keep path
# below still archives. aruco/bridge are terminal by now (cleanup deps afterany on
# both). Compute can't write /bucket -> rsync via the deigo login alias.
archive_deigo_logs() {
	if [[ -f "$JOBS_ROOT/jid_aruco.txt" ]]; then
		local ja; ja=$(cat "$JOBS_ROOT/jid_aruco.txt")
		sacct -j "$ja" \
			--format=JobID,JobName,State,ExitCode,DerivedExitCode,Elapsed,Timelimit,MaxRSS,ReqMem,NodeList,Start,End,Reason%60 \
			-P > "$JOBS_ROOT/sacct_aruco_$ja.tsv" 2>/dev/null || echo "[WARN] aruco sacct failed"
	fi
	shopt -s nullglob
	ssh_retry deigo "mkdir -p '$HPC_LOGS_DIR/aruco' '$HPC_LOGS_DIR/pipeline' && { chgrp $OUTPUT_GROUP '$HPC_LOGS_DIR/aruco' '$HPC_LOGS_DIR/pipeline' 2>/dev/null; chmod 2775 '$HPC_LOGS_DIR/aruco' '$HPC_LOGS_DIR/pipeline' 2>/dev/null; true; }" || true
	local aruco_logs=( "$JOBS_ROOT"/aruco_*.out "$JOBS_ROOT"/aruco_*.err "$JOBS_ROOT"/aruco_*.status "$JOBS_ROOT"/sacct_aruco_*.tsv )
	(( ${#aruco_logs[@]} )) && { rsync_retry -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw --chown=:"$OUTPUT_GROUP" "${aruco_logs[@]}" "deigo:$HPC_LOGS_DIR/aruco/" || echo "[WARN] aruco log archive failed; streamed copies remain"; }
	local ctl_logs=( "$JOBS_ROOT"/chunk_*.out "$JOBS_ROOT"/chunk_*.err "$JOBS_ROOT"/bridge_*.out "$JOBS_ROOT"/bridge_*.err "$JOBS_ROOT"/aruco_datacp_*.out "$JOBS_ROOT"/cleanup_*.out "$JOBS_ROOT"/manifest.csv "$JOBS_ROOT"/aruco_worklist.txt "$JOBS_ROOT"/pipeline.env )
	(( ${#ctl_logs[@]} )) && { rsync_retry -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw --chown=:"$OUTPUT_GROUP" "${ctl_logs[@]}" "deigo:$HPC_LOGS_DIR/pipeline/" || echo "[WARN] pipeline log archive failed"; }
}
archive_deigo_logs || true

# /flash only needs to outlive the chunk-copy to saion, NOT saion's sleap
# inference. Saion holds its own /work copies once self-fetch (or bridge rsync)
# has happened, so /flash is safe to free as soon as all expected chunks are
# visible on the /saion_work read-only mount.
# "Expected" = the chunks saion fetches: the bridge's bucket-skip-filtered list. A
# re-run whose other chunks already have SLEAP on the bucket never sends those, so
# waiting for the whole wave timed out (20260928 block01-k96: 320 / 611). The list
# counts only if THIS run's bridge wrote it, i.e. it is newer than the bridge's
# submission record: a re-submitted wave reuses JOBS_ROOT, and a previous run's
# (smaller, maybe empty) list left by a bridge that died early or never ran would
# free chunks saion has not fetched. Otherwise wait for the whole wave as before.
# Freeing below still covers every chunk of the wave.
WORKLIST="$JOBS_ROOT/aruco_worklist.txt"
WAIT_LIST="$JOBS_ROOT/saion_stage/aruco_worklist.txt"
JID_BRIDGE="$JOBS_ROOT/jid_bridge.txt"
if [[ ! -f "$WAIT_LIST" ]] || [[ -f "$JID_BRIDGE" && ! "$WAIT_LIST" -nt "$JID_BRIDGE" ]]; then
	WAIT_LIST="$WORKLIST"
fi
SAION_INPUT_DIR="${CLEANUP_SAION_INPUT_DIR:-/saion_work/ReiterU/$USER/$EXP_NAME/input}"
CHUNK_COPY_TIMEOUT="${CHUNK_COPY_TIMEOUT:-${SLEAP_TIMEOUT:-5400}}"   # 1.5h: stays within the short 2h walltime (leaves margin for archive+rm)
POLL_SECS="${CLEANUP_POLL_SECS:-300}"     # check every 5 min

if [[ ! -f "$WAIT_LIST" ]]; then
	echo "[WARN] no worklist; treating as nothing to wait for"
	expected=0
else
	expected=$(wc -l < "$WAIT_LIST")
	echo "[INFO] chunks saion fetches: $WAIT_LIST ($expected)"
fi

# A chunk counts once saion's copy matches the /flash source in size: input/ is
# shared across runs, so the name alone also matches a stale copy of a chunk that
# has been re-cut since (or one still being copied).
count_chunks_on_saion() {
	local n=0 name
	[[ -d "$SAION_INPUT_DIR" ]] || { echo 0; return; }
	while IFS=$'\t' read -r vname chunk _; do
		[[ -n "$vname" ]] || continue
		name="${vname}_${chunk}.${CHUNK_EXT}"
		[[ -f "$SAION_INPUT_DIR/$name" ]] || continue
		if [[ -f "$FLASH_ROOT/$vname/$name" ]] &&
			[[ "$(stat -c %s "$SAION_INPUT_DIR/$name")" != "$(stat -c %s "$FLASH_ROOT/$vname/$name")" ]]; then
			continue
		fi
		n=$((n+1))
	done < "$WAIT_LIST"
	echo "$n"
}

deadline=$(( $(date +%s) + CHUNK_COPY_TIMEOUT ))
if (( expected > 0 )); then
	echo "[INFO] waiting for $expected chunks on saion ($SAION_INPUT_DIR, timeout ${CHUNK_COPY_TIMEOUT}s)"
	actual=$(count_chunks_on_saion)
	while (( actual < expected )) && (( $(date +%s) < deadline )); do
		echo "[$(date '+%H:%M:%S')] $actual / $expected chunks on saion; sleeping ${POLL_SECS}s"
		sleep "$POLL_SECS"
		actual=$(count_chunks_on_saion)
	done

	if (( actual < expected )); then
		echo "[WARN] timeout: $actual / $expected chunks on saion (path $SAION_INPUT_DIR)"
		echo "       KEEPING $FLASH_ROOT for retry."
		echo "       Causes: bridge rsync incomplete (OLD arch), or self-fetch slow (NEW arch)."
		echo "       To force-delete after manual verification: rm -rf $FLASH_ROOT"
		exit 1
	fi
	echo "[OK] all $actual / $expected chunks present on saion; safe to free /flash"
fi

# Wave-scoped reclamation. A whole-block run still frees everything, but a wave
# (CHUNK_RANGE set) must delete ONLY the chunks it was responsible for: another
# wave's array may still be reading its own chunks out of the same FLASH_ROOT,
# and a blanket rm -rf would pull them away mid-run.
free_wave_chunks() {
	local vname chunk n=0
	while IFS=$'\t' read -r vname chunk _; do
		[[ -n "$vname" && -n "$chunk" ]] || continue
		rm -f "$FLASH_ROOT/$vname/${vname}_${chunk}.${CHUNK_EXT}"
		rm -f "$FLASH_ROOT/aruco/$vname/${vname}_${chunk}"_aruco_* \
		      "$FLASH_ROOT/aruco/$vname/${vname}_${chunk}.aruco.ok"
		n=$((n+1))
	done < "$WORKLIST"
	echo "[INFO] freed $n chunks of wave $CHUNK_RANGE under $FLASH_ROOT"
	# Prune only the directories this wave emptied; -delete skips non-empty ones,
	# so a concurrent wave's files keep their parents alive.
	find "$FLASH_ROOT" -mindepth 1 -type d -empty -delete 2>/dev/null || true
	rmdir "$FLASH_ROOT" 2>/dev/null || true
}

if [[ -d "$FLASH_ROOT" ]]; then
	if [[ -n "${CHUNK_RANGE:-}" && -f "$WORKLIST" ]]; then
		free_wave_chunks
	else
		echo "[INFO] freeing $FLASH_ROOT"
		rm -rf "$FLASH_ROOT"
	fi
fi

# Try to remove parent if empty (best-effort; harmless if not)
rmdir "$(dirname "$FLASH_ROOT")" 2>/dev/null || true

echo "[OK] cleanup done"
