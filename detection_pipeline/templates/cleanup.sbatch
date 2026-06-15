#!/bin/bash -l
#SBATCH -t 0-6
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
WORKLIST="$JOBS_ROOT/aruco_worklist.txt"
SAION_INPUT_DIR="/saion_work/ReiterU/$USER/$EXP_NAME/input"
CHUNK_COPY_TIMEOUT="${CHUNK_COPY_TIMEOUT:-${SLEAP_TIMEOUT:-21600}}"   # 6h default
POLL_SECS="${CLEANUP_POLL_SECS:-300}"     # check every 5 min

if [[ ! -f "$WORKLIST" ]]; then
	echo "[WARN] no worklist; treating as nothing to wait for"
	expected=0
else
	expected=$(wc -l < "$WORKLIST")
fi

count_chunks_on_saion() {
	local n=0
	[[ -d "$SAION_INPUT_DIR" ]] || { echo 0; return; }
	while IFS=$'\t' read -r vname chunk _; do
		[[ -n "$vname" ]] || continue
		[[ -f "$SAION_INPUT_DIR/${vname}_${chunk}.${CHUNK_EXT}" ]] && n=$((n+1))
	done < "$WORKLIST"
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

if [[ -d "$FLASH_ROOT" ]]; then
	echo "[INFO] freeing $FLASH_ROOT"
	rm -rf "$FLASH_ROOT"
fi

# Try to remove parent if empty (best-effort; harmless if not)
rmdir "$(dirname "$FLASH_ROOT")" 2>/dev/null || true

echo "[OK] cleanup done"
