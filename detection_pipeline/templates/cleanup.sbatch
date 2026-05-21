#!/bin/bash -l
#SBATCH -t 1-0
#SBATCH -c 1
#SBATCH --partition=compute
#SBATCH --mem=2G
#SBATCH -J cleanup
#SBATCH -o __JOBS_ROOT__/cleanup_%j.out
#SBATCH -e __JOBS_ROOT__/cleanup_%j.err
set -eo pipefail

JOBS_ROOT="__JOBS_ROOT__"
source "$JOBS_ROOT/pipeline.env"

# Saion compute self-fetches chunks from /deigo_flash, so deigo's /flash must
# stay readable until saion sleap is fully done. We poll the bucket for SLP
# files (shared FS, no ssh needed) until all chunks have landed or we time out.
WORKLIST="$JOBS_ROOT/aruco_worklist.txt"
SLEAP_TIMEOUT="${SLEAP_TIMEOUT:-86400}"   # 24h hard cap
POLL_SECS="${CLEANUP_POLL_SECS:-300}"     # check every 5 min

if [[ ! -f "$WORKLIST" ]]; then
	echo "[WARN] no worklist; treating as nothing to wait for"
	expected=0
else
	expected=$(wc -l < "$WORKLIST")
fi

count_slp() {
	local n=0
	while IFS=$'\t' read -r vname chunk; do
		[[ -n "$vname" ]] || continue
		[[ -f "$DATA_DIR/${vname}_${chunk}.slp" ]] && n=$((n+1))
	done < "$WORKLIST"
	echo "$n"
}

deadline=$(( $(date +%s) + SLEAP_TIMEOUT ))
if (( expected > 0 )); then
	echo "[INFO] waiting for $expected SLP files in $DATA_DIR (timeout ${SLEAP_TIMEOUT}s)"
	actual=$(count_slp)
	while (( actual < expected )) && (( $(date +%s) < deadline )); do
		echo "[$(date '+%H:%M:%S')] $actual / $expected SLP files; sleeping ${POLL_SECS}s"
		sleep "$POLL_SECS"
		actual=$(count_slp)
	done

	if (( actual < expected )); then
		echo "[WARN] timeout: $actual / $expected SLP files present in $DATA_DIR"
		echo "       KEEPING $FLASH_ROOT for retry."
		echo "       To force-delete after manual verification: rm -rf $FLASH_ROOT"
		exit 1
	fi
	echo "[OK] all $actual / $expected SLP files present"
fi

if [[ -d "$FLASH_ROOT" ]]; then
	echo "[INFO] freeing $FLASH_ROOT"
	rm -rf "$FLASH_ROOT"
fi

# Try to remove parent if empty (best-effort; harmless if not)
rmdir "$(dirname "$FLASH_ROOT")" 2>/dev/null || true

echo "[OK] cleanup done"
