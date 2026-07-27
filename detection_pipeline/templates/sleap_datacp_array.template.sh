#!/bin/bash -l
#SBATCH -t 0-8
#SBATCH -c 2
#SBATCH --partition=__SAION_DATACP_PARTITION__
#SBATCH --mem=8G
#SBATCH -J sleap_dc
#SBATCH -o __REMOTE_JOBS__/sleap_datacp_%j.out
#SBATCH -e __REMOTE_JOBS__/sleap_datacp_%j.err
set -eo pipefail

REMOTE_JOBS="__REMOTE_JOBS__"
REMOTE_OUTPUT="__REMOTE_OUTPUT__"
DATA_DIR="__DATA_DIR__"
OUTPUT_GROUP="__OUTPUT_GROUP__"

# Single-job datacp on saion side: scan the worklist, rsync each .slp to bucket
# via ssh saion (login has bucket write). One job to keep us under any per-user
# submit-job limit; rsync_retry handles SSH rate-limit hiccups.

SSH_CMD=(ssh -x
	-oBatchMode=yes
	-oStrictHostKeyChecking=no
	-oUserKnownHostsFile=/dev/null
	-oConnectionAttempts=5
	-oConnectTimeout=30
	-oServerAliveInterval=15
	-oServerAliveCountMax=4)
export RSYNC_RSH="${SSH_CMD[*]}"

ssh_retry()   { local i; for i in 1 2 3 4 5; do "${SSH_CMD[@]}" "$@" && return; echo "[WARN] ssh attempt $i failed; backoff $((10*i))s" >&2; sleep $((10*i)); done; return 1; }
rsync_retry() { local i; for i in 1 2 3 4 5; do rsync "$@" && return; echo "[WARN] rsync attempt $i failed; backoff $((10*i))s" >&2; sleep $((10*i)); done; return 1; }

WORKLIST="$REMOTE_JOBS/aruco_worklist.txt"

ssh_retry saion "mkdir -p '$DATA_DIR' && { chgrp $OUTPUT_GROUP '$DATA_DIR' 2>/dev/null; chmod 2775 '$DATA_DIR' 2>/dev/null; true; }"

uploaded=0
missing=0
no_h5=0
echo "[$(date)] starting sleap datacp -> saion:$DATA_DIR/"

# Ship the .h5 alongside the .slp. This job is the end-of-run safety net, but it
# used to copy only .slp -- and the tracking trigger counts _sleap_data.h5, so
# the net could never clear the very stall it exists to prevent. On
# 20260723/block02 it reported success while 3 chunks stayed absent from the
# bucket and tracking waited indefinitely.
while IFS=$'\t' read -r vname chunk _; do
	src="$REMOTE_OUTPUT/${vname}_${chunk}.slp"
	src_h5="$REMOTE_OUTPUT/${vname}_${chunk}_sleap_data.h5"
	if [[ ! -s "$src" ]]; then
		missing=$((missing+1))
		continue
	fi
	send=( "$src" )
	if [[ -s "$src_h5" ]]; then
		send+=( "$src_h5" )
	else
		# slp->h5 conversion lives in the predict job (which has the sleap-nn
		# venv); this partition may not. Flag it instead of reporting success.
		echo "[GAP] ${vname}_${chunk}: .slp present but no _sleap_data.h5; tracking will stall on this chunk" >&2
		no_h5=$((no_h5+1))
	fi
	rsync_retry -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw --chown=:"$OUTPUT_GROUP" "${send[@]}" "saion:$DATA_DIR/"
	uploaded=$((uploaded+1))
done < "$WORKLIST"

echo "[$(date)] done: uploaded=$uploaded missing=$missing no_h5=$no_h5 total_worklist=$(wc -l < "$WORKLIST")"

# Exit non-zero so an incomplete run is visible in sacct and triggers the
# failure mail. saion_cleanup is submitted --dependency=afterany on this job,
# so log archival, the sacct post-mortem, and the verify-before-delete gate
# still run; only the false "success" goes away.
if (( no_h5 > 0 )); then
	echo "[ERR] $no_h5 chunk(s) uploaded without _sleap_data.h5; tracking gates on that file and will not fire on completeness." >&2
	echo "[ERR] grep '^\[GAP\]' in this log for the chunk list, then backfill with slp2h5_array.sh." >&2
	exit 1
fi
