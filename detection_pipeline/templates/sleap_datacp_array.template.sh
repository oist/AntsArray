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

# A killed sleap2h5 leaves a truncated h5 on scratch that is still non-empty, so
# the old `-s` test shipped it as finished; the tracking trigger then counted the
# filename and fired, and the map job on deigo was the first thing to open it
# (20260723/block01 cam13_005). Validate before shipping.
#
# Best-effort by design: this partition may not carry the sleap-nn venv, so try
# the known interpreters and fall back to the presence test when none can import
# h5py. "Cannot validate" must never become "file is bad" -- that would drop good
# chunks on a host that merely lacks a library.
h5_ok() {
	local f="$1" py
	[[ -s "$f" ]] || return 1
	for py in "${UV_TOOL_DIR:-}/sleap-nn/bin/python" "$(command -v python3 2>/dev/null)"; do
		[[ -n "$py" && -x "$py" ]] || continue
		"$py" -c 'import h5py' 2>/dev/null || continue
		"$py" -c '
import sys, h5py
try:
    with h5py.File(sys.argv[1], "r") as h:
        ok = len(list(h.keys())) > 0
except Exception:
    ok = False
sys.exit(0 if ok else 1)
' "$f" 2>/dev/null
		return $?   # first usable validator decides
	done
	# Once per job, not once per chunk: on a partition with no h5py this fires for
	# every worklist row and buries the [GAP]/[ERR] lines the log exists to show.
	if [[ -z "${_h5_novalidator_warned:-}" ]]; then
		echo "[WARN] no python with h5py on this partition; shipping h5 files on presence alone" >&2
		_h5_novalidator_warned=1
	fi
	return 0
}

ssh_retry saion "mkdir -p '$DATA_DIR' && { chgrp $OUTPUT_GROUP '$DATA_DIR' 2>/dev/null; chmod 2775 '$DATA_DIR' 2>/dev/null; true; }"

uploaded=0
missing=0
no_h5=0
failed=0
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
	if h5_ok "$src_h5"; then
		send+=( "$src_h5" )
	else
		# slp->h5 conversion lives in the predict job (which has the sleap-nn
		# venv); this partition may not. Flag it instead of reporting success.
		# Counted as a GAP whether absent or unreadable: shipping a truncated h5
		# is strictly worse than shipping none, because presence is what the
		# tracking trigger counts, so a corrupt one reads as "detection done".
		echo "[GAP] ${vname}_${chunk}: no usable _sleap_data.h5 on scratch; tracking will stall on this chunk" >&2
		no_h5=$((no_h5+1))
	fi
	# Guarded because this loop body is NOT exempt from `set -e`: an unguarded
	# rsync_retry that exhausts its 5 attempts (150s of backoff -- a network blip
	# outlasting that is enough) would abort the whole job right here, silently
	# skipping every remaining worklist row and never printing the summary or
	# reaching the completeness gate below. That is the exact opposite of what an
	# end-of-run safety net is for: one bad chunk must not hide the other 174.
	if rsync_retry -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw --chown=:"$OUTPUT_GROUP" "${send[@]}" "saion:$DATA_DIR/"; then
		uploaded=$((uploaded+1))
	else
		echo "[ERR] ${vname}_${chunk}: rsync failed after retries; chunk is NOT on the bucket" >&2
		failed=$((failed+1))
	fi
done < "$WORKLIST"

echo "[$(date)] done: uploaded=$uploaded missing=$missing no_h5=$no_h5 failed=$failed total_worklist=$(wc -l < "$WORKLIST")"

# Exit non-zero so an incomplete run is visible in sacct and triggers the
# failure mail. saion_cleanup is submitted --dependency=afterany on this job,
# so log archival, the sacct post-mortem, and the verify-before-delete gate
# still run; only the false "success" goes away.
if (( no_h5 > 0 )); then
	echo "[ERR] $no_h5 chunk(s) uploaded without a usable _sleap_data.h5; tracking gates on that file and will not fire on completeness." >&2
	echo "[ERR] grep '^\[GAP\]' in this log for the chunk list, then backfill with slp2h5_array.sh." >&2
fi
if (( failed > 0 )); then
	echo "[ERR] $failed chunk(s) failed to transfer after retries and are absent from the bucket." >&2
	echo "[ERR] grep '^\[ERR\].*rsync failed' in this log for the chunk list; re-running this job re-sends them." >&2
fi
if (( no_h5 > 0 || failed > 0 )); then
	exit 1
fi
