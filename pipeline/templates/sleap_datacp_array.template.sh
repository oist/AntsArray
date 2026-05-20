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

ssh_retry saion "mkdir -p '$DATA_DIR' && chmod 2775 '$DATA_DIR' 2>/dev/null || true"

uploaded=0
missing=0
echo "[$(date)] starting sleap datacp -> saion:$DATA_DIR/"

while IFS=$'\t' read -r vname chunk; do
	src="$REMOTE_OUTPUT/${vname}_${chunk}.slp"
	if [[ ! -s "$src" ]]; then
		missing=$((missing+1))
		continue
	fi
	rsync_retry -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw "$src" "saion:$DATA_DIR/"
	uploaded=$((uploaded+1))
done < "$WORKLIST"

echo "[$(date)] done: uploaded=$uploaded missing=$missing total_worklist=$(wc -l < "$WORKLIST")"
