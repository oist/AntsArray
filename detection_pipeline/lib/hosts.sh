# Cross-cluster SSH/rsync helpers with retry/backoff.
# source from sbatch templates: `source "$LIB_DIR/hosts.sh"`
#
# Lesson from block01: every cross-cluster SSH must retry, or one transient
# kex_exchange_identification reset wedges the whole pipeline.

SSH_CMD=(ssh -x
	-oBatchMode=yes
	-oStrictHostKeyChecking=no
	-oUserKnownHostsFile=/dev/null
	-oConnectionAttempts=5
	-oConnectTimeout=30
	-oServerAliveInterval=15
	-oServerAliveCountMax=4)
export RSYNC_RSH="${SSH_CMD[*]}"

ssh_retry() {
	local n
	for n in 1 2 3 4 5; do
		"${SSH_CMD[@]}" "$@" && return 0
		echo "[WARN] ssh attempt $n failed (backoff $((10*n))s): $*" >&2
		sleep $((10*n))
	done
	echo "[ERR] ssh failed after 5 attempts: $*" >&2
	return 1
}

rsync_retry() {
	local n
	for n in 1 2 3 4 5; do
		rsync "$@" && return 0
		echo "[WARN] rsync attempt $n failed (backoff $((10*n))s)" >&2
		sleep $((10*n))
	done
	echo "[ERR] rsync failed after 5 attempts" >&2
	return 1
}

host_resolves() {
	getent hosts "$1" >/dev/null 2>&1
}

# sbatch_retry <jobname> <sbatch args...>
# Submit with retry on transient slurmctld failures ("Socket timed out on send/recv"),
# which otherwise abort the whole pipeline under `set -e`. The controller can still
# CREATE the job when the client times out (a leaked job), so before retrying we adopt
# the newest queued job of the same name instead of double-submitting. Adds --parsable
# and echoes the job id. Caveat: adoption matches by job name, so don't submit two
# same-named stages within seconds of each other.
sbatch_retry() {
	local jobname="$1"; shift
	local n out jid
	for n in 1 2 3 4 5; do
		if out=$(sbatch --parsable --job-name="$jobname" "$@" 2>&1); then
			printf '%s\n' "$out"; return 0
		fi
		echo "[WARN] sbatch '$jobname' attempt $n failed: $out" >&2
		sleep $((10 * n))
		jid=$(squeue -u "$USER" -h -n "$jobname" --sort=-V -o '%i' 2>/dev/null | head -1)
		if [[ -n "$jid" ]]; then
			jid=${jid%%_*}
			echo "[INFO] adopting leaked '$jobname' job $jid (submit timed out but job exists)" >&2
			printf '%s\n' "$jid"; return 0
		fi
	done
	echo "[ERR] sbatch '$jobname' failed after 5 attempts" >&2
	return 1
}
