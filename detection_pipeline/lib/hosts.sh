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
