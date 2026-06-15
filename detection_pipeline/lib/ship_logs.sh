# Stream a Slurm task's stdout/stderr to bucket *while it runs*, so logs survive
# mid-run death (walltime, scancel, node fail, OOM) instead of being lost when the
# cleanup stage rm's the scratch tree.
#
# source AFTER hosts.sh (needs rsync_retry, ssh_retry, RSYNC_RSH).
#
# Why streaming and not "upload at cleanup": cleanup is one more job that the same
# failure (cluster drain / mass scancel) also kills, and saion_cleanup actively
# rm -rf's the jobs/ dir that holds the logs. Streaming + a SIGTERM trap means the
# log is already on bucket regardless of whether any downstream job runs.
#
# Compute nodes CANNOT write /bucket (see aruco_array.sbatch). Every ship therefore
# rsyncs over ssh to the cluster's LOGIN alias ("deigo" / "saion"), which has bucket
# write -- the exact pattern the data uploads already use.

# Globals set by log_stream_start and read by the trap handler / background shipper.
_LS_ALIAS=""
_LS_RDIR=""
_LS_STATUS=""
_LS_FILES=()
_LS_PID=""

# rsync whichever of the tracked files currently exist to login:remote_dir.
# Always non-fatal: a failed ship must never abort the inference task.
_log_ship_now() {
	local existing=() f
	for f in "${_LS_FILES[@]}" "$_LS_STATUS"; do
		[[ -e "$f" ]] && existing+=("$f")
	done
	(( ${#existing[@]} )) || return 0
	rsync_retry -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw \
		${OUTPUT_GROUP:+--chown=:"$OUTPUT_GROUP"} \
		"${existing[@]}" "$_LS_ALIAS:$_LS_RDIR/" >/dev/null 2>&1 || true
	return 0
}

# log_stream_start <login_alias> <remote_dir> <status_file> <tracked_file>...
# Creates the remote dir (best-effort), writes a start marker, and launches a
# background loop that ships the tracked files whenever they grow.
log_stream_start() {
	_LS_ALIAS="$1"; _LS_RDIR="$2"; _LS_STATUS="$3"; shift 3
	_LS_FILES=("$@")
	local interval="${LOG_SHIP_INTERVAL:-300}"
	local task="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${SLURM_ARRAY_TASK_ID:-0}"

	ssh_retry "$_LS_ALIAS" "mkdir -p '$_LS_RDIR' && { chgrp ${OUTPUT_GROUP:-} '$_LS_RDIR' 2>/dev/null; chmod 2775 '$_LS_RDIR' 2>/dev/null; true; }" >/dev/null 2>&1 || true

	{
		echo "task=$task"
		echo "host=$(hostname)"
		echo "start=$(date -Is)"
		echo "restart=${SLURM_RESTART_COUNT:-0}"
		echo "state=running"
	} > "$_LS_STATUS" 2>/dev/null || true

	# Per-task jitter so concurrent array tasks don't all hit the login at once.
	local jitter=$(( (${SLURM_ARRAY_TASK_ID:-0} * 7) % 30 ))
	(
		sleep "$jitter"
		declare -A sig
		local f cur changed
		while sleep "$interval"; do
			changed=0
			for f in "${_LS_FILES[@]}"; do
				[[ -e "$f" ]] || continue
				cur=$(stat -c '%s-%Y' "$f" 2>/dev/null) || continue
				if [[ "${sig[$f]:-}" != "$cur" ]]; then sig[$f]="$cur"; changed=1; fi
			done
			(( changed )) && _log_ship_now
		done
	) &
	_LS_PID=$!
	return 0
}

# Trap handler. Records the exit reason, stops the background shipper, and does a
# final synchronous ship so the last lines (traceback / TERM marker) reach bucket.
# LOG_FINISH_REASON is set by the signal trap before exit; defaults to "exit".
log_stream_finish() {
	local code="${1:-$?}"
	[[ -n "$_LS_PID" ]] && kill "$_LS_PID" >/dev/null 2>&1 || true
	{
		echo "end=$(date -Is)"
		echo "exit_code=$code"
		echo "reason=${LOG_FINISH_REASON:-exit}"
		echo "state=finished"
	} >> "$_LS_STATUS" 2>/dev/null || true
	_log_ship_now
	return 0
}

# Mark that we are dying from a signal (walltime/scancel send SIGTERM first), then
# exit so the EXIT trap's log_stream_finish runs and flushes. Pair with
#   #SBATCH --signal=TERM@60
# so Slurm delivers SIGTERM ~60s before the SIGKILL, giving the flush time to run.
log_stream_on_term() {
	LOG_FINISH_REASON="signal (walltime/scancel/preempt)"
	exit 143
}

# install_log_traps : wire the handlers. Call once, right after log_stream_start.
install_log_traps() {
	trap 'log_stream_finish "$?"' EXIT
	trap 'log_stream_on_term' TERM INT
	return 0
}
