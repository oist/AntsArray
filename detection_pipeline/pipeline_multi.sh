#!/bin/bash -l
# pipeline_multi.sh — multi-user wave driver for detection_pipeline.
#
# Run from a deigo login as ANY user named in the block's plan file
# (<exp>/data/MULTIUSER_PLAN.json — see lib/multiuser_plan.py for the schema).
# Slurm limits are per user, so 2-3 users each driving their own slot of waves
# multiplies throughput; this wrapper makes each user's part one command and
# derives every pipeline.sh flag from the shared plan, so settings cannot
# drift between people (the processing contract would refuse a drift anyway —
# this stops it before a refused submission, not after).
#
#   pipeline_multi.sh validate  --plan <plan>    # check the plan, show slots
#   pipeline_multi.sh submit    --plan <plan>    # submit MY next pending wave
#   pipeline_multi.sh auto      --plan <plan>    # nohup poller: submit my waves
#                                                # back to back until done
#   pipeline_multi.sh status    --plan <plan>    # all slots: waves + queues
#   pipeline_multi.sh stop-auto --plan <plan>    # stop my auto poller
#   pipeline_multi.sh agent                      # forced-command SSH entry
#
# Options: --poll-secs N (auto, default 600)  --max-live N (auto, default 1)
#          --dry-run (submit/auto: print the pipeline.sh command, run nothing)
set -uo pipefail

# pwd -P: under a releases/<sha> + current-symlink deploy the submitted wave
# must pin its release dir, not the symlink a later deploy will repoint.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MP="$SCRIPT_DIR/lib/multiuser_plan.py"
PIPELINE="$SCRIPT_DIR/pipeline.sh"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { echo "[ERR] $*" >&2; exit 2; }

usage() { sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

# --- args --------------------------------------------------------------------
CMD="${1:-}"; shift || true
PLAN=""
POLL_SECS=600
MAX_LIVE=1
DRY_RUN=0
while [[ $# -gt 0 ]]; do
	case "$1" in
		--plan) PLAN="$2"; shift 2 ;;
		--poll-secs) POLL_SECS="$2"; shift 2 ;;
		--max-live) MAX_LIVE="$2"; shift 2 ;;
		--dry-run) DRY_RUN=1; shift ;;
		-h|--help) usage ;;
		*) die "unknown arg: $1" ;;
	esac
done

case "$CMD" in
	validate|submit|auto|status|stop-auto) [[ -n "$PLAN" ]] || die "--plan is required" ;;
	agent) ;;
	*) usage ;;
esac

if [[ -n "$PLAN" ]]; then
	[[ -f "$PLAN" ]] || die "plan not found: $PLAN"
	PLAN=$(readlink -f "$PLAN")
	DATA_DIR=$(dirname "$PLAN")
	EXP_DIR=$(dirname "$DATA_DIR")
	# Mirror pipeline.sh's scratch namespacing: <date>_<block> for block dirs.
	EXP_NAME=$(basename "$EXP_DIR")
	if [[ "$EXP_NAME" =~ ^block[0-9] ]]; then
		EXP_NAME="$(basename "$(dirname "$EXP_DIR")")_$EXP_NAME"
	fi
	JOBS_BASE="/flash/ReiterU/$USER/jobs/$EXP_NAME"
	LOCK_DIR="$EXP_DIR/hpc_logs/pipeline"
	LOCK_FILE="$LOCK_DIR/multi_auto_${USER}.lock"
	AUTO_LOG="$LOCK_DIR/multi_auto_${USER}.log"
	# Bucket-side marker: pipeline.sh nohups a NEW track_trigger poller on every
	# --run-tracking submission, and two pollers submit tracking twice. The
	# marker survives scratch cleanup and is visible to every user.
	TRACK_MARKER="$LOCK_DIR/track_trigger_started"
fi

# --- queue helpers -----------------------------------------------------------
Q_DEIGO=""
Q_SAION=""
Q_SAION_OK=1
fetch_queues() {
	Q_SAION_OK=1
	Q_DEIGO=$(squeue -h -u "$USER" -o '%i' 2>/dev/null || true)
	Q_SAION=$(ssh -x -oBatchMode=yes -oStrictHostKeyChecking=no \
		-oConnectTimeout=15 saion "squeue -h -u \$USER -o '%i'" 2>/dev/null) || {
		Q_SAION=""; Q_SAION_OK=0
	}
}

# wave_live <range>: any jid the wave's jobs dir recorded still in a queue?
# Same identification as pipeline.sh's concurrent-wave guard: jid_*.txt files,
# jid_saion_* checked against saion's queue, the rest against deigo's.
wave_live() {
	local d="$JOBS_BASE/wave_$1" f base jid pat
	[[ -d "$d" ]] || return 1
	shopt -s nullglob
	local jid_files=( "$d"/jid_*.txt )
	shopt -u nullglob
	for f in "${jid_files[@]}"; do
		base=$(basename "$f")
		jid=$(tr -d '[:space:]' < "$f")
		[[ -n "$jid" ]] || continue
		pat='^'"$jid"'(_|$)'
		if [[ "$base" == jid_saion_* ]]; then
			grep -qE "$pat" <<<"$Q_SAION" && return 0
		else
			grep -qE "$pat" <<<"$Q_DEIGO" && return 0
		fi
	done
	return 1
}

# --- submit ------------------------------------------------------------------
submit_wave() {  # $1 = range; returns pipeline.sh's rc
	local range="$1" out args=()
	out=$(python3 "$MP" flags --plan "$PLAN" --user "$USER" --wave "$range") \
		|| die "flag generation failed for wave $range"
	mapfile -t args <<<"$out"
	(( ${#args[@]} )) || die "flag generation produced nothing for wave $range"

	# One track_trigger poller per block, ever: a resubmission of the tracking
	# wave (rescue, auto-retry) must not nohup a second poller.
	local wants_tracking=0
	[[ " ${args[*]} " == *" --run-tracking "* ]] && wants_tracking=1
	if (( wants_tracking == 1 )) && [[ -f "$TRACK_MARKER" ]]; then
		log "tracking poller already launched ($(head -1 "$TRACK_MARKER" 2>/dev/null)); dropping --run-tracking"
		local filtered=() tok
		for tok in "${args[@]}"; do
			[[ "$tok" == "--run-tracking" ]] || filtered+=("$tok")
		done
		args=("${filtered[@]}")
		wants_tracking=0
	fi

	log "submitting wave $range: bash $PIPELINE ${args[*]}"
	if (( DRY_RUN == 1 )); then
		log "(dry-run: not submitting)"
		return 0
	fi
	bash "$PIPELINE" "${args[@]}" || return $?
	if (( wants_tracking == 1 )); then
		mkdir -p "$LOCK_DIR"
		printf '%s by %s (wave %s)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$USER" "$range" > "$TRACK_MARKER"
	fi
}

# slot_status_tsv: this slot's wave table, or FAIL LOUDLY. An empty result on
# a query error must never be read as "all waves done" — that is precisely how
# a poller would silently abandon a half-finished block.
slot_status_tsv() {
	python3 "$MP" slot-status --plan "$PLAN" --user "$USER"
}

# next_waves: my waves that are not done and not live, in slot order, one per
# line. "unknown" (no contract yet — nothing submitted for this block) counts
# as pending: only a submission can create the denominator. rc 1 = query failed.
next_waves() {
	local tsv w expected done state
	tsv=$(slot_status_tsv) || return 1
	while IFS=$'\t' read -r w expected done state; do
		[[ -n "$w" ]] || continue
		[[ "$state" == "done" ]] && continue
		wave_live "$w" && continue
		printf '%s\n' "$w"
	done <<<"$tsv"
}

# count_live_waves: rc 1 = query failed (callers must not treat that as 0).
count_live_waves() {
	local tsv w n=0 _e _d _s
	tsv=$(slot_status_tsv) || return 1
	while IFS=$'\t' read -r w _e _d _s; do
		[[ -n "$w" ]] || continue
		wave_live "$w" && n=$((n + 1))
	done <<<"$tsv"
	echo "$n"
}

cmd_submit() {
	python3 "$MP" validate --plan "$PLAN" >/dev/null || exit 2
	fetch_queues
	(( Q_SAION_OK )) || log "WARN: saion unreachable; live-wave check covers deigo only"
	local pending next live
	pending=$(next_waves) || die "wave status query failed (see [ERR] above)"
	next=$(head -1 <<<"$pending")
	if [[ -z "$next" ]]; then
		live=$(count_live_waves) || die "wave status query failed (see [ERR] above)"
		if (( live > 0 )); then
			log "nothing to submit: remaining waves are still running"
		else
			log "nothing to submit: all my waves are done"
		fi
		return 0
	fi
	submit_wave "$next"
}

# --- auto (login-side nohup poller, same pattern as track_trigger.sh) --------
# acquire_lock: atomic create-if-absent (noclobber), so two simultaneous
# `auto` invocations cannot both pass a bare existence test and fork twice.
acquire_lock() {
	( set -o noclobber; printf 'host=%s\npid=starting\n' "$(hostname)" > "$LOCK_FILE" ) 2>/dev/null
}

cmd_auto() {
	python3 "$MP" validate --plan "$PLAN" >/dev/null || exit 2
	if [[ "${_MULTI_AUTO_CHILD:-0}" != "1" ]]; then
		mkdir -p "$LOCK_DIR"
		if ! acquire_lock; then
			local lhost lpid
			lhost=$(sed -n 's/^host=//p' "$LOCK_FILE")
			lpid=$(sed -n 's/^pid=//p' "$LOCK_FILE")
			if [[ "$lhost" == "$(hostname)" ]] && kill -0 "$lpid" 2>/dev/null; then
				die "auto poller already running (host=$lhost pid=$lpid); stop-auto first"
			fi
			# A lock from another login host may be live there — we cannot check.
			[[ "$lhost" != "$(hostname)" ]] && \
				log "WARN: stale-looking lock from $lhost (pid=$lpid); replacing. If a poller is live there, stop it: ssh $lhost kill $lpid"
			rm -f "$LOCK_FILE"
			acquire_lock || die "lost the lock race to a concurrent 'auto'; retry"
		fi
		local extra=()
		(( DRY_RUN == 1 )) && extra+=(--dry-run)
		_MULTI_AUTO_CHILD=1 nohup "$SCRIPT_DIR/pipeline_multi.sh" auto --plan "$PLAN" \
			--poll-secs "$POLL_SECS" --max-live "$MAX_LIVE" "${extra[@]}" \
			>> "$AUTO_LOG" 2>&1 &
		log "auto poller started (pid $!); log: $AUTO_LOG"
		return 0
	fi

	printf 'host=%s\npid=%s\n' "$(hostname)" "$$" > "$LOCK_FILE"
	trap 'rm -f "$LOCK_FILE"' EXIT
	log "auto poller: user=$USER plan=$PLAN poll=${POLL_SECS}s max_live=$MAX_LIVE"

	declare -A attempts=()
	while true; do
		fetch_queues
		local live pending next
		# A failed status query is a transient error, never "done": log and
		# retry rather than exiting a poller mid-block on a bucket hiccup.
		if ! live=$(count_live_waves) || ! pending=$(next_waves); then
			log "ERROR: wave status query failed; retrying in ${POLL_SECS}s"
			sleep "$POLL_SECS"
			continue
		fi
		if [[ -z "$pending" && "$live" -eq 0 ]]; then
			log "all my waves are done; exiting"
			return 0
		fi
		if [[ -n "$pending" && "$live" -lt "$MAX_LIVE" ]]; then
			# A wave with no live jobs but incomplete outputs is resubmitted:
			# pipeline.sh's bucket-aware skip redoes only its gaps. Cap the
			# retries so a persistent failure escalates instead of looping,
			# and skip past an exhausted wave so it cannot block the rest.
			next=""
			local cand
			while IFS= read -r cand; do
				(( ${attempts[$cand]:-0} >= 3 )) && continue
				next="$cand"; break
			done <<<"$pending"
			if [[ -z "$next" ]]; then
				if (( live == 0 )); then
					log "ERROR: every remaining wave exhausted its 3 retries; manual rescue needed"
					log "  inspect: $EXP_DIR/hpc_logs/ and $JOBS_BASE/wave_*/"
					return 1
				fi
			else
				attempts[$next]=$(( ${attempts[$next]:-0} + 1 ))
				if ! submit_wave "$next"; then
					log "ERROR: submission of wave $next failed (attempt ${attempts[$next]}/3); retrying next poll"
				fi
			fi
		fi
		(( DRY_RUN == 1 )) && { log "(dry-run: stopping after one iteration)"; return 0; }
		sleep "$POLL_SECS"
	done
}

cmd_stop_auto() {
	[[ -f "$LOCK_FILE" ]] || { log "no auto poller lock for $USER"; return 0; }
	local lhost lpid
	lhost=$(sed -n 's/^host=//p' "$LOCK_FILE")
	lpid=$(sed -n 's/^pid=//p' "$LOCK_FILE")
	if [[ "$lhost" != "$(hostname)" ]]; then
		die "poller runs on $lhost, not here — run: ssh $lhost '$SCRIPT_DIR/pipeline_multi.sh stop-auto --plan $PLAN'"
	fi
	if kill "$lpid" 2>/dev/null; then
		log "stopped auto poller (pid $lpid)"
	else
		log "poller pid $lpid already gone"
	fi
	rm -f "$LOCK_FILE"
}

# --- status ------------------------------------------------------------------
cmd_status() {
	python3 "$MP" status --plan "$PLAN" || exit 2
	echo
	local users csv u
	# Through load_plan, not raw json.load: slot names end up on squeue/ssh
	# command lines, and load_plan is what enforces they are username-shaped.
	users=$(python3 -c 'import os, sys
sys.path.insert(0, os.path.join(sys.argv[2], "lib"))
import multiuser_plan
for u in sorted(multiuser_plan.load_plan(sys.argv[1])["slots"]): print(u)' \
		"$PLAN" "$SCRIPT_DIR") || die "plan load failed"
	csv=$(paste -sd, <<<"$users")
	# Defense in depth: this string is interpolated into a remote ssh command.
	[[ "$csv" =~ ^[A-Za-z0-9_.,-]+$ ]] || die "slot names contain unsafe characters: $csv"
	echo "queues (job counts):"
	for u in $users; do
		printf '  %-16s deigo=%s' "$u" "$(squeue -h -u "$u" 2>/dev/null | wc -l)"
		local lock="$LOCK_DIR/multi_auto_${u}.lock"
		[[ -f "$lock" ]] && printf '  auto=on(%s)' "$(sed -n 's/^host=//p' "$lock")"
		echo
	done
	local saion_counts
	if saion_counts=$(ssh -x -oBatchMode=yes -oStrictHostKeyChecking=no -oConnectTimeout=15 \
			saion "squeue -h -u '$csv' -o '%u'" 2>/dev/null | sort | uniq -c); then
		echo "  saion:"
		sed 's/^/    /' <<<"${saion_counts:-(empty)}"
	else
		echo "  saion: unreachable"
	fi
}

# --- agent (forced-command SSH entry) ----------------------------------------
# For a restricted authorized_keys line:
#   command=".../pipeline_multi.sh agent",no-port-forwarding,no-pty,... key...
# The key can then ONLY drive this wrapper: the original command line is
# re-validated token by token (no shell metacharacters, subcommand allowlist,
# plan restricted to /bucket) and re-executed through this script — never a
# shell. Anything else the key holder sends is refused.
cmd_agent() {
	[[ -n "${SSH_ORIGINAL_COMMAND:-}" ]] || die "agent: no SSH_ORIGINAL_COMMAND (this entry is for forced-command keys)"
	local -a toks=()
	read -r -a toks <<<"$SSH_ORIGINAL_COMMAND"
	(( ${#toks[@]} >= 1 )) || die "agent: empty command"
	case "${toks[0]}" in
		validate|submit|auto|status|stop-auto) ;;
		*) die "agent: subcommand '${toks[0]}' not allowed" ;;
	esac
	local t
	for t in "${toks[@]}"; do
		[[ "$t" =~ ^[A-Za-z0-9/._=-]+$ ]] || die "agent: token '$t' contains disallowed characters"
	done
	local i plan_real
	for (( i = 1; i < ${#toks[@]}; i++ )); do
		if [[ "${toks[$i]}" == "--plan" ]]; then
			# Canonicalize BEFORE the containment check: a raw prefix match
			# passes /bucket/../../anywhere and defeats the restriction.
			plan_real=$(readlink -f "${toks[$((i+1))]:-}" 2>/dev/null) || plan_real=""
			[[ "$plan_real" == /bucket/* ]] || die "agent: --plan must resolve under /bucket"
			toks[$((i+1))]="$plan_real"
		fi
	done
	exec "$SCRIPT_DIR/pipeline_multi.sh" "${toks[@]}"
}

case "$CMD" in
	validate)  python3 "$MP" validate --plan "$PLAN" ;;
	submit)    cmd_submit ;;
	auto)      cmd_auto ;;
	status)    cmd_status ;;
	stop-auto) cmd_stop_auto ;;
	agent)     cmd_agent ;;
esac
