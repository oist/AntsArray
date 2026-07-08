#!/bin/bash -l
# track_trigger.sh — login-side poller that launches colony tracking for THIS
# block once detection outputs are all in the bucket.
#
# Why a poller and not a SLURM dependency: SLEAP inference runs on SAION and
# deigo cannot `afterok:` a saion job (no cross-cluster Slurm deps). The only
# durable signal that detection is done is the bucket <exp>/data/ filling up
# with per-camera/per-chunk aruco + sleap H5 files. This mirrors how
# cleanup.sbatch polls, and how the tracking transfer watcher runs (login-side
# nohup): it survives SSH logout and keeps submit_blocks_pipeline.sh on the
# login node, where it has /bucket write + sbatch + its own transfer watcher.
#
# Rendered from pipeline.sh (single placeholder __JOBS_ROOT__); all other config
# comes from pipeline.env. Started via:
#   nohup track_trigger.sh >> <exp>/hpc_logs/pipeline/track_trigger.log 2>&1 &
set -uo pipefail

JOBS_ROOT="__JOBS_ROOT__"
source "$JOBS_ROOT/pipeline.env"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

if [[ "${RUN_TRACKING:-0}" -ne 1 ]]; then
	log "RUN_TRACKING != 1; nothing to do"
	exit 0
fi

worklist="$JOBS_ROOT/aruco_worklist.txt"
data_dir="$DATA_DIR"
poll_secs="${TRACKING_POLL_SECS:-300}"
timeout_secs="${TRACKING_TIMEOUT:-172800}"   # 48h overall deadline (aruco+sleap can be long)
deadline=$(( $(date +%s) + timeout_secs ))

blocks_root="$(dirname "$EXP_DIR")"
block="$(basename "$EXP_DIR")"

log "track trigger start: block=$block"
log "  data_dir=$data_dir"
log "  hmats=$TRACKING_HMATS"
log "  submit=$TRACKING_SUBMIT"
log "  python_bin=${TRACKING_PYTHON_BIN:-<submit-script default>}"
log "  output_root=${TRACKING_OUTPUT_ROOT:-<submit-script default>}"
log "  poll=${poll_secs}s timeout=${timeout_secs}s"

count_ext() {  # $1 = filename suffix, e.g. _sleap_data.h5
	local n
	n=$( { find "$data_dir" -maxdepth 1 -type f -name "*$1" 2>/dev/null || true; } | wc -l )
	printf '%s' "${n//[[:space:]]/}"
}

# 1) Wait for chunk_finalize to build the worklist. It has one row per
#    (camera, chunk), so its line count is the expected number of both
#    _aruco_tracks.h5 and _sleap_data.h5 outputs.
while [[ ! -s "$worklist" ]]; do
	if (( $(date +%s) >= deadline )); then
		log "ERROR: deadline reached waiting for worklist $worklist; aborting"
		exit 1
	fi
	log "waiting for worklist $worklist ..."
	sleep "$poll_secs"
done
expected=$(wc -l < "$worklist" 2>/dev/null | tr -d '[:space:]')
[[ "$expected" =~ ^[0-9]+$ ]] || { log "ERROR: could not read expected count from $worklist"; exit 1; }
log "expected per-modality outputs = $expected (from $(basename "$worklist"))"

# 2) Poll the bucket until BOTH modalities are complete, or the deadline passes.
#    Gate on _sleap_data.h5 (NOT .slp): the tracking map stage reads _sleap_data.h5,
#    and detection's inline slp->h5 conversion is best-effort (a silent failure
#    leaves only .slp, which tracking cannot see).
fired_reason=""
aruco_n=0
sleap_n=0
while true; do
	aruco_n=$(count_ext "_aruco_tracks.h5")
	sleap_n=$(count_ext "_sleap_data.h5")
	log "progress aruco=$aruco_n/$expected sleap=$sleap_n/$expected"
	if (( aruco_n >= expected && sleap_n >= expected )); then
		fired_reason="complete"
		break
	fi
	if (( $(date +%s) >= deadline )); then
		if (( aruco_n > 0 && sleap_n > 0 )); then
			fired_reason="timeout-partial"   # tracking processes the contiguous complete prefix
		else
			log "ERROR: deadline reached with aruco=$aruco_n sleap=$sleap_n; nothing complete, aborting"
			exit 1
		fi
		break
	fi
	sleep "$poll_secs"
done
log "firing tracking (reason=$fired_reason): aruco=$aruco_n sleap=$sleap_n expected=$expected"

# 3) Launch colony tracking for THIS one block. submit_blocks_pipeline.sh is a
#    separate script in tracking/colony/; we only invoke it by path (no shared
#    code). It sbatches its own map/track/stitch/interaction DAG and starts its
#    own login-side transfer watcher.
submit_args=(
	--blocks_root "$blocks_root"
	--block_glob "$block"
	--hmats "$TRACKING_HMATS"
)
[[ -n "${TRACKING_OUTPUT_ROOT:-}" ]] && submit_args+=(--output_root "$TRACKING_OUTPUT_ROOT")
[[ -n "${TRACKING_PYTHON_BIN:-}" ]] && submit_args+=(--python_bin "$TRACKING_PYTHON_BIN")
if [[ -n "${TRACKING_EXTRA_ARGS:-}" ]]; then
	# Space-separated simple tokens (no embedded quotes). Main override needs
	# (--python_bin, --output_root) have dedicated flags above.
	read -r -a _extra <<< "$TRACKING_EXTRA_ARGS" || true
	submit_args+=("${_extra[@]}")
fi

log "exec: bash $TRACKING_SUBMIT ${submit_args[*]}"
if bash "$TRACKING_SUBMIT" "${submit_args[@]}"; then
	log "tracking submitted for block=$block"
	# Cluster/log hint: tracking runs on deigo, and outputs copy back to the bucket
	# only after ALL stages finish (stitch is the long pole). Point the user at the
	# right cluster + log so 'no outputs yet' is not mistaken for a failure.
	transfer_log="${TRACKING_OUTPUT_ROOT:+$TRACKING_OUTPUT_ROOT/jobs/$block/logs/transfer_to_bucket_$block.log}"
	log "tracking runs on DEIGO (compute); bucket copy-back happens only after stitch+interaction finish."
	log "watch progress ON deigo:  squeue -u \$USER | grep -Ei 'map_|track_|stitch_|inter_'"
	[[ -n "$transfer_log" ]] && log "  and: tail -f $transfer_log"
	log "NOTE: these /flash paths are on deigo; from saion they are /deigo_flash (read-only) and the jobs are not in saion's squeue."
else
	rc=$?
	log "ERROR: tracking submit failed (rc=$rc) for block=$block"
	exit "$rc"
fi
