#!/bin/bash
# video_rescue: move recorder output off the auto-purged acquisition share into
# /bucket, re-encoding it on deigo to the recorder's new bitrate target.
#
#   rescue.sh inventory [--dest-rel D]          scan $SRC_ROOT into sessions (D names the single session found)
#   rescue.sh copy   --session S|--all [--sidecars-only]
#                                                verbatim copy (datacp) -> $RAW_ROOT, oldest first;
#                                                --sidecars-only ships only diag.json / sess txt / stubs to $DEST_ROOT
#   rescue.sh probe  --session S|--all           count frames per video (compute array)
#   rescue.sh xcode  --session S|--all [--files GLOB] [--dry-run] [--force]
#                                                window-encode + join + verify + upload
#   rescue.sh status [--session S]
#   rescue.sh prune-raw --session S [--yes]      delete raw copies whose transcode is verified
#   rescue.sh prune-source --session S [--symlink] [--yes]
#                                                delete the SOURCE videos themselves (for a source that already
#                                                lives on /bucket); all-or-nothing per session, --symlink leaves
#                                                links to the transcodes in their place
#
# S is a session_id (20260819_181951) or a dest_rel (20260819/block01) from
# sessions.tsv. Settings live in lib.sh and can be overridden by environment
# variables VR_X264_CRF, VR_WINDOW_SEC, VR_ARRAY_THROTTLE, VR_DEST_ROOT, ...
set -eo pipefail
RESCUE_DIR="$(cd -- "$(dirname -- "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
export RESCUE_DIR
source "$RESCUE_DIR/lib.sh"

usage() { sed -n '2,21p' "$RESCUE_DIR/rescue.sh" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# --- session selection ---------------------------------------------------------
# selected_sessions -> session ids, oldest data first, honouring --session/--all
selected_sessions() {
	[[ -s "$SESSIONS" ]] || die "no $SESSIONS (run: rescue.sh inventory)"
	if [[ -n "$SESSION" ]]; then
		local sid; sid=$(session_id_of "$SESSION")
		[[ -n "$sid" ]] || die "unknown session '$SESSION' (see $SESSIONS)"
		echo "$sid"
	elif (( ALL )); then
		# sessions.tsv is already sorted oldest-first; skip stub-only sessions unless asked
		awk -F'\t' -v stubs="$INCLUDE_STUBS" 'NR>1 && ($3>0 || stubs==1) {print $1}' "$SESSIONS"
	else
		die "need --session S or --all"
	fi
}

jobs_running_for() {   # jobs_running_for <jobname> -> 0 if any job with that name is queued/running
	[[ -n "$(squeue -h -u "$USER" -n "$1" -o '%i' 2>/dev/null)" ]]
}

# --- subcommands ---------------------------------------------------------------
cmd_inventory() {
	mkdir -p "$WORK" "$LOGS" "$STATE" "$TASKS"
	python3 "$RESCUE_DIR/inventory.py" --src "$SRC_ROOT" --out-dir "$WORK" "$@"
	echo "[INFO] manifest: $MANIFEST"
}

cmd_copy() {
	mkdir -p "$LOGS"
	local sid dest_rel jid n=0
	for sid in $(selected_sessions); do
		dest_rel=$(session_dest "$sid")
		if (( INCLUDE_VIDEOS )) && [[ -s "$STATE/$dest_rel/copy.done" ]]; then echo "[SKIP] $sid already copied ($dest_rel)"; continue; fi
		if (( ! INCLUDE_VIDEOS )) && [[ -s "$STATE/$dest_rel/copy.done" || -s "$STATE/$dest_rel/sidecars.done" ]]; then echo "[SKIP] $sid: sidecars already copied ($dest_rel)"; continue; fi
		if jobs_running_for "vr_copy_$sid"; then echo "[SKIP] $sid: a copy job is already queued"; continue; fi
		jid=$(sbatch --parsable -J "vr_copy_$sid" -p "$COPY_PARTITION" \
			-o "$LOGS/copy_${sid}_%j.out" --export=ALL,SESSION="$sid",INCLUDE_STUBS="$INCLUDE_STUBS",INCLUDE_VIDEOS="$INCLUDE_VIDEOS" \
			"$RESCUE_DIR/copy_task.sbatch")
		echo "[OK] $sid -> $RAW_ROOT/$dest_rel  job $jid"
		n=$((n+1))
	done
	echo "[INFO] $n copy job(s) submitted on $COPY_PARTITION (that partition runs 2 at a time for this account)"
}

cmd_probe() {
	mkdir -p "$LOGS" "$TASKS"
	local sid dest_rel list n src name resolved jid
	for sid in $(selected_sessions); do
		dest_rel=$(session_dest "$sid")
		list="$TASKS/probe_${sid}_$(date +%Y%m%d_%H%M%S).tsv"; : > "$list"
		while IFS=$'\t' read -r _ _ _ _ _ src; do
			name=$(basename "$src" .mkv)
			[[ -s "$STATE/$dest_rel/$name.probe.json" ]] && continue
			resolved=$(video_source "$src" "$dest_rel") || { echo "[WARN] no readable copy of $name" >&2; continue; }
			printf '%s\t%s\t%s\n' "$resolved" "$dest_rel" "$name" >> "$list"
		done < <(manifest_rows "$sid" video)
		n=$(wc -l < "$list")
		if (( n == 0 )); then echo "[SKIP] $sid: all videos probed"; rm -f "$list"; continue; fi
		jid=$(sbatch --parsable -J "vr_probe_$sid" -p "$COMPUTE_PARTITION" -t "$PROBE_WALL" --array="1-${n}%${PROBE_THROTTLE}" \
			-o "$LOGS/probe_${sid}_%A_%a.out" --export=ALL,TASKLIST="$list" "$RESCUE_DIR/probe_task.sbatch")
		echo "[OK] $sid: probing $n video(s), array job $jid"
	done
}

# plan_windows <sid> <list> -> appends task rows, prints "files windows tasks frames"
plan_windows() {
	local sid="$1" list="$2" dest_rel src name resolved probe frames fps wf nwin w start nf
	local files=0 windows=0 tasks=0 total_frames=0 missing=0
	dest_rel=$(session_dest "$sid")
	while IFS=$'\t' read -r _ _ _ _ _ src; do
		name=$(basename "$src" .mkv)
		[[ -n "$FILES_GLOB" && "$name" != $FILES_GLOB ]] && continue
		[[ -s "$STATE/$dest_rel/$name.uploaded" ]] && continue
		probe="$STATE/$dest_rel/$name.probe.json"
		[[ -s "$probe" ]] || { missing=$((missing+1)); echo "[WARN] $name: not probed" >&2; continue; }
		resolved=$(video_source "$src" "$dest_rel") || { echo "[WARN] $name: no readable copy" >&2; continue; }
		frames=$(probe_field "$probe" frames); fps=$(probe_field "$probe" fps)
		wf=$(( WINDOW_SEC * fps )); nwin=$(( (frames + wf - 1) / wf ))
		files=$((files+1)); windows=$((windows+nwin)); total_frames=$((total_frames+frames))
		rm -rf "$PARTS/$dest_rel/$name/finalize.lock"   # stale lock from a crashed finalize
		local added=0
		for (( w = 0; w < nwin; w++ )); do
			[[ -s "$(printf '%s/%s/%s/w%04d.mkv.ok' "$PARTS" "$dest_rel" "$name" "$w")" ]] && continue
			start=$(( w * WINDOW_SEC )); nf=$(( frames - w * wf )); (( nf > wf )) && nf=$wf
			printf '%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n' "$resolved" "$dest_rel" "$name" "$w" "$nwin" "$start" "$nf" "$frames" "$fps" >> "$list"
			added=$((added+1))
		done
		if (( added == 0 )); then   # every window exists: finalize-only row
			printf '%s\t%s\t%s\t-1\t%d\t0\t0\t%d\t%d\n' "$resolved" "$dest_rel" "$name" "$nwin" "$frames" "$fps" >> "$list"
			added=1
		fi
		tasks=$((tasks+added))
	done < <(manifest_rows "$sid" video)
	(( missing == 0 )) || die "$missing video(s) of $sid have no probe; run: rescue.sh probe --session $sid"
	echo "$files $windows $tasks $total_frames"
}

cmd_xcode() {
	mkdir -p "$LOGS" "$TASKS" "$PARTS" "$FINAL"
	(( WINDOW_SEC % GOP_SEC == 0 )) || die "WINDOW_SEC=$WINDOW_SEC must be a multiple of GOP_SEC=$GOP_SEC"
	local sid list files windows tasks frames jid est_h
	for sid in $(selected_sessions); do
		if jobs_running_for "vr_xcode_$sid" && (( ! FORCE )); then
			echo "[SKIP] $sid: an xcode job is already queued/running (use --force to submit anyway)"; continue
		fi
		list="$TASKS/xcode_${sid}_$(date +%Y%m%d_%H%M%S).tsv"; : > "$list"
		read -r files windows tasks frames < <(plan_windows "$sid" "$list")
		if (( tasks == 0 )); then echo "[SKIP] $sid: nothing left to encode"; rm -f "$list"; continue; fi
		est_h=$(awk -v f="$frames" -v t="$ARRAY_THROTTLE" -v r="$FPS_PER_TASK" 'BEGIN{printf "%.1f", f / (t * r) / 3600}')
		echo "[PLAN] $sid: $files file(s), $windows window(s) of ${WINDOW_SEC}s, $tasks task(s), $frames frames (~${est_h} h at ${ARRAY_THROTTLE} tasks x ${FPS_PER_TASK} fps)"
		if (( DRY_RUN )); then echo "       tasklist: $list"; continue; fi
		jid=$(sbatch --parsable -J "vr_xcode_$sid" -p "$COMPUTE_PARTITION" -c "$TASK_CPUS" --mem="$TASK_MEM" -t "$TASK_WALL" \
			--array="1-${tasks}%${ARRAY_THROTTLE}" -o "$LOGS/xcode_${sid}_%A_%a.out" \
			--export=ALL,TASKLIST="$list" "$RESCUE_DIR/xcode_task.sbatch")
		echo "[OK] $sid: array job $jid ($tasks tasks, x264 $X264_PRESET crf $X264_CRF)"
	done
}

cmd_status() {
	[[ -s "$SESSIONS" ]] || die "no $SESSIONS (run: rescue.sh inventory)"
	printf '%-16s %-20s %5s %6s %6s %-12s %8s %s\n' session dest video copied probed windows uploaded jobs
	local sid dest_rel nv copied probed done_w total_w up name probe frames fps wf nwin w
	while IFS=$'\t' read -r sid dest_rel nv _ _ _ _ _; do
		[[ "$sid" == session_id ]] && continue
		[[ -n "$SESSION" && "$sid" != "$(session_id_of "$SESSION")" ]] && continue
		copied=no; [[ -s "$STATE/$dest_rel/copy.done" ]] && copied=yes
		probed=0; done_w=0; total_w=0; up=0
		while IFS=$'\t' read -r _ _ _ _ _ src; do
			name=$(basename "$src" .mkv); probe="$STATE/$dest_rel/$name.probe.json"
			[[ -s "$probe" ]] || continue
			probed=$((probed+1))
			frames=$(probe_field "$probe" frames); fps=$(probe_field "$probe" fps)
			wf=$(( WINDOW_SEC * fps )); nwin=$(( (frames + wf - 1) / wf )); total_w=$((total_w+nwin))
			if [[ -s "$STATE/$dest_rel/$name.uploaded" ]]; then   # parts are gone once uploaded
				up=$((up+1)); done_w=$((done_w+nwin)); continue
			fi
			for (( w = 0; w < nwin; w++ )); do
				[[ -s "$(printf '%s/%s/%s/w%04d.mkv.ok' "$PARTS" "$dest_rel" "$name" "$w")" ]] && done_w=$((done_w+1))
			done
		done < <(manifest_rows "$sid" video)
		printf '%-16s %-20s %5s %6s %6s %-12s %8s %s\n' "$sid" "$dest_rel" "$nv" "$copied" "$probed/$nv" "$done_w/$total_w" "$up/$nv" \
			"$(squeue -h -u "$USER" -o '%j %T' 2>/dev/null | grep -c "_$sid " || true)"
	done < "$SESSIONS"
}

# transcode_verified <dest_rel> <name> -> 0 when the bucket copy matches its xcode.json
# (upload marker present, size equal, verified flag set); otherwise prints why and returns 1.
transcode_verified() {
	local dest_rel="$1" name="$2" json rbytes
	json="$DEST_ROOT/$dest_rel/$name.mkv.xcode.json"
	if [[ ! -s "$STATE/$dest_rel/$name.uploaded" || ! -s "$json" ]]; then echo "transcode not verified"; return 1; fi
	rbytes=$(probe_field "$json" output_bytes)
	if [[ "$(stat -c %s "$DEST_ROOT/$dest_rel/$name.mkv" 2>/dev/null)" != "$rbytes" || "$(probe_field "$json" verified)" != True ]]; then
		echo "bucket copy does not match its xcode.json"; return 1
	fi
	return 0
}

cmd_prune_raw() {
	[[ -n "$SESSION" ]] || die "prune-raw needs --session S"
	local sid dest_rel name src raw why ok=0 skip=0 bytes=0
	sid=$(session_id_of "$SESSION"); [[ -n "$sid" ]] || die "unknown session '$SESSION' (see $SESSIONS)"
	dest_rel=$(session_dest "$sid")
	while IFS=$'\t' read -r _ _ _ _ _ src; do
		name=$(basename "$src" .mkv); raw="$RAW_ROOT/$dest_rel/$name.mkv"
		[[ -e "$raw" ]] || continue
		if ! why=$(transcode_verified "$dest_rel" "$name"); then echo "[KEEP] $name: $why"; skip=$((skip+1)); continue; fi
		bytes=$(( bytes + $(stat -c %s "$raw") ))
		if (( YES )); then rm -f "$raw"; echo "[RM] $raw"; else echo "[WOULD RM] $raw"; fi
		ok=$((ok+1))
	done < <(manifest_rows "$sid" video)
	if (( YES )); then rmdir "$RAW_ROOT/$dest_rel" 2>/dev/null || true; fi
	if (( YES )); then echo "[INFO] $ok raw file(s), $((bytes/1000000000)) GB removed; $skip kept"
	else echo "[INFO] $ok raw file(s), $((bytes/1000000000)) GB would be removed (add --yes); $skip kept"; fi
}

# prune-source: when SRC_ROOT is a block directory that already lives on /bucket there is no
# raw copy, so the originals themselves go once their transcodes are verified. All-or-nothing:
# with --yes nothing is removed unless every video of the session verifies. --symlink puts an
# absolute link to the transcode where each original was, so the block directory, its view
# blocks (relative ../blockNN/ links) and the detection pipeline keep resolving the same names.
cmd_prune_source() {
	[[ -n "$SESSION" ]] || die "prune-source needs --session S"
	local sid dest_rel name src target why ok=0 skip=0 linked=0 relink=0 bytes=0
	local -a victims=() orphans=()
	sid=$(session_id_of "$SESSION"); [[ -n "$sid" ]] || die "unknown session '$SESSION' (see $SESSIONS)"
	dest_rel=$(session_dest "$sid")
	# Pass 1: classify every video of the session. Anything not provably safe counts as "kept",
	# and one kept video blocks the whole --yes run.
	while IFS=$'\t' read -r _ _ _ _ _ src; do
		name=$(basename "$src" .mkv); target="$DEST_ROOT/$dest_rel/$name.mkv"
		if ! why=$(transcode_verified "$dest_rel" "$name"); then echo "[KEEP] $name: $why"; skip=$((skip+1)); continue; fi
		if [[ -L "$src" ]]; then
			if [[ "$(readlink -f "$src")" == "$(readlink -f "$target")" ]]; then
				echo "[DONE] $name: already linked to its transcode"; linked=$((linked+1))
			else
				echo "[KEEP] $name: is a symlink to $(readlink "$src"), not to its transcode"; skip=$((skip+1))
			fi
			continue
		fi
		if [[ ! -e "$src" ]]; then
			# the original is gone but no link stands in its place (a run died between steps, or a
			# manual deletion): only --symlink can repair it, otherwise it is a reason to stop
			if (( SYMLINK )); then echo "[WOULD LINK] $src -> $target (source already gone)"; orphans+=("$src"); relink=$((relink+1))
			else echo "[KEEP] $name: source missing and no --symlink to put a link in its place"; skip=$((skip+1)); fi
			continue
		fi
		[[ "$(readlink -f "$src")" != "$(readlink -f "$target")" ]] || die "$name: source and transcode are the same file"
		bytes=$(( bytes + $(stat -c %s "$src") )); ok=$((ok+1)); victims+=("$src")
		if (( SYMLINK )); then echo "[WOULD RM+LINK] $src -> $target"; else echo "[WOULD RM] $src"; fi
	done < <(manifest_rows "$sid" video)
	if (( ! YES )); then
		echo "[INFO] $ok source file(s), $((bytes/1000000000)) GB would be removed (add --yes); $relink missing source(s) would get a link; $skip kept; $linked already linked"
		return 0
	fi
	(( skip == 0 )) || die "$skip video(s) of $sid are not verified; prune-source is all-or-nothing, nothing removed"
	# Pass 2: act. With --symlink the original is swapped for the link by one rename(2), so there
	# is never a moment where the name is missing; a failure stops the run with the file untouched.
	for src in "${victims[@]}" "${orphans[@]}"; do
		name=$(basename "$src" .mkv); target="$DEST_ROOT/$dest_rel/$name.mkv"
		if (( SYMLINK )); then
			rm -f "$src.vr_link"
			ln -s "$target" "$src.vr_link" || die "$name: cannot create $src.vr_link; stopped, $src untouched"
			mv -T "$src.vr_link" "$src" || die "$name: cannot replace $src with its link; stopped"
			echo "[RM+LINK] $src -> $target"
		else
			rm -f "$src" || die "$name: cannot remove $src; stopped"
			echo "[RM] $src"
		fi
	done
	echo "[INFO] $ok source file(s), $((bytes/1000000000)) GB removed; $relink link(s) restored; $linked already linked"
}

# --- main ----------------------------------------------------------------------
[[ $# -ge 1 ]] || usage 1
cmd="$1"; shift
SESSION=""; ALL=0; FILES_GLOB=""; DRY_RUN=0; FORCE=0; YES=0; SYMLINK=0; INCLUDE_STUBS=${INCLUDE_STUBS:-1}; INCLUDE_VIDEOS=1
extra=()
while [[ $# -gt 0 ]]; do
	case "$1" in
		--session) SESSION="$2"; shift 2 ;;
		--all) ALL=1; shift ;;
		--files) FILES_GLOB="$2"; shift 2 ;;
		--dry-run) DRY_RUN=1; shift ;;
		--force) FORCE=1; shift ;;
		--yes) YES=1; shift ;;
		--symlink) SYMLINK=1; shift ;;
		--no-stubs) INCLUDE_STUBS=0; shift ;;
		--sidecars-only) INCLUDE_VIDEOS=0; shift ;;
		-h|--help) usage 0 ;;
		*) extra+=("$1"); shift ;;
	esac
done
case "$cmd" in
	inventory) cmd_inventory "${extra[@]}" ;;
	copy) cmd_copy ;;
	probe) cmd_probe ;;
	xcode) cmd_xcode ;;
	status) cmd_status ;;
	prune-raw) cmd_prune_raw ;;
	prune-source) cmd_prune_source ;;
	*) echo "[ERR] unknown command: $cmd" >&2; usage 1 ;;
esac
