#!/bin/bash -l
#SBATCH -J vr_copy
#SBATCH -p datacp
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 2-00:00:00
# Verbatim rescue copy of one session (env SESSION=<session_id>), oldest data
# first. Runs on the datacp partition because those are the only deigo compute
# nodes that can write /bucket. Videos go to $RAW_ROOT/<dest_rel>/ (temporary
# insurance until the transcode is verified); diag.json sidecars and stub
# videos go straight to their final home $DEST_ROOT/<dest_rel>/.
# Idempotent: rsync skips files already complete (size+mtime), and every file
# is size-checked after the copy. Writes $STATE/<dest_rel>/copy.done on success.
set -eo pipefail
# sbatch runs a spooled copy of this script, so locate lib.sh via RESCUE_DIR exported by rescue.sh
source "${RESCUE_DIR:?RESCUE_DIR not set (submit through rescue.sh)}/lib.sh"
[[ -n "${SESSION:-}" ]] || die "SESSION not set"
: "${INCLUDE_STUBS:=1}"
: "${INCLUDE_VIDEOS:=1}"   # 0 = sidecars only (diag.json, sess_*.txt, stubs); for sessions already transcoded

dest_rel=$(session_dest "$SESSION"); [[ -n "$dest_rel" ]] || die "unknown session $SESSION"
mkdir -p "$STATE/$dest_rel" "$RAW_ROOT/$dest_rel" "$DEST_ROOT/$dest_rel"
log "copy session $SESSION -> raw=$RAW_ROOT/$dest_rel final=$DEST_ROOT/$dest_rel (host $(hostname))"

ok=0; bad=0; skipped=0; bytes=0
while IFS=$'\t' read -r _sid _dest kind size _mtime src; do
	case "$kind" in
		video) (( INCLUDE_VIDEOS )) || { skipped=$((skipped+1)); continue; }
		       dst_dir="$RAW_ROOT/$dest_rel" ;;
		diag)  dst_dir="$DEST_ROOT/$dest_rel" ;;
		stub)  (( INCLUDE_STUBS )) || { skipped=$((skipped+1)); continue; }
		       dst_dir="$DEST_ROOT/$dest_rel" ;;
		*) die "unknown kind '$kind' in manifest" ;;
	esac
	if [[ ! -s "$src" ]]; then
		echo "[ERR] source vanished: $src" >&2; bad=$((bad+1)); continue
	fi
	dst="$dst_dir/$(basename "$src")"
	if ! rsync -rlt --partial-dir=.rsync-partial --chmod=Du=rwx,Dg=rx,Fu=rw,Fg=r "$src" "$dst_dir/"; then
		echo "[ERR] rsync failed: $src" >&2; bad=$((bad+1)); continue
	fi
	got=$(stat -c %s "$dst" 2>/dev/null || echo -1)
	if (( got != size )); then
		echo "[ERR] size mismatch after copy: $dst ($got != $size)" >&2; bad=$((bad+1)); continue
	fi
	ok=$((ok+1)); bytes=$((bytes+size))
	log "ok $(basename "$src") ($((size/1000000)) MB) -> $dst_dir"
done < <(manifest_rows "$SESSION")

log "done: ok=$ok bad=$bad skipped=$skipped copied=$((bytes/1000000000)) GB"
if (( bad == 0 )); then
	# copy.done means the RAW videos are on $RAW_ROOT; a sidecars-only run records its own marker
	if (( INCLUDE_VIDEOS )); then marker="$STATE/$dest_rel/copy.done"; else marker="$STATE/$dest_rel/sidecars.done"; fi
	date '+%F %T' > "$marker"
	echo "ok=$ok bytes=$bytes" >> "$marker"
else
	rm -f "$STATE/$dest_rel/copy.done"
	exit 2
fi
