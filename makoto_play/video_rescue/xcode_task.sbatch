#!/bin/bash -l
#SBATCH -J vr_xcode
#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 12:00:00
# Array task: re-encode one WINDOW of one video, then, if this task completed
# the video's last missing window, join all windows, verify, and upload.
#
# Row $SLURM_ARRAY_TASK_ID of env TASKLIST (tab-separated):
#   src dest_rel name widx nwin start_sec nframes total_frames fps
# widx = -1 means "finalize only" (all windows already encoded).
#
# Why windows: a 92 h recording would take ~5 days to encode in one job; 4 h
# windows keep tasks short and let hundreds run at once. Windows start on
# multiples of GOP_SEC, i.e. on source keyframes, so an input-side -ss lands
# exactly (chunk.sbatch relies on the same property) and every window's frame
# count is checked against the plan. Windows are joined with the concat
# demuxer under -c copy, then the packet count and keyframe count of the joined
# file are checked against the probe before it is shipped to $DEST_ROOT.
set -eo pipefail
# sbatch runs a spooled copy of this script, so locate lib.sh via RESCUE_DIR exported by rescue.sh
source "${RESCUE_DIR:?RESCUE_DIR not set (submit through rescue.sh)}/lib.sh"
[[ -n "${TASKLIST:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "TASKLIST / SLURM_ARRAY_TASK_ID not set"
module load "$FFMPEG_MODULE"

IFS=$'\t' read -r src dest_rel name widx nwin start_sec nframes total fps \
	< <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKLIST")
[[ -n "$fps" ]] || die "no row $SLURM_ARRAY_TASK_ID in $TASKLIST"
part_dir="$PARTS/$dest_rel/$name"
state_dir="$STATE/$dest_rel"
mkdir -p "$part_dir" "$state_dir"
gop=$(( fps * GOP_SEC ))

part_path() { printf '%s/w%04d.mkv' "$part_dir" "$1"; }

encode_window() {
	local part ok tmp prog got
	part=$(part_path "$widx"); ok="$part.ok"; tmp="$part.tmp"; prog="$part.progress"
	if [[ -s "$ok" ]]; then log "window $widx already done"; return 0; fi
	[[ -s "$src" ]] || die "source missing: $src"
	local ss_args=()
	(( start_sec > 0 )) && ss_args=(-ss "$start_sec")
	log "encode $name window $widx/$nwin: start=${start_sec}s frames=$nframes (x264 $X264_PRESET crf $X264_CRF, $TASK_CPUS threads)"
	ffmpeg -hide_banner -nostdin -loglevel error -y -threads "$TASK_CPUS" "${ss_args[@]}" -i "$src" \
		-map 0:v:0 -frames:v "$nframes" -fps_mode passthrough \
		-c:v libx264 -preset "$X264_PRESET" -crf "$X264_CRF" -profile:v main -pix_fmt yuv420p \
		-g "$gop" -keyint_min "$gop" -sc_threshold 0 -maxrate "$MAXRATE" -bufsize "$BUFSIZE" \
		-threads "$TASK_CPUS" -progress "$prog" -f matroska "$tmp"
	got=$(grep '^frame=' "$prog" | tail -1 | cut -d= -f2)
	if [[ "$got" != "$nframes" ]]; then
		rm -f "$tmp"
		die "window $widx of $name encoded ${got:-0} frames, expected $nframes (seek landed off a keyframe, or source truncated)"
	fi
	mv -f "$tmp" "$part"
	echo "$got" > "$ok"
	rm -f "$prog"
	log "window $widx ok: $got frames, $(( $(stat -c %s "$part") / 1000000 )) MB"
}

all_windows_done() {
	local w
	for (( w = 0; w < nwin; w++ )); do
		[[ -s "$(part_path "$w").ok" ]] || return 1
	done
	return 0
}

finalize() {
	local final_dir final list w out_frames out_keys expect_keys out_bytes remote_bytes json
	final_dir="$FINAL/$dest_rel"; final="$final_dir/$name.mkv"; list="$part_dir/concat.txt"
	mkdir -p "$final_dir"
	log "finalize $name: joining $nwin windows"
	: > "$list"
	for (( w = 0; w < nwin; w++ )); do
		printf "file '%s'\n" "$(part_path "$w")" >> "$list"
	done
	ffmpeg -hide_banner -nostdin -loglevel warning -y -f concat -safe 0 -i "$list" \
		-map 0:v:0 -c copy -f matroska "$final.tmp"
	# One packet scan gives both counts; no decode.
	read -r out_frames out_keys < <(ffprobe -v error -select_streams v:0 -show_entries packet=flags \
		-of csv=p=0 "$final.tmp" | awk '{n++; if ($0 ~ /K/) k++} END {print n+0, k+0}')
	expect_keys=$(( (total + gop - 1) / gop ))
	if (( out_frames != total )); then
		die "joined $name holds $out_frames frames, source has $total; parts kept in $part_dir"
	fi
	if (( out_keys != expect_keys )); then
		die "joined $name has $out_keys keyframes, expected $expect_keys (GOP $gop); parts kept in $part_dir"
	fi
	mv -f "$final.tmp" "$final"
	out_bytes=$(stat -c %s "$final")
	json="$final.xcode.json"
	python3 - "$json" <<PY
import json, sys, time
json.dump({
  "source": "$src", "source_bytes": $(stat -c %s "$src"), "source_frames": $total,
  "output": "$DEST_ROOT/$dest_rel/$name.mkv", "output_bytes": $out_bytes, "output_frames": $out_frames,
  "keyframes": $out_keys, "gop_frames": $gop, "fps": $fps, "windows": $nwin, "window_sec": $WINDOW_SEC,
  "encoder": "libx264 preset=$X264_PRESET crf=$X264_CRF profile=main yuv420p maxrate=$MAXRATE bufsize=$BUFSIZE",
  "equivalent_recorder_target": "h264_nvenc p3 vbr cq33 maxrate150 bufsize240 gop60s",
  "ffmpeg": "$(ffmpeg -version 2>&1 | awk 'NR==1{print $3}')", "host": "$(hostname)",
  "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "verified": True,
}, open(sys.argv[1], "w"), indent=1)
PY
	log "verified: $out_frames frames, $out_keys keyframes, $((out_bytes/1000000)) MB ($(( 100 * out_bytes / $(stat -c %s "$src") ))% of source); uploading"
	ssh_retry "$LOGIN_ALIAS" "mkdir -p '$DEST_ROOT/$dest_rel'"
	rsync_retry -rlt --partial-dir=.rsync-partial --chmod=Du=rwx,Dg=rx,Fu=rw,Fg=r \
		"$final" "$json" "$LOGIN_ALIAS:$DEST_ROOT/$dest_rel/"
	remote_bytes=$(ssh_retry "$LOGIN_ALIAS" "stat -c %s '$DEST_ROOT/$dest_rel/$name.mkv'")
	(( remote_bytes == out_bytes )) || die "upload size mismatch for $name: $remote_bytes != $out_bytes"
	printf '%s\tbytes=%s\tframes=%s\n' "$(date '+%F %T')" "$out_bytes" "$out_frames" > "$state_dir/$name.uploaded"
	# Scratch only: the window parts and the joined copy on /flash. The bucket copy is verified above.
	rm -rf "$part_dir" "$final" "$json"
	log "uploaded $DEST_ROOT/$dest_rel/$name.mkv"
}

if (( widx >= 0 )); then
	encode_window
fi
# minutes of walltime left for this task (empty when squeue cannot tell)
walltime_left_min() {
	squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | awk -F'[-:]' 'NF>0 {n=NF; s=$n+60*$(n-1); if (n>=3) s+=3600*$(n-2); if (n>=4) s+=86400*$(n-3); print int(s/60)}'
}

if all_windows_done; then
	left=$(walltime_left_min)
	if [[ -n "$left" ]] && (( left < FINALIZE_MIN_LEFT )); then
		# cam22 of 20260820 lost its join to the 12 h limit after a slow window;
		# leave the lock untaken so a later `rescue.sh xcode` adds a finalize-only task.
		log "[WARN] finalize of $name deferred: $left min of walltime left (< $FINALIZE_MIN_LEFT); rerun rescue.sh xcode --session $dest_rel"
		exit 0
	fi
	# The task that completes the last window finalizes; mkdir is the atomic lock.
	if mkdir "$part_dir/finalize.lock" 2>/dev/null; then
		finalize
	else
		log "another task is finalizing $name"
	fi
fi
