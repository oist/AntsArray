#!/bin/bash -l
#SBATCH -t 0-2
#SBATCH -c 4
#SBATCH --partition=short
#SBATCH --mem=8G
#SBATCH -J chunk
#SBATCH -o __JOBS_ROOT__/chunk_%A_%a.out
#SBATCH -e __JOBS_ROOT__/chunk_%A_%a.err
set -eo pipefail
shopt -s nullglob

module load ffmpeg/7.1

JOBS_ROOT="__JOBS_ROOT__"
source "$JOBS_ROOT/pipeline.env"
source "$LIB_DIR/perms.sh"

MANIFEST="$JOBS_ROOT/manifest.csv"
OUT_EXT="$CHUNK_EXT"

# Manifest has a header row; task 0 -> CSV row 2.
row=$(awk -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n' "$MANIFEST")
if [[ -z "$row" ]]; then
	echo "[ERR] no manifest row for task $SLURM_ARRAY_TASK_ID" >&2
	exit 1
fi
vname=$(echo "$row" | awk -F, '{print $1}')
source_path=$(echo "$row" | awk -F, '{print $2}')
# manifest.csv comes from csv.DictWriter, whose default lineterminator is CRLF,
# so the LAST field arrives as "198\r". That made `(( lo < n_chunks - 1 ))` die
# with "invalid arithmetic operator" -- which inside an if-condition is not fatal
# under set -e: the test simply evaluated false, and verify_first_chunk never ran.
# The alignment guard was silently inert for all of wave 2. Strip to digits
# rather than trusting the field, and refuse the task if nothing usable is left.
fps=$(echo "$row" | awk -F, '{print $4}' | tr -dc '0-9.')
n_chunks=$(echo "$row" | awk -F, '{print $7}' | tr -dc '0-9')
if [[ -z "$n_chunks" || -z "$fps" ]]; then
	echo "[ERR] manifest row for task $SLURM_ARRAY_TASK_ID has no usable fps/n_chunks: $row" >&2
	exit 1
fi

flash_dir="$FLASH_ROOT/$vname"
mkdir -p "$flash_dir"
ensure_group_perms "$flash_dir"

chunk_path() { printf '%s/%s_%03d.%s' "$flash_dir" "$vname" "$1" "$OUT_EXT"; }

# --- Which chunk indices does this task have to produce? ----------------------
# CHUNK_RANGE (from pipeline.env; "A-B" inclusive, empty = the whole video) makes
# a run a "wave": one window of the block instead of all of it. A chunk's index
# is a pure function of (video, CHUNK_SEC), so a window always names the same
# span of wall-clock across runs -- see lib/pipeline_state.py.
lo=0
hi=$(( n_chunks - 1 ))
if [[ -n "${CHUNK_RANGE:-}" ]]; then
	lo="${CHUNK_RANGE%%-*}"
	hi="${CHUNK_RANGE##*-}"
	(( hi > n_chunks - 1 )) && hi=$(( n_chunks - 1 ))   # clamp: shorter cameras
	if (( lo > hi )); then
		echo "[SKIP] $vname has $n_chunks chunks; range $CHUNK_RANGE is past its end"
		exit 0
	fi
fi

# Idempotency per index, not "any chunk present": a later wave must not be
# skipped merely because an earlier wave left its own chunks in this dir.
missing=0
for (( i=lo; i<=hi; i++ )); do
	[[ -s "$(chunk_path "$i")" ]] || missing=1
done
if (( missing == 0 )); then
	echo "[SKIP] chunks $lo-$hi already present in $flash_dir"
	exit 0
fi

# Input-side -ss so the seek costs no decoding. Sound only because chunk
# boundaries are keyframe boundaries: under -c copy both the seek and the
# segmenter snap to keyframes, so a CHUNK_SEC that is not a whole number of GOPs
# would shift every boundary in the wave. verify_first_chunk refuses that rather
# than shipping silently misaligned chunks.
ss_args=()
(( lo > 0 )) && ss_args=(-ss "$(( lo * CHUNK_SEC ))")

# -t cannot be trusted to stop the segment muxer exactly at the window's end:
# -reset_timestamps restarts the output clock for every segment, so the duration
# limit never accumulates and ffmpeg opens ONE segment past hi before stopping.
# Wave 5-6 left a 1,442-frame `_007.mkv` behind that way. That stub is the
# dangerous kind of leftover -- a later wave covering chunk 7 would find it
# non-empty, take the per-index idempotency skip, and hand a 1,442-frame chunk to
# detection as if it were a full one. Record which out-of-window chunks existed
# beforehand so the cleanup below removes only what THIS run produced, never a
# complete chunk belonging to another wave.
chunk_index_of() { local b="${1##*/}"; b="${b%.$OUT_EXT}"; echo "$(( 10#${b##*_} ))"; }
pre_out_of_window=" "
for f in "$flash_dir/${vname}"_[0-9][0-9][0-9]."$OUT_EXT"; do
	[[ -e "$f" ]] || continue
	i=$(chunk_index_of "$f")
	(( i < lo || i > hi )) && pre_out_of_window+="$i "
done

prune_out_of_window() {
	local f i n=0
	for f in "$flash_dir/${vname}"_[0-9][0-9][0-9]."$OUT_EXT"; do
		[[ -e "$f" ]] || continue
		i=$(chunk_index_of "$f")
		(( i < lo || i > hi )) || continue
		[[ "$pre_out_of_window" == *" $i "* ]] && continue   # not ours; leave it
		echo "[INFO] removing chunk $i: produced past the wave window $lo-$hi"
		rm -f "$f"
		n=$((n+1))
	done
	(( n )) && echo "[OK] pruned $n stray chunk(s) outside $lo-$hi"
	return 0
}

echo "[$(date)] chunk $vname [$lo-$hi] -> $flash_dir/${vname}_%03d.${OUT_EXT} (segment_time=${CHUNK_SEC}s)"
ffmpeg -hide_banner -nostdin -y "${ss_args[@]}" -i "$source_path" \
	-t "$(( (hi - lo + 1) * CHUNK_SEC ))" -c copy -map 0 \
	-f segment -segment_time "$CHUNK_SEC" -reset_timestamps 1 \
	-segment_start_number "$lo" \
	"$flash_dir/${vname}_%03d.${OUT_EXT}"

# A shifted seek is invisible in the output -- every chunk is a valid file of
# roughly the right size -- so check the packet count of the wave's FIRST chunk
# against the frame cap the worklist will hand inference. Packets, not frames:
# -count_packets never decodes. Skipped when lo==0 (no seek happened) and when
# the first chunk is the video's last (a legitimately short residual).
verify_first_chunk() {
	local first want got
	first=$(chunk_path "$lo")
	want=$(awk -v f="$fps" -v c="$CHUNK_SEC" 'BEGIN{printf "%d", f*c+0.5}')
	got=$(ffprobe -v error -select_streams v:0 -count_packets \
		-show_entries stream=nb_read_packets -of csv=p=0 "$first" 2>/dev/null | tr -d '\r,')
	if [[ -z "$got" ]]; then
		echo "[WARN] could not count packets in $first; skipping alignment check" >&2
		return 0
	fi
	if (( got != want )); then
		echo "[ERR] chunk $lo of $vname holds $got frames, expected $want." >&2
		echo "      The -ss seek did not land on a chunk boundary, so every chunk in" >&2
		echo "      this wave is offset. CHUNK_SEC=$CHUNK_SEC must be a whole number" >&2
		echo "      of GOPs for wave processing. Removing the misaligned output." >&2
		for (( i=lo; i<=hi; i++ )); do rm -f "$(chunk_path "$i")"; done
		return 1
	fi
	echo "[OK] chunk $lo alignment verified ($got frames)"
}
prune_out_of_window

if (( lo > 0 && lo < n_chunks - 1 )); then
	verify_first_chunk || exit 1
fi

n_produced=0
for (( i=lo; i<=hi; i++ )); do
	[[ -s "$(chunk_path "$i")" ]] && n_produced=$((n_produced+1))
done
echo "[OK] $vname produced $n_produced chunks for range $lo-$hi"
if (( n_produced != hi - lo + 1 )); then
	echo "[ERR] expected $(( hi - lo + 1 )) chunks, got $n_produced" >&2
	exit 1
fi
