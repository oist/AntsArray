#!/bin/bash -l
#SBATCH -t 4-00:00:00
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --partition=compute
#SBATCH -J orchestrator
#SBATCH -o orchestrator_%j.out
#SBATCH -e orchestrator_%j.err

# ------------------------------------------------------------
#  transcode_sleap_aruco.sh — Deigo↔Saion orchestration
# ------------------------------------------------------------
#  Stage per video:
#    1. split         -> raw chunks on /flash
#    2. encode array  -> re-encode segments on /flash
#    3. encode sync   -> push segments to /bucket, mark encode.ok
#    4. bridge        -> stage to Saion + submit SLEAP array/collect
#    5. aruco array   -> detect ArUco markers on Deigo
#    6. aruco sync    -> collate outputs, mark aruco.ok
#    7. sleap sync    -> collate outputs, mark sleap.ok
# ------------------------------------------------------------
set -eo pipefail
shopt -s nullglob
IFS=$'\n\t'

# Resolve script directory for relative path usage
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- Section: Helper utilities ---

usage() {
	cat <<'EOT'
Usage: 
  Interactive: bash transcode_sleap_aruco.sh --dir <folder> ...
  Batch:       sbatch transcode_sleap_aruco.sh --dir <folder> ...
Environment:
	ENC_CONCURRENCY    Max concurrent encoder tasks (default 8)
	ARUCO_CONCURRENCY  Max concurrent ArUco tasks (default 12)
	SLEAP_CONCURRENCY  Max concurrent SLEAP tasks on Saion (default 4)
	CHUNK_BUFFER       Extra indices when sizing arrays (default 2)
	SENTINEL_TIMEOUT   Seconds cleanup waits for sleap.ok (default 86400)
	BUCKET_WRITE_HOST  SSH host with /bucket write access from Deigo jobs (default datacp)
	BUCKET_WRITE_HOST_CANDIDATES
	                    Comma/space list of fallback hosts when BUCKET_WRITE_HOST is unset
	                    (default "datacp,deigo-login1,deigo-login2,deigo-login3")
	SAION_BUCKET_HOST  SSH host with /bucket write access from Saion jobs (default saion-login1)
	SAION_BUCKET_HOST_CANDIDATES
	                    Comma/space list of fallbacks for Saion bucket host
	                    (default "saion-login1,saion-login2,saion-login3")
	DATA_COPY_PARTITION Partition used for helper copy jobs targeting /bucket (default datacp)
EOT
	exit 1
}

require_cmd() {
	command -v "$1" >/dev/null 2>&1 || { echo "[ERR] missing command: $1" >&2; exit 2; }
}

calc_chunk_count() {
	local video="$1"
	local seg_sec="$2"
	local duration
	duration=$(ffprobe -v error -select_streams v:0 -show_entries format=duration -of csv=p=0 "$video" 2>/dev/null || echo 0)
	python3 - "$duration" "$seg_sec" <<'PY'
import math, sys
try:
		duration = float(sys.argv[1])
except (ValueError, TypeError):
		duration = 0.0
try:
		seg = int(sys.argv[2])
except ValueError:
		seg = 3600
if seg <= 0:
		seg = 3600
count = max(1, math.ceil(duration / seg)) if duration > 0 else 1
print(int(count))
PY
}

ensure_dir() {
	local path="$1"
	mkdir -p "$path"
	chmod 2775 "$path"
}

escape_sed() {
	printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

replace_placeholders() {
	local file="$1"
	shift
	while (( $# )); do
		local key="$1"
		local value="$2"
		shift 2
		sed -i "s#__${key}__#$(escape_sed "$value")#g" "$file"
	done
}

host_resolves() {
	local host="$1"
	if [[ -z "$host" ]]; then
		return 1
	fi
	if python3 - "$host" <<'PY' >/dev/null 2>&1; then
import socket, sys
try:
	socket.getaddrinfo(sys.argv[1], None)
except socket.gaierror:
	sys.exit(1)
PY
		return 0
	fi
	return 1
}

# --- Section: Host selection helpers ---

select_resolvable_host() {
	local label="$1"
	local raw="$2"
	local normalized
	local host
	local first=""
	local chosen=""

	normalized=${raw//$'\t\n' /,}
	normalized=${normalized//,,/,}
	IFS=',' read -ra hosts <<< "$normalized"
	for host in "${hosts[@]}"; do
		# trim leading/trailing whitespace
		host="${host#${host%%[![:space:]]*}}"
		host="${host%${host##*[![:space:]]}}"
		[[ -z "$host" ]] && continue
		[[ -z "$first" ]] && first="$host"
		if host_resolves "$host"; then
			chosen="$host"
			break
		fi
	done

	if [[ -n "$chosen" ]]; then
		if [[ -n "$first" && "$chosen" != "$first" ]]; then
			echo "[WARN] ${label}: using fallback host $chosen (primary $first unreachable)" >&2
		fi
		echo "$chosen"
		return 0
	fi

	if [[ -n "$first" ]]; then
		echo "[ERR] ${label}: none of the candidates resolved (tried: $raw)" >&2
	else
		echo "[ERR] ${label}: candidate list empty" >&2
	fi
	return 1
}

# --- Section: CLI parsing ---

DIR=""
SAION_NODE="${SAION_NODE:-largegpu}"
SEG_SEC="${SEG_SEC:-3600}"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--dir) DIR="$2"; shift 2 ;;
		--node) SAION_NODE="$2"; shift 2 ;;
		--seg-sec) SEG_SEC="$2"; shift 2 ;;
		*) usage ;;
	esac
done

DIR=${DIR%/}
[[ -n "$DIR" && -d "$DIR" ]] || usage
[[ "$SEG_SEC" =~ ^[0-9]+$ ]] || { echo "[ERR] --seg-sec must be an integer (seconds)" >&2; exit 2; }

# --- Section: Runtime configuration ---

ENC_CONCURRENCY="${ENC_CONCURRENCY:-16}"
ARUCO_CONCURRENCY="${ARUCO_CONCURRENCY:-16}"
SLEAP_CONCURRENCY="${SLEAP_CONCURRENCY:-8}"
CHUNK_BUFFER="${CHUNK_BUFFER:-2}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-2000}"
SENTINEL_TIMEOUT="${SENTINEL_TIMEOUT:-86400}"

for var in ENC_CONCURRENCY ARUCO_CONCURRENCY SLEAP_CONCURRENCY CHUNK_BUFFER MAX_ARRAY_TASKS SENTINEL_TIMEOUT; do
	[[ "${!var}" =~ ^[0-9]+$ ]] || { echo "[ERR] $var must be numeric" >&2; exit 2; }
done

if [[ -n "$BATCH_SIZE" && ! "$BATCH_SIZE" =~ ^[0-9]+$ ]]; then
	echo "[ERR] BATCH_SIZE must be numeric" >&2
	exit 2
fi

case "$SAION_NODE" in
	gpu) saion_gres="gpu:v100:1" ;;
	largegpu) saion_gres="gpu:1" ;;
	*) saion_gres="gpu:1" ;;
esac

# --- Section: Path setup ---

base_folder="$(basename "$DIR")"
data_folder="$DIR/data"
flash_root="/flash/ReiterU/ant_tmp/$base_folder"
saion_root="/work/ReiterU/ant_tmp/$base_folder"
jobs_root="/flash/ReiterU/ant_tmp/jobs/$base_folder"
saion_jobs_root="/work/ReiterU/ant_tmp/jobs/$base_folder"
sentinel_root="$data_folder/sentinels"
aruco_bucket_root="$data_folder"
aruco_flash_root="$flash_root/aruco"

# --- Section: Directory preparation ---

ensure_dir "$data_folder"
ensure_dir "$flash_root"
ensure_dir "$jobs_root"
ensure_dir "$sentinel_root"
ensure_dir "$aruco_bucket_root"
ensure_dir "$aruco_flash_root"

# --- Section: Dependency checks ---

require_cmd ffmpeg
require_cmd ffprobe
require_cmd sbatch
require_cmd rsync
require_cmd python3

SLEAP_MODEL_CENTROID="${SLEAP_MODEL_CENTROID:-/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centroid/training_config.json}"
SLEAP_MODEL_INSTANCE="${SLEAP_MODEL_INSTANCE:-/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centered_instance/training_config.json}"
SLEAP2H5_SCRIPT="${SLEAP2H5_SCRIPT:-$SCRIPT_DIR/sleap2h5.py}"
SLEAP2CSV_SCRIPT="${SLEAP2CSV_SCRIPT:-$SCRIPT_DIR/sleap2csv.py}"
SLEAP_MODULE="${SLEAP_MODULE:-sleap/1.4.1}"
SAION_COLLECT_PARTITION="${SAION_COLLECT_PARTITION:-test-gpu}"
ARUCO_SCRIPT="${ARUCO_SCRIPT:-$SCRIPT_DIR/run_aruco.py}"
ARUCO_ENV_ACTIVATE="${ARUCO_ENV_ACTIVATE:-module load opencv/4.9.0}"
RSYNC_FLAGS="--chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw"
BUCKET_WRITE_HOST_CANDIDATES="${BUCKET_WRITE_HOST_CANDIDATES:-datacp,deigo-login1,deigo-login2,deigo-login3}"
SAION_BUCKET_HOST_CANDIDATES="${SAION_BUCKET_HOST_CANDIDATES:-saion-login1,saion-login2,saion-login3}"
DATA_COPY_PARTITION="${DATA_COPY_PARTITION:-datacp}"

# Determine time limits based on partition
case "$SAION_NODE" in
	test-gpu) sleap_time="0-08:00:00" ;;
	largegpu) sleap_time="1-00:00:00" ;;
	gpu)      sleap_time="2-00:00:00" ;;
	*)        sleap_time="0-08:00:00" ;;
esac

case "$SAION_COLLECT_PARTITION" in
	test-gpu) collect_time="0-08:00:00" ;;
	largegpu) collect_time="1-00:00:00" ;;
	gpu)      collect_time="2-00:00:00" ;;
	*)        collect_time="0-08:00:00" ;;
esac

bucket_host_source="${BUCKET_WRITE_HOST:-}"
[[ -n "$bucket_host_source" ]] || bucket_host_source="$BUCKET_WRITE_HOST_CANDIDATES"
BUCKET_WRITE_HOST="$(select_resolvable_host "Deigo bucket host" "$bucket_host_source")" || { echo "[ERR] Failed to resolve BUCKET_WRITE_HOST" >&2; exit 2; }
echo "[INFO] Deigo bucket host: $BUCKET_WRITE_HOST" >&2

saion_host_source="${SAION_BUCKET_HOST:-}"
[[ -n "$saion_host_source" ]] || saion_host_source="$SAION_BUCKET_HOST_CANDIDATES"
SAION_BUCKET_HOST="$(select_resolvable_host "Saion bucket host" "$saion_host_source")" || { echo "[ERR] Failed to resolve SAION_BUCKET_HOST" >&2; exit 2; }
echo "[INFO] Saion bucket host: $SAION_BUCKET_HOST" >&2

[[ -n "$DATA_COPY_PARTITION" ]] || { echo "[ERR] DATA_COPY_PARTITION must be set" >&2; exit 2; }

[[ -f "$ARUCO_SCRIPT" ]] || echo "[WARN] ArUco script not found at $ARUCO_SCRIPT" >&2

# --- Section: Source video scan ---

videos=( "$DIR"/*.avi "$DIR"/*.mkv )
(( ${#videos[@]} > 0 )) || { echo "[WARN] No .avi or .mkv videos found in $DIR" >&2; exit 0; }

# --- Section: Job Rate Limiting ---

MAX_SUBMITTED_JOBS="${MAX_SUBMITTED_JOBS:-1500}"

get_job_count() {
	squeue -u "$USER" -h -r | wc -l
}

# --- Section: Per-video pipeline orchestration ---

for video in "${videos[@]}"; do
	# Check job limit (Simplified for staged submission: only counting split jobs)
	# Since we only submit 1 job per video initially, we can use a simpler check or rely on SLURM's internal queuing.
	# However, to avoid flooding the queue with thousands of split jobs, we keep a loose check.
	while true; do
		current_jobs=$(get_job_count)
		if (( current_jobs < MAX_SUBMITTED_JOBS )); then
			break
		fi
		echo "[INFO] Job limit reached ($current_jobs/$MAX_SUBMITTED_JOBS). Waiting 60s..."
		sleep 60
	done

	b="$(basename "$video")"
	[[ "$b" =~ ^\. ]] && continue
	[[ "$b" =~ _renc\.(avi|mkv)$ || "$b" =~ _nvenc\.(avi|mkv)$ ]] && continue
	[[ "$b" =~ ^global_cam ]] && continue
	vname="${b%.*}"

	chunk_count=$(calc_chunk_count "$video" "$SEG_SEC")
	[[ "$chunk_count" =~ ^[0-9]+$ ]] || chunk_count=1
	(( chunk_count >= 1 )) || chunk_count=1
	(( chunk_count >= 1 )) || chunk_count=1
	
	# Add buffer to chunk count before batching
	chunk_count_buffered=$(( chunk_count + CHUNK_BUFFER ))
	
	if [[ -n "$BATCH_SIZE" ]]; then
		b_size="$BATCH_SIZE"
	else
		if (( chunk_count_buffered > MAX_ARRAY_TASKS )); then
			# Auto-scale batch size to keep tasks within limit
			b_size=$(( (chunk_count_buffered + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS ))
		else
			b_size=1
		fi
	fi

	# Ceiling division for array count
	array_count=$(( (chunk_count_buffered + b_size - 1) / b_size ))
	array_upper=$(( array_count - 1 ))
	
	(( array_upper >= 0 )) || array_upper=0

	video_flash_dir="$flash_root/$vname"
	video_job_dir="$jobs_root/$vname"
	manifest_path="$video_flash_dir/${vname}_raw_manifest.txt"
	frame_dir="$video_flash_dir/frame_counts"
	summary_file="$video_job_dir/pipeline.jobs"

	remote_root="$saion_root/$vname"
	remote_input="$remote_root/input"
	remote_output="$remote_root/output"
	remote_logs="$saion_jobs_root/$vname"

	enc_ok="$sentinel_root/${vname}.encode.ok"
	aruco_ok="$sentinel_root/${vname}.aruco.ok"
	sleap_submit_ok="$sentinel_root/${vname}.sleap.submit.ok"
	sleap_done="$sentinel_root/${vname}.sleap.ok"
	cleanup_ok="$sentinel_root/${vname}.cleanup.ok"
	enc_ok_dir="$(dirname "$enc_ok")"
	aruco_ok_dir="$(dirname "$aruco_ok")"
	sleap_submit_ok_dir="$(dirname "$sleap_submit_ok")"
	sleap_done_dir="$(dirname "$sleap_done")"
	cleanup_ok_dir="$(dirname "$cleanup_ok")"

	rm -f "$enc_ok" "$aruco_ok" "$sleap_submit_ok" "$sleap_done" "$cleanup_ok"

	ensure_dir "$video_flash_dir"
	ensure_dir "$video_job_dir"
	ensure_dir "$frame_dir"
	aruco_flash_dir="$aruco_flash_root/$vname"
	aruco_bucket_dir="$aruco_bucket_root"
	ensure_dir "$aruco_flash_dir"
	ensure_dir "$aruco_bucket_dir"

	: > "$summary_file"
	chmod 664 "$summary_file"

# --- Stage: Template script paths ---

	split_script="$video_job_dir/split-$vname.sh"
	encode_script="$video_job_dir/encode-$vname.sh"
	enc_finalize_script="$video_job_dir/encode-finalize-$vname.sh"
	aruco_script_path="$video_job_dir/aruco-$vname.sh"
	aruco_finalize_script="$video_job_dir/aruco-finalize-$vname.sh"
	bridge_script="$video_job_dir/bridge-$vname.sh"
	cleanup_script="$video_job_dir/cleanup-$vname.sh"

	# --- Stage Script: Split raw segments ---
	cat > "$split_script" <<'EOS'
#!/bin/bash -l
#SBATCH -t 0-2
#SBATCH -c 4
#SBATCH --partition=short
#SBATCH --mem=8G
#SBATCH -J split-__BASE__
#SBATCH -o __JOBDIR__/split-__BASE___%j.out
#SBATCH -e __JOBDIR__/split-__BASE___%j.err
set -eo pipefail
shopt -s nullglob
module load ffmpeg/7.1	

video="__VIDEO__"
flash_dir="__FLASH_DIR__"
manifest="__MANIFEST__"
seg_sec=__SEG_SEC__

mkdir -p "$flash_dir"
chmod 2775 "$flash_dir"

if compgen -G "$flash_dir/__BASE___raw_*.avi" > /dev/null; then
	echo "[SKIP] raw chunks already present for $video"
else
	ffmpeg -hide_banner -y -i "$video" \
		-c copy -bsf:v h264_mp4toannexb -map 0:v:0 -f segment -segment_time "$seg_sec" \
		-reset_timestamps 1 "$flash_dir/__BASE___raw_%03d.avi"
	fi

raw_chunks=("$flash_dir"/__BASE___raw_*.avi)
if (( ${#raw_chunks[@]} )); then
	printf '%s\n' "${raw_chunks[@]}" | sort > "$manifest"
else
	: > "$manifest"
fi
chmod 664 "$manifest"
# Submit next stage: Encode Array & Sync
jid_enc=$(sbatch --parsable "__ENCODE_SCRIPT__")
echo "Submitted Encode Array: $jid_enc"
sbatch --dependency=afterok:$jid_enc "__ENC_FINALIZE_SCRIPT__"
echo "Submitted Encode Sync (dep: $jid_enc)"
EOS

	# --- Stage Script: Encode segments ---
	cat > "$encode_script" <<'EOS'
#!/bin/bash -l
#SBATCH -t 0-16
#SBATCH -c 8
#SBATCH --partition=compute
#SBATCH --mem=16G
#SBATCH -J enc-__BASE__
#SBATCH -o __JOBDIR__/enc-__BASE___%A_%a.out
#SBATCH -e __JOBDIR__/enc-__BASE___%A_%a.err
#SBATCH --array=0-__ARRAY_MAX__%__ENC_CONCURRENCY__
set -eo pipefail

source ~/.bashrc
module load ffmpeg/7.1

flash_dir="__FLASH_DIR__"
frame_dir="$flash_dir/frame_counts"
mkdir -p "$frame_dir"
chmod 2775 "$frame_dir"

batch_size=__BATCH_SIZE__
start_idx=$(( SLURM_ARRAY_TASK_ID * batch_size ))
end_idx=$(( start_idx + batch_size ))

for (( i=start_idx; i<end_idx; i++ )); do
	idx=$(printf "%03d" "$i")
	raw="$flash_dir/__BASE___raw_${idx}.avi"
	out="$flash_dir/__BASE___${idx}.avi"

	if [[ ! -s "$raw" ]]; then
		echo "[SKIP] no raw chunk $raw"
		continue
	fi

	ffmpeg -hide_banner -y -i "$raw" -c:v libx264 -pix_fmt yuv420p \
		-preset fast -crf 23 -threads 8 "$out"

	nb=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
			-of default=nk=1:nw=1 "$out" 2>/dev/null || echo 0)

	tmp_csv="$frame_dir/${idx}.csv"
	printf "%s,%s\n" "__BASE___${idx}.avi" "$nb" > "$tmp_csv"
	chmod 664 "$tmp_csv"
	touch "$flash_dir/__BASE___raw_${idx}.encoded.ok"
	rm -f "$raw"
done
EOS

	# --- Stage Script: Finalize encoding outputs ---
	cat > "$enc_finalize_script" <<'EOS'
#!/bin/bash -l
#SBATCH -t 0-2
#SBATCH -c 2
#SBATCH --partition=short
#SBATCH --mem=8G
#SBATCH -J encfin-__BASE__
#SBATCH -o __JOBDIR__/encfin-__BASE___%j.out
#SBATCH -e __JOBDIR__/encfin-__BASE___%j.err
set -eo pipefail
shopt -s nullglob

flash_dir="__FLASH_DIR__"
frame_dir="$flash_dir/frame_counts"
	data_dir="__DATA_DIR__"
	enc_ok="__ENC_OK__"
	enc_ok_dir="__ENC_OK_DIR__"
	manifest="__MANIFEST__"
	copy_partition="__DATA_COPY_PARTITION__"
	job_tmp_dir="__JOBDIR__"

	raw_chunks=("$flash_dir"/__BASE___raw_*.avi)
	if (( ${#raw_chunks[@]} )); then
		printf '%s\n' "${raw_chunks[@]}" | sort > "$manifest"
	else
		: > "$manifest"
	fi
	chmod 664 "$manifest"

	frame_csv="$flash_dir/__BASE___frame_counts.csv"

	if compgen -G "$frame_dir/"*.csv > /dev/null; then
		python3 - "$frame_dir" "$frame_csv" <<'PY'
import pathlib, sys
frame_dir = pathlib.Path(sys.argv[1])
out_csv = pathlib.Path(sys.argv[2])
rows = []
for path in sorted(frame_dir.glob("*.csv")):
		rows.extend(line.strip() for line in path.open("r", encoding="utf-8") if line.strip())
out_csv.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
PY
		chmod 664 "$frame_csv"
		rm -f "$frame_dir/"*.csv
	else
		: > "$frame_csv"
fi

rm -f "$flash_dir"/__BASE___raw_*.encoded.ok

copy_script=$(mktemp -p "$job_tmp_dir" "__BASE___encsync_XXXXXX.sh")
cat > "$copy_script" <<'COPY'
#!/bin/bash -l
set -eo pipefail
umask 0002

flash_dir="__FLASH_DIR__"
data_dir="__DATA_DIR__"
enc_ok_dir="__ENC_OK_DIR__"
enc_ok="__ENC_OK__"

mkdir -p "$data_dir"
rsync -avh __RSYNC_FLAGS__ \
	--exclude="__BASE___raw_*.avi" \
	--include="__BASE___*.avi" \
	--include="__BASE___frame_counts.csv" \
	--exclude="*" "$flash_dir/" "$data_dir/"
mkdir -p "$enc_ok_dir"
: > "$enc_ok"
COPY
chmod +x "$copy_script"

if ! sbatch --wait -p "$copy_partition" -J "encsync-__BASE__" \
	-o "${job_tmp_dir}/datacp-encsync_%j.out" \
	-e "${job_tmp_dir}/datacp-encsync_%j.err" "$copy_script"; then
	echo "[ERR] datacp copy job failed for __BASE__" >&2
	rm -f "$copy_script"
	exit 1
fi

rm -f "$copy_script"

rm -f "$frame_csv"

# Submit next stages: Bridge, ArUco, Cleanup
jid_bridge=$(sbatch --parsable "__BRIDGE_SCRIPT__")
echo "Submitted Bridge: $jid_bridge"

jid_aruco=$(sbatch --parsable "__ARUCO_SCRIPT_PATH__")
echo "Submitted ArUco Array: $jid_aruco"

jid_aruco_sync=$(sbatch --parsable --dependency=afterok:$jid_aruco "__ARUCO_FINALIZE_SCRIPT__")
echo "Submitted ArUco Sync: $jid_aruco_sync"

sbatch --dependency=afterok:$jid_bridge:$jid_aruco_sync "__CLEANUP_SCRIPT__"
echo "Submitted Cleanup (dep: $jid_bridge, $jid_aruco_sync)"
EOS

	# --- Stage Script: Run ArUco detection ---
	cat > "$aruco_script_path" <<'EOS'
#!/bin/bash -l
#SBATCH -t 0-12
#SBATCH -c 4
#SBATCH --partition=compute
#SBATCH --mem=8G
#SBATCH -J aruco-__BASE__
#SBATCH -o __JOBDIR__/aruco-__BASE___%A_%a.out
#SBATCH -e __JOBDIR__/aruco-__BASE___%A_%a.err
#SBATCH --array=0-__ARRAY_MAX__%__ARUCO_CONCURRENCY__
set -eo pipefail

	__ARUCO_ENV_ACTIVATE__

	segment_dir="__FLASH_DIR__"
	output_dir="__ARUCO_FLASH_DIR__"
mkdir -p "$output_dir"
chmod 2775 "$output_dir"

batch_size=__BATCH_SIZE__
start_idx=$(( SLURM_ARRAY_TASK_ID * batch_size ))
end_idx=$(( start_idx + batch_size ))

for (( i=start_idx; i<end_idx; i++ )); do
	idx=$(printf "%03d" "$i")
	video_path="$segment_dir/__BASE___${idx}.avi"
	if [[ ! -s "$video_path" ]]; then
		echo "[SKIP] missing encoded segment $video_path"
		continue
	fi

	python3 "__ARUCO_SCRIPT__" \
		--video-file "$video_path" \
		--output-path "$output_dir/"

	touch "$output_dir/__BASE___${idx}.aruco.ok"
done
EOS

	# --- Stage Script: Sync ArUco outputs ---
	cat > "$aruco_finalize_script" <<'EOS'
#!/bin/bash -l
#SBATCH -t 0-2
#SBATCH -c 2
#SBATCH --partition=short
#SBATCH --mem=4G
#SBATCH -J arucofin-__BASE__
#SBATCH -o __JOBDIR__/arucofin-__BASE__%j.out
#SBATCH -e __JOBDIR__/arucofin-__BASE__%j.err
set -eo pipefail
shopt -s nullglob

flash_dir="__ARUCO_FLASH_DIR__"
bucket_dir="__ARUCO_BUCKET_DIR__"
aruco_ok="__ARUCO_OK__"
aruco_ok_dir="__ARUCO_OK_DIR__"
copy_partition="__DATA_COPY_PARTITION__"
job_tmp_dir="__JOBDIR__"

copy_script=$(mktemp -p "$job_tmp_dir" "__BASE___aruco_sync_XXXXXX.sh")
cat > "$copy_script" <<'COPY'
#!/bin/bash -l
set -eo pipefail
umask 0002

flash_dir="__ARUCO_FLASH_DIR__"
bucket_dir="__ARUCO_BUCKET_DIR__"
aruco_ok_dir="__ARUCO_OK_DIR__"
aruco_ok="__ARUCO_OK__"

mkdir -p "$bucket_dir"
rsync -avh __RSYNC_FLAGS__ \
	--include="__BASE___*_aruco_tracks_.h5" \
	--exclude="*" "$flash_dir/" "$bucket_dir/"
mkdir -p "$aruco_ok_dir"
: > "$aruco_ok"
COPY
chmod +x "$copy_script"

if ! sbatch --wait -p "$copy_partition" -J "arucofin-__BASE__" \
	-o "${job_tmp_dir}/datacp-aruco_%j.out" \
	-e "${job_tmp_dir}/datacp-aruco_%j.err" "$copy_script"; then
	echo "[ERR] datacp ArUco sync job failed for __BASE__" >&2
	rm -f "$copy_script"
	exit 1
fi

rm -f "$copy_script"

rm -f "$flash_dir"/__BASE___*.aruco.ok
rm -f "$flash_dir"/__BASE___*_aruco_tracks_.h5
EOS

	# --- Stage Script: Bridge to Saion ---
	cat > "$bridge_script" <<'EOS'
#!/bin/bash -l
#SBATCH -t 0-2
#SBATCH -c 2
#SBATCH --partition=short
#SBATCH --mem=8G
#SBATCH -J bridge-__BASE__
#SBATCH -o __JOBDIR__/bridge-__BASE__%j.out
#SBATCH -e __JOBDIR__/bridge-__BASE__%j.err
set -eo pipefail
shopt -s nullglob

data_dir="__DATA_DIR__"
flash_dir="__FLASH_DIR__"
remote_root="__REMOTE_ROOT__"
remote_input="__REMOTE_INPUT__"
remote_output="__REMOTE_OUTPUT__"
remote_logs="__REMOTE_LOGS__"
saion_node="__SAION_NODE__"
sleap_module="__SLEAP_MODULE__"
model1="__MODEL1__"
model2="__MODEL2__"
sleap2h5="__SLEAP2H5__"
sleap2csv="__SLEAP2CSV__"
sleap_submit_ok="__SLEAP_SUBMIT_OK__"
sleap_done_ok="__SLEAP_DONE_OK__"
sleap_submit_ok_dir="__SLEAP_SUBMIT_OK_DIR__"
rsync_flags="__RSYNC_FLAGS__"
chunk_max=__ARRAY_MAX__
expected_chunks=__CHUNK_COUNT__
sleap_concurrency=__SLEAP_CONCURRENCY__
batch_size=__BATCH_SIZE__
summary_file="__SUMMARY_FILE__"
copy_partition="__DATA_COPY_PARTITION__"
job_tmp_dir="__JOBDIR__"

SSH_CMD=(ssh -x -oBatchMode=yes -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null)
export RSYNC_RSH="${SSH_CMD[*]}"

"${SSH_CMD[@]}" saion "mkdir -p '$remote_input' '$remote_output' '$remote_logs'"

echo "[INFO] staging encoded segments to Saion: $remote_input"
rsync -avh $rsync_flags \
	--exclude="__BASE___raw_*.avi" \
	--include="__BASE___*.avi" \
	--exclude="*" "$flash_dir/" "saion:$remote_input/"

"${SSH_CMD[@]}" saion "cat > '$remote_root/sleap_array.sh'" <<'SAION'
#!/bin/bash -l
#SBATCH -t __SLEAP_TIME__
#SBATCH -c 8
#SBATCH --partition=__SAION_NODE__
#SBATCH --mem=128G
#SBATCH --gres=__SAION_GRES__
#SBATCH -J sleap-__BASE__
#SBATCH -o __REMOTE_LOGS__/sleap_%A_%a.out
#SBATCH -e __REMOTE_LOGS__/sleap_%A_%a.err
#SBATCH --array=0-__ARRAY_MAX__%__SLEAP_CONCURRENCY__
set -eo pipefail

SSH_CMD=(ssh -x -oBatchMode=yes -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null)
export RSYNC_RSH="${SSH_CMD[*]}"

source ~/.bashrc
module load __SLEAP_MODULE__
base="__BASE__"
input_dir="__REMOTE_INPUT__"
output_dir="__REMOTE_OUTPUT__"
bucket_dir="__DATA_DIR__"
bucket_host="__SAION_BUCKET_HOST__"
batch_size=__BATCH_SIZE__

start_idx=$(( SLURM_ARRAY_TASK_ID * batch_size ))
end_idx=$(( start_idx + batch_size ))

for (( i=start_idx; i<end_idx; i++ )); do
	idx=$(printf "%03d" "$i")
	input="$input_dir/${base}_${idx}.avi"
	if [[ ! -s "$input" ]]; then
		echo "[SKIP] missing input $input"
		continue
	fi

	out_slp="$output_dir/${base}_${idx}.slp"
	out_h5="$output_dir/${base}_${idx}_sleap_data.h5"
	out_csv="$output_dir/${base}_${idx}_sleap_data.csv"

	sleap-track "$input" -m "__MODEL1__" -m "__MODEL2__" \
		--tracking.tracker none -o "$out_slp" \
		--verbosity json --no-empty-frames --batch_size 2

	python3 "__SLEAP2H5__" "$out_slp" "$output_dir"
	python3 "__SLEAP2CSV__" "$out_slp" "$output_dir"

	"${SSH_CMD[@]}" "$bucket_host" "mkdir -p '$bucket_dir'"

	rsync -avh --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw \
		"$out_slp" "$out_h5" "$out_csv" \
		"$bucket_host:$bucket_dir/"
done
SAION

"${SSH_CMD[@]}" saion "cat > '$remote_root/sleap_collect.sh'" <<'SAION'
#!/bin/bash -l
#SBATCH -t __COLLECT_TIME__
#SBATCH -c 2
#SBATCH --partition=__SAION_COLLECT_PARTITION__
#SBATCH --mem=16G
#SBATCH -J sleapfin-__BASE__
#SBATCH -o __REMOTE_LOGS__/sleapfin_%j.out
#SBATCH -e __REMOTE_LOGS__/sleapfin_%j.err
set -eo pipefail

SSH_CMD=(ssh -x -oBatchMode=yes -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null)
export RSYNC_RSH="${SSH_CMD[*]}"

source ~/.bashrc
remote_root="__REMOTE_ROOT__"
output_dir="__REMOTE_OUTPUT__"
bucket_dir="__DATA_DIR__"
sleap_done="__SLEAP_DONE_OK__"
sleap_done_dir="__SLEAP_DONE_OK_DIR__"
bucket_host="__SAION_BUCKET_HOST__"

"${SSH_CMD[@]}" "$bucket_host" "mkdir -p '$bucket_dir'"

rsync -avh --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw \
	--include="__BASE___*.slp" \
	--include="__BASE___*.h5" \
	--include="__BASE___*_sleap_data.csv" \
	--exclude="*" "$output_dir/" "$bucket_host:$bucket_dir/"

"${SSH_CMD[@]}" "$bucket_host" "bash -lc 'umask 0002 && mkdir -p \"$sleap_done_dir\" && : > \"$sleap_done\"'"

# Local cleanup on Saion
rm -rf "$remote_root"
SAION

"${SSH_CMD[@]}" saion "chmod +x '$remote_root/sleap_array.sh' '$remote_root/sleap_collect.sh'"

sleap_array_job=$("${SSH_CMD[@]}" saion "sbatch --parsable '$remote_root/sleap_array.sh'")
sleap_collect_job=$("${SSH_CMD[@]}" saion "sbatch --dependency=afterok:$sleap_array_job --parsable '$remote_root/sleap_collect.sh'")

exec 9>>"$summary_file"
if flock 9; then
	printf '%-16s %s\n' 'saion-array' "$sleap_array_job" >&9
	printf '%-16s %s\n' 'saion-collect' "$sleap_collect_job" >&9
	flock -u 9
fi
exec 9>&-

copy_script=$(mktemp -p "$job_tmp_dir" "__BASE___sleap_submit_mark_XXXXXX.sh")
cat > "$copy_script" <<'COPY'
#!/bin/bash -l
set -eo pipefail
umask 0002

mkdir -p "__SLEAP_SUBMIT_OK_DIR__"
: > "__SLEAP_SUBMIT_OK__"
COPY
chmod +x "$copy_script"

if ! sbatch --wait -p "$copy_partition" -J "sleapmark-__BASE__" \
	-o "${job_tmp_dir}/datacp-sleapmark_%j.out" \
	-e "${job_tmp_dir}/datacp-sleapmark_%j.err" "$copy_script"; then
	echo "[ERR] datacp sleap submit marker job failed for __BASE__" >&2
	rm -f "$copy_script"
	exit 1
fi

rm -f "$copy_script"

echo "[INFO] submitted SLEAP array $sleap_array_job and collect $sleap_collect_job for $expected_chunks chunks (array 0-$chunk_max%$sleap_concurrency)"
EOS

	# --- Stage Script: Cleanup sentinels and storage ---
	cat > "$cleanup_script" <<'EOS'
#!/bin/bash -l
#SBATCH -t 0-2
#SBATCH -c 2
#SBATCH --partition=short
#SBATCH --mem=8G
#SBATCH -J cleanup-__BASE__
#SBATCH -o __JOBDIR__/cleanup-__BASE__%j.out
#SBATCH -e __JOBDIR__/cleanup-__BASE__%j.err
set -eo pipefail

sleap_done="__SLEAP_DONE_OK__"
aruco_ok="__ARUCO_OK__"
cleanup_ok="__CLEANUP_OK__"
cleanup_ok_dir="__CLEANUP_OK_DIR__"
flash_dir="__FLASH_DIR__"
aruco_flash_dir="__ARUCO_FLASH_DIR__"
flash_root="__FLASH_ROOT__"
aruco_flash_root="__ARUCO_FLASH_ROOT__"
remote_root="__REMOTE_ROOT__"
saion_root="__SAION_ROOT__"
summary_file="__SUMMARY_FILE__"
timeout_secs=__SENTINEL_TIMEOUT__
copy_partition="__DATA_COPY_PARTITION__"
job_tmp_dir="__JOBDIR__"

SSH_CMD=(ssh -x -oBatchMode=yes -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null)

wait_for_file() {
	local path="$1"
	local label="$2"
	local deadline=$(( $(date +%s) + timeout_secs ))
	while (( $(date +%s) <= deadline )); do
		if [[ -f "$path" ]]; then
			return 0
		fi
		sleep 60
	done
	echo "[ERR] timeout waiting for $label ($path)" >&2
	return 1
}

wait_for_file "$aruco_ok" "ArUco completion marker"
# Note: We do not wait for SLEAP completion here anymore.
# Saion cleanup is handled locally on Saion.

echo "[INFO] removing flash directory $flash_dir"
rm -rf "$flash_dir"

echo "[INFO] removing ArUco flash directory $aruco_flash_dir"
rm -rf "$aruco_flash_dir"

if [[ -d "$aruco_flash_root" ]]; then
	rmdir "$aruco_flash_root" 2>/dev/null || true
fi

if [[ -d "$flash_root" ]]; then
	rmdir "$flash_root" 2>/dev/null || true
fi

copy_script=$(mktemp -p "$job_tmp_dir" "__BASE___cleanup_mark_XXXXXX.sh")
cat > "$copy_script" <<'COPY'
#!/bin/bash -l
set -eo pipefail
umask 0002

mkdir -p "__CLEANUP_OK_DIR__"
: > "__CLEANUP_OK__"
COPY
chmod +x "$copy_script"

if ! sbatch --wait -p "$copy_partition" -J "cleanupmark-__BASE__" \
	-o "${job_tmp_dir}/datacp-cleanup_%j.out" \
	-e "${job_tmp_dir}/datacp-cleanup_%j.err" "$copy_script"; then
	echo "[ERR] datacp cleanup sentinel job failed for __BASE__" >&2
	rm -f "$copy_script"
	exit 1
fi

rm -f "$copy_script"

if [[ -f "$summary_file" ]]; then
	echo "[INFO] job summary:"
	cat "$summary_file"
	deigo_jobs=$(awk '$2 ~ /^[0-9]+$/ {print $2}' "$summary_file" | paste -sd, -)
	if [[ -n "$deigo_jobs" ]]; then
		sacct --jobs="$deigo_jobs" --format=JobID,JobName%20,State%12,Elapsed -n || true
	fi
fi
EOS

# --- Stage: Parameterize templates ---

	replace_placeholders "$split_script" \
		BASE "$vname" JOBDIR "$video_job_dir" VIDEO "$video" \
		SEG_SEC "$SEG_SEC" FLASH_DIR "$video_flash_dir" \
		ENCODE_SCRIPT "$encode_script" \
		ENC_FINALIZE_SCRIPT "$enc_finalize_script" \
		MANIFEST "$manifest_path" \
		STAGED_SUBMISSION "$STAGED_SUBMISSION"

	replace_placeholders "$encode_script" \
		BASE "$vname" JOBDIR "$video_job_dir" ARRAY_MAX "$array_upper" \
		ENC_CONCURRENCY "$ENC_CONCURRENCY" FLASH_DIR "$video_flash_dir" \
		BATCH_SIZE "$b_size"

	replace_placeholders "$enc_finalize_script" \
		BASE "$vname" JOBDIR "$video_job_dir" FLASH_DIR "$video_flash_dir" \
		BUCKET_DIR "$bucket_dir" BUCKET_HOST "$BUCKET_WRITE_HOST" \
		DATA_DIR "$data_folder" ENC_OK "$enc_ok" \
		BRIDGE_SCRIPT "$bridge_script" \
		ARUCO_SCRIPT_PATH "$aruco_script_path" \
		ARUCO_FINALIZE_SCRIPT "$aruco_finalize_script" \
		CLEANUP_SCRIPT "$cleanup_script" ENC_OK_DIR "$enc_ok_dir" \
		RSYNC_FLAGS "$RSYNC_FLAGS" MANIFEST "$manifest_path" DATA_COPY_PARTITION "$DATA_COPY_PARTITION" \
		STAGED_SUBMISSION "$STAGED_SUBMISSION"

	replace_placeholders "$aruco_script_path" \
		BASE "$vname" JOBDIR "$video_job_dir" ARRAY_MAX "$array_upper" \
		ARUCO_CONCURRENCY "$ARUCO_CONCURRENCY" FLASH_DIR "$video_flash_dir" \
		ARUCO_FLASH_DIR "$aruco_flash_dir" ARUCO_SCRIPT "$ARUCO_SCRIPT" \
		ARUCO_ENV_ACTIVATE "$ARUCO_ENV_ACTIVATE" \
		BATCH_SIZE "$b_size" RSYNC_FLAGS "$RSYNC_FLAGS" \
		ARUCO_BUCKET_DIR "$aruco_bucket_root"

	replace_placeholders "$aruco_finalize_script" \
		BASE "$vname" JOBDIR "$video_job_dir" ARUCO_FLASH_DIR "$aruco_flash_dir" \
		ARUCO_BUCKET_DIR "$aruco_bucket_dir" ARUCO_OK "$aruco_ok" \
		ARUCO_OK_DIR "$aruco_ok_dir" RSYNC_FLAGS "$RSYNC_FLAGS" DATA_COPY_PARTITION "$DATA_COPY_PARTITION"

	replace_placeholders "$bridge_script" \
		BASE "$vname" JOBDIR "$video_job_dir" DATA_DIR "$data_folder" FLASH_DIR "$video_flash_dir" \
		REMOTE_ROOT "$remote_root" REMOTE_INPUT "$remote_input" REMOTE_OUTPUT "$remote_output" \
		REMOTE_LOGS "$remote_logs" SAION_NODE "$SAION_NODE" SLEAP_MODULE "$SLEAP_MODULE" \
		MODEL1 "$SLEAP_MODEL_CENTROID" MODEL2 "$SLEAP_MODEL_INSTANCE" \
		SLEAP2H5 "$SLEAP2H5_SCRIPT" SLEAP2CSV "$SLEAP2CSV_SCRIPT" \
		SLEAP_SUBMIT_OK "$sleap_submit_ok" SLEAP_SUBMIT_OK_DIR "$sleap_submit_ok_dir" \
		SLEAP_DONE_OK "$sleap_done" SLEAP_DONE_OK_DIR "$sleap_done_dir" \
		RSYNC_FLAGS "$RSYNC_FLAGS" ARRAY_MAX "$array_upper" CHUNK_COUNT "$chunk_count" \
		SLEAP_CONCURRENCY "$SLEAP_CONCURRENCY" SUMMARY_FILE "$summary_file" \
		DATA_COPY_PARTITION "$DATA_COPY_PARTITION" SAION_BUCKET_HOST "$SAION_BUCKET_HOST" \
		SAION_GRES "$saion_gres" \
		SAION_COLLECT_PARTITION "$SAION_COLLECT_PARTITION" \
		BATCH_SIZE "$b_size" \
		SLEAP_TIME "$sleap_time" COLLECT_TIME "$collect_time"

	replace_placeholders "$cleanup_script" \
		BASE "$vname" JOBDIR "$video_job_dir" SLEAP_DONE_OK "$sleap_done" \
		ARUCO_OK "$aruco_ok" CLEANUP_OK "$cleanup_ok" CLEANUP_OK_DIR "$cleanup_ok_dir" \
		FLASH_DIR "$video_flash_dir" ARUCO_FLASH_DIR "$aruco_flash_dir" \
		FLASH_ROOT "$flash_root" ARUCO_FLASH_ROOT "$aruco_flash_root" SAION_ROOT "$saion_root" \
		REMOTE_ROOT "$remote_root" SUMMARY_FILE "$summary_file" \
		SENTINEL_TIMEOUT "$SENTINEL_TIMEOUT" DATA_COPY_PARTITION "$DATA_COPY_PARTITION"

# --- Stage: Grant execute permissions ---

	chmod +x "$split_script" "$encode_script" "$enc_finalize_script" \
		"$aruco_script_path" "$aruco_finalize_script" "$bridge_script" "$cleanup_script"

# --- Stage: Submission helpers ---

	log_stage() {
		local stage="$1"
		local job_id="$2"
		printf '%-16s %s\n' "$stage" "$job_id" >> "$summary_file"
	}

# --- Stage:	# --- Submission ---
	
	# Only submit the first stage (Split)
	jid_split=$(sbatch --parsable "$split_script")
	
	echo "Submitted pipeline for $vname (chunks: $chunk_count, array 0-$array_upper)"
	echo "  split        $jid_split"
	
	# Append to summary
	echo "split: $jid_split" >> "$summary_file"
done

echo "Pipelines scheduled. Monitor via: squeue -u $(id -un) and consult per-video summary files under $jobs_root/<video>/pipeline.jobs."
