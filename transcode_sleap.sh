#!/bin/bash -l
#SBATCH -t 4-00:00:00
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --partition=compute
#SBATCH -J orchestrator
#SBATCH -o orchestrator_%j.out
#SBATCH -e orchestrator_%j.err

# ------------------------------------------------------------
# transcode_sleap_aruco.sh — Deigo↔Saion orchestration (ROBUST RESUME)
#
# SINGLE-UNDERSCORE CANONICAL NAMING (FIXED):
#   Encoded chunk:    BASE_###.avi
#   ArUco output:     BASE_###_aruco_tracks_.h5
#   SLEAP outputs:    BASE_###.slp, BASE_###_sleap_data.h5, BASE_###_sleap_data.csv
#
# Notes:
# - Raw split chunks remain internal: BASE_raw_###.avi
# - Coverage checks use bucket outputs as truth.
# - Saion SLEAP array uses conda env sleap15 and SLEAP 1.5.2 (sleap-nn-track).
# ------------------------------------------------------------

set -eo pipefail
shopt -s nullglob
IFS=$'\n\t'

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# -----------------------------
# Helper utilities
# -----------------------------
usage() {
	cat <<'EOT'
Usage:
  Interactive: bash transcode_sleap_aruco.sh --dir <folder> ...
  Batch:       sbatch transcode_sleap_aruco.sh --dir <folder> ...
Options:
  --dir PATH                   (required) Directory containing .avi/.mkv videos to process
  --node PARTITION             Saion GPU partition: gpu, largegpu, test-gpu (default: largegpu)
  --seg-sec SECONDS            Segment duration in seconds for splitting (default: 3600)
  --sleap-version VERSION      SLEAP version: 1.4.1 or 1.5.2 (default: 1.5.2)
  --sleap-model-centroid PATH  Path to SLEAP centroid model training_config.json
  --sleap-model-instance PATH  Path to SLEAP instance model training_config.json

Environment:
  ENC_CONCURRENCY     Max concurrent encoder tasks (default 16)
  ARUCO_CONCURRENCY   Max concurrent ArUco tasks (default 16)
  SLEAP_CONCURRENCY   Max concurrent SLEAP tasks on Saion (default 8)
  CHUNK_BUFFER        Extra indices when sizing arrays (default 2)
  BATCH_SIZE          Override batch size per array task (default auto)
  MAX_ARRAY_TASKS     Cap array upper bound sizing (default 2000)
  MAX_SUBMITTED_JOBS  Throttle total submitted jobs (default 1500)
  SENTINEL_TIMEOUT    Cleanup waits for aruco.ok (default 86400)
  CONDA_ENV_SLEAP     Conda env for SLEAP jobs (default: sleap15)
  ARUCO_CONDA_ENV     Conda env for run_aruco.py (default: aruco_env)
  DATA_COPY_PARTITION Data-copy partition (default: datacp)
EOT
	exit 1
}

require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERR] missing command: $1" >&2; exit 2; }; }

ensure_dir() {
	local path="$1"
	mkdir -p "$path"
	chmod 2775 "$path"
}

escape_sed() { printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'; }

replace_placeholders() {
	local file="$1"; shift
	while (( $# )); do
		local key="$1"; local value="$2"; shift 2
		sed -i "s#__${key}__#$(escape_sed "$value")#g" "$file"
	done
}

calc_chunk_count() {
	local video="$1"; local seg_sec="$2"
	local duration
	duration=$(ffprobe -v error -select_streams v:0 -show_entries format=duration -of csv=p=0 "$video" 2>/dev/null || echo 0)
	python3 - "$duration" "$seg_sec" <<'PY'
import math, sys
try: duration=float(sys.argv[1])
except: duration=0.0
try: seg=int(sys.argv[2])
except: seg=3600
seg=max(seg,1)
print(int(max(1, math.ceil(duration/seg)) if duration>0 else 1))
PY
}

host_resolves() {
	local host="$1"
	[[ -n "$host" ]] || return 1
	python3 - "$host" <<'PY' >/dev/null 2>&1
import socket, sys
try: socket.getaddrinfo(sys.argv[1], None)
except socket.gaierror: raise SystemExit(1)
PY
}

select_resolvable_host() {
	local label="$1" raw="$2"
	local normalized host first="" chosen=""
	normalized=${raw//$'\t\n' /,}
	normalized=${normalized//,,/,}
	IFS=',' read -ra hosts <<< "$normalized"
	for host in "${hosts[@]}"; do
		host="${host#${host%%[![:space:]]*}}"
		host="${host%${host##*[![:space:]]}}"
		[[ -z "$host" ]] && continue
		[[ -z "$first" ]] && first="$host"
		if host_resolves "$host"; then chosen="$host"; break; fi
	done
	if [[ -n "$chosen" ]]; then
		[[ -n "$first" && "$chosen" != "$first" ]] && echo "[WARN] ${label}: using fallback host $chosen (primary $first unreachable)" >&2
		echo "$chosen"; return 0
	fi
	echo "[ERR] ${label}: none of the candidates resolved (tried: $raw)" >&2
	return 1
}

# -----------------------------
# Coverage checks (bucket outputs are truth)
# -----------------------------
check_video_coverage() {
	local base="$1" data_dir="$2" expected_chunks="$3"
	local have_enc=1 have_aruco=1 have_sleap=1
	if (( expected_chunks <= 0 )); then
		echo "enc=0 aruco=0 sleap=0"; return 0
	fi
	for (( i=0; i<expected_chunks; i++ )); do
		local idx; idx=$(printf "%03d" "$i")
		[[ -s "$data_dir/${base}_${idx}.avi" ]] || have_enc=0
		[[ -s "$data_dir/${base}_${idx}_aruco_tracks_.h5" ]] || have_aruco=0
		[[ -s "$data_dir/${base}_${idx}.slp" ]] || have_sleap=0
	done
	echo "enc=$have_enc aruco=$have_aruco sleap=$have_sleap"
}

# -----------------------------
# CLI parsing
# -----------------------------
DIR=""
SAION_NODE="${SAION_NODE:-largegpu}"
SEG_SEC="${SEG_SEC:-3600}"
CLI_SLEAP_VERSION=""
CLI_SLEAP_MODEL_CENTROID=""
CLI_SLEAP_MODEL_INSTANCE=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--dir) DIR="$2"; shift 2 ;;
		--node) SAION_NODE="$2"; shift 2 ;;
		--seg-sec) SEG_SEC="$2"; shift 2 ;;
		--sleap-version) CLI_SLEAP_VERSION="$2"; shift 2 ;;
		--sleap-model-centroid) CLI_SLEAP_MODEL_CENTROID="$2"; shift 2 ;;
		--sleap-model-instance) CLI_SLEAP_MODEL_INSTANCE="$2"; shift 2 ;;
		*) usage ;;
	esac
done

DIR=${DIR%/}
[[ -n "$DIR" && -d "$DIR" ]] || usage
[[ "$SEG_SEC" =~ ^[0-9]+$ ]] || { echo "[ERR] --seg-sec must be integer seconds" >&2; exit 2; }

# -----------------------------
# Runtime configuration
# -----------------------------
ENC_CONCURRENCY="${ENC_CONCURRENCY:-16}"
ARUCO_CONCURRENCY="${ARUCO_CONCURRENCY:-16}"
SLEAP_CONCURRENCY="${SLEAP_CONCURRENCY:-8}"
CHUNK_BUFFER="${CHUNK_BUFFER:-2}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-2000}"
MAX_SUBMITTED_JOBS="${MAX_SUBMITTED_JOBS:-1500}"
SENTINEL_TIMEOUT="${SENTINEL_TIMEOUT:-86400}"

CONDA_ENV_SLEAP="${CONDA_ENV_SLEAP:-sleap15}"
ARUCO_CONDA_ENV="${ARUCO_CONDA_ENV:-aruco_env}"

for var in ENC_CONCURRENCY ARUCO_CONCURRENCY SLEAP_CONCURRENCY CHUNK_BUFFER MAX_ARRAY_TASKS MAX_SUBMITTED_JOBS SENTINEL_TIMEOUT; do
	[[ "${!var}" =~ ^[0-9]+$ ]] || { echo "[ERR] $var must be numeric" >&2; exit 2; }
done
if [[ -n "$BATCH_SIZE" && ! "$BATCH_SIZE" =~ ^[0-9]+$ ]]; then
	echo "[ERR] BATCH_SIZE must be numeric" >&2; exit 2
fi

case "$SAION_NODE" in
	gpu) saion_gres="gpu:v100:1" ;;
	largegpu) saion_gres="gpu:1" ;;
	test-gpu) saion_gres="gpu:1" ;;
	*) saion_gres="gpu:1" ;;
esac

ENV_ACTIVATE="source ~/.bashrc && conda activate ${CONDA_ENV_SLEAP}"
ARUCO_ENV_ACTIVATE="source ~/.bashrc && conda activate ${ARUCO_CONDA_ENV}"

# -----------------------------
# Paths
# -----------------------------
base_folder="$(basename "$DIR")"
data_folder="$DIR/data"
flash_root="/flash/ReiterU/ant_tmp/$base_folder"
saion_root="/work/ReiterU/ant_tmp/$base_folder"
jobs_root="/flash/ReiterU/ant_tmp/jobs/$base_folder"
saion_jobs_root="/work/ReiterU/ant_tmp/jobs/$base_folder"
sentinel_root="$data_folder/sentinels"
aruco_bucket_root="$data_folder"
aruco_flash_root="$flash_root/aruco"

ensure_dir "$data_folder"
ensure_dir "$flash_root"
ensure_dir "$jobs_root"
ensure_dir "$sentinel_root"
ensure_dir "$aruco_bucket_root"
ensure_dir "$aruco_flash_root"

# -----------------------------
# Dependency checks
# -----------------------------
require_cmd ffmpeg
require_cmd ffprobe
require_cmd sbatch
require_cmd rsync
require_cmd python3

# Models
if [[ -n "$CLI_SLEAP_MODEL_CENTROID" ]]; then
	SLEAP_MODEL_CENTROID="$CLI_SLEAP_MODEL_CENTROID"
else
	SLEAP_MODEL_CENTROID="${SLEAP_MODEL_CENTROID:-/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centroid/}"
fi
if [[ -n "$CLI_SLEAP_MODEL_INSTANCE" ]]; then
	SLEAP_MODEL_INSTANCE="$CLI_SLEAP_MODEL_INSTANCE"
else
	SLEAP_MODEL_INSTANCE="${SLEAP_MODEL_INSTANCE:-/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centered_instance/}"
fi

SLEAP2H5_SCRIPT="${SLEAP2H5_SCRIPT:-$SCRIPT_DIR/sleap2h5.py}"
SLEAP2CSV_SCRIPT="${SLEAP2CSV_SCRIPT:-$SCRIPT_DIR/sleap2csv.py}"

# Default to SLEAP 1.5.2 and enforce correct CLI for 1.5.2
SLEAP_VERSION="${CLI_SLEAP_VERSION:-${SLEAP_VERSION:-1.5.2}}"
case "$SLEAP_VERSION" in
	1.5.2)
		SLEAP_TRACK_CMD='sleap-nn-track -i "$input" -m "$model1" -m "$model2" -o "$out_slp" --no_empty_frames -b 2'
		;;
	1.4.1)
		SLEAP_TRACK_CMD='sleap-track "$input" -m "$model1" -m "$model2" --tracking.tracker none -o "$out_slp" --verbosity json --no-empty-frames -b 2'
		;;
	*) echo "[ERR] Unsupported SLEAP version: $SLEAP_VERSION" >&2; exit 2 ;;
esac

echo "[INFO] Using SLEAP version: $SLEAP_VERSION" >&2
echo "[INFO] CONDA_ENV_SLEAP: $CONDA_ENV_SLEAP" >&2

SAION_COLLECT_PARTITION="${SAION_COLLECT_PARTITION:-test-gpu}"
ARUCO_SCRIPT="${ARUCO_SCRIPT:-$SCRIPT_DIR/run_aruco.py}"
RSYNC_FLAGS="--chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw -O"

BUCKET_WRITE_HOST_CANDIDATES="${BUCKET_WRITE_HOST_CANDIDATES:-datacp,deigo-login1,deigo-login2,deigo-login3}"
SAION_BUCKET_HOST_CANDIDATES="${SAION_BUCKET_HOST_CANDIDATES:-saion-login1,saion-login2,saion-login3}"
DATA_COPY_PARTITION="${DATA_COPY_PARTITION:-datacp}"

case "$SAION_NODE" in
	test-gpu) sleap_time="0-08:00:00" ;;
	largegpu) sleap_time="0-18:00:00" ;;
	gpu)      sleap_time="2-00:00:00" ;;
	*)        sleap_time="0-08:00:00" ;;
esac
case "$SAION_COLLECT_PARTITION" in
	test-gpu) collect_time="0-08:00:00" ;;
	largegpu) collect_time="0-18:00:00" ;;
	gpu)      collect_time="2-00:00:00" ;;
	*)        collect_time="0-08:00:00" ;;
esac

BUCKET_WRITE_HOST="$(select_resolvable_host "Deigo bucket host" "${BUCKET_WRITE_HOST:-$BUCKET_WRITE_HOST_CANDIDATES}")" || exit 2
SAION_BUCKET_HOST="$(select_resolvable_host "Saion bucket host" "${SAION_BUCKET_HOST:-$SAION_BUCKET_HOST_CANDIDATES}")" || exit 2
echo "[INFO] Deigo bucket host: $BUCKET_WRITE_HOST" >&2
echo "[INFO] Saion bucket host: $SAION_BUCKET_HOST" >&2

[[ -f "$ARUCO_SCRIPT" ]] || echo "[WARN] ArUco script not found at $ARUCO_SCRIPT" >&2

# -----------------------------
# Source video scan (non-recursive)
# -----------------------------
videos=( "$DIR"/*.avi "$DIR"/*.mkv )
(( ${#videos[@]} > 0 )) || { echo "[WARN] No .avi or .mkv videos found in $DIR" >&2; exit 0; }

get_job_count() { squeue -u "$USER" -h -r | wc -l; }

# -----------------------------
# Per-video pipeline orchestration
# -----------------------------
for video in "${videos[@]}"; do
	while true; do
		current_jobs=$(get_job_count)
		(( current_jobs < MAX_SUBMITTED_JOBS )) && break
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
	expected_chunks="$chunk_count"

	# Batch sizing
	chunk_count_buffered=$(( chunk_count + CHUNK_BUFFER ))
	if [[ -n "$BATCH_SIZE" ]]; then
		b_size="$BATCH_SIZE"
	else
		if (( chunk_count_buffered > MAX_ARRAY_TASKS )); then
			b_size=$(( (chunk_count_buffered + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS ))
		else
			b_size=1
		fi
	fi
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

	ensure_dir "$video_flash_dir"
	ensure_dir "$video_job_dir"
	ensure_dir "$frame_dir"

	aruco_flash_dir="$aruco_flash_root/$vname"
	aruco_bucket_dir="$aruco_bucket_root"
	ensure_dir "$aruco_flash_dir"
	ensure_dir "$aruco_bucket_dir"

	: > "$summary_file"
	chmod 664 "$summary_file"

	# Decide what is missing based on coverage
	cov="$(check_video_coverage "$vname" "$data_folder" "$expected_chunks")"
	# shellcheck disable=SC2086
	eval "$cov"

	if (( enc==1 && aruco==1 && sleap==1 )); then
		echo "[SKIP] $vname fully complete (all enc+aruco+sleap outputs present on bucket)"
		if [[ ! -f "$cleanup_ok" ]]; then
			echo "[INFO] $vname complete but cleanup.ok missing; leaving as-is."
		fi
		continue
	fi

	# Template script paths
	split_script="$video_job_dir/split-$vname.sh"
	encode_script="$video_job_dir/encode-$vname.sh"
	enc_finalize_script="$video_job_dir/encode-finalize-$vname.sh"
	aruco_script_path="$video_job_dir/aruco-$vname.sh"
	aruco_finalize_script="$video_job_dir/aruco-finalize-$vname.sh"
	bridge_script="$video_job_dir/bridge-$vname.sh"
	cleanup_script="$video_job_dir/cleanup-$vname.sh"

	# -----------------------------
	# Split (decides next steps based on coverage; repairs sentinels via datacp)
	# -----------------------------
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

__ENV_ACTIVATE__

video="__VIDEO__"
flash_dir="__FLASH_DIR__"
manifest="__MANIFEST__"
seg_sec=__SEG_SEC__
data_dir="__DATA_DIR__"
expected_chunks=__CHUNK_COUNT__
copy_partition="__DATA_COPY_PARTITION__"
job_tmp_dir="__JOBDIR__"

enc_ok="__ENC_OK__"
aruco_ok="__ARUCO_OK__"
sleap_done="__SLEAP_DONE_OK__"
cleanup_ok="__CLEANUP_OK__"

have_all_encoded=1
have_all_aruco=1
have_all_sleap=1
if (( expected_chunks <= 0 )); then
	have_all_encoded=0; have_all_aruco=0; have_all_sleap=0
else
	for (( i=0; i<expected_chunks; i++ )); do
		idx=$(printf "%03d" "$i")
		[[ -s "$data_dir/__BASE___${idx}.avi" ]] || have_all_encoded=0
		[[ -s "$data_dir/__BASE___${idx}_aruco_tracks_.h5" ]] || have_all_aruco=0
		[[ -s "$data_dir/__BASE___${idx}.slp" ]] || have_all_sleap=0
	done
fi

touch_on_datacp() {
	local path="$1"
	local tmp
	tmp=$(mktemp -p "$job_tmp_dir" "__BASE___touch_XXXXXX.sh")
	cat > "$tmp" <<'COPY'
#!/bin/bash -l
set -eo pipefail
umask 0002
p="__TOUCH_PATH__"
mkdir -p "$(dirname "$p")"
: > "$p"
COPY
	sed -i "s#__TOUCH_PATH__#$(printf '%s' "$path" | sed -e 's/[\/&]/\\&/g')#g" "$tmp"
	chmod +x "$tmp"
	sbatch --wait -p "$copy_partition" -J "touch-__BASE__" \
		-o "${job_tmp_dir}/datacp-touch_%j.out" \
		-e "${job_tmp_dir}/datacp-touch_%j.err" "$tmp" >/dev/null
	rm -f "$tmp"
}

# Repair sentinels if coverage is complete (advisory only)
if (( have_all_encoded )) && [[ ! -f "$enc_ok" ]]; then touch_on_datacp "$enc_ok"; fi
if (( have_all_aruco )) && [[ ! -f "$aruco_ok" ]]; then touch_on_datacp "$aruco_ok"; fi
if (( have_all_sleap )) && [[ ! -f "$sleap_done" ]]; then touch_on_datacp "$sleap_done"; fi

mkdir -p "$flash_dir"
chmod 2775 "$flash_dir"

# If encoded chunks not complete on bucket: do split+encode path
if (( have_all_encoded == 0 )); then
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

	jid_enc=$(sbatch --parsable "__ENCODE_SCRIPT__")
	echo "Submitted Encode Array: $jid_enc"
	sbatch --dependency=afterok:$jid_enc "__ENC_FINALIZE_SCRIPT__"
	echo "Submitted Encode Sync (dep: $jid_enc)"
	exit 0
fi

# Encoded chunks exist on bucket: schedule only missing stages
need_bridge=1
need_aruco=1

if (( have_all_sleap == 1 )); then
	need_bridge=0
	echo "[SKIP] SLEAP already complete by coverage"
fi
if (( have_all_aruco == 1 )); then
	need_aruco=0
	echo "[SKIP] ArUco already complete by coverage"
fi

jid_bridge=""
jid_aruco_sync=""

if (( need_bridge )); then
	jid_bridge=$(sbatch --parsable "__BRIDGE_SCRIPT__")
	echo "Submitted Bridge: $jid_bridge"
fi

if (( need_aruco )); then
	jid_aruco=$(sbatch --parsable "__ARUCO_SCRIPT_PATH__")
	echo "Submitted ArUco Array: $jid_aruco"
	jid_aruco_sync=$(sbatch --parsable --dependency=afterok:$jid_aruco "__ARUCO_FINALIZE_SCRIPT__")
	echo "Submitted ArUco Sync: $jid_aruco_sync"
fi

# Cleanup: only useful if flash exists; safe to run; it waits for aruco_ok
if [[ -d "$flash_dir" || -d "__ARUCO_FLASH_DIR__" ]]; then
	deps=()
	[[ -n "$jid_bridge" ]] && deps+=("$jid_bridge")
	[[ -n "$jid_aruco_sync" ]] && deps+=("$jid_aruco_sync")

	if (( ${#deps[@]} )); then
		sbatch --dependency=afterok:$(IFS=:; echo "${deps[*]}") "__CLEANUP_SCRIPT__"
	else
		sbatch "__CLEANUP_SCRIPT__"
	fi
fi
EOS

	# -----------------------------
	# Encode (per-chunk: skip if output exists on flash OR bucket)
	# Canonical: BASE_###.avi  (single underscore)
	# -----------------------------
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

__ENV_ACTIVATE__

flash_dir="__FLASH_DIR__"
data_dir="__DATA_DIR__"
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
	out_bucket="$data_dir/__BASE___${idx}.avi"

	if [[ -s "$out" || -s "$out_bucket" ]]; then
		echo "[SKIP] re-encoded chunk exists idx=$idx"
		src="$out"; [[ -s "$src" ]] || src="$out_bucket"
		nb=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nk=1:nw=1 "$src" 2>/dev/null || echo 0)
		printf "%s,%s\n" "__BASE___${idx}.avi" "$nb" > "$frame_dir/${idx}.csv"
		chmod 664 "$frame_dir/${idx}.csv"
		rm -f "$raw"
		continue
	fi

	if [[ ! -s "$raw" ]]; then
		echo "[SKIP] no raw chunk $raw"
		continue
	fi

	ffmpeg -hide_banner -y -i "$raw" -c:v libx264 -pix_fmt yuv420p -preset fast -crf 23 -threads 8 "$out"
	nb=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nk=1:nw=1 "$out" 2>/dev/null || echo 0)
	printf "%s,%s\n" "__BASE___${idx}.avi" "$nb" > "$frame_dir/${idx}.csv"
	chmod 664 "$frame_dir/${idx}.csv"
	rm -f "$raw"
done
EOS

	# -----------------------------
	# Encode finalize (datacp rsync --ignore-existing; writes enc_ok)
	# -----------------------------
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
job_tmp_dir="__JOBDIR__"

raw_chunks=("$flash_dir"/__BASE___raw_*.avi)
if (( ${#raw_chunks[@]} )); then printf '%s\n' "${raw_chunks[@]}" | sort > "$manifest"; else : > "$manifest"; fi
chmod 664 "$manifest"

frame_csv="$flash_dir/__BASE___frame_counts.csv"
if compgen -G "$frame_dir/"*.csv > /dev/null; then
	python3 - "$frame_dir" "$frame_csv" <<'PY'
import pathlib, sys
d=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
rows=[]
for p in sorted(d.glob("*.csv")):
    rows += [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
out.write_text("\n".join(rows)+("\n" if rows else ""))
PY
	chmod 664 "$frame_csv"
	rm -f "$frame_dir/"*.csv
else
	: > "$frame_csv"
fi

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
rsync -avh __RSYNC_FLAGS__ --ignore-existing \
	--exclude="__BASE___raw_*.avi" \
	--include="__BASE___*.avi" \
	--include="__BASE___frame_counts.csv" \
	--exclude="*" "$flash_dir/" "$data_dir/"

mkdir -p "$enc_ok_dir"
: > "$enc_ok"
COPY
chmod +x "$copy_script"

sbatch --wait -p "__DATA_COPY_PARTITION__" -J "encsync-__BASE__" \
	-o "${job_tmp_dir}/datacp-encsync_%j.out" \
	-e "${job_tmp_dir}/datacp-encsync_%j.err" "$copy_script" || { echo "[ERR] datacp encsync failed" >&2; exit 1; }

rm -f "$copy_script" "$frame_csv"

# After encsync, schedule only missing stages; split script will compute coverage.
sbatch "__SPLIT_SCRIPT__"
EOS

	# -----------------------------
	# ArUco (uses aruco env; per-chunk skip if bucket output exists; reads segments from flash else bucket)
	# Canonical: BASE_###.avi, BASE_###_aruco_tracks_.h5
	# -----------------------------
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

__ENV_ACTIVATE__

flash_dir="__FLASH_DIR__"
data_dir="__DATA_DIR__"
output_dir="__ARUCO_FLASH_DIR__"
bucket_dir="__ARUCO_BUCKET_DIR__"

mkdir -p "$output_dir"
chmod 2775 "$output_dir"

segment_dir="$flash_dir"
if ! compgen -G "$flash_dir/__BASE___*.avi" > /dev/null; then
	segment_dir="$data_dir"
fi

batch_size=__BATCH_SIZE__
start_idx=$(( SLURM_ARRAY_TASK_ID * batch_size ))
end_idx=$(( start_idx + batch_size ))

for (( i=start_idx; i<end_idx; i++ )); do
	idx=$(printf "%03d" "$i")
	video_path="$segment_dir/__BASE___${idx}.avi"
	[[ -s "$video_path" ]] || { echo "[SKIP] missing segment $video_path"; continue; }

	bucket_out="$bucket_dir/__BASE___${idx}_aruco_tracks_.h5"
	if [[ -s "$bucket_out" ]]; then
		echo "[SKIP] ArUco bucket output exists idx=$idx"
		touch "$output_dir/__BASE___${idx}.aruco.ok"
		continue
	fi

	python3 "__ARUCO_SCRIPT__" --video-file "$video_path" --output-path "$output_dir/"
	touch "$output_dir/__BASE___${idx}.aruco.ok"
done
EOS

	# -----------------------------
	# ArUco finalize (datacp rsync --ignore-existing; writes aruco_ok)
	# -----------------------------
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
rsync -avh __RSYNC_FLAGS__ --ignore-existing \
	--include="__BASE___*_aruco_tracks_.h5" --exclude="*" "$flash_dir/" "$bucket_dir/"

mkdir -p "$aruco_ok_dir"
: > "$aruco_ok"
COPY
chmod +x "$copy_script"

sbatch --wait -p "__DATA_COPY_PARTITION__" -J "arucofin-__BASE__" \
	-o "${job_tmp_dir}/datacp-aruco_%j.out" \
	-e "${job_tmp_dir}/datacp-aruco_%j.err" "$copy_script" || { echo "[ERR] datacp aruco sync failed" >&2; exit 1; }

rm -f "$copy_script"
rm -f "$flash_dir"/__BASE___*.aruco.ok
rm -f "$flash_dir"/__BASE___*_aruco_tracks_.h5
EOS

	# -----------------------------
	# Bridge to Saion (stages from flash else bucket; SLEAP per-chunk skip if bucket outputs exist)
	# Canonical input: BASE_###.avi (single underscore)
	# -----------------------------
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
sleap_done_ok="__SLEAP_DONE_OK__"
summary_file="__SUMMARY_FILE__"
job_tmp_dir="__JOBDIR__"

if [[ -f "$sleap_done_ok" ]]; then
	echo "[INFO] sleap_done sentinel exists; proceeding anyway (coverage drives actual work)."
fi

command -v sbatch >/dev/null 2>&1 || { [[ -r /etc/profile ]] && source /etc/profile || true; }
command -v sbatch >/dev/null 2>&1 || { echo "[ERR] sbatch not found on Deigo in bridge job environment" >&2; exit 2; }

SSH_BIN="${SSH_BIN:-/usr/bin/ssh}"
ENV_CLEAN=(/usr/bin/env -u LD_LIBRARY_PATH -u LD_PRELOAD -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONPATH -u PYTHONHOME)
SSH_CMD=("${ENV_CLEAN[@]}" "$SSH_BIN" -x -oBatchMode=yes -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null)
export RSYNC_RSH="${SSH_CMD[*]}"

"${SSH_CMD[@]}" saion "mkdir -p '$remote_input' '$remote_output' '$remote_logs'"

stage_src="$flash_dir"
if ! compgen -G "$flash_dir/__BASE___*.avi" > /dev/null; then
	stage_src="$data_dir"
fi

echo "[INFO] staging encoded segments to Saion from: $stage_src"
rsync -avh __RSYNC_FLAGS__ --ignore-existing \
	--exclude="__BASE___raw_*.avi" \
	--include="__BASE___*.avi" \
	--exclude="*" "$stage_src/" "saion:$remote_input/"


# --- Saion array job (SLP-only; minimal logging; bucket-as-truth; writes to /work; sync to bucket via ssh+rsync)
"${SSH_CMD[@]}" saion "cat > '$remote_root/sleap_array.sh'" <<'SAION'
#!/bin/bash -l
#SBATCH -t __SLEAP_TIME__
#SBATCH --cpus-per-task=12
#SBATCH --partition=__SAION_NODE__
#SBATCH --mem=16G
#SBATCH --gres=__SAION_GRES__
#SBATCH -J sleap-__BASE__
#SBATCH -o __REMOTE_LOGS__/sleap_%A_%a.out
#SBATCH -e __REMOTE_LOGS__/sleap_%A_%a.err
#SBATCH --array=0-__ARRAY_MAX__%__SLEAP_CONCURRENCY__

set -euo pipefail
shopt -s nullglob
umask 0002

ts() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

# ---- minimal context ----
ts "job_id=${SLURM_JOB_ID} task_id=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
ts "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

# ---- ssh wrapper for bucket host + rsync transport ----
SSH_BIN="${SSH_BIN:-/usr/bin/ssh}"
ENV_CLEAN=(/usr/bin/env -u LD_LIBRARY_PATH -u LD_PRELOAD -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONPATH -u PYTHONHOME)
SSH_CMD=("${ENV_CLEAN[@]}" "$SSH_BIN" -x -oBatchMode=yes -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null)
export RSYNC_RSH="${SSH_CMD[*]}"

# ---- conda activation (provided by orchestrator) ----
__ENV_ACTIVATE__

# ---- threads: use all allocated CPUs ----
CPUS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export TORCH_NUM_THREADS="$CPUS"
export TORCH_NUM_INTEROP_THREADS=1

# ---- paths/params ----
base="__BASE__"
input_dir="__REMOTE_INPUT__"      # /work/.../input
output_dir="__REMOTE_OUTPUT__"    # /work/.../output

bucket_dir="__DATA_DIR__"         # /bucket/.../data (sync target) on bucket host
bucket_host="__SAION_BUCKET_HOST__"

batch_size=__BATCH_SIZE__
model1="__MODEL1__"
model2="__MODEL2__"

mkdir -p "$output_dir"

start_idx=$(( SLURM_ARRAY_TASK_ID * batch_size ))
end_idx=$(( start_idx + batch_size ))

# Ensure bucket target exists (idempotent)
"${SSH_CMD[@]}" "$bucket_host" "mkdir -p '$bucket_dir'"

for (( i=start_idx; i<end_idx; i++ )); do
  idx=$(printf "%03d" "$i")
  input="$input_dir/${base}_${idx}.avi"
  out_slp_name="${base}_${idx}.slp"
  out_slp="$output_dir/$out_slp_name"

  if [[ ! -s "$input" ]]; then
    ts "SKIP missing input idx=$idx"
    continue
  fi

  # Skip if output already exists on bucket (truth)
  if "${SSH_CMD[@]}" "$bucket_host" bash -lc "test -s '$bucket_dir/$out_slp_name'"; then
    ts "SKIP bucket slp exists idx=$idx"
    continue
  fi

  # Optional: skip if already present on Saion /work (can happen on retries)
  if [[ -s "$out_slp" ]]; then
    ts "SKIP /work slp exists idx=$idx"
    continue
  fi

  # Avoid stale partials
  rm -f "$out_slp" || true

  ts "RUN idx=$idx"
  START_TS=$(date +%s)

  # Track (must use $input and $out_slp)
  set +e
  __SLEAP_TRACK_CMD__
  rc=$?
  set -e

  END_TS=$(date +%s)
  ts "DONE idx=$idx rc=$rc runtime=$((END_TS-START_TS))s"

  (( rc == 0 )) || { ts "ERR sleap-nn-track failed idx=$idx"; continue; }

  if [[ ! -s "$out_slp" ]]; then
    ts "ERR missing output slp idx=$idx: $out_slp"
    continue
  fi

  # Sync result to bucket (never clobber)
  rsync -ah --ignore-existing --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw \
    "$out_slp" \
    "$bucket_host:$bucket_dir/"

  ts "SYNCED idx=$idx"
done
SAION

# --- Saion collect job (SLP-only; marks done only if bucket coverage complete; then cleanup /work)
"${SSH_CMD[@]}" saion "cat > '$remote_root/sleap_collect.sh'" <<'SAION'
#!/bin/bash -l
#SBATCH -t __COLLECT_TIME__
#SBATCH -c 2
#SBATCH --partition=__SAION_COLLECT_PARTITION__
#SBATCH --mem=8G
#SBATCH -J sleapfin-__BASE__
#SBATCH -o __REMOTE_LOGS__/sleapfin_%j.out
#SBATCH -e __REMOTE_LOGS__/sleapfin_%j.err

set -euo pipefail
shopt -s nullglob

SSH_BIN="${SSH_BIN:-/usr/bin/ssh}"
ENV_CLEAN=(/usr/bin/env -u LD_LIBRARY_PATH -u LD_PRELOAD -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONPATH -u PYTHONHOME)
SSH_CMD=("${ENV_CLEAN[@]}" "$SSH_BIN" -x -oBatchMode=yes -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null)
export RSYNC_RSH="${SSH_CMD[*]}"

__ENV_ACTIVATE__

base="__BASE__"
remote_root="__REMOTE_ROOT__"
output_dir="__REMOTE_OUTPUT__"
bucket_dir="__DATA_DIR__"
bucket_host="__SAION_BUCKET_HOST__"
sleap_done="__SLEAP_DONE_OK__"
sleap_done_dir="__SLEAP_DONE_OK_DIR__"
expected_chunks=__CHUNK_COUNT__
if [[ ! "$expected_chunks" =~ ^[0-9]+$ ]]; then
  echo "[ERR] expected_chunks not numeric: '$expected_chunks' (placeholder substitution failed)" >&2
  exit 2
fi

# Ensure bucket dir exists
"${SSH_CMD[@]}" "$bucket_host" "mkdir -p '$bucket_dir'"

# Copy any remaining SLPs from Saion /work to bucket (never clobber)
rsync -avh --ignore-existing --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw \
  --include="${base}_*.slp" \
  --exclude="*" "$output_dir/" "$bucket_host:$bucket_dir/"

# Verify bucket coverage (SLP-only)
missing=0
if (( expected_chunks <= 0 )); then
  echo "[ERR] expected_chunks invalid: $expected_chunks" >&2
  exit 2
fi

for (( i=0; i<expected_chunks; i++ )); do
  idx=$(printf "%03d" "$i")
  slp="$bucket_dir/${base}_${idx}.slp"
  if ! "${SSH_CMD[@]}" "$bucket_host" bash -lc "test -s '$slp'"; then
    echo "[WARN] missing bucket slp for idx=$idx" >&2
    missing=1
  fi
done

if (( missing == 1 )); then
  echo "[ERR] SLEAP bucket coverage incomplete. NOT marking sleap_done and NOT removing $remote_root" >&2
  exit 1
fi

# Mark done (idempotent) on bucket host (via datacp in your pipeline, or directly here—keeping your existing approach)
"${SSH_CMD[@]}" "$bucket_host" "bash -lc 'umask 0002 && mkdir -p \"$sleap_done_dir\" && : > \"$sleap_done\"'"

# Cleanup Saion /work
rm -rf "$remote_root"
SAION

# Make executable and submit
"${SSH_CMD[@]}" saion "chmod +x '$remote_root/sleap_array.sh' '$remote_root/sleap_collect.sh'"

sleap_array_job=$("${SSH_CMD[@]}" saion "bash -lc \"sbatch --parsable '$remote_root/sleap_array.sh'\"")
sleap_collect_job=$("${SSH_CMD[@]}" saion "bash -lc \"sbatch --dependency=afterok:$sleap_array_job --parsable '$remote_root/sleap_collect.sh'\"")

exec 9>>"$summary_file"
if flock 9; then
	printf '%-16s %s\n' 'saion-array' "$sleap_array_job" >&9
	printf '%-16s %s\n' 'saion-collect' "$sleap_collect_job" >&9
	flock -u 9
fi
exec 9>&-

# Mark submit-ok via datacp (idempotent)
copy_script=$(mktemp -p "$job_tmp_dir" "__BASE___sleap_submit_mark_XXXXXX.sh")
cat > "$copy_script" <<'COPY'
#!/bin/bash -l
set -eo pipefail
umask 0002
mkdir -p "__SLEAP_SUBMIT_OK_DIR__"
: > "__SLEAP_SUBMIT_OK__"
COPY
chmod +x "$copy_script"

sbatch --wait -p "__DATA_COPY_PARTITION__" -J "sleapmark-__BASE__" \
	-o "${job_tmp_dir}/datacp-sleapmark_%j.out" \
	-e "${job_tmp_dir}/datacp-sleapmark_%j.err" "$copy_script" || { echo "[ERR] datacp sleap marker failed" >&2; exit 1; }

rm -f "$copy_script"
echo "[INFO] submitted SLEAP array $sleap_array_job and collect $sleap_collect_job"
EOS

	# -----------------------------
	# Cleanup (removes flash dirs after aruco is done; marks cleanup_ok via datacp)
	# -----------------------------
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

aruco_ok="__ARUCO_OK__"
cleanup_ok="__CLEANUP_OK__"
cleanup_ok_dir="__CLEANUP_OK_DIR__"
flash_dir="__FLASH_DIR__"
aruco_flash_dir="__ARUCO_FLASH_DIR__"
flash_root="__FLASH_ROOT__"
aruco_flash_root="__ARUCO_FLASH_ROOT__"
summary_file="__SUMMARY_FILE__"
timeout_secs=__SENTINEL_TIMEOUT__
job_tmp_dir="__JOBDIR__"

wait_for_file() {
	local path="$1" label="$2"
	local deadline=$(( $(date +%s) + timeout_secs ))
	while (( $(date +%s) <= deadline )); do
		[[ -f "$path" ]] && return 0
		sleep 60
	done
	echo "[ERR] timeout waiting for $label ($path)" >&2
	return 1
}

wait_for_file "$aruco_ok" "ArUco completion marker"

echo "[INFO] removing flash directory $flash_dir"
rm -rf "$flash_dir" || true
echo "[INFO] removing ArUco flash directory $aruco_flash_dir"
rm -rf "$aruco_flash_dir" || true

[[ -d "$aruco_flash_root" ]] && rmdir "$aruco_flash_root" 2>/dev/null || true
[[ -d "$flash_root" ]] && rmdir "$flash_root" 2>/dev/null || true

copy_script=$(mktemp -p "$job_tmp_dir" "__BASE___cleanup_mark_XXXXXX.sh")
cat > "$copy_script" <<'COPY'
#!/bin/bash -l
set -eo pipefail
umask 0002
mkdir -p "__CLEANUP_OK_DIR__"
: > "__CLEANUP_OK__"
COPY
chmod +x "$copy_script"

sbatch --wait -p "__DATA_COPY_PARTITION__" -J "cleanupmark-__BASE__" \
	-o "${job_tmp_dir}/datacp-cleanup_%j.out" \
	-e "${job_tmp_dir}/datacp-cleanup_%j.err" "$copy_script" || { echo "[ERR] datacp cleanup mark failed" >&2; exit 1; }

rm -f "$copy_script"

if [[ -f "$summary_file" ]]; then
	echo "[INFO] job summary:"
	cat "$summary_file"
fi
EOS

	# -----------------------------
	# Parameterize templates
	# -----------------------------
	replace_placeholders "$split_script" \
		BASE "$vname" JOBDIR "$video_job_dir" VIDEO "$video" \
		SEG_SEC "$SEG_SEC" FLASH_DIR "$video_flash_dir" \
		MANIFEST "$manifest_path" DATA_DIR "$data_folder" CHUNK_COUNT "$expected_chunks" \
		ENCODE_SCRIPT "$encode_script" ENC_FINALIZE_SCRIPT "$enc_finalize_script" \
		BRIDGE_SCRIPT "$bridge_script" ARUCO_SCRIPT_PATH "$aruco_script_path" \
		ARUCO_FINALIZE_SCRIPT "$aruco_finalize_script" CLEANUP_SCRIPT "$cleanup_script" \
		ENC_OK "$enc_ok" ARUCO_OK "$aruco_ok" SLEAP_DONE_OK "$sleap_done" CLEANUP_OK "$cleanup_ok" \
		DATA_COPY_PARTITION "$DATA_COPY_PARTITION" \
		ARUCO_FLASH_DIR "$aruco_flash_dir" \
		ENV_ACTIVATE "$ENV_ACTIVATE"

	replace_placeholders "$encode_script" \
		BASE "$vname" JOBDIR "$video_job_dir" ARRAY_MAX "$array_upper" \
		ENC_CONCURRENCY "$ENC_CONCURRENCY" FLASH_DIR "$video_flash_dir" \
		DATA_DIR "$data_folder" BATCH_SIZE "$b_size" \
		ENV_ACTIVATE "$ENV_ACTIVATE"

	replace_placeholders "$enc_finalize_script" \
		BASE "$vname" JOBDIR "$video_job_dir" FLASH_DIR "$video_flash_dir" \
		DATA_DIR "$data_folder" ENC_OK "$enc_ok" ENC_OK_DIR "$enc_ok_dir" \
		RSYNC_FLAGS "$RSYNC_FLAGS" MANIFEST "$manifest_path" \
		DATA_COPY_PARTITION "$DATA_COPY_PARTITION" \
		SPLIT_SCRIPT "$split_script"

	replace_placeholders "$aruco_script_path" \
		BASE "$vname" JOBDIR "$video_job_dir" ARRAY_MAX "$array_upper" \
		ARUCO_CONCURRENCY "$ARUCO_CONCURRENCY" FLASH_DIR "$video_flash_dir" \
		DATA_DIR "$data_folder" ARUCO_FLASH_DIR "$aruco_flash_dir" \
		ARUCO_BUCKET_DIR "$aruco_bucket_dir" ARUCO_SCRIPT "$ARUCO_SCRIPT" \
		BATCH_SIZE "$b_size" \
		ENV_ACTIVATE "$ARUCO_ENV_ACTIVATE"

	replace_placeholders "$aruco_finalize_script" \
		BASE "$vname" JOBDIR "$video_job_dir" ARUCO_FLASH_DIR "$aruco_flash_dir" \
		ARUCO_BUCKET_DIR "$aruco_bucket_dir" ARUCO_OK "$aruco_ok" ARUCO_OK_DIR "$aruco_ok_dir" \
		RSYNC_FLAGS "$RSYNC_FLAGS" DATA_COPY_PARTITION "$DATA_COPY_PARTITION"

	replace_placeholders "$bridge_script" \
		BASE "$vname" JOBDIR "$video_job_dir" DATA_DIR "$data_folder" FLASH_DIR "$video_flash_dir" \
		REMOTE_ROOT "$remote_root" REMOTE_INPUT "$remote_input" REMOTE_OUTPUT "$remote_output" \
		REMOTE_LOGS "$remote_logs" SAION_NODE "$SAION_NODE" \
		MODEL1 "$SLEAP_MODEL_CENTROID" MODEL2 "$SLEAP_MODEL_INSTANCE" \
    SLEAP_SUBMIT_OK "$sleap_submit_ok" SLEAP_SUBMIT_OK_DIR "$sleap_submit_ok_dir" \
		SLEAP_DONE_OK "$sleap_done" SLEAP_DONE_OK_DIR "$sleap_done_dir" \
		RSYNC_FLAGS "$RSYNC_FLAGS" ARRAY_MAX "$array_upper" CHUNK_COUNT "$expected_chunks"\
		SLEAP_CONCURRENCY "$SLEAP_CONCURRENCY" SUMMARY_FILE "$summary_file" \
		DATA_COPY_PARTITION "$DATA_COPY_PARTITION" SAION_BUCKET_HOST "$SAION_BUCKET_HOST" \
		SAION_GRES "$saion_gres" SAION_COLLECT_PARTITION "$SAION_COLLECT_PARTITION" \
		BATCH_SIZE "$b_size" SLEAP_TIME "$sleap_time" COLLECT_TIME "$collect_time" \
		SLEAP_TRACK_CMD "$SLEAP_TRACK_CMD" \
		ENV_ACTIVATE "$ENV_ACTIVATE"

	replace_placeholders "$cleanup_script" \
		BASE "$vname" JOBDIR "$video_job_dir" \
		ARUCO_OK "$aruco_ok" CLEANUP_OK "$cleanup_ok" CLEANUP_OK_DIR "$cleanup_ok_dir" \
		FLASH_DIR "$video_flash_dir" ARUCO_FLASH_DIR "$aruco_flash_dir" \
		FLASH_ROOT "$flash_root" ARUCO_FLASH_ROOT "$aruco_flash_root" \
		SUMMARY_FILE "$summary_file" \
		SENTINEL_TIMEOUT "$SENTINEL_TIMEOUT" DATA_COPY_PARTITION "$DATA_COPY_PARTITION"

	chmod +x "$split_script" "$encode_script" "$enc_finalize_script" \
		"$aruco_script_path" "$aruco_finalize_script" "$bridge_script" "$cleanup_script"

	# -----------------------------
	# Scheduling:
	# Always start with split_script; it will decide what to do based on bucket coverage.
	# -----------------------------
	jid_split=$(sbatch --parsable "$split_script")
	echo "Submitted pipeline for $vname (expected_chunks: $expected_chunks, array 0-$array_upper)"
	echo "  split $jid_split"
	echo "split: $jid_split" >> "$summary_file"
done

echo "Pipelines scheduled. Monitor with: squeue -u $(id -un)"
echo "Per-video summaries: $jobs_root/<video>/pipeline.jobs"
