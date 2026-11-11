# Shared configuration and helpers for video_rescue. Sourced by rescue.sh and
# by every sbatch task; every setting can be overridden from the environment as
# VR_<NAME> (the login shell already defines WORK, so bare names are not read).
#
# Layout (all per-user):
#   SRC_ROOT   recorder output on the auto-purged acquisition share
#   DEST_ROOT  final home of the transcoded videos:   $DEST_ROOT/<date>/blockNN/
#   RAW_ROOT   temporary verbatim copies (insurance):  $RAW_ROOT/<date>/blockNN/
#   WORK       scratch for manifest, parts, logs, and state markers

SRC_ROOT="${VR_SRC_ROOT:-/hpacquire/users/$USER}"
DEST_ROOT="${VR_DEST_ROOT:-/bucket/StephensU/$USER}"
RAW_ROOT="${VR_RAW_ROOT:-$DEST_ROOT/_raw_cq31}"
WORK="${VR_WORK:-/flash/ReiterU/$USER/video_rescue}"
LOGIN_ALIAS="${VR_LOGIN_ALIAS:-deigo}"          # ssh alias that can write $DEST_ROOT (login node)
FFMPEG_MODULE="${VR_FFMPEG_MODULE:-ffmpeg/7.1}"

# Encoder: libx264 tuned to match the recorder's new NVENC target
# (p3 / vbr / cq 33 / maxrate 150M / bufsize 240M / GOP 60 s). Measured on a
# 60 s cq31 sample: NVENC cq33 -> 14.7 MB, SSIM 0.9878; x264 faster crf29 ->
# 13.5 MB, SSIM 0.9891; x264 medium crf29 -> 14.4 MB, SSIM 0.9890.
X264_PRESET="${VR_X264_PRESET:-faster}"
X264_CRF="${VR_X264_CRF:-29}"
GOP_SEC="${VR_GOP_SEC:-60}"                 # must equal the recording's GOP so chunking stays keyframe-aligned
MAXRATE="${VR_MAXRATE:-150M}"
BUFSIZE="${VR_BUFSIZE:-240M}"

# Window = the slice of one video that one array task encodes. Must be a
# multiple of GOP_SEC so every window starts on a keyframe of the source.
WINDOW_SEC="${VR_WINDOW_SEC:-14400}"
TASK_CPUS="${VR_TASK_CPUS:-4}"                # measured 4024x3036 x264 faster: 2c 9.6 fps, 4c 14.9, 8c 15.6, 16c 23.6
FPS_PER_TASK="${VR_FPS_PER_TASK:-14}"            # planning estimate only
TASK_MEM="${VR_TASK_MEM:-8G}"
TASK_WALL="${VR_TASK_WALL:-1-00:00:00}"        # a 4 h window took 12 h once when the share was saturated
PROBE_WALL="${VR_PROBE_WALL:-12:00:00}"        # 11 probes of 300 GB files hit a 4 h limit under the same load
FINALIZE_MIN_LEFT="${VR_FINALIZE_MIN_LEFT:-120}"  # minutes of walltime a task needs before it may join+upload
ARRAY_THROTTLE="${VR_ARRAY_THROTTLE:-400}"         # concurrent encode tasks (x TASK_CPUS cores of the 2000-core cap)
PROBE_THROTTLE="${VR_PROBE_THROTTLE:-16}"         # concurrent probe scans; each pulls ~500 MB/s from the share
COPY_PARTITION="${VR_COPY_PARTITION:-datacp}"      # the only deigo partition whose nodes can write /bucket
COMPUTE_PARTITION="${VR_COMPUTE_PARTITION:-compute}"

MANIFEST="$WORK/manifest.tsv"
SESSIONS="$WORK/sessions.tsv"
STATE="$WORK/state"
PARTS="$WORK/parts"
FINAL="$WORK/final"
LOGS="$WORK/logs"
TASKS="$WORK/tasks"

die() { echo "[ERR] $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }

SSH_CMD=(ssh -x -oBatchMode=yes -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null
	-oConnectionAttempts=5 -oConnectTimeout=30 -oServerAliveInterval=15 -oServerAliveCountMax=4)
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

# manifest_rows <session_id|dest_rel> [kind]  -> matching manifest lines
manifest_rows() {
	local key="$1" kind="${2:-}"
	[[ -s "$MANIFEST" ]] || die "no manifest at $MANIFEST (run: rescue.sh inventory)"
	awk -F'\t' -v k="$key" -v kind="$kind" '($1==k || $2==k) && (kind=="" || $3==kind)' "$MANIFEST"
}

# manifest_first <session_id|dest_rel> <column> -> that column of the first matching row.
# awk exits on the first match: a `| head -1` here would SIGPIPE awk and, under
# pipefail + errexit, silently kill the caller.
manifest_first() {
	[[ -s "$MANIFEST" ]] || die "no manifest at $MANIFEST (run: rescue.sh inventory)"
	awk -F'	' -v k="$1" -v col="$2" '($1==k || $2==k) {print $col; exit}' "$MANIFEST"
}
session_dest() { manifest_first "$1" 2; }
session_id_of() { manifest_first "$1" 1; }

# video_source <hpacquire_path> <dest_rel> -> readable source path (original if
# it still exists, else the raw copy on $RAW_ROOT), or empty.
video_source() {
	local src="$1" dest_rel="$2" raw
	[[ -s "$src" ]] && { echo "$src"; return 0; }
	raw="$RAW_ROOT/$dest_rel/$(basename "$src")"
	[[ -s "$raw" ]] && { echo "$raw"; return 0; }
	return 1
}

# fps from a probe json ("fps": 24)
probe_field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
