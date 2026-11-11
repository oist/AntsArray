#!/bin/bash -l
#SBATCH -J vr_probe
#SBATCH -p compute
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 12:00:00
# Array task: count the frames of one video (row $SLURM_ARRAY_TASK_ID of env
# TASKLIST: src<TAB>dest_rel<TAB>name). The recorder writes live-mode Matroska
# without cues or a duration, so the only exact frame count is a full packet
# scan (no decode; ~11 min for a 340 GB file). The count drives window planning
# and is the reference the finalize step verifies the transcode against.
# Writes $STATE/<dest_rel>/<name>.probe.json.
set -eo pipefail
# sbatch runs a spooled copy of this script, so locate lib.sh via RESCUE_DIR exported by rescue.sh
source "${RESCUE_DIR:?RESCUE_DIR not set (submit through rescue.sh)}/lib.sh"
[[ -n "${TASKLIST:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "TASKLIST / SLURM_ARRAY_TASK_ID not set"
module load "$FFMPEG_MODULE"

IFS=$'\t' read -r src dest_rel name < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKLIST")
[[ -n "$name" ]] || die "no row $SLURM_ARRAY_TASK_ID in $TASKLIST"
out="$STATE/$dest_rel/$name.probe.json"
[[ -s "$out" ]] && { log "skip: $out exists"; exit 0; }
[[ -s "$src" ]] || die "source missing: $src"
mkdir -p "$STATE/$dest_rel"

log "probe $src"
json=$(ffprobe -v error -select_streams v:0 -count_packets \
	-show_entries stream=codec_name,width,height,r_frame_rate,nb_read_packets -of json "$src")
diag="$src.diag.json"
[[ -s "$diag" ]] || diag="$SRC_ROOT/$name.mkv.diag.json"
python3 - "$json" "$src" "$diag" "$out" <<'PY'
import json, os, sys, time
info = json.loads(sys.argv[1])["streams"][0]
src, diag, out = sys.argv[2], sys.argv[3], sys.argv[4]
num, den = [int(x) for x in info["r_frame_rate"].split("/")]
if den != 1:
    sys.exit("[ERR] non-integer frame rate %s: window planning assumes whole fps" % info["r_frame_rate"])
frames = int(info["nb_read_packets"])
rec = {"source": src, "bytes": os.path.getsize(src), "frames": frames, "fps": num,
       "width": info["width"], "height": info["height"], "codec": info["codec_name"],
       "probed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
if os.path.isfile(diag):
    try:
        d = json.load(open(diag))
        rec["diag_duration_ms"] = d.get("context", {}).get("durationMs")
        rec["diag_frames_emitted"] = d.get("capture", {}).get("framesEmitted")
        rec["diag_clean_close"] = d.get("context", {}).get("cleanClose")
    except Exception as e:  # a broken sidecar must not block the probe
        rec["diag_error"] = str(e)
    dur = rec.get("diag_duration_ms")
    if dur:
        expect = dur / 1000.0 * num
        if abs(expect - frames) > max(0.02 * expect, 2 * num * 60):
            print("[WARN] frame count %d differs from diag durationMs estimate %.0f" % (frames, expect))
tmp = out + ".tmp"
json.dump(rec, open(tmp, "w"), indent=1)
os.rename(tmp, out)
print("[OK] %s: %d frames @ %d fps, %dx%d %s" % (os.path.basename(src), frames, num, info["width"], info["height"], info["codec_name"]))
PY
