#!/bin/bash -l
# tag_roster.sh — which ArUco IDs each colony of a block uses / has dropped, and cuttable sheets
# of the IDs to re-tag. Run on a deigo login node (2-4) once the block's ArUco leg is on the bucket.
#
#   bash tag_roster.sh --block /bucket/ReiterU/Ants/basler/20260928/block01
#        [--chunks 0-11]        default: every chunk whose ArUco files exist for all cameras
#        [--recent 4]           chunks that decide the current status (default 4 = 2 h)
#        [--copies 2]           tags per ID on the sheets
#        [--previous CSV]       roster_status.csv of an earlier run: present then, gone now = dropped
#        [--after JOBID]        wait for this job (e.g. the pipeline's aruco_datacp); needs --chunks
#        [--hmats NPZ] [--x-threshold X] [--out DIR]
#
# Results: <out>/out/{needed_ids.txt, roster_status.csv, retag_left.png/.svg, retag_right.png/.svg}
# default <out> = /flash/ReiterU/$USER/tag_roster/<date>_<block>_c<first>-<last>
# (= V:\<user>\tag_roster\... on the lab Windows PCs; open the PNG in Liene to print).
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TAG_ROSTER_REPO:-$(cd "$HERE/../.." && pwd)}"   # where tracking/colony/map_combine.py lives
PY=/apps/unit/ReiterU/ant_tracking/venv/bin/python
CALIB_ROOT=/bucket/ReiterU/Ants/basler/cameraArray_calib
BLOCK="" CHUNKS="" RECENT=4 COPIES=2 PREVIOUS="" AFTER="" HMATS="" XTHR="" OUT=""
die() { echo "[ERR] $*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
	case "$1" in
		--block) BLOCK="$2"; shift 2 ;;
		--chunks) CHUNKS="$2"; shift 2 ;;
		--recent) RECENT="$2"; shift 2 ;;
		--copies) COPIES="$2"; shift 2 ;;
		--previous) PREVIOUS="$2"; shift 2 ;;
		--after) AFTER="$2"; shift 2 ;;
		--hmats) HMATS="$2"; shift 2 ;;
		--x-threshold) XTHR="$2"; shift 2 ;;
		--out) OUT="$2"; shift 2 ;;
		-h|--help) sed -n '2,15p' "$0"; exit 0 ;;
		*) die "unknown option $1 (see --help)" ;;
	esac
done
[[ -d "$BLOCK/data" ]] || die "--block must be a block folder with data/: '$BLOCK'"
BLOCK="$(cd "$BLOCK" && pwd)"
DATE="$(basename "$(dirname "$BLOCK")")"; NAME="$(basename "$BLOCK")"
[[ -z "$PREVIOUS" || -f "$PREVIOUS" ]] || die "--previous not found: $PREVIOUS"
MAP_COMBINE="$REPO/tracking/colony/map_combine.py"
[[ -f "$MAP_COMBINE" ]] || die "no $MAP_COMBINE (set TAG_ROSTER_REPO to a checkout that has it)"

# Cameras and ArUco dictionary from the block's processing contract (fallback: videos, dict A).
read -r N_CAMS DICT < <("$PY" - "$BLOCK" <<'EOF'
import glob, json, os, sys
b = sys.argv[1]
s = os.path.join(b, "data", "PIPELINE_STATE.json")
st = json.load(open(s)) if os.path.isfile(s) else {}
n = len(st.get("chunking", {}).get("videos", {})) or len(glob.glob(os.path.join(b, "cam[0-9]*.mkv")))
print(n, st.get("detection", {}).get("aruco_dict") or "-")
EOF
)
# A sheet printed from the wrong dictionary carries the right numbers under the wrong patterns.
if [[ "$DICT" == "-" ]]; then
	DICT="$HERE/../custom_dicts/custom_4x4_A100_d4_20260410_103938.npz"
	echo "[WARN] $BLOCK records no ArUco dictionary; assuming $(basename "$DICT")" >&2
elif [[ ! -f "$DICT" ]]; then
	die "the block was detected with $DICT, which is not readable here"
fi
(( N_CAMS > 0 )) || die "cannot tell how many cameras $BLOCK has"

# Chunks: given, or every chunk whose ArUco detections exist for all cameras.
if [[ -n "$CHUNKS" ]]; then
	[[ "$CHUNKS" =~ ^[0-9]+-[0-9]+$ ]] || die "--chunks must be A-B"
	ARRAY="$CHUNKS"; FIRST="${CHUNKS%-*}"; LAST="${CHUNKS#*-}"
else
	[[ -z "$AFTER" ]] || die "--after needs --chunks: the ArUco files it waits for do not exist yet"
	LIST=$(ls "$BLOCK/data" | sed -n 's/^cam[0-9]*_.*_\([0-9]\{3\}\)_aruco_detections\.h5$/\1/p' |
		sort | uniq -c | awk -v n="$N_CAMS" '$1 == n {print $2 + 0}' | sort -n)
	[[ -n "$LIST" ]] || die "no chunk of $BLOCK has ArUco files for all $N_CAMS cameras yet"
	ARRAY=$(echo $LIST | tr ' ' ','); FIRST=$(echo "$LIST" | head -1); LAST=$(echo "$LIST" | tail -1)
	CHUNKS="$ARRAY"
fi

# Homography: the newest calibration recorded on or before the block's date.
if [[ -z "$HMATS" ]]; then
	for d in $(ls -d "$CALIB_ROOT"/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_* 2>/dev/null | sort); do
		[[ "$(basename "$d" | cut -c1-8)" > "${DATE:0:8}" ]] && continue
		[[ -f "$d/frame0/aruco_stitch/aruco_H_mats.npz" ]] && HMATS="$d/frame0/aruco_stitch/aruco_H_mats.npz"
	done
fi
[[ -f "$HMATS" ]] || die "no homography found; pass --hmats"

OUT="${OUT:-/flash/ReiterU/$USER/tag_roster/${DATE}_${NAME}_c$(printf %03d "$FIRST")-$(printf %03d "$LAST")}"
mkdir -p "$OUT/pano" "$OUT/logs" "$OUT/out"
# A rerun into the same folder must not mix old pickles (other split/homography) into the new
# roster, nor leave old sheets looking current if this run's summary never happens.
rm -f "$OUT"/pano/*.pkl "$OUT"/out/*
echo "block  $BLOCK ($N_CAMS cams)   chunks $CHUNKS"
echo "hmats  $HMATS"
echo "dict   $DICT"

DEP=()
[[ -n "$AFTER" ]] && DEP=(--dependency="afterok:$AFTER" --kill-on-invalid-dep=yes)
MAPJ=$(sbatch --parsable --array="$ARRAY" "${DEP[@]}" -o "$OUT/logs/map_%A_%a.out" \
	--export=ALL,BLOCK="$BLOCK",HMATS="$HMATS",PANO="$OUT/pano",MAP_COMBINE="$MAP_COMBINE",PY="$PY",N_CAMS="$N_CAMS",X_THRESHOLD="$XTHR" \
	"$HERE/map_chunk.sbatch")
SUMMARY=("$PY" "$HERE/roster_ids.py" --pano "$OUT/pano" --out "$OUT/out" --npz "$DICT" --block "$BLOCK"
	--chunks "$CHUNKS" --recent "$RECENT" --copies "$COPIES")
[[ -n "$PREVIOUS" ]] && SUMMARY+=(--previous "$PREVIOUS")
SUMJ=$(sbatch --parsable -p compute -c 4 --mem=64G -t 0-02 -J tag_roster_sum \
	--dependency="afterok:$MAPJ" --kill-on-invalid-dep=yes -o "$OUT/logs/summary_%j.out" \
	--wrap "$(printf '%q ' "${SUMMARY[@]}")")
echo "map $MAPJ -> summary $SUMJ" | tee "$OUT/jobs.txt"
WIN="V:\\${OUT#/flash/ReiterU/}"
echo "results: $OUT/out   (Windows: ${WIN//\//\\}\\out)"
