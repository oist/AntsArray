#!/usr/bin/env bash
# analysis_after_stitch.sh — wait for a stitch completion marker, then fan out
# the per-track analysis routines over the stitched per-track parquets, conda-free.
#
# Routines dispatched (one Slurm job per TrackID_*.parquet, via
# scripts/per_track_slurm_fanout.sh):
#   colony_presence  (analysis/compute_track_colony_presence_vector.py)
#   speed_vector     (analysis/compute_track_speed_vector.py)   [needs scipy]
#   sleep_motion     (analysis/compute_track_sleep_motion.py)
#   grid_occupancy   (analysis/compute_track_grid_occupancy.py)
#   sleep_prediction (optional; --run_sleep, needs joblib+scikit-learn + a model)
#
# Conda-free: each worker runs the operation with an ABSOLUTE venv python path
# (--python_bin) and --no_conda, so nothing depends on conda or PATH. Bucket
# destination is passed explicitly (--bucket_output_dir) so it works for both
# <blocks_root>/<block> and date-only layouts.
#
# Designed to be launched (nohup) by submit_blocks_pipeline.sh after the tracking
# DAG is submitted; also runnable standalone.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"

stitch_ok=""
per_track_dir=""
bucket_stitched=""
python_bin="/apps/unit/ReiterU/ant_tracking/venv/bin/python"
repo="$REPO_DEFAULT"
partition="compute"
sbatch_bin="sbatch"
cpus=4
mem="16G"
time_limit="0-12:00:00"
poll_seconds=120
timeout_secs=172800   # 48h overall deadline
run_sleep=0
sleep_model=""
fps="${FPS:-24.0}"
sleep_motion_mm_per_px="${SLEEP_MOTION_MM_PER_PX:-0.016}"
sleep_motion_cache_max_gap_frames="${SLEEP_MOTION_CACHE_MAX_GAP_FRAMES:-120}"
grid_size_mm="${GRID_OCCUPANCY_GRID_SIZE_MM:-0.25}"
grid_bounds_json="${GRID_OCCUPANCY_BOUNDS_JSON:-}"
grid_output_name="${GRID_OCCUPANCY_OUTPUT_NAME:-grid_occupancy_histograms}"
notify_email=""

usage() {
  cat <<EOF
Wait for a stitch marker, then fan out per-track analysis routines conda-free.

Required:
  --per_track_dir PATH    Stitched per-track parquet folder (TrackID_*.parquet).
  --bucket_stitched PATH  Bucket <block>/stitched dir; routine outputs go under it.

Options:
  --stitch_ok PATH        Marker file to wait for (fresh mtime) before starting.
                          Omit to start immediately.
  --python_bin PATH       Worker python (ant_tracking venv). Default: $python_bin
  --repo PATH             Repo root (worker working dir). Default: $repo
  --partition NAME        SLURM partition. Default: compute
  --sbatch_bin PATH       sbatch executable. Default: sbatch
  --cpus N                CPUs per per-track job. Default: 4
  --mem MEM               Memory per per-track job. Default: 16G
  --time TIME             Time per per-track job. Default: 0-12:00:00
  --poll_seconds N        Poll interval while waiting for the marker. Default: 120
  --timeout N             Overall wait deadline (s). Default: 172800 (48h)
  --fps FLOAT             Frame rate for all-bodypoint sleep motion. Default: $fps
  --sleep_motion_mm_per_px FLOAT
                          Spatial scale for all-bodypoint sleep motion. Default: $sleep_motion_mm_per_px
  --sleep_motion_cache_max_gap_frames N
                          Largest retained pose gap. Default: $sleep_motion_cache_max_gap_frames
  --run_sleep             Also fan out sleep predictions (needs --sleep_model).
  --sleep_model PATH      Trained sleep classifier for --run_sleep.
  --grid_size_mm FLOAT    Occupancy histogram bin size in mm. Default: $grid_size_mm
  --grid_bounds_json PATH Optional inferred bounds JSON for occupancy histograms.
  --grid_output_name NAME Occupancy output folder. Default: $grid_output_name
  --notify_email ADDR     Email ADDR if this watcher aborts (stitch deadline,
                          no per-track parquets). Default: off.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stitch_ok) stitch_ok="$2"; shift 2 ;;
    --per_track_dir) per_track_dir="$2"; shift 2 ;;
    --bucket_stitched) bucket_stitched="$2"; shift 2 ;;
    --python_bin) python_bin="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --sbatch_bin) sbatch_bin="$2"; shift 2 ;;
    --cpus) cpus="$2"; shift 2 ;;
    --mem) mem="$2"; shift 2 ;;
    --time) time_limit="$2"; shift 2 ;;
    --poll_seconds) poll_seconds="$2"; shift 2 ;;
    --timeout) timeout_secs="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --sleep_motion_mm_per_px) sleep_motion_mm_per_px="$2"; shift 2 ;;
    --sleep_motion_cache_max_gap_frames) sleep_motion_cache_max_gap_frames="$2"; shift 2 ;;
    --run_sleep) run_sleep=1; shift ;;
    --sleep_model) sleep_model="$2"; run_sleep=1; shift 2 ;;
    --grid_size_mm) grid_size_mm="$2"; shift 2 ;;
    --grid_bounds_json) grid_bounds_json="$2"; shift 2 ;;
    --grid_output_name) grid_output_name="$2"; shift 2 ;;
    --notify_email) notify_email="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
fmt_elapsed() { local s="$1"; printf '%dh%02dm' "$(( s / 3600 ))" "$(( (s % 3600) / 60 ))"; }

[[ -n "$per_track_dir" ]] || { echo "ERROR: --per_track_dir is required" >&2; exit 2; }
[[ -n "$bucket_stitched" ]] || { echo "ERROR: --bucket_stitched is required" >&2; exit 2; }
fanout="$repo/scripts/per_track_slurm_fanout.sh"
[[ -f "$fanout" ]] || { echo "ERROR: fan-out script not found: $fanout" >&2; exit 2; }

# Best-effort email on abort (helper lives in the detection tree; repo deploys
# as one unit). Missing helper degrades to log-only, never to a crash.
NOTIFY_EMAIL="$notify_email"
notify_lib="$repo/detection_pipeline/lib/notify.sh"
if [[ -n "$NOTIFY_EMAIL" && -f "$notify_lib" ]]; then
  source "$notify_lib"
else
  [[ -n "$NOTIFY_EMAIL" ]] && log "WARN: notify helper not found: $notify_lib; emails disabled"
  notify_send() { :; }
fi

start_epoch="$(date +%s)"
deadline=$(( start_epoch + timeout_secs ))

log "analysis fan-out watcher start"
log "  per_track_dir=$per_track_dir"
log "  bucket_stitched=$bucket_stitched"
log "  python_bin=$python_bin  repo=$repo"
log "  stitch_ok=${stitch_ok:-<none: start now>}"

# 1) Wait for the stitch marker (fresh) if one was given.
if [[ -n "$stitch_ok" ]]; then
  log "waiting for stitch marker: $stitch_ok"
  while true; do
    if [[ -s "$stitch_ok" ]]; then
      m="$(stat -c %Y "$stitch_ok" 2>/dev/null || echo 0)"
      if [[ "$m" =~ ^[0-9]+$ ]] && (( m >= start_epoch )); then
        log "fresh stitch marker found (mtime=$m)"
        break
      fi
    fi
    if (( $(date +%s) >= deadline )); then
      log "ERROR: deadline reached waiting for stitch marker; aborting analysis"
      notify_send "[AntsArray] analysis watcher TIMEOUT waiting for stitch" \
"analysis_after_stitch.sh gave up after ${timeout_secs}s waiting for a fresh stitch marker:
$stitch_ok
No per-track analysis jobs were submitted for $per_track_dir.
The stitch job likely failed or was cancelled; check its Slurm FAIL mail / logs."
      exit 1
    fi
    log "still waiting for stitch (elapsed $(fmt_elapsed $(( $(date +%s) - start_epoch ))))"
    sleep "$poll_seconds"
  done
fi

# 2) Ensure per-track parquets exist.
n_tracks=$( { find "$per_track_dir" -maxdepth 1 -type f -name 'TrackID_*.parquet' 2>/dev/null || true; } | wc -l )
if (( n_tracks == 0 )); then
  log "ERROR: no TrackID_*.parquet under $per_track_dir; nothing to analyze"
  notify_send "[AntsArray] analysis watcher FAILED: no per-track parquets" \
"analysis_after_stitch.sh found a fresh stitch marker but no TrackID_*.parquet under:
$per_track_dir
No analysis jobs were submitted; the stitch output looks empty or misplaced."
  exit 1
fi
log "found $n_tracks per-track parquet(s); fanning out analysis routines"

# 3) Fan out each routine. One routine failing must not stop the others.
run_routine() {  # $1 operation_script(basename)  $2 operation_name  $3 output_name  $4 extra operation args
  local script="$1" opname="$2" outname="$3" extra_args="${4:-}"
  # Absolute venv python + relative script (worker cd's into --run_workdir=repo).
  local op_cmd="$python_bin analysis/$script --track \"\$TRACK_PATH\" --out \"\$TASK_OUTPUT_DIR\""
  if [[ -n "$extra_args" ]]; then
    op_cmd+=" $extra_args"
  fi
  log "=== fan-out $opname ($script) ==="
  if bash "$fanout" \
      --per_track_dir "$per_track_dir" \
      --operation_cmd "$op_cmd" \
      --operation_name "$opname" \
      --output_name "$outname" \
      --no_conda \
      --run_workdir "$repo" \
      --bucket_output_dir "$bucket_stitched/$outname" \
      --partition "$partition" \
      --cpus "$cpus" --mem "$mem" --time "$time_limit" \
      --sbatch_bin "$sbatch_bin"; then
    log "dispatched $opname"
  else
    log "WARN: fan-out $opname failed (see per_track_slurm_fanout output above)"
  fi
}

run_routine compute_track_colony_presence_vector.py colony_presence colony_presence_vectors
run_routine compute_track_speed_vector.py           speed_vector    speed_vectors
sleep_motion_args=(
  --fps "$fps"
  --mm_per_px "$sleep_motion_mm_per_px"
  --cache_max_gap_frames "$sleep_motion_cache_max_gap_frames"
)
sleep_motion_args_text="$(printf ' %q' "${sleep_motion_args[@]}")"
sleep_motion_args_text="${sleep_motion_args_text# }"
run_routine compute_track_sleep_motion.py           sleep_motion    sleep_motion "$sleep_motion_args_text"
grid_args=(--grid_size_mm "$grid_size_mm")
if [[ -n "$grid_bounds_json" ]]; then
  grid_args+=(--bounds_json "$grid_bounds_json")
fi
grid_args_text="$(printf ' %q' "${grid_args[@]}")"
grid_args_text="${grid_args_text# }"
run_routine compute_track_grid_occupancy.py         grid_occupancy  "$grid_output_name" "$grid_args_text"

if (( run_sleep == 1 )); then
  if [[ -z "$sleep_model" ]]; then
    log "WARN: --run_sleep set but no --sleep_model; skipping sleep predictions"
  else
    speed_root="$(dirname "$per_track_dir")/speed_vectors"
    op_cmd="$python_bin analysis/compute_track_sleep_predictions.py --track \"\$TRACK_PATH\" --out \"\$TASK_OUTPUT_DIR\" --model $sleep_model --speed_root $speed_root"
    log "=== fan-out sleep_prediction ==="
    bash "$fanout" \
      --per_track_dir "$per_track_dir" \
      --operation_cmd "$op_cmd" \
      --operation_name sleep_prediction \
      --output_name sleep_predictions \
      --no_conda \
      --run_workdir "$repo" \
      --bucket_output_dir "$bucket_stitched/sleep_predictions" \
      --partition "$partition" \
      --cpus "$cpus" --mem "$mem" --time "$time_limit" \
      --sbatch_bin "$sbatch_bin" \
      || log "WARN: sleep_prediction fan-out failed"
  fi
fi

log "analysis fan-out dispatch complete"
