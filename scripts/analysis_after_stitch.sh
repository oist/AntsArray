#!/usr/bin/env bash
# analysis_after_stitch.sh — wait for a stitch completion marker, then fan out
# the per-track analysis routines over the stitched per-track parquets, conda-free.
#
# Routines dispatched (one Slurm job per TrackID_*.parquet, via
# scripts/per_track_slurm_fanout.sh):
#   colony_presence  (analysis/compute_track_colony_presence_vector.py)
#   speed_vector     (analysis/compute_track_speed_vector.py)   [needs scipy]
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
  --run_sleep             Also fan out sleep predictions (needs --sleep_model).
  --sleep_model PATH      Trained sleep classifier for --run_sleep.
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
    --run_sleep) run_sleep=1; shift ;;
    --sleep_model) sleep_model="$2"; run_sleep=1; shift 2 ;;
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
  exit 1
fi
log "found $n_tracks per-track parquet(s); fanning out analysis routines"

# 3) Fan out each routine. One routine failing must not stop the others.
run_routine() {  # $1 operation_script(basename)  $2 operation_name  $3 output_name
  local script="$1" opname="$2" outname="$3"
  # Absolute venv python + relative script (worker cd's into --run_workdir=repo).
  local op_cmd="$python_bin analysis/$script --track \"\$TRACK_PATH\" --out \"\$TASK_OUTPUT_DIR\""
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
run_routine compute_track_grid_occupancy.py         grid_occupancy  grid_occupancy_histograms

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
