#!/bin/bash
# SLURM DAG driver for the W1.1 PDA-C / TagProbe centroid evaluation harness.
#
# Contract: group_labelling/EVAL_HARNESS_DESIGN.md
#
#   ./group_labelling/submit_eval.sh [--out-dir DIR] [--cameras cam10,cam22]
#                                    [--blocks block01,block02,block03]
#                                    [--concurrency 8] [--force_submit]
#                                    [--notify-email you@example.com]
#
# Chain:
#   0 sample anchors        (login, serial)
#   A extract + ArUco       (compute array, %16)
#   B predict both models   (largegpu array, %8 -- exactly the association slice)
#   C cluster + stratify    (compute, serial, after B)
#   -- MANUAL: human adjudication of the sampled clusters --
#   E estimate              (compute, serial)
#   F report                (compute, serial)
#
# Stages E/F run without verdicts too: they emit the reference-free Cobs and the
# disagreement inventory and explicitly refuse to estimate Delta or to decide.
#
# Concurrency: at --cpus-per-task=16 --mem=128G --gres=gpu:a100:1 the Stage B
# array at %8 sits on all three association caps at once (cpu=128, gpu=8,
# mem=1T). Any concurrent production submission will block it or be blocked by
# it -- use --concurrency 4 outside an agreed window.

set -euo pipefail

OUT_DIR=/work/ReiterU/centroid_eval
BLOCKS=block01,block02,block03
CAMERAS=
CONCURRENCY=8
FORCE=0
NOTIFY=
PY=/apps/unit/ReiterU/sleap-nn/0.3.1/tools/sleap-nn/bin/python
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)       OUT_DIR="$2"; shift 2 ;;
    --blocks)        BLOCKS="$2"; shift 2 ;;
    --cameras)       CAMERAS="$2"; shift 2 ;;
    --concurrency)   CONCURRENCY="$2"; shift 2 ;;
    --notify-email)  NOTIFY="$2"; shift 2 ;;
    --force_submit)  FORCE=1; shift ;;
    -h|--help)       sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

LOGS="$OUT_DIR/logs"
mkdir -p "$LOGS"
SUBMIT_LOG="$LOGS/submit_${STAMP}.log"
exec > >(tee -a "$SUBMIT_LOG") 2>&1
echo "=== centroid eval submit $STAMP ==="
echo "out_dir=$OUT_DIR blocks=$BLOCKS cameras=${CAMERAS:-ALL} concurrency=$CONCURRENCY"

# Refuse duplicate concurrent DAGs. Two harness runs sharing an out-dir
# interleave their arrays and silently corrupt each other's outputs.
if [[ $FORCE -eq 0 ]]; then
  if squeue -u "$USER" -h -o "%j" | grep -q "^cev[ABCEF]_"; then
    echo "ERROR: a centroid-eval DAG is already queued or running:" >&2
    squeue -u "$USER" -o "%.10i %.20j %.8T %.10M" | grep -E "JOBID|cev[ABCEF]_" >&2
    echo "Re-run with --force_submit only if you are certain they do not share" >&2
    echo "an --out-dir." >&2
    exit 1
  fi
fi

MAIL=()
if [[ -n "$NOTIFY" ]]; then
  MAIL=(--mail-user="$NOTIFY" --mail-type=END,FAIL)
fi

# ---- Stage 0: anchors (login node, seconds to minutes) ---------------------
echo "--- stage 0: sampling anchors ---"
CAM_ARG=()
[[ -n "$CAMERAS" ]] && CAM_ARG=(--cameras "$CAMERAS")
"$PY" "$REPO/group_labelling/eval_sample_anchors.py" \
  --out-dir "$OUT_DIR" --blocks "$BLOCKS" "${CAM_ARG[@]}"

N_KEYS=$("$PY" -c "
import json
s = json.load(open(r'$OUT_DIR/strata.json'))
print(sum(len(v) for v in s['strata'].values()))")
if [[ "$N_KEYS" -eq 0 ]]; then
  echo "ERROR: no camera-runs sampled; nothing to submit." >&2
  exit 1
fi
LAST=$((N_KEYS - 1))
echo "$N_KEYS camera-runs -> array 0-$LAST"

COMMON=(--cpus-per-task=16 --mem=128G --exclude=saion-gpu25 "${MAIL[@]}")

# ---- Stage A: extract + ArUco (CPU) ----------------------------------------
JID_A=$(sbatch --parsable \
  --job-name="cevA_${STAMP}" --partition=compute \
  --array=0-${LAST}%16 --time=4:00:00 \
  --output="$LOGS/cevA_%A_%a.log" \
  "${COMMON[@]}" \
  --wrap "$PY $REPO/group_labelling/eval_extract_and_aruco.py \
            --out-dir $OUT_DIR --array-index \$SLURM_ARRAY_TASK_ID")
echo "stage A: $JID_A"

# ---- Stage B: both models on the same clip (GPU) ---------------------------
JID_B=$(sbatch --parsable --dependency=afterok:"$JID_A" \
  --job-name="cevB_${STAMP}" --partition=largegpu --gres=gpu:a100:1 \
  --array=0-${LAST}%${CONCURRENCY} --time=2:00:00 \
  --output="$LOGS/cevB_%A_%a.log" \
  "${COMMON[@]}" \
  --wrap "$PY $REPO/group_labelling/eval_predict_centroids.py \
            --out-dir $OUT_DIR --array-index \$SLURM_ARRAY_TASK_ID")
echo "stage B: $JID_B (after A)"

# ---- Stage C: clusters + HT sampling ---------------------------------------
JID_C=$(sbatch --parsable --dependency=afterok:"$JID_B" \
  --job-name="cevC_${STAMP}" --partition=compute --time=4:00:00 \
  --output="$LOGS/cevC_%j.log" \
  "${COMMON[@]}" \
  --wrap "$PY $REPO/group_labelling/eval_cluster_and_stratify.py \
            --out-dir $OUT_DIR --r-pair-envelope")
echo "stage C: $JID_C (after B)"

# ---- Stages E/F: estimate + report -----------------------------------------
# afterany, not afterok: if C partially fails we still want the inventory and a
# report saying what is missing, rather than silence.
JID_E=$(sbatch --parsable --dependency=afterany:"$JID_C" \
  --job-name="cevE_${STAMP}" --partition=compute --time=2:00:00 \
  --output="$LOGS/cevE_%j.log" \
  "${COMMON[@]}" \
  --wrap "$PY $REPO/group_labelling/eval_estimate.py --out-dir $OUT_DIR")
echo "stage E: $JID_E (after C)"

JID_F=$(sbatch --parsable --dependency=afterany:"$JID_E" \
  --job-name="cevF_${STAMP}" --partition=compute --time=1:00:00 \
  --output="$LOGS/cevF_%j.log" \
  "${COMMON[@]}" \
  --wrap "$PY $REPO/group_labelling/eval_report.py --out-dir $OUT_DIR")
echo "stage F: $JID_F (after E)"

cat <<EOF

Submitted: A=$JID_A B=$JID_B C=$JID_C E=$JID_E F=$JID_F
Watch:   squeue -u $USER -o '%.10i %.20j %.8T %.10M %R'
Logs:    $LOGS
Report:  $OUT_DIR/report/report.md

NEXT, AND IT IS MANUAL: stage F will report AWAITING_ADJUDICATION. The primary
statistic Delta cannot be computed until a human adjudicates the sampled
disagreement clusters listed in $OUT_DIR/clusters/*.parquet (rows where
sampled == True). Write verdicts to $OUT_DIR/adjudication/verdicts.jsonl, one
object per line:

  {"key":"block01_cam10","operating_point":"pi_match","clip_frame":0,
   "cluster":3,"k":2,"tp_A0":2,"tp_all6":1,"cant_tell":false,"flags":[]}

then re-run stages E and F. /bucket is read-only on compute nodes, so copy the
finished report from the login node:
  cp -r $OUT_DIR/report /bucket/ReiterU/Ants/SLEAP_files/Group_labelling/
EOF
