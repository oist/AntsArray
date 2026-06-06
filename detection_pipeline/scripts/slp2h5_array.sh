#!/bin/bash -l
# slp2h5_array.sh — backfill .slp -> _sleap_data.h5 on deigo.
#
# Reads $WORKLIST (one absolute .slp path per line), processes BATCH lines per
# array task. Output .h5 is written under $OUT_BASE/<block>/ on /flash because
# /bucket is read-only from deigo compute nodes. A separate post-stage rsync
# from a login node syncs /flash/.../h5_out/<block>/ -> /bucket/.../<block>/data/.
# Idempotent: skips files where the .h5 already exists on either bucket or flash.
#
# Submission (one-time, after staging the script + worklist):
#   sbatch --parsable --array=0-${UPPER}%${CONC} ~/detection_pipeline/scripts/slp2h5_array.sh
#
# WORKLIST default lives at /flash/ReiterU/$USER/slp2h5/worklist.txt
# OUT_BASE default  lives at /flash/ReiterU/$USER/slp2h5/h5_out
# Logs land in /flash/ReiterU/$USER/slp2h5/<jid>_<task>.{out,err}
#SBATCH -p compute
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0-4
#SBATCH -J slp2h5
#SBATCH -o /flash/ReiterU/%u/slp2h5/%A_%a.out
#SBATCH -e /flash/ReiterU/%u/slp2h5/%A_%a.err
set -eo pipefail
module load python/3.11.4

SCRIPT="${SLEAP2H5_SCRIPT:-$HOME/detection_pipeline/scripts/sleap2h5.py}"
WORKLIST="${WORKLIST:-/flash/ReiterU/$USER/slp2h5/worklist.txt}"
OUT_BASE="${OUT_BASE:-/flash/ReiterU/$USER/slp2h5/h5_out}"
BATCH="${BATCH:-50}"

start=$(( SLURM_ARRAY_TASK_ID * BATCH ))
end=$(( start + BATCH ))
echo "[task $SLURM_ARRAY_TASK_ID] worklist=$WORKLIST  rows $start..$((end-1))"

for (( i=start; i<end; i++ )); do
    slp=$(sed -n "$((i+1))p" "$WORKLIST")
    [[ -z "$slp" ]] && break
    stem=$(basename "$slp" .slp)
    block=$(echo "$slp" | sed -n 's|.*/20260515/\([^/]*\)/data/.*|\1|p')
    out_dir="$OUT_BASE/$block"
    flash_out="$out_dir/${stem}_sleap_data.h5"
    bucket_out="$(dirname "$slp")/${stem}_sleap_data.h5"
    if [[ -s "$bucket_out" || -s "$flash_out" ]]; then
        echo "[SKIP] $stem"
        continue
    fi
    mkdir -p "$out_dir"
    echo "[$(date +%H:%M:%S)] $stem -> $flash_out"
    python3 "$SCRIPT" "$slp" "$out_dir" || echo "[WARN] failed: $slp" >&2
done
echo "[task $SLURM_ARRAY_TASK_ID] DONE"
