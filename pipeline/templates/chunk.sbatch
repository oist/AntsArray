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

flash_dir="$FLASH_ROOT/$vname"
mkdir -p "$flash_dir"
chmod 2775 "$flash_dir"

# Idempotency: skip if any chunk already present (assume prior successful run)
if compgen -G "$flash_dir/${vname}_[0-9][0-9][0-9].${OUT_EXT}" > /dev/null; then
	echo "[SKIP] chunks already present in $flash_dir"
	exit 0
fi

echo "[$(date)] chunk $vname -> $flash_dir/${vname}_%03d.${OUT_EXT} (segment_time=${CHUNK_SEC}s)"
ffmpeg -hide_banner -nostdin -y -i "$source_path" \
	-c copy -map 0 \
	-f segment -segment_time "$CHUNK_SEC" -reset_timestamps 1 \
	"$flash_dir/${vname}_%03d.${OUT_EXT}"

n_produced=$(ls -1 "$flash_dir/${vname}"_[0-9][0-9][0-9].${OUT_EXT} 2>/dev/null | wc -l)
echo "[OK] $vname produced $n_produced chunks"
