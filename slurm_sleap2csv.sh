#!/bin/bash -l
# File: slurm_sleap2csv.sh
set -euo pipefail

# Usage: ./slurm_sleap2csv.sh /path/to/slp_dir /path/to/output_dir
INPUT_DIR="${1:-}"
OUTPUT_DIR="${2:-}"
[[ -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" ]] && { echo "Usage: $0 <input_slp_dir> <output_dir>"; exit 1; }

# Slurm resources (override via env)
PARTITION="${PARTITION:-compute}"
TIME="${TIME:-0-06:00}"
CPUS="${CPUS:-4}"
MEM="${MEM:-8G}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-0.05}"

# Paths
INPUT_DIR="$(readlink -f "$INPUT_DIR")"
OUTPUT_DIR="$(readlink -f "$OUTPUT_DIR")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/sleap2csv.py"
[[ -f "$CONVERTER" ]] || { echo "Converter not found: $CONVERTER" >&2; exit 1; }

JOBS_DIR="${JOBS_DIR:-jobs_slp2csv_$(date +%Y%m%d-%H%M%S)}"
JOBS_DIR="$(readlink -m "$JOBS_DIR")"
mkdir -p "$OUTPUT_DIR" "$JOBS_DIR"

# Helper: sed-escape
esc() { sed 's/[\/&]/\\&/g' <<<"$1"; }

# Submit
shopt -s nullglob
submitted=0 skipped=0 idx=0
echo "Scanning: $INPUT_DIR"

for slp in "$INPUT_DIR"/*.slp; do
  ((++idx))
  b="$(basename "$slp")"
  [[ "$b" =~ ^\. ]] && continue
  vname="${b%.slp}"
  csv_out="$OUTPUT_DIR/${vname}.csv"

  if [[ -e "$csv_out" ]]; then
    echo "[SKIP] ${vname} (exists)"
    ((++skipped))
    continue
  fi

  job="$JOBS_DIR/slp2csv-${vname}.sh"
  cat > "$job" <<'EOFJOB'
#!/bin/bash -l
#SBATCH -t __TIME__
#SBATCH -c __CPUS__
#SBATCH --partition=__PARTITION__
#SBATCH --mem=__MEM__
#SBATCH -J slp2csv-__VNAME__
#SBATCH -o __JOBS_DIR__/slp2csv-__VNAME__%j.out
#SBATCH -e __JOBS_DIR__/slp2csv-__VNAME__%j.err
set -euo pipefail
shopt -s nullglob

source ~/.bashrc 
conda activate sleap

slp_file="__SLP_FILE__"
out_dir="__OUT_DIR__"
converter="__CONVERTER__"

# Skip inside the job too (race-safe)
stem="$(basename "${slp_file%.slp}")"
csv="${out_dir}/${stem}.csv"
if [[ -e "$csv" ]]; then
  echo "[SKIP] ${stem} (exists)"
  exit 0
fi

python "$converter" "$slp_file" "$out_dir"

echo "Done: $(date -Iseconds)"
EOFJOB

  # Token replace
  sed -i \
    -e "s/__TIME__/$(esc "$TIME")/" \
    -e "s/__CPUS__/$(esc "$CPUS")/" \
    -e "s/__PARTITION__/$(esc "$PARTITION")/" \
    -e "s/__MEM__/$(esc "$MEM")/" \
    -e "s/__VNAME__/$(esc "$vname")/g" \
    -e "s#__JOBS_DIR__#$(esc "$JOBS_DIR")#g" \
    -e "s#__SLP_FILE__#$(esc "$slp")#" \
    -e "s#__OUT_DIR__#$(esc "$OUTPUT_DIR")#" \
    -e "s#__CONVERTER__#$(esc "$CONVERTER")#" \
    "$job"
  chmod +x "$job"

  # Submit
  set +e
  jid_line=$(sbatch "$job" 2> "${JOBS_DIR}/sbatch-${vname}.err")
  rc=$?
  set -e

  if [[ $rc -eq 0 && -n "$jid_line" ]]; then
    jid="${jid_line##* }"
    echo "${vname} -> job ${jid}"
    ((++submitted))
  else
    echo "[FAIL] ${vname} (rc=$rc). See ${JOBS_DIR}/sbatch-${vname}.err" >&2
  fi

  sleep "$SLEEP_BETWEEN"
done
shopt -u nullglob

echo "Submitted: ${submitted}  Skipped: ${skipped}"
echo "Logs: ${JOBS_DIR}"
echo "Outputs: ${OUTPUT_DIR}"

# ---------- final move step (login node) ----------
# Move any CSVs currently in OUTPUT_DIR back into INPUT_DIR.
# Safe if none exist; does not overwrite existing files.
shopt -s nullglob
moved=0
for csv in "$OUTPUT_DIR"/*.csv; do
  bn="$(basename "$csv")"
  if mv -n -- "$csv" "$INPUT_DIR/$bn"; then
    echo "[MOVED] $bn -> $INPUT_DIR/"
    ((++moved))
  else
    echo "[MOVE-FAIL] $bn" >&2
  fi
done
shopt -u nullglob
echo "Move summary: moved=${moved} from ${OUTPUT_DIR} to ${INPUT_DIR}"
