#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  ./submit_chunks.sh --input_folder /path/to/input --output_path /path/to/output

This version submits sbatch jobs based on *files* in input_folder (no chunk directories).
It expects filenames like:
  20251118_121514_chunk000_aruco_panorama_x_left1740.pkl
  20251118_121514_chunk000_aruco_panorama_x_right1740.pkl
(and similarly for sleap, etc.)

Behavior
--------
- Finds all *.pkl in --input_folder matching *_x_left*.pkl and *_x_right*.pkl
- Creates separate job submissions for LEFT and RIGHT (so you can run them as separate srun jobs)
- Within each side, submits one job per unique dataset+chunk prefix:
    <dataset>_chunkNNN
  derived from the filename.
- Calls python worker as:
    python <py_script> --input_file <pkl> --output_path <output_path>

Optional:
  --py_script combine_tracks_one_chunk.py   (default: combine_tracks_one_chunk.py in same dir as submit_chunks.sh)
  --partition compute                      (default: compute)
  --cpus 32                                (default: 32)
  --mem 32G                                (default: 32G)
  --time 0-24:00:00                        (default: 0-24:00:00)
  --job_name combine_tracks                (default: combine_tracks)
  --logs_dir logs                          (default: logs)
  --side left|right|both                   (default: both)
EOF
}

# Defaults
PARTITION="compute"
CPUS="32"
MEM="32G"
TIME="0-24:00:00"
JOB_NAME="combine_tracks"
LOGS_DIR="logs"
SIDE="both"

INPUT_FOLDER=""
OUTPUT_PATH=""
PY_SCRIPT_BASENAME="combine_tracks_one_chunk.py"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_folder) INPUT_FOLDER="$2"; shift 2 ;;
    --output_path)  OUTPUT_PATH="$2"; shift 2 ;;
    --py_script)    PY_SCRIPT_BASENAME="$2"; shift 2 ;;
    --partition)    PARTITION="$2"; shift 2 ;;
    --cpus)         CPUS="$2"; shift 2 ;;
    --mem)          MEM="$2"; shift 2 ;;
    --time)         TIME="$2"; shift 2 ;;
    --job_name)     JOB_NAME="$2"; shift 2 ;;
    --logs_dir)     LOGS_DIR="$2"; shift 2 ;;
    --side)         SIDE="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${INPUT_FOLDER}" || -z "${OUTPUT_PATH}" ]]; then
  echo "ERROR: --input_folder and --output_path are required." >&2
  usage
  exit 2
fi

if [[ ! -d "${INPUT_FOLDER}" ]]; then
  echo "ERROR: input_folder does not exist or is not a directory: ${INPUT_FOLDER}" >&2
  exit 2
fi

case "${SIDE}" in
  left|right|both) ;;
  *) echo "ERROR: --side must be left|right|both (got: ${SIDE})" >&2; exit 2 ;;
esac

# Resolve script directory and python worker path (same folder as this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/${PY_SCRIPT_BASENAME}"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "ERROR: Python worker script not found next to submit_chunks.sh: ${PY_SCRIPT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_PATH}"
mkdir -p "${LOGS_DIR}"

# Extract a stable key "<dataset>_chunkNNN" from filenames:
#   20251118_121514_chunk000_aruco_panorama_x_left1740.pkl  -> 20251118_121514_chunk000
extract_key() {
  local bn="$1"
  # Use sed to capture "<anything>_chunkDDD" (DDD = 3 digits) at the start of name.
  # If not matched, return empty.
  echo "${bn}" | sed -nE 's/^(.+_chunk[0-9]{3}).*/\1/p'
}

submit_side() {
  local side="$1"

  mapfile -t FILES < <(find "${INPUT_FOLDER}" -maxdepth 1 -type f -name "*_x_${side}*.pkl" | sort)
  local N=${#FILES[@]}

  if [[ ${N} -eq 0 ]]; then
    echo "No *${side}*.pkl files found in ${INPUT_FOLDER}"
    return 0
  fi

  echo
  echo "Side=${side}: Found ${N} candidate PKL files."

  # Choose one representative file per key (<dataset>_chunkNNN) for this side.
  # If there are multiple matches for a key+side, we warn and pick the first lexicographically.
  declare -A chosen=()
  declare -A dupcount=()

  for fp in "${FILES[@]}"; do
    bn="$(basename "${fp}")"
    key="$(extract_key "${bn}")"
    if [[ -z "${key}" ]]; then
      echo "WARN: Could not extract <dataset>_chunkNNN key from filename, skipping: ${bn}" >&2
      continue
    fi
    if [[ -z "${chosen[$key]+x}" ]]; then
      chosen["$key"]="$fp"
      dupcount["$key"]=1
    else
      dupcount["$key"]=$((dupcount["$key"] + 1))
    fi
  done

  local keys=("${!chosen[@]}")
  IFS=$'\n' keys=($(sort <<<"${keys[*]}"))
  unset IFS

  local K=${#keys[@]}
  if [[ ${K} -eq 0 ]]; then
    echo "ERROR: No usable keys found for side=${side} in ${INPUT_FOLDER}" >&2
    return 1
  fi

  echo "Side=${side}: Submitting ${K} jobs (one per <dataset>_chunkNNN)."
  echo "sbatch: -p ${PARTITION} -c ${CPUS} --mem=${MEM} -t ${TIME}"
  echo "Python worker: ${PY_SCRIPT}"
  echo "Logs dir: ${LOGS_DIR}"
  echo

  local submitted=0
  for key in "${keys[@]}"; do
    fp="${chosen[$key]}"
    bn="$(basename "${fp}")"

    if [[ "${dupcount[$key]}" -gt 1 ]]; then
      echo "WARN: ${key} side=${side} has ${dupcount[$key]} matching files; using: ${bn}" >&2
    fi

    # per-job script
    jobfile="$(mktemp "${LOGS_DIR}/sbatch_${JOB_NAME}_${key}_${side}_XXXXXX.sh")"

    cat > "${jobfile}" <<EOF
#!/usr/bin/env bash
#SBATCH -J ${JOB_NAME}_${side}_${key}
#SBATCH -p ${PARTITION}
#SBATCH -c ${CPUS}
#SBATCH --mem=${MEM}
#SBATCH -t ${TIME}
#SBATCH -o ${LOGS_DIR}/${JOB_NAME}_${side}_${key}_%j.out
#SBATCH -e ${LOGS_DIR}/${JOB_NAME}_${side}_${key}_%j.err

set -euo pipefail
source ~/.bashrc
conda activate aruco_env

echo "Running on host: \$(hostname)"
echo "Input file: ${fp}"
echo "Side: ${side}"
echo "Key: ${key}"
echo "Output root: ${OUTPUT_PATH}"
echo "Python: \$(which python)"

python "${PY_SCRIPT}" --input_file "${fp}" --output_path "${OUTPUT_PATH}"
EOF

    chmod +x "${jobfile}"

    if jobid=$(sbatch --parsable "${jobfile}"); then
      echo "Submitted side=${side} key=${key} -> job ${jobid}"
      submitted=$((submitted + 1))
    else
      echo "ERROR: sbatch submission failed for side=${side} key=${key}" >&2
    fi

    rm -f "${jobfile}"
  done

  echo "Side=${side}: Submitted ${submitted}/${K} jobs."
}

# Run submissions
case "${SIDE}" in
  left)
    submit_side left
    ;;
  right)
    submit_side right
    ;;
  both)
    submit_side left
    submit_side right
    ;;
esac

echo
echo "Done."
echo "Check status: squeue -u \$USER"
