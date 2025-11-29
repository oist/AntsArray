#!/bin/bash
# compare_sleap_inference.sh
# Usage: ./compare_sleap_inference.sh --version <1.4.1|1.5.2|sleap-nn> --node <gpu|largegpu> --dir <video_dir>

set -euo pipefail

# Defaults
DEFAULT_MODEL_CENTROID="/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centroid/"
DEFAULT_MODEL_INSTANCE="/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centered_instance/"

VERSION=""
NODE=""
DIR=""
MODEL_CENTROID="$DEFAULT_MODEL_CENTROID"
MODEL_INSTANCE="$DEFAULT_MODEL_INSTANCE"

usage() {
    echo "Usage: $0 --version <1.4.1|1.5.2|sleap-nn> --node <gpu|largegpu> --dir <video_dir> [--model_centroid <path>] [--model_instance <path>]"
    exit 1
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --node) NODE="$2"; shift 2 ;;
        --dir) DIR="$2"; shift 2 ;;
        --model_centroid) MODEL_CENTROID="$2"; shift 2 ;;
        --model_instance) MODEL_INSTANCE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "$VERSION" || -z "$NODE" || -z "$DIR" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

if [[ ! -d "$DIR" ]]; then
    echo "Error: Directory '$DIR' does not exist."
    exit 1
fi

# Setup paths
BASE_NAME=$(basename "$DIR")
WORK_DIR="/work/ReiterU/ant_tmp/$BASE_NAME"
JOBS_DIR="/work/ReiterU/ant_tmp/jobs/$BASE_NAME"

echo "Work directory: $WORK_DIR"
echo "Jobs directory: $JOBS_DIR"

mkdir -p "$WORK_DIR" "$JOBS_DIR"
chmod 2775 "$WORK_DIR" "$JOBS_DIR" 2>/dev/null || echo "Warning: Could not chmod directories."

# Configure version-specific settings
MODULE_CMD="module load sleap/$VERSION"
VERBOSITY_FLAG="--verbosity json"
TRACK_CMD="sleap-track"

if [[ "$VERSION" == "sleap-nn" ]]; then
    MODULE_CMD="module load python/3.12.9 sleap-nn"
    VERBOSITY_FLAG=""
    TRACK_CMD="sleap-nn track"
elif [[ "$VERSION" =~ ^1\.[5-9] ]] || [[ "$VERSION" =~ ^[2-9] ]]; then
    MODULE_CMD="module load python/3.12.9 sleap/$VERSION"
    VERBOSITY_FLAG="" # 1.5+ doesn't support --verbosity
    TRACK_CMD="sleap-nn-track"
fi

# Configure Node settings
TIME_LIMIT="1-00:00:00"
GRES="gpu:1"
if [[ "$NODE" == "gpu" ]]; then
    GRES="gpu:v100:1"
fi

# Scan for videos
shopt -s nullglob
VIDEOS=("$DIR"/*.avi)
NUM_VIDEOS=${#VIDEOS[@]}

if [[ "$NUM_VIDEOS" -eq 0 ]]; then
    echo "No .avi videos found in $DIR"
    exit 0
fi

echo "Found $NUM_VIDEOS videos. Submitting jobs..."

for VIDEO in "${VIDEOS[@]}"; do
    VIDEO_NAME=$(basename "$VIDEO" .avi)
    JOB_NAME="bench_${VIDEO_NAME}_${VERSION}"
    SCRIPT_PATH="$JOBS_DIR/${JOB_NAME}.sh"
    LOG_PATH="$JOBS_DIR/${JOB_NAME}_%j.log"
    OUTPUT_SLP="$WORK_DIR/${VIDEO_NAME}.${VERSION}.slp"
    RESULT_CSV="$WORK_DIR/${VIDEO_NAME}.${VERSION}.csv"

    cat <<EOF | tr -d '\r' > "$SCRIPT_PATH"
#!/bin/bash -l
#SBATCH -t $TIME_LIMIT
#SBATCH -c 8
#SBATCH --partition=$NODE
#SBATCH --mem=64G
#SBATCH --gres=$GRES
#SBATCH --exclude=saion-gpu22
#SBATCH -J $JOB_NAME
#SBATCH -o $LOG_PATH
#SBATCH -e $LOG_PATH

set -euo pipefail

$MODULE_CMD

echo "Starting inference on $VIDEO with SLEAP $VERSION"
START_TIME=\$(date +%s)
echo "Start Time: \$START_TIME"

# Run SLEAP
if [[ "$VERSION" == "sleap-nn" ]] || [[ "$VERSION" =~ ^1\.[5-9] ]] || [[ "$VERSION" =~ ^[2-9] ]]; then
    # SLEAP 1.5.2 syntax
    $TRACK_CMD \
        -i "$VIDEO" \
        -m "$MODEL_CENTROID" \
        -m "$MODEL_INSTANCE" \
        --no_empty_frames \
        -o "$OUTPUT_SLP"
else
    # SLEAP 1.4.1 syntax
    $TRACK_CMD "$VIDEO" \
        -m "$MODEL_CENTROID" \
        -m "$MODEL_INSTANCE" \
        --tracking.tracker none \
        $VERBOSITY_FLAG \
        --no-empty-frames \
        -o "$OUTPUT_SLP"
fi

END_TIME=\$(date +%s)
echo "End Time: \$END_TIME"

# Parse results using Python (pandas is available in sleap env)
python3 -c '
import pandas as pd
import re
import sys
import os

log_path = sys.argv[1]
video_name = sys.argv[2]
version = sys.argv[3]
node = sys.argv[4]
start_time = int(sys.argv[5])
end_time = int(sys.argv[6])
csv_path = sys.argv[7]
hostname = sys.argv[8]
duration = end_time - start_time
fps = None

if os.path.exists(log_path):
    with open(log_path, "r") as f:
        content = f.read()
        match = re.search(r"Inference: ([\d\.]+) FPS", content)
        if match:
            fps = float(match.group(1))

df = pd.DataFrame([{
    "Video": video_name,
    "SLEAP Version": version,
    "Node Partition": node,
    "Hostname": hostname,
    "Inference FPS": fps,
    "Duration (s)": duration,
    "Log File": log_path
}])

df.to_csv(csv_path, index=False)
print(f"Saved result to {csv_path}")
' "${JOBS_DIR}/${JOB_NAME}_\${SLURM_JOB_ID}.log" "$VIDEO_NAME" "$VERSION" "$NODE" "\$START_TIME" "\$END_TIME" "$RESULT_CSV" "\$HOSTNAME"
EOF

    # Submit job
    JOB_ID=$(sbatch --parsable "$SCRIPT_PATH")
    echo "Submitted job $JOB_ID for $VIDEO_NAME"

done

echo "All jobs submitted."
echo "Results will be saved as individual CSV files in $WORK_DIR"
echo "To combine them later, run:"
echo "  awk '(NR == 1) || (FNR > 1)' $WORK_DIR/*.${VERSION}.csv > $DIR/inference_results_${VERSION}_${NODE}.csv"
