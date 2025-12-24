#!/bin/bash
# combine_results.sh
# Usage: ./combine_results.sh [directory]
# Combines all *.csv files in the specified directory (or current directory) into combined_inference_results.csv

set -euo pipefail

DIR="${1:-.}"

if [[ ! -d "$DIR" ]]; then
    echo "Error: Directory '$DIR' does not exist."
    exit 1
fi

OUTPUT_FILE="${DIR}/combined_inference_results.csv"

# Enable nullglob to handle no matches gracefully
shopt -s nullglob
CSVS=("$DIR"/*.csv)

# Filter out the output file itself if it exists in the list to avoid infinite loops or duplication
INPUT_CSVS=()
for csv in "${CSVS[@]}"; do
    if [[ "$(realpath "$csv")" != "$(realpath "$OUTPUT_FILE")" ]]; then
        INPUT_CSVS+=("$csv")
    fi
done

if [[ ${#INPUT_CSVS[@]} -eq 0 ]]; then
    echo "No CSV files found in $DIR"
    exit 0
fi

echo "Combining ${#INPUT_CSVS[@]} CSV files in $DIR..."

# Combine files:
# 1. Print the header from the first file (NR==1)
# 2. For all files (including first), print lines that are NOT the header (FNR > 1)
# Actually, simpler logic:
# Print line if it's the first line of the first file (NR==1)
# OR if it's NOT the first line of the current file (FNR > 1)

awk '(NR == 1) || (FNR > 1)' "${INPUT_CSVS[@]}" > "$OUTPUT_FILE"

echo "Combined results saved to $OUTPUT_FILE"
