#!/usr/bin/env bash
set -euo pipefail

# Usage: ./batch_slep2csv.sh /path/to/slp_folder /path/to/output_csv_folder

INPUT_DIR="${1:-}"
OUTPUT_DIR="${2:-}"

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 <input_slp_folder> <output_h5_folder>"
  exit 1
fi

# Create output dir if needed
mkdir -p "$OUTPUT_DIR"

# Loop over every .slp file
shopt -s nullglob
for slp in "$INPUT_DIR"/*.slp; do
  echo "Processing: $(basename "$slp")"
  python sleap2h5.py "$slp" "$OUTPUT_DIR"
done
shopt -u nullglob

echo "All done! CSVs in $OUTPUT_DIR."
