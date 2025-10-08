#!/bin/bash

DATA_FOLDER="/bucket/ReiterU/Ants/basler/2025_Sep_no_pertubation/20250908_09/data"
OUTPUT_FOLDER="/flash/ReiterU/ant_tmp/20250908_09/aruco"
SCRIPT_PATH="$HOME/AntsArray/aruco_detection/opencv_aruco.py"

mkdir -p "$OUTPUT_FOLDER"

shopt -s nullglob  # prevents literal patterns when no matches
for video_file in "$DATA_FOLDER"/*.{mp4,avi,mov}; do
    filename=$(basename -- "$video_file")
    base_name="${filename%.*}"
 
    # define the file that signals completion
    marker_file="$OUTPUT_FOLDER/${base_name}_aruco_detections.csv"   # <-- adjust to your script’s actual output
    echo $marker_file
    if [ -f "$marker_file" ]; then
        echo "Skipping $base_name (output $marker_file exists)"
        continue
    fi

    sbatch_script="$OUTPUT_FOLDER/run_${base_name}.sh"
    cat <<EOF > "$sbatch_script"
#!/bin/bash -l
#SBATCH -t 0-24
#SBATCH -c 16
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH -J seg-$base_name
#SBATCH -o $OUTPUT_FOLDER/%x_%j.out
#SBATCH -e $OUTPUT_FOLDER/%x_%j.err

source ~/.bashrc
conda activate torch

python3 "$SCRIPT_PATH" \
  --video-file "$video_file" \
  --output-path "$OUTPUT_FOLDER" \
  --dictionary-size 1000 \
  --max-gap 100 \
  --min-fraction 0.125
EOF

    sbatch "$sbatch_script"
done
