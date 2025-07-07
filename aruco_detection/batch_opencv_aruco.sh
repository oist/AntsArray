#!/bin/bash

# Define directories
DATA_FOLDER="/bucket/ReiterU/Ants/basler/20250321_2_test/data"       
OUTPUT_FOLDER="/flash/ReiterU/ant_tmp"   
SCRIPT_PATH="$HOME/AntsArray/aruco_detection/opencv_aruco.py"   

mkdir -p "$OUTPUT_FOLDER"


for video_file in "$DATA_FOLDER"/*.{mp4,avi,mov}; do
    [ -e "$video_file" ] || continue

    filename=$(basename -- "$video_file")
    base_name="${filename%.*}"

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


source ~/mambaforge/etc/profile.d/conda.sh
conda activate torch

python3 "$SCRIPT_PATH" \
  --video-file "$video_file" \
  --output-path "$OUTPUT_FOLDER/${base_name}" \
  --dictionary-size 1000 \
  --max-gap 100 \
  --min-fraction 0.125 \

EOF

        sbatch "$sbatch_script"

done
