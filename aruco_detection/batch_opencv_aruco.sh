#!/bin/bash

DATA_FOLDER="/bucket/ReiterU/Ants/basler/single_ants/test_12h_vib/20260122-002925/data"
OUTPUT_FOLDER="/flash/ReiterU/ant_tmp/20260122-002925/aruco/"
SCRIPT_PATH="$HOME/AntsArray/run_aruco.py"

mkdir -p "$OUTPUT_FOLDER"

shopt -s nullglob  # prevents literal patterns when no matches
for video_file in "$DATA_FOLDER"/*.{mp4,avi,mov}; do
    echo "Processing $video_file"
    filename=$(basename -- "$video_file")
    base_name="${filename%.*}"
 
    # completion markers (adjust suffix to match your real output name)
    marker_in_data="$DATA_FOLDER/${base_name}_aruco_tracks_.h5"
    marker_in_out="$OUTPUT_FOLDER/${base_name}_aruco_tracks_.h5"

    echo "Marker (data): $marker_in_data"
    echo "Marker (out):  $marker_in_out"

    if [ -f "$marker_in_data" ] || [ -f "$marker_in_out" ]; then
        echo "Skipping $base_name (marker exists in data or output folder)"
        continue
    fi

    sbatch_script="$OUTPUT_FOLDER/run_${base_name}.sh"
    cat <<EOF > "$sbatch_script"
#!/bin/bash -l
#SBATCH -t 0-24
#SBATCH -c 4
#SBATCH --partition=compute
#SBATCH --mem=4G
#SBATCH -J seg-$base_name
#SBATCH -o $OUTPUT_FOLDER/%x_%j.out
#SBATCH -e $OUTPUT_FOLDER/%x_%j.err

source ~/.bashrc
conda activate aruco_env

python3 "$SCRIPT_PATH" \
  --video-file "$video_file" \
  --output-path "$OUTPUT_FOLDER" \
  --dictionary-size 1000 
EOF

    sbatch "$sbatch_script"
done
