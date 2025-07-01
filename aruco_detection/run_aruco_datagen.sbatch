#!/bin/bash

# Define input arguments
VIDEO_FOLDER='/bucket/ReiterU/Ants/basler/20250321_2_test/data'
OUTPUT_FOLDER='/flash/ReiterU/sam/aruco/'

# Ensure output folder exists
mkdir -p "$OUTPUT_FOLDER"

# Iterate over all video files in the data folder
for VIDEO_FILE in "$VIDEO_FOLDER"/*.avi; do
    # Extract base name without extension
    BASE_NAME=$(basename "$VIDEO_FILE" .avi)

    # Define corresponding SLEAP file
    SLEAP_FILE="$VIDEO_FOLDER/$BASE_NAME.h5"
    
    # Define corresponding aruco file
    ARUCO_FILE="$VIDEO_FOLDER/$BASE_NAME.aviaruco_tracks_.npy"
  
    # Create a temporary SLURM script for the job
    SLURM_SCRIPT=$(mktemp)

    cat <<EOF > "$SLURM_SCRIPT"
#!/bin/bash -l
#SBATCH -t 0-24
#SBATCH -c 32
#SBATCH --partition=compute
#SBATCH --mem=64G
#SBATCH --job-name=$BASE_NAME
#SBATCH --output=$OUTPUT_FOLDER/${BASE_NAME}_%j.out

# Activate the Python environment
source ~/mambaforge/etc/profile.d/conda.sh
conda activate torch

# Run the Python script
python aruco_datagen.py \\
  --video_name "$VIDEO_FILE" \\
  --sleap_file_name "$SLEAP_FILE" \\
  --aruco_name "$ARUCO_FILE" \\
  --output_path "$OUTPUT_FOLDER" \\
 
EOF

    # Submit the SLURM job
    sbatch "$SLURM_SCRIPT"

    # Clean up temporary SLURM script
    rm "$SLURM_SCRIPT"
done
