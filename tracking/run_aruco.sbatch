#!/bin/bash

# Define input arguments
VIDEO_FOLDER='/home/sam/bucket/Ants/trials/20241108_1/20241108_1_first_hour/data/'
SLP_FOLDER='/home/sam/bucket/sam/ant_tracking/20241108_1/slp_files/'
MODEL_PATH='/work/ReiterU/sam/antnet.pth'
OUTPUT_FOLDER='/work/ReiterU/sam/'

# Ensure output folder exists
mkdir -p "$OUTPUT_FOLDER"

# Iterate over all video files in the data folder
for VIDEO_FILE in "$VIDEO_FOLDER"/*.avi; do
    # Extract base name without extension
    BASE_NAME=$(basename "$VIDEO_FILE" .avi)

    # Define corresponding SLEAP file
    SLEAP_FILE="$SLP_FOLDER/$BASE_NAME.csv"

    # Define output HDF5 file
    OUTPUT_FILE="$OUTPUT_FOLDER/$BASE_NAME.pkl"

    # Create a temporary SLURM script for the job
    SLURM_SCRIPT=$(mktemp)

    cat <<EOF > "$SLURM_SCRIPT"
#!/bin/bash -l
#SBATCH -t 0-24
#SBATCH -c 32
#SBATCH --partition=largegpu
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=$BASE_NAME
#SBATCH --output=$OUTPUT_FOLDER/${BASE_NAME}_%j.out

# Activate the Python environment
source /opt/conda/etc/profile.d/conda.sh
conda activate torch

# Run the Python script
python /home/s/samuel-reiter/AntsArray/sam/aruco_inference.py \\
  --video_file "$VIDEO_FILE" \\
  --sleap_file "$SLEAP_FILE" \\
  --model_name "$MODEL_PATH" \\
  --output_file "$OUTPUT_FILE" \\
  --crop_size 128 \\
  --brightness_factor 1.2 \\
  --batch_size 64 \\
  --num_tags 111 \\
  --confidence_threshold 0.99 \\
  --visualize 0
EOF

    # Submit the SLURM job
    sbatch "$SLURM_SCRIPT"

    # Clean up temporary SLURM script
    rm "$SLURM_SCRIPT"
done
