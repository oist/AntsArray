#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 15:11:39 2024

@author: sam
"""

#!/bin/bash

# Define input arguments
DATA_FOLDER='/bucket/ReiterU/Ants/trials/20241108_1/20241108_1_first_hour/data'
MODEL_PATH='/work/ReiterU/sam/antnet.pth'
OUTPUT_FOLDER='/work/ReiterU/sam/'


# Ensure output folder exists
mkdir -p "$OUTPUT_FOLDER"

# Iterate over all video files in the data folder
for VIDEO_FILE in "$DATA_FOLDER"/*.avi; do
  # Extract base name without extension
  BASE_NAME=$(basename "$VIDEO_FILE" .avi)

  # Define corresponding SLEAP file
  SLEAP_FILE="$DATA_FOLDER/$BASE_NAME.csv"

  # Define output HDF5 file
  OUTPUT_FILE="$OUTPUT_FOLDER/$BASE_NAME.h5"

  # Submit job to the cluster
  sbatch -p largegpu -c 32 --x11 --mem=64G -t 0-24 --gres=gpu:1 <<EOF
#!/bin/bash
module load python  # Load appropriate Python module
python /path/to/aruco_detection_script.py \\
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
done
