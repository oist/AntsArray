#!/bin/bash
#SBATCH --job-name=mosaic_creation
#SBATCH --partition=short
#SBATCH --output=./output/jobs/mosaic_%A_%a.out
#SBATCH --error=./output/jobs/mosaic_%A_%a.err
#SBATCH --array=1-300  # Launch 100 jobs with up to 20 running in parallel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1     # One core per task
#SBATCH --mem-per-cpu=30G

# Define base directory for videos
video_dir="/flash/ReiterU/makoto/20240415"

# Define start frame offset
startframe=0
frame=$((SLURM_ARRAY_TASK_ID + startframe))

# Define directory for storing extracted frames
# Format the frame number to have 8 digits, padded with zeros
formatted_frame=$(printf "%08d" $frame)
output_dir="${video_dir}/frame_${formatted_frame}"
mkdir -p ${output_dir}

# Create an array of video files sorted by camera number
mapfile -t sorted_videos < <(find $video_dir -name 'cam*.avi' | sort -t '_' -k 3)

# Loop through sorted video array to extract the corresponding frame
for video_path in "${sorted_videos[@]}"; do
  # Extract camera number
  cam_number=$(echo $video_path | grep -oP 'cam\d+.avi$' | grep -oP '\d+')
  ffmpeg -y -i ${video_path} -vf "select=eq(n\,${frame})" -vframes 1 ${output_dir}/cam${cam_number}_fr${formatted_frame}.png
done

module load matlab  # Load MATLAB module, adjust as needed
matlab -nosplash -nodisplay -nojvm -nodesktop -r "addpath('/home/m/makoto-hiroi/ants_mosaic/'); MultiCamMosaicGenerator_arrayjob('${output_dir}'); exit;"

rm -r ${output_dir}