#!/bin/bash

# Define the base directory for videos and output
video_dir="/flash/ReiterU/makoto/20240415"
output_dir=${video_dir}"/tmp"
mkdir -p $output_dir
mkdir -p $output_dir/jobs

# Frame extraction parameters
start_frame=1289
total_frames=220
frame_interval=1  # Adjust the frame interval if needed

# File names for the sbatch scripts
extract_frames_sbatch="extract_frames.sbatch"
mosaic_generator_sbatch="mosaic_generator.sbatch"

# Create the first sbatch file for frame extraction
cat > ${output_dir}/$extract_frames_sbatch << EOF
#!/bin/bash
#SBATCH --job-name=extract_frames
#SBATCH --partition=short
#SBATCH --output=$output_dir/jobs/%A_%a.out
#SBATCH --error=$output_dir/jobs/%A_%a.err
#SBATCH --array=1-25
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cam_number=\$(printf "cam%02d" \${SLURM_ARRAY_TASK_ID})
mkdir $output_dir/\${cam_number}

# Create an array of video files sorted by camera number
# Assuming filenames are like cam01_2024-04-15-09-12-43.avi or similar
mapfile -t sorted_videos < <(find $video_dir -name 'cam*.avi' | sort -t '_' -k 3)

# Use SLURM_ARRAY_TASK_ID to fetch the appropriate video file
video_index=\$((SLURM_ARRAY_TASK_ID - 1))  # Adjust index since array indices start at 0
video_file="\${sorted_videos[\$video_index]}"

# Extract frames from each video in an array job
ffmpeg -y -i "\$video_file" -vf "select='gte(n,$start_frame)'" -vframes $total_frames -vsync vfr -start_number $start_frame $output_dir/\${cam_number}/frame%08d.png
EOF

# Create the second sbatch file for running the mosaic generator
cat > ${output_dir}/$mosaic_generator_sbatch << EOF
#!/bin/bash
#SBATCH --job-name=mosaic_generator
#SBATCH --partition=short
#SBATCH --output=$output_dir/jobs/mosaic_%A_%a.out
#SBATCH --error=$output_dir/jobs/mosaic_%A_%a.err
#SBATCH --array=1-$total_frames%100
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Run the mosaic generator
module load matlab
matlab -nosplash -nodisplay -nojvm -nodesktop -r "addpath('/home/m/makoto-hiroi/ants_mosaic/'); MultiCamMosaicGenerator_arrayjob('$output_dir',$start_frame,\${SLURM_ARRAY_TASK_ID}); exit;"
EOF

# Create the cleanup sbatch file to delete the output directory
cat > ${output_dir}/cleanup.sbatch << EOF
#!/bin/bash
#SBATCH --job-name=cleanup_job
#SBATCH --partition=short
#SBATCH --output=$output_dir/jobs/cleanup_%j.out
#SBATCH --error=$output_dir/jobs/cleanup_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# Command to remove the output directory
rm -rf $output_dir
EOF

# Submit the first job and capture its job ID
job_id1=$(sbatch ${output_dir}/$extract_frames_sbatch | cut -d ' ' -f 4)

# Submit the second job with dependency
job_id2=$(sbatch --dependency=afterok:$job_id1 ${output_dir}/$mosaic_generator_sbatch | cut -d ' ' -f 4)

# Submit the cleanup job with dependency on the second job
sbatch --dependency=afterok:$job_id2 ${output_dir}/cleanup.sbatch
