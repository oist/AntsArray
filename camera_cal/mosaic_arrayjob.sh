#!/bin/bash

# Initialize variables
video_dir=""
start_frame=""
total_frames=""

# Parse command line options
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --dir)
        video_dir="$2"
        shift # past argument
        shift # past value
        ;;
        --st)
        start_frame="$2"
        shift # past argument
        shift # past value
        ;;
        --to)
        total_frames="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        echo "Unknown option: $1"
        echo "Usage: $0 --dir <video_dir> --st <start_frame> --to <total_frames>"
        exit 1
        ;;
    esac
done

# Check if all parameters are set
if [ -z "$video_dir" ] || [ -z "$start_frame" ] || [ -z "$total_frames" ]; then
    echo "All parameters are required."
    echo "Usage: $0 --dir <video_dir> --st <start_frame> --to <total_frames>"
    exit 1
fi

frame_interval=1  # Adjust the frame interval if needed

# Generate a unique directory name using the date and time
unique_dir=$(date +%Y%m%d_%H%M%S)

# Remove trailing slash if it exists
[[ "${video_dir}" == */ ]] && output_dir="${video_dir%/}"

# Create a new working directory in the video directory
output_dir="${video_dir}/${unique_dir}"
mkdir -p $output_dir
mkdir -p $output_dir/jobs

# File names for the sbatch scripts
extract_frames_sbatch="extract_frames.sbatch"
mosaic_generator_sbatch="mosaic_generator.sbatch"

# Create the first sbatch file for frame extraction
cat > ${output_dir}/$extract_frames_sbatch << EOF
#!/bin/bash
#SBATCH --job-name=extract_frames
#SBATCH --partition=compute
#SBATCH --output=$output_dir/jobs/%A_%a.out
#SBATCH --error=$output_dir/jobs/%A_%a.err
#SBATCH --array=1-25
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
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
#SBATCH --array=1-${total_frames}%65
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
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
