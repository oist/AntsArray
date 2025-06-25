#!/bin/bash -l

# Initialize variables
DIR=""
SAION_NODE="gpu" # Default value

# Function to display usage
usage() {
    echo "Usage: $0 --dir <directory> [--node <node>]"
    echo "  --dir  Specify the directory path containing .avi files."
    echo "  --node Specify the node name (optional). Default is 'gpu'."
    exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dir) DIR="$2"; shift ;; # Shift twice to skip argument
        --node) SAION_NODE="$2"; shift ;;
        *) usage ;; # Display usage for unrecognized options
    esac
    shift # Move to next argument
done

# Remove trailing slash from DIR if it exists
DIR="${DIR%/}"

# Check if the directory path is not empty
if [[ -z "$DIR" ]]; then
    echo "Error: No directory provided."
    usage
fi

echo "Directory: $DIR"
echo "Node: $SAION_NODE"

# Specify the gpu type 
if [ "$SAION_NODE" = "gpu" ]; then
    gputype="gpu:V100:1"
else
    gputype="gpu:1"
fi

# Email address for job notifications
emailurl=makoto.hiroi@oist.jp

# Create the output folder and set the environment variables
base_folder=$(basename "$DIR")
data_folder=$DIR/data
output_folder=/flash/ReiterU/ant_tmp/${base_folder}
deigo_folder=/deigo_flash/ReiterU/ant_tmp/${base_folder}
sleap_model1=/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centroid/training_config.json
sleap_model2=/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton/20250408_models_LATESTWORKINGMODEL/250408_141245.centered_instance/training_config.json

mkdir -p ~/output/jobs
mkdir -p $data_folder
chmod 2775 $data_folder
mkdir -p $output_folder
chmod 2775 $output_folder
export output_folder
export deigo_folder

# Loop through each .avi file in the directory
for video_file in ${DIR}/*.avi
do
  # Check if the file name ends with '_renc.avi' or '_nvenc.avi'
  if [[ ! $video_file =~ ^\. ]] && [[ ! $video_file =~ _renc\.avi$ ]] && [[ ! $video_file =~ _nvenc\.avi$ ]]; then
  
    # Extract the base name of the file for job naming
    video_name=$(basename "$video_file" .avi)
  
    # Create a job submission script for the current file
    cat > "${output_folder}/job1-$video_name.sh" <<EOF
#!/bin/bash -l
#SBATCH -t 0-72
#SBATCH -c 32
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH --job-name=transcode-${video_name}
#SBATCH --output=./output/jobs/%x_%j.out
#SBATCH --error=./output/jobs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$emailurl

# Use exiftool to get the frame count and fps, filtering the output
FRAME_COUNT=\$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 ${video_file})
FPS=\$(exiftool "${video_file}" | grep 'Video Frame Rate' | awk -F': ' '{print \$2}' | tr -d ' ')
echo "FPS: \${FPS}"
echo "FRAME_COUNT: \${FRAME_COUNT}"

# Calculate segment time in seconds for 20 minutes
SEG_TIME_SEC=\$((60*20))
echo "SEG_TIME_SEC: \${SEG_TIME_SEC}"

# Calculate total segments based on video duration in frames and segment duration in frames
SEG_FRAMES=\$((\$SEG_TIME_SEC * \$FPS))
echo "SEG_FRAMES: \${SEG_FRAMES}"
SEG_FRAMES_STRING=""

# Assuming we need a segment every 20 minutes, calculate how many segments we have
# We'll divide the total frame count by SEG_FRAMES to determine the number of segments
TOTAL_SEGMENTS=\$((\$FRAME_COUNT / \$SEG_FRAMES))
echo "TOTAL_SEGMENTS: \${TOTAL_SEGMENTS}"

# Check if TOTAL_SEGMENTS is 0 and assign SEG_FRAMES_STRING to SEG_FRAMES if true
if [ "\$TOTAL_SEGMENTS" -eq 0 ]; then
    SEG_FRAMES_STRING=\$SEG_FRAMES
else
	# Generate SEG_FRAMES_STRING
	for ((i = 1; i <= \${TOTAL_SEGMENTS}; i++)); do
	  SEG_POINT=\$((\$i * \$SEG_FRAMES))
	  if [ "\$i" -gt 1 ]; then
		SEG_FRAMES_STRING+=","
	  fi
	  SEG_FRAMES_STRING+="\${SEG_POINT}"
	done
fi

echo "Segmenting at frames: \${SEG_FRAMES_STRING}"

if [[ -f "${output_folder}/${video_name}_frame_counts.csv" ]]; then
  rm "${output_folder}/${video_name}_frame_counts.csv"
fi

# Transcode the video to segmented frames with ffmpeg
ffmpeg -fflags +genpts -y -i ${video_file} -threads 32 -c:v libx264 -pix_fmt yuv420p -preset slow -crf 23 -f segment -reset_timestamps 1 -segment_list ${output_folder}/${video_name}_frame_counts.csv -segment_frames \${SEG_FRAMES_STRING} -break_non_keyframes 1 -force_key_frames "expr:gte(t,n_forced*\${SEG_TIME_SEC})" ${output_folder}/${video_name}_%03d.avi

# Section to calculate and save frame counts for each segment
csv_file="${output_folder}/${video_name}_frame_counts.csv"
temp_file="${output_folder}/${video_name}_frame_counts_temp.csv"

# Read the CSV line by line
while IFS=, read -r name start_in_sec end_in_sec
do
    # Use ffprobe to get the frame count for the segment
    nb_frames=\$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "${output_folder}/\$name")
    
    # Append the nb_frames to the line
    echo "\$name,\$start_in_sec,\$end_in_sec,\$nb_frames" >> "\$temp_file"
done < "\$csv_file"

# Overwrite the original CSV with the temporary file
mv "\$temp_file" "\$csv_file"
EOF