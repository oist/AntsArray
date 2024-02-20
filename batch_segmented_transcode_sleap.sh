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

# Check if the directory path is not empty
if [[ -z "$DIR" ]]; then
    echo "Error: No directory provided."
    usage
fi

# Your script logic here
echo "Directory: $DIR"
echo "Node: $SAION_NODE"

# Email address for job notifications
emailurl=makoto.hiroi@oist.jp

# Create the output folder and set the environment variables
base_folder=$(basename "$DIR")
data_folder=$DIR/data
output_folder=/flash/ReiterU/ant_tmp/${base_folder}
deigo_folder=/deigo_flash/ReiterU/ant_tmp/${base_folder}
sleap_model1=/bucket/ReiterU/Ants/SLEAP_files/topdown/IR/231223_113827.centroid.n=82/training_config.json
sleap_model2=/bucket/ReiterU/Ants/SLEAP_files/topdown/IR/231223_142806.centered_instance.n=82/training_config.json

mkdir -p $data_folder
mkdir -p $output_folder
export output_folder
export deigo_folder

# Loop through each .avi file in the directory
for video_file in ${DIR}*.avi
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

# Calculate segment time in seconds for 2 hours
SEG_TIME_SEC=\$((60*60*2))
echo "SEG_TIME_SEC: \${SEG_TIME_SEC}"

# Calculate total segments based on video duration in frames and segment duration in frames
SEG_FRAMES=\$((\$SEG_TIME_SEC * \$FPS))
echo "SEG_FRAMES: \${SEG_FRAMES}"
SEG_FRAMES_STRING=""

# Assuming we need a segment every 2 hours, calculate how many segments we have
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
ffmpeg -fflags +genpts -y -i ${video_file} -threads 32 -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 -f segment -reset_timestamps 1 -segment_list ${output_folder}/${video_name}_frame_counts.csv -segment_frames \${SEG_FRAMES_STRING} -break_non_keyframes 1 -force_key_frames "expr:gte(t,n_forced*\${SEG_TIME_SEC})" ${output_folder}/${video_name}_%03d.avi

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

    # Create a chain job submission script for initiating a saion job
    cat > "${output_folder}/job2-$video_name.sh" <<EOF
#!/bin/bash -l
#SBATCH -t 0-1
#SBATCH -c 1
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --job-name=init_saionjob-${video_name}
#SBATCH --output=./output/jobs/%x_%j.out
#SBATCH --error=./output/jobs/%x_%j.err

# Submit the job to the Saion system
ssh saion sbatch "${deigo_folder}/job3-$video_name.sh"

# This wrapper script will count the number of files you want to process, apply a maximum cap if necessary, and then submit the job to SLURM with the appropriate --array parameter.
# Path to the text file containing the file names
FILE_LIST=${output_folder}/${video_name}_frame_counts.csv

# Count the number of non-empty lines in the file
NUM_FILES=\$(grep -cve '^\s*\$' "\${FILE_LIST}")

# Define the maximum cap for the array jobs
MAX_CAP=100

# Create a chain job script (job5) on the Deigo system (run_aruco.py)
cat > "${output_folder}/job5-$video_name.sh" <<EOJ
#!/bin/bash -l
#SBATCH -t 0-24
#SBATCH -c 32
#SBATCH --partition=compute
#SBATCH --mem=0
#SBATCH --job-name=aruco-${video_name}
#SBATCH --array=1-10
#SBATCH --output=./output/jobs/%x_%A_%a.out
#SBATCH --error=./output/jobs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$emailurl

# Load the required modules
ml use /apps/unit/ReiterU/.modulefiles
ml load opencv/4.9.0

# Read the first column from the CSV into an array
mapfile -t video_files < <(cut -d',' -f1 "${output_folder}/${video_name}_frame_counts.csv")

# Calculate the array index
index=\\\$((\\\${SLURM_ARRAY_TASK_ID} - 1))
echo "SLURM_ARRAY_TASK_ID: \\\${SLURM_ARRAY_TASK_ID}"

# Execute the Python script for the current video file
python /apps/unit/ReiterU/ant_tracking/run_aruco.py --video-file ${output_folder}/\\\${video_files[\\\$index]} --output-path ${output_folder}/

# Transfer the output file to the Saion system
scp $output_folder/\\\${video_files[\\\$index]}aruco_tracks_.npy saion:/work/ReiterU/ant_tmp/${base_folder}/
rm $output_folder/\\\${video_files[\\\$index]}aruco_tracks_.npy

echo "Processed: \\\${video_files[\\\$index]}"
EOJ

# Submit job5 to deigo with the calculated array size
if [ "\$NUM_FILES" -gt 1 ]; then
  sbatch --array=1-\$NUM_FILES%\$MAX_CAP ${output_folder}/job5-$video_name.sh
else
  sbatch --array=1 ${output_folder}/job5-$video_name.sh
fi
EOF

    # Create the follow-up job script on the Saion system
    cat > "${output_folder}/job3-$video_name.sh" <<EOF
#!/bin/bash -l
#SBATCH -t 0-1
#SBATCH -c 1
#SBATCH --partition=test-gpu
#SBATCH --mem=32G
#SBATCH --job-name=list_submit-${video_name}
#SBATCH --output=./output/jobs/%x_%j.out
#SBATCH --error=./output/jobs/%x_%j.err

# Directory where the segmented files are stored
SEGMENT_DIR=${deigo_folder}
# Base name for the segmented files
BASE_NAME=${video_name}

mkdir -p /work/ReiterU/ant_tmp/${base_folder}

# List and iterate over each line in the frame counts file
while IFS=, read -r line; do
    # Extract the base name and the number of frames
    IFS=',' read -ra ADDR <<< "\${line}"
    segmented_file_base=\${ADDR[0]%.avi}
    frames=\${ADDR[3]}
    echo "segmented_file_base: \${segmented_file_base}"
    echo "BASE_NAME: \${BASE_NAME}"
    echo "frames: \${frames}"

    # Check if the segmented file base name matches the pattern ***_NNN
    if [[ \${segmented_file_base} =~ ^\${BASE_NAME}_[0-9]{3}\$ ]]; then

		# Dynamically create a job4 script for the current segmented file
		cat > "/work/ReiterU/ant_tmp/${base_folder}/job4-\${segmented_file_base}.sh" <<EOJ
#!/bin/bash -l
#SBATCH -t 0-48
#SBATCH -c 8
#SBATCH --partition=$SAION_NODE
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --job-name=process-\${segmented_file_base}
#SBATCH --output=./output/jobs/%x_%j.out
#SBATCH --error=./output/jobs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$emailurl

ml use /apps/unit/ReiterU/.modulefiles
ml load sleap

echo "Processing \${segmented_file_base} with \${frames} frames."

# Now we have the total_frames variable available for use
total_frames=\${frames}

sleap-track ${deigo_folder}/\${segmented_file_base}.avi --batch_size 1 --frames 0-\$((\${frames} - 1)) \
-m ${sleap_model1} \
-m ${sleap_model2} \
--tracking.tracker none \
-o /work/ReiterU/ant_tmp/${base_folder}/\${segmented_file_base}.slp --verbosity json --no-empty-frames

# Load Matlab module
module --ignore_cache load matlab

# Run a small script that sets our variables then calls the real script
matlab -nosplash -nodisplay -nojvm -nodesktop -r "addpath('/apps/unit/ReiterU/makoto/mfiles/'); slp2csv('/work/ReiterU/ant_tmp/${base_folder}/\${segmented_file_base}.slp'); exit;"

# Delete video files on flash
# ssh deigo rm -rf ${output_folder}/\${segmented_file_base}.avi
EOJ

		# Submit the job4 script
		sbatch "/work/ReiterU/ant_tmp/${base_folder}/job4-\${segmented_file_base}.sh"
    fi
done < "${deigo_folder}/${video_name}_frame_counts.csv"
EOF

    # Submit the first job and get its job ID
    job1_path="${output_folder}/job1-$video_name.sh"
    jobstring=$(sbatch "${job1_path}")
    jobid=${jobstring##* }

    # Submit the second job with a dependency on the first job
    job2_path="${output_folder}/job2-$video_name.sh"
    sbatch --dependency=afterok:$jobid "${job2_path}"
  fi
done

# Create a folder monitoring job
cat > "${data_folder}/monitor-$base_folder.sh" <<EOF
#!/bin/bash -l
#SBATCH -t 0-48
#SBATCH -c 1
#SBATCH --partition=compute
#SBATCH --mem=8G
#SBATCH --job-name=monitor-${base_folder}
#SBATCH --output=./output/jobs/%x_%j.out
#SBATCH --error=./output/jobs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$emailurl

monitor_dir=/saion_work/ReiterU/ant_tmp/$base_folder/ # Directory to monitor
TARGET_DIR=$data_folder # Directory to move files to
INTERVAL=30 # How often to check the directory (in seconds)

while true; do
	# Loop through each .csv file in the directory
	for csv_file in ${monitor_dir}/*.csv; do
		# Skip if no .csv files are found
		[ -e "\$csv_file" ] || continue

		# Extract the filename without the extension
		filename=\$(basename "\$csv_file" .csv)

		# Define the companion file
		companion_file="${monitor_dir}/\${filename}.aviaruco_tracks_.npy"

		# Check if both files exist
		if [ -f "\$csv_file" ] && [ -f "\$companion_file" ]; then
			echo "Both files for \$filename found."

			# Move all files starting with \${filename} to the target directory
			ssh saion mv "/work/ReiterU/ant_tmp/${base_folder}/\${filename}"* "$TARGET_DIR"/
			echo "Moved files starting with \${filename} to $TARGET_DIR on saion"
			
			mv "${output_folder}/\${filename}"* "$TARGET_DIR"/
			echo "Moved files starting with \${filename} to $TARGET_DIR on deigo"
		done
	sleep $INTERVAL
done
EOF

# Submit the monitoring job
sbatch ${data_folder}/monitor-$base_folder.sh