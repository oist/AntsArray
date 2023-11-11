import re
import csv

def read_video_filenames_from_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        return [row[0] for row in reader]

def read_video_filenames_from_txt(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

# Choose the appropriate function based on your file type
# For CSV file
# video_filenames = read_video_filenames_from_csv("path_to_your_csv_file.csv")

# For Text file
# video_filenames = read_video_filenames_from_txt("path_to_your_text_file.txt")

def extract_number(file_name):
    # Extract the number after '_cam' and before '.avi'
    match = re.search(r'_cam(\d+).avi', file_name)
    if match:
        return int(match.group(1))
    return 0  # Default if no number is found

def generate_ffmpeg_command(video_directory, video_filenames, output_filename, frames):
    base_command = "ffmpeg"
    input_str = ""
    filter_complex_str = ""
    for i, filename in enumerate(video_filenames):
        # Add the input file to the command
        input_str += f' -i {video_directory}{filename}'
        # Prepare the filter_complex part
        filter_complex_str += f"[{i}:v]scale=804x606[v{i}]; "

    # Define the layout for the xstack filter
    xstack_layout = ("0_0|w0_0|w0+w1_0|w0+w1+w2_0|w0+w1+w2+w3_0"
                     "|0_h0|w0_h0|w0+w1_h0|w0+w1+w2_h0|w0+w1+w2+w3_h0"
                     "|0_h0+h1|w0_h0+h1|w0+w1_h0+h1|w0+w1+w2_h0+h1|w0+w1+w2+w3_h0+h1"
                     "|0_h0+h1+h2|w0_h0+h1+h2|w0+w1_h0+h1+h2|w0+w1+w2_h0+h1+h2|w0+w1+w2+w3_h0+h1+h2"
                     "|0_h0+h1+h2+h3|w0_h0+h1+h2+h3|w0+w1_h0+h1+h2+h3|w0+w1+w2_h0+h1+h2+h3|w0+w1+w2+w3_h0+h1+h2+h3[out]")

    # Add all video labels to the xstack filter
    xstack_input_str = "".join(f"[v{i}]" for i in range(len(video_filenames)))

    # Assemble the complete filter_complex string
    filter_complex_str += f"{xstack_input_str}xstack=inputs={len(video_filenames)}:layout={xstack_layout} "

    # The final part of the command with the mapping and output file
    final_str = f"-frames:v {frames} -c:v h264_nvenc -b:v 20M -map \"[out]\" \"{output_filename}\""

    # Combine all parts into the final command
    complete_command = f"{base_command}{input_str} -filter_complex \"{filter_complex_str}\" {final_str}"
    return complete_command

# List of video file names
video_filenames = read_video_filenames_from_txt(r"C:\Users\makoto\Desktop\video_files.txt")

# Sort the file names using the extracted number
sorted_video_filenames = sorted(video_filenames, key=extract_number)

# Assuming the video files are in the specified directory
video_directory = ""
output_filename = ".\\Desktop\\output.mp4"
frames = 750  # If you want to process more frames, change this number accordingly

# Generate the ffmpeg command
ffmpeg_command = generate_ffmpeg_command(video_directory, sorted_video_filenames, output_filename, frames)
print(ffmpeg_command)
