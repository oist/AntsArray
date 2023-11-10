# Here's a Python script that generates the ffmpeg command to tile 25 videos into a 5x5 grid.
# The script will include the necessary [v0][v1]...[v24] labels for the xstack filter.

def generate_ffmpeg_command(video_directory, video_filenames, output_filename, frames):
    base_command = "ffmpeg"
    input_str = ""
    filter_complex_str = ""
    for i, filename in enumerate(video_filenames):
        # Add the input file to the command
        input_str += f' -i "{video_directory}\\{filename}"'
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
    final_str = f"-frames:v {frames} -map \"[out]\" \"{output_filename}\""

    # Combine all parts into the final command
    complete_command = f"{base_command}{input_str} -filter_complex \"{filter_complex_str}\" {final_str}"
    return complete_command

# Assuming the video files are named sequentially from cam0...cam24.avi
video_directory = "C:\\Users\\machi\\Desktop\\numbering"
video_filenames = [
    'cam0_2023-11-10-01-42-10_cam1.avi',
    'cam1_2023-11-10-01-42-11_cam2.avi',
    'cam2_2023-11-10-01-42-12_cam3.avi',
    'cam3_2023-11-10-01-42-11_cam4.avi',
    'cam4_2023-11-10-01-42-12_cam5.avi',
    'cam5_2023-11-10-01-42-12_cam6.avi',
    'cam6_2023-11-10-01-42-12_cam7.avi',
    'cam7_2023-11-10-01-42-13_cam8.avi',
    'cam8_2023-11-10-01-42-12_cam9.avi',
    'cam0_2023-11-10-01-41-53_cam10.avi',
    'cam1_2023-11-10-01-41-53_cam11.avi',
    'cam2_2023-11-10-01-41-53_cam12.avi',
    'cam3_2023-11-10-01-41-54_cam13.avi',
    'cam4_2023-11-10-01-41-54_cam14.avi',
    'cam5_2023-11-10-01-41-54_cam15.avi',
    'cam6_2023-11-10-01-41-54_cam16.avi',
    'cam7_2023-11-10-01-41-54_cam17.avi',
    'cam0_2023-11-10-01-41-37_cam18.avi',
    'cam1_2023-11-10-01-41-38_cam19.avi',
    'cam2_2023-11-10-01-41-37_cam20.avi',
    'cam3_2023-11-10-01-41-39_cam21.avi',
    'cam4_2023-11-10-01-41-39_cam22.avi',
    'cam5_2023-11-10-01-41-39_cam23.avi',
    'cam6_2023-11-10-01-41-39_cam24.avi',
    'cam7_2023-11-10-01-41-39_cam25.avi'
]
output_filename = "output.mp4"
frames = 250  # If you want to process more frames, change this number accordingly

# Generate the ffmpeg command
ffmpeg_command = generate_ffmpeg_command(video_directory, video_filenames, output_filename, frames)
print(ffmpeg_command)
