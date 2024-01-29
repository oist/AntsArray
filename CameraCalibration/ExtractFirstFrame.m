% Directory where your videos are stored
videoDir = '\\wsl.localhost\Ubuntu-22.04\home\makoto\bucketReiter\Ants\basler\cameraArray_calib\2023-12-26-23-01_AruCo_DICT_5X5_1000_glass';

% Directory where you want to save the extracted frames
outputDir = fullfile(videoDir, 'frame1');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% List of video files
videoFiles = dir(fullfile(videoDir, '*.avi'));

% Loop through each video file
for k = 1:length(videoFiles)
    videoPath = fullfile(videoDir, videoFiles(k).name);
    
    % Create a VideoReader
    vidObj = VideoReader(videoPath);
    
    % Read the first frame
    frame = readFrame(vidObj);
    
    % Construct output filename
    [~, name, ~] = fileparts(videoFiles(k).name);
    outputFilename = fullfile(outputDir, [name, '_frame1.png']);
    
    % Save the frame
    imwrite(frame, outputFilename);

    % disp the file name
    disp(outputFilename);
end
