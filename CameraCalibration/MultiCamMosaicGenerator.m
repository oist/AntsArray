% Specify the directory containing the videos and homography matrice
vExt = '.avi';
videoDir = "D:\OIST Dropbox\Makoto Hiroi\makoto.hiroi@oist.jp’s files\20240415";
HomoParasPath = "D:\OIST Dropbox\Makoto Hiroi\makoto.hiroi@oist.jp’s files\20240415\frame1\bundle_adjustment_paras.mat";

videos = dir(fullfile(videoDir, ['*' vExt]));
videos([videos.isdir]|startsWith({videos.name},'.')) = [];
fileNames = {videos.name};
im_n = numel(fileNames);
fprintf('%d videos.\n', im_n)

% Read homography parameters
paras = load(HomoParasPath,"paras");
paras = paras.paras;
% Recover Hall
H_pair = cell(im_n, im_n);
for i = 1:im_n
    H_pair{i, i} = eye(3);
end
for ii = 2:im_n
    currParams = paras(((4*(ii-2))+1):4*(ii-1));
    S = [currParams(1) currParams(2) currParams(3)
        currParams(2) currParams(1) currParams(4)];
    H_pair{1,ii} = [S;0 0 1];
    H_pair{ii,1} = inv(H_pair{1,ii});
end
for i = 2:im_n-1
    for j = i+1:im_n
        H_pair{i,j} = H_pair{1,j}*H_pair{i,1};
        H_pair{j,i} = inv(H_pair{i,j});
    end
end
% Set cam13 as init
Hall=H_pair(13,:);

%% Sort the filenames based on the camera numbers extracted from the filenames
[~, idx] = sort(cellfun(@(x) sscanf(x, 'cam%*d_%*d-%*d-%*d-%*d-%*d-%*d_cam%d.avi'), fileNames));
sortedFiles = fileNames(idx);

% Create a VideoWriter object to write the mosaic video
outputVideo = VideoWriter(fullfile(videoDir, 'mosaic_video'), 'MPEG-4');
outputVideo.FrameRate = 25;  % Adjust as needed based on the input videos
outputVideo.Quality = 100;
open(outputVideo);
mkdir(fullfile(videoDir, 'mosaic_video'))

% Define the frame number to start from and the total frames to process
startFrame = 10;  % For example, N-th frame
totalFrames = 25;  % Total frames to process

tic;
% Process each frame
parfor frame = startFrame:(startFrame + totalFrames - 1)
    fprintf('frame=%d\n', frame)
    images = cell(1, im_n);  % Assuming 25 videos
    
    % Read the specific frame from each video
    for i = 1:length(sortedFiles)
        v = VideoReader(fullfile(videoDir, sortedFiles{i}));
        if hasFrame(v)
            v.CurrentTime = (frame-1) * (1/v.FrameRate);
            images{i} = readFrame(v);
        else
            images{i} = zeros(v.Height, v.Width, 3);  % Placeholder for missing frames
        end
    end
    
    % Create the mosaic from the array of images
    % Assuming Hall is defined earlier
    mosaic = createMosaicFromImageArray(images, Hall);

    % Resize the mosaic to the required output resolution
    mosaicResized = imresize(mosaic, [2160, NaN]);  % [height width]
    % Clip the mosaic values to be within the range [0, 1]
    if isa(mosaicResized, 'double')
        mosaicResized = max(0, min(1, mosaicResized));  % This will clip the values
    end

    % Write the frame to the video
    writeVideo(outputVideo, mosaicResized);
end

% Close the video file
close(outputVideo);
toc;