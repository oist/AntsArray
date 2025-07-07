function Panorama_Zoom(videoDir)
% videoDir = "C:\Users\machi\Desktop\";
numWorkers = 24;

% camera folder names
nCam = 25;
cam_dir = arrayfun(@(x) sprintf('cam%02d', x), 1:nCam, 'UniformOutput', false);

% Create output directory if it doesn't exist
parent_dir = fileparts(videoDir);
outputDir = fullfile(parent_dir, 'mosaic_zoom_video');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
output_res = [2160 4*2160/3];

% zoom time series
time_control = [...
    4500,7300,3350,1000; ...
    4550,7300,3350,1480; ...
    5670,7300,3350,1480; ...
    6130,9360,3990,1480; ...
    6180,1,1,17633; ...
    8800,1,1,17633; ...
    8875,11600,935,5100; ...
    9855,11600,935,5100; ...
    10050,14100,1000,1500; ...
    10499,14100,1000,1500
    ];

ip_time = interpolateFrameData(time_control);

%% Hall
HomoParasPath = fullfile(parent_dir,"frame1", "bundle_adjustment_paras.mat");
nCam = 25;

% Load homography parameters
paras = load(HomoParasPath, "paras");
paras = paras.paras;

% Similar setup for Hall and sortedFiles as in the original code
% Recover Hall
H_pair = cell(nCam, nCam);
for i = 1:nCam
    H_pair{i, i} = eye(3);
end
for ii = 2:nCam
    currParams = paras(((4*(ii-2))+1):4*(ii-1));
    S = [currParams(1) currParams(2) currParams(3)
        currParams(2) currParams(1) currParams(4)];
    H_pair{1,ii} = [S;0 0 1];
    H_pair{ii,1} = inv(H_pair{1,ii});
end
for i = 2:nCam-1
    for j = i+1:nCam
        H_pair{i,j} = H_pair{1,j}*H_pair{i,1};
        H_pair{j,i} = inv(H_pair{i,j});
    end
end
% Set cam13 as init
Hall=H_pair(13,:);

%% Define the range of file indices
startIndex = ip_time(1,1);
% endIndex = ip_time(end,1);
if isempty(gcp('nocreate'))
    parpool(numWorkers);
end

parfor ii = 1:size(ip_time,1)
    % Read and process the specific frame
    subDir = cam_dir;
    images = cell(1, nCam);
    for i = 1:nCam
        im_path = fullfile(videoDir, subDir{i}, sprintf('frame%08d.png',startIndex+ii-1));
        fprintf('reading form %s\n', im_path);
        if exist(im_path, 'file')
            images{i} = imread(im_path);
        else
            images{i} = zeros(v.Height, v.Width, 3);  % Placeholder for missing frames
        end
    end

    % Create and save the mosaic image
    % mosaic = createMosaicFromImageArray(images, Hall);
    mosaic = createPanorama(images, Hall, 0); %no blending, just overlay

    % Cropping images
    cropping_params = ip_time(ii,:);
    im_crop = imresize(imcrop(mosaic,[cropping_params(2) cropping_params(3) cropping_params(4) cropping_params(4)/4*3]), output_res);

    imwrite(im_crop, fullfile(outputDir, sprintf('frame_%08d.png', startIndex+ii-1)), 'png');
end

function time_ctrl_array = interpolateFrameData(time_control)
% % Define the array
% time_control = [4500, 7280, 3260, 1000; ...
%                 4550, 7280, 3260, 1480; ...
%                 5670, 7280, 3260, 1480; ...
%                 6130, 9360, 3990, 1480; ...
%                 6155, 1, 1, 17633];

% Extract columns for frame number, x, y, and width
frames = time_control(:,1);
x = time_control(:,2);
y = time_control(:,3);
width = time_control(:,4);

% Define new frame range from min to max frame number
new_frames = min(frames):max(frames);

% Interpolate x, y, and width values
itp = @(xx) interp1(frames, xx, new_frames, 'linear', 'extrap');

new_x = itp(x);
new_y = itp(y);
new_width = itp(width);
time_ctrl_array = [new_frames', new_x', new_y', new_width'];
end
end