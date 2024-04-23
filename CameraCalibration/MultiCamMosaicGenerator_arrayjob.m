function MultiCamMosaicGenerator_arrayjob(videoDir,startframe,frame)

% Adjust paths for the cluster environment
% videoDir = "/flash/ReiterU/makoto/20240415/";
HomoParasPath = fullfile("/flash/ReiterU/makoto/20240415/", "frame1/bundle_adjustment_paras.mat");
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

% camera folder names
cam_dir = arrayfun(@(x) sprintf('cam%02d', x), 1:nCam, 'UniformOutput', false);

% Create output directory if it doesn't exist
parent_dir = fileparts(videoDir);
outputDir = fullfile(parent_dir, 'mosaic_video');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Read and process the specific frame
images = cell(1, nCam);
for i = 1:nCam
    im_path = fullfile(videoDir, cam_dir{i}, sprintf('frame%08d.png',startframe+frame-1));
    fprintf('reading form %s\n', im_path);
    if exist(im_path, 'file')
        images{i} = imread(im_path);
    else
        images{i} = zeros(v.Height, v.Width, 3);  % Placeholder for missing frames
    end
end

% Create and save the mosaic image
mosaic = createMosaicFromImageArray(images, Hall);
imwrite(mosaic, fullfile(outputDir, sprintf('frame_%08d.png', startframe+frame-1)), 'png');
