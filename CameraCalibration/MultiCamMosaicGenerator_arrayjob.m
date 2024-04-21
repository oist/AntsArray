function MultiCamMosaicGenerator_arrayjob(videoDir)

% Adjust paths for the cluster environment
% videoDir = "/flash/ReiterU/makoto/20240415/";
HomoParasPath = fullfile("/flash/ReiterU/makoto/20240415/", "frame1/bundle_adjustment_paras.mat");

% Load videos and homography parameters as before
iExt = '.png';
videos = dir(fullfile(videoDir, ['*' iExt]));
fileNames = {videos(~[videos.isdir] & ~startsWith({videos.name}, '.')).name};
im_n = numel(fileNames);

paras = load(HomoParasPath, "paras");
paras = paras.paras;

% Similar setup for Hall and sortedFiles as in the original code
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

% Sort the filenames based on the camera numbers extracted from the filenames
[~, idx] = sort(cellfun(@(x) sscanf(x, 'cam%d_fr%*d.png'), fileNames));
sortedFiles = fileNames(idx);

% Create output directory if it doesn't exist
[baseDir,~,~] = fileparts(videoDir);
outputDir = fullfile(baseDir, 'mosaic_video');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Read and process the specific frame
frame = sscanf(fileNames{1}, 'cam%*d_fr%d.png');
fprintf('Processing frame=%d\n', frame);
images = cell(1, im_n);
for i = 1:length(sortedFiles)
    im_path = fullfile(videoDir, sortedFiles{i});
    fprintf('reading form %s\n', im_path);
    if exist(im_path, 'file')
        images{i} = imread(im_path);
    else
        images{i} = zeros(v.Height, v.Width, 3);  % Placeholder for missing frames
    end
end

% Create and save the mosaic image
mosaic = createMosaicFromImageArray(images, Hall);
imwrite(mosaic, fullfile(outputDir, sprintf('frame_%08d.png', frame)), 'png');
