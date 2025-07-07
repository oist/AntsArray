function panorama = createPanorama(images,matrices,enableBlending)
% Load images and transformation matrices
% Replace these with your actual image loading and matrix setup
% for i = 1:numImages
%     images{i} = imread(sprintf('image%d.jpg', i));  % Example filenames
%     matrices{i} = eye(3);  % Replace with actual transformation matrices
% end
numImages = numel(images);
numChannels = 1;%size(images{1},3);

% Compute the output limits for each image to find the overall panorama size
xLimits = [inf -inf];
yLimits = [inf -inf];

for i = 1:numImages
    tform = affine2d(inv(matrices{i})');
    [xlim, ylim] = outputLimits(tform, [1 size(images{i}, 2)], [1 size(images{i}, 1)]);
    xLimits = [min(xLimits(1), xlim(1)) max(xLimits(2), xlim(2))];
    yLimits = [min(yLimits(1), ylim(1)) max(yLimits(2), ylim(2))];
end

% Define the size of the panorama
width = round(xLimits(2) - xLimits(1));
height = round(yLimits(2) - yLimits(1));
panorama = zeros(height, width, numChannels, 'like', images{1});
panoramaRef = imref2d(size(panorama), xLimits, yLimits);

% Warp each image into the panorama
for i = 1:numImages
    tform = affine2d(inv(matrices{i})');
    warpedImage = imwarp(images{i}, tform, 'OutputView', panoramaRef);

    % Create a mask for the current image
    mask = imwarp(true(size(images{i},1), size(images{i},2)), tform, 'OutputView', panoramaRef);

    if enableBlending
        % Blend using linear weighting
        tmp = panorama + warpedImage;
        panorama = bsxfun(@times, tmp, cast(mask,class(tmp)));
    else
        % No blending, just overwrite
        panorama(mask) = warpedImage(mask);
    end
end

% Normalize the panorama image if blending is enabled
if enableBlending
    % Compute weight sum to normalize
    weightSum = sum(panorama, 3);
    mask = weightSum > 0;
    panorama = bsxfun(@rdivide, panorama, cast(weightSum,class(panorama)));
    % panorama(mask) = panorama(mask) ./ weightSum(mask);
end

% Display the resulting panorama
% figure;
% imshow(panorama);
end
