function panorama = createPanorama_gpu(images, matrices, enableBlending)
% Load images and transformation matrices
% Replace these with your actual image loading and matrix setup
% for i = 1:numImages
%     images{i} = imread(sprintf('image%d.jpg', i));  % Example filenames
%     matrices{i} = eye(3);  % Replace with actual transformation matrices
% end

numImages = numel(images);
numChannels = 1;%size(images{1}, 3);

% Precompute affine matrices to avoid redundant computations
affineMatrices = cell(1, numImages);
for i = 1:numImages
    % Extract the affine portion (first two rows, three columns)
    affineMatrices{i} = matrices{i};
end

% Move images and matrices to GPU
imagesGPU = cell(1, numImages);
for i = 1:numImages
    imagesGPU{i} = gpuArray(images{i});
end

% Compute the output limits for each image to find the overall panorama size
xLimits = [inf, -inf];
yLimits = [inf, -inf];

for i = 1:numImages
    % Use the affine matrix
    tform = affinetform2d(affineMatrices{i});
    [xlim, ylim] = outputLimits(tform, [1, size(imagesGPU{i}, 2)], [1, size(imagesGPU{i}, 1)]);
    xLimits = [min(xLimits(1), xlim(1)), max(xLimits(2), xlim(2))];
    yLimits = [min(yLimits(1), ylim(1)), max(yLimits(2), ylim(2))];
end

% Define the size of the panorama
width = round(xLimits(2) - xLimits(1));
height = round(yLimits(2) - yLimits(1));
panorama = zeros(height, width, numChannels, 'like', imagesGPU{1});
% panorama = zeros(height, width, numChannels, 'like', images{1}, "gpuArray");
panoramaRef = imref2d(size(panorama), xLimits, yLimits);

% Warp each image into the panorama using a regular for loop
for i = 1:numImages
    % Use the affine matrix
    tform = affinetform2d(affineMatrices{i});

    % Warp the image on the GPU
    warpedImage = imwarp(imagesGPU{i}, tform, 'OutputView', panoramaRef, 'Interp', 'linear');

    % Create a mask for the current image on the GPU
    mask = imwarp(gpuArray(true(size(imagesGPU{i}, 1), size(imagesGPU{i}, 2))), tform, 'OutputView', panoramaRef);

    % Accumulate the warped images and masks
    if enableBlending
        % Linear blending: add to panorama and mask
        for c = 1:numChannels
            panorama(:, :, c) = panorama(:, :, c) + warpedImage(:, :, c) .* cast(mask, 'like', warpedImage);
        end
    else
        % Without blending, just overwrite the regions
        panorama(mask) = warpedImage(mask);
    end
end

% Normalize the panorama image if blending is enabled
if enableBlending
    % Compute weight sum to normalize
    weightSum = sum(panorama, 3);
    mask = weightSum > 0;
    panorama = bsxfun(@rdivide, panorama, cast(weightSum, class(panorama)));
end

% Gather the panorama back to the CPU for further usage or display
panorama = gather(panorama);

% Display the resulting panorama
% figure;
% imshow(panorama);
end
