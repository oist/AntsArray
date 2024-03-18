function calculateImageDimensionsInFolder(folderPath)
% Get a list of all files in the folder
files = dir(fullfile(folderPath, '*.png'));

% Loop through each file
for k = 1:length(files)
    fileName = files(k).name;

    % Check if the file name matches the target pattern
    if isTargetFile(fileName)
        % Full path to the file
        filePath = fullfile(folderPath, fileName);

        % Process the image and calculate the zoom ratio
        zoomRatio = calculateImageZoomRatio(filePath);

        % Correct rotation if necessary
        correctedImage = correctImageRotation(filePath);

        % Read the image to get its pixel dimensions
        imageInfo = imfinfo(filePath);
        imageWidthPixels = imageInfo.Width;
        imageHeightPixels = imageInfo.Height;

        % Convert pixel dimensions to physical dimensions using the zoom ratio
        physicalWidth = imageWidthPixels / zoomRatio;
        physicalHeight = imageHeightPixels / zoomRatio;

        % Display the file name, zoom ratio, and image physical dimensions
        disp(['File: ', fileName, ...
            ' - Zoom ratio: ', num2str(zoomRatio), ' pixels/mm', ...
            ' - Image Physical Width: ', num2str(physicalWidth), ' mm', ...
            ' - Image Physical Height: ', num2str(physicalHeight), ' mm']);
    end
end
end

function correctedImage = correctImageRotation(filePath)
    % Read the image
    image = imread(filePath);
    
    % Convert the image to grayscale if it is not already
    if size(image, 3) == 3
        grayImage = rgb2gray(image);
    else
        grayImage = image;
    end
    
    % Apply Gaussian blur to reduce noise
    blurredImage = imgaussfilt(grayImage, 2);
    
    % Use Canny edge detection to find edges
    edges = edge(blurredImage, 'Canny');
    
    % Perform the Hough Transform to find lines
    [H, T, R] = hough(edges);
    P = houghpeaks(H, 5, 'threshold', ceil(0.3*max(H(:))));
    lines = houghlines(edges, T, R, P, 'FillGap', 5, 'MinLength', 7);
    
    % Display the original image
    figure, imshow(image), title('Original Image with Detected Lines');
    hold on;
    
    % Initialize array of angles
    angles = [];
    
    % Plot the detected lines and compute their angles
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
        
        % Compute the angle of the line
        angle = atan2(diff(xy(:,2)), diff(xy(:,1)));
        angles(end+1) = rad2deg(angle);
    end
    
    % Find the most common angle
    mostCommonAngle = mode(round(angles));
    
    % Log the angle
    disp(['Most common angle for ', filePath, ': ', num2str(mostCommonAngle)]);
    
    % Rotate the image to correct the alignment
    correctedImage = imrotate(image, -mostCommonAngle, 'bilinear', 'crop');
    
    % Display the corrected image
    figure, imshow(correctedImage), title('Corrected Image');
    hold off;
end

function isTarget = isTargetFile(fileName)
% Define the pattern for target files: 8 digits followed by optional underscore and any characters and .png
pattern = '^\d{8}(_.*?)?\.png$';

% Check if the fileName matches the pattern
isTarget = ~isempty(regexp(fileName, pattern, 'once'));
end

function [zoomRatio, physicalWidth, physicalHeight] = calculateImageZoomRatio(imagePath)
% Read the image
image = imread(imagePath);

% Convert the image to grayscale if it is not already
if size(image, 3) == 3
    grayImage = rgb2gray(image);
else
    grayImage = image;
end

% Apply Gaussian blur to reduce noise
blurredImage = imgaussfilt(grayImage, 2);

% Use Canny edge detection
edges = edge(blurredImage, 'Canny');

% Find the largest contour
[B, ~] = bwboundaries(edges, 'noholes');

% Calculate the bounding box of the largest contour
[x, y, w, h] = boundingBox(B);

% Known physical size of the lattice in millimeters
realSizeMm = 10; % mm

% Calculate the average size in pixels and the zoom ratio
averageSizePixels = (w + h) / 2;
zoomRatio = averageSizePixels / realSizeMm;

% Calculate the physical width and height based on the zoom ratio
physicalWidth = w / zoomRatio;
physicalHeight = h / zoomRatio;
end

function [x, y, w, h] = boundingBox(B)
maxArea = 0;
largestContour = [];
for k = 1:length(B)
    contour = B{k};
    area = polyarea(contour(:,2), contour(:,1));
    if area > maxArea
        maxArea = area;
        largestContour = contour;
    end
end
x = min(largestContour(:,2));
y = min(largestContour(:,1));
w = max(largestContour(:,2)) - x;
h = max(largestContour(:,1)) - y;
end
