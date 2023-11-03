function calculateZoomRatiosInFolder(folderPath)
    % Get a list of all files in the folder
    files = dir(fullfile(folderPath, '*.png'));
    
    % Filter out files that start with 'n' followed by 8 digits
    files = files(~startsWith({files.name}, 'n') | ~endsWith({files.name}, '.png'));
    
    % Initialize an array to store the zoom ratios
    zoomRatios = [];

    % Loop through each file
    for k = 1:length(files)
        fileName = files(k).name;
        
        % Check if the file name matches the target pattern (8 digits + .png)
        if isTargetFile(fileName)
            % Full path to the file
            filePath = fullfile(folderPath, fileName);
            
            % Process the image and calculate the zoom ratio
            zoomRatio = calculateImageZoomRatio(filePath);
            
            % Store the zoom ratio
            zoomRatios(end+1) = zoomRatio;
            
            % Display the file name and its zoom ratio
            disp(['File: ', fileName, ' - Zoom ratio: ', num2str(zoomRatio), ' pixels/mm']);
        end
    end
end

function isTarget = isTargetFile(fileName)
    % Define the pattern for target files: 8 digits followed by .png
    pattern = '^\d{8}\.png$';
    isTarget = ~isempty(regexp(fileName, pattern, 'once'));
end

function zoomRatio = calculateImageZoomRatio(imagePath)
    % Read the image
    image = imread(imagePath);
    
    % Convert the image to grayscale
    grayImage = rgb2gray(image);
    
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
