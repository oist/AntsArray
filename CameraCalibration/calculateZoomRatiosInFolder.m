function calculateZoomRatiosInFolder(folderPath)
    % Get a list of all files in the folder
    files = dir(fullfile(folderPath, '*.png'));
    
    % Loop through each file
    for k = 1:length(files)
        fileName = files(k).name;
        
        % Check if the file name matches the target pattern
        if isTargetFile(fileName)
            % Full path to the file
            filePath = fullfile(folderPath, fileName);
            
            % Process the image and calculate the zoom ratio and physical dimensions
            [zoomRatio, physicalWidth, physicalHeight] = calculateImageZoomRatio(filePath);
            
            % Display the file name, zoom ratio, and physical dimensions
            disp(['File: ', fileName, ...
                  ' - Zoom ratio: ', num2str(zoomRatio), ' pixels/mm', ...
                  ' - Physical Width: ', num2str(physicalWidth), ' mm', ...
                  ' - Physical Height: ', num2str(physicalHeight), ' mm']);
        end
    end
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
