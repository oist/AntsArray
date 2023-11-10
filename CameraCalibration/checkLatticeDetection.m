function checkLatticeDetection(imagePath)
    % Read the image
    image = imread(imagePath);
    grayImage = rgb2gray(image);
    
    % Threshold the image to get the lattices
    % Assuming the lattices are darker than the background
    thresholdValue = graythresh(grayImage); % Otsu's method
    latticeMask = grayImage < thresholdValue * 255;
    
    % Perform the Hough Transform to find lines
    [H, T, R] = hough(latticeMask);
    P  = houghpeaks(H, 10, 'threshold', ceil(0.5*max(H(:))));
    lines = houghlines(latticeMask, T, R, P, 'FillGap', 20, 'MinLength', 40);
    
    % Create a plot to show the original image and the detected lines
    figure, imshow(image), title('Original Image with Lattice Lines');
    hold on;
    
    % Plot the detected lines on the original image
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'LineWidth',2,'Color','green');

        % Plot beginnings and ends of lines
        plot(xy(1,1), xy(1,2), 'x', 'LineWidth',2,'Color','yellow');
        plot(xy(2,1), xy(2,2), 'x', 'LineWidth',2,'Color','red');
    end
    
    hold off;
end
