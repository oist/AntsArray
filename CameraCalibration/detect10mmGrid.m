function detect10mmGrid(imagePath)
    % Read the image
    image = imread(imagePath);
    grayImage = rgb2gray(image);

    % Apply a median filter to reduce noise while preserving edges
    filteredImage = medfilt2(grayImage, [3 3]);

    % Apply adaptive thresholding to get a binary image
    binImage = imbinarize(filteredImage, 'adaptive', 'ForegroundPolarity','dark','Sensitivity',0.4);

    % Perform the Hough Transform to find lines
    [H, theta, rho] = hough(binImage);
    peaks = houghpeaks(H, 100, 'threshold', ceil(0.3*max(H(:))), 'NHoodSize', [5 5]);
    lines = houghlines(binImage, theta, rho, peaks, 'FillGap', 20, 'MinLength', 40);

    % Display the original image
    figure, imshow(image), title('Detected 10mm Grid');
    hold on;

    % Draw the detected lines
    for k = 1:length(lines)
        endpoints = [lines(k).point1; lines(k).point2];
        plot(endpoints(:,1), endpoints(:,2), 'LineWidth', 2, 'Color', 'green');
    end

    hold off;
end
