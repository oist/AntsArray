%% Camera Calibration Using Functional Interface
% Step 1: Generate the calibration pattern
% Download and prepare tag images

downloadURL  = "https://github.com/AprilRobotics/apriltag-imgs/archive/master.zip";
dataFolder   = fullfile(tempdir,"apriltag-imgs",filesep); 
options      = weboptions('Timeout', Inf);
zipFileName  = fullfile(dataFolder,"apriltag-imgs-master.zip");
folderExists = exist(dataFolder,"dir");

% Create a folder in a temporary directory to save the downloaded file.
if ~folderExists  
    mkdir(dataFolder); 
    disp("Downloading apriltag-imgs-master.zip (60.1 MB)...") 
    websave(zipFileName,downloadURL,options); 
    
    % Extract contents of the downloaded file.
    disp("Extracting apriltag-imgs-master.zip...") 
    unzip(zipFileName,dataFolder); 
end

%% Set the properties of the calibration pattern.
tagArrangement = [12,16];
tagFamily = "tag36h11";

% Generate the calibration pattern using AprilTags.
tagImageFolder = fullfile(dataFolder,"apriltag-imgs-master",tagFamily);
imdsTags = imageDatastore(tagImageFolder);
calibPattern = helperGenerateAprilTagPattern(imdsTags,tagArrangement,tagFamily);

function calibPattern = helperGenerateAprilTagPattern(imdsTags,tagArragement,tagFamily)

numTags = tagArragement(1)*tagArragement(2);
tagIds = zeros(1,numTags);

% Read the first image.
I = readimage(imdsTags,3);
Igray = im2gray(I);

% Adjust the scale factor to modify the tag size. Decrease for smaller tags.
scaleFactor = 20; % Smaller than previous 15

% Scale up the thumbnail tag image.
Ires = imresize(Igray,scaleFactor,"nearest");

% Detect the tag ID and location (in image coordinates).
[tagIds(1), tagLoc] = readAprilTag(Ires,tagFamily);

% Adjust pad size to increase space between tags.
tagSize = round(max(tagLoc(:,2)) - min(tagLoc(:,2)));
extraPad = 10; % Additional padding to increase space between tags
padSize = round(tagSize/2 - (size(Ires,2) - tagSize)/2) + extraPad;
Ires = padarray(Ires,[padSize,padSize],255);

% Initialize tagImages array to hold the scaled tags.
tagImages = zeros(size(Ires,1),size(Ires,2),numTags);
tagImages(:,:,1) = Ires;

for idx = 2:numTags
   
    I = readimage(imdsTags,idx + 2);
    Igray = im2gray(I);
    Ires = imresize(Igray,scaleFactor,"nearest");
    Ires = padarray(Ires,[padSize,padSize],255);
    
    tagIds(idx) = readAprilTag(Ires,tagFamily);
    
    % Store the tag images.
    tagImages(:,:,idx) = Ires;
     
end

% Sort the tag images based on their IDs.
[~, sortIdx] = sort(tagIds);
tagImages = tagImages(:,:,sortIdx);

% Reshape the tag images to ensure that they appear in column-major order
% (montage function places image in row-major order).
columnMajIdx = reshape(1:numTags,tagArragement)';
tagImages = tagImages(:,:,columnMajIdx(:));

% Create the pattern using 'montage'.
imgData = montage(tagImages,Size=tagArragement);
calibPattern = imgData.CData;

end
