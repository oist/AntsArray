% data folder
original_videopath = "D:\OIST Dropbox\Makoto Hiroi\makoto.hiroi@oist.jp’s files\20240415\";
mosaic_impath = "D:\OIST Dropbox\Makoto Hiroi\makoto.hiroi@oist.jp’s files\20240415\mosaic_video";

videofiles = dir(fullfile(original_videopath,'*.avi'));
vfileNames = {videofiles(~[videofiles.isdir] & ~startsWith({videofiles.name}, '.')).name};
[~, idx] = sort(cellfun(@(x) sscanf(x, 'cam%*d_%*d-%*d-%*d-%*d-%*d-%*d_cam%d.avi'), vfileNames));
vfileNames = vfileNames(idx);

imfiles = dir(fullfile(mosaic_impath,'*.png'));
fileNames = {imfiles(~[imfiles.isdir] & ~startsWith({imfiles.name}, '.')).name};
[frames, idx] = sort(cellfun(@(x) sscanf(x, 'frame_%d.png'), fileNames));
fileNames = fileNames(idx);

output_res = [4*2160/3 2160];
fps = 25;

%% zoom out to whole arena
f1 = figure;
ax1 = imshow(uint8(zeros(output_res(2), output_res(1))));
point = [14000, 1150];
startframe = 1;

time_control = [...
    4500,7300,3350,1000; ...
    4550,7300,3350,1480; ...
    5670,7300,3350,1480; ...
    6130,9360,3990,1480; ...
    6180,1,1,17633; ...
    8800,1,1,17633; ...
    8875,11600,935,5100; ...
    9855,11600,935,5100; ...
    10050,14100,1000,1500; ...
    10499,14100,1000,1500
    ];

ip_time = interpolateFrameData(time_control);

% Define the range of file indices
startIndex = ip_time(1,1);
endIndex = ip_time(end,1);

% Generate file names
imgNames = arrayfun(@(x) sprintf('frame_%08d.png', x), startIndex:endIndex, 'UniformOutput', false);

% Create an imageDatastore
imds = imageDatastore(fullfile(mosaic_impath,imgNames));

tic 
rec = VideoWriter(fullfile(original_videopath,'zoom.mp4'),"MPEG-4");
rec.FrameRate = fps;
rec.Quality = 100;
open(rec);

tic
frame = 1;
% while frame < 101
while hasdata(imds)
    im = read(imds);
    fprintf('%d/%d\n',frame,numel(imgNames))
    crop_window = [ip_time(frame,4) ip_time(frame,4)/4*3];
    point = ip_time(frame,2:3);
    % im = padarray(im,[100 100],0,'both');
    im_crop = imresize(imcrop(im,[point crop_window]), [2160, 2880]);
    ax1.CData = im_crop;
    % Write the frame to the video
    writeVideo(rec, im_crop);
    drawnow
    frame = frame+1;
end
close(rec);
toc

function time_ctrl_array = interpolateFrameData(time_control)
% % Define the array
% time_control = [4500, 7280, 3260, 1000; ...
%                 4550, 7280, 3260, 1480; ...
%                 5670, 7280, 3260, 1480; ...
%                 6130, 9360, 3990, 1480; ...
%                 6155, 1, 1, 17633];

% Extract columns for frame number, x, y, and width
frames = time_control(:,1);
x = time_control(:,2);
y = time_control(:,3);
width = time_control(:,4);

% Define new frame range from min to max frame number
new_frames = min(frames):max(frames);

% Interpolate x, y, and width values
itp = @(xx) interp1(frames, xx, new_frames, 'linear', 'extrap');

new_x = itp(x);
new_y = itp(y);
new_width = itp(width);
time_ctrl_array = [new_frames', new_x', new_y', new_width'];
end