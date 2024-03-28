%%

im_path = '/home/sam/bucket/Ants/basler/cameraArray_calib/2024_Feb/2024-02-27-18-39_AprilTag36h10_2mm/';

% load images
cd(im_path)
im_files=dir('*tiff');
im_files(startsWith({im_files.name},'.')) = []; % remove unexpected hidden files

im_file_list={};
im_file_list=fullfile(im_path, {im_files.name}); % fullfile for multi-platform

% Extract 'camXX' pattern using regular expressions
cam_pattern = 'cam\d{2}';
cams = regexp(im_file_list, cam_pattern, 'match', 'once');

% Convert extracted patterns to numeric values
cams_numeric = cellfun(@(x) str2double(x(4:end)), cams);

% Reorder the file list based on the numeric values of 'camXX'
[~, idx] = sort(cams_numeric);
im_file_list_reordered = im_file_list(idx);

im_n=length(im_file_list_reordered);
im=cell(im_n,1);
imsize = zeros(im_n,2);

% Parameters
lambda = 0.001 * imsize(1,1)*imsize(1,2); % weighting parameter to balance the fitting term and the smoothing term
intv_mesh = 50; % interval in pixels for the computing of deformation functions
K_smooth = 5; % the smooth transition width in the non-overlapping region is set to K_smooth times of the maximum bias.


for i = 1:im_n
    currImg=imread(im_file_list_reordered{i});
    im{i}=im2gray(currImg);
    imsize(i,:) = size(im{i});
end

%%
disp('feature detection and matching')

% establish neighbors
array_size = [5,5];
cam_layout = transpose(reshape(0:(im_n-1), array_size(2), array_size(1)));
cam_neighbors = cell(array_size(1), array_size(2));

edge_list=[];
for row = 1:array_size(1)
    for column = 1:array_size(2)
        row_range = max(row - 1, 1):min(row + 1, array_size(1));
        column_range = max(column - 1, 1):min(column + 1, array_size(2));
        cam_neighbors{row, column} = reshape(cam_layout(row_range,column_range),1,[]);
        curr_edges=cam_neighbors{row, column};
        curr_cam=cam_layout(row,column);
        curr_edges(curr_edges<=curr_cam)=[]; %only take greater values to not repeat calculations
        curr_edge_list=zeros(numel(curr_edges),2);
        for pair=1:numel(curr_edges)
            curr_edge_list(pair,:)=[curr_cam, curr_edges(pair)];
        end
        edge_list=[edge_list; curr_edge_list];
    end
end
edge_list=edge_list+1; %the neighbors are 0 based, but I'll access them in matlab with 1 based
edge_n=size(edge_list,1);

% feature detection
points = cell(im_n, 1);
features = cell(im_n, 1);
valid_points = cell(im_n, 1);


for i = 1:im_n
    points{i} = detectSURFFeatures(im{i},...
        'MetricThreshold', 600, 'NumOctaves',1, 'NumScaleLevels', 4);
    [features{i}, valid_points{i}] = extractFeatures(im{i}, points{i});
end

%feature matching
X = cell(edge_n, 2);
Hall_init = cell(edge_n,1);
matchNum=[];
for ei = 1 : edge_n
    i = edge_list(ei, 1);
    j = edge_list(ei, 2);
    
    indexPairs = matchFeatures(features{i}, features{j},'MatchThreshold',10,'Method','Approximate');
    matched_points_1 = valid_points{i}(indexPairs(:, 1), :);
    matched_points_2 = valid_points{j}(indexPairs(:, 2), :);

    try
        [tform, inlierIndices] = ...
            estimateGeometricTransform2D(matched_points_1, matched_points_2, 'similarity','Confidence',99,'MaxDistance',20);
        X_1 = [matched_points_1(inlierIndices).Location'; ones(1,size(matched_points_1(inlierIndices), 1))];
        X_2 = [matched_points_2(inlierIndices).Location'; ones(1,size(matched_points_2(inlierIndices), 1))];
        Hall_init{ei}=eye(3);%tform.T seems simple initiliazation works fine
    catch
        X_1=[];
        X_2=[];
        Hall_init{ei}=eye(3);
    end

    X{ei,1} = double(X_1);
    X{ei,2} = double(X_2);
    matchNum(ei)=size(X_1,2);
    disp(['matched edge ' num2str(ei) ' out of ' num2str(edge_n)])
end
Xorig=X;

% filter out edges that have a small number of shared features (they can
%wreak havoc!) Can happen even where there are lots of points
badEdges=matchNum<20;

%badEdges(7:end)=1; %for debugging!

X(badEdges,:)=[];
edge_list(badEdges,:)=[];
edge_n = size(edge_list, 1);
Hall_init(badEdges)=[];


% 
% % for debugging
% fi=2
% si=6
% 
% figure
% imagesc(im{ fi})
% hold on
% scatter(X{ fi}(1,:),X{ fi}(2,:),10,'r','filled')
% 
% figure
% imagesc(im{6})
% hold on
% scatter(X{2}(1,:),X{2}(2,:),10,'r','filled')
% % 
% figure;
% for x=1:size(X,1)
% showMatchedFeatures(im{edge_list(x,1)}, im{edge_list(x,2)}, X{x,1}(1:2,:)', X{x,2}(1:2,:)','montage');
%  pause
% end

%% Bundle adjustment under similarity transformation
disp('bundle adjustment')



paras_init=[];
for H=1:numel(Hall_init)
    paras_init=double([paras_init, Hall_init{H}(1,1) Hall_init{H}(1,2)  Hall_init{H}(1,3)  Hall_init{H}(2,3)]);
end


% paras_init=[];
% for H=1:numel(Hall_init)
%     paras_init=double([paras_init, 0,0,0,0]);
% end


options = optimoptions('lsqnonlin', 'Algorithm','levenberg-marquardt', 'Display','off',...
    'MaxFunEvals',1000*im_n, 'MaxIter',1e3, 'TolFun',1e-6, 'TolX',1e-6, 'Jacobian','off');

for sigma = [1000, 100, 10]
    [paras, ~ ,~ ,exitflag] = lsqnonlin(...
        @(p)residual_all_robust_similarity(X, edge_list, p, sigma), paras_init,...
        [],[],options);
    if exitflag > 0
        paras_init = paras;
    end
end
save([im_path,'bundle_adjustment_paras.mat'],'paras');

H_pair = cell(im_n, im_n);
for i = 1 : im_n
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

Hall=H_pair(12,:); %taking 1 row as the mappings. They should all be consistent after BA, but I licked a camera in the middle

%% compute mosaic 

ubox = cell(im_n,1);
vbox = cell(im_n,1);
ubox_ = cell(im_n,1);
vbox_ = cell(im_n,1);
ubox_all_ = [];
vbox_all_ = [];
for i = 1 : im_n
    ubox{i} = [1:imsize(i,2)        1:imsize(i,2)                     ones(1,imsize(i,1))  imsize(i,2)*ones(1,imsize(i,1))] ;
    vbox{i} = [ones(1,imsize(i,2))  imsize(i,1)*ones(1,imsize(i,2))  1:imsize(i,1)        1:imsize(i,1) ];
    H=inv(Hall{i});
    z1_ = H(3,1)*ubox{i} + H(3,2)*vbox{i} + H(3,3);
    ubox_{i} = (H(1,1)*ubox{i} +  H(1,2)*vbox{i}  + H(1,3)) ./ z1_;
    vbox_{i} = (H(2,1)*ubox{i} + H(2,2)*vbox{i} + H(2,3)) ./ z1_;
    ubox_all_ = cat(2,ubox_all_,ubox_{i});
    vbox_all_ = cat(2,vbox_all_,vbox_{i});
end

u0 = min(ubox_all_);
u1 = max(ubox_all_);
ur = u0:u1;
v0 = min(vbox_all_);
v1 = max(vbox_all_);
vr = v0:v1;
mosaicw = size(ur, 2);
mosaich = size(vr, 2);

m_u0_ = zeros(im_n,1);
m_u1_ = zeros(im_n,1);
m_v0_ = zeros(im_n,1);
m_v1_ = zeros(im_n,1);
imw_ = zeros(im_n,1);
imh_ = zeros(im_n,1);
for i = 1 : im_n
    % align the sub coordinates with the mosaic coordinates
    margin = 0.2 * min(imsize(1,1),imsize(1,2)); % additional margin of the reprojected image region considering the possilbe deformation
    u0_im_ = max(min(ubox_{i}) - margin, u0);
    u1_im_ = min(max(ubox_{i}) + margin, u1);
    v0_im_ = max(min(vbox_{i}) - margin, v0);
    v1_im_ = min(max(vbox_{i}) + margin, v1);
    m_u0_(i) = ceil(u0_im_ - u0 + 1);
    m_u1_(i) = floor(u1_im_ - u0 + 1);
    m_v0_(i) = ceil(v0_im_ - v0 + 1);
    m_v1_(i) = floor(v1_im_ - v0 + 1);
    imw_(i) = floor(m_u1_(i) - m_u0_(i) + 1);
    imh_(i) = floor(m_v1_(i) - m_v0_(i) + 1);
end

[u,v] = meshgrid(ur,vr) ;

im_p = cell(im_n,1);
mask = cell(im_n,1);
mass = zeros(mosaich, mosaicw);
mosaic = zeros(mosaich, mosaicw);
for i = 1 : im_n
    disp(i)
    u_im = u(m_v0_(i):m_v1_(i),m_u0_(i):m_u1_(i));
    v_im = v(m_v0_(i):m_v1_(i),m_u0_(i):m_u1_(i));

    H=Hall{i};
    z1_ = H(3,1)*u_im + H(3,2)*v_im + H(3,3);
    u_im_ = (H(1,1)*u_im + H(1,2)*v_im + H(1,3)) ./ z1_;
    v_im_ = (H(2,1)*u_im + H(2,2)*v_im + H(2,3)) ./ z1_;



    im_p{i} = interp2(im2double(im{i}),u_im_,v_im_);

    mask{i} = double(~isnan(im_p{i}));
    im_p{i}(isnan(im_p{i})) = 0;
    
    mass(m_v0_(i):m_v1_(i),m_u0_(i):m_u1_(i),:)...
        = mass(m_v0_(i):m_v1_(i),m_u0_(i):m_u1_(i),:) + mask{i};
    mosaic(m_v0_(i):m_v1_(i),m_u0_(i):m_u1_(i),:)...
        = mosaic(m_v0_(i):m_v1_(i),m_u0_(i):m_u1_(i),:) + im_p{i};

    % imshow(mosaic, 'border', 'tight') ;
    % pause

end

mosaic = mosaic ./ mass;
mosaic(isnan(mosaic)) = 0;

figure ;
imshow(mosaic, 'border', 'tight') ;
drawnow;

%imwrite(mosaic, [im_path, 'mosaic_global.png']);


%     %     if save_results
%             imwrite(mosaic, [imfolder, 'mosaic_global.jpg']);
%         end

