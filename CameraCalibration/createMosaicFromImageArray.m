function mosaic = createMosaicFromImageArray(images, Hall)
    % images: a cell array of 25 images
    % Hall: a cell array of precomputed homography matrices for each image
    
    im_n = length(images);  % Number of images
    imsize = zeros(im_n, 2);  % Store the size of each image
    
    % Calculate image sizes
    parfor i = 1:im_n
        if numel(size(images{i})) >= 3
            images{i} = im2gray(images{i});
        end
        imsize(i,:) = size(images{i});
    end

    % Compute bounds for the mosaic
    ubox = cell(im_n,1);
    vbox = cell(im_n,1);
    ubox_ = cell(im_n,1);
    vbox_ = cell(im_n,1);
    ubox_all_ = [];
    vbox_all_ = [];
    for i = 1 : im_n
        ubox{i} = [1:imsize(i,2) 1:imsize(i,2) ones(1,imsize(i,1)) imsize(i,2)*ones(1,imsize(i,1))];
        vbox{i} = [ones(1,imsize(i,2)) imsize(i,1)*ones(1,imsize(i,2)) 1:imsize(i,1) 1:imsize(i,1)];
        H = inv(Hall{i});
        z1_ = H(3,1)*ubox{i} + H(3,2)*vbox{i} + H(3,3);
        ubox_{i} = (H(1,1)*ubox{i} + H(1,2)*vbox{i} + H(1,3)) ./ z1_;
        vbox_{i} = (H(2,1)*ubox{i} + H(2,2)*vbox{i} + H(2,3)) ./ z1_;
        ubox_all_ = cat(2, ubox_all_, ubox_{i});
        vbox_all_ = cat(2, vbox_all_, vbox_{i});
    end

    % Determine the bounds of the mosaic
    u0 = min(ubox_all_);
    u1 = max(ubox_all_);
    v0 = min(vbox_all_);
    v1 = max(vbox_all_);
    ur = u0:u1;
    vr = v0:v1;

    % Prepare the output mosaic
    mosaicw = numel(ur);
    mosaich = numel(vr);
    mosaic = zeros(mosaich, mosaicw);
    mass = zeros(mosaich, mosaicw);

    m_u0_ = zeros(im_n, 1);
    m_u1_ = zeros(im_n, 1);
    m_v0_ = zeros(im_n, 1);
    m_v1_ = zeros(im_n, 1);
    imw_ = zeros(im_n, 1);
    imh_ = zeros(im_n, 1);
    for i = 1:im_n
        % align the sub coordinates with the mosaic coordinates
        margin = 0.2 * min(imsize(i,1),imsize(i,2)); % additional margin of the reprojected image region considering the possilbe deformation
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

    % Reproject and blend each image into the mosaic
    [u, v] = meshgrid(ur, vr);

    im_p = cell(im_n, 1);
    mask = cell(im_n, 1);

    for i = 1:im_n
        u_im = u(m_v0_(i):m_v1_(i), m_u0_(i):m_u1_(i));
        v_im = v(m_v0_(i):m_v1_(i), m_u0_(i):m_u1_(i));

        H = Hall{i};
        z1_ = H(3, 1) * u_im + H(3, 2) * v_im + H(3, 3);
        u_im_ = (H(1, 1) * u_im + H(1, 2) * v_im + H(1, 3)) ./ z1_;
        v_im_ = (H(2, 1) * u_im + H(2, 2) * v_im + H(2, 3)) ./ z1_;

        im_p{i} = interp2(im2double(images{i}), u_im_, v_im_);

        mask{i} = double(~isnan(im_p{i}));
        im_p{i}(isnan(im_p{i})) = 0;

        mass(m_v0_(i):m_v1_(i), m_u0_(i):m_u1_(i), :) ...
            = mass(m_v0_(i):m_v1_(i), m_u0_(i):m_u1_(i), :) + mask{i};
        mosaic(m_v0_(i):m_v1_(i), m_u0_(i):m_u1_(i), :) ...
            = mosaic(m_v0_(i):m_v1_(i), m_u0_(i):m_u1_(i), :) + im_p{i};
    end

    % Normalize the mosaic to account for overlapping areas
    mosaic(mass > 0) = mosaic(mass > 0) ./ mass(mass > 0);
    mosaic(isnan(mosaic)) = 0;  % Clean up any NaN resulting from zero division

    % % Display and save the mosaic image
    % figure;
    % imshow(mosaic, 'border', 'tight');
    % drawnow;
end