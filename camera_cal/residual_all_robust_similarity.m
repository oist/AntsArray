function [ err ] = residual_all_robust_similarity( X, edge_list, paras, sigma )

% parameretes sigma indicates the distance scope for inliers with homopraphy matrix, in pixels
indexes=unique(edge_list);

im_n = numel(unique(edge_list));

edge_n = size(X, 1);


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

err = [];
for ei = 1 : edge_n
    i = edge_list(ei, 1);
    j = edge_list(ei, 2);
    
    i=find(i==indexes);
    j=find(j==indexes);
    
    X1_p2 = H_pair{i,j} * double(X{ei,1});
    X1_p2(1,:) = X1_p2(1,:) ./ X1_p2(3,:) ;
    X1_p2(2,:) = X1_p2(2,:) ./ X1_p2(3,:) ;
    errH1_2 = double(X{ei,2}) - X1_p2;
    X2_p1 =  H_pair{i,j} \ double(X{ei,2});
    X2_p1(1,:) = X2_p1(1,:) ./ X2_p1(3,:) ;
    X2_p1(2,:) = X2_p1(2,:) ./ X2_p1(3,:) ;
    errH2_1 = double(X{ei,1}) - X2_p1;
    %  err = [errH1_2(1,:)' errH1_2(2,:)' errH2_1(1,:)' errH2_1(2,:)'];
    
    % err_ij = residual_KR( double(X{ei,1}), double(X{ei,2}), Ki, Kj, R_pair{i,j});
    %   err_ji = residual_KR( double(X{ei,2}), double(X{ei,1}), Kj, Ki, R_pair{j,i});
    err = cat(1,err, errH1_2(1,:)', errH1_2(2,:)', errH2_1(1,:)', errH2_1(2,:)');
end

outlier = (abs(err) > sigma);
err(outlier) = sign(err(outlier)) .* (sigma + sigma * log(abs(err(outlier))/sigma));
% err(outlier) = sign(err(outlier)) .* sqrt(2*sigma*abs(err(outlier)) - sigma*sigma);

end