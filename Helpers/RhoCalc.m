%%% Input:
% X: N by P data matrix

function rhos = RhoCalc(X)

N = size(X, 1);
rhos = zeros(N, 1);
epsilon = 1e-4;

for i=1:N
    
    idx = 1:N;
    x = X(i,:)';
    idx(i) = [];
    Y = X(idx,:)';
    
    cand_vals = zeros(N-2, 1);
    dists_sq = sum((repmat(x, 1, N-1) - Y).^2, 1);
    dists = arrayfun(@(x) sqrt(x), dists_sq);
    [val, ind] = sort(dists, 'ascend');
    
    % re-order the columns of Y
    Y0 = Y;
    Y = Y0(:,ind);
    
    for j=2:(N-1)
        cand_vals(j) = ((Y(:,1)-Y(:,j))'*(Y(:,1)-x)+epsilon*val(1))/(val(j)-val(1));
    end
    
    rhos(i) = max(cand_vals);
    
end
end