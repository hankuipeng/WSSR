% This function is an adaptation of WSSR_le.m. It still solves the problem
% through a system of linear equations. The only difference being how the
% weight matrix D is calculated. Previously we used absolute cosine 
% similarity, but here Euclidean distance is adopted instead. It is 
% suitable for data from affine subspaces or with manifold structure.

% Last updated: 1 Apr. 2020


function W = WSSR_le_euclid(X, k, rho, normalize, weight)

N = size(X, 1);
W = zeros(N);
epsilon = 1e-4;

if length(rho) == 1
    rhos = ones(N,1)*rho;
else
    rhos = rho;
end

if nargin < 4
    normalize = 1;
end

if normalize == 1
    X0 = X;
    X = norml2(X0, 1);
end

if nargin < 5
    weight = 1;
end


%%
for i = 1:N
    
    rho = rhos(i);
    yopt = X(i,:)';
    
    dists_sq = sum((repmat(yopt', N, 1) - X).^2, 2);
    dists = arrayfun(@(x) sqrt(x), dists_sq);
    dists(i) = Inf; % don't choose itself 
    
    [vals, inds]= sort(dists, 'ascend');    
    nn = inds(1:k);
    dk = max(vals(1:k), epsilon);
    
    if weight == 1
        D = diag(dk); 
    else
        D = diag(ones(length(dk),1));
    end
    
    
    %% set up the linear system
    Y = X(nn,:)'; % P x k
    a = Y'*Y + epsilon.*D'*D;
    b = ones(k, 1);
    
    A = [a, b; b', 0];
    B = [Y'*yopt-rho*D*b; 1];
    
    
    %% since n<p we add small ridge penalty to the formulation
    beta = linsolve(A,B);
    W(i,nn) = beta(1:k);
    
    
end

end