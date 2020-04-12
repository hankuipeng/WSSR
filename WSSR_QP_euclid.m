% This function constructs a N by N similarity matrix based on an input
% data matrix. It is done by solving N weighted sparse simplex
% representation (WSSR) problems. 

%%%% Inputs:
% X: N by P data matrix.
% k: the number of nearest neighbors to consider. 
% rho: the l1 penalty parameter.
% normalize: 1 or 0 depending on whether to normalize the data or not. 
% weight: 1 or 0 depending on whether to apply a weight matrix within the 
% l1 and l2 norm penalty of the objective.

%%% Outputs:
% W0: the N by N coefficient matrix that consists of all the solution
% vectors.
% objs: a vector of length N that stores all the objective function values
% for all points given their solution vectors.

% Last updated: 12 Apr. 2020


function [W0, objs] = WSSR_QP_euclid(X, k, rhos, normalize, weight)

N = size(X, 1);
objs = zeros(N, 1);
epsilon = 1e-4;
W0 = zeros(N);

if nargin < 4
    normalize = 0;
end

if normalize == 1
    X0 = X;
    X = norml2(X0, 1);
end

if length(rhos) == 1
    rhos = ones(N, 1)*rhos;
end

if nargin < 5
    weight = 1;
end


%%
for i = 1:N
    
    rho = rhos(i);
    
    %% calculate the Euclidean distances
    yopt = X(i,:)';
    dists_sq = sum((repmat(yopt', N, 1) - X).^2, 2);
    dists = arrayfun(@(x) sqrt(x), dists_sq);
    dists(i) = Inf; % don't choose itself 
    
    [vals, inds]= sort(dists, 'ascend');    
    nn = inds(1:k);
    dk = max(vals(1:k), epsilon);
    Y = X(nn,:)'; % P x k
    
    if weight == 1
        D = diag(dk);
    else
        D = eye(length(dk));
    end
   
    
    %% QP for Constrained LASSO
    H = Y'*Y+epsilon.*D*D;
    f = rho.*diag(D)-Y'*yopt; 
    
    
    %% solve the QP
    options = optimoptions(@quadprog,'Display','off');
    [beta,fopt,flag,out,lambda] = quadprog(H, f, -eye(k), zeros(k,1), ones(1,k), 1, ...
        zeros(k,1), [], [], options);
    W0(i,nn) = beta;
    
    
    %% calculate objective function value for point i
    partA = sum((yopt-Y*beta).^2);
    partB = sum(D*beta);
    partC = sum((D*beta).^2);
    objs(i) = partA/2 + partB*rho + partC*epsilon/2;

end

end