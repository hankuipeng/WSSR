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

% Last updated: 28th March 2020


function [W0, objs] = WSSR_euclid(X, k, rho, normalize, weight)

N = size(X, 1);
objs = zeros(N, 1);
epsilon = 1e-4;
sqeps = 1.0e-2; % square root of epsilon

if nargin < 4
    normalize = 0;
end

if normalize == 1
    X0 = X;
    X = norml2(X0, 1);
end

if nargin < 5
    weight = 1;
end


%%
W0 = zeros(N);
for i = 1:N
    
    idx = 1:N;
    idx(i) = [];
    
    Xopt = X(idx,:)';
    yopt = X(i,:)';
    
    
    %% calculate the Euclidean distances
    dists = [];
    for ii = 1:size(Xopt, 2)
        dists(ii) = sqrt(sum((yopt-Xopt(:,ii)).^2));
    end
    
    [vals, inds]= sort(dists, 'ascend');
    
    nn = inds(1:k);
    dk = vals(1:k);
    
    if weight == 1
        D = diag(dk);
        Dinv = diag(1./max(1e-4, dk)); % prevent the scenario where dk=0
    else
        D = eye(length(dk));
        Dinv = D;
    end
   
    
    %% QP for Constrained LASSO
    Xstar = [Xopt(:,nn)*Dinv; sqeps*eye(k)]; 
    ystar = [yopt; zeros(k,1)];
    
    A = Xstar'*Xstar;
    B = -Xstar'*Xstar;
    
    H = [A, B; B, A];
    H = (H+H')/2;
    f = rho*ones(2*k,1) - [Xstar'*ystar; -Xstar'*ystar]; 
    
    
    %% solve the QP
    options = optimoptions(@quadprog,'Display','off');
    [alpha,fopt,flag,out,lambda] = quadprog(H, f, [-Dinv, Dinv], zeros(k,1), [diag(Dinv)',-diag(Dinv)'], 1, ...
        zeros(2*k,1), [], [], options);
    beta = Dinv * (alpha(1:k) - alpha(k+1:2*k));
    W0(i,idx(nn)) = beta;
    
    
    %% calculate objective function value for point i
    partA = sum((yopt-Xopt(:,nn)*beta).^2);
    partB = sum(D*beta);
    partC = sum((D*beta).^2);
    objs(i) = partA/2 + partB*rho + partC*epsilon/2;

end

end