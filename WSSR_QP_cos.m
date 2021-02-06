% This function constructs an N by N similarity matrix based on an input
% data matrix, which is done by solving N weighted sparse simplex
% representation (WSSR) problems. 

%%%% Inputs:
% X: N by P data matrix.
% k: the number of nearest neighbours to consider. 
% rho: the l1-penalty parameter.
% normalize: 1 or 0 depending on whether to normalise the data or not. 
% stretch: 1 or 0 depending on whether to stretch X_{-i} to touch the
% perpendicular hyperplane of x_{i} or not.
% weight: 1 or 0 depending on whether to apply a weight matrix within the 
% l1 and l2 norm penalty of the objective.

%%% Outputs:
% W0: the N by N coefficient matrix that consists of all the solution
% vectors.
% objs: a vector of length N that stores all the objective function values
% for all points given their solution vectors.

% Last updated: 8 Jul. 2020


function [W0, objs] = WSSR_QP_cos(X, k, rho, normalize, stretch, weight)

N = size(X, 1);
objs = zeros(N, 1);
W0 = zeros(N);
epsilon = 1e-4; 

if nargin < 4
    normalize = 1;
end

if normalize == 1
    X0 = X;
    X = norml2(X0, 1);
end

if nargin < 5
    stretch = 1; 
end

if nargin < 6
    weight = 1;
end


%%
for i = 1:N
    
    %% We remove any zero cosine similarities
    idx = 1:N;
    idx(i) = [];
    
    Xopt = X(idx,:)';
    yopt = X(i,:)';
   
    % calculate the cosine similarities
    sims = abs(yopt'*Xopt);
    
    if sum(sims <= 1e-4) ~= 0 
        ind = find(sims >= 1e-4);
        if length(ind) > 0
            sims = sims(ind);
            idx = idx(ind);
        else
            sims = 1e-4;
            idx = randsample(idx, 1);
        end
    end
    
    
    %% sort the similarity values in descending order 
    [vals, inds]= sort(abs(sims), 'descend');
    
    if k == 0 % consider only the positive similarity values 
        dk = vals(vals>0);
        nn = inds(vals>0);
        k = length(dk);
    else
        if k > length(vals) % if some zero entries have been removed from sims
            dk = vals;
            nn = inds;
            k = length(dk);
        else
            dk = vals(1:k);
            nn = inds(1:k);
        end
    end
    
    
    %% calculate the weight matrix
    if weight == 1
        D = diag(1./dk);
    else
        D = eye(length(dk));
    end 
    
    
    %% stretch the data points that will be considered in the program
    Y = X(idx(nn),:)';
    
    if stretch
        Xst = Y;
        Ts = 1./(yopt'*Xst);
        Xst = Xst*diag(Ts);
        Y = Xst;
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
