% This function constructs a N by N similarity matrix based on an input
% data matrix. It is done by solving N weighted sparse simplex
% representation (WSSR) problems. 

%%%% Inputs:
% X: N by P data matrix
% k: the number of nearest neighbors to consider 
% rho: the l1 penalty parameter
% normalize: 1 or 0 depending on whether to normalize the data or not 
% stretch: 1 or 0 depending on whether to stretch X_{-i} to touch the
% perpendicular hyperplane of x_{i}

%%% Outputs:
% W0: the N by N coefficient matrix that consists of all the solution
% vectors.
% objs: a vector of length N that stores all the objective function values
% for all points given their solution vectors.

% Last updated: 28th March 2020


function [W0, objs] = WSSR_cos(X, k, rho, normalize, stretch, weight)

N = size(X, 1);
objs = zeros(N, 1);
epsilon = 1e-4;
sqeps = 1.0e-2; % square root of epsilon

if nargin < 4
    normalize = 1;
end

if normalize == 1
    X0 = X;
    X = norml2(X0, 1);
end

if nargin < 5
    stretch = 0; % the default setting does not stretch the data points
end

if nargin < 6
    weight = 1;
end


%%
W0 = zeros(N);
for i = 1:N
    
    idx = 1:N;
    idx(i) = [];
    
    Xopt = X(idx,:)';
    yopt = X(i,:)';
    
    sims = yopt'*Xopt;
    
    %% We remove any zero cosine similarities
    if sum(sims <= 1e-4) ~= 0 
        ind = find(sims >= 1e-4);
        sims = sims(ind);
        idx = idx(ind);
        Xopt = Xopt(:,ind);
    end
    
    
    %% sort the similarity values in descending order 
    [vals, inds]= sort(abs(sims), 'descend');
    %[vals, inds]= sort(sims, 'descend');
    
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
        Dinv = diag(dk);
    else
        D = eye(length(dk));
        Dinv = D;
    end 
    
    
    %% stretch the data points that will be considered in the program
    if stretch
        Xst = Xopt(:,nn);
        Ts = 1./(yopt'*Xst);
        Xst = Xst*diag(Ts);
        Xopt(:,nn) = Xst;
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
    
    
    %%
    partA = sum((yopt-Xopt(:,nn)*beta).^2);
    partB = sum(D*beta);
    partC = sum((D*beta).^2);
    objs(i) = partA/2 + partB*rho + partC*epsilon/2;
    
end

end