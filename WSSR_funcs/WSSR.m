% This function constructs a N by N similarity matrix based on an input
% data matrix. It is done by solving N weighted sparse simplex
% representation (WSSR) problems. 

% The changes we made in this version (based on WSSR_v2.m): 
% (a) we remove x_{j}s if x_{i}^{T}x_{j} = 0,
% (b) we also order the cosine similarities instead of the absolute values.

% Inputs:
% X: N by P data matrix
% k: the number of nearest neighbors to consider 
% rho: the l1 penalty parameter
% normalize: 1 or 0 depending on whether to normalize the data or not 
% stretch: 1 or 0 depending on whether to stretch X_{-i} to touch the
% perpendicular hyperplane of x_{i}


function W0 = WSSR(X, k, rho, normalize, stretch)

N = size(X, 1);

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


%%
W0 = zeros(N);
for i = 1:N
    
    idx = 1:N;
    idx(i) = [];
    
    Xopt = X(idx,:)';
    yopt = X(i,:)';
    
    sims = yopt'*Xopt;
    
    %% We remove any zero cosine similarities
    if sum(sims == 0) ~= 0 
        idx = idx(find(sims~=0));
        sims = sims(find(sims~=0));
    end
    
    
    %% sort the similarity values in descending order 
    %[vals, inds]= sort(sims, 'descend');
    [vals, inds]= sort(abs(sims), 'descend');
    
    if k == 0 % consider only the positive similarity values 
        dk = vals(find(vals>0));
        nn = inds(find(vals>0));
        k = length(dk);
    else
        dk = vals(1:k);
        nn = inds(1:k);
    end
    
    Dinv = diag(dk); 
    
    
    %% stretch the data points that will be considered in the program
    if stretch
        Xst = Xopt(:,nn);
        Ts = 1./(yopt'*Xst);
        Xst = Xst*diag(Ts);
        Xopt(:,nn) = Xst;
    end
    
    
    %% since n<p we add small ridge penalty to the formulation
    sqeps = 1.0e-2; % squared epsilon
    
    %Xstar = [Xopt(:,nn); sqeps*Dinv]; % no penalty on the second l2 norm
    %Xstar = [Xopt(:,nn)*Dinv; sqeps*Dinv]; % no penalty on both of the l2 norms
    %Xstar = [Xopt(:,nn); sqeps*eye(k)]; % penalty on all three terms 
    Xstar = [Xopt(:,nn)*Dinv; sqeps*eye(k)]; % no penalty on the first l2 norm
    ystar = [yopt; zeros(k,1)];

    
    %% QP for Constrained LASSO
    H = [Xstar'*Xstar, -Xstar'*Xstar; -Xstar'*Xstar, Xstar'*Xstar];
    f = rho*ones(2*k,1) - [Xstar'*ystar; -Xstar'*ystar]; 
    %f = rho*ones(2*k,1) - [(Xstar*Dinv)'*ystar; -(Xstar*Dinv)'*ystar];
    
    
    %% the inequality constraint [-Dinv, Dinv]*[alpha^+; alpha^-] <= zeros(k,1) is actually necessary
    options = optimoptions(@quadprog,'Display','off');
    [alpha,fopt,flag,out,lambda] = quadprog(H, f, [-Dinv, Dinv], zeros(k,1), [diag(Dinv)',-diag(Dinv)'], 1, ...
        zeros(2*k,1), [], [], options);
    beta = Dinv * (alpha(1:k) - alpha(k+1:2*k));
    W0(i,idx(nn)) = beta;
    
    
end

%W0(W0<=sqeps) = 0; % for numerical stability

end