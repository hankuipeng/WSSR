% This function constructs a N by N similarity matrix based on an input
% data matrix. It is done by solving N weighted sparse simplex
% representation (WSSR) problems. This is the l1-version of the original
% WSSR problem, in which we consider only the l1 penalty term. 

% Inputs:
% X: N by P data matrix
% k: the number of nearest neighbors to consider 
% rho: the l1 penalty parameter
% normalize: 1 or 0 depending on whether to normalize the data or not 

% Last updated: 20th Nov. 2019

function W0 = WSSR_l1(X, k, rho, normalize)

N = size(X, 1);

if nargin < 4
    normalize = 1;
end

if normalize == 1
    X0 = X;
    X = norml2(X0, 1);
end

% detect if the function is given one rho or a sequence of rhos (one per point)
if length(rho) == 1
    rhos = ones(N,1)*rho;
else
    rhos = rho;
end

sqeps = 1.0e-2; % squared epsilon


%%
W0 = zeros(N);
for i = 1:N
    
    idx = 1:N;
    idx(i) = [];
    
    Xopt = X(idx,:)';
    yopt = X(i,:)';
    
    % appproach 1: identify kNNs
    d = []; % pairwise similarity vector
    for ind=1:size(Xopt,2)
        d(ind) = yopt'*Xopt(:,ind);
    end
    
    
    [val ind]= sort(abs(d), 'descend');
    dk = val(1:k);
    nn = ind(1:k);

    
    %% We need to ensure that D is invertible (this may require more thought)
    dk = max(dk, 1.0e-4); % make sure the similarity values are greater than 0 
    Dinv = diag(dk);

    
    %% since n<p we add small ridge penalty to the formulation
    ystar = yopt;
    Xstar = Xopt(:,nn)*diag(dk);

    
    %% QP for Constrained LASSO
    H = [Xstar'*Xstar, -Xstar'*Xstar; -Xstar'*Xstar, Xstar'*Xstar];
    f = rhos(i)*ones(2*k,1) - [Xstar'*ystar; -Xstar'*ystar]; 
    
    
    %% the inequality constraint [-Dinv, Dinv]*[alpha^+; alpha^-] <= zeros(k,1) is actually necessary
    options = optimoptions(@quadprog,'Display','off');
    [alpha,fopt,flag,out,lambda] = quadprog(H, f, [-Dinv, Dinv], zeros(k,1), [diag(Dinv)',-diag(Dinv)'], 1, ...
        zeros(2*k,1), [], [], options);
                                        
    beta = Dinv * (alpha(1:k) - alpha(k+1:2*k));                                 
    %%
    
    W0(i,idx(nn)) = beta;
    
    
end

W0(W0<=sqeps) = 0;

end