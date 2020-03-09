% This function is an adaptation of WSSR_le.m. It still solves the problem
% through a system of linear equations. The only difference being how the
% weight matrix D is calculated. Previously we used cosine similarity, but
% here Euclidean distance is adopted instead.

% Suitable for data with manifold structure.

% Last updated: 9 Mar. 2020

function W0 = WSSR_le_euclid(X, k, rho, weight)

N = size(X, 1);

if length(rho) == 1
    rhos = ones(N,1)*rho;
else
    rhos = rho;
end

if nargin < 3
    weight = 1;
end


%%
W0 = zeros(N);
for i = 1:N
    
    rho = rhos(i);
    
    idx = 1:N;
    idx(i) = [];
    
    Xopt = X(idx,:)';
    yopt = X(i,:)';
    
    %sims = yopt'*Xopt;
    dists = [];
    for ii = 1:size(Xopt, 2)
        dists(ii) = sqrt(sum((yopt-Xopt(:,ii)).^2));
    end
    
    %%
    [vals inds]= sort(dists, 'ascend');
    %[vals inds]= sort(sims, 'descend');
    
    nn = inds(1:k);
    dk = vals(1:k);
    
    if weight == 1
        D = diag(dk);
    else
        D = diag(ones(length(dk),1));
    end
    
    
    %%
    Y = X(idx(nn),:)'; % P x k
    epsilon = 1.0e-4; 
    a = Y'*Y + epsilon.*D'*D;
    b = ones(k, 1);
    
    
    %%
    A = [a, b; b', 0];
    B = [Y'*yopt-rho*D*b; 1];
    
    
    %% since n<p we add small ridge penalty to the formulation
    beta = linsolve(A,B);
    W0(i,idx(nn)) = beta(1:k);
    
    
end

end