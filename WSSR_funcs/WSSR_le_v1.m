function W0 = WSSR_le_v1(X, k, rho, normalize)

N = size(X, 1);

if nargin < 4
    normalize = 1;
end

if normalize == 1
    X0 = X;
    X = norml2(X0, 1);
end


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
    D = diag(1./dk);
    Y = X(idx(nn),:)'; % P x k
    
    %%
    epsilon = 1.0e-4; 
    a = Y'*Y + epsilon.*D'*D;
    b = ones(k, 1);
    
    %%
    A = [a, b; b', 0];
    B = [Y'*yopt-rho*D*b; 1];
    
    %% since n<p we add small ridge penalty to the formulation
    beta = linsolve(A,B);
    %W0(2,idx(nn)) = beta2(1:33);
    
    
    
    W0(i,idx(nn)) = beta(1:k);
    
    
end

end