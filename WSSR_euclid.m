function W0 = WSSR_euclid(X, k, rho, normalize, weight)

N = size(X, 1);
sqeps = 1.0e-2; % square root of epsilon

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
    
    
end

end