% This function is a variant of WSSR_le.m. In this function, the entries in
% D are calculated using the absolute cosine similarity value. 

% Last updated: 18 Mar. 2020

function W0 = WSSR_le_cos(X, k, rho, normalize, stretch, weight)

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

if nargin < 6
    weight = 1; % whether we use cosine similarity (1) to create D or not (0)
end

if length(rho) == 1
    rhos = ones(N,1)*rho;
else
    rhos = rho;
end


%%
epsilon = 1.0e-4; 
W0 = zeros(N);

for i = 1:N
    
    rho = rhos(i);
    
    idx = 1:N;
    idx(i) = [];
    
    Xopt = X(idx,:)';
    yopt = X(i,:)';
   
    
    %% calculate the cosine similarities
    sims = yopt'*Xopt;
    
    if sum(sims <= 1e-4) ~= 0 
        ind = find(sims >= 1e-4);
        sims = sims(ind);
        idx = idx(ind);
    end
    
    [vals inds]= sort(abs(sims), 'descend');
    %[vals inds]= sort(sims, 'descend');
    
    
    %%
    if k == 0 % consider only the positive similarity values 
        dk = vals(find(vals>0));
        nn = inds(find(vals>0));
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

    
    %% We need to ensure that D is invertible
    if weight == 1 
        D = diag(1./dk);
    else
        D = diag(ones(1, length(dk)));
    end
    
    Y = X(idx(nn),:)'; % P x k
    
    
    %% stretch the data points that will be considered in the program
    if stretch
        Xst = Y;
        Ts = 1./(yopt'*Xst);
        Xst = Xst*diag(Ts);
        Y = Xst;
    end
    
    a = Y'*Y + epsilon.*D'*D;
    b = ones(length(dk), 1);
    
    
    %%
    A = [a, b; b', 0];
    B = [Y'*yopt-rho*D*b; 1];
    
    
    %% since n<p we add small ridge penalty to the formulation
    beta = linsolve(A,B);
    W0(i,idx(nn)) = beta(1:length(dk));
    
    
end

end