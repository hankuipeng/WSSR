% The changes we made in this version (based on WSSR_le_v1.m): 
% (a) we remove x_{j}s if x_{i}^{T}x_{j} = 0,
% (b) we also order the cosine similarities instead of the absolute values.
% (3) we allow for an additional option for an equal weighting D

% Last updated: 4 Mar. 2020

function W0 = WSSR_le(X, k, rho, normalize, stretch, weight, euclid)

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

if nargin < 7 % whether euclidean is used or not
    euclid = 0;
end

if length(rho) == 1
    rhos = ones(N,1)*rho;
else
    rhos = rho;
end


%%
W0 = zeros(N);
for i = 1:N
    
    rho = rhos(i);
    
    idx = 1:N;
    idx(i) = [];
    
    Xopt = X(idx,:)';
    yopt = X(i,:)';
   
    
    %% calculate the cosine similarities
    if euclid == 1
        dists = [];
        for ii = 1:size(Xopt, 2)
            dists(ii) = sqrt(sum((yopt-Xopt(:,ii)).^2));
        end
        
        %%
        [vals inds]= sort(dists, 'ascend');
    else
        sims = yopt'*Xopt;
        if sum(sims == 0) ~= 0
            idx = idx(find(sims~=0));
            sims = sims(find(sims~=0));
        end
        
        [vals inds]= sort(abs(sims), 'descend');
        %[vals inds]= sort(sims, 'descend');
    end
    
    
    %%
    if k == 0 % consider only the positive similarity values 
        dk = vals(find(vals>0));
        nn = inds(find(vals>0));
        k = length(dk);
    else
        if k > length(vals) % if some zero entries have been removed from sims
            dk = vals;
            nn = inds;
        else
            dk = vals(1:k);
            nn = inds(1:k);
        end
    end
    
    %dk = val(1:k);
    %nn = ind(1:k);

    
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
    
    
    %%
    %epsilon = 1;
    epsilon = 1.0e-4; 
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