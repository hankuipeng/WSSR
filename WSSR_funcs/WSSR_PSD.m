function [W obj_stars] = WSSR_PSD(X, k, rho, normalize, denom, MaxIter, stretch)

% denom: the denominator in the initial step size 1/denom 
% MaxIter: the maximum number of iterations to run projected gradient
% descent for 

% Last edited: 20 Feb. 2020


N = size(X, 1);

if nargin < 4
    normalize = 1;
end

if normalize == 1
    X0 = X;
    X = norml2(X0, 1);
end

if nargin < 5
    denom = 50;
end

if nargin < 6
    MaxIter = 100;
end

if nargin < 7
    stretch = 0;
end


%%
W = zeros(N);
obj_stars = [];

for i = 1:N
    
    idx = 1:N;
    idx(i) = [];
    
    yopt = X(i,:)';
    Xopt = X(idx,:)';
    
    sims = yopt'*Xopt;
    
    
    %% We remove any zero cosine similarities
    if sum(sims == 0) ~= 0 
        idx = idx(find(sims~=0));
        sims = sims(find(sims~=0));
    end
    
    
    %% sort the similarity values in descending order 
    [vals, inds]= sort(abs(sims), 'descend'); % absolute cosine similarity values 
    %[vals, inds]= sort(sims, 'descend'); % cosine similarity values 
    
    if k == 0 % consider only the positive similarity values 
        dk = vals(find(vals>0));
        nn = inds(find(vals>0));
        k = length(dk);
    else
        dk = vals(1:k);
        nn = inds(1:k);
    end
 
    
    %% We need to ensure that D is invertible
    D = diag(1./dk);
    Dinv = diag(dk);
    Ynew = X(idx(nn),:)'; % P x k
    
    
    %% stretch the data points that will be considered in the program
    if stretch
        Xst = Ynew;
        Ts = 1./(yopt'*Xst);
        Xst = Xst*diag(Ts);
        Ynew = Xst;
    end
    
    
    %%
    epsilon = 1.0e-4; 
    a = Ynew'*Ynew + epsilon.*D'*D;
    b = ones(k, 1);
    
    
    %% solve the WSSR problem without the non-negativity constraint to obtain
    % an initial solution for \beta
    A = [a, b; b', 0];
    B = [Ynew'*yopt-rho*D*b; 1];
    % since n<p we add small ridge penalty to the formulation
    beta_le = linsolve(A,B); % this is our initial value for projected subgradient descent
    beta_cur = beta_le(1:k); % ignore the lambda values obtained
    
    
    %% now let's start projected subgradient descent 
    objs = [];
    for iter = 1:MaxIter
        
        % step1: calculate the current step size (diminishing step sizes)
        % I adopted the step size rule in 'sungradient methods stanford
        % notes' from: https://web.stanford.edu/class/ee392o/subgrad_method.pdf
        ss = 1/(denom + iter);
        
        % step 2: calculate the subgradient
        %v = randsample([-1,1], k, true)';
        v = rho.*Dinv*(Ynew'*yopt-(Ynew'*Ynew+epsilon.*D'*D)*beta_cur);
        v(find(beta_cur>0)) = 1;
        v(find(beta_cur>0)) = -1;
        g = -Ynew'*yopt+Ynew'*Ynew*beta_cur+rho.*D*v+epsilon.*D'*D*beta_cur;
        
        % step 3: calculate the new beta
        beta1 = beta_cur - ss.*g;
        
        % step 4: project beta1 onto the probability simplex
        beta_cur = SimplexProj(beta1);
        betas(:,iter) = beta_cur;
        
        % step 5: recore the current objective function value
        obj_cur = ObjVal(X, i, k, beta_cur, rho);
        objs = [objs; obj_cur];
        
        % stop when the objective stops decreasing 
        if iter > 1 && abs(objs(iter) - objs(iter-1)) <= 0.01
            break
        end
        
    end
    
    [obj_stars(i) ind]=min(objs);
    beta_best = betas(:,ind);
    W(i,idx(nn)) = beta_best;
    
    
end

end