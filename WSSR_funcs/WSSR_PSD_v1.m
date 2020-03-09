function [W obj_stars] = WSSR_PSD_v1(X, k, rho, normalize, denom, MaxIter)

% denom: the denominator in the initial step size 1/denom 
% MaxIter: the maximum number of iterations to run projected gradient
% descent for 

% Last edited: 1st Feb. 2020


N = size(X, 1);

if nargin < 4
    normalize = 1;
end

if normalize == 1
    X0 = X;
    X = norml2(X0, 1);
end


%%
W = zeros(N);
obj_stars = [];

for i = 1:N
    
    idx = 1:N;
    idx(i) = [];
    
    x = X(i,:)';
    Y = X(idx,:)';
    
    d = x'*Y;
    
    
    [val ind]= sort(abs(d), 'descend');
    dk = val(1:k);
    nn = ind(1:k);

    
    %% We need to ensure that D is invertible
    dk = max(dk, 1.0e-4); % make sure the similarity values are greater than 0 
    D = diag(1./dk);
    Ynew = X(idx(nn),:)'; % P x k
    
    %%
    epsilon = 1.0e-4; 
    a = Ynew'*Ynew + epsilon.*D'*D;
    b = ones(k, 1);
    
    %% solve the WSSR problem without the non-negativity constraint to obtain
    % an initial solution for \beta
    A = [a, b; b', 0];
    B = [Ynew'*x-rho*D*b; 1];
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
        v = randsample([-1,1], k, true)';
        v(find(beta_cur>0)) = 1;
        v(find(beta_cur>0)) = -1;
        g = -Ynew'*x+Ynew'*Ynew*beta_cur+rho.*D*v+epsilon.*D'*D*beta_cur;
        
        % step 3: calculate the new beta
        beta1 = beta_cur - ss.*g;
        
        % step 4: project beta1 onto the probability simplex
        beta_cur = SimplexProj(beta1);
        betas(:,iter) = beta_cur;
        
        % step 5: recore the current objective function value
        obj_cur = ObjVal(X, i, k, beta_cur, rho);
        objs = [objs; obj_cur];
        
        % calculate the new step size -- using line search
%         if iter>=2 && g'*(betas(:,iter)-betas(:,iter-1))+sum((betas(:,iter)-betas(:,iter-1)).^2)/(2*ss) <= 0
%             ss = ss/diminisher;
%         end
        
%         if length(objs)>1 && objs(length(objs)) - objs(length(objs)-1) < 1e-3
%             break
%         end

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