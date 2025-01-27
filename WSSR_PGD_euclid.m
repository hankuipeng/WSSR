% This function solves the Weighted Sparse Simplex Representation (WSSR)
% problem through Projected Gradient Descent (PGD). We first solves the 
% subproblem of WSSR analytically to obtain \beta_0, then we project
% \beta_0 to the probability simplex to obtain \beta_1. We use \beta_1 as
% the initial solution vector to the PGD algorithm. 

%%%% Inputs:
% X: N by P data matrix.
% k: the number of nearest neighbors to consider. 
% rho: the l1 penalty parameter.
% normalize: 1 or 0 depending on whether to normalize the data or not. 
% ss: the initial step size -- we use backtracking line search.
% MaxIter: the maximum number of iterations to run PGD.

%%% Outputs:
% W: the N by N coefficient matrix.
% obj_stars: a vector of length N whosen entries contain the objective
% function values for each point.

% Last edited: 15 Apr. 2020


function [W, obj_stars, obj_mat] = WSSR_PGD_euclid(X, k, rhos, normalize, ss, MaxIter, thr)

N = size(X, 1);
W = zeros(N);
obj_stars = zeros(N ,1);
obj_mat = zeros(N, MaxIter);
epsilon = 1e-4; 
beta = 0.8;
alpha = 0.3;

if nargin < 4
    normalize = 1;
end

if normalize == 1
    X0 = X;
    X = norml2(X0, 1);
end

if length(rhos) == 1
    rhos = ones(N, 1)*rhos;
end

if nargin < 5
    num = 1;
end

if nargin < 6
    MaxIter = 100;
end

if nargin < 7
    thr = 1e-4;
end


%%
for i = 1:N
    
    rho = rhos(i);
    
    %% calculate the Euclidean distances
    yopt = X(i,:)';
    dists_sq = sum((repmat(yopt', N, 1) - X).^2, 2);
    dists = arrayfun(@(x) sqrt(x), dists_sq);
    dists(i) = Inf; % don't choose itself 
    
    [vals, inds]= sort(dists, 'ascend');    
    nn = inds(1:k);
    dk = max(vals(1:k), epsilon);
    D = diag(dk);
    Ynew = X(nn,:)'; % P x k
    
    
    %% solve a system of linear equations for the subproblem
    a = Ynew'*Ynew + epsilon.*D'*D;
    b = ones(k, 1);
    A = [a, b; b', 0];
    B = [Ynew'*yopt-rho*D*b; 1];
    
    beta_le = linsolve(A,B); % solve the system of linear equations
    beta_cur = beta_le(1:k); % \beta_0
    beta_cur = SimplexProj(beta_cur); % \beta_1
    
    
    %% Projected Gradient Descent (PGD) 
    betas = [];
    iter = 1;
    while iter <= MaxIter
        
        % calculate the gradient
        g = -Ynew'*yopt + Ynew'*Ynew*beta_cur + rho.*diag(D) + epsilon.*D'*D*beta_cur;
        
        % gradient update step 
        beta1 = beta_cur - ss.*g;
        
        left = ObjVal(yopt, Ynew, beta1, D, rho);
        right = ObjVal(yopt, Ynew, beta_cur, D, rho) - alpha*ss*norm(g).^2;
        
        % backtracking line search
        while left > right
            ss = beta*ss;
            beta1 = beta_cur - ss.*g;
            left = ObjVal(yopt, Ynew, beta1, D, rho);
            right = ObjVal(yopt, Ynew, beta_cur, D, rho) - 0.5*ss*norm(g).^2;
        end
        
        % gradient update step (using updated step size)
        beta1 = beta_cur - ss.*g;
        
        % projection onto the probability simplex
        beta_cur = SimplexProj(beta1);
        betas(iter,:) = beta_cur;
        
        % calculate the current objective function value
        obj = ObjVal(yopt, Ynew, beta_cur, D, rho);
        obj_mat(i,iter) = obj; % the objective function value over iterations for one point 
        
        % stop when the objective stops decreasing 
         if obj < thr
            obj_mat(i,iter:end) = obj;
            break
        end
        
        iter = iter + 1;
        
    end
    
    [obj_stars(i), id] = min(obj_mat(i,:)); % the vector of objective function values for all points 
    W(i,nn) = betas(id,:);
    
    
end

end