% This function solves the Weighted Sparse Simplex Representation (WSSR)
% problem through Projected Gradient Descent (PGD). We first solves the 
% subproblem of WSSR analytically to obtain \beta_0, then we project
% \beta_0 to the probability simplex to obtain \beta_1. We use \beta_1 as
% the initial solution vector to the PGD algorithm. 

% Last edited: 15 Apr. 2020


function [W, obj_star, obj_mat] = WSSR_PGD_cos(X, k, rho, normalize, ss, MaxIter, stretch, thr)


%%% Inputs:
% X: the N by P data matrix.
% k: the number of nearest neighbours.
% rho: the penalty parameter on the l1 norm of the WSSR objective.
% normalize: 1 or 0, whether we normalize the data to unit length or not. 
% ss: initial step size -- we use backtracking line search.
% MaxIter: the maximum number of iterations to run PGD.
% stretch: whether to stretch the data points or not. 

%%% Outputs:
% W: the N by N coefficient matrix.
% obj_stars: a vector of length N whosen entries contain the objective
% function values for each point.
% obj_mat: an N by MaxIter matrix that stores the objective function values over all
% iterations for all points.


if nargin < 4
    normalize = 1;
end

if normalize == 1
    X0 = X;
    X = norml2(X0, 1);
end

if nargin < 5
    num = 1;
end

if nargin < 6
    MaxIter = 100;
end

if nargin < 7
    stretch = 1;
end

if nargin < 8
    thr = 1e-4;
end

N = size(X, 1);
W = zeros(N);
obj_mat = zeros(N ,MaxIter);
obj_star = zeros(N, 1);
epsilon = 1e-4;
beta = 0.8;
alpha = 0.3;


%%
for i = 1:N
    
    %% We remove any zero cosine similarities
    idx = 1:N;
    idx(i) = [];
    
    Xopt = X(idx,:)';
    yopt = X(i,:)';
   
    % calculate the cosine similarities
    sims = abs(yopt'*Xopt);
    
    if sum(sims <= 1e-4) ~= 0 
        ind = find(sims >= 1e-4);
        sims = sims(ind);
        idx = idx(ind);
    end
    
    
    %% sort the similarity values in descending order 
    [vals, inds]= sort(abs(sims), 'descend'); % absolute cosine similarity values 
    
    if k == 0 % consider only the positive similarity values 
        dk = vals(vals > epsilon);
        nn = inds(vals > epsilon);
        k = length(dk);
    else
        if k > length(inds)
            dk = vals;
            nn = inds;
            k = length(inds);
        else
            dk = vals(1:k);
            nn = inds(1:k);
        end
    end
    
    D = diag(1./dk);
    Y = X(idx(nn),:)';
    
    
    %% stretch the data points that will be considered in the program
    if stretch == 1
        Xst = Y;
        Ts = 1./(yopt'*Xst);
        Xst = Xst*diag(Ts);
        Y = Xst;
    end
    
    
    %% solve a system of linear equations for the subproblem
    a = Y'*Y + epsilon.*D'*D;
    b = ones(k, 1);
    A = [a, b; b', 0];
    B = [Y'*yopt-rho*D*b; 1];
    
    beta_le = linsolve(A,B); % solve the system of linear equations
    beta_cur = beta_le(1:k); % \beta_0
    beta_cur = SimplexProj(beta_cur);
    
    
    %% Projected Gradient Descent (PGD) 
    betas = [];
    iter = 1;
    
    while iter <= MaxIter
        
        % calculate the gradient
        g = -Y'*yopt + Y'*Y*beta_cur + rho.*diag(D) + epsilon.*D'*D*beta_cur;
        
        % gradient update step
        beta1 = beta_cur - ss.*g;
        
        left = ObjVal(yopt, Y, beta1, D, rho);
        right = ObjVal(yopt, Y, beta_cur, D, rho) - alpha*ss*norm(g).^2;
        
        % backtracking line search
        while left > right
            ss = beta*ss;
            beta1 = beta_cur - ss.*g;
            left = ObjVal(yopt, Y, beta1, D, rho);
            right = ObjVal(yopt, Y, beta_cur, D, rho) - 0.5*ss*norm(g).^2;
        end
        
        % gradient update step (using updated step size)
        beta1 = beta_cur - ss.*g;
        
        % project \beta onto the probability simplex
        beta_cur = SimplexProj(beta1);
        betas(iter,:) = beta_cur;
        
        % calculate the current objective function value
        obj = ObjVal(yopt, Y, beta_cur, D, rho);
        obj_mat(i,iter) = obj; % the objective function value over iterations for one point 
        
        if obj < thr
            obj_mat(i,iter:end) = obj;
            break
        end
        
        iter = iter + 1;
        
    end
    
    obj_star(i) = min(obj_mat(i,:));
    [~, id] = min(obj_mat(i,:));
    W(i,idx(nn)) = betas(id,:);
    
end

end
