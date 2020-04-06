% This function solves the Weighted Sparse Simplex Representation (WSSR)
% problem through Projected Gradient Descent (PGD). We first solves the 
% subproblem of WSSR analytically to obtain \beta_0, then we project
% \beta_0 to the probability simplex to obtain \beta_1. We use \beta_1 as
% the initial solution vector to the PGD algorithm. 

% Last edited: 6 Apr. 2020


function [W, obj_stars] = WSSR_PGD_cos(X, k, rho, normalize, denom, MaxIter, stretch, thr)


%%% Inputs:
% X: the N by P data matrix.
% k: the number of nearest neighbours.
% rho: the penalty parameter on the l1 norm of the WSSR objective.
% normalize: 1 or 0, whether we normalize the data to unit length or not. 
% denom: the step size parameter (in the denominator part).
% MaxIter: the maximum number of iterations to run PGD.
% stretch: whether to stretch the data points or not. 

%%% Outputs:
% W: the N by N coefficient matrix.
% obj_stars: a vector of length N whosen entries contain the objective
% function values for each point.


if nargin < 4
    normalize = 1;
end

if normalize == 1
    X0 = X;
    X = norml2(X0, 1);
end

if nargin < 5
    denom = 1;
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
obj_stars = zeros(N ,1);
epsilon = 1e-4; 


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
    if stretch
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
    beta_cur = SimplexProj(beta_cur); % \beta_1
    
    
    %% Projected Gradient Descent (PGD) 
    objs = [];
    betas = [];
    for iter = 1:MaxIter
        
        % step1: calculate the current step size (diminishing step sizes)
        % I adopted the step size rule in 'sungradient methods stanford
        % notes' from: https://web.stanford.edu/class/ee392o/subgrad_method.pdf
        ss = 1/(denom + iter);
        
        % step 2: calculate the gradient
        g = -Y'*yopt + Y'*Y*beta_cur + rho.*diag(D) + epsilon.*D'*D*beta_cur;
        
        % step 3: gradient update step 
        beta1 = beta_cur - ss.*g;
        
        % step 4: projection onto the probability simplex
        beta_cur = SimplexProj(beta1);
        betas(iter,:) = beta_cur;
        
        % step 5: record the current objective function value
        partA = sum((yopt-Y*beta_cur).^2);
        partB = sum(D*beta_cur);
        partC = sum((D*beta_cur).^2);
        obj_cur = partA/2 + partB*rho + partC*epsilon/2;
        
        objs = [objs; obj_cur]; % the objective function value over iterations for one point 
        
        % stop when the objective stops decreasing 
        if iter > 1 && abs(objs(iter) - objs(iter-1)) <= thr
            break
        end
        
    end
    
    [obj_stars(i), id] = min(objs); % the vector of objective function values for all points 
    W(i,idx(nn)) = betas(id,:);
    
    
end

end
