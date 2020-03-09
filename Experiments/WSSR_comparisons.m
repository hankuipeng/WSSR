
% This script originally comes from work_0130.m
% This script provides a performance and time comparison of the different
% versions / implementations of WSSR.

%% import paths
acsr_path = '/home/hankui/Dropbox/Clustering/WorkLog/By_Algorithm/ACSR';
addpath(genpath(acsr_path))

toolbox = '/home/hankui/Dropbox/Clustering/WorkLog/By_Language/MATLAB/MyToolBox';
addpath(genpath(toolbox))

clear acsr_path toolbox 


%% generate the noise-free data first (and keep this base X0 fixed)
P = 20;
q = 5;
Nk = 100;
K = 4;
N = Nk*K;
[X0 Truth] = GenSubDat(P, q, Nk, K, 0);

% add some noise to the data 
noi = 0.2;
X = X0 + normrnd(0, noi, size(X0));


%% plot 3D data -- only if 3D data are generated 
scatter3(X(1:200,1),X(1:200,2),X(1:200,3))
hold on
scatter3(X(201:400,1),X(201:400,2),X(201:400,3))
hold on
scatter3(X(401:600,1),X(401:600,2),X(401:600,3))
hold off


%% plot 2D data -- only if 2D data are generated 
scatter(X(1:Nk,1),X(1:Nk,2))
hold on
scatter(X((Nk+1):(Nk*2),1),X((Nk+1):(Nk*2),2))
hold on
% scatter(X(401:600,1),X(401:600,2))
% hold off


%% weighted SSR (WSSR) -- QP solver
tic
knn = round(N/(K)); % number of nearest neighbours to consider
rho = 0.001;

W = WSSR(X, knn, rho, 1); 
A = (W + W')./2;

% spectral clustering approach 1
grps = SpectralClustering(A,K);
wssr_perf = cluster_performance(grps, Truth)
time = toc


%% objective function value from QP solver for the first point 

% fist obtain W0 by running WSSR without the non-negativity constraint --
% WSSR_QP_i
i = 1;
out_QP = ObjVal(X, i, k, W0(1,nn)', rho)


%% WSSR without the non-negativity constraint 
tic
knn = round(N/(K)); % number of nearest neighbours to consider
rho = 0.001;

W = WSSR_le(X, knn, rho, 0); 
A = abs(W + W')./2;

% spectral clustering approach 1
grps = SpectralClustering(A,K);
wssr_perf = cluster_performance(grps, Truth)
time2 = toc

out_LinSys = ObjVal(X, i, knn, W(1,nn)', rho)


%% now try projected gradient descent (for point 1)
epsilon = 1.0e-4;
idx = 1:N;
i = 1;
idx(i) = [];

x = X(i,:)';
Y = X(idx,:)';

d = x'*Y;
[val ind]= sort(abs(d), 'descend');
dk = val(1:k);
nn = ind(1:k);
dk = max(dk, epsilon); % make sure the similarity values are greater than 0
D = diag(1./dk);
Ynew = Y(:,nn);

objs = out_LinSys; % the vector that stores the objective function values


%% step 1: choose the starting point (the current beta)
beta_cur = W(1,nn)'; % the solution that we obtained from solving the linear system

% step 2: choose a step size -- diminishing step size
iter = 1;
ss = 1/(1000+iter); 

% step 3: calculate the subgradient 
v = randsample([-1,1], k, true)';
v(find(beta_cur>0)) = 1;
v(find(beta_cur>0)) = -1;
g = -Ynew'*x+Ynew'*Ynew*beta_cur+rho.*D*v+epsilon.*D'*D*beta_cur;

% step 4: calculate the new beta
beta1 = beta_cur - ss.*g;

% step 5: project beta1 onto the probability simplex 
beta_new = SimplexProj(beta1);

% step 6: recore the current objective function value 
obj_cur = ObjVal(X, i, k, beta_new, rho);
objs = [objs; obj_cur];


%% put the above in a loop
MaxIter = 100;
beta_cur = W(1,nn)';

for iter = 1:MaxIter
    
    % step1: calculate the current step size 
    ss = 1/(1000+iter); 
    
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
    
end

[val ind] = min(objs)


%% comparison of all three methods 
tic
knn = round(N/(K)); % number of nearest neighbours to consider
rho = 0.001;

%W = WSSR(X, knn, rho, 1, 1); % QP
%W = WSSR_le(X, knn, rho, 1, 0); % solving system of linear equations
[W obj_stars] = WSSR_PSD(X, knn, rho, 1, 20, 150, 1); % projected subgradient descent
%[W obj_stars] = WSSR_PSD_v2(X, knn, rho, 0, .01, 100); % PSD with backtracking line search
%[W obj_stars] = WSSR_APSD(X, knn, rho, 0, 100, 10); % accelerated projected subgradient descent 
A = abs(W + W')./2;

% spectral clustering approach 1
grps = SpectralClustering(A,K);
wssr_perf = cluster_performance(grps, Truth)
time = toc
%imshow(W*100)

% projected gradient descent can be just as good, but we need to be careful
% with the step size and the number of iterations 


%%
[W objs] = WSSR_PSD_i(X, knn, 2, rho, 0, 10, 100);
plot(objs)