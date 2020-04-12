%% set path and add the main code repo to path
repo_path = pwd;
addpath(genpath(repo_path))

clear repo_path


%% generate the noise-free data first (and keep this base X0 fixed)
P = 3;
q = 1;
Nk = 100;
K = 3;
N = Nk*K;
rng(1)

% if we want to generate data from linear subspaces 
[X0, Truth] = GenSubDat(P, q, Nk, K, 0, 'linear');

% if we want to generate data from affine subspaces 
%[X0, Truth] = GenSubDat(P, q, Nk, K, 0, 'affine');

% add some noise to the data 
noi = 0.15;
X = X0 + normrnd(0, noi, size(X0));


%% plot the data if the data are 2D
for k = 1:K
    scatter(X(Truth==k,1), X(Truth==k,2))
    hold on 
end
hold off


%% plot the data if the data are 3D
for k = 1:K
    scatter3(X(Truth==k,1), X(Truth==k,2), X(Truth==k,3))
    hold on 
end
hold off


%% parameter settings 
knn = 10;
rho = 0.001;
weight = 1; % whether to use the weight matrix or not 
normalize = 1; % whether to normalise the data to have unit length
stretch = 1; % whether to stretch the points to reach the unit sphere
ss = 10;
MaxIter = 50;
thr = 1e-5;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Linear Subspace %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% WSSR_PGD_cos (use absolute cosine similarity)
tic;
[W, obj, obj_mat] = WSSR_PGD_cos(X, knn, rho, normalize, ss, MaxIter, stretch, thr);
time = toc
A = (abs(W) + abs(W'))./2;
grps = SpectralClustering(A, K);
cluster_performance(grps, Truth)

% visualise the changes of objective function values for one point 
i = 10; % pick a point
plot(obj_mat(i,:))


%% WSSR_QP_cos (use absolute cosine similarity)
tic;
[W, obj] = WSSR_QP_cos(X, knn, rho, normalize, stretch, weight);
time = toc
A = (abs(W) + abs(W'))./2;
grps = SpectralClustering(A, K);
cluster_performance(grps, Truth)


%% WSSR_le_cos (use absolute cosine similarity)
tic;
W = WSSR_le_cos(X, knn, rho, normalize, stretch, weight);
time = toc
A = (abs(W) + abs(W'))./2;
grps = SpectralClustering(A, K);
cluster_performance(grps, Truth)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Affine Subspace %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% WSSR_PGD_euclid (use Euclidean distance)
% The Euclidean version of PGD requires a much larger step size than the
% absolute cosine similarity version.
tic;
W = WSSR_PGD_euclid(X, knn, rho, normalize, ss, MaxIter, thr);
time = toc
A = (abs(W) + abs(W'))./2;
grps = SpectralClustering(A, K);
cluster_performance(grps, Truth)


%% WSSR_QP_euclid (use Euclidean distance)
tic;
W = WSSR_QP_euclid(X, knn, rho, normalize, weight);
time = toc
A = (abs(W) + abs(W'))./2;
grps = SpectralClustering(A, K);
cluster_performance(grps, Truth)


%% WSSR_le_euclid (use Euclidean distance)
tic;
W = WSSR_le_euclid(X, knn, rho, normalize, weight);
time = toc
A = (abs(W) + abs(W'))./2;
grps = SpectralClustering(A, K);
cluster_performance(grps, Truth)


%% visualise the similarity matrix
imshow(A*100)


%% visualise the clustering result
hold on
for k = 1:K
    scatter(X(grps == k,1), X(grps == k,2))
end
hold off

