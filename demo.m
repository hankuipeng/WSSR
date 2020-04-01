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


%% visualize the data (if 2-D data are generated)
scatter(X(1:Nk,1),X(1:Nk,2))
hold on 
scatter(X((Nk+1):(Nk*2),1),X((Nk+1):(Nk*2),2))
hold on
scatter(X((Nk*2+1):(Nk*3),1),X((Nk*2+1):(Nk*3),2))
hold off 


%% visualize the data (if 3-D data are generated) 
scatter3(X(1:Nk,1),X(1:Nk,2),X(1:Nk,3))
hold on
scatter3(X((Nk+1):(Nk*2),1),X((Nk+1):(Nk*2),2),X((Nk+1):(Nk*2),3))
hold on
scatter3(X((Nk*2+1):(Nk*3),1),X((Nk*2+1):(Nk*3),2),X((Nk*2+1):(Nk*3),3))
hold off 


%% parameter settings 
knn = 10;
rho = 0.001;
weight = 1; % whether to use the weight matrix or not 
normalize = 1; % whether to normalise the data to have unit length
stretch = 0; % whether to stretch the points to reach the unit sphere
denom = 5;
MaxIter = 500;


%% WSSR_PGD_cos
tic;
W = WSSR_PGD_cos(X, knn, rho, normalize, denom, MaxIter, stretch);
time = toc
A = (abs(W) + abs(W'))./2;
grps = SpectralClustering(A, K);
cluster_performance(grps, Truth)


%% WSSR_cos (using cosine similarity -- for linear subspace)
tic;
W = WSSR_cos(X, knn, rho, normalize, stretch, weight);
time = toc
A = (abs(W) + abs(W'))./2;
grps = SpectralClustering(A, K);
cluster_performance(grps, Truth)


%% WSSR-LE (using cosine similarity -- for linear subspace)
tic;
W = WSSR_le_cos(X, knn, rho, normalize, stretch, weight);
time = toc
A = (abs(W) + abs(W'))./2;
grps = SpectralClustering(A, K);
cluster_performance(grps, Truth)


%% WSSR_PGD_euclid
tic;
W = WSSR_PGD_euclid(X, knn, rho, normalize, denom, MaxIter);
time = toc
A = (abs(W) + abs(W'))./2;
grps = SpectralClustering(A, K);
cluster_performance(grps, Truth)


%% WSSR_euclid (using cosine similarity -- for affine subspace)
tic;
W = WSSR_euclid(X, knn, rho, normalize, weight);
time = toc
A = (abs(W) + abs(W'))./2;
grps = SpectralClustering(A, K);
cluster_performance(grps, Truth)


%% WSSR-LE (using Euclidean distance -- for affine subspace)
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

