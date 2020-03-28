%% set path and add the main code repo to path
cd ../.. % go to the main code repo
repo_path = pwd;
addpath(genpath(repo_path))


%% load existing data examples 
load('Example3_27Mar.mat')


%% or generate some data 
P = 3;
q = 1;
Nk = 100;
K = 3;
N = Nk*K;

% make sure the results are reproducible
rng(1)

% if we want to generate data from linear subspaces 
[X0, Truth] = GenSubDat(P, q, Nk, K, 0, 'linear');

% if we want to generate data from affine subspaces 
%[X0 Truth] = GenSubDat(P, q, Nk, K, 0, 'affine');

% add some noise to the data 
noi = 0.001;
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


%% parameter settings for WSSR_QP
knn = 10;
rho = 0.01;
weight = 1; % whether to use the weight matrix or not 
normalize = 1; % whether to normalise the data to have unit length
stretch = 1; % whether to stretch the points to reach the unit sphere


%% obtain the objective function values from WSSR_QP
[W1, objs1] = WSSR_cos(X, knn, rho, normalize, stretch, weight);
A1 = (abs(W1) + abs(W1'))./2;
A1(A1<=1e-4) = 0; 
grps = SpectralClustering(A1, K);
cluster_performance(grps, Truth)


%% additional parameters for WSSR_PSGD
denom = 10; % one part of the denominator of the step size
MaxIter = 100;


%% obtain the objective function values from WSSR_PSGD
[W2, objs2] = WSSR_PSGD_cos(X, knn, rho, normalize, denom, MaxIter, stretch);
A2 = (abs(W2) + abs(W2'))./2;
A2(A2<=1e-4) = 0;
grps = SpectralClustering(A2, K);
cluster_performance(grps, Truth)


%% 1. compare the objective function values of these two 
plot(1:N, objs1)
hold on 
plot(1:N, objs2)
hold on
legend('QP', 'PSGD')
hold off


%% 2. compare the solution vectors of these two 
i = 5; % pick a point 

vals_qp_i = W1(i, W1(i,:) >= 1e-4)
inds_qp_i = find(W1(i,:) >= 1e-4)

vals_psgd_i = W2(i, W2(i,:) >= 1e-4)
inds_psgd_i = find(W2(i,:) >= 1e-4)


%% 3. compare the similarity matrix of these two
figure;
subplot(1,2,1)
imshow(A1*100)
title('QP')
subplot(1,2,2)
imshow(A2*100)
title('PSGD')
