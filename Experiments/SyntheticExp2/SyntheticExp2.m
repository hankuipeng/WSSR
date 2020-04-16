%% set path and add the main code repo to path
cd ../.. % go to the main code repo
repo_path = pwd;
addpath(genpath(repo_path))

clear repo_path


%% load existing data examples 
load('edge_case_dat.mat')


%% or generate some data 
P = 5;
q = 3;
Nk = 100;
K = 3;
N = Nk*K;

% make sure the results are reproducible
rng(K)

% if we want to generate data from affine subspaces 
[X0, Truth] = GenSubDat(P, q, Nk, K, 0, 'affine');

% add some noise to the data 
noi = 0.3;
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


%% parameters for QP
knn = 10;
rho = 0.01;
weight = 1; % whether to use the weight matrix or not 
normalize = 0; % whether to normalise the data to have unit length


%% WSSR -- QP
tic;
[W1, objs1] = WSSR_QP_euclid(X, knn, rho, normalize, weight);
time1 = toc;
A1 = (abs(W1) + abs(W1'))./2;
grps = SpectralClustering(A1, K);
cluster_performance(grps, Truth)


%% additional parameters for PGD
ss = 10;
MaxIter = 500;


%% WSSR -- PGD
tic;

% use fixed step size 
% [W2, objs2, obj_mat2] = WSSR_PGD_euclid_fixed(X, knn, rho, normalize, ss, MaxIter);

% use backtracking line search to determine step size 
[W2, objs2, obj_mat2] = WSSR_PGD_euclid(X, knn, rho, normalize, ss, MaxIter);

time2 = toc;

A2 = (abs(W2) + abs(W2'))./2;
grps = SpectralClustering(A2, K);
cluster_performance(grps, Truth)

% plot the change of objective function values for one point
i = 4; % pick a point 
plot(obj_mat2(i,:))


%% 1a. compare the objective function values of these two 
plot(1:N, objs1)
hold on 
plot(1:N, objs2)
hold on
legend('QP', 'PGD')
hold off


%% 1b. plot the difference in the objective function values 
plot(objs1-objs2)


%% 2. compare the solution vectors of these two 
i = 1; % pick a point 

vals_qp_i = W1(i, W1(i,:) >= 1e-4)
inds_qp_i = find(W1(i,:) >= 1e-4)

vals_pgd_i = W2(i, W2(i,:) >= 1e-4)
inds_pgd_i = find(W2(i,:) >= 1e-4)


%% 3. compare the similarity matrix of these two
figure;
subplot(1,2,1)
imshow(A1*100)
title('QP')
subplot(1,2,2)
imshow(A2*100)
title('PGD')
