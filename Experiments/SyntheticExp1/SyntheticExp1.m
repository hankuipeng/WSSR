%% set path and add the main code repo to path
cd ../.. % go to the main code repo
repo_path = pwd;
addpath(genpath(repo_path))

clear repo_path


%% load existing data examples 
load('Example1_26Mar.mat')


%% or generate some data 
P = 10;
q = 5;
Nk = 100;
K = 3;
N = Nk*K;

% make sure the results are reproducible
rng(1)

% if we want to generate data from linear subspaces 
[X0, Truth] = GenSubDat(P, q, Nk, K, 0, 'linear');

% if we want to generate data from affine subspaces 
%[X0, Truth] = GenSubDat(P, q, Nk, K, 0, 'affine');

% add some noise to the data 
noi = 0.2;
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
knn = 20;
rho = 0.001;
weight = 1; % whether to use the weight matrix or not 
normalize = 1; % whether to normalise the data to have unit length
stretch = 1; % whether to stretch the points to reach the unit sphere


%% obtain the objective function values from WSSR_QP
tic;
[W1, objs1] = WSSR_QP_cos(X, knn, rho, normalize, stretch, weight);
time1 = toc;
A1 = (abs(W1) + abs(W1'))./2;
grps = SpectralClustering(A1, K);
cluster_performance(grps, Truth)


%% additional parameters for WSSR_PGD
ss = .5; % one part of the denominator of the step size
MaxIter = 100;
thr = 1e-5;


%% obtain the objective function values from WSSR_PGD
tic;
[W2, objs2, obj_mat] = WSSR_PGD_cos(X, knn, rho, normalize, ss, MaxIter, stretch, thr);
time2 = toc;
A2 = (abs(W2) + abs(W2'))./2;
grps = SpectralClustering(A2, K);
cluster_performance(grps, Truth)


%% check the change of objective function values over iterations (for one point)
i = 6;
plot(obj_mat(i,:))


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
i = 15; % pick a point 

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
title('PGD')
