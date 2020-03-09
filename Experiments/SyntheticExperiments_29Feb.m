%% import paths
acsr_path = '/home/hankui/Dropbox/Clustering/WorkLog/By_Algorithm/ACSR';
addpath(genpath(acsr_path))

toolbox = '/home/hankui/Dropbox/Clustering/WorkLog/By_Language/MATLAB/MyToolBox';
addpath(genpath(toolbox))

ssc_path = '/home/hankui/Dropbox/Clustering/WorkLog/By_Algorithm/SSC_ADMM';
addpath(genpath(ssc_path))

clear acsr_path toolbox ssc_path


%% generate the noise-free data first (and keep this base X0 fixed)
P = 3;
q = 1;
Nk = 100;
K = 4;
N = Nk*K;
[X0 Truth] = GenSubDat(P, q, Nk, K, 0);


%% add some noise to the data 
noi = 0.4;
X = X0 + normrnd(0, noi, size(X0));


%% normalise the data
X_noNormalise = X;
X = norml2(X_noNormalise, 1);


%% visualize the data 
scatter3(X(1:Nk,1),X(1:Nk,2),X(1:Nk,3))
hold on
scatter3(X((Nk+1):(Nk*2),1),X((Nk+1):(Nk*2),2),X((Nk+1):(Nk*2),3))
hold on
scatter3(X((Nk*2+1):(Nk*3),1),X((Nk*2+1):(Nk*3),2),X((Nk*2+1):(Nk*3),3))
hold off 


%% SSC
tic
alpha = 20; r = 0; affine = false; outlier = false; rho = 1;
[missrate, lbs_ssc, C] = SSC(X', r, affine, alpha, outlier, rho, Truth);
%A = abs(C+C')./2;
%lbs_ssc = SpectralClustering(A,K);
perf_ssc = cluster_performance(lbs_ssc, Truth)
time_ssc = toc


%% comparison of all three methods 
tic
knn = round(N/(K)); % number of nearest neighbours to consider
rho = 0.001;

%W = WSSR(X, knn, rho, 1, 1); % QP
%W = WSSR_le(X, knn, rho, 1, 1); % solving system of linear equations
W = WSSR_le_euclid(X, knn, rho); % solving system of linear equations, with Euclidean distances
%[W obj_stars] = WSSR_PSD(X, knn, rho, 1, 50, 10, 0); % projected subgradient descent
%[W obj_stars] = WSSR_PSD_v2(X, knn, rho, 0, .01, 100); % PSD with backtracking line search
%[W obj_stars] = WSSR_APSD(X, knn, rho, 0, 100, 10); % accelerated projected subgradient descent 
A = abs(W + W')./2;

% spectral clustering approach 1
grps = SpectralClustering(A,K);
wssr_perf = cluster_performance(grps, Truth)
time = toc

% calculate the missrate
grps = bestMap(Truth,grps);
missrate = sum(Truth(:) ~= grps(:)) / length(Truth)

%%
time_wssr_qp = time;