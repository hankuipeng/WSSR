% /home/hankui/Dropbox/WorkLog/By_Algorithm/WSSR/Experiments/SyntheticExperiments

%% add necessary paths
wssr_pa = '/home/hankui/Dropbox/WorkLog/By_Algorithm/WSSR/WSSR_clean';
addpath(genpath(wssr_pa))

ssr_pa = '/home/hankui/Dropbox/WorkLog/By_Algorithm/SSR';
addpath(genpath(ssr_pa))

ssc_pa = '/home/hankui/Dropbox/WorkLog/By_Algorithm/SSC_ADMM';
addpath(genpath(ssc_pa))

omp_pa = '/home/hankui/Dropbox/WorkLog/By_Algorithm/SSCOMP_Code';
addpath(genpath(omp_pa))

lsr_pa = '/home/hankui/Dropbox/WorkLog/By_Algorithm/LSR-master';
addpath(genpath(lsr_pa))

fgnsc_pa = '/home/hankui/Dropbox/WorkLog/By_Algorithm/FGNSC-master';
addpath(genpath(fgnsc_pa))

clear wssr_pa ssr_pa ssc_pa omp_pa lsr_pa fgnsc_pa 


%% data parameters
P = 3;
q = 2;
Nk = 200;
K = 2;


%% generate the data (for playing)
% noi = 0.01;
% 
% rng(1)
% R = RotMatrix(degtorad(60), [1 1 0]); % rotation in 3 dimensions
% %R = RotMatrix(degtorad(45)); % rotation in 2 dimensions
% 
% [X, Truth] = GenSubDatR(P, q, Nk, K, noi, R);


%% load the generated data (noise-free) 
% it has two clusters: one of them is two-dimensional, another is
% one-dimensional
load('VaryingNoiseDat_0.mat');
X0 = X;
noi = 0;
X = X0 + normrnd(0, noi, size(X0));


%% plot the data if the data are 3D
for k = 1:K
    scatter3(X(Truth==k,1), X(Truth==k,2), X(Truth==k,3))
    hold on 
end
% set(gca,'visible','off')
hold off


%%%%%%% run the experiments for varying noise %%%%%%%


%% WSSR-QP 
% WSSR parameter settings
knn = 10;
rho = 0.01;
weight = 1; % whether to use the weight matrix or not 
normalize = 1; % whether to normalise the data to have unit length
stretch = 1; % whether to stretch the points to reach the unit sphere

%
purrs = zeros(6,1); % vector to store purities
for i=1:6
    
    rng(i)
    noi = (i-1)*0.1;
    X = X0 + normrnd(0, noi, size(X));    
    
    [W, obj] = WSSR_QP_cos(X, knn, rho, normalize, stretch, weight);
    A = (abs(W) + abs(W'))./2;
    grps = SpectralClustering(A, K);
    
    perf = cluster_performance(grps, Truth);
    purrs(i) = perf.Purity;
    
end

purrs


%% SSR
rho = 0;
normalize = 0;

purrs_ssr = zeros(6,1); % vector to store purities

for i=1:6
    
    rng(i)
    noi = (i-1)*0.1;
    X = X0 + normrnd(0, noi, size(X));    
    [N, P] = size(X);
    
    W = SSR(X, knn, rho, normalize);
    A = (abs(W) + abs(W'))./2;
    grps = SpectralClustering(A, K);
    
    perf = cluster_performance(grps, Truth);
    purrs_ssr(i) = perf.Purity;
    
end

purrs_ssr


%% SSC
% parameters for SSC
alpha = 20; 
r = 0; 
affine = false; 
outlier = false; 
rho = 1;

%
purrs_ssc = zeros(6,1); % vector to store purities

for i=1:6
    
    rng(i)
    noi = (i-1)*0.1;
    X = X0 + normrnd(0, noi, size(X)); 
    
    [missrate, grps, W] = SSC(X', r, affine, alpha, outlier, rho, Truth);
    
    perf = cluster_performance(grps, Truth);
    purrs_ssc(i) = perf.Purity;
    
end

purrs_ssc


%% S3C
% add path
cs3c_pa = '/home/hankui/Dropbox/Clustering/WorkLog/By_Algorithm/softS3C+CS3C_ver1.01';
addpath(genpath(cs3c_pa))

% S3C parameters
opt.sc_method ='softStrSSC2';

lambda0 = 10;
gamma0 = 0.02;
opt.affine = 0;
opt.outliers = 1;
opt.T = 1; % T=1 for spectral clustering and T>2 for spectral assumble clustering
opt.iter_max = 10;
opt.nu = 1;
opt.r = 0;

opt.gamma0 = gamma0; % This is for reweighting the off-diagonal entries in Z
opt.lambda = lambda0;

opt.mu_max = 1e8;
opt.tol = 1e-5;
opt.rho = 1.1;
opt.maxIter = 150;

opt.error_norm = 'L1'; 
opt.SSCrho = 1; 
opt.DEBUG = 0;

%
purrs_s3c = zeros(6,1); % vector to store purities

for i=1:6
    
    rng(i)
    noi = (i-1)*0.1;
    X = X0 + normrnd(0, noi, size(X)); 
    
    [acc, grps, Theta, W] = StrSSCplus(X', Truth, opt);  
    
    perf = cluster_performance(grps, Truth);
    purrs_s3c(i) = perf.Purity;
    
end

purrs_s3c

rmpath(cs3c_pa)


%% ASSC
% parameters for ASSC
alpha = 20; 
r = 0; 
affine = true; 
outlier = false; 
rho = 1;

%
purrs_ssc = zeros(6,1); % vector to store purities

for i=1:6
    
    rng(i)
    
    noi = (i-1)*0.1;
    X = X0 + normrnd(0, noi, size(X)); 
    
    [missrate, grps, W] = SSC(X', r, affine, alpha, outlier, rho, Truth);
    
    perf = cluster_performance(grps, Truth);
    purrs_assc(i) = perf.Purity;
    
end

purrs_assc


%% SSC-OMP
purrs_omp = zeros(6,1); % vector to store purities

for i=1:6
    
    rng(i)
    
    noi = (i-1)*0.1;
    X = X0 + normrnd(0, noi, size(X)); 
    
    W = OMP_mat_func(X', 10, 1e-6);
    A = abs(W) + abs(W)';
    grps = SpectralClustering(A, K);
    
    perf = cluster_performance(grps, Truth);
    purrs_omp(i) = perf.Purity;
    
end

purrs_omp


%% LSR
purrs_lsr = zeros(6,1); % vector to store purities

for i=1:6
    
    rng(i)
    
    noi = (i-1)*0.1;
    X = X0 + normrnd(0, noi, size(X)); 
    
    [A, grps, acc] = SubspaceSegmentation_edit('LSR1', X', Truth, 4.6*1e-3);
    
    perf = cluster_performance(grps, Truth);
    purrs_lsr(i) = perf.Purity;
    
end

purrs_lsr


%% SMR
% SMR parameter settings 
para.gamma = 5;
para.knn = 4;
para.elpson = 0.001;
para.alpha = 2.^-16;
para.aff_type = 'J1';

%
purrs_smr = zeros(6,1); % vector to store purities

for i=1:6
    
    rng(i)
    
    noi = (i-1)*0.1;
    X = X0 + normrnd(0, noi, size(X)); 
    
    W = smr(X', para);
    A = abs(W) + abs(W)';
    grps = SpectralClustering(A, K);
    
    perf = cluster_performance(grps, Truth);
    purrs_smr(i) = perf.Purity;
    
end

purrs_smr


%% FGNSC
% FGNSC parameters
num_closer = 8;
num_judge = 20;

%
rng(1)
purrs_fgnsc = zeros(6,1); % vector to store purities

for i=1:6
    
    noi = (i-1)*0.1;
    X = X0 + normrnd(0, noi, size(X)); 
    
    W0 = smr(X', para);
    
    % FGNSC
    [S_weight, S_number, S_weight_max, S_max] = choose_value(W0, num_closer, num_judge);
    S_number = S_number(:,2:size(S_number,2));
    S_max = S_max(:,2:size(S_max,2));

    [R_,R] = omp(X, S_number, S_max, S_weight_max, num_closer-1);
    if size(R,2)==size(X,2)-1
        R(:,size(X,2))=zeros(size(X,2),1);
        R(size(X,2),size(X,2)-1)=1;
    end
    R(1:N+1:end) = 0;
    A = (abs(R) + abs(R)')./2;
    grps = SpectralClustering(A, K);
    
    perf = cluster_performance(grps, Truth);
    purrs_fgnsc(i) = perf.Purity;
    
end

purrs_fgnsc


%% visualise the affinity matrix
A = (abs(W) + abs(W'))./2;
imshow(A)

grps = SpectralClustering(A, K);
cluster_performance(grps, Truth)

width  = 600; % Width of figure
height = 400; % Height of figure (by default in pixels)

truesize([600 400])
imshow(A*500)




