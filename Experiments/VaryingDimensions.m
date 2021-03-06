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


%% generate the data 
P = 20;
q = 2;
Nk = 200;
K = 4;
noi = 0.01;

rng(1)

[X, Truth] = GenSubDat(P, q, Nk, K, noi);


%%%%%%% run the experiments for varying subspace dimension %%%%%%%


%% generate the corresponding data
qs = [2,4,6,8,10,12,14,16,18,20];
l = 8;

NumCores = l;
rng(1)

for i=1:l
    
    q = qs(i);
    [X, Truth] = GenSubDat(P, q, Nk, K, noi);
    Xs{i} = X;
    Truths{i} = Truth;
    
end


%% WSSR-QP 
% WSSR parameter settings
knn = 50;
rho = 0.01;
weight = 1; % whether to use the weight matrix or not 
normalize = 1; % whether to normalise the data to have unit length
stretch = 1; % whether to stretch the points to reach the unit sphere

%
rng(1)
purrs = zeros(l,1); % vector to store purities

parfor (i=1:l, NumCores)
    
    X = Xs{i};
    Truth = Truths{i};
    
    [W, obj] = WSSR_QP_cos(X, knn, rho, normalize, stretch, weight);
    A = (abs(W) + abs(W'))./2;
    grps = SpectralClustering(A, K);
    
    perf = cluster_performance(grps, Truth);
    purrs(i) = perf.Purity;
    
    fprintf('\n Just finished iteration %f \n', i)
    
end

purrs


%% SSR
rho = 0;
normalize = 0;
rng(1)

purrs_ssr = zeros(l,1); % vector to store purities

parfor (i=1:l, NumCores)
    
    X = Xs{i};
    Truth = Truths{i};
    [N, P] = size(X);
    
    W = SSR(X, knn, rho, normalize);
    A = (abs(W) + abs(W'))./2;
    grps = SpectralClustering(A, K);
    
    perf = cluster_performance(grps, Truth);
    purrs_ssr(i) = perf.Purity;
    
    fprintf('\n Just finished iteration %f \n', i)
    
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
rng(1)
purrs_ssc = zeros(l,1); % vector to store purities

parfor (i=1:l, NumCores)
    
    X = Xs{i};
    Truth = Truths{i};
    
    [missrate, grps, W] = SSC(X', r, affine, alpha, outlier, rho, Truth);
    
    perf = cluster_performance(grps, Truth);
    purrs_ssc(i) = perf.Purity;
    
    fprintf('\n Just finished iteration %f \n', i)
    
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
rng(1)
purrs_s3c = zeros(l,1); % vector to store purities

parfor (i=1:l, NumCores)
    
    X = Xs{i};
    Truth = Truths{i};
    
    [acc, grps] = StrSSCplus(X', Truth, opt);  
    
    perf = cluster_performance(grps, Truth);
    purrs_s3c(i) = perf.Purity;
    
    fprintf('\n Just finished iteration %f \n', i)
    
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
rng(1)
purrs_assc = zeros(l,1); % vector to store purities

parfor (i=1:l, NumCores)
    
    X = Xs{i};
    Truth = Truths{i};
    
    [missrate, grps, W] = SSC(X', r, affine, alpha, outlier, rho, Truth);
    
    perf = cluster_performance(grps, Truth);
    purrs_assc(i) = perf.Purity;
    
    fprintf('\n Just finished iteration %f \n', i)
    
end

purrs_assc


%% SSC-OMP
rng(1)
purrs_omp = zeros(l,1); % vector to store purities

parfor (i=1:l, NumCores)
    
    X = Xs{i};
    Truth = Truths{i};
    
    W = OMP_mat_func(X', 10, 1e-6);
    A = abs(W) + abs(W)';
    grps = SpectralClustering(A, K);
    
    perf = cluster_performance(grps, Truth);
    purrs_omp(i) = perf.Purity;
    
    fprintf('\n Just finished iteration %f \n', i)
    
end

purrs_omp


%% LSR
rng(1)
purrs_lsr = zeros(l,1); % vector to store purities

parfor (i=1:l, NumCores)
    
    X = Xs{i};
    Truth = Truths{i};
    
    [A, grps, acc] = SubspaceSegmentation_edit('LSR1', X', Truth, 4.6*1e-3);
    
    perf = cluster_performance(grps, Truth);
    purrs_lsr(i) = perf.Purity;
    
    fprintf('\n Just finished iteration %f \n', i)
    
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
rng(1)
purrs_smr = zeros(l,1); % vector to store purities

parfor (i=1:l, NumCores)
    
    X = Xs{i};
    Truth = Truths{i};
    
    W = smr(X', para);
    A = abs(W) + abs(W)';
    grps = SpectralClustering(A, K);
    
    perf = cluster_performance(grps, Truth);
    purrs_smr(i) = perf.Purity;
    
    fprintf('\n Just finished iteration %f \n', i)
    
end

purrs_smr

