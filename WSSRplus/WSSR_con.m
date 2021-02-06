% WSSR with constraints

% Last updated: 15 May 2020


function labels = WSSR_con(X, K, asmt, Truth, rho, knn, queried_id, NumCores)

%%% Inputs:
% X: the N by P data matrix. 
% K: number of clusters. 
% asmt: the initial cluster assignment.
% Truth: the vector of true classes. 
% q: subspace dimension.
% rho: l1 penalty parameter.
% knn: the neighbourhood size.
% queried_id: the indices for the points whose class labels are assumed
% known.
% NumCores: the number of cores to use. 


%% initial parameters
epsilon = 1.0e-4; % squared epsilon
N = size(X,1);
% P = size(X,2);
% lbs_mat = zeros(T,N);
% unqueried_id = 1:N;
% queried_01 = zeros(N,1);
% queried_id = [];
% iter = 1;
% NMI = zeros(T,1);


if (nargin < 8)
    NumCores = 8;
end


%% normalize the data
X0 = X;
X = norml2(X0, 1);
queryT = Truth(queried_id); % the true classes of queried points


%% update the affinity matrix (with queried info incorporated)
Psi = ones(N,N);
for k=1:K

    % indices that satisfy the must-link constraints
    currset = queried_id(queryT==k);

    % indices that satisfy the cannot-link constraints
    Nind = ~ismember(queried_id, currset);
    Nset = queried_id(Nind);

    for i=1:length(currset)
        Psi(currset(i), currset) = exp(-1); % must-link
        Psi(currset(i), Nset) = exp(1); % cannot-link 
    end

end


%% form the segmentation matrix Q
Q = zeros(N, K);
for k = 1:K
    Q(find(asmt==k), k) = 1;
end


%% solve the optimisation program
A = zeros(N);
for i = 1:N

    yopt = X(i,:)';

    %% update D to incorporate the must-link and cannot-link constraints 
    % prior information 
    sims = abs(X*yopt); % similarity vector 
    sims = max(sims, 1.0e-4); % ensure that there is no similarity value 0

    % cluster assignment information 
    q0_diff = Q*Q(i,:)';
    q_diff = zeros(size(q0_diff));
    q_diff(q0_diff==0) = 1;

    % combining everything together 
    alpha = length(queried_id)/N; % the amount of belief on the cluster assignment 
    d = Psi(:,i)./sims(:) + alpha.*q_diff;
    d(i) = []; % don't choose itself 

    [Fval, Find] = sort(abs(d), 'ascend');
    dk = Fval(1:knn);
    nn = Find(1:knn);

    idx = 1:N;
    idx(i) = [];
    Xopt = X(idx(nn),:)';
    D = diag(dk);

    % stretch the points to reach the sphere 
    Xst = Xopt;
    Ts = 1./(yopt'*Xst);
    Xst = Xst*diag(Ts);
    Y = Xst;


    %% formulate the optimisation program 
    H = Y'*Y+epsilon.*D*D;
    f = rho.*diag(D)-Y'*yopt;


    %% solve the quadratic programming problem 
    options = optimoptions(@quadprog,'Display','off');
    [beta,fopt,flag,out,lambda] = quadprog(H, f, -eye(knn), zeros(knn,1), ones(1,knn), 1, ...
        zeros(knn,1), [], [], options);
    A(i,nn) = beta;

end


%% obtain the similarity matrix W
W = (abs(A)+abs(A'))./2;
W(Psi==exp(1)) = 0; % satisfy the cannot-link constraints 
%W = round(W,4);
asmt = SpectralClustering(W, K); % spectral clustering


%% satisfy the constraints 
labels = asmt;
%labels = KSCCq(X0, K, 50, q, asmt, Truth, queried_id, @LabUpdateHun, NumCores);

end