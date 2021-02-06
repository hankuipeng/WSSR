% WSSR with active learning

% Last updated: 28 May 2020


function [lbs_mat, pur, queried_id] = WSSR_al(X, K, q, asmt, T, b, Truth, rho, knn, strategy, NumCores)

%%% Inputs:
% X: the N by P data matrix. 
% K: number of clusters.
% asmt: the initial cluster membership.
% T: total number of iterations. 
% b: number of points to query in every iteration. 
% Truth: the vector of true classes. 

%%% Outputs:
% lbs_mat: the matrix that contains the cluster assignments for all T
% iterations. It is of size T by N.
% pur: The clustering accuracy over all iterations.
% queried_id: the points that are queried over iterations.


%% initial parameters
epsilon = 1.0e-4; % squared epsilon
N = size(X,1);
P = size(X,2);
lbs_mat = zeros(T,N);
unqueried_id = 1:N;
queried_01 = zeros(N,1);
queried_id = [];
iter = 1;
NMI = zeros(T,1);

if (nargin < 10)
    strategy = 1; % don't do random sampling
end

if (nargin < 11)
    NumCores = 4;
end


%% normalize the data
X0 = X;
X = norml2(X0, 1);


%%
while iter <= T
    
    
    %% stop if all points have been queried
    if length(unqueried_id) == 0
        NMI(iter:T) = NMI(iter-1);
        iter = T+1;
        break
    end
    
    
    %% KSC: update the subspace bases
    parfor (k = 1:K, NumCores)
        
        % obtain zero mean data
        mu(k,:) = mean(X0(asmt==k,:));
        
        % eigen decomposition
        [eigvecs, eigvals_m] = eig(cov(X0(asmt==k,:)));
        eigvals = diag(eigvals_m);
        
        % re-order the eigenvalues and eigenvectors in decreasing order
        ord = P:-1:1;
        val_all{k} = eigvals(ord);
        vec_all{k} = eigvecs(:,ord);
        
    end
    
    clear eigvals_m ord eigvals eigvecs 
    
    
    %% query stage
    if strategy == 0
        querying_id = randsample(unqueried_id, b, false);
    else
        re_diff = InfBoth(X0, asmt, vec_all, val_all, q, mu, NumCores);
        %re_diff = InfAdd(X0, asmt, vec_all, val_all, q, mu, NumCores);
        [~, diff_id] = query0_max(re_diff(unqueried_id), b, asmt(unqueried_id), K);
        querying_id = unqueried_id(diff_id); % the indices of points that we are querying in this iteration
    end    
    
    queried_id = [queried_id(:); querying_id(:)];
    queried_01(querying_id) = 1;
    unqueried_id = find(queried_01==0);
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
        %alpha = 0;
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
%         Xstar = [Xopt*Dinv; sqeps*eye(knn)]; 
%         ystar = [yopt; zeros(knn,1)];
%         H1 = Xstar'*Xstar;
%         H2 = -Xstar'*Xstar;
%         H = [H1, H2; H2, H1];
%         f = rho*ones(2*knn,1) - [Xstar'*ystar; -Xstar'*ystar];
        
        H = Y'*Y+epsilon.*D*D;
        f = rho.*diag(D)-Y'*yopt;
        
        
        %% solve the quadratic programming problem 
%         options = optimoptions(@quadprog,'Display','off');
%         [alpha,fopt,flag,out,lambda] = quadprog(H, f, [-Dinv, Dinv], zeros(knn,1), [diag(Dinv)',-diag(Dinv)'], 1, ...
%             zeros(2*knn,1), [], [], options);
%         beta = Dinv * (alpha(1:knn) - alpha(knn+1:2*knn));
%         A(i,nn) = beta;
%         
        options = optimoptions(@quadprog,'Display','off');
        [beta,fopt,flag,out,lambda] = quadprog(H, f, -eye(knn), zeros(knn,1), ones(1,knn), 1, ...
            zeros(knn,1), [], [], options);
        A(i,nn) = beta;
        
    end

    
    %% obtain the similarity matrix W
    W = (abs(A)+abs(A'))./2;
    W(Psi==exp(1)) = 0; % satisfy the cannot-link constraints 
    W = round(W,4);
    asmt = SpectralClustering(W, K); % spectral clustering
    
    
    %% satisfy the constraints 
    tmp = KSCCq(X0, K, 50, q, asmt, Truth, queried_id, @LabUpdateHun, NumCores);
    asmt = tmp;
    lbs_mat(iter,:) = asmt;
    clear tmp
    
    
    %% cluster performance 
    perf = cluster_performance(asmt,Truth);
    pur(iter) = perf.Purity;
    pur(iter)
    fprintf('rho: %d \n', rho);
    fprintf('Just finished iteration: %d \n', iter);
    
    
%     %% stop if perfection is reached
%     if perf.Purity == 1
%         pur(iter:T) = 1;
%         iter = T+1;
%     end
    
    iter = iter + 1;
    
end

end