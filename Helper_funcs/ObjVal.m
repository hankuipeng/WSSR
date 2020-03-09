function out = ObjVal(X, i, k, beta, rho)

% i: the i-th point to be regressed over
% k: the number of nearest neighbours to consider
% rho: the penalty parameter

N = size(X,1);
idx = 1:N;
idx(i) = [];

x = X(i,:)';
Y = X(idx,:)';

% identify kNNs
d = x'*Y;
[val ind]= sort(abs(d), 'descend');
dk = val(1:k);
nn = ind(1:k);
dk = max(dk, 1.0e-4); % make sure the similarity values are greater than 0
D = diag(1./dk);
Ynew = Y(:,nn);

partA = sum((x-Ynew*beta).^2);
partB = sum(D*beta);
partC = sum((D*beta).^2);

epsilon = 1.0e-4; 
out = partA/2 + partB*rho + partC*epsilon/2;


end