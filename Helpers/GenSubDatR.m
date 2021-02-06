% A function to generate data that lie in different subspaces, with a 
% specific rotation angle between subspaces.

%%% Inputs:
% P: full dimensionality of the data 
% q: subspace dimensionality 
% Nk: number of points per cluster 
% K: number of clusters 
% noi: level of noise 
% R: the P by P rotation matrix 

%%% Ouputs:
% X; the N by P data generated 
% Truth: the ground truth vectors 

% Last updated: 11 May. 2020

function [X, Truth] = GenSubDatR(P, q, Nk, K, noi, R)


%% generate the first subspace basis and the data 
basis{1} = orth(randn(P,q)); 
X = basis{1}*randn(q,Nk);
Truth = ones(Nk, 1);


%% generate the subspaces and the data for the remaining clusters
for k = 2:K
    
    basis{k} = R*basis{k-1};
    X = [X basis{k}*randn(q,Nk)];
	Truth = [Truth; k*ones(Nk, 1)];
    
end


%% add noise
X = X + normrnd(0, noi, size(X));
X=X';


end