% This function generates K groups of data points. Each group of data
% points come from a linear / affine subspace, and each group contains Nk 
% number of points.

%%% Inputs:
% P: full data dimension. 
% q: subspace dimension. 
% Nk: the number of points in each cluster. 
% K: the total number of clusters. 
% noi: the noise level. 
% type: specify whether to generate data from 'linear' or 'affine'
% subspaces.

%%% Ouputs:
% X; the N by P data generated 
% Truth: the ground truth vectors 

% Last updated: 30 Mar. 2020

function [X, Truth] = GenSubDat(P, q, Nk, K, noi, type)

if nargin < 6
    type = 'linear';
end


%% prepare dataset
X = [];
Truth = [];


%% create the data
for in=1:K
    
    basis=orth(randn(P,q));
    
    if strcmp(type, 'affine')
        disp = randi(10,P,q)*ones(q,Nk);
        X = [X basis*randn(q,Nk) + disp];
    else
        X = [X basis*randn(q,Nk)];
    end
    
	Truth = [Truth in*ones(1,Nk)];
    
end


%% add noise if asked for
X = X + normrnd(0, noi, size(X));
X = X';


end