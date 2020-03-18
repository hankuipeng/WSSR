% This function generates K groups of data points. Each group of data
% points come from a linear / affine subspace, and each group contains Nk 
% number of points.


function [X Truth] = GenSubDat(P,q,Nk,K,noi,type)

if nargin < 6
    type = 'linear';
end


%% prepare dataset
X = [];
Truth = [];


%% create the data
for in=1:K
    basis=orth(randn(P,q));
    
    if type == 'affine'
        %X = x + ;
        disp = randi(10,P,1)*ones(q,Nk);
        X=[X basis*randn(q,Nk)+disp];
    else
        X=[X basis*randn(q,Nk)];
    end
	Truth=[Truth in*ones(1,Nk)];
end


%% add noise if asked for
X = X + normrnd(0, noi, size(X));

%% add displacement if data are from affine subspaces



X=X';
Truth = Truth';

end