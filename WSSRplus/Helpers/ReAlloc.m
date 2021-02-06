% This function calculates the reconstruction error of allocating data in
% 'dat' to the label 'k'. E.g. 'dat' can be X(class{1},:), and 'k' can 
% be alloc(1)

% Last updated: 3 Oct. 2019

function dist = ReAlloc(dat,k,vec_all,Q,mu)

n = size(dat,1); % number of data objects in dat
Vuse = vec_all{k};
quse = Q(k);

if size(Vuse,2) == 0
    dist = sum(sum((dat-repmat(mu(k,:),n,1)).^2, 2));
else
    Px = (dat-repmat(mu(k,:),n,1))*Vuse(:,1:quse)*Vuse(:,1:quse)' + repmat(mu(k,:),n,1);
    dist=sum(sum((dat-Px).^2,2));
end

