% The active strategy in this function queries entries in a similarity
% matrix.

% Last edited: 18 Mar. 2020


function querying_id = PairwiseQuery(A, grps, queried_id, b)


%%% Inputs:
% A: the current pairwise similarity matrix
% grps: the current cluster assignment 
% queried_id: the points that have been queried so far 
% b: the number of similarity entries to query


%% basic parameters
K = length(unique(grps));
N = length(grps);


%% step 1: form the block diagonal assignment matrix 
AMat = zeros(N, N); % the block-diagonal zero-one matrix indicating the assignment 
Gl = 0; % a vector that indicates the number of points in each assigned cluster 
Pind = []; % a sequence of point indices ordered by assignd clusters

for k = 1:K
    Gind = find(grps==k); % points that are assigned to cluster k
    Pind = [Pind(:); Gind(:)];
    Gl = [Gl length(Gind)];
    idx = sum(Gl(1:k))+1 : sum(Gl(1:k+1)); % row and column indices to be set to 1s
    AMat(idx,idx) = 1;
end

% re-order the similarity matrix to make the entries match the
% block-diagonal 0-1 assignment matrix AMat.
NewW = A(Pind,Pind);

% make it symmetric
NewW = max(NewW,NewW');

clear Gind idx


%% step 2: identify those non-zero off-diagonal entries from NewW
rr = [];
cc = [];

for k = 1:(K-1) % there are K-1 blocks to tally from
    
    % the row indices for the k-th block
    idx = sum(Gl(1:k))+1 : sum(Gl(1:k+1)); 
    
    % the row and column indices of the non-zero entries within the block
    [rind, cind] = find(NewW(idx,(max(idx)+1):N) ~= 0);
    
    % the row and column indices of the non-zero entries in NewW
    rr = [rr; sum(Gl(1:k)) + rind(:)]; % row indices 
    cc = [cc; max(idx) + cind(:)]; % column indices 
    
end

% get the corresponding similarity values (Wij) for those off-diagonal 
% non-zero entries
Wvals = [];
for i = 1:length(rr)
    Wvals(i) = NewW(rr(i), cc(i));
end

clear idx rind cind


%% step 3: match the indices (of the non-zero entries in NewW) 
% with their similarity values 

% create the match table
match_tb = zeros(length(rr)*2, 2);
match_tb(:,1) = [rr(:); cc(:)];
match_tb(:,2) = [Wvals(:); Wvals(:)];

% sort the similarity values in descending order  
[~, sind] = sort(match_tb(:,2), 'descend'); 
match_tb1 = match_tb(sind,:);

% keep the first unique value (for point indices)
[~, ia, ~] = unique(match_tb1(:,1)); 
match_tb2 = match_tb1(ia,:);

% reorder the match table again 
[~, sind] = sort(match_tb2(:,2),'descend'); 
match_tb3 = match_tb2(sind,:);

% remove the queried points 
logi_idx = ismember(Pind(match_tb3(:,1)), queried_id); 
match_tb4 = match_tb3(logi_idx==0,:);

clear sind match_tb match_tb1 ia match_tb2 match_tb3 logi_idx


%% step 4: determine the indices of (b*2) points to query 

% if the number of points corresponding to non-zero off-diagonal 
% similarity entries is more than the query budget
if size(match_tb4, 1) >= b*2 
    
    querying_id = Pind(match_tb4(1:b*2, 1));

% if that's not the case 
else 
    
    % (a) first query points that correspond to all the remaining 
    % off-diagonal entries
    part1_l = size(match_tb4, 1);
    querying_id1 = Pind(match_tb4(1:part1_l, 1));
    newb = b*2 - part1_l; % the number of points left to query 
    
    % (b) then query the zero diagonal entries to strengthen the 
    % block-diagonal structure 
    rr = [];
    cc = [];
    NewW = NewW - diag(diag(NewW)) - eye(N);
    for k = 1:K
        idx = sum(Gl(1:k))+1 : sum(Gl(1:k+1));
        [rind, cind] = find(NewW(idx,idx) == 0);
        
        rr = [rr; sum(Gl(1:k)) + rind]; % row point indices
        cc = [cc; sum(Gl(1:k)) + cind]; % column point indices
    end
    
    Wvals = [];
    for i = 1:length(rr)
        Wvals(i) = NewW(rr(i), cc(i));
    end
    
    match_tb = zeros( length(rr)*2,2);
    match_tb(:,1) = [rr(:); cc(:)];
    match_tb(:,2) = [Wvals(:); Wvals(:)];
    
    [sval sind] = sort(match_tb(:,2),'descend'); % sort the W values in descending order
    match_tb1 = match_tb(sind,:);
    
    [val ia ib] = unique(match_tb1(:,1)); % keep the first unique value
    match_tb2 = match_tb1(ia,:);
    
    [sval sind] = sort(match_tb2(:,2),'descend'); % reorder it again
    match_tb3 = match_tb2(sind,:);
    
    logi_idx = ismember(Pind(match_tb3(:,1)), queried_id); % remove the queried points
    match_tb4 = match_tb3(logi_idx==0,:);
    
    if newb > size(match_tb4,1)
        querying_id2 = Pind(match_tb4(:,1));
    else
        querying_id2 = Pind(match_tb4(1:newb,1));
    end
    
    querying_id = [querying_id1(:); querying_id2(:)];
    
end

end