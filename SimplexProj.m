function out = SimplexProj(in)

% in: the input vector, which we want to project onto the probability
% simplex

u = sort(in, 'descend');

new_u = [];
for j=1:length(in)
    new_u(j) = u(j) + (1-sum(u(1:j)))/j;
end

rho_proj = sum(new_u>0);
lambda = (1-sum(u(1:rho_proj)))/rho_proj;

out = in + lambda;
out(find(out <= 0)) = 0;

end
