% This function normalize the row/column of a matrix to have unit length

function y = norml2(x,ind)

if ind == 1 % normalize each row in 'before' 
    x = x';
    scale = diag(1./sqrt(sum(abs(x).^2)));
    y = (x*scale)';
end

if ind == 2 % normalize each column in 'before'
    scale = diag(1./sqrt(sum(abs(x).^2)));
    y = x*scale;
end

end
