function obj = ObjVal(yopt, Y, beta_cur, D, rho)

epsilon = 1e-4;
partA = sum((yopt-Y*beta_cur).^2);
partB = sum(D*beta_cur);
partC = sum((D*beta_cur).^2);
obj = partA/2 + partB*rho + partC*epsilon/2;

end