# WSSR

This is a toolbox for Weighted Sparse Simplex Representation (WSSR).

We provide two implementations for the full WSSR problem (`WSSR`): a quadratic programming (QP) implementation, and a projected subgradient descent implementation (PSGD). We also provide an implementation for solving the subproblem of WSSR (`WSSR-LE`) by setting a system of linear equations. 

There are two specifications for each problem: `WSSR_cos.m`, `WSSR_PSGD_cos.m` and `WSSR_le_cos.m` use absolute cosine similarity to construct the weight matrix in the case of linear subspaces; `WSSR_euclid.m`, `WSSR_PSGD_euclid.m` and `WSSR_le_euclid.m` use Euclidean distances to construct the weight matrix in the case of affine subspaces and manifolds.
