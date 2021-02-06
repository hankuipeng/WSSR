# Weighted Sparse Simplex Representation (WSSR)

In this repository, we provide the MATLAB implementation for **Weighted Sparse Simplex Representation**. 

We provide two implementations for the full WSSR problem (`WSSR`): a quadratic programming (QP) implementation, and a projected gradient descent implementation (PGD). We also provide an implementation for solving a subproblem of WSSR (`WSSR-LE`) by setting a system of linear equations. 

There are two specifications for each problem: 

- `WSSR_cos.m`, `WSSR_PGD_cos.m` and `WSSR_le_cos.m` use absolute cosine similarity to construct the weight matrix in the case of linear subspaces; 

- `WSSR_euclid.m`, `WSSR_PGD_euclid.m` and `WSSR_le_euclid.m` use Euclidean distances to construct the weight matrix in the case of affine subspaces and manifolds.
