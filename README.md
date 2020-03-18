# WSSR

This is a toolbox for Weighted Sparse Simplex Representation.

We provide a quadratic programming (QP) implementation for the full WSSR problem (`WSSR`), and an implementation by setting a system of linear equations for the sub WSSR problem (`WSSR-LE`). There are two specifications for each problem: `WSSR_cos.m` and `WSSR_le_cos.m` use absolute cosine similarity to construct the weight matrix in the case of linear subspaces; `WSSR_euclid.m` and `WSSR_le_euclid.m` use Euclidean distances to construct the weight matrix in the case of affine subspaces and manifolds.
