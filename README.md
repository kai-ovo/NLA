# Numerical Linear Algebra
This repository contains implementations of numerous fundamental and advanced algorithms in Numerical Linear Algebra, including GMRES and its variants, CG and its variants, smoothing methods, Golub-Kahan bidiagonalization, ILU, iChol, etc.

---

 `CG.py` contains the following algorithms:
- Conjugate Gradient (CG)
- Bi-Conjugate Gradient (BiCG)
- Bi-Conjugate Gradient Stabilized (BiCGStab)
- Preconditioned Conjugate Gradient (PCG)
- Conjugate Residual (CR) 

`GMRES.py` contains the following algorithms:
- Generalized Minimal Residual (GMRES)
- Minimum Residual (MINRES)
- Flexible GMRES (FGMRES)
- Restarted GMRES (GMRES(m))
- Quasi-Minimal Residual (QMR) 
- Left Preconditioned GMRES

`golub_kahan` contains the Golub-Kahan bidiagonalization algorithm

`basis.py` contains the following algorithms:
- Lanczos Method
- Arnoldi Method

`preconditioner.py` contains the following preconditioners:
- Incomplete LU (ILU)
- Incomplete Cholesky
- Jacobi Preconditioner
- Gauss-Seidel Preconditioner
- Successive Over-Relaxation (SOR) Preconditioner
- Symmetric Successive Over-Relaxation (SSOR) Preconditioner

`smoothing.py` contains the following algorithms:
- Jacobi Method
- Gauss-Seidel Method
- Successive Over-Relaxation (SOR)
- Symmetric Successive Over-Relaxation (SSOR)
- Richardson Iteration (Damped Jacobi)

`test_cg.py` and `test_gmres.py` contain scripts for testing CG and GMRES on example test cases.

## References

- Golub, Gene H., and Charles F. Van Loan. Matrix computations. JHU press, 2013.
- Saad, Yousef. Iterative methods for sparse linear systems. Society for Industrial and Applied Mathematics, 2003.