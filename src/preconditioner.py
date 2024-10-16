import numpy as np
import scipy.sparse as sp

def ilu(A):
    """
    Incomplete LU (ILU):
    Computes the ILU of A using the ikj variant of Gaussian elimination.
    
    Parameters:
    - A: matrix 
    
    Returns:
    - L: Lower triangular matrix from ILU factorization
    - U: Upper triangular matrix from ILU factorization
    """
    A = A.copy()
    n = A.shape[0]
    
    for i in range(n):
        row_indices = np.nonzero(A[i, :])[0]
        nzr = len(row_indices)
        p = 0
        while p < nzr and row_indices[p] < i:
            k = row_indices[p]
            A[i, k] = A[i, k] / A[k, k]
            piv = A[i, k]
            for j in range(p + 1, nzr):
                A[i, row_indices[j]] = A[i, row_indices[j]] - piv * A[k, row_indices[j]]
            p += 1
    
    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)
    
    return L, U

def incomplete_cholesky(A):
    """
    Incomplete Cholesky Factorization
    
    Parameters:
    - A: SPD matrix
    
    Returns:
    - L: Lower triangular matrix from iChol (A \approx L * L.T)
    """
    A = A.copy()
    n = A.shape[0]
    
    L = np.zeros_like(A)
    
    for i in range(n):
        L[i, i] = np.sqrt(A[i, i])
        for j in range(i + 1, n):
            if A[i, j] != 0:  
                L[j, i] = A[j, i] / L[i, i]
                A[j, j] = A[j, j] - L[j, i] ** 2
    
    return L


def jacobi_precond(A):
    """
    Jacobi Preconditioner:
    
    """
    D = np.diag(A)  
    M_inv = np.diag(1.0 / D) 
    return M_inv

def gauss_seidel_precond(A):
    """
    Gauss-Seidel Preconditioner
    """
    n = A.shape[0]
    M = np.tril(A) 
    M_inv = np.linalg.inv(M)  
    return M_inv


def sor_precond(A, omega=1.0):
    """
    SOR preconditioner
    """
    n = A.shape[0]
    M = np.tril(A) / omega  
    M_inv = np.linalg.inv(M)  
    return M_inv


def ssor_precond(A, omega=1.0):
    """
    SSOR Preconditioner:
    """
    n = A.shape[0]
    M_forward = np.tril(A) / omega  
    M_backward = np.triu(A) / omega 
    M = M_forward + M_backward - np.diag(np.diag(A))  
    M_inv = np.linalg.inv(M)  
    return M_inv
