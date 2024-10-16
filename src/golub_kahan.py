import numpy as np

def golub_kahan_bidiag(A, m):
    """
    Golub-Kahan Bidiagonalization:
    Computes the bidiagonalization of matrix A using Golub-Kahan method.
    
    Parameters:
    - A: matrix
    - m: Number of bi-diag steps (dim of the Krylov subspace)
    
    Returns:
    - U: Left orthogonal matrix (A ≈ U * B * V^T)
    - B: Bidiagonal matrix
    - V: Right orthogonal matrix
    """
    n, p = A.shape
    U = np.zeros((n, m))
    V = np.zeros((p, m))
    B = np.zeros((m, m))

    beta = 0
    u = np.random.rand(n)
    u = u / np.linalg.norm(u)

    for j in range(m):
        # compute v from u
        v = np.dot(A.T, u)
        if j > 0:
            v = v - beta * V[:, j - 1]
        alpha = np.linalg.norm(v)
        if alpha == 0:
            break
        V[:, j] = v / alpha
        B[j, j] = alpha

        # compute u from v
        u = np.dot(A, V[:, j])
        if j > 0:
            u = u - alpha * U[:, j - 1]
        beta = np.linalg.norm(u)
        if beta == 0:
            break
        U[:, j] = u / beta
        if j < m - 1:
            B[j + 1, j] = beta

    return U[:, :m], B[:m, :m], V[:, :m]
