import numpy as np

def lanczos(A, v, m):
    """
    Lanczos Method:
    
    Parameters:
    - A: Symmetric matrix (must be SPD)
    - v: non-zero initialization
    - m: Number of Lanczos iterations (dim of Krylov subspace)
    
    Returns:
    - Q: Orthonormal basis for the Krylov subspace (n x m matrix)
    - T: Tridiagonal matrix (m x m matrix)
    """
    n = A.shape[0]
    Q = np.zeros((n, m))
    T = np.zeros((m, m))
    beta = 0
    q = v / np.linalg.norm(v)
    Q[:, 0] = q

    for j in range(m):
        w = np.dot(A, q)
        if j > 0:
            w = w - beta * Q[:, j - 1]
        alpha = np.dot(q.T, w)
        w = w - alpha * q

        beta = np.linalg.norm(w)
        if beta < 1e-10:
            T[j, j] = alpha
            break
        
        q = w / beta

        T[j, j] = alpha
        if j < m - 1:
            T[j, j + 1] = beta
            T[j + 1, j] = beta
        
        if j < m - 1:
            Q[:, j + 1] = q

    return Q, T

def arnoldi(A, v, m):
    """
    Arnoldi Iteration:
    
    Inputs:
    - A: matrix (can be non-symmetric)
    - v: Initial vector (must be non-zero)
    - m: Number of Arnoldi iterations (dimension of the Krylov subspace)
    
    Outputs:
    - Q: Orthonormal basis for the Krylov subspace
    - H: Upper Hessenberg matrix
    """
    n = A.shape[0]
    Q = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))

    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:  # Prevent division by zero or small norms
        raise ValueError("Input vector v has near-zero norm, \
                         cannot perform Arnoldi iteration.")
    
    q = v / norm_v
    Q[:, 0] = q

    for k in range(m):
        w = np.dot(A, Q[:, k])
        for j in range(k + 1):
            H[j, k] = np.dot(Q[:, j].T, w)
            w = w - H[j, k] * Q[:, j]
        
        H[k + 1, k] = np.linalg.norm(w)
        if H[k + 1, k] != 0:
            Q[:, k + 1] = w / H[k + 1, k]
    
    return Q[:, :m], H[:m, :]
