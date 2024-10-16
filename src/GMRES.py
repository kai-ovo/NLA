import numpy as np
from basis import arnoldi

import numpy as np
from basis import arnoldi

"""
variants of GMRES for solving Ax = b

these algorithms can be merged into a big one, but I coded them individually
to get myself more familiar
"""

def gmres(A, b, x0=None, tol=1e-10, max_iter=None, restart=None):
    """
    Generalized Minimal Residual (GMRES) Method:

    Inputs:
    - A, b: set up Ax = b
    - x0: Initial guess for the solution (default is a zero vector)
    - tol: rel. err. tolerance
    - max_iter: Maximum number of iterations (default is the size of A)
    - restart: Number of iterations after which the process restarts (None means no restart)

    Outputs:
    - x: Approximate solution vector
    - rel_errors: rel. err. history
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = n
    if restart is None:
        restart = max_iter

    x = x0
    rel_errors = []

    for _ in range(0, max_iter, restart):
        r = b - np.dot(A, x)
        beta = np.linalg.norm(r)
        
        if beta < tol:
            return x, rel_errors
        
        V = np.zeros((n, restart + 1))
        H = np.zeros((restart + 1, restart))
        V[:, 0] = r / beta
        g = np.zeros(restart + 1)
        g[0] = beta
        
        for j in range(restart):
            if np.linalg.norm(V[:, j]) < tol:
                return x, rel_errors

            V[:, :j + 1], H[:j + 2, :j + 1] = arnoldi(A, V[:, j], j + 1)
            
            e1 = np.zeros(j + 2)
            e1[0] = beta
            y = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)[0]
            x = x + np.dot(V[:, :j + 1], y)
            
            r = b - np.dot(A, x)
            rel_error = np.linalg.norm(r) / np.linalg.norm(b)
            rel_errors.append(rel_error)

            if rel_error < tol:
                return x, rel_errors

    return x, rel_errors


def minres(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    MINRES (Minimum Residual) Method
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = n

    x = x0
    r = b - np.dot(A, x)
    v_old = np.zeros_like(r)
    v = r / np.linalg.norm(r)
    beta_old = 0
    w = np.zeros_like(r)
    alpha_old = 0
    rel_errors = []

    for i in range(max_iter):
        v_new = np.dot(A, v)
        alpha = np.dot(v.T, v_new)
        v_new = v_new - alpha * v - beta_old * v_old
        beta = np.linalg.norm(v_new)
        v_old = v
        v = v_new / beta
        
        if i == 0:
            w = v
        else:
            w_new = (v - alpha_old * w) / beta_old
            w = w_new

        x = x + np.dot(r.T, w) * v

        rel_error = np.linalg.norm(b - np.dot(A, x)) / np.linalg.norm(b)
        rel_errors.append(rel_error)
        
        if rel_error < tol:
            break

        alpha_old = alpha
        beta_old = beta

    return x, rel_errors




def fgmres(A, b, M_inv_list, x0=None, tol=1e-10, max_iter=None, restart=None):
    """
    Flexible GMRES (FGMRES) Method:
    
    inputs:
    - M_inv_list: List of preconditioners, one for each iteration 
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = n
    if restart is None:
        restart = max_iter
        
    x = x0
    r = b - np.dot(A, x)
    beta = np.linalg.norm(r)
    V = np.zeros((n, restart + 1))
    H = np.zeros((restart + 1, restart))
    rel_errors = []

    for cycle in range(0, max_iter, restart):
        if beta < tol:
            return x, rel_errors
        
        V[:, 0] = r / beta
        g = np.zeros(restart + 1)
        g[0] = beta
        
        for j in range(restart):
            M_inv = M_inv_list[cycle + j]
            z = np.dot(M_inv, V[:, j])

            if np.linalg.norm(z) < tol:
                return x, rel_errors
            
            V[:, :j+1], H[:j+2, :j+1] = arnoldi(A, z, j + 1)
            
            y = np.linalg.lstsq(H[:j + 2, :j + 1], g[:j + 2], rcond=None)[0]
            x = x + np.dot(V[:, :j + 1], y)

            r = b - np.dot(A, x)
            rel_error = np.linalg.norm(r) / np.linalg.norm(b)
            rel_errors.append(rel_error)
            
            if rel_error < tol:
                return x, rel_errors

    return x, rel_errors


def restarted_gmres(A, b, x0=None, tol=1e-10, max_iter=None, restart=20):
    """
    Restarted GMRES (GMRES(m)):

    Inputs:
    - restart: Number of iterations after which the method is restarted

    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = n

    x = x0
    rel_errors = []

    for _ in range(0, max_iter, restart):
        r = b - np.dot(A, x)
        beta = np.linalg.norm(r)

        if beta < tol:
            return x, rel_errors
        
        V = np.zeros((n, restart + 1))
        H = np.zeros((restart + 1, restart))

        V[:, 0] = r / beta
        g = np.zeros(restart + 1)
        g[0] = beta

        for j in range(restart):
            if np.linalg.norm(V[:, j]) < tol:
                return x, rel_errors

            V[:, :j + 1], H[:j + 2, :j + 1] = arnoldi(A, V[:, j], j + 1)

            y = np.linalg.lstsq(H[:j + 2, :j + 1], g[:j + 2], rcond=None)[0]
            x = x + np.dot(V[:, :j + 1], y)

            r = b - np.dot(A, x)
            rel_error = np.linalg.norm(r) / np.linalg.norm(b)
            rel_errors.append(rel_error)

            if rel_error < tol:
                return x, rel_errors

    return x, rel_errors


def qmr(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    Quasi-Minimal Residual (QMR) Method
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = n

    x = x0
    r = b - np.dot(A, x)
    r_tilde = r.copy()
    p = r
    p_tilde = r_tilde

    v = np.dot(A, p)
    w = v.copy()
    
    rho_old = np.dot(r.T, r_tilde)
    eta_old = np.linalg.norm(r)
    rel_errors = []

    for i in range(max_iter):
        v = np.dot(A, p)
        alpha = rho_old / np.dot(p_tilde.T, v)

        x = x + alpha * p
        r = r - alpha * v

        rel_error = np.linalg.norm(r) / np.linalg.norm(b)
        rel_errors.append(rel_error)
        
        if rel_error < tol:
            break

        rho_new = np.dot(r.T, r_tilde)
        beta = rho_new / rho_old
        p = r + beta * p

        rho_old = rho_new

    return x, rel_errors

def left_preconditioned_gmres(A, b, M_inv, x0=None, tol=1e-10, max_iter=None):
    """
    Left Preconditioned GMRES
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = n

    b_preconditioned = np.dot(M_inv, b)
    x = x0
    r = b_preconditioned - np.dot(np.dot(M_inv, A), x)
    rel_errors = []

    beta = np.linalg.norm(r)

    if beta < tol:
        return x, rel_errors

    V = np.zeros((n, max_iter + 1))
    H = np.zeros((max_iter + 1, max_iter))
    
    V[:, 0] = r / beta
    g = np.zeros(max_iter + 1)
    g[0] = beta
    
    for j in range(max_iter):
        if np.linalg.norm(V[:, j]) < tol:
            return x, rel_errors

        V[:, :j + 1], H[:j + 2, :j + 1] = arnoldi(np.dot(M_inv, A), V[:, j], j + 1)

        e1 = np.zeros(j + 2)
        e1[0] = beta
        y = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)[0]
        x = x + np.dot(V[:, :j + 1], y)

        r = b_preconditioned - np.dot(np.dot(M_inv, A), x)
        rel_error = np.linalg.norm(r) / np.linalg.norm(b_preconditioned)
        rel_errors.append(rel_error)
        
        if rel_error < tol:
            return x, rel_errors

    return x, rel_errors
