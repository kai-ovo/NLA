import numpy as np

""" 
variants of the conjugate gradient algorithm to solve Ax = b
                                                      A is SPD
"""
def cg(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    Conjugate Gradient (CG) Method:

    Inputs:
    - A, b: set up Ax = b; A is SPD
    - x0: Initial guess for the solution (default is a zero vector)
    - tol: Convergence tolerance
    - max_iter: Maximum number of iterations (default is the size of A)

    Outputs:
    - x: Approximate solution
    - errors: rel. err. history
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = n

    x = x0
    r = b - np.dot(A, x)
    p = r
    rs_old = np.dot(r.T, r)
    errors = []

    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rs_old / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r.T, r)
        rel_error = np.sqrt(rs_new) / np.linalg.norm(b)
        errors.append(rel_error)
        if rel_error < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, errors


def bicg(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    Bi-Conjugate Gradient (BiCG) Method
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = n

    x = x0
    r = b - np.dot(A, x)
    r_hat = r.copy()
    p = r
    p_hat = r_hat
    rs_old = np.dot(r.T, r_hat)
    errors = []

    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rs_old / np.dot(p_hat.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r.T, r_hat)
        rel_error = np.sqrt(rs_new) / np.linalg.norm(b)
        errors.append(rel_error)
        if rel_error < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        p_hat = r_hat + beta * p_hat
        rs_old = rs_new

    return x, errors


def bicgstab(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    Bi-Conjugate Gradient Stabilized (BiCGStab) Method
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = n

    x = x0
    r = b - np.dot(A, x)
    r_hat = r.copy()
    p = r
    rs_old = np.dot(r.T, r_hat)
    errors = []
    
    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rs_old / np.dot(r_hat.T, Ap)
        s = r - alpha * Ap
        As = np.dot(A, s)
        omega = np.dot(As.T, s) / np.dot(As.T, As)
        x = x + alpha * p + omega * s
        r = s - omega * As
        rs_new = np.dot(r.T, r_hat)
        rel_error = np.sqrt(rs_new) / np.linalg.norm(b)
        errors.append(rel_error)
        if rel_error < tol:
            break
        beta = (rs_new / rs_old) * (alpha / omega)
        p = r + beta * (p - omega * Ap)
        rs_old = rs_new

    return x, errors


def pcg(A, b, M_inv, x0=None, tol=1e-10, max_iter=None):
    """
    Preconditioned Conjugate Gradient (PCG) Method:

    Inputs:
    - M_inv: Preconditioner such as iChol, Gauss-Seidel, etc.
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = n

    x = x0
    r = b - np.dot(A, x)
    z = np.dot(M_inv, r)
    p = z
    rs_old = np.dot(r.T, z)
    errors = []

    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rs_old / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        rel_error = np.linalg.norm(r) / np.linalg.norm(b)
        errors.append(rel_error)

        if rel_error < tol:
            break

        z = np.dot(M_inv, r)
        rs_new = np.dot(r.T, z)
        beta = rs_new / rs_old
        p = z + beta * p
        rs_old = rs_new

    return x, errors


def cr(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    Conjugate Residual (CR) Method
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = n

    x = x0
    r = b - np.dot(A, x)
    p = r.copy()
    rTr = np.dot(r.T, r)
    errors = []

    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rTr / np.dot(p.T, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        rTr_new = np.dot(r_new.T, r_new)
        rel_error = np.sqrt(rTr_new) / np.linalg.norm(b)
        errors.append(rel_error)
        if rel_error < tol:
            break
        beta = rTr_new / rTr
        p = r_new + beta * p
        r = r_new
        rTr = rTr_new

    return x, errors
