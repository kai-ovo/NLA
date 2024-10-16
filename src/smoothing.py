import numpy as np

"""
Some popular smoothing methods. They form the core of multigrid methods.
"""

def jacobi(A, b, x0, tol=1e-10, max_iter=1000):
    """
    Jacobi Method
    
    inputs:
    - A, b: set up Ax = b
    - x0: Initial guess
    - tol: rel. err. tolerance
    - max_iter: Maximum number of iterations

    outputs:
    - x: solution
    """
    n = len(b)
    x = np.copy(x0)
    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sigma) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def gauss_seidel(A, b, x0, tol=1e-10, max_iter=1000):
    """
    Gauss-Seidel Method
    
    """
    n = len(b)
    x = np.copy(x0)
    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sigma) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def sor(A, b, x0, omega=1.0, tol=1e-10, max_iter=1000):
    """
    Successive Over-Relaxation (SOR)
    """
    n = len(b)
    x = np.copy(x0)
    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - omega) * x[i] + (omega * (b[i] - sigma) / A[i, i])
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def ssor(A, b, x0, omega=1.0, tol=1e-10, max_iter=1000):
    """
    Symmetric Successive Over-Relaxation (SSOR)

    inputs:
    - omega: Relaxation factor
    """
    n = len(b)
    x = np.copy(x0)
    for _ in range(max_iter):
        x_new = np.copy(x)
        # Forward sweep
        for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
        # Backward sweep 
        for i in range(n-1, -1, -1):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def richardson(A, b, x0, omega=1.0, tol=1e-10, max_iter=1000):
    """
    Richardson Iteration (Damped Jacobi):
    
    inputs:
    - A, b : set up Ax = b
    - x0: Initial guess for the solution
    - omega: damping factor
    - tol: rel. err. tolerance
    - max_iter: Maximum number of iterations

    outputs:
    - x: solution
    """
    x = np.copy(x0)
    for _ in range(max_iter):
        r = b - np.dot(A, x)
        x_new = x + omega * r
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x


def test_smoothing_methods():
    """
    Example test cases
    """
    A = np.array([[4, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 3]], dtype=float)
    b = np.array([15, 10, 10], dtype=float)
    
    x0 = np.zeros_like(b)
    
    tol = 1e-6
    max_iter = 100
    
    jacobi_result = jacobi(A, b, x0, tol, max_iter)
    print("Jacobi method result:", jacobi_result)
    
    gauss_seidel_result = gauss_seidel(A, b, x0, tol, max_iter)
    print("Gauss-Seidel method result:", gauss_seidel_result)
    
    sor_result = sor(A, b, x0, omega=1.25, tol=tol, max_iter=max_iter)
    print("SOR method result (omega=1.25):", sor_result)
    
    ssor_result = ssor(A, b, x0, omega=1.25, tol=tol, max_iter=max_iter)
    print("SSOR method result (omega=1.25):", ssor_result)
    
    richardson_result = richardson(A, b, x0, omega=1.0, tol=tol, max_iter=max_iter)
    print("Richardson Iteration result (omega=1.0):", richardson_result)
    
    print("Expected solution (by numpy):", np.linalg.solve(A, b))

if __name__ == "__main__":
    test_smoothing_methods()
