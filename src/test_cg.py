import numpy as np
import matplotlib.pyplot as plt
from CG import cg, bicg, bicgstab, pcg, cr
from preconditioner import jacobi_preconditioner

def plot_relative_errors(errors, labels):
    """
    Plot the relative errors against the number of iterations for each method.
    """
    plt.figure(figsize=(10, 6))
    for err, label in zip(errors, labels):
        plt.plot(err, label=label)
    
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Relative Error')
    plt.title('CG Variants: Relative Error vs Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_cg_algorithms():
    """
    Test and plot the relative error of various CG-related algorithms.
    """
    n = 100
    A = np.random.rand(n, n)
    A = 0.5 * (A + A.T) + n * np.eye(n) 
    b = np.random.rand(n)
    
    x0 = np.zeros_like(b)
    tol = 1e-6
    max_iter = 50

    errors = []
    labels = []
    
    # Test Conjugate Gradient (CG)
    x_cg, errors_cg = cg(A, b, x0=x0, tol=tol, max_iter=max_iter)
    errors.append(errors_cg)
    labels.append('CG')

    # Test Preconditioned Conjugate Gradient (PCG) with Jacobi preconditioner
    M_inv = jacobi_preconditioner(A)
    x_pcg, errors_pcg = pcg(A, b, M_inv, x0=x0, tol=tol, max_iter=max_iter)
    errors.append(errors_pcg)
    labels.append('PCG')

    # Test BiCG
    x_bicg, errors_bicg = bicg(A, b, x0=x0, tol=tol, max_iter=max_iter)
    errors.append(errors_bicg)
    labels.append('BiCG')

    # Test BiCGStab
    x_bicgstab, errors_bicgstab = bicgstab(A, b, x0=x0, tol=tol, max_iter=max_iter)
    errors.append(errors_bicgstab)
    labels.append('BiCGStab')

    # Test Conjugate Residual (CR)
    x_cr, errors_cr = cr(A, b, x0=x0, tol=tol, max_iter=max_iter)
    errors.append(errors_cr)
    labels.append('CR')

    # Plot the relative errors
    print(errors)
    plot_relative_errors(errors, labels)

if __name__ == "__main__":
    test_cg_algorithms()
