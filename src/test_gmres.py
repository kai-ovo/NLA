import numpy as np
import matplotlib.pyplot as plt
from basis import arnoldi
from preconditioner import jacobi_preconditioner
from GMRES import gmres, minres, fgmres, qmr, restarted_gmres, left_preconditioned_gmres

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
    plt.title('GMRES Variants: Relative Error vs Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_gmres_algorithms():
    """
    Test and plot the relative error of various GMRES-related algorithms.
    """
    n = 100
    A = np.random.rand(n, n)
    # A = 0.5 * (A + A.T) + n * np.eye(n) 
    b = np.random.rand(n)
    x0 = np.zeros_like(b)
    
    tol = 1e-6
    max_iter = 50

    errors = []
    labels = []
    
    # Test GMRES
    x_gmres, errors_gmres = gmres(A, b, x0=x0, tol=tol, max_iter=max_iter)
    errors.append(errors_gmres)
    labels.append('GMRES')

    # # Test MINRES
    # x_minres, errors_minres = minres(A, b, x0=x0, tol=tol, max_iter=max_iter)
    # errors.append(errors_minres)
    # labels.append('MINRES')

    # Test Flexible GMRES (FGMRES)
    M_inv_list = [jacobi_preconditioner(A) for _ in range(max_iter)]
    x_fgmres, errors_fgmres = fgmres(A, b, M_inv_list, x0=x0, tol=tol, max_iter=max_iter)
    errors.append(errors_fgmres)
    labels.append('FGMRES')

    # Test QMR
    x_qmr, errors_qmr = qmr(A, b, x0=x0, tol=tol, max_iter=max_iter)
    errors.append(errors_qmr)
    labels.append('QMR')

    # Test Restarted GMRES
    x_restart_gmres, errors_restart_gmres = restarted_gmres(A, b, x0=x0, tol=tol, max_iter=max_iter, restart=10)
    errors.append(errors_restart_gmres)
    labels.append('Restarted GMRES')

    # Test Left Preconditioned GMRES
    M_inv = jacobi_preconditioner(A)
    x_left_gmres, errors_left_gmres = left_preconditioned_gmres(A, b, M_inv, x0=x0, tol=tol, max_iter=max_iter)
    errors.append(errors_left_gmres)
    labels.append('Left Preconditioned GMRES')
    
    # Plot the relative errors
    print(errors)
    plot_relative_errors(errors, labels)

if __name__ == "__main__":
    test_gmres_algorithms()
