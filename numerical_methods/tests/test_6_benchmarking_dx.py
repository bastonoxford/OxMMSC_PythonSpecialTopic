"""Benchmark Spatial testing methods vs the very fine Explicit Scheme."""

import numpy as np
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
from numerical_methods import Explicit, Implicit, ImexA,\
                       ImexB, ImexC, ImexD,\
                       laplacian2D, initialise2D
from math import ceil, sqrt

# MAIN INPUTS
T = 1*10**-6
eps = 0.01
TOL = 10**-8
max_its = 1000

N = 256
x_domain = np.linspace(0, 1, N+1)
omega = np.meshgrid(x_domain, x_domain)
xx, yy = omega
h = x_domain[1] - x_domain[0]
dt = h**4


def coarsener(vec):
    """Take in a solution vector and coarsen it to a finer grid.
       vec must be of shape ((2**n + 1)**2, ) for some integer n.
       Return coarse solution vector of shape ((2**(n-1) + 1)**2, )."""
    grid = np.reshape(vec, (int(sqrt(vec.size)), int(sqrt(vec.size))))
    sz = int((grid.shape[0]-1)/2)
    sim_mat = np.array([[1, 0],
                        [0, 0]])
    big_sim_mat = np.kron(np.identity(sz), sim_mat)
    big_sim_mat = sparse.block_diag((big_sim_mat, 1))
    out = big_sim_mat@grid@big_sim_mat
    out = np.reshape(out, out.size)
    out = out[out != 0]
    return out


def make_benchmark():
    """Generate benchmark solution using the Explicit scheme\
    with fine Spatial discretisation. Return coarse grid \
    representations reshaped as vectors."""
    N = 256
    x_domain = np.linspace(0, 1, N+1)
    omega = np.meshgrid(x_domain, x_domain)
    xx, yy = omega
    h = x_domain[1] - x_domain[0]
    dt = h**4
    J = ceil(T/dt)
    Lp = laplacian2D(N, h)
    explicit = Explicit(dt, eps, Lp)
    (c0, w0, name) = initialise2D(omega_domain=omega, laplacian=Lp,
                                  epsilon=eps, switch=0)
    c, w = c0, w0
    for k in range(1, J):
        print(f"Generating benchmark, percentage\
              complete: {ceil(k/J*1000)/10}")
        c, w = explicit(c, w)
    c128 = coarsener(c)
    c64 = coarsener(c128)
    c32 = coarsener(c64)
    c16 = coarsener(c32)
    c8 = coarsener(c16)

    return [c8, c16, c32, c64]


benchmark = make_benchmark()


N_values = [8, 16, 32]


def method_tester_dx(methodClass):
    """Instantiate method and test against benchmark solution at increasingly\
       finer levels of spatial discretisation"""
    errors = np.zeros(len(N_values))
    idx = 0
    for N in N_values:
        x_domain = np.linspace(0, 1, N+1)
        omega = np.meshgrid(x_domain, x_domain)
        xx, yy = omega
        h = x_domain[1] - x_domain[0]
        J = ceil(T/dt)
        Lp = laplacian2D(N, h)
        method = methodClass(dt, eps, Lp, TOL, max_its)
        (c0, w0, name) = initialise2D(omega_domain=omega, laplacian=Lp, epsilon=eps, switch=0) # noqa E501
        c, w = c0, w0
        for j in range(1, J):
            print(f"Percentage complete for {method.name}\
                  with N = {N}: {ceil(j/J*1000)/10}")
            c, w = method(c, w)
        errors[idx] = scipy.linalg.norm(c - benchmark[idx], 2)/sqrt(c.size)
        idx += 1
    return errors


def error_ratios(ls):
    return [ls[i]/ls[i+1] for i in range(0, len(ls) - 1)]


def test_implicit_benchmark_dx():
    implicit_errors = method_tester_dx(Implicit)
    implicit_ratios = error_ratios(implicit_errors)
    assert implicit_ratios[0]/2 > 2 and implicit_ratios[1]/2 > 2,\
           "Second order convergence is not being observed\
            for the Implicit class"
    return implicit_errors, implicit_ratios


def test_imex_a_benchmark_dx():
    imexA_errors = method_tester_dx(ImexA)
    imexA_ratios = error_ratios(imexA_errors)
    assert imexA_ratios[0]/2 > 2 and imexA_ratios[1]/2 > 2,\
           "Second order convergence is not being observed\
            for the ImexA class"
    return imexA_errors, imexA_ratios


def test_imex_b_benchmark_dx():
    imexB_errors = method_tester_dx(ImexB)
    imexB_ratios = error_ratios(imexB_errors)
    assert imexB_ratios[0]/2 > 2 and imexB_ratios[1]/2 > 2,\
           "Second order convergence is not being observed\
            for the ImexB class"
    return imexB_errors, imexB_ratios


def test_imex_c_benchmark_dx():
    imexC_errors = method_tester_dx(ImexC)
    imexC_ratios = error_ratios(imexC_errors)
    assert imexC_ratios[0]/2 > 2 and imexC_ratios[1]/2 > 2,\
           "Second order convergence is not being observed\
            for the ImexC class"
    return imexC_errors, imexC_ratios


def test_imex_d_benchmark_dx():
    imexD_errors = method_tester_dx(ImexD)
    imexD_ratios = error_ratios(imexD_errors)
    assert imexD_ratios[0]/2 > 2 and imexD_ratios[1]/2 > 2,\
           "Second order convergence is not being observed\
            for the ImexD class"
    return imexD_errors, imexD_ratios


# implicit_errors, implicit_ratios = test_implicit_benchmark_dx()
# imexA_errors, imexA_ratios = test_imex_a_benchmark_dx()
# imexB_errors, imexB_ratios = test_imex_b_benchmark_dx()
# imexC_errors, imexC_ratios = test_imex_c_benchmark_dx()
# imexD_errors, imexD_ratios = test_imex_d_benchmark_dx()

# dx_s = [1/i for i in N_values]

# plt.loglog(dx_s, implicit_errors, 'b', marker="x", label="Implicit")
# plt.xlabel("$\\Delta x$")
# plt.ylabel("$||C - C_{Exp}||_2/\\sqrt{C.size}$")
# plt.legend(loc="lower right")
# plt.title("Convergence of methods to the benchmarked solution.")
# plt.show()

# plt.loglog(dx_s, imexA_errors, 'g', marker="x", label="ImexA")
# plt.xlabel("$\\Delta x$")
# plt.ylabel("$||C - C_{Exp}||_2/\\sqrt{C.size}$")
# plt.legend(loc="lower right")
# plt.title("Convergence of methods to the benchmarked solution.")
# plt.show()

# plt.loglog(dx_s, imexB_errors, 'r', marker="x", label="ImexB")
# plt.xlabel("$\\Delta x$")
# plt.ylabel("$||C - C_{Exp}||_2/\\sqrt{C.size}$")
# plt.legend(loc="lower right")
# plt.title("Convergence of methods to the benchmarked solution.")
# plt.show()

# plt.loglog(dx_s, imexC_errors, 'c', marker="x", label="ImexC")
# plt.xlabel("$\\Delta x$")
# plt.ylabel("$||C - C_{Exp}||_2/\\sqrt{C.size}$")
# plt.legend(loc="lower right")
# plt.title("Convergence of methods to the benchmarked solution.")
# plt.show()

# plt.loglog(dx_s, imexD_errors, 'm', marker="x", label="ImexD")
# plt.xlabel("$\\Delta x$")
# plt.ylabel("$||C - C_{Exp}||_2/\\sqrt{C.size}$")
# plt.legend(loc="lower right")
# plt.title("Convergence of methods to the benchmarked solution.")
# plt.show()

# print("Run complete.")
