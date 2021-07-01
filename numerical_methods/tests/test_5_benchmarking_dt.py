"""Benchmark Temporal testing methods vs the very fine Explicit Scheme."""
import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import Explicit, Implicit, ImexA, ImexB, ImexC,\
                              ImexD, laplacian2D, initialise2D
from math import ceil

# MAIN INPUTS
N = 50
eps = 0.01
x_domain = np.linspace(0, 1, N+1)
y_domain = np.linspace(0, 1, N+1)
h = x_domain[1] - x_domain[0]
TOL = 10**-8
max_its = 1000
omega = np.meshgrid(x_domain, y_domain)
Lp = laplacian2D(N, h)
(c0, w0, name) = initialise2D(omega_domain=omega, laplacian=Lp,
                              epsilon=eps, switch=0)
xx, yy = omega

T = 1*10**-5


def make_benchmark():
    """Generate benchmark solution using the Explicit scheme\
     with fine temporal discretisation."""
    dt = 1/256*h**4
    J = ceil(T/dt)
    explicit = Explicit(dt, eps, Lp)
    c, w = c0, w0
    c_evol = np.zeros((c0.size, J))
    for k in range(1, J):
        print(f"Generating benchmark, percentage complete:\
             {ceil(k/J*1000)/10}")
        c, w = explicit(c, w)
        c_evol[:, k] = c
    return c_evol[:, -1]


benchmark = make_benchmark()
dt_s = [2**(-i)*h**4 for i in range(1, 4)]


def method_tester(methodClass):
    """Instantiate method and test against benchmark solution at increasingly\
       finer levels of temporal discretisation"""
    errors = np.zeros(len(dt_s))
    idx = 0
    for dt in dt_s:
        J = ceil(T/dt)
        method = methodClass(dt, eps, Lp, TOL, max_its)
        c, w = c0, w0
        for k in range(1, J):
            print(f"Percentage complete, {method.name} with dt = {dt}:\
                 {ceil(k/J*1000)/10}")
            c, w = method(c, w)
        errors[idx] = np.linalg.norm(c - benchmark, 2)
        idx += 1
    assert abs(errors[0]/errors[1] - 2) < 10**-1\
           and abs(errors[1]/errors[2] - 2) < 10**-1,\
           f"Temporal Convergence is not linear as expected for\
             {method.name} scheme."
    return errors


def test_implicit_benchmark_dt():
    """Test Implicit class time convergence against benchmark."""
    implicit_errors = method_tester(Implicit)
    return implicit_errors


def test_imex_a_benchmark_dt():
    """Test ImexA class time convergence against benchmark."""
    imexA_errors = method_tester(ImexA)
    return imexA_errors


def test_imex_b_benchmark_dt():
    """Test ImexB class time convergence against benchmark."""
    imexB_errors = method_tester(ImexB)
    return imexB_errors


def test_imex_c_benchmark_dt():
    """Test ImexC class time convergence against benchmark."""
    imexC_errors = method_tester(ImexC)
    return imexC_errors


def test_imex_d_benchmark_dt():
    """Test ImexD class time convergence against benchmark."""
    imexD_errors = method_tester(ImexD)
    return imexD_errors


def error_ratios(ls):
    return [ls[i]/ls[i+1] for i in range(0, len(ls) - 1)]


implicit_errors = test_implicit_benchmark_dt()
implicit_ratios = error_ratios(implicit_errors)

imexA_errors = test_imex_a_benchmark_dt()
imexA_ratios = error_ratios(imexA_errors)

imexB_errors = test_imex_b_benchmark_dt()
imexB_ratios = error_ratios(imexB_errors)

imexC_errors = test_imex_c_benchmark_dt()
imexC_ratios = error_ratios(imexC_errors)

imexD_errors = test_imex_d_benchmark_dt()
imexD_ratios = error_ratios(imexD_errors)


plt.loglog(dt_s, implicit_errors, label="Implicit", marker="x")
plt.loglog(dt_s, imexA_errors, label="ImexA", marker="x")
plt.loglog(dt_s, imexB_errors, label="ImexB", marker="x")
plt.loglog(dt_s, imexC_errors, label="ImexC", marker="x")
plt.loglog(dt_s, imexD_errors, label="ImexD", marker="x")

plt.xlabel("$\\Delta t$")
plt.ylabel("$||C - C_{Exp}||_2$")
plt.legend(loc="lower right")
plt.title("Convergence of methods to the benchmarked solution.")
plt.show()
