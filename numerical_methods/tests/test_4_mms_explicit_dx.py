import numpy as np
import scipy
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from numerical_methods import Explicit, laplacian2D, initialise2D
from math import inf, pi, ceil, exp, sqrt


T = 1*10**-6
eps = 0.01
dt = (1/256)**4
J = ceil(T/dt)


def Q(c0, t):
    qq = (4*eps*pi**4 - 1)*c0*exp(-t)
    return qq


def actual_exp(c0, t):
    return c0*exp(-t)


def test_space_convergence_exp():
    N_values = [2, 4, 8, 16, 32, 64, 128]
    max_errors = np.zeros(len(N_values))
    k = 0
    for n in N_values:
        x_domain = np.linspace(0, 1, n+1)
        y_domain = np.linspace(0, 1, n+1)
        h = x_domain[1] - x_domain[0]
        omega = np.meshgrid(x_domain, y_domain)
        Lp = laplacian2D(n, h)
        (c0, w0, name) = initialise2D(omega_domain=omega, laplacian=Lp, epsilon=eps, switch=0) # noqa E501
        explicit = Explicit(dt, eps, Lp)
        c = c0
        w = w0 - 1/eps * (np.power(c0, 3) - c0)
        c_evol = np.zeros((c.size, J))
        c_evol[:, 0] = c0
        for i in range(1, J):
            print(f"Percentage complete: {ceil(i/J*1000)/10}")
            out = explicit(c, w)
            c = out[0] + dt*Q(c0, i*dt)
            w = out[1] - 1/eps*(np.power(c, 3) - c)
            c_evol[:, i] = c
            if i == ceil(J/2):
                max_errors[k] = scipy.linalg.norm(actual_exp(c0, i*dt)-c, inf)
                k += 1

    error_ratio = [max_errors[j]/max_errors[j+1] for j in range(len(max_errors) - 1)]
    plt.loglog([1/i for i in N_values], max_errors, marker="+")
    matplotlib.rcParams.update({"text.usetex": True})
    plt.xlabel("$\\Delta x$")
    plt.ylabel("$||C - \\widehat{c}||_\\infty$")
    plt.title("Convergence for $\\Delta x$ - Explicit Class.")
    plt.show()


test_space_convergence_exp()
