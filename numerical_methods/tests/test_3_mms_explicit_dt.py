import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from numerical_methods import Explicit, laplacian2D, initialise2D
from math import pi, ceil, exp


T = 1*10**-5
N = 50
eps = 0.01
x_domain = np.linspace(0, 1, N+1)
y_domain = np.linspace(0, 1, N+1)
h = x_domain[1] - x_domain[0]
TOL = 10**-8
max_its = 1000
omega = np.meshgrid(x_domain, y_domain)
Lp = laplacian2D(N, h)
(c0, w0, name) = initialise2D(omega_domain=omega, laplacian=Lp, epsilon=eps, switch=0) # noqa E501


def Q(t):
    qq = (4*eps*pi**4 - 1)*c0*exp(-t)
    return qq


def actual_exp(t):
    return c0*exp(-t)


def test_time_convergence_exp():
    c = c0
    w = w0 - 1/eps*(np.power(c0, 3) - c0)
    ll = 7
    max_errors = np.zeros(ll)
    dt_s = [2**(i - ll)*h**4 for i in range(0, ll)]
    k = 0
    for dt in dt_s:
        J = ceil(T/dt)
        explicit = Explicit(dt, eps, Lp)
        c = c0
        w = w0
        c_evol = np.zeros((c0.size, J))
        c_evol[:, 0] = c0
        for i in range(1, J):
            print(f"Percentage complete: {ceil(i/J*1000)/10}")
            out = explicit(c, w)
            c = out[0] + dt*Q(i*dt)
            w = out[1] - 1/eps*(np.power(c, 3) - c)
            c_evol[:, i] = c
            if i == ceil(J/2):
                max_errors[k] = scipy.linalg.norm(actual_exp(i*dt) - c, 2)
                k += 1
    assert [abs(max_errors[i]/max_errors[i+1] - 1/2) < 10**-15
           for i in range(ll-1)], "Convergence is not linear as expected."

    plt.loglog(dt_s, max_errors, marker="+")
    matplotlib.rcParams.update({"text.usetex": True})
    plt.xlabel("$\\Delta t$")
    plt.ylabel("$||C - \\widehat{c}||_2$")
    plt.title("Convergence for $\\Delta t$ - Explicit Class.")
    plt.show()


test_time_convergence_exp()
