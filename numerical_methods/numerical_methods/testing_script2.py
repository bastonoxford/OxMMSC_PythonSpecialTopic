"""This script is only meant to be used during development.

TESTING SCRIPT VERSION 2 For sparse diagonals.
"""
import numpy as np
import scipy
from scipy import sparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from numerical_methods import Explicit, Implicit, laplacian2D, initialise2D
from math import pi, cos, ceil, sqrt, exp, sin, log
from get_quantities import get_mass1D, get_mass2D, get_energy1D, get_energy2D

# MAIN INPUTS


T = 1*10**-5
eps = 0.01
TOL = 10**-8
max_its = 1000

N = 256
x_domain = np.linspace(0, 1, N+1)
omega = np.meshgrid(x_domain, x_domain)
xx, yy = omega
h = x_domain[1] - x_domain[0]
dt = h**4
J = ceil(T/dt)
Lp = laplacian2D(N, h)
(c0, w0, name) = initialise2D(omega_domain=omega, laplacian=Lp, epsilon=eps, switch=0) # noqa E501


def coarsener(vec):
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
    N = 256
    x_domain = np.linspace(0, 1, N+1)
    omega = np.meshgrid(x_domain, x_domain)
    xx, yy = omega
    h = x_domain[1] - x_domain[0]
    dt = h**4
    J = ceil(T/dt)
    Lp = laplacian2D(N, h)
    explicit = Explicit(dt, eps, Lp)
    (c0, w0, name) = initialise2D(omega_domain=omega, laplacian=Lp, epsilon=eps, switch=0) # noqa E501
    c, w = c0, w0
    for k in range(1, J):
        print(f"Generating benchmark, percentage complete: {ceil(k/J*1000)/10}")
        c, w = explicit(c, w)
    c128 = coarsener(c)
    c64 = coarsener(c128)
    c32 = coarsener(c64)
    c16 = coarsener(c32)
    c8 = coarsener(c16)
    return [c8, c16, c32, c64]


benchmark = make_benchmark()
c8_grid = np.reshape(benchmark[0], (int(sqrt(benchmark[0].size)), int(sqrt(benchmark[0].size))))
xd8 = np.linspace(0, 1, int(sqrt(benchmark[0].size)))
xx8, yy8 = np.meshgrid(xd8, xd8)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xx8, yy8, c8_grid, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
plt.show()

N_values = [8, 16, 32, 64]


def implicit_dx_bench():
    errors = np.zeros(len(N_values))
    idx = 0
    for N in N_values:
        x_domain = np.linspace(0, 1, N+1)
        omega = np.meshgrid(x_domain, x_domain)
        xx, yy = omega
        h = x_domain[1] - x_domain[0]
        J = ceil(T/dt)
        Lp = laplacian2D(N, h)
        implicit = Implicit(dt, eps, Lp, TOL, max_its)
        (c0, w0, name) = initialise2D(omega_domain=omega, laplacian=Lp, epsilon=eps, switch=0) # noqa E501
        c, w = c0, w0
        for j in range(1, J):
            print(f"Percentage complete for N = {N}: {ceil(j/J*1000)/10}")
            c, w = implicit(c, w)
        errors[idx] = scipy.linalg.norm(c - benchmark[idx], 2)
        idx += 1
    return errors


implicit_errors = implicit_dx_bench()

#j_values = range(10)
#sz = c_evol[:, 0].shape
#xx, yy = omega
#for j in j_values:
#    j_ = ceil(J/10)*j
#    c_grid = np.reshape(c_evol[:, j_-1], (int(sqrt(sz[0])), int(sqrt(sz[0]))))
#    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#    ax.plot_surface(xx, yy, c_grid, cmap=cm.coolwarm,
#                    linewidth=0, antialiased=False)
#    plt.show()

print("Run complete.")
