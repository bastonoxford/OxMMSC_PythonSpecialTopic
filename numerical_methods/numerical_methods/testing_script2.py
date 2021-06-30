"""This script is only meant to be used during development.

TESTING SCRIPT VERSION 2 For sparse diagonals.
"""
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from numerical_methods import *
from math import pi, cos, ceil, sqrt, exp, sin, log
from get_quantities import get_mass1D, get_mass2D, get_energy1D, get_energy2D

# MAIN INPUTS
T = 5*10**-4
N = 50
eps = 0.01
x_domain = np.linspace(0, 1, N+1)
y_domain = np.linspace(0, 1, N+1)
h = x_domain[1] - x_domain[0]
dt = h**4
dt = dt
J = ceil(T/dt)
TOL = 10**-8
max_its = 1000
omega = np.meshgrid(x_domain, y_domain)
Lp = laplacian2D(N, h)
(c0, w0, name) = initialise2D(omega_domain=omega, laplacian=Lp, epsilon=eps, switch=0) # noqa E501

xx, yy = omega

explicit = Explicit(dt, eps, Lp)
implicit = Implicit(dt, eps, Lp, TOL, max_its)
imexA = ImexA(dt, eps, Lp, TOL, max_its)
imexB = ImexB(dt, eps, Lp)
imexC = ImexC(dt, eps, Lp)
imexD = ImexD(dt, eps, Lp)


T = 1*10**-5
def make_benchmark():
    dt = 1/256*h**4
    J = ceil(T/dt)
    explicit = Explicit(dt, eps, Lp)
    c, w = c0, w0
    c_evol = np.zeros((c0.size, J))
    for k in range(1, J):
        print(f"Percentage complete: {ceil(k/J*1000)/10}")
        c, w = explicit(c, w)
        c_evol[:, k] = c
    return c_evol[:, -1]


benchmark = make_benchmark()


dt_s = [2**(-i)*h**4 for i in range(1, 4)]
errors = np.zeros(len(dt_s))
idx = 0
for dt in dt_s:
    J = ceil(T/dt)
    implicit = Implicit(dt, eps, Lp, TOL, max_its)
    c, w = c0, w0
    for k in range(1, J):
        print(f"Percentage complete: {ceil(k/J*1000)/10}")
        c, w = implicit(c, w)
    errors[idx] = np.linalg.norm(c - benchmark, 2)
    idx += 1


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


print("Determining solution mass across time.")
mass_out = get_mass2D(c_evol, h)
plt.figure(6)
plt.plot(mass_out)
plt.show()


print("Determining the free energy of the solution across time.")
energy_out = get_energy2D(c_evol, eps, h)
plt.figure(7)
plt.plot(energy_out)
plt.show()

print("Run complete.")
