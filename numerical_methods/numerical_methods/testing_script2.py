"""This script is only meant to be used during development.

Anthony Baston - Oxford - 2021

TESTING SCRIPT VERSION 2 For sparse diagonals.
"""
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from numerical_methods2 import *
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


def Q(t):
    qq = (4*eps*pi**4 - 1)*c0*exp(-t)
    return qq


v1 = np.power((np.cos(pi*xx)*np.cos(pi*yy)), 3)
v1 = np.transpose(np.reshape(v1, v1.size))

v2 = np.cos(pi*xx) * np.power(np.sin(pi*xx), 2) * np.power(np.cos(pi*yy), 3)
v2 = np.transpose(np.reshape(v2, v2.size))

v3 = np.cos(pi*yy) * np.power(np.sin(pi*yy), 2) * np.power(np.cos(pi*xx), 3)
v3 = np.transpose(np.reshape(v3, v3.size))


def Q2(t):
    qq2 = 2*pi**2*c0*exp(-t) - 6*pi**2*v1*exp(-3*t) + 6*pi**2*v2*exp(-3*t) + 6*pi**2*v3*exp(-3*t)
    return Q(t) - 1/eps*qq2


def actual(t):
    return c0*exp(-t)


c = c0
w = w0 - 1/eps*(np.power(c, 3) - c)
c_evol = np.zeros((c0.size, J))
for k in range(1, J):
    print(f"Percentage complete: {ceil(k/J*1000)/10}")
    out = explicit(c, w)
    c = out[0] + dt*Q(k*dt)
    w = out[1] - 1/eps*(np.power(c, 3) - c)
    c_evol[:, k] = c


j_values = range(10)
sz = c_evol[:, 0].shape
xx, yy = omega
for j in j_values:
    j_ = ceil(J/10)*j
    c_grid = np.reshape(c_evol[:, j_-1], (int(sqrt(sz[0])), int(sqrt(sz[0]))))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xx, yy, c_grid, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    plt.show()


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
