"""This script is only meant to be used during development.

Anthony Baston - Oxford - 2021

TESTING SCRIPT VERSION 2 For sparse diagonals.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numerical_methods2 import *
from math import pi, cos, ceil, sqrt
from get_quantities import get_mass1D, get_energy1D

# MAIN INPUTS
T = 5*10**-4
N = 50
eps = 0.01
x_domain = np.linspace(0, 1, N+1)
y_domain = np.linspace(0, 1, N+1)
h = x_domain[1] - x_domain[0]
dt = 5*10**-7
J = ceil(T/dt)
TOL = 10**-8
max_its = 1000
omega = np.meshgrid(x_domain, y_domain)

Lp = laplacian2D(N, h)
(c0, w0) = initialise2D(omega_domain=omega, laplacian=Lp, epsilon=eps, switch=0) # noqa E501

c_evol = np.zeros((c0.size, J))
c_evol[:, 0] = c0


explicit = Explicit(dt, eps, Lp)
implicit = Implicit(dt, eps, Lp, TOL, max_its)
imexA = ImexA(dt, eps, Lp, TOL, max_its)
imexB = ImexB(dt, eps, Lp)
imexC = ImexC(dt, eps, Lp)
imexD = ImexD(dt, eps, Lp)

c = c0
w = w0

for i in range(1, J):
    print(f"Percentage complete: {ceil(i/J*1000)/10}")
    out = imexD(c, w)
    c = out[0]
    w = out[1]
    c_evol[:, i] = c


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



mass_out = get_mass1D(c_evol, h)
plt.figure(6)
plt.plot(mass_out)
plt.show()

energy_out = get_energy1D(c_evol, eps, h)
plt.figure(7)
plt.plot(energy_out)
plt.show()

print("Pause")
