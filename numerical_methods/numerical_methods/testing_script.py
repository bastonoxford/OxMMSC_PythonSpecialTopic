"""This script is only meant to be used during development.

Anthony Baston - Oxford - 2021
"""
import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import *
from math import pi, cos, ceil
from get_quantities import get_mass1D, get_energy1D

# MAIN INPUTS
T = 5*10**-4
N = 100
eps = 0.01
x_domain = np.linspace(0, 1, N+1)
h = x_domain[1] - x_domain[0]
dt = h**4
J = ceil(T/dt)
TOL = 10**-8
max_its = 1000

Lp = laplacian1D(N, h)
(c0, w0) = initialise(omega_domain=x_domain, laplacian=Lp, epsilson=eps, switch=0) # noqa E501

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
    out = explicit(c, w)
    c = out[0]
    w = out[1]
    c_evol[:, i] = c

j_values = range(5)
for j in j_values:
    j_ = ceil(J/5)*j
    plt.figure(j)
    plt.plot(c_evol[:, j_-1])
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
