"""This script illustrates the working of the numerical methods in 1D and 2D.

--------------------------------------------------------------------------------

Required variables (adjust as desired):
------------
T : float           
    End time.
N : int             
    Number of discretised points in space.
J_im : int          
    Number of discretised points in time for methods with an implicit component. # noqa E501
eps : float         
    The epsilon parameter of the PDE.
x_domain : np.array 
    The spatial mesh points, this is fixed to be between 0 and 1.
h : float           
    The spatial mesh size.
TOL : float         
    The tolerance for the Newton method in the Implicit and ImexA schemes.
max_its : int       
    The maximum number of interations in Newton's methods.
dt_ee : float       
    The timestep for the Explicit Euler method, must be < 8*h**4 for stability.
J_ee : int          
    Number of discretised points in time for the explcit method.
dt_im : float       
    The timestep for all methods with an implicit component.

--------------------------------------------------------------------------------

User inputs:
------------
intial : int
    Type of initial conditions, 0 for smooth and 1 for random
choice : int
    Choice of numerical method, 0, 1, 2, 3, 4 and 5 corresponding 
    to Explicit, Implicit, ImexA, ImexB, ImexC and ImexD respectively.
dimension : int
    The dimension in the Cahn-Hilliard equation is to be solved.
    If dimension == 1, solve on the interval [0, 1] in 1D, or if
    dimension == 2, solve on the square [0, 1]**2.

--------------------------------------------------------------------------------

Output:
c_evol: np.array
    Solution of the Cahn-Hilliard equation in, using desired method.


--------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from numerical_methods import laplacian1D, laplacian2D, Explicit, Implicit, \
                               ImexA, ImexB, ImexC, ImexD, \
                               initialise1D, initialise2D
from math import ceil, sqrt
from get_quantities import get_mass1D, get_energy1D, \
                           get_mass2D, get_energy2D

# Choose method and initial conditions.
choice = 1
initial = 1
dimension = 2

# Required variables - adjust as desired.
T = 5*10**-4
N = 75
J_im = 1000
eps = 0.01
x_domain = np.linspace(0, 1, N+1)
y_domain = np.linspace(0, 1, N+1)
h = x_domain[1] - x_domain[0]
TOL = 10**-8
max_its = 1000
dt_ee = h**4
J_ee = ceil(T/dt_ee)
dt_im = T/J_im

# Set up Laplacian Operator and Initial Conditions based on Dimension selection
if dimension == 1:
    Lp = laplacian1D(N, h)
    (c0, w0, name) = initialise1D(omega_domain=x_domain, laplacian=Lp, epsilon=eps, switch=initial) # noqa E501

elif dimension == 2:
    Lp = laplacian2D(N, h)
    omega = np.meshgrid(x_domain, y_domain)
    (c0, w0, name) = initialise2D(omega_domain=omega, laplacian=Lp, epsilon=eps, switch=initial) # noqa E501

else:
    raise NotImplementedError(f"Please select dimension to be an int \
                                with value either 1 or 2, not {dimension} \
                                of type {type(dimension)}")

# Instatiate numerical methods.
methods = [Explicit(dt_ee, eps, Lp), Implicit(dt_im, eps, Lp, TOL, max_its),
           ImexA(dt_im, eps, Lp, TOL, max_its), ImexB(dt_im, eps, Lp),
           ImexC(dt_im, eps, Lp), ImexD(dt_im, eps, Lp)]

# Check choice input.
method = methods[choice]
if choice == 0:
    J = J_ee
    assert dt_ee < 8*h**4, f"For stability, please ensure that the \
                             timestep for the Explicit Euler method \
                             is < 8*h**4 you have entered a value that \
                             is {dt_ee/h**4} times larger that h**4."

elif type(choice) == int and 0 < choice <= 5:
    J = J_im

else:
    raise NotImplementedError(f"Please choose a method by setting choice  \
                                equal to an int from 0 to 5, not {choice} \
                                of type {type(choice)}")

# Solve the equation.
print(f"Solving Cahn-Hilliard in {dimension}D, with the {method.name} scheme, and {name} initial conditions. About to start.") # noqa E501
time.sleep(5)

c_evol = np.zeros((c0.size, J))
w_evol = np.zeros((w0.size, J))
c = c0
w = w0
c_evol[:, 0] = c0
w_evol[:, 0] = w0
for i in range(1, J):
    print(f"Percentage complete: {ceil(i/J*1000)/10}")
    out = method(c, w)
    c = out[0]
    w = out[1]
    c_evol[:, i] = c


if dimension == 1:
    # Plot the evolution of the solution.
    fig, ax = plt.subplots(5, 1, sharex=True)
    ax[0].plot(x_domain, c_evol[:, 0])
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([-1, 1])
    for j in range(1, 5):
        ax[j].plot(x_domain, c_evol[:, j*ceil(J/4)-1])
        ax[j].set_xlim([0, 1])
        ax[j].set_ylim([-1, 1])
    plt.show()

    # Get solution mass evolution over time.
    mass_out = get_mass1D(c_evol, h)
    plt.figure(6)
    plt.plot(mass_out)
    plt.show()

    # Get solution energy evolution over time.
    energy_out = get_energy1D(c_evol, eps, h)
    plt.figure(7)
    plt.plot(energy_out)
    plt.show()
else:
    # Plot the evolution of the solution.
    j_values = range(10)
    sz = c_evol[:, 0].shape
    xx, yy = omega
    for j in j_values:
        j_ = ceil(J/10)*j
        c_grid = np.reshape(c_evol[:, j_-1], (int(sqrt(sz[0])), int(sqrt(sz[0])))) # noqa E501
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(xx, yy, c_grid, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        plt.show()

    # Get solution mass evolution over time
    mass_out = get_mass2D(c_evol, h)
    plt.figure(11)
    plt.plot(mass_out)
    plt.show()

    # Get solution energy evolution over time
    energy_out = get_energy2D(c_evol, eps, h)
    plt.figure(12)
    plt.plot(energy_out)
    plt.show()

print("Run complete")
