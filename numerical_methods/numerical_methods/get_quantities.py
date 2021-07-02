"""
Script defining functions that return mass, M[c],
and energy, J[c], of the evolution of the mass fraction c
in the Cahn-Hilliard equation.
"""

from math import sqrt
import numpy as np


def potential_phi(c):
    """The potential function associated with the Cahn-Hilliard equation.

    Parameters
    ----------
    c : float
        The value of c at a given point in the domain.

    Returns
    ----------
    potential : float
        The value of the potential given c.
    """
    return 1/4*(1 - c**2)**2


def get_mass1D(c_evol, h):
    """The mass of the solution at a given time, for a 1D solution.

    Parameters
    ----------
    c_evol : np.array
        The c value from Cahn-Hilliard across both space and time.
    h : float
        The spatial mesh size.

    Returns
    ----------
    mass : np.array
        The total mass of the solution across time.
    """
    sz = np.shape(c_evol)
    mass = np.zeros(sz[1])
    for i in range(sz[1]):
        mass[i] = h/2 * (2 * np.sum(c_evol[:, i]) - (c_evol[0, i]
                         + c_evol[sz[0] - 1, i]))
    mass = np.array(mass)
    return mass


def get_mass2D(c_evol, h):
    """The mass of the solution at a given time, for a 2D solution.

    Parameters
    ----------
    c_evol : np.array
        The c value from Cahn-Hilliard across both space and time.
    h : float
        The spatial mesh size.

    Returns
    ----------
    mass : np.array
        The total mass of the solution across time.
    """
    sz = np.shape(c_evol)
    mass = np.zeros(sz[1])
    for i in range(sz[1]):
        c_grid = np.reshape(c_evol[:, i], (int(sqrt(sz[0])), int(sqrt(sz[0]))))
        proto_mass = get_mass1D(c_grid, h)
        mass[i] = h/2 * (2 * np.sum(proto_mass)
                         - (proto_mass[0] + proto_mass[-1]))
    mass = np.array(mass)
    return mass


def get_energy1D(c_evol, eps, h):
    """The free energy of the solution at a given time in 1D.

    Parameters
    ----------
    c_evol : np.array
        The c value from Cahn-Hilliard across both space and time.
    eps : float
        The PDE epsilon parameter.
    h : float
        The spatial mesh size.

    Returns
    ----------
    energy : np.array
        The free energy of solutions at given times.

    """
    sz = np.shape(c_evol)
    energy = np.zeros(sz[1])
    for i in range(sz[1]):
        energy_value = 1/eps*(potential_phi(c_evol[1, i])
                              + potential_phi(c_evol[sz[0]-1, i]))
        for j in range(1, sz[0]-1):
            energy_value = energy_value \
                            + 2 * (1/eps * potential_phi(c_evol[j, i])) \
                            + eps/2 * ((c_evol[j+1, i] - c_evol[j-1, i]) / (2*h)) ** 2 # noqa E501
        energy[i] = energy_value * h/2
    return energy


def get_energy2D(c_evol, eps, h):
    """The free energy of the solution at a given time in 2D.

    Parameters
    ----------
    c_evol : np.array
        The c value from Cahn-Hilliard across both space and time.
    eps : float
        The PDE epsilon parameter.
    h : float
        The spatial mesh size.

    Returns
    ----------
    energy : np.array
        The free energy of solutions at given times.

    """
    sz = np.shape(c_evol)
    energy = np.zeros(sz[1])
    square = int(sqrt(sz[0]))
    for i in range(sz[1]):
        c_grid = np.reshape(c_evol[:, i], (square, square))
        energy_value = 0
        for j in range(square):
            energy_value += 1/eps * (potential_phi(c_grid[0, j])
                                     + potential_phi(c_grid[-1, j])
                                     + potential_phi(c_grid[j, 0])
                                     + potential_phi(c_grid[j, -1]))
        energy_value -= 1/eps * (potential_phi(c_grid[0, 0]) + potential_phi(c_grid[0, -1]) # noqa E501
                         + potential_phi(c_grid[-1, 0]) + potential_phi(c_grid[-1, -1])) # noqa E501
        for kk in range(1, square-1):
            for ll in range(1, square-1):
                energy_value += 2 * (1/eps * potential_phi(c_grid[kk, ll])) \
                                 + eps/2 * ((c_grid[kk+1, ll] - c_grid[kk-1, ll]) / (2*h)  # noqa E501
                                            + (c_grid[kk, ll+1] - c_grid[kk, ll-1]) / (2*h)) ** 2 # noqa E501
        energy[i] = h**2/4 * energy_value
    return energy
