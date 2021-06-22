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
    """The mass of the solution at a given time.

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


def get_energy1D(c_evol, eps, h):
    """The free energy of the solution at a given time.

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
