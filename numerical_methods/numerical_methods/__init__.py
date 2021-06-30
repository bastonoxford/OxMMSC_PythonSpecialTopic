"""Init script to indicate existence of a package and allow for ease of import""" # noqa E501

from .numerical_methods import NumericalMethod, Explicit, Implicit, \
                               ImexA, ImexB, ImexC, ImexD, \
                               ConvergenceError, laplacian1D, laplacian2D,\
                               initialise1D, initialise2D # noqa E401

from .get_quantities import get_mass1D, get_mass2D, get_energy1D, get_energy2D # noqa E401
