"Test imports."
import sys
print(sys.executable)


def test_numerical_import():
    "Ensure that we can import the numerical methods."
    from numerical_methods.numerical_methods import NumericalMethod, Explicit, \
                                                    Implicit, ImexA, ImexB, \
                                                    ImexC, ImexD # noqa F401


def test_additional_numerical_import():
    "Ensure we can also import the additional associated functions."
    from numerical_methods.numerical_methods import initialise1D, \
                                                    initialise2D,laplacian1D,\
                                                    laplacian2D, ConvergenceError # noqa F401


def test_get_quantities_import():
    "Ensure we can import the functions to get mass and energy."
    from numerical_methods.get_quantities import potential_phi, get_mass1D,\
                                                 get_mass2D, get_energy1D, \
                                                 get_energy2D # noqa F401


test_numerical_import()
test_additional_numerical_import()
test_get_quantities_import()
