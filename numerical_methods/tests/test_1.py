
def test_numerical_import():
    "Ensure that we can import the numerical methods."
    from numerical_methods import NumericalMethod, Explicit,\
                                                     Implicit, ImexA, ImexB, \
                                                     ImexC, ImexD

def test_additional_function_import():
    from numerical_methods import Newton, initialise1D, \
                                                     initialise2D,laplacian1D,\
                                                     laplacian2D, ConvergenceError