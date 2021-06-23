from numerical_methods import laplacian1D, laplacian2D, newton, \
                              initialise1D, initialise2D, Implicit,\
                              ImexA
import scipy.sparse as sparse
import numpy as np
from math import cos, pi

dt = 10**-7
eps = 0.01
tol = 10**-10
max_its = 1000
N = 100
x_domain = np.linspace(0, 1, N+1)
omega = np.meshgrid(x_domain, x_domain)
h = x_domain[1] - x_domain[0]


def test_laplacian_1D():
    Lap = laplacian1D(N, h).toarray()
    assert Lap[0, 0] == -Lap[0, 1] and Lap[-1, -1] == -Lap[-1, -2], \
           "Neumann Boundary conditions not implemented in Laplacian operator."


def test_laplacian_2D():
    Lap2 = laplacian2D(N, h)
    Lap1 = laplacian1D(N, h)
    eye = sparse.identity(N+1)
    Lap3 = sparse.kron(Lap1, eye) + sparse.kron(eye, Lap1)
    assert Lap2.toarray().all() == Lap3.toarray().all(),\
           "2D finite difference Laplacian is not the relevant kronecker product \
            of the 1D laplacians"


def test_initialise_1D():
    Lp = laplacian1D(N, h)
    smooth = initialise1D(x_domain, Lp, eps, switch=0)
    random = initialise1D(x_domain, Lp, eps, switch=1)
    assert smooth[0].all() == np.array([cos(pi*x) for x in x_domain]).all() \
           and random, "Initialisation of initial conditions\
                        in 1D has not worked."


def test_initialise_2D():
    Lp = laplacian2D(N, h)
    smooth = initialise2D(omega, Lp, eps, switch=0)
    random = initialise2D(omega, Lp, eps, switch=1)
    assert smooth and random, "Initialisation of the initial conditions\
                               in 2D has not worked"


def test_newton_1D_implicit_smoothICs():
    Lp = laplacian1D(N, h)
    implicit = Implicit(dt, eps, Lp, tol, max_its)
    c0, w0, name = initialise1D(x_domain, Lp, eps, switch=0)
    out = implicit(c0, w0)
    assert out, "Newton iterations failed to converge."


def test_newton_2D_imexA_randomICs():
    Lp = laplacian2D(N, h)
    imexA = ImexA(dt, eps, Lp, tol, max_its)
    c0, w0, name = initialise2D(omega, Lp, eps, switch=1)
    out = imexA(c0, w0)
    assert out, "Newton iterations failed to converge."


test_laplacian_1D()
test_laplacian_2D()
test_initialise_1D()
test_initialise_2D()
test_newton_1D_implicit_smoothICs()
test_newton_2D_imexA_randomICs()
