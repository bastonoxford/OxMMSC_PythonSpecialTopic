"""
A Class hierarchy defining six Numerical Methods for the Cahn-Hilliard PDE.

Anthony Baston - Oxford - June 2021
"""
import numpy as np
from math import pi, cos


class ConvergenceError(Exception):
    """Exception raised if the solver fails to converge."""
    pass


def newton(f, df, c0, w0, tol, max_iter):
    """Solve a multivariate non-linear eqaution using the Newton-Raphson iteration.

    Solve f = 0 using the Newton-Raphson iteration.

    Parameters
    ----------
    f    function(c0: np.array, c: np.array, w: np.array) -> np.array
        The function whose root is being found
    df : function(c: np.array) -> np.array
        The Jacobian of f
    c0 : np.array
        The initial guess for c
    w0 : np.array
        The initial guess for w
    tol : float
        The solver tolerance, convergence is achieved when
        max(abs(f(c0, c, w))) < 0
    max_iter : int
        The maximum number of acceptable iterations before the solver \
        is determined to have failed.

    Returns
    ----------
    tuple(np.array, np.array)
        The approximate root computed using Newton iteration.
    """
    n = 0
    c = c0
    w = w0
    while max(abs(f(c0, c, w))) > tol and n <= max_iter:
        out = - np.linalg.solve(df(c), f(c0, c, w))
        c = c + out[:c.size]
        w = w + out[c.size:]
        n += 1

    if n <= max_iter and max(abs(f(c0, c, w))) <= tol:
        return (c, w)
    else:
        raise ConvergenceError("Convergence Failure")


def laplacian1D(n, h):
    """Construct a second order finite difference Laplacian operator,\
       associated with Neumann boundary conditions.

    Parameters
    ----------
    n : int
        The total number of for the discretised mesh.
    h : float
        The size of the mesh

    Returns
    ----------
    np.array
        The discretised Laplacian operator, of shape (N+1, N+1)
    """
    M = np.diag(-2*np.ones(n+1)) + np.diag(np.ones(n), 1) \
        + np.diag(np.ones(n), -1)
    M[0, 1] = 2
    M[n, n-1] = 2
    M = 1/h**2 * M
    return M


def initialise(omega_domain, laplacian, epsilson, switch):
    """Set initial conditions to be either smooth or random

    Parameters
    ----------
    omega_domain : np.array
        The domain on which the Cahn-Hilliard equation is to be solved.
    laplacian : np.array
        The Laplacian operator from laplacian1D, to be used to construct
        the w0 initial conditions
    epsilon : float
        The PDE epsilon parameter.
    switch : int
        Either 0 or 1. An input of 0 returns smooth initial data, an input
        of 1 returns random initial data

    Returns
    ----------
    tuple(np.array, np.array)
        The c0 and w0 initial data respectively.
    """
    if switch == 0:
        c0 = np.transpose(np.array([cos(pi*x) for x in omega_domain]))
        w0 = 1/epsilson*(np.power(c0, 3) - c0) + epsilson*pi**2*c0
        return c0, w0
    elif switch == 1:
        c0 = np.transpose(np.random.normal(0, 1, len(omega_domain)))
        w0 = 1/epsilson*(np.power(c0, 3) - c0) + epsilson*laplacian@c0
        return c0, w0
    else:
        raise NotImplementedError(f"You must set the switch = 0 or 1,\
                                  not {switch}")


class NumericalMethod:
    """A general class of numerical method for the Cahn-Hilliard equation."""
    def __init__(self, dt, eps, lap, *args, **kwargs):
        """Initialisation of general numerical method.

        Parameters
        ----------
        dt : float
            The timestep size of the method to be implemented.
        eps : float
            The PDE epsilon parameter
        lap : np.array
            The discretised laplacian operator
        """
        self.timestep = dt
        self.epsilon = eps
        self.laplacian = lap
        self.args = tuple(args)
        self.kwargs = tuple(kwargs)

    def __call__(self):
        """General call method will intentionally not work \
           for this superclass."""
        raise NotImplementedError


class Explicit(NumericalMethod):
    """Implementation of Euler's Explicit method for Cahn-Hilliard."""

    name = "Explicit"

    def __call__(self, c, w):
        """Single step of Explicit Euler.

        Parameters
        ----------
        c : np.array
            The concentration by mass fraction, as in the equation
        w : np.array
            The second order variable, as in the equation
        """
        c_next = c + self.timestep*self.laplacian@w
        w_next = (1/self.epsilon)*(np.power(c_next, 3) - c_next) - self.epsilon*self.laplacian@c_next # noqa E501

        return (c_next, w_next)


class Implicit(NumericalMethod):
    """Implementation of Euler's Implicit method for Cahn-Hilliard."""

    name = "Implicit"

    def __init__(self, dt, eps, lap, tolerance, max_iter):
        """Initialisation of Implicit method, call superclass __init__\
           and sets additional attributes.

        Additional Parameters
        ----------
        tolerance : float
            The tolerance to be used in Newton-Raphson root finding.
        max_iter : int
            The maximum number of iteration to be used in Newton-Raphson \
            root finding.
        """
        super().__init__(dt, eps, lap)
        self.tolerance = tolerance
        self.max_iter = max_iter

    def jacobian(self, c):
        """Jacobian associated with the Implicit scheme.

        Parameters
        ----------
        c : np.array
            The vector on which the Jacobian depends.

        Returns
        ----------
        jacobian : np.block
            A block matrix representation of the Jacobian.
        """
        eye = np.identity(c.size)
        return np.block([[eye, -self.timestep*self.laplacian],
                         [1/self.epsilon*(-3*np.diag(np.power(c, 2)) + eye) + self.epsilon*self.laplacian, eye]]) # noqa E501

    def implicit_function(self, c, c_next, w_next):
        """Function associated with the implicit method.

        Parameters
        ----------
        c : np.array
            The value of c at the previous timestep that defines the current \
            function, a constant at each iteration.
        c_next : np.array
            The value of c at the next timestep, that we seek to solve for \
            using Newton iterations.
        w_next : np.array
            The value of w at the next timestep, that we also seek to solve \
            for using Newton iterations.

        Returns
        ----------
        output : np.array
            The output of the function.
        """
        vec_1 = c_next-c-self.timestep*self.laplacian@w_next
        vec_2 = w_next - 1/self.epsilon*np.power(c_next, 3) + 1/self.epsilon*c_next + self.epsilon*self.laplacian@c_next # noqa E501
        return np.concatenate((vec_1, vec_2))

    def __call__(self, c, w):
        """Call one step of the Implicit method to solve at the next timestep.

        Parameters
        ----------
        c : np.array
            The value of c at the current timestep.
        w : np.array
            The value of w at the current timestep.

        Returns
        ----------
        c_next, w_next : tuple(np.array, np.array)
            The values of c and w at the next timestep.
        """

        (c_next, w_next) = newton(self.implicit_function, self.jacobian, c, w, self.tolerance, self.max_iter) # noqa E501
        return (c_next, w_next)


class ImexA(NumericalMethod):
    """Implementation of the ImexA scheme for Cahn-Hilliard."""

    name = "ImexA"

    def __init__(self, dt, eps, lap, tolerance, max_iter):
        """Initialisation of ImexA method, call superclass __init__ and sets additional attributes. # noqa E501

        Additional Parameters
        ----------
        tolerance : float
            The tolerance to be used in Newton-Raphson root finding.
        max_iter : int
            The maximum number of iteration to be used in Newton-Raphson root finding. # noqa E501
        """
        super().__init__(dt, eps, lap)
        self.tolerance = tolerance
        self.max_iter = max_iter

    def jacobian(self, c):
        """Jacobian associated with the ImexA scheme.

        Parameters
        ----------
        c : np.array
            The vector on which the Jacobian depends.

        Returns
        ----------
        jacobian : np.block
            A block matrix representation of the Jacobian.
        """
        eye = np.identity(c.size)
        return np.block([[eye, -self.timestep*self.laplacian],
                         [-3/self.epsilon*np.diag(np.power(c, 2))+self.epsilon*self.laplacian, eye]]) # noqa E501

    def implicit_function(self, c, c_next, w_next):
        """Function associated with the ImexA method.

        Parameters
        ----------
        c : np.array
            The value of c at the previous timestep that defines the current \
            function, a constant at each iteration.
        c_next : np.array
            The value of c at the next timestep, that we seek to solve for \
            using Newton iterations.
        w_next : np.array
            The value of w at the next timestep, that we also seek to solve \
            for using Newton iterations.

        Returns
        ----------
        output : np.array
            The output of the function.
        """
        vec_1 = c_next-c-self.timestep*self.laplacian@w_next
        vec_2 = w_next - 1/self.epsilon*np.power(c_next, 3) + 1/self.epsilon*c + self.epsilon*self.laplacian@c_next # noqa E501
        return np.concatenate((vec_1, vec_2))

    def __call__(self, c, w):
        """Call one step of the ImexA method to solve at the next timestep.

        Parameters
        ----------
        c : np.array
            The value of c at the current timestep.
        w : np.array
            The value of w at the current timestep.

        Returns
        ----------
        c_next, w_next : tuple(np.array, np.array)
            The values of c and w at the next timestep.
        """
        (c_next, w_next) = newton(self.implicit_function, self.jacobian, c, w, self.tolerance, self.max_iter) # noqa E501
        return (c_next, w_next)


class ImexB(NumericalMethod):
    """Implementation of the ImexB scheme for Cahn-Hilliard."""

    name = "ImexB"

    def __init__(self, dt, eps, lap):
        """Initialisation of ImexB method, call superclass __init__ and sets additional attributes. # noqa E501

        Additional Attributes
        ----------
        self.matrix : np.block
            A constant matrix equivalent to the Jacobian of the method, to be used \
            to solve the linear system of ImexB for the relevant values at the \
            next timesteps.
        """
        super().__init__(dt, eps, lap)
        sz = np.shape(self.laplacian)
        self.matrix = np.block([[np.identity(sz[0]), -self.timestep*self.laplacian], # noqa E501
                                [self.epsilon*self.laplacian - 2/self.epsilon*np.identity(sz[0]), np.identity(sz[0])]]) # noqa E501

    def __call__(self, c, *args):
        """Call one step of the ImexB method to solve at the next timestep.

        Parameters
        ----------
        c : np.array
            The value of c at the current timestep.

        Returns
        ----------
        c_next, w_next : tuple(np.array, np.array)
            The values of c and w at the next timestep.
        """

        vec = np.concatenate((c, 1/self.epsilon*np.power(c, 3) - 3/self.epsilon*c)) # noqa E501
        out = np.linalg.solve(self.matrix, vec)
        c_next = out[:c.size]
        w_next = out[c.size:]
        return (c_next, w_next)


class ImexC(NumericalMethod):
    """Implementation of the ImexC scheme for Cahn-Hilliard."""

    name = "ImexC"

    def __call__(self, c, *args):
        """Call one step of the ImexC method to solve at the next timestep.

        Parameters
        ----------
        c : np.array
            The value of c at the current timestep.

        Returns
        ----------
        c_next, w_next : tuple(np.array, np.array)
            The values of c and w at the next timestep.
        """
        vec = np.concatenate((c, -2/self.epsilon*np.power(c, 3)))
        matrix = np.block([[np.identity(c.size), -self.timestep*self.laplacian], # noqa E501
                           [self.epsilon*self.laplacian - 3/self.epsilon*np.diag(np.power(c, 2)) \
                           + 1/self.epsilon*np.identity(c.size), np.identity(c.size)]]) # noqa E501
        out = np.linalg.solve(matrix, vec)
        c_next = out[:c.size]
        w_next = out[c.size:]
        return (c_next, w_next)


class ImexD(NumericalMethod):
    """Implementation of the ImexD scheme for Cahn-Hilliard."""

    name = "ImexD"

    def __call__(self, c, *args):
        """Call one step of the ImexD method to solve at the next timestep.

        Parameters
        ----------
        c : np.array
            The value of c at the current timestep.

        Returns
        ----------
        c_next, w_next : tuple(np.array, np.array)
            The values of c and w at the next timestep.
        """

        vec = np.concatenate((c, -1/self.epsilon*c))
        matrix = np.block([[np.identity(c.size), -self.timestep*self.laplacian], # noqa E501
                           [-1/self.epsilon*np.diag(np.power(c,2)) \
                            + self.epsilon*self.laplacian, np.identity(c.size)]]) # noqa E501
        out = np.linalg.solve(matrix, vec)
        c_next = out[:c.size]
        w_next = out[c.size:]
        return (c_next, w_next)
