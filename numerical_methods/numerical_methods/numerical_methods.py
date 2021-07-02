"""
A Class hierarchy defining six Numerical Methods for the Cahn-Hilliard PDE.

This script also contains functions to produce initial conditions in 1D and 2D
and functions that construct the finite difference Laplacian in 1D and 2D
"""
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from math import pi, cos


class ConvergenceError(Exception):
    """Exception raised if the solver fails to converge."""
    pass


def laplacian1D(n, h):
    """Construct a second order finite difference Laplacian operator,
       associated with Neumann boundary conditions, in 1D.

    Parameters
    ----------
    n : int
        The total number of for the discretised mesh.
    h : float
        The size of the mesh

    Returns
    ----------
    M : sparse.dia_matrix
        The discretised Laplacian operator, in sparse form.
    """
    M = np.diag(-2*np.ones(n+1)) + np.diag(np.ones(n), 1) \
        + np.diag(np.ones(n), -1)
    M[0, 1] = 2
    M[n, n-1] = 2
    M = 1/h**2 * M
    M = sparse.dia_matrix(M)
    return M


def laplacian2D(n, h):
    """Construct a second order finite difference Laplacian operator,
    associated with Neumann boundary conditions, in 2D.

    Parameters
    ----------
    n : int
        The total number of for the discretised mesh.
    h : float
        The size of the mesh

    Returns
    ----------
    M : sparse.csr_matrix
        The discretised Laplacian operator, in sparse form.
    """
    m = laplacian1D(n, h)
    M = sparse.kron(sparse.identity(n+1), m) \
        + sparse.kron(m, sparse.identity(n+1))
    return M


def initialise1D(omega_domain, laplacian, epsilon, switch):
    """Set initial conditions to be either smooth or random in 1D

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
    c0 : np.array
        The c0 initial data.
    w0 : np.array
        The w0 initial data.
    name : str
        "smooth" or "random" stating the type of initial conditions.
    """
    if switch == 0:
        c0 = np.transpose(np.array([cos(pi*x) for x in omega_domain]))
        w0 = 1/epsilon*(np.power(c0, 3) - c0) + epsilon*pi**2*c0
        name = "smooth"
        return c0, w0, name
    elif switch == 1:
        c0 = np.transpose(np.random.normal(0, 0.25, len(omega_domain)))
        w0 = 1/epsilon*(np.power(c0, 3) - c0) + epsilon*laplacian@c0
        name = "random"
        return c0, w0, name
    else:
        raise NotImplementedError(f"You must set the switch = 0 or 1,\
                                  not {switch}")


def initialise2D(omega_domain, laplacian, epsilon, switch):
    """Set initial conditions to be either smooth or random in 2D

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
    c0 : np.array
        The c0 initial data.
    w0 : np.array
        The w0 initial data.
    name : str
        "smooth" or "random" stating the type of initial conditions.
    """
    xx, yy = omega_domain
    if switch == 0:
        c_grid = np.cos(pi*xx)*np.cos(pi*yy)
        c0 = np.transpose(np.reshape(c_grid, c_grid.size))
        name = "smooth"
    elif switch == 1:
        c0 = np.transpose(np.random.normal(0, 1, xx.size))
        name = "random"
    else:
        raise NotImplementedError(f"You must set the switch = 0 or 1,\
                                  not {switch}")
    w0 = 1/epsilon*(np.power(c0, 3) - c0) + epsilon*laplacian@c0

    return c0, w0, name


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
        lap : sparse.dia_matrix or sparse.csr_matrix
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

    def __eq__(self, other):
        """Equality check for numerical methods."""
        return isinstance(other, type(self))\
            and self.timestep == other.timestep \
            and self.epsilon == other.epsilon \
            and self.laplacian.toarray().all() \
            == other.laplacian.toarray().all()

    def newton(self, f, df, c0, w0, tolerance, max_iter):
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
        tolerance : float
            The solver tolerance, convergence is achieved when
            max(abs(f(c0, c, w))) < 0
        max_iter : int
            The maximum number of acceptable iterations before the solver
            is determined to have failed.

        Returns
        ----------
        c, w : tuple(np.array, np.array)
            The approximate root computed using Newton iteration.
        """
        n = 0
        c = c0
        w = w0
        while max(abs(f(c0, c, w))) > tolerance and n <= max_iter:
            out = - linalg.spsolve(df(c), f(c0, c, w))
            c = c + out[:c.size]
            w = w + out[c.size:]
            n += 1

        if n <= max_iter and max(abs(f(c0, c, w))) <= tolerance:
            return (c, w)
        else:
            raise ConvergenceError("Convergence Failure")


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
        w_next = (1/self.epsilon)*(np.power(c_next, 3) - c_next)\
            - self.epsilon*self.laplacian@c_next

        return (c_next, w_next)


class Implicit(NumericalMethod):
    """Implementation of Euler's Implicit method for Cahn-Hilliard."""

    name = "Implicit"

    def __init__(self, dt, eps, lap, tolerance, max_iter):
        """Initialisation of Implicit method, call superclass __init__
           and sets additional attributes.

        Additional Parameters
        ----------
        tolerance : float
            The tolerance to be used in Newton-Raphson root finding.
        max_iter : int
            The maximum number of iteration to be used in Newton-Raphson
            root finding.
        """
        super().__init__(dt, eps, lap)
        self.tolerance = tolerance
        self.max_iter = max_iter

    def __eq__(self, other):
        return super().__eq__(self, other) \
               and self.tolerance == other.tolerance \
               and self.max_iter == other.max_iter

    def jacobian(self, c):
        """Jacobian associated with the Implicit scheme.

        Parameters
        ----------
        c : np.array
            The vector on which the Jacobian depends.

        Returns
        ----------
        jacobian : sparse.csr_matrix
            A block matrix representation of the Jacobian.
        """
        eye = sparse.identity(c.size)
        jacobian = sparse.bmat([[eye, -self.timestep*self.laplacian],
                                [1/self.epsilon*(-3*sparse.diags(np.power(c, 2)) # noqa E501
                                 + eye) + self.epsilon*self.laplacian, eye]])
        jacobian = sparse.csr_matrix(jacobian)
        return jacobian

    def residual_function(self, c, c_next, w_next):
        """Function associated with the implicit method.

        Parameters
        ----------
        c : np.array
            The value of c at the previous timestep that defines the current
            function, a constant at each iteration.
        c_next : np.array
            The value of c at the next timestep, that we seek to solve for
            using Newton iterations.
        w_next : np.array
            The value of w at the next timestep, that we also seek to solve
            for using Newton iterations.

        Returns
        ----------
        output : np.array
            The output of the function.
        """
        vec_1 = c_next-c-self.timestep*self.laplacian@w_next
        vec_2 = w_next - 1/self.epsilon*np.power(c_next, 3)\
            + 1/self.epsilon*c_next + self.epsilon*self.laplacian@c_next
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

        c_next, w_next = super().newton(self.residual_function, self.jacobian,
                                        c, w, self.tolerance, self.max_iter)
        return (c_next, w_next)


class ImexA(NumericalMethod):
    """Implementation of the ImexA scheme for Cahn-Hilliard."""

    name = "ImexA"

    def __init__(self, dt, eps, lap, tolerance, max_iter):
        """Initialisation of ImexA method, call superclass __init__
        and sets additional attributes.

        Additional Parameters
        ----------
        tolerance : float
            The tolerance to be used in Newton-Raphson root finding.
        max_iter : int
            The maximum number of iteration to be used in Newton-Raphson
            root finding.
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
        eye = sparse.identity(c.size)
        jacobian = sparse.bmat([[eye, -self.timestep*self.laplacian],
                                [- 3/self.epsilon*sparse.diags(np.power(c, 2))
                                 + self.epsilon*self.laplacian, eye]])
        jacobian = sparse.csr_matrix(jacobian)
        return jacobian

    def residual_function(self, c, c_next, w_next):
        """Function associated with the ImexA method.

        Parameters
        ----------
        c : np.array
            The value of c at the previous timestep that defines the current
            function, a constant at each iteration.
        c_next : np.array
            The value of c at the next timestep, that we seek to solve for
            using Newton iterations.
        w_next : np.array
            The value of w at the next timestep, that we also seek to solve
            for using Newton iterations.

        Returns
        ----------
        output : np.array
            The output of the function.
        """
        vec_1 = c_next-c-self.timestep*self.laplacian@w_next
        vec_2 = w_next - 1/self.epsilon*np.power(c_next, 3) + 1/self.epsilon*c\
            + self.epsilon*self.laplacian@c_next
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
        c_next, w_next = super().newton(self.residual_function, self.jacobian,
                                        c, w, self.tolerance, self.max_iter)
        return (c_next, w_next)


class ImexB(NumericalMethod):
    """Implementation of the ImexB scheme for Cahn-Hilliard."""

    name = "ImexB"

    def __init__(self, dt, eps, lap, *args, **kwargs):
        """Initialisation of ImexB method, call superclass __init__
        and sets additional attributes.

        Additional Attributes
        ----------
        self.matrix : np.block
            A constant matrix equivalent to the Jacobian of the method,
            to be used to solve the linear system of ImexB for the
            relevant values at the next timesteps.
        """
        super().__init__(dt, eps, lap)
        sz = np.shape(self.laplacian)
        eye = sparse.identity(sz[0])
        matrix = sparse.bmat([[eye, -self.timestep*self.laplacian],
                              [self.epsilon*self.laplacian
                               - 2/self.epsilon*eye, eye]])
        self.matrix = sparse.csr_matrix(matrix)

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

        vec = np.concatenate((c, 1/self.epsilon*np.power(c, 3)
                              - 3/self.epsilon*c))
        out = linalg.spsolve(self.matrix, vec)
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
        eye = sparse.identity(c.size)
        matrix = sparse.bmat([[eye, -self.timestep*self.laplacian],
                              [self.epsilon*self.laplacian
                               - 3/self.epsilon*sparse.diags(np.power(c, 2))
                               + 1/self.epsilon*eye, eye]])
        matrix = sparse.csr_matrix(matrix)
        out = linalg.spsolve(matrix, vec)
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
        eye = sparse.identity(c.size)
        matrix = sparse.bmat([[eye, -self.timestep*self.laplacian],
                             [-1/self.epsilon*sparse.diags(np.power(c, 2))
                             + self.epsilon*self.laplacian, eye]])
        matrix = sparse.csr_matrix(matrix)
        out = linalg.spsolve(matrix, vec)
        c_next = out[:c.size]
        w_next = out[c.size:]
        return (c_next, w_next)
