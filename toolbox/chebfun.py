from contextlib import suppress

import numpy as np

from . import typehints as types


def chebspace(a: float = -1, b: float = 1, n: int = 50) -> types.chebfun:
    """Creates a Chebyshev grid of size n on [a, b].

    Args:
        a (float, optional): The first endpoint of the grid. Defaults to -1.
        b (float, optional): The second endpoint of the grid. Defaults to 1.
        n (int, optional): The number of endpoints in the grid. Defaults to 50.

    Signatures:
        (...),(...)->(n,...)

    Returns:
        _type_: A range between a and b that is evenly spaced in theta.
    """

    pi = np.pi * np.ones_like(b)
    theta = np.linspace(pi, 0, n)
    grid = np.cos(theta)

    return (b + a) / 2 + (b - a) / 2 * grid


@types.ureg.wraps("=L**-1", "=L")
def _D_op(x_i: types.chebfun) -> types.chebop:
    N = x_i.size
    i = np.arange(N)

    x_i = x_i[None, :]

    c_i = np.ones_like(i)
    c_i[0] = c_i[-1] = 2
    c_i = (c_i * (-1) ** i)[:, None]

    dx = x_i - x_i.T
    identity = np.eye(N)

    D1 = (c_i) @ (1 / c_i.T) / (dx + identity)
    D1[i, i] -= D1.sum(axis=1)
    return D1


def D_op(x_i: types.chebfun | types.Quantity, k: int = 1) -> types.chebop:
    """Returns the k-th derivative matrix on a Chebyshev grid.

    Args:
        x_i (1d array, optional): A Chebyshev grid of size N.
        k (non-negative int): What number derivative to take.

    Returns:
        D_k (2d array): D is the N by N matrix that satisfies
        `f_j l_j'(x_i) = f_j D_ij and thus (f')_j = D_ij f_j. D**k
        is thus the matrix for the k-th derivative of data on the grid.


    Signatures:
        (N)->(N,N)
    """

    assert x_i.ndim == 1
    N = x_i.size

    if k == 0:
        return np.eye(N)

    D1 = _D_op(x_i)
    D = D1.copy()

    for _ in range(k - 1):
        D @= D1
    with suppress(AttributeError):
        if D.dimensionless:
            return D.m
    return D


def derivative(y_i: types.chebfun, x_i: types.chebfun, k: int = 1) -> types.chebfun:
    """Computes the n-th derivative of
    data y_i on a Chebyshev grid x_i.

    Args:
        y_i (1d array): The data to differentiate.
        x_i (1d array, optional): The grid to differentiate on.
        Defaults to a grid on [-1, 1].
        k (int, optional): The number of times to differentiate. Defaults to 1.

    Signatures:
        (N),(N)->(N)
    """

    return D_op(x_i, k) @ y_i


@types.ureg.wraps(
    ("joule", "meter**-0.5"),
    ("joule", "meter", "joule**0.5 * meter", None),
    strict=False,
)
def quantumstates(
    V: types.chebfun, x: types.chebfun, h: float = 0.1, N: int = 10
) -> tuple[types.npt.NDArray, types.npt.NDArray]:
    """Finds the eigenenergies and eigenstates of a one-dimensional,
    one-particle quantum system, given a Chebyshev grid that the
    particle is confined to.

    Args:
        V (types.chebfun): The values of the potential on the grid.
        x (types.chebfun): The Chebyshev grid the particle is confined to.
        h (float, optional): The value of -hbar/sqrt(2m). Defaults to 0.1.
        N (int, optional): The number of eigensolutions to find. Defaults to 10.

    Returns:
        E (types.npt.NDArray): The first N eigenenergies.
        psi (types.npt.NDArray): The first N eigenstates.
    """

    # constructs the Hamiltonian operator
    H = -(h**2) * D_op(x, 2) + np.diag(V)

    # applies dirichlet boundary conditions
    H[0, 0] = H[-1, -1] = 1
    H[0, 1:] = H[-1, :-1] = 0

    # solves the eigenvalue problem
    E, psi = np.linalg.eig(H)

    # removes nonsensical results by re-enforcing the boundary conditions
    valid = np.isclose(psi[0], 0) & np.isclose(psi[-1], 0)
    E, psi = E[valid], psi[:, valid]

    # sorts the results and selects the first N of them
    idx = E.argsort()
    E, psi = E[idx][:N], psi[:, idx].T[:N]

    # normalizes the wavefunctions
    C2 = np.trapz(np.abs(psi) ** 2, x, axis=1)
    psi /= np.sqrt(C2)[:, None]

    return E, psi
