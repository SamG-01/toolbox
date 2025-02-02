from contextlib import contextmanager
from functools import partial

import numpy as np
from numpy.typing import NDArray

from .typehints import Callable


@contextmanager
def rel_coords(coords: NDArray[np.complex128], origin: complex, angle: float = 0):
    """Context manager for performing operations on complex vectors relative to some origin and axes.

    Args:
        coords (NDArray[np.complex128]): The coordinates to temporarily transform.
        origin (complex): The point to treat as the origin.
        angle (float, optional): The angle to consider as the positive real axis. Defaults to 0.
    """

    rotation = np.exp(1j * angle)
    try:
        coords -= origin
        coords /= rotation
        yield
    finally:
        coords *= rotation
        coords += origin


def nonlinearinterp(
    x,
    xp,
    fp,
    left,
    right,
    func: Callable[[float], float],
    inverse_func: Callable[[float], float],
):
    """_summary_

    Args:
        x (_type_): TODO
        xp (_type_): TODO
        fp (_type_): TODO
        left (_type_): TODO
        right (_type_): TODO
        func (Callable[[float], float]): TODO
        inverse_func (Callable[[float], float]): TODO

    Returns:
        _type_: _description_

    Signatures:
        (...),(n),(n),(),()->(...)
    """

    if left is None:
        left = fp[0]
    if right is None:
        right = fp[-1]

    return inverse_func(np.interp(func(x), func(xp), func(fp), func(left), func(right)))


def loginterpolate(x, xp, fp, left, right, base: int = 10) -> float:
    func = partial(np.emath.logn, n=base)
    inverse_func = partial(np.power, base)
    return nonlinearinterp(x, xp, fp, left, right, func, inverse_func)


def ndtrapz(y, *xn):
    out = y
    for x in xn:
        out = np.trapz(out, x, axis=0)
    return out
