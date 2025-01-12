from functools import partial

import numpy as np

from .typehints import Callable


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
