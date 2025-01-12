from collections.abc import Callable
from typing import Any, Concatenate, Protocol, Self

import nptyping as npt
from pint import Quantity, UnitRegistry

ureg = UnitRegistry(auto_reduce_dimensions=True)

__all__ = [
    "Any",
    "Callable",
    "Concatenate",
    "Decorator",
    "DecoratorFactory",
    "npt",
    "Quantity",
    "ureg",
    "Self",
]

type Decorator[**P, T] = Callable[[Callable[P, T]], Callable[P, T]]
type DecoratorFactory = Callable[..., Decorator] | Decorator

type chebfun[n: int] = npt.NDArray[npt.Shape[f"{n}"], npt.Float64]
type chebop[n: int] = npt.NDArray[npt.Shape[f"{n}, {n}"], npt.Float64]


class Descriptor(Protocol):
    def __get__(self, instance, owner=None): ...


class FDescriptor(Protocol):
    def __get__(self, instance, owner=None) -> Callable: ...
