from functools import wraps

from ..typehints import Any, Callable, Concatenate, Decorator, DecoratorFactory

__all__ = ["parametrized"]


def parametrized[**P, T](
    decorator: Callable[Concatenate[Callable[P, T], ...], Decorator[P, T]],
) -> DecoratorFactory:
    """Decorator for decorating a decorator with additional arguments
    to convert it into a decorator factory. In other words, this
    is a decorator factory factory.

    https://stackoverflow.com/questions/5929107/decorators-with-parameters

    Args:
        decorator (Callable[Concatenate[Function, ...], Decorator]): A
        callable that takes a function and any number of parameters
        and spits out a wrapped function.

    Returns:
        DecoratorFactory: A proper decorator factory that applies
        functools.wraps to the wrapper and takes the previously
        mentioned parameters as arguments.
    """

    def decorator_factory(*dargs: Any, **dkwargs: Any) -> Decorator[P, T]:
        # we can't use functools.wraps here, as you can't
        # specify *dargs to start at index 1

        try:
            f = dargs[0]
        except IndexError:
            f = None
        else:
            if callable(f):
                dargs = dargs[1:]
            else:
                f = None

        @wraps(decorator)
        def filled_decorator(function: Callable[P, T]) -> Decorator[P, T]:
            wrapper = decorator(function, *dargs, **dkwargs)
            return wraps(function)(wrapper)

        if f is None:
            return filled_decorator
        return filled_decorator(f)

    return decorator_factory
