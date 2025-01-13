from inspect import _empty, signature
from types import UnionType

from ..typehints import Any, Callable, FDescriptor, Iterable, Self
from .general import parametrized

__all__ = ["check_type", "check_call", "typeguard", "multipledispatch"]

types = type | Iterable[type] | UnionType


class TypeCheckError(TypeError):
    pass


def check_type(arg: Any, expected_type: types, label: str | Any) -> None:
    """Checks the type of arg.

    Args:
        arg (Any): The variable to check the type of.
        expected_type (type | Iterable[type] | UnionType): The
        expected type(s) to check against.
        label (str | Any): What to refer to arg as in the TypeError.

    Raises:
        TypeCheckError: If type(arg) is not (in) expected_type.
    """

    actual = type(arg)
    if isinstance(expected_type, type):
        # normal case where expected_type is a type
        expected_types = (expected_type,)
        error = f"Argument {label} has type {actual} and not {expected_type}"
    else:
        # if expected_type is a tuple or UnionType, then check its elements
        expected_types = tuple(getattr(expected_type, "__args__", expected_type))
        error = f"Type {actual} of argument {label} is not in {expected_types}"

    for expected in expected_types:
        # if expected_type is Any or an empty annotation,
        # then the check automatically succeeds
        if expected in (actual, Any, _empty):
            return
    raise TypeCheckError(error)


def check_call[**P, T](
    func: Callable[P, T],
    /,
    fargs: tuple[Any],
    fkwargs: dict[str, Any],
    targs: tuple[types] = None,
    tkwargs: dict[str, types] = None,
    check_types: bool = True,
    use_annotations: bool = True,
) -> None:
    """Checks that the arguments passed into the function are
    compatible with its signature and the supplied types, if any.

    Args:
        func (Callable): The function to check.
        fargs (tuple[Any]): The positional arguments to check against func.
        fkwargs (dict[str, Any]): The keyword arguments to check against func.
        targs (tuple[types], optional): The expected types for the positional
        arguments.
        tkwargs (dict[str, types], optional): The expected types for the
        keyword arguments.
        check_types (bool, optional): Whether to check the types of
        fargs and fkwargs against those of targs and tkwargs. Defaults to True.
        use_annotations (bool, optional): Whether to fall back on func's
        type annotations for gaps in targs and tkwargs. Defaults to True.

    Raises:
        TypeError: If func(*fargs, **fkwargs) is not a valid
        signature binding.
        TypeCheckError: If any of the call arguments don't correspond
        to the type arguments.
    """

    targs = () if targs is None else targs
    tkwargs = {} if tkwargs is None else tkwargs

    fsig = signature(func)
    fcallargs = fsig.bind_partial(*fargs, **fkwargs)

    if not check_types:
        return

    tcallargs = fsig.bind_partial(*targs, **tkwargs)

    failures = []
    for fparamname, fcallarg in fcallargs.arguments.items():
        fparam = fsig.parameters[fparamname]
        tcallarg = tcallargs.arguments.get(
            fparamname, fparam.annotation if use_annotations else _empty
        )

        # if this is a regular param, just check it normally
        try:
            if int(fparam.kind) == 1:
                check_type(fcallarg, tcallarg, fparamname)
                continue

            # otherwise, treat it as an *args or **kwargs parameter
            if isinstance(fcallarg, tuple):
                fcallarg = dict(enumerate(fcallarg))
            for k, v in fcallarg.items():
                check_type(v, tcallarg, f"{fparamname}[{k}]")
        except TypeCheckError as e:
            failures.append(str(e))

    if failures:
        failures.insert(0, "Function arguments did not match expected types")
        raise TypeCheckError("\n".join(failures))


@parametrized
def typeguard[**P, T](
    func: Callable[P, T], /, *targs: type | tuple[type], **tkwargs: type
) -> Callable[P, T]:
    """A decorator factory that checks the arguments of a function
    call against the supplied types before running it and raises a
    TypeError if they differ. If a type isn't provided for a parameter,
    this function will use its type annotations as a fallback.

    https://stackoverflow.com/a/26151604

    Args:
        func (Callable): The function to be decorated.
        targs (type): The types for the positional arguments.
        tkwargs (type): The types for the keyword arguments.
    """

    def wrapper(*fargs: P.args, **fkwargs: P.kwargs) -> T:
        # checks the call beforehand
        check_call(func, fargs, fkwargs, targs, tkwargs, True, True)

        # otherwise, run the function as normal
        return func(*fargs, **fkwargs)

    return wrapper


class multipledispatch:
    """Overload a function via explicitly matching arguments to parameters on call.

    Attributes:
        registry (list[Callable | FDescriptor]): A list of function
        implementations to dispatch to according to arguments and
        argument type, in order of most to least recently registered.
        These must be callables or descriptors that return callables,
        such as classmethods and staticmethods.

    https://stackoverflow.com/a/63810326"""

    def __init__(
        self,
        base_case: Callable | FDescriptor | None = None,
        typecheck: bool = True,
        _instance: object | None = None,
        _owner: type | None = None,
    ) -> None:
        """Decorates a function or descriptor.

        Args:
            base_case (Callable | FDescriptor | None, optional): The base
            implementation to consider. If None, this can be added later
            via __call__.
            typecheck (bool, optional): Whether to compare against each
            implementation's type annotations in addition to comparing
            signatures. Defaults to True.
            _instance (object | None, optional): The class instance to bind
            any descriptor implementations to. Supplied automatically by
            __get__. Defaults to None.
            _owner (type | None, optional): The class to bind any
            descriptor implementations to. Supplied automatically by
            __get__. Defaults to None.
        """

        self.registry: list[Callable | FDescriptor] = []
        self.typecheck = typecheck

        if base_case is not None:
            self.register(base_case)

        self._instance = _instance
        self._owner = _owner

    def __len__(self) -> int:
        return len(self.registry)

    def register(self, *implementations: Callable | FDescriptor) -> Self:
        """Registers one or more function implementations.

        Arguments:
            *implementations (Callable | FDescriptor): The implementation(s)
            to add to the registry.

        Returns:
            Self: The newly wrapped multipledispatch object.
        """

        for implementation in implementations:
            self.registry.append(implementation)
        return self

    def dispatcher(self, *fargs: Any, **fkwargs: Any) -> tuple[Callable, list[str]]:
        """Finds the implementation that fits the call arguments.

        Returns:
            func (Callable): The appropriate implementation, bound to
            the same instance and/or class as the multipledispatch object
            if appropriate.
            failures (list[str]): The error messages for each failed implementation.
        """

        failures = ["No implementation found for the supplied arguments"]

        # checks the registry for implementations that work
        for n, implementation in enumerate(self.registry[::-1]):
            # if self._instance and self._owner are specified,
            # bind the implementation to them using
            # its __get__ method
            if self._instance is None and self._owner is None:
                func: Callable = implementation
            else:
                func: Callable = implementation.__get__(self._instance, self._owner)  # pylint: disable=C2801

            # checks if the implementation works for these arguments
            try:
                check_call(func, fargs, fkwargs, check_types=self.typecheck)
            except TypeCheckError as e:
                failures.append(f"Implementation {len(self) - n}: {e}")
            else:
                return func, failures

        # if a good implementation isn't found, return NotImplemented
        return NotImplemented, failures

    def __call__(self, *fargs: Any, **fkwargs: Any) -> Any:
        """Finds the right implementation for the supplied
        arguments and passes them into it.

        Raises:
            NotImplementedError: If no valid implementations are found.

        Returns:
            Self: If the registry is empty, decorate the new base case.
            Any: If the registry isn't empty, find the appropriate
            implementation and call it.
        """

        # if there is no base case, decorate it now
        if callable(fargs[0]) and not self.registry:
            self.register(fargs[0])
            return self

        # otherwise, get the correct implementation from the dispatcher
        implementation, failures = self.dispatcher(*fargs, **fkwargs)
        if implementation is NotImplemented:
            raise NotImplementedError("\n".join(failures))
        return implementation(*fargs, **fkwargs)

    def __get__(self, instance: object, owner: type | None = None) -> Self:
        """Returns Self or a copy of Self bound to the
        given instance/owner, as appropriate.

        Args:
            instance (object): The instance to bind to.
            owner (type | None, optional): The class of instance,
            or the class to bind to. Defaults to None.

        Returns:
            Self: Self, or a copy with the appropriate bindings
            if Self's bindings are different.
        """

        if instance is self._instance and owner is self._owner:
            return self
        copy = self.__class__(None, self.typecheck, instance, owner)
        return copy.register(*self.registry)
