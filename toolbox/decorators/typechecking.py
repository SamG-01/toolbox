from inspect import _empty, signature

from ..typehints import Any, Callable, FDescriptor, Iterable, Self
from .general import parametrized


def check_type(arg: Any, expected_type: type | Iterable[type], label: Any) -> None:
    """Checks the type of arg.

    Args:
        arg (Any): The variable to check the type of.
        expected_type (type): The expected type to check against.
        label (Any): What to refer to arg as in the TypeError.

    Raises:
        TypeError: If type(arg) is not expected_type, an exception is raised.
    """

    # if expected_type is Any or an empty annotation,
    # then the check automatically succeeds
    if expected_type in (Any, _empty):
        return

    # otherwise, check the two types
    actual = type(arg)
    try:
        expected_types = tuple(expected_type)
        error = f"Type {actual} of argument {label} is not in {expected_types}"
    except TypeError:
        expected_types = (expected_type,)
        error = f"Argument {label} has type {actual} and not {expected_type}"

    for expected in expected_types:
        if actual is expected:
            return
    raise TypeError(error)


@parametrized
def typeguard[**P, T](
    func: Callable[P, T], /, *targs: type, **tkwargs: type
) -> Callable[P, T]:
    """A decorator factory that checks the arguments of a function
    call against the supplied types before running it and raises a
    TypeError if they differ. If a type isn't provided for a parameter,
    this function will use its type annotations as a fallback.

    against
    the supplied types before running it and raises a TypeError if they differ.

    https://stackoverflow.com/a/26151604

    Args:
        func (Callable): The function to be decorated.
        targs (type): The types for the positional arguments.
        tkwargs (type): The types for the keyword arguments.
    """

    # gets the function signature
    fsig = signature(func)

    # identifies which parameters are type guarded
    tcallargs = fsig.bind_partial(*targs, **tkwargs)

    def wrapper(*fargs: P.args, **fkwargs: P.kwargs) -> T:
        # binds fargs and fkwargs to the signature
        fcallargs = fsig.bind_partial(*fargs, **fkwargs)

        # loops through the typeguarded parameters
        # and confirms their types
        for fparamname, fcallarg in fcallargs.arguments.items():
            fparam = fsig.parameters[fparamname]
            tcallarg = tcallargs.arguments.get(fparamname, fparam.annotation)

            # if this is a regular param, just check it normally
            if int(fparam.kind) == 1:
                check_type(fcallarg, tcallarg, fparamname)
                continue

            # otherwise, treat it as an *args or **kwargs parameter
            if isinstance(fcallarg, tuple):
                fcallarg = dict(enumerate(fcallarg))
            for k, v in fcallarg.items():
                check_type(v, tcallarg, f"{fparamname}[{k}]")

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
                # validates the arguments against the implementation signature
                fsig = signature(func)
                fcallargs = fsig.bind(*fargs, **fkwargs)

                # if typechecking is enabled, check the argument
                # types against the implementation's type annotations
                if self.typecheck:
                    for fparam, fcallarg in fcallargs.arguments.items():
                        tcallarg = fsig.parameters[fparam].annotation
                        check_type(fcallarg, tcallarg, fparam)
            except TypeError as e:
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
            Any: The output of the implementation.
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
        """Returns a copy of the multipledispatch object
        bound to its instance/owner, as appropriate.

        Args:
            instance (object): The instance to bind to.
            owner (type | None, optional): The class of instance,
            or the class to bind to. Defaults to None.

        Returns:
            Self: A copy of self with the appropriate bindings.
        """

        copy = self.__class__(None, self.typecheck, instance, owner)
        return copy.register(*self.registry)
