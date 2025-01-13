from time import sleep

from ..typehints import Callable
from .general import parametrized


@parametrized
def retry[**P, T](
    func: Callable[P, T],
    /,
    max_retries: int = 3,
    exceptions: tuple[type[Exception]] = (Exception,),
    delay: float = 0,
) -> Callable[P, T]:
    """Attempts to retry a function a set number of times.

    Args:
        func: The function to decorate.
        max_retries: The maximum number of times to attempt to
        rerun the function if an exception is caught.
        exceptions: The exceptions to catch and rerun the function for.
        delay: The delay to add between attempts. Defaults to 0.

    https://favtutor.com/articles/top-python-decorators/

    Raises:
        RuntimeError: If all attempts fail, this is raised.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                print(f"Attempt {attempt} failed: {e}")
                if delay > 0 and attempt < max_retries:
                    sleep(delay)
                elif attempt == max_retries:
                    raise RuntimeError(
                        f"Function {func.__name__} failed after {max_retries + 1} attempts."
                    ) from e
        return func(*args, **kwargs)

    return wrapper
