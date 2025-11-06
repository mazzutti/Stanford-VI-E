# Stub file for numba decorators
from typing import Callable, TypeVar, overload, Any

F = TypeVar("F", bound=Callable[..., Any])

@overload
def njit(
    func: F,
    *,
    parallel: bool = False,
    cache: bool = False,
    fastmath: bool = False,
    error_model: str = "default",
    nogil: bool = False,
    locals: dict[str, Any] | None = None,
    **kwargs: Any,
) -> F: ...
@overload
def njit(
    func: None = None,
    *,
    parallel: bool = False,
    cache: bool = False,
    fastmath: bool = False,
    error_model: str = "default",
    nogil: bool = False,
    locals: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Callable[[F], F]: ...
def njit(
    func: F | None = None,
    *,
    parallel: bool = False,
    cache: bool = False,
    fastmath: bool = False,
    error_model: str = "default",
    nogil: bool = False,
    locals: dict[str, Any] | None = None,
    **kwargs: Any,
) -> F | Callable[[F], F]:
    """JIT compile function for performance.

    Decorator that compiles a function to machine code at import time.

    Args:
        func: Function to compile
        parallel: Enable automatic parallelization
        cache: Cache compiled version
        fastmath: Enable fast math
        error_model: Error handling mode
        nogil: Release GIL during execution
        locals: Type map for local variables

    Returns:
        Compiled function
    """
    ...

def prange(start: int, stop: int | None = None, step: int = 1) -> range:
    """Parallel range for use in njit functions.

    Use instead of range() inside njit functions to parallelize the loop.

    Args:
        start: Start value or stop (if stop is None)
        stop: Stop value (exclusive)
        step: Step size

    Returns:
        Range object that will be parallelized
    """
    ...
