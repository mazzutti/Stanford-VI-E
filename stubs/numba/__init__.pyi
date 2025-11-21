from collections.abc import Iterable
from typing import Any, Callable, TypeVar, overload

from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

@overload
def njit(func: Callable[P, R]) -> Callable[P, R]: ...
@overload
def njit(
    *, cache: Any = ..., parallel: Any = ..., nogil: Any = ..., fastmath: Any = ...
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def njit(arg: Any) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def prange(n: int) -> Iterable[int]: ...
@overload
def prange(start: int, stop: int) -> Iterable[int]: ...
@overload
def prange(start: int, stop: int, step: int) -> Iterable[int]: ...

# No runtime implementations in stubs: keep only overload signatures for mypy.
def typeof(obj: Any) -> Any: ...

__all__ = ["njit", "prange", "typeof"]
