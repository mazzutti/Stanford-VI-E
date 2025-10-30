"""Compatibility helpers for optional accelerated libraries (migrated).

Provides a thin shim when Numba isn't available so callers can safely
decorate functions with `njit` and use `prange` without conditional logic.
"""

from typing import Callable, Optional, Any

try:
    from numba import njit, prange  # type: ignore

    _NUMBA_AVAILABLE: bool = True
except Exception:
    _NUMBA_AVAILABLE = False

    def njit(func: Optional[Callable] = None, **kwargs: Any) -> Callable:
        """Simple no-op decorator replacement for numba.njit.

        When Numba is missing this returns the original function unchanged.
        """

        if func is None:

            def wrapper(f: Callable) -> Callable:
                return f

            return wrapper
        return func

    prange = range  # type: ignore


def numba_available() -> bool:
    """Return True when Numba is available in this environment."""

    return bool(_NUMBA_AVAILABLE)


__all__ = ["njit", "prange", "_NUMBA_AVAILABLE", "numba_available"]
