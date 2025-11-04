"""Accelerated library utilities.

NOTE: This module is deprecated. Import numba directly instead:
    from numba import njit, prange

Numba is a required dependency and should be imported directly
from its main package rather than through this compatibility layer.
"""

from numba import njit, prange

__all__ = ["njit", "prange"]
