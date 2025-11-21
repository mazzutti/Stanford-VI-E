"""Type aliases and type variables for processor operations."""

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

# Duplicate-code warnings are expected for similar __all__ exports across
# small model modules; silence for this module only (placed before __all__).
# pylint: disable=duplicate-code

__all__ = [
    "Float64Array",
    "Int64Array",
    "BoolArray",
    "FloatingArray",
    "IntegerArray",
    "T",
    "StatsType",
    "FilterResult",
    "CorrelationResult",
    "OptionalArrayPair",
    "ArrayNamePair",
    "FloatingArrayNamePair",
    "CorrelationFunction",
    "AttributeArrayDict",
]
# Allow non-PascalCase TypeVar names used across the codebase for readability
# in type hints (StatsType etc.). This is stylistic and safe to relax.

# Specific array types for cleaner type hints
Float64Array = NDArray[np.float64]
"""Type alias for 64-bit float arrays."""

Int64Array = NDArray[np.int64]
"""Type alias for 64-bit integer arrays."""

BoolArray = NDArray[np.bool_]
"""Type alias for boolean arrays."""

FloatingArray = NDArray[np.floating[Any]]
"""Type alias for any floating-point precision arrays."""

IntegerArray = NDArray[np.integer[Any]]
"""Type alias for integer arrays (includes facies labels and indices)."""

# Type variables for generic programming
T = TypeVar("T", bound=NDArray[Any])
"""Generic type variable for array types."""

StatsType = TypeVar("StatsType")
"""Generic type variable for statistics types."""

# Allow non-PascalCase TypeVar names used across the codebase for readability
# in type hints (StatsType etc.). This is stylistic and safe to relax.

# Common result type aliases
FilterResult = tuple[Float64Array, Float64Array, int]
"""Type alias for (filtered_arr1, filtered_arr2, n_removed)."""

CorrelationResult = tuple[float, float]
"""Type alias for (correlation_coefficient, p_value) result tuples."""

OptionalArrayPair = tuple[Float64Array | None, Float64Array | None]
"""Type alias for validation results: (filtered_arr1, filtered_arr2) or (None, None)."""

ArrayNamePair = tuple[NDArray[Any], str]
"""Type alias for generic (array, name) tuples used in validation."""

FloatingArrayNamePair = tuple[FloatingArray, str]
"""Type alias for (floating_array, name) tuples used in array validation."""

CorrelationFunction = Callable[
    [NDArray[np.float64], NDArray[np.float64]], CorrelationResult
]
"""Type alias for correlation functions: takes two float arrays, returns (coeff, p_value)."""

AttributeArrayDict = dict[str, FloatingArray]
"""Type alias for dict mapping attribute names to floating-point arrays."""
