"""Strategy pattern implementations for flexible component behavior.

This module provides abstract strategy classes and composable implementations
that allow flexible swapping of algorithms and behaviors throughout the
analysis module.

For validation strategies, see src.analysis.validators instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Union, cast

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=Union[int, float, np.number[Any]])

__all__ = [
    "ArrayStatisticsStrategy",
    "StandardArrayStatistics",
    "RobustArrayStatistics",
]


# Array Statistics Strategies
class ArrayStatisticsStrategy(ABC):
    """Abstract base for array statistics computation strategies.

    Enables pluggable computation of array statistics with different
    algorithms (standard vs. robust, for example).
    """

    @abstractmethod
    def compute_mean(self, arr: NDArray[Any]) -> Union[int, float, np.number[Any]]:
        """Compute mean of array."""
        pass

    @abstractmethod
    def compute_std(self, arr: NDArray[Any]) -> Union[int, float, np.number[Any]]:
        """Compute standard deviation of array."""
        pass

    @abstractmethod
    def compute_median(self, arr: NDArray[Any]) -> Union[int, float, np.number[Any]]:
        """Compute median of array."""
        pass


class StandardArrayStatistics(ArrayStatisticsStrategy):
    """Standard array statistics using numpy defaults."""

    def compute_mean(self, arr: NDArray[Any]) -> Union[int, float, np.number[Any]]:
        """Compute mean using numpy."""
        return cast(Union[int, float, np.number[Any]], np.mean(arr))

    def compute_std(self, arr: NDArray[Any]) -> Union[int, float, np.number[Any]]:
        """Compute standard deviation using numpy."""
        return cast(Union[int, float, np.number[Any]], np.std(arr))

    def compute_median(self, arr: NDArray[Any]) -> Union[int, float, np.number[Any]]:
        """Compute median using numpy."""
        return cast(Union[int, float, np.number[Any]], np.median(arr))


class RobustArrayStatistics(ArrayStatisticsStrategy):
    """Robust array statistics using median and IQR."""

    def compute_mean(self, arr: NDArray[Any]) -> Union[int, float, np.number[Any]]:
        """Compute robust mean (median)."""
        return cast(Union[int, float, np.number[Any]], np.median(arr))

    def compute_std(self, arr: NDArray[Any]) -> Union[int, float, np.number[Any]]:
        """Compute robust std (IQR)."""
        q75, q25 = np.percentile(arr, [75, 25])
        return cast(Union[int, float, np.number[Any]], (q75 - q25) / 1.35)

    def compute_median(self, arr: NDArray[Any]) -> Union[int, float, np.number[Any]]:
        """Compute median."""
        return cast(Union[int, float, np.number[Any]], np.median(arr))
