"""Strategy pattern implementations for flexible component behavior.

This module provides abstract strategy classes and composable implementations
that allow flexible swapping of algorithms and behaviors throughout the
analysis module.

For validation strategies, see src.analysis.validators instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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
    def compute_mean(self, arr):
        """Compute mean of array."""
        pass

    @abstractmethod
    def compute_std(self, arr):
        """Compute standard deviation of array."""
        pass

    @abstractmethod
    def compute_median(self, arr):
        """Compute median of array."""
        pass


class StandardArrayStatistics(ArrayStatisticsStrategy):
    """Standard array statistics using numpy defaults."""

    def compute_mean(self, arr):
        """Compute mean using numpy."""
        import numpy as np

        return np.mean(arr)

    def compute_std(self, arr):
        """Compute standard deviation using numpy."""
        import numpy as np

        return np.std(arr)

    def compute_median(self, arr):
        """Compute median using numpy."""
        import numpy as np

        return np.median(arr)


class RobustArrayStatistics(ArrayStatisticsStrategy):
    """Robust array statistics using median and IQR."""

    def compute_mean(self, arr):
        """Compute robust mean (median)."""
        import numpy as np

        return np.median(arr)

    def compute_std(self, arr):
        """Compute robust std (IQR)."""
        import numpy as np

        q75, q25 = np.percentile(arr, [75, 25])
        return (q75 - q25) / 1.35

    def compute_median(self, arr):
        """Compute median."""
        import numpy as np

        return np.median(arr)
