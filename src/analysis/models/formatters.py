"""Shared formatting strategy for statistical models.

This module provides consistent formatting for statistical models across
the analysis package, eliminating duplication of __repr__ and __str__
methods in individual model classes.

Strategy Pattern: FormattableModel uses composition with StatisticsFormatter
for flexible, reusable formatting behavior.
"""

from __future__ import annotations
from typing import Dict, ClassVar
from abc import ABC, abstractmethod
import numpy as np

__all__ = ["StatisticsFormatter", "FormattableModel"]


class StatisticsFormatter:
    """Strategy for formatting statistical data with configurable precision.

    Handles formatting of individual statistics and collections of statistics
    with consistent precision control.

    Example:
        >>> formatter = StatisticsFormatter(precision=4)
        >>> formatter.format_stat("mean", 3.14159)
        'mean=3.1416'
    """

    def __init__(self, precision: int = 4):
        """Initialize formatter with desired precision.

        Args:
            precision: Number of decimal places for floating point values
        """
        self.precision = precision

    def format_stat(self, name: str, value: float) -> str:
        """Format a single statistic value.

        Args:
            name: Name of the statistic
            value: Float value (may be NaN)

        Returns:
            Formatted string like "mean=3.1416" or "count=nan"
        """
        if np.isnan(value):
            return f"{name}=nan"
        return f"{name}={value:.{self.precision}f}"

    def format_stats_dict(self, stats_dict: Dict[str, float]) -> str:
        """Format all statistics as comma-separated string.

        Args:
            stats_dict: Dictionary mapping stat names to float values

        Returns:
            Comma-separated formatted stats like "mean=3.1416, std=0.5000"
        """
        parts = [self.format_stat(k, v) for k, v in stats_dict.items()]
        return ", ".join(parts)

    def format_table(self, stats_dict: Dict[str, float]) -> str:
        """Format statistics as table-like string for multi-line output.

        Args:
            stats_dict: Dictionary mapping stat names to float values

        Returns:
            Multi-line formatted string with right-aligned names
        """
        lines = []
        max_key_len = max(len(k) for k in stats_dict.keys()) if stats_dict else 0

        for k, v in stats_dict.items():
            if np.isnan(v):
                lines.append(f"  {k:<{max_key_len}}: nan")
            else:
                lines.append(f"  {k:<{max_key_len}}: {v:.{self.precision}f}")

        return "\n".join(lines)


class FormattableModel(ABC):
    """Base class for models that need consistent statistical formatting.

    Provides standardized __repr__ and __str__ implementations using
    the Strategy pattern with StatisticsFormatter.

    Subclasses must implement get_stats_dict() to provide statistics
    for formatting.

    Example:
        >>> @dataclass
        ... class MyStats(FormattableModel):
        ...     count: int
        ...     mean: float
        ...
        ...     def get_stats_dict(self) -> Dict[str, float]:
        ...         return {"count": float(self.count), "mean": self.mean}
        >>> stats = MyStats(count=10, mean=3.14)
        >>> print(stats)
        MyStats(count=10.0000, mean=3.1400)
    """

    _REPR_PRECISION: ClassVar[int] = 6
    """Precision for __repr__ (higher precision for debugging)"""

    _STR_PRECISION: ClassVar[int] = 4
    """Precision for __str__ (moderate precision for display)"""

    @abstractmethod
    def get_stats_dict(self) -> Dict[str, float]:
        """Return statistics as dictionary for formatting.

        Must be implemented by subclasses to provide the statistics
        that will be formatted for string representation.

        Returns:
            Dictionary mapping stat names to float values (may contain NaN)

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    def __repr__(self) -> str:
        """Return repr with high precision (6 decimals by default).

        Used for debugging and interactive shell display.
        """
        formatter = StatisticsFormatter(self._REPR_PRECISION)
        stats = self.get_stats_dict()
        formatted = formatter.format_stats_dict(stats)
        return f"{self.__class__.__name__}({formatted})"

    def __str__(self) -> str:
        """Return str with moderate precision (4 decimals by default).

        Used for regular string conversion and printing.
        """
        formatter = StatisticsFormatter(self._STR_PRECISION)
        stats = self.get_stats_dict()
        formatted = formatter.format_stats_dict(stats)
        return f"{self.__class__.__name__}({formatted})"

    def to_table_string(self, precision: int | None = None) -> str:
        """Format statistics as table for display.

        Args:
            precision: Decimal places (uses _STR_PRECISION if None)

        Returns:
            Multi-line formatted string with statistics
        """
        p = precision or self._STR_PRECISION
        formatter = StatisticsFormatter(p)
        stats = self.get_stats_dict()
        return formatter.format_table(stats)
