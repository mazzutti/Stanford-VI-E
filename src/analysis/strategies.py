"""Strategy pattern implementations for flexible component behavior.

This module provides abstract strategy classes and composable implementations
that allow flexible swapping of algorithms and behaviors throughout the
analysis module.

For validation strategies, see src.analysis.validators instead.

Note: Conversion strategies now use parameterized implementations from
src.core.parameterized_conversions. The classes here are kept for backward
compatibility but delegate to the shared implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from src.utils.quantity import Quantity
from src.core.parameterized_conversions import ConversionRegistry

T = TypeVar("T", bound=int | float | np.number[Any])

__all__ = [
    # Array Statistics
    "ArrayStatisticsStrategy",
    "StandardArrayStatistics",
    "RobustArrayStatistics",
    # Unit Conversions
    "ConversionStrategy",
    "VelocityConversionStrategy",
    "TimeConversionStrategy",
    "DepthConversionStrategy",
    "AmplitudeConversionStrategy",
    "UnitSystem",
]


# ============================================================================
# Array Statistics Strategies
# ============================================================================


class ArrayStatisticsStrategy(ABC):
    """Abstract base for array statistics computation strategies.

    Enables pluggable computation of array statistics with different
    algorithms (standard vs. robust, for example).
    """

    @abstractmethod
    def compute_mean(self, arr: NDArray[Any]) -> int | float | np.number[Any]:
        """Compute mean of array."""
        pass

    @abstractmethod
    def compute_std(self, arr: NDArray[Any]) -> int | float | np.number[Any]:
        """Compute standard deviation of array."""
        pass

    @abstractmethod
    def compute_median(self, arr: NDArray[Any]) -> int | float | np.number[Any]:
        """Compute median of array."""
        pass


class StandardArrayStatistics(ArrayStatisticsStrategy):
    """Standard array statistics using numpy defaults."""

    def compute_mean(self, arr: NDArray[Any]) -> int | float | np.number[Any]:
        """Compute mean using numpy."""
        return cast(int | float | np.number[Any], np.mean(arr))

    def compute_std(self, arr: NDArray[Any]) -> int | float | np.number[Any]:
        """Compute standard deviation using numpy."""
        return cast(int | float | np.number[Any], np.std(arr))

    def compute_median(self, arr: NDArray[Any]) -> int | float | np.number[Any]:
        """Compute median using numpy."""
        return cast(int | float | np.number[Any], np.median(arr))


class RobustArrayStatistics(ArrayStatisticsStrategy):
    """Robust array statistics using median and IQR."""

    def compute_mean(self, arr: NDArray[Any]) -> int | float | np.number[Any]:
        """Compute robust mean (median)."""
        return cast(int | float | np.number[Any], np.median(arr))

    def compute_std(self, arr: NDArray[Any]) -> int | float | np.number[Any]:
        """Compute robust std (IQR)."""
        q75, q25 = np.percentile(arr, [75, 25])
        return cast(int | float | np.number[Any], (q75 - q25) / 1.35)

    def compute_median(self, arr: NDArray[Any]) -> int | float | np.number[Any]:
        """Compute median."""
        return cast(int | float | np.number[Any], np.median(arr))


# ============================================================================
# Unit Conversion Strategies
# ============================================================================


class UnitSystem(Enum):
    """Enumeration of supported unit systems."""

    SI = "si"  # m, s, m/s
    METRIC = "metric"  # km, s, km/s
    IMPERIAL = "imperial"  # ft, s, ft/s
    CGS = "cgs"  # cm, s, cm/s


class ConversionStrategy(ABC):
    """Abstract base class for unit conversion strategies.

    Defines the interface for converting between different unit systems
    for seismic physical quantities.
    """

    @abstractmethod
    def convert(
        self,
        value: NDArray[Any] | Quantity,
        from_unit: str,
        to_unit: str,
    ) -> NDArray[Any] | Quantity:
        """Convert a value from one unit to another.

        Args:
            value: Input array or Quantity to convert
            from_unit: Source unit identifier (e.g., 'm/s', 'km/s')
            to_unit: Target unit identifier (e.g., 'm/s', 'km/s')

        Returns:
            Converted value as NDArray or Quantity
        """

    @abstractmethod
    def validate_units(self, unit: str) -> bool:
        """Validate if a unit string is supported by this strategy.

        Args:
            unit: Unit identifier to validate

        Returns:
            True if unit is supported, False otherwise
        """

    def _preserve_quantity_wrapper(
        self,
        converted: NDArray[Any],
        was_quantity: bool,
        target_unit: str,
    ) -> NDArray[Any] | Quantity:
        """Preserve Quantity wrapper if input was Quantity.

        Args:
            converted: Converted array
            was_quantity: Whether original value was Quantity
            target_unit: Target unit for Quantity wrapper

        Returns:
            Converted value as NDArray or Quantity
        """
        return Quantity(converted, target_unit) if was_quantity else converted


class VelocityConversionStrategy(ConversionStrategy):
    """Velocity conversion strategy supporting m/s, km/s, ft/s conversions.

    Now delegates to parameterized implementation for DRY principle.
    """

    def __init__(self) -> None:
        """Initialize with shared converter."""
        self._converter = ConversionRegistry.VELOCITY

    def convert(
        self,
        value: NDArray[Any] | Quantity,
        from_unit: str,
        to_unit: str,
    ) -> NDArray[Any] | Quantity:
        """Convert velocity between units."""
        return self._converter.convert(value, from_unit, to_unit)

    def validate_units(self, unit: str) -> bool:
        """Validate velocity unit."""
        return self._converter.validate_units(unit)


class TimeConversionStrategy(ConversionStrategy):
    """Time conversion strategy supporting s, ms, µs conversions.

    Now delegates to parameterized implementation for DRY principle.
    """

    def __init__(self) -> None:
        """Initialize with shared converter."""
        self._converter = ConversionRegistry.TIME

    def convert(
        self,
        value: NDArray[Any] | Quantity,
        from_unit: str,
        to_unit: str,
    ) -> NDArray[Any] | Quantity:
        """Convert time between units."""
        return self._converter.convert(value, from_unit, to_unit)

    def validate_units(self, unit: str) -> bool:
        """Validate time unit."""
        return self._converter.validate_units(unit)


class DepthConversionStrategy(ConversionStrategy):
    """Depth conversion strategy supporting m, km, ft conversions.

    Now delegates to parameterized implementation for DRY principle.
    """

    def __init__(self) -> None:
        """Initialize with shared converter."""
        self._converter = ConversionRegistry.DEPTH

    def convert(
        self,
        value: NDArray[Any] | Quantity,
        from_unit: str,
        to_unit: str,
    ) -> NDArray[Any] | Quantity:
        """Convert depth between units."""
        return self._converter.convert(value, from_unit, to_unit)

    def validate_units(self, unit: str) -> bool:
        """Validate depth unit."""
        return self._converter.validate_units(unit)


class AmplitudeConversionStrategy(ConversionStrategy):
    """Amplitude/scaling conversion strategy supporting linear and dB scales.

    Now delegates to parameterized implementation for DRY principle.
    """

    def __init__(self) -> None:
        """Initialize with shared converter."""
        self._converter = ConversionRegistry.AMPLITUDE

    def convert(
        self,
        value: NDArray[Any] | Quantity,
        from_unit: str,
        to_unit: str,
    ) -> NDArray[Any] | Quantity:
        """Convert amplitude between scales."""
        return self._converter.convert(value, from_unit, to_unit)

    def validate_units(self, unit: str) -> bool:
        """Validate amplitude scale."""
        return self._converter.validate_units(unit)
