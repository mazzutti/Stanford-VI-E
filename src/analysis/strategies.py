"""Strategy pattern implementations for flexible component behavior.

This module provides abstract strategy classes and composable implementations
that allow flexible swapping of algorithms and behaviors throughout the
analysis module.

For validation strategies, see src.analysis.validators instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypeVar, Union, cast

import numpy as np
from numpy.typing import NDArray

from src.utils.quantity import Quantity

T = TypeVar("T", bound=Union[int, float, np.number[Any]])

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
        value: Union[NDArray[Any], Quantity],
        from_unit: str,
        to_unit: str,
    ) -> Union[NDArray[Any], Quantity]:
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
    ) -> Union[NDArray[Any], Quantity]:
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
    """Velocity conversion strategy supporting m/s, km/s, ft/s conversions."""

    # Conversion factors to m/s (base unit)
    _FACTORS = {
        "m/s": 1.0,
        "km/s": 1000.0,
        "ft/s": 0.3048,
    }

    def convert(
        self,
        value: Union[NDArray[Any], Quantity],
        from_unit: str,
        to_unit: str,
    ) -> Union[NDArray[Any], Quantity]:
        """Convert velocity between units.

        Args:
            value: Input velocity array or Quantity
            from_unit: Source velocity unit
            to_unit: Target velocity unit

        Returns:
            Converted velocity
        """
        self.validate_units(from_unit)
        self.validate_units(to_unit)

        is_quantity = isinstance(value, Quantity)
        arr = value.array if is_quantity else value

        # Convert to base unit (m/s) then to target
        factor_from = self._FACTORS[from_unit]
        factor_to = self._FACTORS[to_unit]
        converted = arr * (factor_from / factor_to)

        return self._preserve_quantity_wrapper(converted, is_quantity, to_unit)

    def validate_units(self, unit: str) -> bool:
        """Validate velocity unit."""
        return unit in self._FACTORS


class TimeConversionStrategy(ConversionStrategy):
    """Time conversion strategy supporting s, ms, µs conversions."""

    # Conversion factors to seconds (base unit)
    _FACTORS = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "µs": 1e-6,  # Alternative µ symbol
    }

    def convert(
        self,
        value: Union[NDArray[Any], Quantity],
        from_unit: str,
        to_unit: str,
    ) -> Union[NDArray[Any], Quantity]:
        """Convert time between units.

        Args:
            value: Input time array or Quantity
            from_unit: Source time unit
            to_unit: Target time unit

        Returns:
            Converted time
        """
        self.validate_units(from_unit)
        self.validate_units(to_unit)

        is_quantity = isinstance(value, Quantity)
        arr = value.array if is_quantity else value

        # Convert to base unit (s) then to target
        factor_from = self._FACTORS[from_unit]
        factor_to = self._FACTORS[to_unit]
        converted = arr * (factor_from / factor_to)

        return self._preserve_quantity_wrapper(converted, is_quantity, to_unit)

    def validate_units(self, unit: str) -> bool:
        """Validate time unit."""
        return unit in self._FACTORS


class DepthConversionStrategy(ConversionStrategy):
    """Depth conversion strategy supporting m, km, ft conversions."""

    # Conversion factors to meters (base unit)
    _FACTORS = {
        "m": 1.0,
        "km": 1000.0,
        "ft": 0.3048,
    }

    def convert(
        self,
        value: Union[NDArray[Any], Quantity],
        from_unit: str,
        to_unit: str,
    ) -> Union[NDArray[Any], Quantity]:
        """Convert depth between units.

        Args:
            value: Input depth array or Quantity
            from_unit: Source depth unit
            to_unit: Target depth unit

        Returns:
            Converted depth
        """
        self.validate_units(from_unit)
        self.validate_units(to_unit)

        is_quantity = isinstance(value, Quantity)
        arr = value.array if is_quantity else value

        # Convert to base unit (m) then to target
        factor_from = self._FACTORS[from_unit]
        factor_to = self._FACTORS[to_unit]
        converted = arr * (factor_from / factor_to)

        return self._preserve_quantity_wrapper(converted, is_quantity, to_unit)

    def validate_units(self, unit: str) -> bool:
        """Validate depth unit."""
        return unit in self._FACTORS


class AmplitudeConversionStrategy(ConversionStrategy):
    """Amplitude/scaling conversion strategy supporting linear and dB scales."""

    _VALID_UNITS = {"linear", "dB", "log10", "log2"}

    def convert(
        self,
        value: Union[NDArray[Any], Quantity],
        from_unit: str,
        to_unit: str,
    ) -> Union[NDArray[Any], Quantity]:
        """Convert amplitude between scales.

        Args:
            value: Input amplitude array or Quantity
            from_unit: Source scale ('linear', 'dB', 'log10', 'log2')
            to_unit: Target scale

        Returns:
            Converted amplitude
        """
        self.validate_units(from_unit)
        self.validate_units(to_unit)

        is_quantity = isinstance(value, Quantity)
        arr = value.array if is_quantity else value

        # Convert through linear scale as intermediate
        if from_unit != "linear":
            arr = self._from_scale(arr, from_unit)

        if to_unit != "linear":
            arr = self._to_scale(arr, to_unit)

        return self._preserve_quantity_wrapper(arr, is_quantity, to_unit)

    @staticmethod
    def _from_scale(arr: NDArray[Any], scale: str) -> NDArray[Any]:
        """Convert from non-linear scale to linear."""
        if scale == "dB":
            return 10.0 ** (arr / 20.0)
        elif scale == "log10":
            return 10.0**arr
        elif scale == "log2":
            return 2.0**arr
        return arr

    @staticmethod
    def _to_scale(arr: NDArray[Any], scale: str) -> NDArray[Any]:
        """Convert from linear scale to non-linear."""
        if scale == "dB":
            return 20.0 * np.log10(np.abs(arr) + 1e-12)
        elif scale == "log10":
            return np.log10(np.abs(arr) + 1e-12)
        elif scale == "log2":
            return np.log2(np.abs(arr) + 1e-12)
        return arr

    def validate_units(self, unit: str) -> bool:
        """Validate amplitude scale."""
        return unit in self._VALID_UNITS
