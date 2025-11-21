"""Unit conversion helpers with OOP design using converter strategies.

This module provides a clean object-oriented interface for unit conversions
through converter classes that handle specific unit types (velocity, density, etc.).
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

def _nanmax_abs(a: NDArray[np.floating[Any]]) -> float:
    """Helper to safely compute max absolute value."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return float(np.nanmax(np.abs(a)))
    except (ValueError, TypeError, FloatingPointError):
        return float("inf")

class Converter(ABC):
    """Abstract base class for unit converters.

    Defines the interface that all converters must implement.
    """

    @abstractmethod
    def convert(
        self,
        value: NDArray[np.floating[Any]] | float,
        from_unit: str,
        to_unit: str,
    ) -> NDArray[np.floating[Any]] | tuple[float, bool]:
        """Convert array from one unit to another.

        Args:
            value: Input numpy array or scalar value
            from_unit: Source unit string
            to_unit: Target unit string

        Returns:
            Converted array

        Raises:
            ValueError: If conversion is not supported
        """
        raise NotImplementedError()

    @abstractmethod
    def can_convert(self, from_unit: str, to_unit: str) -> bool:
        """Check if this converter can handle the conversion."""

class VelocityConverter(Converter):
    """Converts between velocity units (m/s <-> km/s)."""

    def __init__(self) -> None:
        self.canonical_units = ("m/s", "km/s")
        self.conversion_factor = 1000.0

    def convert(
        self, value: NDArray[np.floating[Any]] | float, from_unit: str, to_unit: str
    ) -> NDArray[np.floating[Any]]:
        """Convert velocity between m/s and km/s."""
        val = np.asarray(value)

        if from_unit == to_unit:
            return val

        if not self.can_convert(from_unit, to_unit):
            raise ValueError(f"Cannot convert velocity from {from_unit} to {to_unit}")

        if from_unit == "km/s" and to_unit == "m/s":
            return val * self.conversion_factor
        if from_unit == "m/s" and to_unit == "km/s":
            return val / self.conversion_factor

        raise ValueError(f"Unsupported velocity conversion: {from_unit} -> {to_unit}")

    def can_convert(self, from_unit: str, to_unit: str) -> bool:
        """Check if both units are supported velocity units."""
        return from_unit in self.canonical_units and to_unit in self.canonical_units

class DensityConverter(Converter):
    """Converts between density units (kg/m3 <-> g/cc)."""

    def __init__(self) -> None:
        self.canonical_units = ("kg/m3", "g/cc")
        self.conversion_factor = 1000.0

    def convert(
        self, value: NDArray[np.floating[Any]] | float, from_unit: str, to_unit: str
    ) -> NDArray[np.floating[Any]]:
        """Convert density between kg/m3 and g/cc."""
        val = np.asarray(value)

        if from_unit == to_unit:
            return val

        if not self.can_convert(from_unit, to_unit):
            raise ValueError(f"Cannot convert density from {from_unit} to {to_unit}")

        if from_unit == "g/cc" and to_unit == "kg/m3":
            return val * self.conversion_factor
        if from_unit == "kg/m3" and to_unit == "g/cc":
            return val / self.conversion_factor

        raise ValueError(f"Unsupported density conversion: {from_unit} -> {to_unit}")

    def can_convert(self, from_unit: str, to_unit: str) -> bool:
        """Check if both units are supported density units."""
        return from_unit in self.canonical_units and to_unit in self.canonical_units

class TimeConverter(Converter):
    """Converts time values with heuristic detection."""

    def __init__(
        self,
        convert_threshold_low: float = 0.01,
        convert_threshold_high: float = 100.0,
    ) -> None:
        self.threshold_low = convert_threshold_low
        self.threshold_high = convert_threshold_high

    def convert(
        self,
        value: NDArray[np.floating[Any]] | float,
        from_unit: str = "unknown",
        to_unit: str = "s",
    ) -> NDArray[np.floating[Any]] | tuple[float, bool]:
        """Convert time to seconds with heuristic detection.

        Returns (converted_value, was_converted) for scalar or array for ndarray
        """
        if isinstance(value, np.ndarray):
            raise TypeError("TimeConverter expects scalar values")

        try:
            v = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("Value must be numeric") from exc

        if self.threshold_low <= v < self.threshold_high:
            # Likely milliseconds
            return v / 1000.0, True
        return v, False

    def can_convert(self, from_unit: str, to_unit: str) -> bool:
        """Time converter works with any time-like units."""
        return to_unit in ("s", "seconds")

class LengthConverter(Converter):
    """Converts length values with heuristic detection."""

    def __init__(self, convert_threshold: float = 0.1) -> None:
        self.threshold = convert_threshold

    def convert(
        self,
        value: NDArray[np.floating[Any]] | float,
        from_unit: str = "unknown",
        to_unit: str = "m",
    ) -> NDArray[np.floating[Any]] | tuple[float, bool]:
        """Convert length to meters with heuristic detection.

        Returns (converted_value, was_converted) for scalar or array for ndarray
        """
        if isinstance(value, np.ndarray):
            raise TypeError("LengthConverter expects scalar values")

        try:
            v = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("Value must be numeric") from exc

        if v < self.threshold:
            # Likely kilometers
            return v * 1000.0, True
        return v, False

    def can_convert(self, from_unit: str, to_unit: str) -> bool:
        """Length converter works with any length-like units."""
        return to_unit in ("m", "meters")

class UnitRegistry:
    """Central registry managing all unit conversions.

    Uses converter strategy pattern for clean, extensible design.
    """

    def __init__(self) -> None:
        self.converters: list[Converter] = [
            VelocityConverter(),
            DensityConverter(),
            TimeConverter(),
            LengthConverter(),
        ]

    def convert(
        self, array: NDArray[np.floating[Any]], from_unit: str, to_unit: str
    ) -> NDArray[np.floating[Any]]:
        """Convert array from one unit to another using registered converters."""
        if from_unit == to_unit:
            return array

        for converter in self.converters:
            if converter.can_convert(from_unit, to_unit):
                return cast(
                    NDArray[np.floating[Any]],
                    converter.convert(array, from_unit, to_unit),
                )

        raise ValueError(f"No converter found for {from_unit} -> {to_unit}")

    def is_likely_in_unit(self, arr: ArrayLike, unit: str) -> bool:
        """Heuristic check whether array is likely in the requested unit."""

        a = np.asarray(arr)
        maxabs = _nanmax_abs(a)

        if unit in ("km/s",):
            return maxabs < 100
        if unit in ("m/s",):
            return maxabs >= 100
        if unit in ("g/cc",):
            return maxabs < 100
        if unit in ("kg/m3",):
            return maxabs >= 100
        return False

    @staticmethod
    def ensure_m_per_s(
        arr: ArrayLike, copy_on_convert: bool = False
    ) -> tuple[NDArray[np.floating[Any]], bool]:
        """Ensure velocity array is in m/s, converting from km/s if needed.

        Uses heuristic: values < 100 are likely km/s (seismic P-wave in km/s),
        values >= 100 are likely m/s.

        Args:
            arr: Input array or array-like
            copy_on_convert: If True, always return a copy; if False, return
                original if no conversion

        Returns:
            Tuple of (converted_array, was_converted)
        """
        a = np.asarray(arr)
        maxabs = _nanmax_abs(a)

        # If likely in km/s, convert to m/s
        if maxabs < 100:
            converted = a * 1000.0
            return converted, True

        # Already in m/s (or close enough)
        if copy_on_convert:
            return a.copy(), False
        return a, False

    @staticmethod
    def ensure_meters(
        arr: ArrayLike, copy_on_convert: bool = False
    ) -> tuple[NDArray[np.floating[Any]], bool]:
        """Ensure length array is in meters, converting from km if needed.

        Uses heuristic: values < 0.1 are likely km, values >= 0.1 are likely meters.

        Args:
            arr: Input array or array-like
            copy_on_convert: If True, always return a copy; if False, return
                original if no conversion

        Returns:
            Tuple of (converted_array, was_converted)
        """
        a = np.asarray(arr)
        try:
            maxabs = _nanmax_abs(a)
        except (ValueError, TypeError, FloatingPointError):
            maxabs = float("inf")

        # If likely in km, convert to meters
        if maxabs < 0.1:
            converted = a * 1000.0
            return converted, True

        # Already in meters (or close enough)
        if copy_on_convert:
            return a.copy(), False
        return a, False

__all__ = [
    "Converter",
    "VelocityConverter",
    "DensityConverter",
    "TimeConverter",
    "LengthConverter",
    "UnitRegistry",
]

# Module-level registry singleton (eagerly initialized)
unit_registry: UnitRegistry = UnitRegistry()

def get_unit_registry(instance: UnitRegistry | None = None) -> UnitRegistry:
    """Return provided UnitRegistry or the module-level singleton."""
    return instance if instance is not None else unit_registry

__all__.extend(["unit_registry", "get_unit_registry"])
