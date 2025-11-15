"""Parameterized conversion strategies to eliminate duplication.

This module provides a data-driven approach to unit conversions, replacing
multiple similar strategy classes with a single parameterized implementation.
"""

from __future__ import annotations

from typing import Dict, Callable, Any, Union, Optional, cast

import numpy as np
from numpy.typing import NDArray

from src.utils.quantity import Quantity

__all__ = ["ParameterizedConversionStrategy", "ConversionRegistry"]


class ParameterizedConversionStrategy:
    """Generic conversion strategy driven by conversion rules.

    Eliminates duplicate code across VelocityConversionStrategy,
    TimeConversionStrategy, DepthConversionStrategy by using data-driven
    configuration instead of separate classes.

    Example:
        >>> # Create velocity converter
        >>> velocity_converter = ParameterizedConversionStrategy(
        ...     base_unit="m/s",
        ...     factors={"m/s": 1.0, "km/s": 1000.0, "ft/s": 0.3048}
        ... )
        >>> result = velocity_converter.convert(value, "km/s", "m/s")
    """

    def __init__(
        self,
        base_unit: str,
        factors: Dict[str, float],
        custom_converters: Optional[Dict[tuple[str, str], Callable[[Any], Any]]] = None,
    ) -> None:
        """Initialize parameterized converter.

        Args:
            base_unit: The base unit for conversions (e.g., "m/s", "s", "m")
            factors: Conversion factors to base unit {unit: factor}
            custom_converters: Optional custom conversion functions
                               {(from_unit, to_unit): converter_func}
        """
        self.base_unit = base_unit
        self._factors = factors
        self._custom: Dict[tuple[str, str], Callable[[Any], Any]] = (
            custom_converters or {}
        )

    def convert(
        self,
        value: Any,
        from_unit: str,
        to_unit: str,
    ) -> Union[NDArray[Any], Quantity]:
        """Convert value between units.

        Args:
            value: Input array or Quantity
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value
        """
        if not self.validate_units(from_unit):
            raise ValueError(f"Invalid from_unit: {from_unit}")
        if not self.validate_units(to_unit):
            raise ValueError(f"Invalid to_unit: {to_unit}")

        is_quantity = isinstance(value, Quantity)
        if is_quantity:
            arr = cast(Any, value.to_numpy())
        else:
            arr = value

        # Check for custom converter
        key = (from_unit, to_unit)
        if key in self._custom:
            converted = self._custom[key](arr)
        else:
            # Standard factor-based conversion
            factor_from = self._factors[from_unit]
            factor_to = self._factors[to_unit]
            converted = arr * (factor_from / factor_to)

        return self._preserve_quantity(converted, is_quantity, to_unit)

    def validate_units(self, unit: str) -> bool:
        """Check if unit is supported."""
        return unit in self._factors

    def _preserve_quantity(
        self,
        converted: NDArray[Any],
        was_quantity: bool,
        target_unit: str,
    ) -> Union[NDArray[Any], Quantity]:
        """Preserve Quantity wrapper if input was Quantity."""
        return Quantity(converted, target_unit) if was_quantity else converted


class ConversionRegistry:
    """Central registry for all conversion strategies.

    Provides pre-configured converters for common physical quantities,
    eliminating the need for separate strategy classes.
    """

    # Velocity conversions
    VELOCITY = ParameterizedConversionStrategy(
        base_unit="m/s",
        factors={
            "m/s": 1.0,
            "km/s": 1000.0,
            "ft/s": 0.3048,
        },
    )

    # Time conversions
    TIME = ParameterizedConversionStrategy(
        base_unit="s",
        factors={
            "s": 1.0,
            "ms": 1e-3,
            "us": 1e-6,
            "µs": 1e-6,
        },
    )

    # Depth/Length conversions
    DEPTH = ParameterizedConversionStrategy(
        base_unit="m",
        factors={
            "m": 1.0,
            "km": 1000.0,
            "ft": 0.3048,
            "cm": 0.01,
        },
    )

    # Amplitude conversions (with custom converters for log scales)
    AMPLITUDE = ParameterizedConversionStrategy(
        base_unit="linear",
        factors={"linear": 1.0, "dB": 1.0, "log10": 1.0, "log2": 1.0},
        custom_converters={
            # dB conversions
            ("linear", "dB"): lambda x: 20.0 * np.log10(np.abs(x) + 1e-12),
            ("dB", "linear"): lambda x: 10.0 ** (x / 20.0),
            # log10 conversions
            ("linear", "log10"): lambda x: np.log10(np.abs(x) + 1e-12),
            ("log10", "linear"): lambda x: 10.0**x,
            # log2 conversions
            ("linear", "log2"): lambda x: np.log2(np.abs(x) + 1e-12),
            ("log2", "linear"): lambda x: 2.0**x,
        },
    )

    @classmethod
    def get(cls, quantity_type: str) -> ParameterizedConversionStrategy:
        """Get converter for a specific quantity type.

        Args:
            quantity_type: Type of quantity ("velocity", "time", "depth", "amplitude")

        Returns:
            Configured converter

        Raises:
            ValueError: If quantity_type is unknown
        """
        converters = {
            "velocity": cls.VELOCITY,
            "time": cls.TIME,
            "depth": cls.DEPTH,
            "length": cls.DEPTH,  # Alias
            "amplitude": cls.AMPLITUDE,
        }

        if quantity_type.lower() not in converters:
            available = ", ".join(converters.keys())
            raise ValueError(
                f"Unknown quantity type: {quantity_type}. " f"Available: {available}"
            )

        return converters[quantity_type.lower()]
