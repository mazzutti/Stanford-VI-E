"""Conversion Strategy Pattern for unit conversions.

This module provides strategy implementations for converting between different
units. Each conversion strategy encapsulates the conversion logic for a specific
unit type, making it easy to extend with new conversions and avoiding scattered
conversion logic throughout the codebase.

Patterns Used:
  - Strategy: Different conversion strategies for different unit types
  - Factory: ConversionStrategyFactory for creating converters

Example:
    >>> velocity_converter = VelocityConversionStrategy("km/s", "m/s")
    >>> velocity_m_per_s = velocity_converter.convert(3.0)
    >>> time_converter = TimeConversionStrategy("seconds", "milliseconds")
    >>> time_ms = time_converter.convert(0.5)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ConversionStrategy",
    "VelocityConversionStrategy",
    "TimeConversionStrategy",
    "DepthConversionStrategy",
    "AmplitudeConversionStrategy",
    "ConversionStrategyFactory",
]


class UnitType(str, Enum):
    """Enumeration of supported unit types."""

    VELOCITY = "velocity"
    TIME = "time"
    DEPTH = "depth"
    AMPLITUDE = "amplitude"


class ConversionStrategy(ABC):
    """Abstract base class for unit conversion strategies.

    Each strategy encapsulates conversion logic for a specific unit type,
    providing consistent interface for all conversions.
    """

    @abstractmethod
    def convert(self, value: float) -> float:
        """Convert value from source unit to target unit.

        Parameters
        ----------
        value : float
            Value in source unit.

        Returns
        -------
        float
            Converted value in target unit.

        Raises
        ------
        ValueError
            If value is invalid for the conversion.
        """
        pass

    @abstractmethod
    def reverse_convert(self, value: float) -> float:
        """Convert value from target unit back to source unit.

        Inverse of convert().

        Parameters
        ----------
        value : float
            Value in target unit.

        Returns
        -------
        float
            Converted value in source unit.

        Raises
        ------
        ValueError
            If value is invalid for the conversion.
        """
        pass

    @property
    @abstractmethod
    def from_unit(self) -> str:
        """Source unit name."""
        pass

    @property
    @abstractmethod
    def to_unit(self) -> str:
        """Target unit name."""
        pass

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}({self.from_unit} → {self.to_unit})"


class VelocityConversionStrategy(ConversionStrategy):
    """Strategy for velocity unit conversions.

    Supports conversions between:
      - m/s (SI unit)
      - km/s
      - ft/s
      - mile/s

    Example:
        >>> conv = VelocityConversionStrategy("km/s", "m/s")
        >>> print(conv.convert(5.0))  # 5000.0
    """

    # Conversion factors to m/s (SI unit)
    _FACTORS: Dict[str, float] = {
        "m/s": 1.0,
        "km/s": 1000.0,
        "ft/s": 0.3048,
        "mile/s": 1609.344,
    }

    def __init__(self, from_unit: str, to_unit: str) -> None:
        """Initialize velocity converter.

        Parameters
        ----------
        from_unit : str
            Source unit (e.g., 'km/s').
        to_unit : str
            Target unit (e.g., 'm/s').

        Raises
        ------
        ValueError
            If unit is not supported.
        """
        if from_unit not in self._FACTORS:
            raise ValueError(
                f"Unsupported velocity unit: {from_unit}. "
                f"Supported: {list(self._FACTORS.keys())}"
            )
        if to_unit not in self._FACTORS:
            raise ValueError(
                f"Unsupported velocity unit: {to_unit}. "
                f"Supported: {list(self._FACTORS.keys())}"
            )
        self._from_unit = from_unit
        self._to_unit = to_unit
        logger.debug(f"Created VelocityConversionStrategy: {from_unit} → {to_unit}")

    def convert(self, value: float) -> float:
        """Convert velocity from source to target unit."""
        if value < 0:
            raise ValueError(f"Velocity cannot be negative: {value}")
        # Convert to SI (m/s), then to target unit
        si_value = value * self._FACTORS[self._from_unit]
        return si_value / self._FACTORS[self._to_unit]

    def reverse_convert(self, value: float) -> float:
        """Convert velocity from target back to source unit."""
        if value < 0:
            raise ValueError(f"Velocity cannot be negative: {value}")
        # Convert from target to SI, then to source unit
        si_value = value * self._FACTORS[self._to_unit]
        return si_value / self._FACTORS[self._from_unit]

    @property
    def from_unit(self) -> str:
        """Source unit name."""
        return self._from_unit

    @property
    def to_unit(self) -> str:
        """Target unit name."""
        return self._to_unit


class TimeConversionStrategy(ConversionStrategy):
    """Strategy for time unit conversions.

    Supports conversions between:
      - s (seconds, SI unit)
      - ms (milliseconds)
      - us (microseconds)
      - ns (nanoseconds)

    Example:
        >>> conv = TimeConversionStrategy("ms", "s")
        >>> print(conv.convert(1000.0))  # 1.0
    """

    # Conversion factors to seconds (SI unit)
    _FACTORS: Dict[str, float] = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
    }

    def __init__(self, from_unit: str, to_unit: str) -> None:
        """Initialize time converter.

        Parameters
        ----------
        from_unit : str
            Source unit (e.g., 'ms').
        to_unit : str
            Target unit (e.g., 's').

        Raises
        ------
        ValueError
            If unit is not supported.
        """
        if from_unit not in self._FACTORS:
            raise ValueError(
                f"Unsupported time unit: {from_unit}. "
                f"Supported: {list(self._FACTORS.keys())}"
            )
        if to_unit not in self._FACTORS:
            raise ValueError(
                f"Unsupported time unit: {to_unit}. "
                f"Supported: {list(self._FACTORS.keys())}"
            )
        self._from_unit = from_unit
        self._to_unit = to_unit
        logger.debug(f"Created TimeConversionStrategy: {from_unit} → {to_unit}")

    def convert(self, value: float) -> float:
        """Convert time from source to target unit."""
        if value < 0:
            raise ValueError(f"Time cannot be negative: {value}")
        # Convert to SI (seconds), then to target unit
        si_value = value * self._FACTORS[self._from_unit]
        return si_value / self._FACTORS[self._to_unit]

    def reverse_convert(self, value: float) -> float:
        """Convert time from target back to source unit."""
        if value < 0:
            raise ValueError(f"Time cannot be negative: {value}")
        # Convert from target to SI, then to source unit
        si_value = value * self._FACTORS[self._to_unit]
        return si_value / self._FACTORS[self._from_unit]

    @property
    def from_unit(self) -> str:
        """Source unit name."""
        return self._from_unit

    @property
    def to_unit(self) -> str:
        """Target unit name."""
        return self._to_unit


class DepthConversionStrategy(ConversionStrategy):
    """Strategy for depth/distance unit conversions.

    Supports conversions between:
      - m (meters, SI unit)
      - km (kilometers)
      - ft (feet)
      - mile (miles)

    Example:
        >>> conv = DepthConversionStrategy("km", "m")
        >>> print(conv.convert(1.0))  # 1000.0
    """

    # Conversion factors to meters (SI unit)
    _FACTORS: Dict[str, float] = {
        "m": 1.0,
        "km": 1000.0,
        "ft": 0.3048,
        "mile": 1609.344,
    }

    def __init__(self, from_unit: str, to_unit: str) -> None:
        """Initialize depth converter.

        Parameters
        ----------
        from_unit : str
            Source unit (e.g., 'km').
        to_unit : str
            Target unit (e.g., 'm').

        Raises
        ------
        ValueError
            If unit is not supported.
        """
        if from_unit not in self._FACTORS:
            raise ValueError(
                f"Unsupported depth unit: {from_unit}. "
                f"Supported: {list(self._FACTORS.keys())}"
            )
        if to_unit not in self._FACTORS:
            raise ValueError(
                f"Unsupported depth unit: {to_unit}. "
                f"Supported: {list(self._FACTORS.keys())}"
            )
        self._from_unit = from_unit
        self._to_unit = to_unit
        logger.debug(f"Created DepthConversionStrategy: {from_unit} → {to_unit}")

    def convert(self, value: float) -> float:
        """Convert depth from source to target unit."""
        if value < 0:
            raise ValueError(f"Depth cannot be negative: {value}")
        # Convert to SI (meters), then to target unit
        si_value = value * self._FACTORS[self._from_unit]
        return si_value / self._FACTORS[self._to_unit]

    def reverse_convert(self, value: float) -> float:
        """Convert depth from target back to source unit."""
        if value < 0:
            raise ValueError(f"Depth cannot be negative: {value}")
        # Convert from target to SI, then to source unit
        si_value = value * self._FACTORS[self._to_unit]
        return si_value / self._FACTORS[self._from_unit]

    @property
    def from_unit(self) -> str:
        """Source unit name."""
        return self._from_unit

    @property
    def to_unit(self) -> str:
        """Target unit name."""
        return self._to_unit


class AmplitudeConversionStrategy(ConversionStrategy):
    """Strategy for seismic amplitude unit conversions.

    Supports conversions between:
      - raw (dimensionless amplitude, default)
      - normalized (0-1 range)
      - percent (0-100 range)

    Example:
        >>> conv = AmplitudeConversionStrategy("normalized", "percent")
        >>> print(conv.convert(0.5))  # 50.0
    """

    def __init__(self, from_unit: str, to_unit: str) -> None:
        """Initialize amplitude converter.

        Parameters
        ----------
        from_unit : str
            Source unit ('raw', 'normalized', 'percent').
        to_unit : str
            Target unit ('raw', 'normalized', 'percent').

        Raises
        ------
        ValueError
            If unit is not supported.
        """
        supported = {"raw", "normalized", "percent"}
        if from_unit not in supported:
            raise ValueError(
                f"Unsupported amplitude unit: {from_unit}. " f"Supported: {supported}"
            )
        if to_unit not in supported:
            raise ValueError(
                f"Unsupported amplitude unit: {to_unit}. " f"Supported: {supported}"
            )
        self._from_unit = from_unit
        self._to_unit = to_unit
        logger.debug(f"Created AmplitudeConversionStrategy: {from_unit} → {to_unit}")

    def convert(self, value: float) -> float:
        """Convert amplitude from source to target unit."""
        # First convert to normalized (0-1 range)
        if self._from_unit == "normalized":
            normalized = value
        elif self._from_unit == "percent":
            normalized = value / 100.0
        else:  # raw
            # Assume raw is already normalized
            normalized = value

        # Then convert to target unit
        if self._to_unit == "normalized":
            return normalized
        elif self._to_unit == "percent":
            return normalized * 100.0
        else:  # raw
            return normalized

    def reverse_convert(self, value: float) -> float:
        """Convert amplitude from target back to source unit."""
        # First convert to normalized
        if self._to_unit == "normalized":
            normalized = value
        elif self._to_unit == "percent":
            normalized = value / 100.0
        else:  # raw
            normalized = value

        # Then convert to source unit
        if self._from_unit == "normalized":
            return normalized
        elif self._from_unit == "percent":
            return normalized * 100.0
        else:  # raw
            return normalized

    @property
    def from_unit(self) -> str:
        """Source unit name."""
        return self._from_unit

    @property
    def to_unit(self) -> str:
        """Target unit name."""
        return self._to_unit


class ConversionStrategyFactory:
    """Factory for creating conversion strategies.

    Provides static methods for creating converters for common unit types.

    Example:
        >>> converter = ConversionStrategyFactory.create("velocity", "km/s", "m/s")
        >>> result = converter.convert(3.0)
    """

    _CONVERTERS = {
        UnitType.VELOCITY: VelocityConversionStrategy,
        UnitType.TIME: TimeConversionStrategy,
        UnitType.DEPTH: DepthConversionStrategy,
        UnitType.AMPLITUDE: AmplitudeConversionStrategy,
    }

    @staticmethod
    def create(
        unit_type: str,
        from_unit: str,
        to_unit: str,
    ) -> ConversionStrategy:
        """Create a conversion strategy.

        Parameters
        ----------
        unit_type : str
            Type of unit to convert ('velocity', 'time', 'depth', 'amplitude').
        from_unit : str
            Source unit name.
        to_unit : str
            Target unit name.

        Returns
        -------
        ConversionStrategy
            Appropriate converter instance.

        Raises
        ------
        ValueError
            If unit_type is not supported.
        """
        try:
            unit_enum = UnitType(unit_type)
        except ValueError:
            raise ValueError(
                f"Unsupported unit type: {unit_type}. "
                f"Supported: {[t.value for t in UnitType]}"
            )

        converter_class = ConversionStrategyFactory._CONVERTERS.get(unit_enum)
        if converter_class is None:
            raise ValueError(f"No converter for unit type: {unit_type}")

        return converter_class(from_unit, to_unit)

    @staticmethod
    def create_velocity(from_unit: str, to_unit: str) -> VelocityConversionStrategy:
        """Create a velocity converter.

        Parameters
        ----------
        from_unit : str
            Source velocity unit.
        to_unit : str
            Target velocity unit.

        Returns
        -------
        VelocityConversionStrategy
            Velocity converter.
        """
        return VelocityConversionStrategy(from_unit, to_unit)

    @staticmethod
    def create_time(from_unit: str, to_unit: str) -> TimeConversionStrategy:
        """Create a time converter.

        Parameters
        ----------
        from_unit : str
            Source time unit.
        to_unit : str
            Target time unit.

        Returns
        -------
        TimeConversionStrategy
            Time converter.
        """
        return TimeConversionStrategy(from_unit, to_unit)

    @staticmethod
    def create_depth(from_unit: str, to_unit: str) -> DepthConversionStrategy:
        """Create a depth converter.

        Parameters
        ----------
        from_unit : str
            Source depth unit.
        to_unit : str
            Target depth unit.

        Returns
        -------
        DepthConversionStrategy
            Depth converter.
        """
        return DepthConversionStrategy(from_unit, to_unit)

    @staticmethod
    def create_amplitude(from_unit: str, to_unit: str) -> AmplitudeConversionStrategy:
        """Create an amplitude converter.

        Parameters
        ----------
        from_unit : str
            Source amplitude unit.
        to_unit : str
            Target amplitude unit.

        Returns
        -------
        AmplitudeConversionStrategy
            Amplitude converter.
        """
        return AmplitudeConversionStrategy(from_unit, to_unit)
