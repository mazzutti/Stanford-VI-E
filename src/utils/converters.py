"""Unit conversion utilities using Strategy pattern for extensible conversions.

Provides abstract base class and implementations for converting between
different unit systems (e.g., km/s ↔ m/s, g/cc ↔ kg/m³).
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class UnitConverter(ABC):
    """Abstract base class for unit converters using Strategy pattern.

    Provides a cleaner, more extensible approach to unit conversion logic
    compared to static methods with repeated patterns. Each converter
    specializes in a specific unit conversion pair.

    Subclasses should implement:
    - is_likely_in_unit: Check if array is in target unit
    - convert_if_needed: Convert if needed, returning (array, was_converted)
    """

    @abstractmethod
    def is_likely_in_unit(self, arr: NDArray[Any]) -> bool:
        """Check if array is likely already in target unit.

        Parameters
        ----------
        arr : NDArray
            Array to check

        Returns
        -------
        bool
            True if array appears to be in target unit
        """

    @abstractmethod
    def convert_if_needed(
        self, arr: NDArray[Any], copy_on_convert: bool = True
    ) -> tuple[NDArray[Any], bool]:
        """Convert array to target unit if needed.

        Parameters
        ----------
        arr : NDArray
            Array to convert
        copy_on_convert : bool, optional
            If True, create copy when converting. If False, convert in-place.

        Returns
        -------
        Tuple[NDArray, bool]
            Tuple of (converted_array, was_converted)
        """

    def _nanmax_abs(self, a: NDArray[Any]) -> float:
        """Helper to safely get max absolute value, handling NaNs.

        Parameters
        ----------
        a : NDArray
            Array to check

        Returns
        -------
        float
            Maximum absolute value, or inf if cannot compute
        """
        try:
            return float(np.nanmax(np.abs(a)))
        except (ValueError, TypeError, FloatingPointError):
            return float("inf")

    def _ensure_numeric(self, arr: NDArray[Any]) -> NDArray[Any]:
        """Ensure array is numeric type.

        Parameters
        ----------
        arr : NDArray
            Array to check

        Returns
        -------
        NDArray
            Array with numeric dtype
        """
        if not np.issubdtype(arr.dtype, np.number):
            arr = arr.astype(float)
        return arr


class VelocityConverter(UnitConverter):
    """Converts velocity values between km/s and m/s.

    Uses a configurable threshold to detect unit. Values below threshold
    are assumed to be in km/s (smaller magnitude), values above are assumed
    to be in m/s (larger magnitude).

    Examples:
        >>> converter = VelocityConverter(threshold=100)
        >>> vel_km_s = np.array([3.0, 4.0, 5.0])  # km/s
        >>> vel_m_s, was_converted = converter.convert_if_needed(vel_km_s)
        >>> was_converted
        True
        >>> vel_m_s[0]
        3000.0
    """

    def __init__(self, threshold: float = 100.0):
        """Initialize converter with unit detection threshold.

        Parameters
        ----------
        threshold : float, optional
            Values < threshold assumed km/s, >= threshold assumed m/s
        """
        self.threshold = threshold
        self.conversion_factor = 1000.0

    def is_likely_in_unit(self, arr: NDArray[Any]) -> bool:
        """Check if array is likely in m/s (not km/s).

        Parameters
        ----------
        arr : NDArray
            Velocity array to check

        Returns
        -------
        bool
            True if max absolute value >= threshold (assumed m/s)
        """
        maxabs = self._nanmax_abs(arr)
        return maxabs >= self.threshold

    def convert_if_needed(
        self, arr: NDArray[Any], copy_on_convert: bool = True
    ) -> tuple[NDArray[Any], bool]:
        """Convert from km/s to m/s if needed.

        Parameters
        ----------
        arr : NDArray
            Velocity array
        copy_on_convert : bool, optional
            If True, create copy when converting

        Returns
        -------
        Tuple[NDArray, bool]
            (converted_array, was_converted)
        """
        arr = self._ensure_numeric(arr)
        maxabs = self._nanmax_abs(arr)

        if maxabs == float("inf"):
            return arr, False

        if maxabs < self.threshold:  # Likely km/s, convert to m/s
            if copy_on_convert:
                return arr.astype(float) * self.conversion_factor, True
            arr[...] = arr * self.conversion_factor
            return arr, True

        return arr, False


class DensityConverter(UnitConverter):
    """Converts density values between g/cc and kg/m³.

    Uses a configurable threshold to detect unit. Values below threshold
    are assumed to be in g/cc (smaller magnitude), values above are assumed
    to be in kg/m³ (larger magnitude).

    Examples:
        >>> converter = DensityConverter(threshold=100)
        >>> rho_g_cc = np.array([2.3, 2.4, 2.5])  # g/cc
        >>> rho_kg_m3, was_converted = converter.convert_if_needed(rho_g_cc)
        >>> was_converted
        True
        >>> rho_kg_m3[0]
        2300.0
    """

    def __init__(self, threshold: float = 100.0):
        """Initialize converter with unit detection threshold.

        Parameters
        ----------
        threshold : float, optional
            Values < threshold assumed g/cc, >= threshold assumed kg/m³
        """
        self.threshold = threshold
        self.conversion_factor = 1000.0

    def is_likely_in_unit(self, arr: NDArray[Any]) -> bool:
        """Check if array is likely in kg/m³ (not g/cc).

        Parameters
        ----------
        arr : NDArray
            Density array to check

        Returns
        -------
        bool
            True if max absolute value >= threshold (assumed kg/m³)
        """
        maxabs = self._nanmax_abs(arr)
        return maxabs >= self.threshold

    def convert_if_needed(
        self, arr: NDArray[Any], copy_on_convert: bool = True
    ) -> tuple[NDArray[Any], bool]:
        """Convert from g/cc to kg/m³ if needed.

        Parameters
        ----------
        arr : NDArray
            Density array
        copy_on_convert : bool, optional
            If True, create copy when converting

        Returns
        -------
        Tuple[NDArray, bool]
            (converted_array, was_converted)
        """
        arr = self._ensure_numeric(arr)
        maxabs = self._nanmax_abs(arr)

        if maxabs == float("inf"):
            return arr, False

        if maxabs < self.threshold:  # Likely g/cc, convert to kg/m³
            if copy_on_convert:
                return arr.astype(float) * self.conversion_factor, True
            arr[...] = arr * self.conversion_factor
            return arr, True

        return arr, False
