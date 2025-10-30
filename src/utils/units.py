"""Unit conversion helpers and simple registry for common geophysical units.

This module centralizes heuristics used across the project to detect and
convert common units (velocity in km/s -> m/s, density in g/cc -> kg/m3, etc.).
Functions return a tuple (array, converted_bool) so callers can log or decide
whether to persist conversions.
"""

from __future__ import annotations

from typing import Tuple
from numpy.typing import ArrayLike

import numpy as np
import logging


def _nanmax_abs(a: np.ndarray) -> float:
    try:
        return float(np.nanmax(np.abs(a)))
    except Exception:
        # if array is empty or not numeric, return large sentinel to avoid conversion
        return float("inf")


class UnitRegistry:
    """Small helper class with conversion heuristics.

    Methods are intentionally conservative and non-destructive:
    - They accept array-like input and return either the original array (no copy)
      when no conversion was needed, or a new numpy array when conversion applied.
    - They return a boolean flag indicating whether a conversion took place.
    """

    @staticmethod
    def ensure_m_per_s(
        arr: ArrayLike, *, copy_on_convert: bool = True
    ) -> Tuple[np.ndarray, bool]:
        """Ensure a velocity array is in meters/second.

        Heuristic: if max(abs(arr)) < 100, treat as km/s and multiply by 1000.

        Returns (array_in_m_per_s, converted_flag)
        """
        if arr is None:
            raise ValueError("arr is None")
        a = np.asarray(arr)
        # preserve object for non-numeric arrays
        if not np.issubdtype(a.dtype, np.number):
            # try to coerce to float
            a = a.astype(float)

        maxabs = _nanmax_abs(a)
        if maxabs == float("inf"):
            return a, False

        # If max is small (e.g., 0-10) it's likely km/s or similar
        if maxabs < 100:
            if copy_on_convert:
                return (a.astype(float) * 1000.0, True)
            else:
                a[...] = a * 1000.0
                return a, True

        # already in m/s (or other large unit) — do nothing
        return a, False

    @staticmethod
    def ensure_kg_per_m3(
        arr: ArrayLike, *, copy_on_convert: bool = True
    ) -> Tuple[np.ndarray, bool]:
        """Ensure a density array is in kg/m3.

        Heuristic: if max(abs(arr)) < 100, treat as g/cc and multiply by 1000.

        Returns (array_in_kg_per_m3, converted_flag)
        """
        if arr is None:
            raise ValueError("arr is None")
        a = np.asarray(arr)
        if not np.issubdtype(a.dtype, np.number):
            a = a.astype(float)

        maxabs = _nanmax_abs(a)
        if maxabs == float("inf"):
            return a, False

        if maxabs < 100:
            if copy_on_convert:
                return (a.astype(float) * 1000.0, True)
            else:
                a[...] = a * 1000.0
                return a, True

        return a, False

    @staticmethod
    def is_likely_in_unit(arr: ArrayLike, unit: str) -> bool:
        """Best-effort check whether an array is likely already in the requested unit.

        Supported units: 'm/s', 'kg/m3', 'km/s', 'g/cc'
        """
        if arr is None:
            return False
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
    def ensure_seconds(
        value: float,
        *,
        convert_threshold_low: float = 0.01,
        convert_threshold_high: float = 100.0,
    ) -> tuple[float, bool]:
        """Ensure a time value is in seconds.

        Heuristic: if the value is between `convert_threshold_low` and
        `convert_threshold_high` it's likely expressed in milliseconds (ms)
        as a small integer (e.g., 1, 2, ...). In that case divide by 1000.

        Returns (seconds, converted_bool).
        """
        try:
            v = float(value)
        except Exception:
            raise ValueError("value must be numeric")

        if convert_threshold_low <= v < convert_threshold_high:
            # treat as milliseconds
            return v / 1000.0, True
        return v, False

    @staticmethod
    def ensure_meters(
        value: float, *, convert_threshold: float = 0.1
    ) -> tuple[float, bool]:
        """Ensure a length value is in meters.

        Heuristic: if the provided value is smaller than `convert_threshold`
        it's likely in kilometers (e.g., 0.001 -> 1m); multiply by 1000.

        Returns (meters, converted_bool).
        """
        try:
            v = float(value)
        except Exception:
            raise ValueError("value must be numeric")

        if v < convert_threshold:
            return v * 1000.0, True
        return v, False


__all__ = ["UnitRegistry"]

# Module logger
logger = logging.getLogger(__name__)


from src.utils.facades import LazyObjectProxy


# Module-level lazy registry for gradual migration
unit_registry: UnitRegistry = LazyObjectProxy(lambda: UnitRegistry())


def get_unit_registry(instance: UnitRegistry | None = None) -> UnitRegistry:
    """Return provided UnitRegistry or the module-level lazy singleton."""
    return instance if instance is not None else unit_registry


__all__.extend(["unit_registry", "get_unit_registry"])
