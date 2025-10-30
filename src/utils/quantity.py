"""Lightweight Quantity (unit-aware array) helper.

Provides a small wrapper around numpy arrays that carries a unit string and
offers safe conversions to common geophysical units used in this project
(m/s, km/s, kg/m3, g/cc). The goal is to make unit conversions explicit and
easy to test while remaining lightweight.
"""

from __future__ import annotations

from typing import Any
from numpy.typing import ArrayLike

import numpy as np
import logging

from src.utils.units import UnitRegistry


class Quantity:
    """A minimal unit-aware array wrapper.

    Usage:
        q = Quantity(np.array([1.5, 2.0]), 'km/s')
        q_m = q.to('m/s')
    """

    def __init__(self, array: ArrayLike, unit: str):
        self._array = np.asarray(array)
        # normalize unit aliases
        unit = unit.strip()

        # common velocity aliases
        if unit in ("m_per_s", "m/s"):
            unit = "m/s"
        if unit in ("km_per_s", "km/s"):
            unit = "km/s"

        # density aliases
        if unit in ("g/cc", "g/cm3", "g/cm^3"):
            unit = "g/cc"
        if unit in ("kg/m3", "kg/m^3", "kg/m³"):
            unit = "kg/m3"

        self.unit = unit

    @property
    def array(self) -> np.ndarray:
        return self._array

    def copy(self) -> "Quantity":
        return Quantity(self._array.copy(), self.unit)

    def to(self, unit: str, copy: bool = True) -> "Quantity":
        """Return a Quantity converted to `unit`.

        Supports conversions between m/s <-> km/s and kg/m3 <-> g/cc. If an
        unknown combination is requested, best-effort conversions via
        `UnitRegistry` are attempted.
        """
        unit = unit.strip()
        if unit == self.unit:
            return self.copy() if copy else self

        # Velocity conversions
        if unit in ("m/s", "m_per_s") or self.unit in ("m/s", "m_per_s"):
            # If target is m/s, try to coerce from km/s or via UnitRegistry
            if unit in ("m/s", "m_per_s"):
                if self.unit in ("km/s", "km_per_s"):
                    return Quantity(self._array * 1000.0, "m/s")
                # Best-effort: let UnitRegistry handle heuristics
                arr, _ = UnitRegistry.ensure_m_per_s(self._array, copy_on_convert=True)
                return Quantity(arr, "m/s")
            else:  # target is km/s
                if self.unit in ("m/s", "m_per_s"):
                    return Quantity(self._array / 1000.0, "km/s")
                arr, _ = UnitRegistry.ensure_m_per_s(self._array, copy_on_convert=True)
                return Quantity(arr / 1000.0, "km/s")

        # Density conversions
        if unit in ("kg/m3", "kg/m^3", "kg/m³") or self.unit in (
            "kg/m3",
            "kg/m^3",
            "kg/m³",
        ):
            if unit in ("kg/m3", "kg/m^3", "kg/m³"):
                if self.unit in ("g/cc", "g/cm3", "g/cm^3"):
                    return Quantity(self._array * 1000.0, "kg/m3")
                arr, _ = UnitRegistry.ensure_kg_per_m3(
                    self._array, copy_on_convert=True
                )
                return Quantity(arr, "kg/m3")
            else:
                # convert to g/cc
                if self.unit in ("kg/m3", "kg/m^3", "kg/m³"):
                    return Quantity(self._array / 1000.0, "g/cc")
                arr, _ = UnitRegistry.ensure_kg_per_m3(
                    self._array, copy_on_convert=True
                )
                return Quantity(arr / 1000.0, "g/cc")

        # Fallback: if UnitRegistry can help, prefer it and mark unit as target
        if unit in ("m/s", "km/s"):
            arr, _ = UnitRegistry.ensure_m_per_s(self._array, copy_on_convert=True)
            if unit == "m/s":
                return Quantity(arr, "m/s")
            return Quantity(arr / 1000.0, "km/s")

        if unit in ("kg/m3", "g/cc"):
            arr, _ = UnitRegistry.ensure_kg_per_m3(self._array, copy_on_convert=True)
            if unit == "kg/m3":
                return Quantity(arr, "kg/m3")
            return Quantity(arr / 1000.0, "g/cc")

        raise ValueError(f"Unsupported target unit: {unit}")

    def to_numpy(self) -> np.ndarray:
        return self._array

    def __array__(self):
        # Support numpy's array protocol
        return self._array

    @property
    def shape(self):
        return self._array.shape

    def __len__(self):
        try:
            return len(self._array)
        except Exception:
            return 0

    def __repr__(self) -> str:
        return f"Quantity(shape={self._array.shape}, unit='{self.unit}')"

    # Basic arithmetic helpers
    def __add__(self, other: Any) -> "Quantity":
        if isinstance(other, Quantity):
            if other.unit == self.unit:
                return Quantity(self._array + other._array, self.unit)
            other_conv = other.to(self.unit)
            return Quantity(self._array + other_conv._array, self.unit)
        return Quantity(self._array + other, self.unit)

    def __mul__(self, other: Any):
        if isinstance(other, (int, float)):
            return Quantity(self._array * other, self.unit)
        if isinstance(other, Quantity):
            # ambiguous unit result — return raw ndarray product
            return self._array * other._array
        return Quantity(self._array * other, self.unit)

    __rmul__ = __mul__


__all__ = ["Quantity"]
# Module logger
logger = logging.getLogger(__name__)
