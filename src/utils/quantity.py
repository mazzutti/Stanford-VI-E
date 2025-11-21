"""Lightweight Quantity (unit-aware array) helper using OOP composition.

Provides a simple wrapper around numpy arrays that carries a unit string and
offers safe conversions to common geophysical units via converter strategies.
"""

from __future__ import annotations

import logging
from typing import Any, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.utils.units import UnitRegistry, get_unit_registry

logger = logging.getLogger(__name__)


class Quantity:
    """Unit-aware array wrapper with conversion support.

    Uses composition with UnitRegistry converters to handle unit conversions
    in a clean, extensible OOP manner.

    Usage:
        q = Quantity(np.array([1.5, 2.0]), 'km/s')
        q_m = q.to('m/s')
    """

    def __init__(
        self, array: ArrayLike, unit: str, registry: UnitRegistry | None = None
    ):
        """Initialize Quantity with array, unit, and optional registry.

        Args:
            array: Array-like data
            unit: Unit string (e.g., 'm/s', 'kg/m3')
            registry: Optional UnitRegistry; uses singleton if not provided
        """
        self._array = np.asarray(array)
        self.unit = unit.strip()
        self._registry = registry or get_unit_registry()

    @property
    def array(self) -> NDArray[Any]:
        """Get underlying numpy array."""
        return self._array

    def copy(self) -> Quantity:
        """Create an independent copy with same registry."""
        return Quantity(self._array.copy(), self.unit, self._registry)

    def to(self, unit: str, copy: bool = True) -> Quantity:
        """Convert to target unit using registry converters.

        Args:
            unit: Target unit string
            copy: Whether to copy the array

        Returns:
            New Quantity in target unit

        Raises:
            ValueError: If conversion is not supported
        """
        unit = unit.strip()
        if unit == self.unit:
            return self.copy() if copy else self

        # Delegate to registry
        try:
            converted_array = self._registry.convert(self._array, self.unit, unit)
            return Quantity(converted_array, unit, self._registry)
        except ValueError as e:
            raise ValueError(f"Cannot convert from {self.unit} to {unit}") from e

    def to_numpy(self) -> NDArray[Any]:
        """Export as numpy array."""
        return self._array

    def __array__(self) -> NDArray[Any]:
        """Support numpy's array protocol."""
        return self._array

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape."""
        return tuple(self._array.shape)

    def __len__(self) -> int:
        """Length of first dimension."""
        try:
            return len(self._array)
        except (TypeError, AttributeError):
            return 0

    def __repr__(self) -> str:
        """String representation."""
        return f"Quantity(shape={self._array.shape}, unit='{self.unit}')"

    # Arithmetic operations
    @overload
    def __add__(self, other: Quantity) -> Quantity: ...

    @overload
    def __add__(self, other: float | int | NDArray[np.floating[Any]]) -> Quantity: ...

    def __add__(self, other: Quantity | float | int | NDArray[Any]) -> Quantity:
        """Addition with automatic unit conversion."""
        if isinstance(other, Quantity):
            if other.unit != self.unit:
                other = other.to(self.unit)
            return Quantity(self._array + other._array, self.unit, self._registry)
        return Quantity(self._array + other, self.unit, self._registry)

    def __radd__(self, other: float | int | NDArray[np.floating[Any]]) -> Quantity:
        """Right addition."""
        return self.__add__(other)

    @overload
    def __mul__(self, other: float | int) -> Quantity: ...

    @overload
    def __mul__(self, other: Quantity) -> NDArray[np.floating[Any]]: ...

    @overload
    def __mul__(
        self, other: NDArray[np.floating[Any]]
    ) -> Quantity | NDArray[np.floating[Any]]: ...

    def __mul__(
        self, other: Quantity | float | int | NDArray[Any]
    ) -> Quantity | NDArray[Any]:
        """Multiplication."""
        if isinstance(other, (int, float)):
            return Quantity(self._array * other, self.unit, self._registry)
        if isinstance(other, Quantity):
            # Ambiguous unit result; return raw product
            arr_result = self._array * other._array
            return np.asarray(arr_result)
        # Multiply by ndarray - result is still a Quantity with same unit
        result = self._array * other
        return Quantity(result, self.unit, self._registry)

    @overload
    def __rmul__(self, other: float | int) -> Quantity: ...

    @overload
    def __rmul__(
        self, other: NDArray[np.floating[Any]]
    ) -> Quantity | NDArray[np.floating[Any]]: ...

    def __rmul__(self, other: float | int | NDArray[Any]) -> Quantity | NDArray[Any]:
        """Right multiplication."""
        return self.__mul__(other)


__all__ = ["Quantity"]


def to_ndarray(obj: NDArray[Any] | Quantity | Any) -> NDArray[Any]:
    """Normalize a `Quantity` or array-like into a plain `ndarray[Any]`.

    This helper centralizes the runtime check and gives the type checker a
    single, well-typed place to convert Quantity->ndarray.
    """
    return obj.array if isinstance(obj, Quantity) else np.asarray(obj)


__all__.append("to_ndarray")
