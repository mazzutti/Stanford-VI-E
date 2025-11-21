"""Base class for unit-aware material property models."""

from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray

__all__ = ["MaterialModel"]

class MaterialModel(ABC):
    """Base class for unit-aware material property models.

    Subclasses provide specific conversions and validation logic for different
    property types (velocity, density, etc.).
    """

    @abstractmethod
    def get_data(self) -> NDArray[Any]:
        """Return the underlying data array."""

    @abstractmethod
    def set_data(self, data: NDArray[Any]) -> None:
        """Update the underlying data array."""

    @abstractmethod
    def ensure_units(self) -> bool:
        """Convert to expected units if needed.

        Returns:
            True if conversion occurred, False otherwise
        """

    @abstractmethod
    def validate(self) -> None:
        """Validate data is finite and physically meaningful.

        Raises:
            ValueError: If validation fails
        """
