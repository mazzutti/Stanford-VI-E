"""Base class for unit-aware material property models."""

from abc import ABC, abstractmethod
import numpy as np

__all__ = ["MaterialModel"]


class MaterialModel(ABC):
    """Base class for unit-aware material property models.

    Subclasses provide specific conversions and validation logic for different
    property types (velocity, density, etc.).
    """

    @abstractmethod
    def get_data(self) -> np.ndarray:
        """Return the underlying data array."""
        pass

    @abstractmethod
    def set_data(self, data: np.ndarray) -> None:
        """Update the underlying data array."""
        pass

    @abstractmethod
    def ensure_units(self) -> bool:
        """Convert to expected units if needed.

        Returns:
            True if conversion occurred, False otherwise
        """
        pass

    @abstractmethod
    def validate(self) -> None:
        """Validate data is finite and physically meaningful.

        Raises:
            ValueError: If validation fails
        """
        pass
