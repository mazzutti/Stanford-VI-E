"""Core abstractions and interfaces for the processing module.

Provides abstract base classes and interfaces that define contracts
for different processing components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, TypeVar
import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "Processor",
    "Manager",
    "Resampler",
    "MaterialProperty",
    "Validator",
]

T = TypeVar("T")
R = TypeVar("R")


class Processor(ABC):
    """Abstract base class for data processors.

    Processors transform input data through a well-defined process method.
    """

    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """Process input data.

        Args:
            data: Input data to process
            **kwargs: Additional processing parameters

        Returns:
            Processed data
        """
        pass


class Manager(ABC):
    """Abstract base class for managers.

    Managers handle resource lifecycle operations (clearing, summarizing, etc.).
    """

    @abstractmethod
    def clear(self, *args, **kwargs) -> int:
        """Clear managed resources.

        Returns:
            Number of resources cleared
        """
        pass

    @abstractmethod
    def summarize(self, *args, **kwargs) -> None:
        """Print summary of managed resources."""
        pass


class Resampler(ABC):
    """Abstract base class for resampling operations.

    Resampler implementations handle depth/time conversions and domain transformations.
    """

    @abstractmethod
    def resample(self, data: ArrayLike, plan: Any) -> np.ndarray:
        """Resample data according to plan.

        Args:
            data: Input data to resample
            plan: Resampling plan with target specifications

        Returns:
            Resampled data array
        """
        pass

    @abstractmethod
    def inverse_resample(self, data: ArrayLike, plan: Any) -> np.ndarray:
        """Inverse resampling operation.

        Args:
            data: Input data to inverse resample
            plan: Resampling plan

        Returns:
            Inverse resampled data array
        """
        pass


class MaterialProperty(ABC):
    """Abstract base class for material properties.

    Represents physical properties with unit awareness and validation.
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


class Validator(ABC):
    """Abstract base class for validators.

    Validators check data/model validity and provide diagnostic reports.
    """

    @abstractmethod
    def validate(self, *args, **kwargs) -> Dict[str, Any]:
        """Validate input and return report.

        Returns:
            Dictionary with validation results
        """
        pass

    @abstractmethod
    def is_valid(self, *args, **kwargs) -> bool:
        """Quick validity check.

        Returns:
            True if valid, False otherwise
        """
        pass
