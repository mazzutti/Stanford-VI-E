"""Core abstractions and interfaces for the processing module.

Provides abstract base classes and interfaces that define contracts
for different processing components.

Note: The Processor ABC has been consolidated into src.core.processors
for unified use across both analysis and processing modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, Any, Optional, List
from numpy.typing import ArrayLike, NDArray

# Import unified Processor from src.core
from src.core.processors import Processor

if TYPE_CHECKING:
    from src.processing.resampling._plan import ResamplePlan


__all__ = [
    "Processor",
    "Manager",
    "Resampler",
    "MaterialProperty",
    "Validator",
]


T = TypeVar("T")
R = TypeVar("R")


class Manager(ABC):
    """Abstract base class for managers.

    Managers handle resource lifecycle operations (clearing, summarizing, etc.).
    """

    @abstractmethod
    def clear(
        self,
        patterns: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
        prefix: str = "",
    ) -> int:
        """Clear managed resources.

        Args:
            patterns: Optional glob patterns to match
            cache_dir: Optional cache directory
            prefix: Prefix for log messages

        Returns:
            Number of resources cleared
        """
        pass

    @abstractmethod
    def summarize(
        self,
        cache_dir: str = ".cache",
        keys: Optional[List[str]] = None,
        prefix: str = "",
    ) -> None:
        """Print summary of managed resources.

        Args:
            cache_dir: Cache directory to summarize
            keys: Optional keys to filter
            prefix: Prefix for log messages
        """
        pass


class Resampler(ABC):
    """Abstract base class for resampling operations.

    Resampler implementations handle depth/time conversions and domain transformations.
    """

    @abstractmethod
    def resample(self, data: ArrayLike, plan: ResamplePlan) -> NDArray[Any]:
        """Resample data according to plan.

        Args:
            data: Input data to resample
            plan: Resampling plan with target specifications

        Returns:
            Resampled data array
        """
        pass

    @abstractmethod
    def inverse_resample(self, data: ArrayLike, plan: ResamplePlan) -> NDArray[Any]:
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
    def get_data(self) -> NDArray[Any]:
        """Return the underlying data array."""
        pass

    @abstractmethod
    def set_data(self, data: NDArray[Any]) -> None:
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
    def validate(self, *args: NDArray[Any], **kwargs: NDArray[Any]) -> Any:
        """Validate input and return report.

        Returns:
            Validation report (type depends on implementation)
        """
        pass

    @abstractmethod
    def is_valid(self, *args: NDArray[Any], **kwargs: NDArray[Any]) -> bool:
        """Quick validity check.

        Returns:
            True if valid, False otherwise
        """
        pass
