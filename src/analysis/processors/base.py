"""Base classes for processor hierarchy."""

import logging
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .boundary import CubeAligner

logger = logging.getLogger(__name__)

__all__ = ["Processor", "BaseProcessor"]


class Processor(ABC):
    """Abstract base class for all data processors (polymorphic interface).

    Defines the common interface that all processors must implement.
    This enables treating different processor types uniformly while maintaining
    their specific implementations.

    Notes
    -----
    All concrete processor subclasses must implement the process() method,
    which represents the main computational task for that processor.
    """

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the processor's main computational task.

        All subclasses must implement this method. This ensures all processors
        can be called with a consistent interface.

        Parameters
        ----------
        *args
            Variable positional arguments (specific to each processor type).
        **kwargs
            Variable keyword arguments (specific to each processor type).

        Returns
        -------
        Any
            Result of the processor operation (type varies by processor).
        """
        pass


class BaseProcessor(Processor):
    """Base class for data processors providing shared initialization and utilities.

    Eliminates code duplication across processor classes by providing common
    dependencies like CubeAligner in a single location. Also implements smart
    process() delegation to domain-specific methods (detect, extract, calculate, analyze).

    Subclasses should implement one of the domain-specific methods:
    - detect() for boundary detection operations
    - extract() for amplitude extraction operations
    - calculate() for correlation/computation operations
    - analyze() for analysis operations

    The process() method will automatically delegate to whichever method
    is implemented by the subclass, eliminating boilerplate code.
    """

    # Define the method resolution order for finding domain methods
    _DOMAIN_METHODS = ["detect", "extract", "calculate", "analyze"]

    def __init__(self) -> None:
        """Initialize base processor with shared dependencies."""
        # Use lazy initialization to avoid circular imports and infinite recursion
        self._aligner_instance: "CubeAligner | None" = None
        logger.debug(f"Initialized {self.__class__.__name__}")

    @property
    def _aligner(self) -> "CubeAligner":
        """Lazy-load the CubeAligner on first access.

        Avoids circular imports and prevents infinite recursion issues
        that could occur if CubeAligner were eagerly initialized.

        Returns
        -------
        CubeAligner
            Shared aligner instance for this processor.
        """
        if self._aligner_instance is None:
            # Import here to avoid circular imports
            from .boundary import CubeAligner

            self._aligner_instance = CubeAligner()
        return self._aligner_instance

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Execute processor via automatic domain-method delegation.

        Implements the polymorphic interface by finding and delegating to
        the appropriate domain-specific method for this processor instance.
        This eliminates repetitive process() implementations across subclasses.

        The method resolution order is: detect → extract → calculate → analyze

        Parameters
        ----------
        *args
            Variable positional arguments (specific to the domain method).
        **kwargs
            Variable keyword arguments (specific to the domain method).

        Returns
        -------
        Any
            Result of the domain operation (type varies by processor).

        Raises
        ------
        NotImplementedError
            If the processor implements none of the expected domain methods.

        Examples
        --------
        >>> detector = BoundaryDetector()
        >>> boundaries = detector.process(facies_cube)  # Calls detect()

        >>> extractor = BoundaryAmplitudeExtractor()
        >>> result = extractor.process(seismic, boundaries)  # Calls extract()
        """
        for method_name in self._DOMAIN_METHODS:
            if hasattr(self, method_name):
                domain_method = getattr(self, method_name)
                return domain_method(*args, **kwargs)

        raise NotImplementedError(
            f"{self.__class__.__name__} must implement one of: "
            f"{', '.join(self._DOMAIN_METHODS)}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make processor callable via process() method.

        Enables using processor instance as a function:
            detector = BoundaryDetector()
            boundaries = detector(facies_cube)  # Calls process() internally

        Parameters
        ----------
        *args
            Variable positional arguments.
        **kwargs
            Variable keyword arguments.

        Returns
        -------
        Any
            Result of the process() method.
        """
        return self.process(*args, **kwargs)
