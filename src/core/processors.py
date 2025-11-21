"""Unified processor base classes for the Stanford-VI-E framework.

Provides abstract base classes and concrete implementations that define contracts
for different processing components across both analysis and processing modules.

This module consolidates processor abstractions to eliminate duplication and
provide a single source of truth for processor interface definitions.

Patterns Used:
- Abstract Base Class: Define processor interface contract
- Template Method: Optional smart delegation to domain-specific methods
- Strategy: Allow different processor implementations

Example:
    >>> from src.core.processors import Processor, BaseProcessor
    >>>
    >>> class MyProcessor(BaseProcessor):
    ...     def detect(self, data):
    ...         '''Implement detection logic'''
    ...         return process_data(data)
    >>>
    >>> processor = MyProcessor()
    >>> result = processor(input_data)  # Calls detect() via process()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from src.analysis.processors.boundary import CubeAligner

logger = logging.getLogger(__name__)

__all__ = ["Processor", "BaseProcessor", "AutoLoggingMixin"]

# The processor module defines a few small base classes and mixins that
# intentionally expose a compact public API. Suppress the simple
# too-few-public-methods warning to focus lint on real issues.


class AutoLoggingMixin:
    """Mixin that automatically configures logging for any class.

    Eliminates the need for module-level logger declarations in every file.
    Each instance gets a logger named after its module and class.
    """

    _logger: logging.Logger | None = None

    @property
    def logger(self) -> logging.Logger:
        """Lazy logger initialization with class-based naming."""
        if self._logger is None:
            self._logger = logging.getLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger

    def log_operation(self, operation: str, level: int = logging.DEBUG) -> None:
        """Log an operation with automatic context."""
        self.logger.log(level, "%s: %s", self.__class__.__name__, operation)

    def log_error_with_context(self, error: Exception, context: str = "") -> None:
        """Log error with class context."""
        self.logger.error(
            "%s error in %s: %s",
            self.__class__.__name__,
            context,
            error,
            exc_info=True,
        )


# Module note: processor base classes are intentionally compact wrappers
# used across the codebase to provide consistent interfaces.
# Some methods intentionally perform imports inside properties to avoid
# circular imports; silence import-outside-toplevel so pylint focuses
# on actionable problems in the implementations.


class Processor(ABC):
    """Abstract base class for all data processors (unified interface).

    Defines the common interface that all processors must implement.
    This enables treating different processor types uniformly while maintaining
    their specific implementations across both analysis and processing modules.

    This unified class replaces previous duplicate implementations in:
    - src/analysis/processors/base.py
    - src/processing/core/abstracts.py

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


class BaseProcessor(Processor, AutoLoggingMixin):
    """Base class for data processors providing shared initialization and utilities.

    Eliminates code duplication across processor classes by providing common
    dependencies like CubeAligner in a single location. Also implements smart
    process() delegation to domain-specific methods (detect, extract, calculate, analyze).

    Subclasses should implement one of the domain-specific methods:
    - detect() for boundary detection operations
    - extract() for amplitude extraction operations
    - calculate() for correlation/computation operations
    - analyze() for analysis operations
    - align() for cube alignment and registration utilities

    The process() method will automatically delegate to whichever method
    is implemented by the subclass, eliminating boilerplate code.

    This implementation was previously in src/analysis/processors/base.py
    and is now unified in src/core.
    """

    # Define the method resolution order for finding domain methods
    _DOMAIN_METHODS = ["detect", "extract", "calculate", "analyze", "align"]

    def __init__(self) -> None:
        """Initialize base processor with shared dependencies."""
        # Use lazy initialization to avoid circular imports and infinite recursion
        self._aligner_instance: CubeAligner | None = None
        self.logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _aligner(self) -> CubeAligner:
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
            from src.analysis.processors.boundary import (
                CubeAligner,
            )

            self._aligner_instance = CubeAligner()
        return self._aligner_instance

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Execute processor via automatic domain-method delegation.

            Implements the polymorphic interface by finding and delegating to
            the appropriate domain-specific method for this processor instance.
            This eliminates repetitive process() implementations across subclasses.

        The method resolution order is: detect → extract → calculate → analyze → align

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
        method = self._find_domain_method()
        if method:
            return method(*args, **kwargs)
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement one of: "
            f"{', '.join(self._DOMAIN_METHODS)}"
        )

    def _find_domain_method(self) -> Callable[..., Any] | None:
        """Find first available domain method and return it if callable."""
        for method_name in self._DOMAIN_METHODS:
            if hasattr(self, method_name):
                attr = getattr(self, method_name)
                if callable(attr):
                    return cast(Callable[..., Any], attr)
        return None

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
