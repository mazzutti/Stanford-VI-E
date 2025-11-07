"""Domain-specific handling strategies for depth/time processing.

This module provides polymorphic domain handlers that eliminate
string-based conditionals and follow the Strategy pattern.

Example:
    Get a handler for a specific domain and process cubes:

    >>> handler = DomainHandlerFactory.get_handler(Domain.TIME)
    >>> avo_display, facies_display = handler.prepare_display_cubes(
    ...     vm=time_resampler,
    ...     facies_depth=facies_cube,
    ...     avo=avo_cube,
    ...     grid_spec=grid_spec
    ... )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generator, NamedTuple, Protocol
from types import TracebackType
import logging
from contextlib import contextmanager

import numpy as np
from numpy.typing import NDArray

from src.analysis.domain.enum import Domain
from src.analysis.types.protocols import TimeResampler
from src.io.grid import GridSpec

__all__ = [
    "DisplayCubes",
    "CubeProcessor",
    "DomainHandler",
    "DepthDomainHandler",
    "TimeDomainHandler",
    "DomainHandlerRegistry",
    "DomainHandlerFactory",
    "HandlerStatistics",
]

logger = logging.getLogger(__name__)

# Type aliases for better clarity
HandlerFactory = Callable[[], "DomainHandler"]


class HandlerStatistics(NamedTuple):
    """Statistics about handler usage and performance.

    Attributes
    ----------
    domain : Domain
        The domain this handler manages.
    is_initialized : bool
        Whether the handler has been initialized.
    call_count : int
        Number of times prepare_display_cubes was called.
    total_runtime_ms : float
        Total runtime in milliseconds for all calls.
    average_runtime_ms : float
        Average runtime per call in milliseconds.
    """

    domain: Domain
    is_initialized: bool
    call_count: int
    total_runtime_ms: float
    average_runtime_ms: float

    def __str__(self) -> str:
        """Return formatted statistics string."""
        return (
            f"Handler({self.domain.name}): "
            f"initialized={self.is_initialized}, "
            f"calls={self.call_count}, "
            f"avg_time={self.average_runtime_ms:.2f}ms"
        )


class DisplayCubes(NamedTuple):
    """Semantic container for display-ready cubes.

    Attributes
    ----------
    avo : NDArray[np.float64]
        AVO cube prepared for display.
    facies : NDArray[np.int64]
        Facies cube prepared for display.
    """

    avo: NDArray[np.float64]
    facies: NDArray[np.int64]


class CubeProcessor(Protocol):
    """Protocol defining the interface for cube processors.

    This Protocol documents the expected interface for domain-specific handlers
    and enables better type checking and IDE support without requiring inheritance.

    Example:
        Any object implementing this protocol can be used as a cube processor:

        >>> def process_with_any_handler(processor: CubeProcessor) -> DisplayCubes:
        ...     return processor.prepare_display_cubes(...)
    """

    @property
    def domain(self) -> Domain:
        """The domain this processor manages."""
        ...

    def prepare_display_cubes(
        self,
        vm: TimeResampler,
        facies_depth: NDArray[np.int64],
        avo: NDArray[np.float64],
        grid_spec: GridSpec,
    ) -> DisplayCubes:
        """Prepare AVO and facies cubes for display in this domain."""
        ...


@dataclass(frozen=True)
class DomainHandler(ABC):
    """Abstract base for domain-specific processing strategies.

    Immutable handler that implements the Strategy pattern for domain-specific
    processing. Subclasses must implement the abstract methods.

    Supports lifecycle management via initialize()/cleanup() and context manager
    protocol for automatic resource cleanup.

    Example:
        Using as context manager for automatic cleanup:

        >>> with DomainHandlerFactory.get_handler(Domain.TIME) as handler:
        ...     result = handler.prepare_display_cubes(...)
    """

    domain: Domain
    """The domain this handler manages."""

    def __post_init__(self) -> None:
        """Initialize handler state tracking."""
        object.__setattr__(self, "_is_initialized", False)
        object.__setattr__(self, "_call_count", 0)
        object.__setattr__(self, "_total_runtime_ms", 0.0)

    def __repr__(self) -> str:
        """Return detailed string representation."""
        init_status = "initialized" if self.is_initialized else "not-initialized"
        return (
            f"{self.__class__.__name__}("
            f"domain={self.domain.name}, "
            f"status={init_status})"
        )

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        return f"{self.__class__.__name__}({self.domain.name})"

    @abstractmethod
    def prepare_display_cubes(
        self,
        vm: TimeResampler,
        facies_depth: NDArray[np.int64],
        avo: NDArray[np.float64],
        grid_spec: GridSpec,
    ) -> DisplayCubes:
        """Prepare AVO and facies cubes for display in this domain.

        Parameters
        ----------
        vm
            Time resampler instance.
        facies_depth
            Facies cube in depth domain.
        avo
            AVO cube.
        grid_spec
            Grid specification.

        Returns
        -------
        DisplayCubes
            Named tuple with avo and facies cubes prepared for this domain.
        """
        pass

    def initialize(self) -> None:
        """Initialize handler resources if needed.

        Override this method to perform setup tasks when the handler
        is first registered or activated. The base implementation
        marks the handler as initialized.
        """
        object.__setattr__(self, "_is_initialized", True)
        logger.debug(f"Handler for {self.domain.name} initialized")

    def cleanup(self) -> None:
        """Clean up handler resources if needed.

        Override this method to perform teardown tasks such as
        releasing cached data or closing connections.
        """
        logger.debug(f"Handler for {self.domain.name} cleaned up")

    @property
    def is_initialized(self) -> bool:
        """Check if handler has been initialized."""
        return getattr(self, "_is_initialized", False)

    @property
    def call_count(self) -> int:
        """Get number of times prepare_display_cubes was called."""
        return getattr(self, "_call_count", 0)

    @property
    def total_runtime_ms(self) -> float:
        """Get total runtime in milliseconds for all calls."""
        return getattr(self, "_total_runtime_ms", 0.0)

    @property
    def average_runtime_ms(self) -> float:
        """Get average runtime per call in milliseconds."""
        if self.call_count == 0:
            return 0.0
        return self.total_runtime_ms / self.call_count

    def get_statistics(self) -> HandlerStatistics:
        """Get usage statistics for this handler.

        Returns
        -------
        HandlerStatistics
            Detailed statistics about handler usage.
        """
        return HandlerStatistics(
            domain=self.domain,
            is_initialized=self.is_initialized,
            call_count=self.call_count,
            total_runtime_ms=self.total_runtime_ms,
            average_runtime_ms=self.average_runtime_ms,
        )

    def __enter__(self) -> "DomainHandler":
        """Enter context manager - ensures handler is initialized.

        Returns
        -------
        DomainHandler
            Self for use in with statement.
        """
        if not self.is_initialized:
            self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - performs cleanup.

        Parameters
        ----------
        exc_type
            Exception type if an exception occurred.
        exc_val
            Exception value if an exception occurred.
        exc_tb
            Exception traceback if an exception occurred.
        """
        try:
            self.cleanup()
        except Exception as e:
            logger.exception(f"Error during cleanup of {self.domain.name} handler: {e}")


class DepthDomainHandler(DomainHandler):
    """Handler for depth-domain processing (no transformation needed)."""

    def __init__(self) -> None:
        """Initialize the depth domain handler."""
        object.__setattr__(self, "domain", Domain.DEPTH)

    def __repr__(self) -> str:
        """Return detailed representation."""
        return super().__repr__()

    def prepare_display_cubes(
        self,
        vm: TimeResampler,
        facies_depth: NDArray[np.int64],
        avo: NDArray[np.float64],
        grid_spec: GridSpec,
    ) -> DisplayCubes:
        """Return cubes unchanged (already in depth domain).

        Parameters
        ----------
        vm
            Time resampler instance.
        facies_depth
            Facies cube in depth domain.
        avo
            AVO cube.
        grid_spec
            Grid specification.

        Returns
        -------
        DisplayCubes
            Named tuple with (avo, facies_depth) unchanged.
        """
        logger.info("Processing in DEPTH domain (no transformation required)")
        return DisplayCubes(avo=avo, facies=facies_depth)


class TimeDomainHandler(DomainHandler):
    """Handler for time-domain processing (requires resampling)."""

    def __init__(self) -> None:
        """Initialize the time domain handler."""
        object.__setattr__(self, "domain", Domain.TIME)

    def __repr__(self) -> str:
        """Return detailed representation."""
        return super().__repr__()

    def prepare_display_cubes(
        self,
        vm: TimeResampler,
        facies_depth: NDArray[np.int64],
        avo: NDArray[np.float64],
        grid_spec: GridSpec,
    ) -> DisplayCubes:
        """Resample facies to time domain.

        Parameters
        ----------
        vm
            Time resampler instance.
        facies_depth
            Facies cube in depth domain.
        avo
            AVO cube.
        grid_spec
            Grid specification with target dt.

        Returns
        -------
        DisplayCubes
            Named tuple with (avo_display, facies_time) where facies is
            resampled to time domain.
        """
        logger.info("Processing in TIME domain (resampling facies to time)")
        facies_time, _dt = vm.resample_to_time(
            facies_depth,
            is_categorical=True,
            target_dt=grid_spec.dt,
        )
        return DisplayCubes(avo=avo, facies=facies_time)


class DomainHandlerRegistry:
    """Registry for domain handler instances using the registry pattern.

    This design allows new handlers to be registered without modifying
    the factory class, supporting the Open/Closed principle.

    Features:
    - Lazy initialization of handlers (on first access)
    - Lifecycle management (initialize/cleanup)
    - State tracking for debugging and monitoring
    - Exception-safe cleanup
    - Usage statistics per handler

    Example:
        Create and use a registry:

        >>> registry = DomainHandlerRegistry()
        >>> handler = registry.get_handler(Domain.DEPTH)
        >>> stats = registry.get_handler_statistics(Domain.DEPTH)
        >>> registry.cleanup_all()
    """

    def __init__(self) -> None:
        """Initialize the registry with built-in handler factories."""
        self._handlers: dict[Domain, DomainHandler] = {}
        self._handler_factories: dict[Domain, HandlerFactory] = {
            Domain.DEPTH: DepthDomainHandler,
            Domain.TIME: TimeDomainHandler,
        }
        self._initialized: set[Domain] = set()
        logger.debug("DomainHandlerRegistry initialized")

    def __repr__(self) -> str:
        """Return string representation of registry state."""
        return (
            f"DomainHandlerRegistry("
            f"initialized={len(self._initialized)}, "
            f"factories={len(self._handler_factories)})"
        )

    def _ensure_initialized(self, domain: Domain) -> None:
        """Lazily initialize a handler on first access.

        Parameters
        ----------
        domain
            The domain to initialize.

        Raises
        ------
        RuntimeError
            If handler initialization fails.
        """
        if domain in self._initialized:
            return

        if domain not in self._handler_factories:
            return

        try:
            factory = self._handler_factories[domain]
            handler = factory()
            handler.initialize()
            self._handlers[domain] = handler
            self._initialized.add(domain)
            logger.debug(
                f"Lazy-initialized handler for domain {domain.name} "
                f"({handler.__class__.__name__})"
            )
        except Exception as e:
            logger.exception(
                f"Failed to initialize handler for domain {domain.name}: {e}"
            )
            raise RuntimeError(f"Cannot initialize handler for {domain.name}") from e

    def register(self, domain: Domain, handler: DomainHandler) -> None:
        """Register a handler for a specific domain.

        Parameters
        ----------
        domain
            The domain enum value.
        handler
            The handler instance to register.

        Raises
        ------
        ValueError
            If a handler is already registered and initialized for this domain.
        """
        if domain in self._initialized and domain in self._handlers:
            logger.warning(
                f"Overwriting initialized handler for domain {domain.name}. "
                f"Consider calling cleanup() first."
            )
        self._handlers[domain] = handler
        self._handler_factories.pop(domain, None)
        self._initialized.add(domain)
        logger.debug(
            f"Registered handler for domain {domain.name} "
            f"({handler.__class__.__name__})"
        )

    def get_handler(self, domain: Domain) -> DomainHandler:
        """Get the handler for a specific domain.

        Lazily initializes the handler on first access if needed.

        Parameters
        ----------
        domain
            A Domain enum value.

        Returns
        -------
        DomainHandler
            Handler implementing domain-specific logic.

        Raises
        ------
        ValueError
            If domain is not registered or recognized.
        RuntimeError
            If handler initialization fails.
        """
        self._ensure_initialized(domain)

        if domain not in self._handlers:
            available = list(self._handler_factories.keys()) + list(
                self._handlers.keys()
            )
            available_names = [d.name for d in available]
            raise ValueError(
                f"No handler registered for domain: {domain.name}. "
                f"Available domains: {available_names}"
            )
        return self._handlers[domain]

    def get_all_handlers(self) -> dict[Domain, DomainHandler]:
        """Get all currently initialized handlers.

        Returns
        -------
        dict[Domain, DomainHandler]
            Dictionary mapping domains to their handlers.
        """
        return dict(self._handlers)

    def is_initialized(self, domain: Domain) -> bool:
        """Check if a domain's handler has been initialized.

        Parameters
        ----------
        domain
            The domain to check.

        Returns
        -------
        bool
            True if handler is initialized, False otherwise.
        """
        return domain in self._initialized

    def get_handler_statistics(self, domain: Domain) -> HandlerStatistics:
        """Get usage statistics for a specific handler.

        Parameters
        ----------
        domain
            The domain to get statistics for.

        Returns
        -------
        HandlerStatistics
            Statistics about the handler's usage.

        Raises
        ------
        ValueError
            If domain is not registered.
        """
        handler = self.get_handler(domain)
        return handler.get_statistics()

    def get_all_statistics(self) -> list[HandlerStatistics]:
        """Get statistics for all initialized handlers.

        Returns
        -------
        list[HandlerStatistics]
            List of statistics for each initialized handler.
        """
        return [handler.get_statistics() for handler in self._handlers.values()]

    def cleanup_all(self) -> None:
        """Clean up all registered handlers.

        Call this when shutting down the application to allow handlers
        to release resources. Continues cleanup even if one handler fails.
        """
        errors = []
        for domain, handler in self._handlers.items():
            try:
                handler.cleanup()
                logger.debug(f"Cleaned up handler for domain {domain.name}")
            except Exception as e:
                error_msg = f"Error cleaning up {domain.name} handler: {e}"
                logger.exception(error_msg)
                errors.append(error_msg)

        if errors:
            logger.warning(f"Cleanup completed with {len(errors)} error(s)")
        else:
            logger.debug("All handlers cleaned up successfully")


# Global registry instance for singleton access
_default_registry = DomainHandlerRegistry()


class DomainHandlerFactory:
    """Factory for creating and accessing appropriate domain handlers.

    Provides convenient static methods for accessing handlers from the global
    registry. For more control over lifecycle, instantiate DomainHandlerRegistry.

    Features:
    - Lazy initialization of handlers
    - Automatic resource cleanup via cleanup()
    - Context manager support for automatic cleanup
    - State tracking and introspection
    - Usage statistics and monitoring

    Example:
        Basic usage:

        >>> handler = DomainHandlerFactory.get_handler(Domain.DEPTH)
        >>> result = handler.prepare_display_cubes(...)

        Using context manager for automatic cleanup:

        >>> with DomainHandlerFactory.get_handler(Domain.TIME) as handler:
        ...     result = handler.prepare_display_cubes(...)
        >>> DomainHandlerFactory.cleanup()  # Clean up remaining handlers

        Using context manager factory (recommended):

        >>> with DomainHandlerFactory.handler_context(Domain.TIME) as handler:
        ...     result = handler.prepare_display_cubes(...)
        # Automatic cleanup at the end of the with block
    """

    @classmethod
    def get_handler(cls, domain: Domain) -> DomainHandler:
        """Get the handler for a specific domain.

        Lazily initializes the handler on first access.

        Parameters
        ----------
        domain
            A Domain enum value.

        Returns
        -------
        DomainHandler
            Handler implementing domain-specific logic.

        Raises
        ------
        ValueError
            If domain is not recognized.
        RuntimeError
            If handler initialization fails.
        """
        return _default_registry.get_handler(domain)

    @classmethod
    @contextmanager
    def handler_context(cls, domain: Domain) -> Generator[DomainHandler, None, None]:
        """Get a handler as a context manager for automatic cleanup.

        This is the recommended way to use handlers when you want automatic
        cleanup after use.

        Parameters
        ----------
        domain
            A Domain enum value.

        Yields
        ------
        DomainHandler
            Handler for use within the context.

        Example:
            >>> with DomainHandlerFactory.handler_context(Domain.TIME) as handler:
            ...     result = handler.prepare_display_cubes(...)
            # Automatic cleanup happens here
        """
        handler = cls.get_handler(domain)
        try:
            yield handler
        finally:
            try:
                handler.cleanup()
            except Exception as e:
                logger.exception(f"Error in handler cleanup for {domain.name}: {e}")

    @classmethod
    def register_handler(cls, domain: Domain, handler: DomainHandler) -> None:
        """Register a custom handler for a domain.

        Allows runtime registration of custom handlers, overriding the default.

        Parameters
        ----------
        domain
            The domain enum value.
        handler
            The handler instance to register.
        """
        _default_registry.register(domain, handler)
        logger.info(
            f"Custom handler registered for domain {domain.name}: "
            f"{handler.__class__.__name__}"
        )

    @classmethod
    def cleanup(cls) -> None:
        """Clean up all registered handlers.

        Call this during application shutdown to allow handlers to release
        resources like cached data or external connections. This should be
        called in try/finally or via context manager to ensure cleanup occurs.

        Example:
            >>> try:
            ...     # Process data with handlers
            ...     handler = DomainHandlerFactory.get_handler(Domain.TIME)
            ... finally:
            ...     DomainHandlerFactory.cleanup()
        """
        logger.info("Cleaning up all domain handlers")
        _default_registry.cleanup_all()

    @classmethod
    def is_handler_initialized(cls, domain: Domain) -> bool:
        """Check if a handler has been initialized for a domain.

        Parameters
        ----------
        domain
            The domain to check.

        Returns
        -------
        bool
            True if handler is initialized, False otherwise.
        """
        return _default_registry.is_initialized(domain)

    @classmethod
    def get_all_handlers(cls) -> dict[Domain, DomainHandler]:
        """Get all currently initialized handlers.

        Returns
        -------
        dict[Domain, DomainHandler]
            Dictionary mapping domains to initialized handlers.
        """
        return _default_registry.get_all_handlers()

    @classmethod
    def get_statistics(cls, domain: Domain) -> HandlerStatistics:
        """Get usage statistics for a specific handler.

        Parameters
        ----------
        domain
            The domain to get statistics for.

        Returns
        -------
        HandlerStatistics
            Statistics about the handler's usage.

        Raises
        ------
        ValueError
            If domain is not registered.
        """
        return _default_registry.get_handler_statistics(domain)

    @classmethod
    def get_all_statistics(cls) -> list[HandlerStatistics]:
        """Get statistics for all initialized handlers.

        Returns
        -------
        list[HandlerStatistics]
            List of statistics for each handler.
        """
        return _default_registry.get_all_statistics()

    @classmethod
    def print_statistics(cls) -> None:
        """Print formatted statistics for all handlers to logger.

        Useful for debugging and monitoring handler usage.
        """
        stats_list = cls.get_all_statistics()
        if not stats_list:
            logger.info("No handler statistics available yet")
            return

        logger.info(f"Handler Statistics ({len(stats_list)} handlers):")
        for stats in stats_list:
            logger.info(f"  {stats}")
