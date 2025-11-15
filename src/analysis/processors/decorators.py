"""Decorators for processor logging and performance monitoring."""

import logging
from functools import wraps
from typing import Any, TypeVar, cast
from collections.abc import Callable
from numpy.typing import NDArray

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ProcessorDecorators"]

F = TypeVar("F", bound=Callable[..., Any])


class ProcessorDecorators:
    """Factory for processor decorators providing logging and performance monitoring.

    Centralizes decorator logic for consistent behavior across all processor methods.
    """

    @staticmethod
    def log_debug(message_template: str) -> Callable[[F], F]:
        """Decorator to add consistent debug logging to processor methods.

        Reduces logging boilerplate by automatically logging method entry with
        a consistent format. The decorator extracts the method name from the
        calling function.

        Parameters
        ----------
        message_template : str
            Template string for the log message. Will be formatted with the
            method name from the calling function.

        Returns
        -------
        callable
            Decorator function that adds logging to the wrapped method.

        Examples
        --------
        >>> @ProcessorDecorators.log_debug("Starting {}...")
        ... def detect(self, facies_cube):
        ...     # Method body
        """

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                logger.debug(message_template.format(func.__name__))
                return func(*args, **kwargs)

            return cast(F, wrapper)

        return decorator

    @staticmethod
    def time_operation(
        operation_name: str, threshold_ms: float = 100.0
    ) -> Callable[..., Any]:
        """Decorator to profile expensive operations and log execution time.

        Automatically measures and logs execution time for expensive processor
        methods. Logs at debug level when below threshold, warning level when exceeded.

        Parameters
        ----------
        operation_name : str
            Human-readable name for the operation in log messages.
        threshold_ms : float, optional
            Warning threshold in milliseconds (default: 100.0). If execution
            time exceeds this, logs as warning instead of debug.

        Returns
        -------
        callable
            Decorator function that times the wrapped method.

        Examples
        --------
        >>> @ProcessorDecorators.time_operation("gradient computation", threshold_ms=50.0)
        ... def calculate(self, seismic_cube, facies_cube):
        ...     # Computation body
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                import time

                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                log_msg = f"{operation_name} completed in {elapsed_ms:.2f}ms"
                if elapsed_ms > threshold_ms:
                    logger.warning(log_msg)
                else:
                    logger.debug(log_msg)

                return result

            return wrapper

        return decorator

    @staticmethod
    def validate_cube_shape(expected_dims: int = 3) -> Callable[..., Any]:
        """Decorator to validate cube dimensionality before processing.

        Reduces boilerplate validation code by automatically checking cube
        dimensions at method entry. Logs and raises ValueError if invalid.

        Parameters
        ----------
        expected_dims : int, optional
            Expected number of dimensions (default: 3 for 3D cubes).

        Returns
        -------
        callable
            Decorator function that validates shape before method execution.

        Examples
        --------
        >>> @ProcessorDecorators.validate_cube_shape(expected_dims=3)
        ... def detect(self, facies_cube):
        ...     # facies_cube is guaranteed to be 3D
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(self: Any, cube: Any, *args: Any, **kwargs: Any) -> Any:
                if not isinstance(cube, np.ndarray):
                    raise TypeError(f"Expected ndarray, got {type(cube).__name__}")
                # Inform static type checkers that `cube` is an ndarray after the runtime check
                cube = cast(NDArray[Any], cube)
                if cube.ndim != expected_dims:
                    raise ValueError(
                        f"{func.__name__}() expects {expected_dims}D cube, "
                        f"got {cube.ndim}D with shape {cube.shape}"
                    )
                if cube.size == 0:
                    raise ValueError(f"Cube is empty (shape: {cube.shape})")
                logger.debug(f"Validated {expected_dims}D cube: shape={cube.shape}")
                return func(self, cube, *args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def cache_on_attributes(attributes: tuple[str, ...]) -> Callable[..., Any]:
        """Decorator to cache processor results based on specified attributes.

        Caches method results using attribute values as cache keys, improving
        performance for repeated calls with the same configuration.

        Parameters
        ----------
        attributes : tuple of str
            Names of attributes to use for cache key generation (e.g., ('threshold',))

        Returns
        -------
        callable
            Decorator function that caches method results.

        Examples
        --------
        >>> @ProcessorDecorators.cache_on_attributes(('dilation_window',))
        ... def extract(self, seismic_cube, boundaries):
        ...     # Method body
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            cache: dict[tuple[Any, ...], Any] = {}

            @wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                # Build cache key from specified attributes
                cache_attrs = tuple(getattr(self, attr, None) for attr in attributes)
                key = (cache_attrs, tuple(args), tuple(sorted(kwargs.items())))

                if key not in cache:
                    cache[key] = func(self, *args, **kwargs)
                    logger.debug(
                        "Cached result for %s with key %s", func.__name__, cache_attrs
                    )

                return cache[key]

            return wrapper

        return decorator
