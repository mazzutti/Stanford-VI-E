"""Decorator Pattern support for cross-cutting concerns.

This module provides decorator implementations for common cross-cutting concerns
in analyzers and processors. Decorators offer an alternative to mixins for
adding behavior dynamically without inheritance chains.

Patterns Used:
  - Decorator: Wrap functions with additional behavior
  - Composition: Combine multiple decorators on single function
  
Decorators provided:
  - log_execution: Automatic logging of function entry/exit
  - time_operation: Track execution time and log warnings if threshold exceeded
  - validate_input: Validate input parameters before execution
  - memoize: Cache function results
  - retry: Automatically retry failed operations

Example:
    >>> @log_execution
    ... @time_operation(threshold_ms=100)
    ... @validate_input(lambda data: data is not None)
    ... def analyze(self, data: np.ndarray) -> dict:
    ...     return process(data)
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

__all__ = [
    "log_execution",
    "time_operation",
    "validate_input",
    "memoize",
    "retry",
]

F = TypeVar("F", bound=Callable[..., Any])


def log_execution(func: F) -> F:
    """Decorator for automatic execution logging.
    
    Logs function entry, successful completion, and any exceptions that occur
    during execution. Useful for debugging and tracing execution flow.
    
    Parameters
    ----------
    func : Callable
        Function to be decorated.
    
    Returns
    -------
    Callable
        Wrapped function with logging.
    
    Example
    -------
    >>> @log_execution
    ... def analyze(self, data):
    ...     return process(data)
    >>> analyzer.analyze(data)  # Logs entry and exit automatically
    """
    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger_inst = logging.getLogger(self.__class__.__module__)
        func_name = func.__qualname__
        logger_inst.info(f"Starting {func_name}")
        try:
            result = func(self, *args, **kwargs)
            logger_inst.debug(f"Completed {func_name}")
            return result
        except Exception as e:
            logger_inst.error(f"Failed {func_name}: {e}", exc_info=True)
            raise

    return wrapper  # type: ignore


def time_operation(
    label: str = "",
    threshold_ms: float = 100.0,
) -> Callable[[F], F]:
    """Decorator for operation timing with threshold warnings.
    
    Measures function execution time and logs a warning if execution time
    exceeds the specified threshold. Useful for identifying performance issues.
    
    Parameters
    ----------
    label : str, optional
        Operation label for logging (e.g., "boundary detection").
        If not provided, uses function name.
    threshold_ms : float, optional
        Time threshold in milliseconds for warning. Default: 100ms.
    
    Returns
    -------
    Callable
        Decorator function.
    
    Example
    -------
    >>> @time_operation("complex analysis", threshold_ms=500)
    ... def analyze(self, data):
    ...     return process(data)
    >>> analyzer.analyze(data)  # Logs warning if > 500ms
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            logger_inst = logging.getLogger(self.__class__.__module__)
            op_label = label or func.__qualname__
            
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > threshold_ms:
                    logger_inst.warning(
                        f"{op_label} took {elapsed_ms:.1f}ms (threshold: {threshold_ms}ms)"
                    )
                else:
                    logger_inst.debug(
                        f"{op_label} completed in {elapsed_ms:.1f}ms"
                    )

        return wrapper  # type: ignore

    return decorator


def validate_input(
    validator: Callable[[Any], bool],
    error_msg: str = "Input validation failed",
) -> Callable[[F], F]:
    """Decorator for input validation before execution.
    
    Calls validator function before executing the wrapped function.
    If validator returns False, raises ValueError without executing function.
    
    Parameters
    ----------
    validator : Callable
        Function that takes input and returns True if valid, False otherwise.
    error_msg : str, optional
        Error message if validation fails.
    
    Returns
    -------
    Callable
        Decorator function.
    
    Raises
    ------
    ValueError
        If validator returns False.
    
    Example
    -------
    >>> @validate_input(
    ...     lambda data: data is not None and len(data) > 0,
    ...     "Input data must be non-empty"
    ... )
    ... def analyze(self, data):
    ...     return process(data)
    >>> analyzer.analyze([])  # Raises ValueError
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Validate first argument (usually 'data')
            if args and not validator(args[0]):
                raise ValueError(error_msg)
            return func(self, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def memoize(func: F) -> F:
    """Decorator for function result caching.
    
    Caches function results based on input arguments. Subsequent calls with
    same arguments return cached result without re-execution. Useful for
    expensive computations that are called multiple times with same inputs.
    
    Note:
        Arguments must be hashable. Uses instance + arguments as cache key.
    
    Parameters
    ----------
    func : Callable
        Function to be decorated.
    
    Returns
    -------
    Callable
        Wrapped function with result caching.
    
    Example
    -------
    >>> @memoize
    ... def expensive_calculation(self, x):
    ...     return complex_math(x)
    >>> result1 = obj.expensive_calculation(5)  # Computed
    >>> result2 = obj.expensive_calculation(5)  # From cache
    """
    cache: Dict[int, Any] = {}

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Create cache key from instance id and arguments
        try:
            key = hash((id(self), args, tuple(sorted(kwargs.items()))))
        except TypeError:
            # Arguments not hashable, skip caching
            return func(self, *args, **kwargs)

        if key in cache:
            logger.debug(f"Cache hit for {func.__qualname__}")
            return cache[key]

        result = func(self, *args, **kwargs)
        cache[key] = result
        logger.debug(f"Cache miss for {func.__qualname__}, storing result")
        return result

    # Add cache management methods
    wrapper.cache_clear = cache.clear  # type: ignore
    wrapper.cache_info = lambda: f"Cache size: {len(cache)}"  # type: ignore

    return wrapper  # type: ignore


def retry(
    max_attempts: int = 3,
    delay_sec: float = 1.0,
    backoff_factor: float = 1.0,
    retryable_exceptions: Optional[tuple[type[Exception], ...]] = None,
) -> Callable[[F], F]:
    """Decorator for automatic retry on failure.
    
    Automatically retries function execution if it raises one of the specified
    exceptions. Supports exponential backoff with configurable delay multiplier.
    
    Parameters
    ----------
    max_attempts : int, optional
        Maximum number of attempts (including first). Default: 3.
    delay_sec : float, optional
        Initial delay between retries in seconds. Default: 1.0.
    backoff_factor : float, optional
        Multiplier for delay after each retry. Default: 1.0 (no backoff).
    retryable_exceptions : tuple[Exception], optional
        Tuple of exception types to catch. Default: (Exception,).
    
    Returns
    -------
    Callable
        Decorator function.
    
    Raises
    ------
    Exception
        Original exception after max_attempts exceeded.
    
    Example
    -------
    >>> @retry(max_attempts=5, delay_sec=0.5, backoff_factor=2.0)
    ... def unreliable_operation(self, data):
    ...     if random() > 0.8:
    ...         raise RuntimeError("Random failure")
    ...     return result
    """
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            logger_inst = logging.getLogger(self.__class__.__module__)
            current_delay = delay_sec
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(self, *args, **kwargs)
                    if attempt > 1:
                        logger_inst.info(
                            f"{func.__qualname__} succeeded on attempt {attempt}"
                        )
                    return result
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger_inst.warning(
                            f"{func.__qualname__} failed (attempt {attempt}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay:.2f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger_inst.error(
                            f"{func.__qualname__} failed after {max_attempts} attempts"
                        )

            # All retries exhausted, raise last exception
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__qualname__} failed unexpectedly")

        return wrapper  # type: ignore

    return decorator
