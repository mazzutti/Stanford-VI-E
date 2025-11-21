"""Small utility decorators shared across packages.

Move common cross-cutting decorators here so light-weight modules can depend
on them without importing heavy analysis/processing packages.

Only includes behavior-preserving implementations used in both analysis and
processing code: `log_execution`, `time_operation`, `validate_input`,
`memoize`, and `retry`.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

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
    """Decorator that logs start and completion of method execution."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Wrapper that logs execution start/completion and forwards the call."""
        logger_inst = logging.getLogger(self.__class__.__module__)
        func_name = func.__qualname__
        logger_inst.info("Starting %s", func_name)
        try:
            result = func(self, *args, **kwargs)
            logger_inst.debug("Completed %s", func_name)
            return result
        except (RuntimeError, ValueError, TypeError) as e:
            logger_inst.error("Failed %s: %s", func_name, e, exc_info=True)
            raise

    return cast(F, wrapper)

# Decorators here are intentionally lightweight and dependency-free so they
# can be used in performance-sensitive modules without pulling heavy deps.

def time_operation(
    label: str = "",
    threshold_ms: float = 100.0,
) -> Callable[[F], F]:
    """Decorator factory that logs slow operations exceeding `threshold_ms`."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Wrapper that times the operation and logs slow executions."""
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
                        "%s took %.1fms (threshold: %sms)",
                        op_label,
                        elapsed_ms,
                        threshold_ms,
                    )
                else:
                    logger_inst.debug("%s completed in %.1fms", op_label, elapsed_ms)

        return cast(F, wrapper)

    return decorator

def validate_input(
    validator: Callable[[Any], bool],
    error_msg: str = "Input validation failed",
) -> Callable[[F], F]:
    """Decorator factory that validates the first argument using `validator`."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Wrapper validating the first argument using provided validator."""
            if args and not validator(args[0]):
                raise ValueError(error_msg)
            return func(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator

def memoize(func: F) -> F:
    """Simple per-process memoize decorator for instance methods."""
    cache: dict[int, Any] = {}

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Memoize wrapper that caches results per-instance."""
        try:
            key = hash((id(self), args, tuple(sorted(kwargs.items()))))
        except TypeError:
            return func(self, *args, **kwargs)

        if key in cache:
            logger.debug("Cache hit for %s", func.__qualname__)
            return cache[key]

        result = func(self, *args, **kwargs)
        cache[key] = result
        logger.debug("Cache miss for %s, storing result", func.__qualname__)
        return result

    wrapper_any = cast(Any, wrapper)
    wrapper_any.cache_clear = cache.clear
    wrapper_any.cache_info = lambda: f"Cache size: {len(cache)}"

    return cast(F, wrapper)

def retry(
    max_attempts: int = 3,
    delay_sec: float = 1.0,
    backoff_factor: float = 1.0,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
) -> Callable[[F], F]:
    """Retry decorator factory with optional backoff for retryable exceptions."""
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Retry wrapper that retries the decorated function on failure."""
            logger_inst = logging.getLogger(self.__class__.__module__)
            current_delay = delay_sec
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(self, *args, **kwargs)
                    if attempt > 1:
                        logger_inst.info(
                            "%s succeeded on attempt %d",
                            func.__qualname__,
                            attempt,
                        )
                    return result
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger_inst.warning(
                            "%s failed (attempt %d/%d): %s. Retrying in %.2fs...",
                            func.__qualname__,
                            attempt,
                            max_attempts,
                            e,
                            current_delay,
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger_inst.error(
                            "%s failed after %d attempts",
                            func.__qualname__,
                            max_attempts,
                        )

            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__qualname__} failed unexpectedly")

        return cast(F, wrapper)

    return decorator
