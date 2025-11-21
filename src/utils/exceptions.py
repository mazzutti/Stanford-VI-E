"""Exception handling utilities for streamlined error management.

This module provides decorators and context managers for consolidating
common exception handling patterns across the codebase, reducing boilerplate
and improving maintainability.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar, cast

__all__ = [
    "safe_call",
    "ignore_errors",
    "log_errors",
    "handle_errors",
    "safe_context",
]

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

def safe_call(
    func: Callable[..., Any],
    *args: Any,
    exc_types: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    default: Any = None,
    log_exceptions: bool = False,
    **kwargs: Any,
) -> Any:
    """Safely call a function, returning default on exception.

    Args:
        func: Callable to invoke.
        *args: Positional arguments for func.
        exceptions: Exception type(s) to catch (default: Exception).
        default: Value to return if exception occurs (default: None).
        log_errors: Whether to log caught exceptions (default: False).
        **kwargs: Keyword arguments for func.

    Returns:
        Result of func() or default if exception occurs.
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        # These helpers intentionally start from catching broad exceptions
        # and then filter by `exc_types` at runtime. Narrowing here would
        # make the helpers less flexible for callers that provide custom
        # exception tuples.
        if not isinstance(exc, exc_types):
            raise
        if log_exceptions:
            logger.error("Error calling %s: %s", func.__name__, exc, exc_info=True)
        return default

def ignore_errors(
    exc_types: type[BaseException] | tuple[type[BaseException], ...] = Exception,
) -> Callable[[F], F]:
    """Decorator to silently ignore specified exceptions.

    Args:
        exceptions: Exception type(s) to ignore (default: Exception).

    Returns:
        Decorator function.

    Example:
        @ignore_errors(ValueError)
        def risky_operation():
            ...
    """
    if isinstance(exc_types, type):
        exc_types = (exc_types,)
    else:
        exc_types = tuple(exc_types)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if not isinstance(exc, exc_types):
                    raise
            return None

        return cast(F, wrapper)

    return decorator

def log_errors(
    exc_types: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    message: str | None = None,
    level: int = logging.ERROR,
) -> Callable[[F], F]:
    """Decorator to log exceptions without suppressing them.

    Args:
        exceptions: Exception type(s) to catch (default: Exception).
        message: Optional custom message to log.
        level: Logging level (default: logging.ERROR).

    Returns:
        Decorator function.

    Example:
        @log_errors(ValueError, message="Validation failed")
        def validate_data(data):
            ...
    """
    if isinstance(exc_types, type):
        exc_types = (exc_types,)
    else:
        exc_types = tuple(exc_types)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if not isinstance(exc, exc_types):
                    raise
                log_msg = message or "Error in %s"
                logger.log(level, log_msg, func.__name__, exc_info=True)
                raise

        return cast(F, wrapper)

    return decorator

def handle_errors(
    exc_types: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    handler: Callable[[BaseException], Any] | None = None,
    default: Any = None,
    suppress: bool = False,
) -> Callable[[F], F]:
    """Decorator to handle exceptions with custom handler.

    Args:
        exceptions: Exception type(s) to catch (default: Exception).
        handler: Optional callable to invoke with caught exception.
        default: Value to return if exception occurs.
        suppress: Whether to suppress exception after handling (default: False).

    Returns:
        Decorator function.

    Example:
        def on_error(e):
            logger.warning("Handled: %s", e)

        @handle_errors(ValueError, handler=on_error, suppress=True)
        def risky_operation():
            ...
    """
    if isinstance(exc_types, type):
        exc_types = (exc_types,)
    else:
        exc_types = tuple(exc_types)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if not isinstance(exc, exc_types):
                    raise
                if handler:
                    handler(exc)
                if suppress:
                    return default
                raise

        return cast(F, wrapper)

    return decorator

@contextmanager
def safe_context(
    exc_types: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    action: Callable[[BaseException], None] | None = None,
    suppress: bool = False,
) -> Generator[None]:
    """Context manager for exception handling.

    Args:
        exceptions: Exception type(s) to catch (default: Exception).
        action: Optional callable to invoke with caught exception.
        suppress: Whether to suppress exception (default: False).

    Yields:
        None

    Example:
        with safe_context(ValueError, suppress=True):
            risky_operation()
    """
    if isinstance(exc_types, type):
        exc_types = (exc_types,)
    else:
        exc_types = tuple(exc_types)

    try:
        yield
    except Exception as exc:
        if not isinstance(exc, exc_types):
            raise
        if action:
            action(exc)
        if not suppress:
            raise
