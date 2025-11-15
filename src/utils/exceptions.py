"""Exception handling utilities for streamlined error management.

This module provides decorators and context managers for consolidating
common exception handling patterns across the codebase, reducing boilerplate
and improving maintainability.
"""

from __future__ import annotations
from typing import Callable, TypeVar, Any, Optional, Type, Union, Tuple, cast, Generator
from functools import wraps
from contextlib import contextmanager
import logging

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
    exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = Exception,
    default: Any = None,
    log_errors: bool = False,
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
    except exceptions as e:
        if log_errors:
            logger.error(f"Error calling {func.__name__}: {e}", exc_info=True)
        return default


def ignore_errors(
    exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = Exception,
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
    if isinstance(exceptions, type):
        exceptions = (exceptions,)
    else:
        exceptions = tuple(exceptions)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions:
                pass
            return None

        return cast(F, wrapper)

    return decorator


def log_errors(
    exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = Exception,
    message: Optional[str] = None,
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
    if isinstance(exceptions, type):
        exceptions = (exceptions,)
    else:
        exceptions = tuple(exceptions)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions:
                log_msg = message or f"Error in {func.__name__}"
                logger.log(level, log_msg, exc_info=True)
                raise

        return cast(F, wrapper)

    return decorator


def handle_errors(
    exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = Exception,
    handler: Optional[Callable[[BaseException], Any]] = None,
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
            logger.warning(f"Handled: {e}")

        @handle_errors(ValueError, handler=on_error, suppress=True)
        def risky_operation():
            ...
    """
    if isinstance(exceptions, type):
        exceptions = (exceptions,)
    else:
        exceptions = tuple(exceptions)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if handler:
                    handler(e)
                if suppress:
                    return default
                raise

        return cast(F, wrapper)

    return decorator


@contextmanager
def safe_context(
    exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = Exception,
    action: Optional[Callable[[BaseException], None]] = None,
    suppress: bool = False,
) -> Generator[None, None, None]:
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
    if isinstance(exceptions, type):
        exceptions = (exceptions,)
    else:
        exceptions = tuple(exceptions)

    try:
        yield
    except exceptions as e:
        if action:
            action(e)
        if not suppress:
            raise
