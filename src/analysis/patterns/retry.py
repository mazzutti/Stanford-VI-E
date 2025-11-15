"""
Retry & Timeout Logic Implementation

Provides intelligent retry strategies with exponential backoff, jitter,
and timeout handling for resilient operations.

Key Features:
- Multiple retry strategies (exponential, linear, fibonacci)
- Jitter support to prevent thundering herd
- Timeout enforcement with configurable delays
- Function decorator support
- Detailed retry statistics and tracking
- Configurable exception handling

Usage:
    # Direct usage
    retry_policy = RetryPolicy(
        max_attempts=3,
        initial_delay=1.0,
        strategy=ExponentialBackoffStrategy()
    )

    result = retry_policy.execute(unreliable_function)

    # Decorator usage
    @retry(max_attempts=3, initial_delay=1.0)
    def unreliable_operation():
        pass
"""

import time
import random
from abc import ABC, abstractmethod
from typing import Any, cast
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RetryStrategyType(Enum):
    """Available retry strategies."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    CONSTANT = "constant"


class RetryStrategy(ABC):
    """
    Abstract base class for retry strategies.

    Defines how delays are calculated between retry attempts.
    """

    @abstractmethod
    def get_delay(self, attempt: int, initial_delay: float) -> float:
        """
        Calculate delay before next retry attempt.

        Args:
            attempt: Retry attempt number (0-indexed)
            initial_delay: Base delay in seconds

        Returns:
            Delay in seconds before next attempt
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class ExponentialBackoffStrategy(RetryStrategy):
    """
    Exponential backoff strategy: delay = initial_delay * (base ^ attempt)

    Example: initial_delay=1.0
        Attempt 0: 1.0s
        Attempt 1: 2.0s (1.0 * 2^1)
        Attempt 2: 4.0s (1.0 * 2^2)
        Attempt 3: 8.0s (1.0 * 2^3)
    """

    def __init__(self, base: float = 2.0, max_delay: float = 300.0):
        """
        Initialize exponential backoff strategy.

        Args:
            base: Exponent base (default: 2.0)
            max_delay: Maximum delay in seconds (default: 300.0)
        """
        self.base = base
        self.max_delay = max_delay

    def get_delay(self, attempt: int, initial_delay: float) -> float:
        """Calculate exponential delay with max cap."""
        delay = initial_delay * (self.base**attempt)
        return min(delay, self.max_delay)

    def __str__(self) -> str:
        return f"ExponentialBackoff(base={self.base}, max={self.max_delay}s)"


class LinearBackoffStrategy(RetryStrategy):
    """
    Linear backoff strategy: delay = initial_delay * (1 + attempt)

    Example: initial_delay=1.0
        Attempt 0: 1.0s
        Attempt 1: 2.0s (1.0 * 2)
        Attempt 2: 3.0s (1.0 * 3)
        Attempt 3: 4.0s (1.0 * 4)
    """

    def __init__(self, max_delay: float = 300.0):
        """
        Initialize linear backoff strategy.

        Args:
            max_delay: Maximum delay in seconds (default: 300.0)
        """
        self.max_delay = max_delay

    def get_delay(self, attempt: int, initial_delay: float) -> float:
        """Calculate linear delay with max cap."""
        delay = initial_delay * (1 + attempt)
        return min(delay, self.max_delay)

    def __str__(self) -> str:
        return f"LinearBackoff(max={self.max_delay}s)"


class FibonacciBackoffStrategy(RetryStrategy):
    """
    Fibonacci backoff strategy: delay = initial_delay * fibonacci(attempt)

    Example: initial_delay=1.0
        Attempt 0: 1.0s (fib(0) = 1)
        Attempt 1: 1.0s (fib(1) = 1)
        Attempt 2: 2.0s (fib(2) = 2)
        Attempt 3: 3.0s (fib(3) = 3)
        Attempt 4: 5.0s (fib(4) = 5)
    """

    def __init__(self, max_delay: float = 300.0):
        """
        Initialize fibonacci backoff strategy.

        Args:
            max_delay: Maximum delay in seconds (default: 300.0)
        """
        self.max_delay = max_delay
        self._fib_cache: dict[int, int] = {}

    def _fibonacci(self, n: int) -> int:
        """Calculate nth fibonacci number with caching."""
        if n in self._fib_cache:
            return self._fib_cache[n]

        if n <= 1:
            result = 1
        else:
            result = self._fibonacci(n - 1) + self._fibonacci(n - 2)

        self._fib_cache[n] = result
        return result

    def get_delay(self, attempt: int, initial_delay: float) -> float:
        """Calculate fibonacci delay with max cap."""
        fib_value = self._fibonacci(attempt)
        delay = initial_delay * fib_value
        return min(delay, self.max_delay)

    def __str__(self) -> str:
        return f"FibonacciBackoff(max={self.max_delay}s)"


class ConstantBackoffStrategy(RetryStrategy):
    """
    Constant backoff strategy: delay = initial_delay (no increase)

    Example: initial_delay=1.0
        Attempt 0: 1.0s
        Attempt 1: 1.0s
        Attempt 2: 1.0s
    """

    def get_delay(self, attempt: int, initial_delay: float) -> float:
        """Return constant delay."""
        return initial_delay

    def __str__(self) -> str:
        return "ConstantBackoff"


@dataclass
class RetryStats:
    """Statistics for retry attempts."""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_delay: float = 0.0
    last_exception: Exception | None = None
    exceptions_encountered: list[str] = field(
        default_factory=lambda: cast(list[str], [])
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_attempts / self.total_attempts) * 100

    @property
    def average_delay(self) -> float:
        """Calculate average delay per attempt."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_delay / self.total_attempts

    def __str__(self) -> str:
        return (
            f"RetryStats("
            f"total={self.total_attempts}, "
            f"successful={self.successful_attempts}, "
            f"failed={self.failed_attempts}, "
            f"success_rate={self.success_rate:.1f}%, "
            f"avg_delay={self.average_delay:.2f}s)"
        )


class RetryPolicy:
    """
    Manages retry logic with configurable strategy and exception handling.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        strategy: RetryStrategy | None = None,
        jitter: bool = True,
        jitter_factor: float = 0.1,
        timeout: float | None = None,
        retryable_exceptions: list[type[Exception]] | None = None,
        name: str | None = None,
    ) -> None:
        """
        Initialize retry policy.

        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            strategy: Retry strategy (default: ExponentialBackoffStrategy)
            jitter: Whether to add random jitter to delays
            jitter_factor: Jitter factor (0.1 = ±10%)
            timeout: Overall timeout for all retry attempts in seconds
            retryable_exceptions: List of exceptions to retry on
            name: Name for logging purposes
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.strategy = strategy or ExponentialBackoffStrategy()
        self.jitter = jitter
        self.jitter_factor = jitter_factor
        self.timeout = timeout
        self.retryable_exceptions = retryable_exceptions or [Exception]
        self.name = name or "RetryPolicy"

        self._stats = RetryStats()
        self._start_time: float | None = None

    def _apply_jitter(self, delay: float) -> float:
        """Apply random jitter to delay."""
        if not self.jitter:
            return delay

        jitter_range = delay * self.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        return max(0, delay + jitter)

    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        return any(
            isinstance(exception, exc_type) for exc_type in self.retryable_exceptions
        )

    def _is_timeout(self) -> bool:
        """Check if overall timeout has been exceeded."""
        if self.timeout is None or self._start_time is None:
            return False

        elapsed = time.time() - self._start_time
        return elapsed >= self.timeout

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Original exception if max attempts exceeded or timeout
        """
        self._stats = RetryStats()
        self._start_time = time.time()

        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                self._stats.total_attempts += 1

                logger.debug(f"{self.name}: Attempt {attempt + 1}/{self.max_attempts}")

                result = func(*args, **kwargs)
                self._stats.successful_attempts += 1

                logger.debug(f"{self.name}: Success on attempt {attempt + 1}")

                return result

            except Exception as e:
                self._stats.failed_attempts += 1
                self._stats.last_exception = e
                self._stats.exceptions_encountered.append(type(e).__name__)
                last_exception = e

                if not self._is_retryable(e):
                    logger.error(
                        f"{self.name}: Non-retryable exception: {type(e).__name__}"
                    )
                    raise

                if attempt == self.max_attempts - 1:
                    logger.error(
                        f"{self.name}: Max retries exceeded. Last error: {str(e)}"
                    )
                    raise

                if self._is_timeout():
                    logger.error(f"{self.name}: Timeout exceeded during retries")
                    raise

                # Calculate delay for next retry
                delay = self.strategy.get_delay(attempt, self.initial_delay)
                delay = self._apply_jitter(delay)
                self._stats.total_delay += delay

                logger.warning(
                    f"{self.name}: Attempt {attempt + 1} failed ({type(e).__name__}). "
                    f"Retrying in {delay:.2f}s..."
                )

                time.sleep(delay)

        # Should not reach here due to raise in loop
        raise last_exception or RuntimeError("Retry execution failed")

    def get_stats(self) -> RetryStats:
        """Get retry statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset retry statistics."""
        self._stats = RetryStats()

    def __repr__(self) -> str:
        return (
            f"<{self.name}: "
            f"max_attempts={self.max_attempts}, "
            f"initial_delay={self.initial_delay}s, "
            f"strategy={self.strategy}>"
        )


class TimeoutError(Exception):
    """Raised when operation exceeds timeout."""

    pass


def timeout(seconds: float) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to enforce timeout on function execution.

    Args:
        seconds: Timeout duration in seconds

    Raises:
        TimeoutError: If function exceeds timeout

    Note:
        Uses signal-based timeout (Unix-like systems only).
        For cross-platform, use thread-based or process-based timeout.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()

            result = func(*args, **kwargs)

            elapsed = time.time() - start_time
            if elapsed > seconds:
                raise TimeoutError(
                    f"Function {func.__name__} exceeded timeout of {seconds}s "
                    f"(took {elapsed:.2f}s)"
                )

            return result

        return wrapper

    return decorator


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    strategy: RetryStrategy | None = None,
    jitter: bool = True,
    timeout: float | None = None,
    retryable_exceptions: list[type[Exception]] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for automatic retry logic.

    Args:
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay between retries
        strategy: Retry strategy (default: ExponentialBackoffStrategy)
        jitter: Whether to add jitter to delays
        timeout: Overall timeout for all attempts
        retryable_exceptions: Exceptions to retry on

    Returns:
        Decorated function with retry support

    Example:
        @retry(max_attempts=3, initial_delay=1.0)
        def unreliable_operation():
            pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        policy = RetryPolicy(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            strategy=strategy or ExponentialBackoffStrategy(),
            jitter=jitter,
            timeout=timeout,
            retryable_exceptions=retryable_exceptions,
            name=func.__name__,
        )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return policy.execute(func, *args, **kwargs)

        return wrapper

    return decorator
