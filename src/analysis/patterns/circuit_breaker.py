"""
Circuit Breaker Pattern Implementation

Provides fault tolerance through circuit breaker state management.
Prevents cascading failures by detecting failure patterns and
temporarily blocking requests to failing services.

Key Features:
- Three states: CLOSED (normal), OPEN (failing), HALF_OPEN (recovery)
- Configurable failure thresholds and recovery delays
- Thread-safe state management
- Function decorator support
- Automatic recovery mechanisms
- Per-service circuit breaker management

Usage:
    # Direct usage
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    try:
        result = breaker.call(unreliable_function)
    except CircuitBreakerOpen:
        # Handle open circuit
        pass

    # Decorator usage
    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    def unreliable_api_call():
        pass
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from threading import RLock
from time import time
from typing import Any

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"  # Normal operation, allowing requests
    OPEN = "open"  # Failing state, blocking requests
    HALF_OPEN = "half_open"  # Recovery state, testing with limited requests


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is in OPEN state."""


class CircuitBreakerException(Exception):
    """Base exception for circuit breaker errors."""


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    state: CircuitBreakerState
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: float | None = None
    state_changed_at: float = field(default_factory=time)

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.failed_calls / self.total_calls) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    def __str__(self) -> str:
        time_in_state = time() - self.state_changed_at
        return (
            f"CircuitBreakerStats("
            f"state={self.state.value}, "
            f"total_calls={self.total_calls}, "
            f"success_rate={self.success_rate:.1f}%, "
            f"failure_rate={self.failure_rate:.1f}%, "
            f"rejected_calls={self.rejected_calls}, "
            f"time_in_state={time_in_state:.1f}s)"
        )


class CircuitBreaker:
    """
    Circuit Breaker implementation for fault tolerance.

    Manages state transitions and failure tracking.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type[BaseException] = Exception,
        name: str | None = None,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before trying recovery (HALF_OPEN state)
            expected_exception: Exception type to catch and count as failure
            name: Name of the circuit breaker for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "CircuitBreaker"

        self._lock = RLock()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._state_changed_at = time()
        self._total_calls = 0
        self._rejected_calls = 0

    @property
    def state(self) -> CircuitBreakerState:
        """Get current state of circuit breaker."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_recovery():
                    self._transition_to(CircuitBreakerState.HALF_OPEN)
                    logger.info("%s: Transitioning to HALF_OPEN state", self.name)

            return self._state

    def _should_attempt_recovery(self) -> bool:
        """Check if recovery timeout has elapsed."""
        if self._last_failure_time is None:
            return False
        return (time() - self._last_failure_time) >= self.recovery_timeout

    def _transition_to(self, new_state: CircuitBreakerState) -> None:
        """Transition to a new state."""
        if self._state != new_state:
            logger.info(
                "%s: State transition %s -> %s",
                self.name,
                self._state.value,
                new_state.value,
            )
            self._state = new_state
            self._state_changed_at = time()

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Original exception from function
        """
        with self._lock:
            self._total_calls += 1

            if self.state == CircuitBreakerState.OPEN:
                self._rejected_calls += 1
                logger.warning(
                    "%s: Circuit is OPEN, rejecting call (rejected: %s, total: %s)",
                    self.name,
                    self._rejected_calls,
                    self._total_calls,
                )
                raise CircuitBreakerOpen(f"Circuit breaker {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self._success_count += 1

            if self._state == CircuitBreakerState.HALF_OPEN:
                # Success during recovery, close the circuit
                self._failure_count = 0
                self._success_count = 0
                self._transition_to(CircuitBreakerState.CLOSED)
                logger.info(
                    "%s: Recovery successful, transitioning to CLOSED", self.name
                )
            elif self._state == CircuitBreakerState.CLOSED:
                # Normal operation, reset failure count
                self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time()

            logger.warning(
                "%s: Failure detected (count: %s/%s)",
                self.name,
                self._failure_count,
                self.failure_threshold,
            )

            if self._failure_count >= self.failure_threshold:
                if self._state != CircuitBreakerState.OPEN:
                    self._transition_to(CircuitBreakerState.OPEN)
                    logger.error(
                        "%s: Failure threshold reached, opening circuit", self.name
                    )
            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Failure during recovery, reopen the circuit
                self._transition_to(CircuitBreakerState.OPEN)
                logger.error(
                    "%s: Failure during recovery, reopening circuit", self.name
                )

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._transition_to(CircuitBreakerState.CLOSED)
            logger.info("%s: Circuit breaker manually reset", self.name)

    def get_stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        with self._lock:
            return CircuitBreakerStats(
                state=self._state,
                total_calls=self._total_calls,
                successful_calls=self._success_count,
                failed_calls=self._failure_count,
                rejected_calls=self._rejected_calls,
                last_failure_time=self._last_failure_time,
                state_changed_at=self._state_changed_at,
            )

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"<{self.name}: {stats}>"


class CircuitBreakerPool:
    """
    Manages multiple circuit breakers for different services.
    """

    def __init__(self) -> None:
        """Initialize circuit breaker pool."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = RLock()

    def get_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type[BaseException] = Exception,
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker.

        Args:
            name: Unique name for the circuit breaker
            failure_threshold: Failures before opening
            recovery_timeout: Recovery attempt delay
            expected_exception: Exception type to track

        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    expected_exception=expected_exception,
                    name=name,
                )
            return self._breakers[name]

    def get_all_breakers(self) -> dict[str, CircuitBreaker]:
        """Get all registered circuit breakers."""
        with self._lock:
            return dict(self._breakers)

    def reset_breaker(self, name: str) -> None:
        """Reset a specific circuit breaker."""
        with self._lock:
            if name in self._breakers:
                self._breakers[name].reset()

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_stats_all(self) -> dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_stats() for name, breaker in self._breakers.items()
            }

    def remove_breaker(self, name: str) -> None:
        """Remove a circuit breaker."""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]

    def __len__(self) -> int:
        """Get number of registered circuit breakers."""
        return len(self._breakers)


# Global pool instance
_global_pool = CircuitBreakerPool()


def circuit_breaker(
    name: str | None = None,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type[BaseException] = Exception,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for circuit breaker protection.

    Args:
        name: Name for the circuit breaker (defaults to function name)
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        expected_exception: Exception type to catch

    Returns:
        Decorated function with circuit breaker protection

    Example:
        @circuit_breaker(failure_threshold=5, recovery_timeout=60)
        def unreliable_api_call():
            pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        breaker = _global_pool.get_breaker(
            breaker_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
        )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


def get_circuit_breaker(name: str) -> CircuitBreaker | None:
    """Get a circuit breaker by name."""
    return _global_pool.get_breaker(name)


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Get all circuit breakers."""
    return _global_pool.get_all_breakers()


def reset_circuit_breaker(name: str) -> None:
    """Reset a circuit breaker."""
    _global_pool.reset_breaker(name)


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers."""
    _global_pool.reset_all()
