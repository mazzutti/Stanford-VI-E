"""
Rate Limiting Implementation

Provides multiple rate limiting algorithms for traffic control and
overload prevention in distributed systems.

Key Features:
- Token Bucket algorithm (smooth rate enforcement)
- Sliding Window Counter algorithm (precise rate limiting)
- Leaky Bucket algorithm (smooth output rate)
- Request queuing and scheduling
- Thread-safe operations with RLock
- Configurable burst sizes and rate limits
- Statistics tracking and monitoring

Usage:
    # Token bucket - allows bursts
    limiter = TokenBucketLimiter(capacity=100, refill_rate=10.0)
    if limiter.allow_request():
        process_request()

    # Sliding window - precise limits
    limiter = SlidingWindowLimiter(max_requests=100, window_size=60)
    if limiter.allow_request():
        process_request()

    # Leaky bucket - smooth rate
    limiter = LeakyBucketLimiter(capacity=100, leak_rate=10.0)
    if limiter.allow_request():
        process_request()
"""

import time
from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from collections import deque
from enum import Enum
from functools import wraps


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: float | None = None
    ) -> None:
        """
        Initialize rate limit exceeded exception.

        Args:
            message: Exception message
            retry_after: Seconds to wait before retry
        """
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitStats:
    """Statistics for rate limiter."""

    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    last_reset: float = field(default_factory=time.time)

    @property
    def rejection_rate(self) -> float:
        """Calculate rejection rate."""
        if self.total_requests == 0:
            return 0.0
        return self.rejected_requests / self.total_requests

    def reset(self) -> None:
        """Reset statistics."""
        self.total_requests = 0
        self.allowed_requests = 0
        self.rejected_requests = 0
        self.last_reset = time.time()


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    def allow_request(self, tokens: int = 1) -> bool:
        """
        Check if request is allowed.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if request allowed, False otherwise
        """
        pass

    @abstractmethod
    def get_stats(self) -> RateLimitStats:
        """Get limiter statistics."""
        pass


class TokenBucketLimiter(RateLimiter):
    """
    Token Bucket algorithm implementation.

    Allows bursts up to capacity while maintaining average rate.
    Good for: APIs that can handle occasional bursts.
    """

    def __init__(
        self, capacity: float, refill_rate: float, name: str = "token_bucket"
    ) -> None:
        """
        Initialize token bucket limiter.

        Args:
            capacity: Maximum tokens in bucket (burst size)
            refill_rate: Tokens per second to refill
            name: Name of the limiter
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("Refill rate must be positive")

        self.capacity = capacity
        self.refill_rate = refill_rate
        self.name = name
        self.tokens = capacity  # Start with full bucket
        self.last_refill = time.time()
        self._lock = RLock()
        self._stats = RateLimitStats()

    def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now

    def allow_request(self, tokens: int = 1) -> bool:
        """
        Check if tokens are available.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens available, False otherwise
        """
        if tokens <= 0:
            raise ValueError("Tokens must be positive")

        with self._lock:
            self._refill()
            self._stats.total_requests += 1

            if self.tokens >= tokens:
                self.tokens -= tokens
                self._stats.allowed_requests += 1
                return True
            else:
                self._stats.rejected_requests += 1
                return False

    def try_consume(self, tokens: int = 1) -> bool:
        """Alias for allow_request for consistency."""
        return self.allow_request(tokens)

    def get_stats(self) -> RateLimitStats:
        """Get statistics."""
        with self._lock:
            return RateLimitStats(
                total_requests=self._stats.total_requests,
                allowed_requests=self._stats.allowed_requests,
                rejected_requests=self._stats.rejected_requests,
                last_reset=self._stats.last_reset,
            )

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._stats.reset()

    def __repr__(self) -> str:
        return f"TokenBucketLimiter(name={self.name}, capacity={self.capacity}, refill_rate={self.refill_rate})"


class SlidingWindowLimiter(RateLimiter):
    """
    Sliding Window Counter algorithm implementation.

    Precisely limits requests within a rolling window.
    Good for: Strict rate limiting with exact limits.
    """

    def __init__(
        self, max_requests: int, window_size: float, name: str = "sliding_window"
    ) -> None:
        """
        Initialize sliding window limiter.

        Args:
            max_requests: Maximum requests in window
            window_size: Window duration in seconds
            name: Name of the limiter
        """
        if max_requests <= 0:
            raise ValueError("Max requests must be positive")
        if window_size <= 0:
            raise ValueError("Window size must be positive")

        self.max_requests = max_requests
        self.window_size = window_size
        self.name = name
        self.request_times: deque[float] = deque()
        self._lock = RLock()
        self._stats = RateLimitStats()

    def _cleanup_old_requests(self) -> None:
        """Remove requests outside the current window."""
        now = time.time()
        window_start = now - self.window_size

        while self.request_times and self.request_times[0] < window_start:
            self.request_times.popleft()

    def allow_request(self, tokens: int = 1) -> bool:
        """
        Check if requests are allowed within window.

        Args:
            tokens: Number of requests to check (treated as single request)

        Returns:
            True if requests fit within limit, False otherwise
        """
        if tokens <= 0:
            raise ValueError("Tokens must be positive")

        with self._lock:
            now = time.time()
            self._cleanup_old_requests()
            self._stats.total_requests += 1

            # Count tokens as single requests for sliding window
            requests_needed = tokens

            if len(self.request_times) + requests_needed <= self.max_requests:
                for _ in range(requests_needed):
                    self.request_times.append(now)
                self._stats.allowed_requests += 1
                return True
            else:
                self._stats.rejected_requests += 1
                return False

    def get_current_count(self) -> int:
        """Get current request count in window."""
        with self._lock:
            self._cleanup_old_requests()
            return len(self.request_times)

    def get_stats(self) -> RateLimitStats:
        """Get statistics."""
        with self._lock:
            return RateLimitStats(
                total_requests=self._stats.total_requests,
                allowed_requests=self._stats.allowed_requests,
                rejected_requests=self._stats.rejected_requests,
                last_reset=self._stats.last_reset,
            )

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._stats.reset()

    def __repr__(self) -> str:
        return (
            f"SlidingWindowLimiter(name={self.name}, "
            f"max_requests={self.max_requests}, window_size={self.window_size})"
        )


class LeakyBucketLimiter(RateLimiter):
    """
    Leaky Bucket algorithm implementation.

    Smooths out traffic by allowing fixed rate of output.
    Good for: Preventing traffic spikes while processing at consistent rate.
    """

    def __init__(
        self, capacity: float, leak_rate: float, name: str = "leaky_bucket"
    ) -> None:
        """
        Initialize leaky bucket limiter.

        Args:
            capacity: Bucket capacity (maximum queued requests)
            leak_rate: Requests per second to process
            name: Name of the limiter
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if leak_rate <= 0:
            raise ValueError("Leak rate must be positive")

        self.capacity = capacity
        self.leak_rate = leak_rate
        self.name = name
        self.water_level = 0.0  # Current water in bucket
        self.last_leak = time.time()
        self._lock = RLock()
        self._stats = RateLimitStats()

    def _leak(self) -> None:
        """Simulate water leaking from bucket."""
        now = time.time()
        elapsed = now - self.last_leak
        leak_amount = elapsed * self.leak_rate

        self.water_level = max(0.0, self.water_level - leak_amount)
        self.last_leak = now

    def allow_request(self, tokens: int = 1) -> bool:
        """
        Try to add request to bucket.

        Args:
            tokens: Number of tokens/requests to add

        Returns:
            True if request fits in bucket, False if bucket full
        """
        if tokens <= 0:
            raise ValueError("Tokens must be positive")

        with self._lock:
            self._leak()
            self._stats.total_requests += 1

            if self.water_level + tokens <= self.capacity:
                self.water_level += tokens
                self._stats.allowed_requests += 1
                return True
            else:
                self._stats.rejected_requests += 1
                return False

    def get_queue_size(self) -> float:
        """Get current queue size (water level)."""
        with self._lock:
            self._leak()
            return self.water_level

    def get_stats(self) -> RateLimitStats:
        """Get statistics."""
        with self._lock:
            return RateLimitStats(
                total_requests=self._stats.total_requests,
                allowed_requests=self._stats.allowed_requests,
                rejected_requests=self._stats.rejected_requests,
                last_reset=self._stats.last_reset,
            )

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._stats.reset()

    def __repr__(self) -> str:
        return f"LeakyBucketLimiter(name={self.name}, capacity={self.capacity}, leak_rate={self.leak_rate})"


class RateLimitPolicy:
    """
    Configuration policy for rate limiting.
    """

    def __init__(
        self,
        strategy: RateLimitStrategy,
        max_requests: int = 100,
        window_size: float = 60.0,
        capacity: float | None = None,
        rate: float | None = None,
    ) -> None:
        """
        Initialize rate limit policy.

        Args:
            strategy: Rate limiting strategy to use
            max_requests: Maximum requests (for sliding window)
            window_size: Window size in seconds (for sliding window)
            capacity: Bucket capacity (for token/leaky bucket)
            rate: Refill/leak rate per second (for token/leaky bucket)
        """
        self.strategy = strategy
        self.max_requests = max_requests
        self.window_size = window_size
        self.capacity = capacity
        self.rate = rate

    def create_limiter(self, name: str = "limiter") -> RateLimiter:
        """
        Create rate limiter based on policy.

        Args:
            name: Name of the limiter

        Returns:
            Configured RateLimiter instance
        """
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            if self.capacity is None or self.rate is None:
                raise ValueError("Token bucket requires capacity and rate")
            return TokenBucketLimiter(self.capacity, self.rate, name)

        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return SlidingWindowLimiter(self.max_requests, self.window_size, name)

        elif self.strategy == RateLimitStrategy.LEAKY_BUCKET:
            if self.capacity is None or self.rate is None:
                raise ValueError("Leaky bucket requires capacity and rate")
            return LeakyBucketLimiter(self.capacity, self.rate, name)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class RateLimitPool:
    """
    Pool of rate limiters for managing multiple named limiters.
    """

    def __init__(self) -> None:
        """Initialize rate limiter pool."""
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = RLock()

    def add_limiter(self, name: str, limiter: RateLimiter) -> None:
        """
        Add limiter to pool.

        Args:
            name: Unique limiter name
            limiter: RateLimiter instance
        """
        with self._lock:
            if name in self._limiters:
                raise ValueError(f"Limiter '{name}' already exists")
            self._limiters[name] = limiter

    def get_limiter(self, name: str) -> RateLimiter | None:
        """
        Get limiter by name.

        Args:
            name: Limiter name

        Returns:
            RateLimiter or None if not found
        """
        with self._lock:
            return self._limiters.get(name)

    def remove_limiter(self, name: str) -> bool:
        """
        Remove limiter from pool.

        Args:
            name: Limiter name

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._limiters:
                del self._limiters[name]
                return True
            return False

    def allow_request(self, name: str, tokens: int = 1) -> bool:
        """
        Check if request allowed using named limiter.

        Args:
            name: Limiter name
            tokens: Number of tokens

        Returns:
            True if allowed, False if rate limited
        """
        limiter = self.get_limiter(name)
        if limiter is None:
            raise ValueError(f"Limiter '{name}' not found")
        return limiter.allow_request(tokens)

    def get_stats(self, name: str) -> RateLimitStats | None:
        """
        Get stats for limiter.

        Args:
            name: Limiter name

        Returns:
            RateLimitStats or None if not found
        """
        limiter = self.get_limiter(name)
        if limiter is None:
            return None
        return limiter.get_stats()

    def get_all_stats(self) -> dict[str, RateLimitStats]:
        """Get stats for all limiters."""
        with self._lock:
            return {
                name: limiter.get_stats() for name, limiter in self._limiters.items()
            }


def rate_limit(
    limiter: RateLimiter, tokens: int = 1, raise_on_limit: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for rate limiting function execution.

    Args:
        limiter: RateLimiter instance
        tokens: Tokens to consume per call
        raise_on_limit: Raise exception if rate limited

    Returns:
        Decorated function

    Example:
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)

        @rate_limit(limiter)
        def api_handler():
            pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if limiter.allow_request(tokens):
                return func(*args, **kwargs)
            elif raise_on_limit:
                raise RateLimitExceeded(f"Rate limit exceeded for {func.__name__}")
            else:
                return None

        return wrapper

    return decorator
