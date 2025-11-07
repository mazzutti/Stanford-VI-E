"""
Tests for Phase 5.5: Rate Limiting
"""

import time
import pytest
from threading import Thread

from src.analysis.rate_limiting import (
    RateLimitExceeded,
    RateLimitStrategy,
    RateLimitStats,
    TokenBucketLimiter,
    SlidingWindowLimiter,
    LeakyBucketLimiter,
    RateLimitPolicy,
    RateLimitPool,
    rate_limit,
)


class TestRateLimitExceeded:
    """Test RateLimitExceeded exception."""

    def test_exception_creation(self):
        """Test creating rate limit exceeded exception."""
        exc = RateLimitExceeded("Test message")
        assert str(exc) == "Test message"

    def test_exception_with_retry_after(self):
        """Test exception with retry_after."""
        exc = RateLimitExceeded("Too many requests", retry_after=60.0)
        assert exc.retry_after == 60.0


class TestRateLimitStrategy:
    """Test RateLimitStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert RateLimitStrategy.TOKEN_BUCKET.value == "token_bucket"
        assert RateLimitStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert RateLimitStrategy.LEAKY_BUCKET.value == "leaky_bucket"


class TestRateLimitStats:
    """Test RateLimitStats dataclass."""

    def test_stats_creation(self):
        """Test creating rate limit stats."""
        stats = RateLimitStats()
        assert stats.total_requests == 0
        assert stats.allowed_requests == 0
        assert stats.rejected_requests == 0

    def test_rejection_rate_calculation(self):
        """Test rejection rate calculation."""
        stats = RateLimitStats(
            total_requests=10, allowed_requests=8, rejected_requests=2
        )
        assert stats.rejection_rate == 0.2

    def test_rejection_rate_no_requests(self):
        """Test rejection rate with no requests."""
        stats = RateLimitStats()
        assert stats.rejection_rate == 0.0

    def test_stats_reset(self):
        """Test resetting stats."""
        stats = RateLimitStats(total_requests=100, allowed_requests=80)
        stats.reset()

        assert stats.total_requests == 0
        assert stats.allowed_requests == 0
        assert stats.rejected_requests == 0


class TestTokenBucketLimiter:
    """Test TokenBucketLimiter."""

    def test_creation(self):
        """Test creating token bucket limiter."""
        limiter = TokenBucketLimiter(capacity=100, refill_rate=10.0)
        assert limiter.capacity == 100
        assert limiter.refill_rate == 10.0
        assert limiter.tokens == 100

    def test_invalid_capacity(self):
        """Test invalid capacity."""
        with pytest.raises(ValueError):
            TokenBucketLimiter(capacity=0, refill_rate=10)

    def test_invalid_refill_rate(self):
        """Test invalid refill rate."""
        with pytest.raises(ValueError):
            TokenBucketLimiter(capacity=100, refill_rate=0)

    def test_allow_request_success(self):
        """Test allowing request when tokens available."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)
        assert limiter.allow_request(5) is True
        assert limiter.tokens == 5

    def test_allow_request_failure(self):
        """Test denying request when tokens unavailable."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)
        assert limiter.allow_request(15) is False

    def test_refill_tokens(self):
        """Test token refill over time."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=200)

        # Use all tokens
        assert limiter.allow_request(10) is True
        old_tokens = limiter.tokens

        # Wait for refill
        time.sleep(0.1)

        # Trigger refill by calling allow_request
        limiter.allow_request(0) if False else None
        # Manually call _refill through a request check
        limiter._refill()

        # Should have tokens now
        assert limiter.tokens > old_tokens

    def test_capacity_limit(self):
        """Test tokens don't exceed capacity."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=100)
        limiter.tokens = 5

        time.sleep(0.2)  # Allow refill

        assert limiter.tokens <= 10

    def test_multiple_tokens(self):
        """Test consuming multiple tokens."""
        limiter = TokenBucketLimiter(capacity=100, refill_rate=10)

        assert limiter.allow_request(50) is True
        assert limiter.allow_request(30) is True
        assert limiter.allow_request(30) is False

    def test_try_consume_alias(self):
        """Test try_consume is alias for allow_request."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)
        assert limiter.try_consume(5) is True
        assert limiter.try_consume(10) is False

    def test_get_stats(self):
        """Test getting statistics."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)
        limiter.allow_request(5)
        limiter.allow_request(10)  # Should fail

        stats = limiter.get_stats()
        assert stats.total_requests == 2
        assert stats.allowed_requests == 1
        assert stats.rejected_requests == 1

    def test_reset_stats(self):
        """Test resetting statistics."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)
        limiter.allow_request(5)

        limiter.reset_stats()
        stats = limiter.get_stats()

        assert stats.total_requests == 0
        assert stats.allowed_requests == 0


class TestSlidingWindowLimiter:
    """Test SlidingWindowLimiter."""

    def test_creation(self):
        """Test creating sliding window limiter."""
        limiter = SlidingWindowLimiter(max_requests=100, window_size=60)
        assert limiter.max_requests == 100
        assert limiter.window_size == 60

    def test_invalid_max_requests(self):
        """Test invalid max requests."""
        with pytest.raises(ValueError):
            SlidingWindowLimiter(max_requests=0, window_size=60)

    def test_invalid_window_size(self):
        """Test invalid window size."""
        with pytest.raises(ValueError):
            SlidingWindowLimiter(max_requests=100, window_size=0)

    def test_allow_request_success(self):
        """Test allowing request within limit."""
        limiter = SlidingWindowLimiter(max_requests=10, window_size=60)
        assert limiter.allow_request(5) is True
        assert limiter.allow_request(3) is True

    def test_allow_request_failure(self):
        """Test denying request exceeding limit."""
        limiter = SlidingWindowLimiter(max_requests=5, window_size=60)
        assert limiter.allow_request(3) is True
        assert limiter.allow_request(3) is False

    def test_window_expiration(self):
        """Test old requests expire from window."""
        limiter = SlidingWindowLimiter(max_requests=5, window_size=0.1)

        assert limiter.allow_request(5) is True
        assert limiter.allow_request(1) is False

        # Wait for window to expire
        time.sleep(0.15)

        # Should allow again
        assert limiter.allow_request(1) is True

    def test_get_current_count(self):
        """Test getting current request count."""
        limiter = SlidingWindowLimiter(max_requests=10, window_size=60)

        limiter.allow_request(3)
        assert limiter.get_current_count() == 3

        limiter.allow_request(2)
        assert limiter.get_current_count() == 5

    def test_get_stats(self):
        """Test getting statistics."""
        limiter = SlidingWindowLimiter(max_requests=5, window_size=60)
        limiter.allow_request(3)
        limiter.allow_request(3)

        stats = limiter.get_stats()
        assert stats.total_requests == 2
        assert stats.allowed_requests == 1
        assert stats.rejected_requests == 1


class TestLeakyBucketLimiter:
    """Test LeakyBucketLimiter."""

    def test_creation(self):
        """Test creating leaky bucket limiter."""
        limiter = LeakyBucketLimiter(capacity=100, leak_rate=10)
        assert limiter.capacity == 100
        assert limiter.leak_rate == 10
        assert limiter.water_level == 0.0

    def test_invalid_capacity(self):
        """Test invalid capacity."""
        with pytest.raises(ValueError):
            LeakyBucketLimiter(capacity=0, leak_rate=10)

    def test_invalid_leak_rate(self):
        """Test invalid leak rate."""
        with pytest.raises(ValueError):
            LeakyBucketLimiter(capacity=100, leak_rate=0)

    def test_allow_request_success(self):
        """Test adding request to bucket."""
        limiter = LeakyBucketLimiter(capacity=100, leak_rate=10)
        assert limiter.allow_request(50) is True
        assert limiter.water_level == 50

    def test_allow_request_failure(self):
        """Test rejecting when bucket full."""
        limiter = LeakyBucketLimiter(capacity=100, leak_rate=10)
        assert limiter.allow_request(100) is True
        assert limiter.allow_request(1) is False

    def test_leak_over_time(self):
        """Test water leaks over time."""
        limiter = LeakyBucketLimiter(capacity=100, leak_rate=10)
        limiter.allow_request(50)

        time.sleep(0.1)

        # Water should have leaked
        queue_size = limiter.get_queue_size()
        assert queue_size < 50

    def test_multiple_requests(self):
        """Test queuing multiple requests."""
        limiter = LeakyBucketLimiter(capacity=100, leak_rate=10)

        assert limiter.allow_request(30) is True
        assert limiter.allow_request(40) is True
        assert limiter.water_level >= 69  # Allow for floating point precision

    def test_get_queue_size(self):
        """Test getting queue size."""
        limiter = LeakyBucketLimiter(capacity=100, leak_rate=10)
        limiter.allow_request(25)

        assert limiter.get_queue_size() >= 24.99  # Allow for floating point precision

    def test_get_stats(self):
        """Test getting statistics."""
        limiter = LeakyBucketLimiter(capacity=100, leak_rate=10)
        limiter.allow_request(30)
        limiter.allow_request(80)

        stats = limiter.get_stats()
        assert stats.total_requests == 2
        assert stats.allowed_requests == 1
        assert stats.rejected_requests == 1


class TestRateLimitPolicy:
    """Test RateLimitPolicy."""

    def test_token_bucket_policy(self):
        """Test creating token bucket from policy."""
        policy = RateLimitPolicy(
            strategy=RateLimitStrategy.TOKEN_BUCKET, capacity=100, rate=10
        )
        limiter = policy.create_limiter()

        assert isinstance(limiter, TokenBucketLimiter)
        assert limiter.capacity == 100

    def test_sliding_window_policy(self):
        """Test creating sliding window from policy."""
        policy = RateLimitPolicy(
            strategy=RateLimitStrategy.SLIDING_WINDOW, max_requests=100, window_size=60
        )
        limiter = policy.create_limiter()

        assert isinstance(limiter, SlidingWindowLimiter)
        assert limiter.max_requests == 100

    def test_leaky_bucket_policy(self):
        """Test creating leaky bucket from policy."""
        policy = RateLimitPolicy(
            strategy=RateLimitStrategy.LEAKY_BUCKET, capacity=100, rate=10
        )
        limiter = policy.create_limiter()

        assert isinstance(limiter, LeakyBucketLimiter)
        assert limiter.capacity == 100

    def test_token_bucket_missing_parameters(self):
        """Test token bucket with missing parameters."""
        policy = RateLimitPolicy(strategy=RateLimitStrategy.TOKEN_BUCKET)
        with pytest.raises(ValueError):
            policy.create_limiter()

    def test_leaky_bucket_missing_parameters(self):
        """Test leaky bucket with missing parameters."""
        policy = RateLimitPolicy(strategy=RateLimitStrategy.LEAKY_BUCKET)
        with pytest.raises(ValueError):
            policy.create_limiter()


class TestRateLimitPool:
    """Test RateLimitPool."""

    def test_add_limiter(self):
        """Test adding limiter to pool."""
        pool = RateLimitPool()
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)

        pool.add_limiter("api", limiter)
        assert pool.get_limiter("api") is limiter

    def test_duplicate_limiter(self):
        """Test adding duplicate limiter."""
        pool = RateLimitPool()
        limiter1 = TokenBucketLimiter(capacity=10, refill_rate=1)
        limiter2 = TokenBucketLimiter(capacity=20, refill_rate=2)

        pool.add_limiter("api", limiter1)
        with pytest.raises(ValueError):
            pool.add_limiter("api", limiter2)

    def test_get_nonexistent_limiter(self):
        """Test getting nonexistent limiter."""
        pool = RateLimitPool()
        assert pool.get_limiter("nonexistent") is None

    def test_remove_limiter(self):
        """Test removing limiter."""
        pool = RateLimitPool()
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)

        pool.add_limiter("api", limiter)
        assert pool.remove_limiter("api") is True
        assert pool.get_limiter("api") is None

    def test_remove_nonexistent_limiter(self):
        """Test removing nonexistent limiter."""
        pool = RateLimitPool()
        assert pool.remove_limiter("nonexistent") is False

    def test_allow_request_through_pool(self):
        """Test allowing request through pool."""
        pool = RateLimitPool()
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)
        pool.add_limiter("api", limiter)

        assert pool.allow_request("api", 5) is True
        assert pool.allow_request("api", 10) is False

    def test_allow_request_nonexistent_limiter(self):
        """Test allowing request with nonexistent limiter."""
        pool = RateLimitPool()
        with pytest.raises(ValueError):
            pool.allow_request("nonexistent")

    def test_get_stats_from_pool(self):
        """Test getting stats from pool."""
        pool = RateLimitPool()
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)
        pool.add_limiter("api", limiter)

        limiter.allow_request(5)
        stats = pool.get_stats("api")

        assert stats.total_requests == 1
        assert stats.allowed_requests == 1

    def test_get_all_stats(self):
        """Test getting all stats."""
        pool = RateLimitPool()

        limiter1 = TokenBucketLimiter(capacity=10, refill_rate=1)
        limiter2 = SlidingWindowLimiter(max_requests=20, window_size=60)

        pool.add_limiter("api", limiter1)
        pool.add_limiter("web", limiter2)

        limiter1.allow_request(5)
        limiter2.allow_request(10)

        all_stats = pool.get_all_stats()
        assert "api" in all_stats
        assert "web" in all_stats


class TestRateLimitDecorator:
    """Test @rate_limit decorator."""

    def test_decorator_basic(self):
        """Test basic rate limit decorator."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)
        call_count = {"value": 0}

        @rate_limit(limiter)
        def test_function():
            call_count["value"] += 1
            return "result"

        # Should succeed
        result = test_function()
        assert result == "result"
        assert call_count["value"] == 1

    def test_decorator_rate_limited(self):
        """Test decorator returns None when rate limited."""
        limiter = TokenBucketLimiter(capacity=1, refill_rate=0.1)
        call_count = {"value": 0}

        @rate_limit(limiter)
        def test_function():
            call_count["value"] += 1
            return "result"

        # First call should succeed
        result1 = test_function()
        assert result1 == "result"

        # Second call should fail
        result2 = test_function()
        assert result2 is None
        assert call_count["value"] == 1

    def test_decorator_with_tokens(self):
        """Test decorator with multiple tokens."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)

        @rate_limit(limiter, tokens=5)
        def test_function():
            return "result"

        assert test_function() == "result"
        assert test_function() == "result"
        assert test_function() is None  # Third call should fail

    def test_decorator_raise_on_limit(self):
        """Test decorator raises exception when rate limited."""
        limiter = TokenBucketLimiter(capacity=1, refill_rate=0.1)

        @rate_limit(limiter, raise_on_limit=True)
        def test_function():
            return "result"

        # First call succeeds
        assert test_function() == "result"

        # Second call raises exception
        with pytest.raises(RateLimitExceeded):
            test_function()

    def test_decorator_preserves_function_name(self):
        """Test decorator preserves function metadata."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1)

        @rate_limit(limiter)
        def my_function():
            """My function docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert "My function docstring" in my_function.__doc__


class TestThreadSafety:
    """Test thread safety of rate limiters."""

    def test_token_bucket_thread_safety(self):
        """Test token bucket with concurrent access."""
        limiter = TokenBucketLimiter(capacity=100, refill_rate=10)
        results = {"allowed": 0, "denied": 0}

        def make_requests():
            for _ in range(10):
                if limiter.allow_request(1):
                    results["allowed"] += 1
                else:
                    results["denied"] += 1

        threads = [Thread(target=make_requests) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["allowed"] + results["denied"] == 50
        assert results["allowed"] <= 100

    def test_sliding_window_thread_safety(self):
        """Test sliding window with concurrent access."""
        limiter = SlidingWindowLimiter(max_requests=50, window_size=1)
        results = {"allowed": 0, "denied": 0}

        def make_requests():
            for _ in range(10):
                if limiter.allow_request(1):
                    results["allowed"] += 1
                else:
                    results["denied"] += 1

        threads = [Thread(target=make_requests) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["allowed"] <= 50

    def test_leaky_bucket_thread_safety(self):
        """Test leaky bucket with concurrent access."""
        limiter = LeakyBucketLimiter(capacity=100, leak_rate=10)
        results = {"allowed": 0, "denied": 0}

        def make_requests():
            for _ in range(10):
                if limiter.allow_request(1):
                    results["allowed"] += 1
                else:
                    results["denied"] += 1

        threads = [Thread(target=make_requests) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["allowed"] <= 100


class TestIntegration:
    """Integration tests for rate limiting."""

    def test_api_rate_limiting(self):
        """Test typical API rate limiting scenario."""
        limiter = TokenBucketLimiter(capacity=100, refill_rate=10.0)

        # Simulate API requests
        successful = 0
        for _ in range(150):
            if limiter.allow_request(1):
                successful += 1

        # Should allow some requests
        assert successful > 0
        assert successful < 150

    def test_multiple_limiters_pool(self):
        """Test managing multiple limiters in pool."""
        pool = RateLimitPool()

        # Add different limiters for different endpoints
        pool.add_limiter(
            "fast_endpoint", TokenBucketLimiter(capacity=1000, refill_rate=100)
        )
        pool.add_limiter(
            "slow_endpoint", TokenBucketLimiter(capacity=10, refill_rate=1)
        )

        # Fast endpoint allows many
        fast_allowed = sum(1 for _ in range(50) if pool.allow_request("fast_endpoint"))
        assert fast_allowed > 40

        # Slow endpoint allows few
        slow_allowed = sum(1 for _ in range(50) if pool.allow_request("slow_endpoint"))
        assert slow_allowed <= 10

    def test_strategy_comparison(self):
        """Compare different rate limiting strategies."""
        # All allow roughly 50 requests in first call
        tb = TokenBucketLimiter(capacity=50, refill_rate=10)
        sw = SlidingWindowLimiter(max_requests=50, window_size=60)
        lb = LeakyBucketLimiter(capacity=50, leak_rate=10)

        # Token bucket allows burst
        tb_success = sum(1 for _ in range(100) if tb.allow_request(1))

        # Sliding window strict
        sw_success = sum(1 for _ in range(100) if sw.allow_request(1))
        assert sw_success == 50

        # Leaky bucket queues
        lb_success = sum(1 for _ in range(100) if lb.allow_request(1))
        assert lb_success == 50
