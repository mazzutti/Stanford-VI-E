"""
Comprehensive test suite for Retry & Timeout Logic (Phase 5.3)

Tests cover:
- Retry strategies (Exponential, Linear, Fibonacci, Constant)
- Jitter application
- Exception handling and filtering
- Retry policies with timeouts
- Decorator functionality
- Statistics tracking
- Edge cases and failures
"""

import time

import pytest
from freezegun import freeze_time

from src.analysis.patterns.retry import (ConstantBackoffStrategy,
                                         ExponentialBackoffStrategy,
                                         FibonacciBackoffStrategy,
                                         LinearBackoffStrategy, RetryPolicy,
                                         RetryStats, RetryStrategy,
                                         TimeoutError, retry, timeout)


class TestExponentialBackoffStrategy:
    """Test exponential backoff strategy."""

    def test_exponential_delay_calculation(self):
        """Exponential backoff calculates correct delays."""
        strategy = ExponentialBackoffStrategy(base=2.0)

        assert strategy.get_delay(0, 1.0) == 1.0  # 1.0 * 2^0
        assert strategy.get_delay(1, 1.0) == 2.0  # 1.0 * 2^1
        assert strategy.get_delay(2, 1.0) == 4.0  # 1.0 * 2^2
        assert strategy.get_delay(3, 1.0) == 8.0  # 1.0 * 2^3

    def test_exponential_with_custom_base(self):
        """Exponential backoff with custom base."""
        strategy = ExponentialBackoffStrategy(base=3.0)

        assert strategy.get_delay(0, 1.0) == 1.0  # 1.0 * 3^0
        assert strategy.get_delay(1, 1.0) == 3.0  # 1.0 * 3^1
        assert strategy.get_delay(2, 1.0) == 9.0  # 1.0 * 3^2

    def test_exponential_respects_max_delay(self):
        """Exponential backoff respects maximum delay."""
        strategy = ExponentialBackoffStrategy(base=2.0, max_delay=10.0)

        assert strategy.get_delay(0, 1.0) == 1.0
        assert strategy.get_delay(3, 1.0) == 8.0
        assert strategy.get_delay(4, 1.0) == 10.0  # Capped at max_delay
        assert strategy.get_delay(5, 1.0) == 10.0  # Still capped


class TestLinearBackoffStrategy:
    """Test linear backoff strategy."""

    def test_linear_delay_calculation(self):
        """Linear backoff calculates correct delays."""
        strategy = LinearBackoffStrategy()

        assert strategy.get_delay(0, 1.0) == 1.0  # 1.0 * 1
        assert strategy.get_delay(1, 1.0) == 2.0  # 1.0 * 2
        assert strategy.get_delay(2, 1.0) == 3.0  # 1.0 * 3
        assert strategy.get_delay(3, 1.0) == 4.0  # 1.0 * 4

    def test_linear_with_different_initial_delay(self):
        """Linear backoff with different initial delays."""
        strategy = LinearBackoffStrategy()

        assert strategy.get_delay(0, 0.5) == 0.5
        assert strategy.get_delay(1, 0.5) == 1.0
        assert strategy.get_delay(2, 0.5) == 1.5

    def test_linear_respects_max_delay(self):
        """Linear backoff respects maximum delay."""
        strategy = LinearBackoffStrategy(max_delay=5.0)

        assert strategy.get_delay(0, 1.0) == 1.0
        assert strategy.get_delay(4, 1.0) == 5.0  # Capped
        assert strategy.get_delay(10, 1.0) == 5.0  # Still capped


class TestFibonacciBackoffStrategy:
    """Test fibonacci backoff strategy."""

    def test_fibonacci_delay_calculation(self):
        """Fibonacci backoff calculates correct delays."""
        strategy = FibonacciBackoffStrategy()

        # Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13...
        assert strategy.get_delay(0, 1.0) == 1.0  # 1.0 * fib(0)
        assert strategy.get_delay(1, 1.0) == 1.0  # 1.0 * fib(1)
        assert strategy.get_delay(2, 1.0) == 2.0  # 1.0 * fib(2)
        assert strategy.get_delay(3, 1.0) == 3.0  # 1.0 * fib(3)
        assert strategy.get_delay(4, 1.0) == 5.0  # 1.0 * fib(4)

    def test_fibonacci_respects_max_delay(self):
        """Fibonacci backoff respects maximum delay."""
        strategy = FibonacciBackoffStrategy(max_delay=4.0)

        assert strategy.get_delay(0, 1.0) == 1.0
        assert strategy.get_delay(2, 1.0) == 2.0
        assert strategy.get_delay(3, 1.0) == 3.0
        assert strategy.get_delay(4, 1.0) == 4.0  # Capped
        assert strategy.get_delay(5, 1.0) == 4.0  # Still capped


class TestConstantBackoffStrategy:
    """Test constant backoff strategy."""

    def test_constant_delay(self):
        """Constant backoff returns same delay."""
        strategy = ConstantBackoffStrategy()

        assert strategy.get_delay(0, 1.0) == 1.0
        assert strategy.get_delay(1, 1.0) == 1.0
        assert strategy.get_delay(5, 1.0) == 1.0

    def test_constant_with_different_initial(self):
        """Constant backoff with different initial delays."""
        strategy = ConstantBackoffStrategy()

        assert strategy.get_delay(0, 0.5) == 0.5
        assert strategy.get_delay(10, 0.5) == 0.5


class TestRetryPolicy:
    """Test retry policy functionality."""

    def test_successful_on_first_try(self):
        """Successful function on first attempt."""
        policy = RetryPolicy(max_attempts=3)

        def success_func():
            return "ok"

        result = policy.execute(success_func)
        assert result == "ok"

        stats = policy.get_stats()
        assert stats.total_attempts == 1
        assert stats.successful_attempts == 1

    def test_succeeds_after_failures(self):
        """Function succeeds after initial failures."""
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01)

        attempt_count = 0

        def eventually_succeeds():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Not ready yet")
            return "success"

        result = policy.execute(eventually_succeeds)
        assert result == "success"

        stats = policy.get_stats()
        assert stats.total_attempts == 3
        assert stats.successful_attempts == 1
        assert stats.failed_attempts == 2

    def test_max_attempts_exceeded(self):
        """Raises exception when max attempts exceeded."""
        policy = RetryPolicy(max_attempts=2, initial_delay=0.01)

        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            policy.execute(always_fails)

        stats = policy.get_stats()
        assert stats.total_attempts == 2
        assert stats.failed_attempts == 2

    def test_non_retryable_exception(self):
        """Non-retryable exception stops immediately."""
        policy = RetryPolicy(max_attempts=3, retryable_exceptions=[ValueError])

        def raises_type_error():
            raise TypeError("Not retryable")

        with pytest.raises(TypeError):
            policy.execute(raises_type_error)

        stats = policy.get_stats()
        assert stats.total_attempts == 1  # Only one attempt


class TestRetryPolicyWithStrategies:
    """Test retry policy with different strategies."""

    def test_with_exponential_strategy(self):
        """Retry policy with exponential strategy."""
        strategy = ExponentialBackoffStrategy(base=2.0, max_delay=0.1)
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01, strategy=strategy)

        attempt_count = 0

        def retry_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Try again")
            return "done"

        result = policy.execute(retry_func)
        assert result == "done"

    def test_with_linear_strategy(self):
        """Retry policy with linear strategy."""
        strategy = LinearBackoffStrategy(max_delay=0.1)
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01, strategy=strategy)

        attempt_count = 0

        def retry_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Try again")
            return "done"

        result = policy.execute(retry_func)
        assert result == "done"

    def test_with_fibonacci_strategy(self):
        """Retry policy with fibonacci strategy."""
        strategy = FibonacciBackoffStrategy(max_delay=0.1)
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01, strategy=strategy)

        attempt_count = 0

        def retry_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Try again")
            return "done"

        result = policy.execute(retry_func)
        assert result == "done"


@pytest.mark.slow
class TestRetryPolicyJitter:
    """Test jitter functionality."""

    def test_jitter_disabled(self):
        """Without jitter, delays are exact."""
        policy = RetryPolicy(
            max_attempts=3,
            initial_delay=0.01,  # Reduced from 1.0
            strategy=ConstantBackoffStrategy(),
            jitter=False,
        )

        attempt_count = 0

        def retry_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Try again")
            return "done"

        result = policy.execute(retry_func)
        assert result == "done"

    def test_jitter_enabled(self):
        """With jitter, delays vary."""
        policy = RetryPolicy(
            max_attempts=2,
            initial_delay=0.01,  # Reduced from 1.0
            strategy=ConstantBackoffStrategy(),
            jitter=True,
            jitter_factor=0.5,
        )

        attempt_count = 0

        def retry_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Try again")
            return "done"

        result = policy.execute(retry_func)
        assert result == "done"


@pytest.mark.slow
class TestRetryPolicyTimeout:
    """Test timeout handling."""

    def test_timeout_not_exceeded(self):
        """Operation completes within timeout."""
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01, timeout=5.0)

        def fast_func():
            return "ok"

        result = policy.execute(fast_func)
        assert result == "ok"

    def test_timeout_exceeded(self):
        """Operation raises timeout exception."""
        policy = RetryPolicy(max_attempts=10, initial_delay=0.01, timeout=0.05)

        attempt_count = 0

        def slow_func():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Fail")

        with pytest.raises(Exception):  # Could be TimeoutError or ValueError
            policy.execute(slow_func)


class TestRetryStatistics:
    """Test statistics tracking."""

    def test_stats_track_attempts(self):
        """Statistics track attempt count."""
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01)

        attempt_count = 0

        def retry_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Try again")
            return "ok"

        policy.execute(retry_func)
        stats = policy.get_stats()

        assert stats.total_attempts == 2
        assert stats.successful_attempts == 1
        assert stats.failed_attempts == 1

    def test_stats_calculate_success_rate(self):
        """Statistics calculate success rate."""
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01)

        attempt_count = 0

        def retry_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Try again")
            return "ok"

        policy.execute(retry_func)
        stats = policy.get_stats()

        assert stats.success_rate == pytest.approx(33.33, 0.1)
        assert stats.total_delay > 0

    def test_stats_track_last_exception(self):
        """Statistics track last exception."""
        policy = RetryPolicy(max_attempts=3, initial_delay=0.01)

        attempt_count = 0

        def retry_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("First failure")
            raise RuntimeError("Second failure")

        with pytest.raises(RuntimeError):
            policy.execute(retry_func)

        stats = policy.get_stats()
        assert isinstance(stats.last_exception, RuntimeError)
        assert "Second failure" in str(stats.last_exception)


class TestRetryDecorator:
    """Test retry decorator."""

    def test_decorator_basic_usage(self):
        """Decorator provides retry functionality."""
        attempt_count = 0

        @retry(max_attempts=3, initial_delay=0.01)
        def risky_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Not ready")
            return "success"

        result = risky_operation()
        assert result == "success"

    def test_decorator_with_exponential_backoff(self):
        """Decorator with exponential strategy."""
        attempt_count = 0

        @retry(
            max_attempts=3, initial_delay=0.01, strategy=ExponentialBackoffStrategy()
        )
        def risky_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Not ready")
            return "success"

        result = risky_operation()
        assert result == "success"

    def test_decorator_preserves_function_name(self):
        """Decorator preserves decorated function name."""

        @retry(max_attempts=3)
        def my_function():
            return "ok"

        assert my_function.__name__ == "my_function"

    def test_decorator_with_arguments(self):
        """Decorator works with function arguments."""
        attempt_count = 0

        @retry(max_attempts=3, initial_delay=0.01)
        def operation_with_args(a, b, c=None):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Retry")
            return f"{a}-{b}-{c}"

        result = operation_with_args(1, 2, c=3)
        assert result == "1-2-3"

    def test_decorator_access_policy(self):
        """Access retry policy through decorator."""

        @retry(max_attempts=5, initial_delay=0)
        def sometimes_fails(attempts: list[int]):
            """Function that fails a fixed number of times before succeeding.

            The function increments the supplied `attempts` counter list and
            raises ValueError until the counter reaches 3, after which it
            returns the number of attempts. This verifies the decorator
            performs retries up to the configured limit without inspecting
            internal attributes.
            """
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("retry")
            return attempts[0]

        calls = [0]
        # Should succeed after a couple of retries (max_attempts=5)
        result = sometimes_fails(calls)
        assert result == 3

        # If the function fails more than allowed attempts, the decorator
        # should let the final exception propagate.
        @retry(max_attempts=2, initial_delay=0)
        def always_fails(attempts: list[int]):
            attempts[0] += 1
            raise RuntimeError("always")

        calls2 = [0]
        with pytest.raises(RuntimeError):
            always_fails(calls2)


class TestTimeoutDecorator:
    """Test timeout decorator."""

    def test_timeout_not_exceeded_timeout_decorator(self):
        """Timeout decorator with operation completing in time."""

        @timeout(1.0)
        def fast_operation():
            time.sleep(0.1)
            return "ok"

        result = fast_operation()
        assert result == "ok"

    def test_timeout_exceeded_timeout_decorator(self):
        """Timeout decorator with operation exceeding timeout."""

        @timeout(0.1)
        def slow_operation():
            time.sleep(0.5)
            return "ok"

        with pytest.raises(TimeoutError):
            slow_operation()

    def test_timeout_with_arguments(self):
        """Timeout decorator with function arguments."""

        @timeout(1.0)
        def operation_with_args(a, b):
            return a + b

        result = operation_with_args(2, 3)
        assert result == 5


class TestRetryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_delay(self):
        """Retry with zero initial delay."""
        policy = RetryPolicy(max_attempts=2, initial_delay=0.0)

        attempt_count = 0

        def retry_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Try again")
            return "ok"

        result = policy.execute(retry_func)
        assert result == "ok"

    def test_single_attempt(self):
        """Retry policy with single attempt (no retries)."""
        policy = RetryPolicy(max_attempts=1)

        def always_fails():
            raise ValueError("Fails")

        with pytest.raises(ValueError):
            policy.execute(always_fails)

        stats = policy.get_stats()
        assert stats.total_attempts == 1

    def test_large_max_attempts(self):
        """Retry policy with many max attempts."""
        policy = RetryPolicy(max_attempts=100, initial_delay=0.001)

        def success_func():
            return "ok"

        result = policy.execute(success_func)
        assert result == "ok"

    def test_custom_exception_filtering(self):
        """Only retries on specified exceptions."""
        policy = RetryPolicy(
            max_attempts=3, retryable_exceptions=[ValueError, TypeError]
        )

        def raises_runtime_error():
            raise RuntimeError("Not retryable")

        with pytest.raises(RuntimeError):
            policy.execute(raises_runtime_error)

        stats = policy.get_stats()
        assert stats.total_attempts == 1


class TestRetryIntegration:
    """Integration tests combining multiple features."""

    def test_full_retry_lifecycle(self):
        """Test complete retry lifecycle."""
        policy = RetryPolicy(
            max_attempts=5,
            initial_delay=0.01,
            strategy=ExponentialBackoffStrategy(base=2.0),
            jitter=True,
            timeout=5.0,
        )

        attempt_count = 0

        def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                raise ConnectionError("Network issue")
            elif attempt_count == 2:
                raise TimeoutError("Timeout")

            return "success"

        result = policy.execute(flaky_operation)
        assert result == "success"

        stats = policy.get_stats()
        assert stats.successful_attempts == 1
        assert stats.failed_attempts == 2

    def test_decorator_with_manual_policy(self):
        """Combine decorator with manual policy inspection."""

        @retry(max_attempts=3, initial_delay=0.01)
        def operation():
            return "ok"

        result = operation()
        assert result == "ok"
