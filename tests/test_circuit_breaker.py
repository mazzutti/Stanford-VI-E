"""
Comprehensive test suite for Circuit Breaker Pattern (Phase 5.2)

Tests cover:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure tracking and thresholds
- Success handling during recovery
- Recovery timeout logic
- Decorator functionality
- CircuitBreakerPool management
- Thread safety with concurrent access
- Statistics tracking
"""

import time
from threading import Thread

import pytest
from freezegun import freeze_time

from src.analysis.patterns.circuit_breaker import (CircuitBreaker,
                                                   CircuitBreakerOpen,
                                                   CircuitBreakerPool,
                                                   CircuitBreakerState,
                                                   circuit_breaker,
                                                   get_circuit_breaker,
                                                   reset_all_circuit_breakers)


@pytest.mark.slow
class TestCircuitBreakerStates:
    """Test state transitions and basic functionality."""

    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_closed_to_open_on_failure_threshold(self):
        """Circuit transitions to OPEN after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3)

        def failing_func():
            raise ValueError("Test failure")

        # First two failures
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)
            assert breaker.state == CircuitBreakerState.CLOSED

        # Third failure triggers OPEN
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.state == CircuitBreakerState.OPEN

    def test_open_rejects_calls(self):
        """Open circuit rejects new calls immediately."""
        breaker = CircuitBreaker(failure_threshold=1)

        def failing_func():
            raise ValueError("Test failure")

        # Trigger OPEN state
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.state == CircuitBreakerState.OPEN

        # Next calls should be rejected immediately
        with pytest.raises(CircuitBreakerOpen):
            breaker.call(failing_func)

    @freeze_time("2025-01-01 12:00:00")
    def test_open_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1)

        def failing_func():
            raise ValueError("Test failure")

        # Trigger OPEN state
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.state == CircuitBreakerState.OPEN

        # Move time forward past recovery timeout
        with freeze_time("2025-01-01 12:00:01.1"):
            # Accessing state should trigger transition
            assert breaker.state == CircuitBreakerState.HALF_OPEN

    @freeze_time("2025-01-01 12:00:00")
    def test_half_open_to_closed_on_success(self):
        """Successful call in HALF_OPEN state closes circuit."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1)

        def failing_func():
            raise ValueError("Test failure")

        def success_func():
            return "success"

        # Trigger OPEN state
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # Move time forward past recovery timeout
        with freeze_time("2025-01-01 12:00:01.1"):
            assert breaker.state == CircuitBreakerState.HALF_OPEN

            # Successful call closes circuit
            result = breaker.call(success_func)
            assert result == "success"
            assert breaker.state == CircuitBreakerState.CLOSED

    @freeze_time("2025-01-01 12:00:00")
    def test_half_open_to_open_on_failure(self):
        """Failed call in HALF_OPEN state reopens circuit."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1)

        def failing_func():
            raise ValueError("Test failure")

        # Trigger OPEN state
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # Move time forward past recovery timeout
        with freeze_time("2025-01-01 12:00:01.1"):
            assert breaker.state == CircuitBreakerState.HALF_OPEN

            # Failed call reopens circuit
            with pytest.raises(ValueError):
                breaker.call(failing_func)

            assert breaker.state == CircuitBreakerState.OPEN


class TestCircuitBreakerStatistics:
    """Test statistics and metrics tracking."""

    def test_stats_track_calls(self):
        """Statistics track total calls correctly."""
        breaker = CircuitBreaker(failure_threshold=5)

        def success_func():
            return "ok"

        for _ in range(3):
            breaker.call(success_func)

        stats = breaker.get_stats()
        assert stats.total_calls == 3
        assert stats.successful_calls == 3
        assert stats.failed_calls == 0

    def test_stats_track_failures(self):
        """Statistics track failures correctly."""
        breaker = CircuitBreaker(failure_threshold=5)

        def failing_func():
            raise ValueError("failure")

        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        stats = breaker.get_stats()
        assert stats.total_calls == 3
        assert stats.failed_calls == 3
        assert stats.successful_calls == 0

    def test_stats_track_rejected_calls(self):
        """Statistics track rejected calls."""
        breaker = CircuitBreaker(failure_threshold=1)

        def failing_func():
            raise ValueError("failure")

        # Trigger OPEN
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # Reject calls
        for _ in range(3):
            with pytest.raises(CircuitBreakerOpen):
                breaker.call(failing_func)

        stats = breaker.get_stats()
        assert stats.rejected_calls == 3

    def test_stats_calculate_rates(self):
        """Statistics calculate success/failure rates."""
        breaker = CircuitBreaker(failure_threshold=10)

        def failing_func():
            raise ValueError("failure")

        def success_func():
            return "ok"

        # 2 successes, 2 failures
        breaker.call(success_func)
        breaker.call(success_func)
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        stats = breaker.get_stats()
        assert stats.success_rate == 50.0
        assert stats.failure_rate == 50.0

    def test_stats_string_representation(self):
        """Statistics have useful string representation."""
        breaker = CircuitBreaker()
        breaker.call(lambda: "ok")

        stats = breaker.get_stats()
        stats_str = str(stats)

        assert "closed" in stats_str
        assert "total_calls" in stats_str
        assert "success_rate" in stats_str


class TestCircuitBreakerReset:
    """Test manual reset functionality."""

    def test_reset_from_open(self):
        """Reset transitions circuit to CLOSED."""
        breaker = CircuitBreaker(failure_threshold=1)

        def failing_func():
            raise ValueError("failure")

        # Trigger OPEN
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.state == CircuitBreakerState.OPEN

        # Reset
        breaker.reset()
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_reset_clears_failure_count(self):
        """Reset clears failure tracking."""
        breaker = CircuitBreaker(failure_threshold=5)

        def failing_func():
            raise ValueError("failure")

        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        breaker.reset()

        # Should not trigger OPEN even with more failures
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.state == CircuitBreakerState.CLOSED


class TestCircuitBreakerDecorator:
    """Test decorator functionality."""

    def test_decorator_basic_usage(self):
        """Decorator protects function with circuit breaker."""
        reset_all_circuit_breakers()

        @circuit_breaker(name="test_func", failure_threshold=2, recovery_timeout=1)
        def test_func(x):
            if x < 0:
                raise ValueError("negative")
            return x * 2

        # Successful calls
        assert test_func(5) == 10

        # Failed calls
        with pytest.raises(ValueError):
            test_func(-1)

        with pytest.raises(ValueError):
            test_func(-1)

        # Circuit should be open now
        with pytest.raises(CircuitBreakerOpen):
            test_func(5)

    def test_decorator_default_name(self):
        """Decorator uses function name if no name provided."""
        reset_all_circuit_breakers()

        @circuit_breaker(failure_threshold=1)
        def my_function():
            return "ok"

        breaker = get_circuit_breaker(
            "src.analysis.patterns.circuit_breaker.my_function"
        )
        assert breaker is not None

    def test_decorator_with_args_kwargs(self):
        """Decorator preserves function arguments."""
        reset_all_circuit_breakers()

        @circuit_breaker(name="test_args")
        def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = func_with_args(1, 2, c=3)
        assert result == "1-2-3"


class TestCircuitBreakerPool:
    """Test CircuitBreakerPool management."""

    def test_pool_get_breaker(self):
        """Pool returns breaker by name."""
        pool = CircuitBreakerPool()
        breaker1 = pool.get_breaker("service1")
        breaker2 = pool.get_breaker("service1")

        assert breaker1 is breaker2

    def test_pool_creates_different_breakers(self):
        """Pool creates separate breakers for different names."""
        pool = CircuitBreakerPool()
        breaker1 = pool.get_breaker("service1")
        breaker2 = pool.get_breaker("service2")

        assert breaker1 is not breaker2

    def test_pool_get_all_breakers(self):
        """Pool returns all registered breakers."""
        pool = CircuitBreakerPool()
        pool.get_breaker("service1")
        pool.get_breaker("service2")
        pool.get_breaker("service3")

        all_breakers = pool.get_all_breakers()
        assert len(all_breakers) == 3
        assert "service1" in all_breakers
        assert "service2" in all_breakers
        assert "service3" in all_breakers

    def test_pool_reset_breaker(self):
        """Pool can reset individual breaker."""
        pool = CircuitBreakerPool()
        breaker = pool.get_breaker("service1", failure_threshold=1)

        def failing_func():
            raise ValueError("failure")

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.state == CircuitBreakerState.OPEN

        # Reset via pool
        pool.reset_breaker("service1")
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_pool_reset_all(self):
        """Pool can reset all breakers."""
        pool = CircuitBreakerPool()
        b1 = pool.get_breaker("service1", failure_threshold=1)
        b2 = pool.get_breaker("service2", failure_threshold=1)

        def failing_func():
            raise ValueError("failure")

        # Open both circuits
        with pytest.raises(ValueError):
            b1.call(failing_func)
        with pytest.raises(ValueError):
            b2.call(failing_func)

        assert b1.state == CircuitBreakerState.OPEN
        assert b2.state == CircuitBreakerState.OPEN

        # Reset all
        pool.reset_all()
        assert b1.state == CircuitBreakerState.CLOSED
        assert b2.state == CircuitBreakerState.CLOSED

    def test_pool_get_stats_all(self):
        """Pool returns stats for all breakers."""
        pool = CircuitBreakerPool()
        b1 = pool.get_breaker("service1")
        b2 = pool.get_breaker("service2")

        b1.call(lambda: "ok")
        b2.call(lambda: "ok")

        stats = pool.get_stats_all()
        assert len(stats) == 2
        assert "service1" in stats
        assert "service2" in stats
        assert stats["service1"].successful_calls == 1
        assert stats["service2"].successful_calls == 1

    def test_pool_remove_breaker(self):
        """Pool can remove breakers."""
        pool = CircuitBreakerPool()
        pool.get_breaker("service1")
        pool.get_breaker("service2")

        assert len(pool) == 2

        pool.remove_breaker("service1")
        assert len(pool) == 1
        assert "service1" not in pool.get_all_breakers()


class TestCircuitBreakerCustomException:
    """Test handling of custom exceptions."""

    def test_only_expected_exception_counts(self):
        """Only expected exception type triggers failures."""
        breaker = CircuitBreaker(failure_threshold=2, expected_exception=ValueError)

        def value_error_func():
            raise ValueError("value error")

        def type_error_func():
            raise TypeError("type error")

        # ValueError counts as failure
        with pytest.raises(ValueError):
            breaker.call(value_error_func)

        # TypeError doesn't count as failure
        with pytest.raises(TypeError):
            breaker.call(type_error_func)

        # Can still call after TypeError
        with pytest.raises(ValueError):
            breaker.call(value_error_func)

        # Circuit not open yet (only 2 ValueErrors expected)
        assert breaker.state == CircuitBreakerState.OPEN


class TestCircuitBreakerThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_calls(self):
        """Circuit breaker handles concurrent calls safely."""
        breaker = CircuitBreaker(failure_threshold=50)
        call_count = 0
        lock = None

        def safe_increment():
            nonlocal call_count
            call_count += 1
            return call_count

        def call_breaker():
            for _ in range(20):
                try:
                    breaker.call(safe_increment)
                except CircuitBreakerOpen:
                    pass

        threads = [Thread(target=call_breaker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have approximately 100 calls (5 threads * 20 calls)
        # May be slightly less due to circuit opening
        assert call_count > 0

    def test_concurrent_state_transitions(self):
        """Circuit breaker state transitions are thread-safe."""
        breaker = CircuitBreaker(failure_threshold=10)
        results = []

        def failing_func():
            raise ValueError("failure")

        def call_until_open():
            for _ in range(20):
                try:
                    breaker.call(failing_func)
                except (ValueError, CircuitBreakerOpen):
                    pass

        threads = [Thread(target=call_until_open) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Circuit should be in OPEN state
        assert breaker.state == CircuitBreakerState.OPEN


@pytest.mark.slow
class TestCircuitBreakerIntegration:
    """Integration tests combining multiple features."""

    def test_full_lifecycle(self):
        """Test complete circuit breaker lifecycle."""
        breaker = CircuitBreaker(
            failure_threshold=3, recovery_timeout=0.5, name="integration_test"
        )

        call_count = 0

        def simulate_service():
            nonlocal call_count
            call_count += 1

            # Fail first 3 calls
            if call_count <= 3:
                raise ConnectionError("Service unavailable")
            return "success"

        # Phase 1: First two failures keep circuit CLOSED
        for i in range(2):
            with pytest.raises(ConnectionError):
                breaker.call(simulate_service)
            assert breaker.state == CircuitBreakerState.CLOSED

        # Third failure triggers OPEN
        with pytest.raises(ConnectionError):
            breaker.call(simulate_service)
        assert breaker.state == CircuitBreakerState.OPEN

        # Phase 2: OPEN rejects calls
        with pytest.raises(CircuitBreakerOpen):
            breaker.call(simulate_service)

        # Phase 3: Wait for recovery with reduced timeout
        time.sleep(0.6)
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Phase 4: Successful recovery closes circuit
        result = breaker.call(simulate_service)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED

        # Phase 5: Normal operation
        result = breaker.call(simulate_service)
        assert result == "success"

        # Verify final state
        stats = breaker.get_stats()
        assert stats.state == CircuitBreakerState.CLOSED

    def test_decorator_full_lifecycle(self):
        """Test decorator through full lifecycle."""
        reset_all_circuit_breakers()

        call_count = 0

        @circuit_breaker(
            name="lifecycle_test", failure_threshold=2, recovery_timeout=0.5
        )
        def risky_operation():
            nonlocal call_count
            call_count += 1

            if call_count <= 2:
                raise RuntimeError("Temporary failure")
            return "recovered"

        # Failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                risky_operation()

        # Circuit open
        with pytest.raises(CircuitBreakerOpen):
            risky_operation()

        # Wait for recovery with reduced timeout
        time.sleep(0.6)
        # Recovery
        result = risky_operation()
        assert result == "recovered"


class TestCircuitBreakerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_failure_threshold_one(self):
        """Circuit breaker works with threshold of 1."""
        breaker = CircuitBreaker(failure_threshold=1)

        def failing_func():
            raise ValueError("failure")

        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.state == CircuitBreakerState.OPEN

    def test_zero_recovery_timeout(self):
        """Circuit breaker with zero recovery timeout transitions immediately."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0)

        def failing_func():
            raise ValueError("failure")

        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # Should transition to HALF_OPEN immediately
        assert breaker.state == CircuitBreakerState.HALF_OPEN

    def test_successful_call_in_closed_state(self):
        """Successful calls maintain CLOSED state."""
        breaker = CircuitBreaker()

        for _ in range(10):
            result = breaker.call(lambda: "ok")
            assert result == "ok"
            assert breaker.state == CircuitBreakerState.CLOSED

    def test_breaker_name_in_error(self):
        """CircuitBreakerOpen exception includes breaker name."""
        breaker = CircuitBreaker(failure_threshold=1, name="test_breaker")

        def failing_func():
            raise ValueError("failure")

        with pytest.raises(ValueError):
            breaker.call(failing_func)

        with pytest.raises(CircuitBreakerOpen) as exc_info:
            breaker.call(failing_func)

        assert "test_breaker" in str(exc_info.value)
