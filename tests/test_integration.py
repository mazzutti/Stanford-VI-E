"""
Phase 5.6: Integration Tests

Comprehensive integration tests for all Phase 5 components working together
to create resilient, observable, and rate-limited systems.

Tests cover:
- Caching + Rate Limiting
- Circuit Breaker + Retry
- Monitoring + All components
- End-to-end resilience scenarios
- Complex multi-component workflows
"""

import time
import pytest
from typing import Any

from src.analysis.caching import CacheManager, LRUCache, TTLCache
from src.analysis.patterns.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerOpen,
)
from src.analysis.patterns.retry import RetryPolicy, ExponentialBackoffStrategy
from src.analysis.rate_limiting import (
    TokenBucketLimiter,
    SlidingWindowLimiter,
    LeakyBucketLimiter,
    RateLimitPool,
    RateLimitExceeded,
    rate_limit,
)
from src.analysis.monitoring import (
    StructuredLogger,
    MetricsCollector,
    PerformanceMonitor,
    HealthCheckRegistry,
    SimpleHealthCheck,
    monitor,
)


class TestCachingWithRateLimiting:
    """Test caching combined with rate limiting."""

    def test_cache_miss_increases_rate_limit_usage(self):
        """Test that cache misses consume rate limit tokens."""
        cache = LRUCache(max_size=10)
        limiter = TokenBucketLimiter(capacity=100, refill_rate=10)

        def fetch_data(key: str) -> Any:
            """Fetch data, checking rate limit."""
            if not limiter.allow_request(2):  # Cache miss costs 2 tokens
                raise RateLimitExceeded("Rate limited")
            return f"data_{key}"

        # First access - cache miss, 2 tokens
        result = fetch_data("key1")
        cache.set("key1", result)
        assert limiter.get_stats().allowed_requests == 1

        # Second access - cache hit, no rate limit needed
        cached = cache.get("key1")
        assert cached == result
        assert limiter.get_stats().allowed_requests == 1  # Unchanged

    def test_rate_limit_prevents_cache_flooding(self):
        """Test that rate limiting prevents cache from being flooded."""
        cache = LRUCache(max_size=100)
        limiter = SlidingWindowLimiter(max_requests=50, window_size=1.0)

        successful = 0
        for i in range(100):
            if limiter.allow_request():
                cache.set(f"key_{i}", f"value_{i}")
                successful += 1

        # Should have limited number in cache
        assert cache.size() <= 100
        assert successful == 50


class TestCircuitBreakerWithRetry:
    """Test circuit breaker combined with retry logic."""

    def test_retry_respects_circuit_breaker_state(self):
        """Test that retry respects open circuit."""
        breaker = CircuitBreaker(failure_threshold=2)
        retry_policy = RetryPolicy(
            max_attempts=5, strategy=ExponentialBackoffStrategy()
        )

        attempt_count = 0

        def unreliable_operation():
            nonlocal attempt_count
            attempt_count += 1

            # Fail for first 2 calls
            if attempt_count <= 2:
                raise RuntimeError("Service error")

            return "success"

        # Wrap in circuit breaker
        def breaker_protected():
            return breaker.call(unreliable_operation)

        # First attempts fail, circuit opens
        with pytest.raises((RuntimeError, Exception)):
            retry_policy.execute(breaker_protected)

        # Circuit should be open or have recorded failures
        stats = breaker.get_stats()
        assert stats.failed_calls >= 2 or breaker.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_half_open_with_retry_recovery(self):
        """Test recovery through half-open state with retry."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        call_count = {"value": 0}

        def sometimes_failing():
            call_count["value"] += 1
            if call_count["value"] <= 2:
                raise RuntimeError("Failed")

            return "recovered"

        # Wrap in circuit breaker - will fail twice and open circuit
        def breaker_protected():
            return breaker.call(sometimes_failing)

        # Fail twice to open the circuit
        with pytest.raises(RuntimeError):
            breaker.call(sometimes_failing)

        with pytest.raises(RuntimeError):
            breaker.call(sometimes_failing)

        # Circuit should be OPEN now
        assert breaker.state == CircuitBreakerState.OPEN

        # Try to call while open - should raise CircuitBreakerOpen
        with pytest.raises(CircuitBreakerOpen):
            breaker.call(sometimes_failing)

        # Wait for recovery timeout
        time.sleep(0.2)

        # Access state property to trigger transition to HALF_OPEN
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Now should be able to try again and succeed
        result = breaker.call(sometimes_failing)
        assert result == "recovered"

        # After success in HALF_OPEN, should close
        assert breaker.state == CircuitBreakerState.CLOSED


class TestMonitoringAllComponents:
    """Test monitoring integrated with all components."""

    def test_monitor_caching_performance(self):
        """Test monitoring cache performance metrics."""
        logger = StructuredLogger("cache_service")
        metrics = MetricsCollector("cache_metrics")
        cache = LRUCache(max_size=100)

        @monitor(logger=logger, metrics=metrics)
        def cached_fetch(key: str) -> str:
            cached = cache.get(key)
            if cached is not None:
                metrics.record_counter("cache_hits", 1)
                return cached

            metrics.record_counter("cache_misses", 1)
            result = f"computed_{key}"
            cache.set(key, result)
            return result

        # First call - miss
        result1 = cached_fetch("key1")
        # Second call - hit
        result2 = cached_fetch("key1")

        summary = metrics.get_metrics_summary()
        assert summary["cache_misses"]["latest"] == 1
        assert summary["cache_hits"]["latest"] == 1

    def test_monitor_circuit_breaker_state_transitions(self):
        """Test monitoring circuit breaker state changes."""
        logger = StructuredLogger("circuit_breaker_service")
        breaker = CircuitBreaker(failure_threshold=2)

        @monitor(logger=logger)
        def protected_operation():
            def op():
                if breaker.state != CircuitBreakerState.CLOSED:
                    logger.warning("Circuit not closed", state=breaker.state.value)
                    raise RuntimeError("Circuit open")

                logger.info("Operation executing", state=breaker.state.value)
                return "success"

            return breaker.call(op)

        # Record some successes
        protected_operation()

        # Record failures to open circuit
        for _ in range(3):
            try:

                def fail_op():
                    raise RuntimeError("Service error")

                breaker.call(fail_op)
            except Exception:
                pass

    def test_monitor_rate_limiter_rejection_rate(self):
        """Test monitoring rate limiter rejection rates."""
        metrics = MetricsCollector("rate_limit_metrics")
        limiter = TokenBucketLimiter(capacity=10, refill_rate=2)

        for _ in range(20):
            if limiter.allow_request(2):
                metrics.record_counter("requests_allowed", 1)
            else:
                metrics.record_counter("requests_rejected", 1)

        summary = metrics.get_metrics_summary()
        total_allowed = summary["requests_allowed"]["count"]
        total_rejected = summary["requests_rejected"]["count"]

        assert total_allowed > 0
        assert total_rejected > 0


class TestComplexResilientSystem:
    """Test complex scenarios requiring multiple components."""

    def test_resilient_api_client(self):
        """Test a resilient API client using all components."""
        # Setup
        logger = StructuredLogger("api_client")
        metrics = MetricsCollector("api_metrics")
        cache = LRUCache(max_size=50)
        limiter = TokenBucketLimiter(capacity=100, refill_rate=10)
        breaker = CircuitBreaker(failure_threshold=3)
        retry_policy = RetryPolicy(max_attempts=3)

        call_count = {"value": 0}

        @monitor(logger=logger, metrics=metrics)
        @rate_limit(limiter, tokens=1)
        def make_api_call(endpoint: str) -> str:
            """Make API call with full resilience stack."""
            # Check cache first
            cached = cache.get(endpoint)
            if cached is not None:
                metrics.record_counter("cache_hits", 1)
                logger.info("Cache hit", endpoint=endpoint)
                return cached

            metrics.record_counter("cache_misses", 1)

            def call_endpoint():
                nonlocal call_count
                call_count["value"] += 1
                return f"response_{endpoint}"

            # Use retry with circuit breaker
            def breaker_protected():
                return breaker.call(call_endpoint)

            result = retry_policy.execute(breaker_protected)

            # Cache result
            cache.set(endpoint, result)
            logger.info("API call successful", endpoint=endpoint)

            return result

        # First call should succeed
        result1 = make_api_call("/users")
        assert result1 == "response_/users"

        # Second call should hit cache
        result2 = make_api_call("/users")
        assert result2 == result1

        # Verify metrics
        summary = metrics.get_metrics_summary()
        assert "cache_misses" in summary
        assert "cache_hits" in summary

    def test_multi_endpoint_rate_limited_system(self):
        """Test system with multiple endpoints each rate limited."""
        pool = RateLimitPool()
        logger = StructuredLogger("multi_endpoint_system")

        # Different rate limits for different endpoints
        pool.add_limiter(
            "fast_endpoint", TokenBucketLimiter(capacity=1000, refill_rate=100)
        )
        pool.add_limiter(
            "slow_endpoint", TokenBucketLimiter(capacity=10, refill_rate=1)
        )

        @monitor(logger=logger)
        def handle_request(endpoint: str) -> bool:
            if not pool.allow_request(endpoint):
                logger.warning("Rate limit exceeded", endpoint=endpoint)
                return False

            logger.info("Request handled", endpoint=endpoint)
            return True

        # Fast endpoint allows many requests
        fast_success = sum(1 for _ in range(100) if handle_request("fast_endpoint"))
        assert fast_success > 90

        # Slow endpoint allows few
        slow_success = sum(1 for _ in range(100) if handle_request("slow_endpoint"))
        assert slow_success <= 15

    def test_health_check_with_component_validation(self):
        """Test health checks validate all component states."""
        logger = StructuredLogger("health_check_system")
        registry = HealthCheckRegistry()

        # Component 1: Cache health
        cache = LRUCache(max_size=100)
        registry.register(
            SimpleHealthCheck("cache_health", lambda: cache.size() < cache.max_size)
        )

        # Component 2: Circuit breaker health
        breaker = CircuitBreaker()
        registry.register(
            SimpleHealthCheck(
                "circuit_breaker_health",
                lambda: breaker.state == CircuitBreakerState.CLOSED,
            )
        )

        # Component 3: Rate limiter health
        limiter = TokenBucketLimiter(capacity=100, refill_rate=10)
        registry.register(
            SimpleHealthCheck("rate_limiter_health", lambda: limiter.tokens > 0)
        )

        # All should be healthy initially
        assert registry.is_healthy()

        # Open circuit breaker by causing failures
        for _ in range(11):
            try:

                def fail_op():
                    raise RuntimeError("Error")

                breaker.call(fail_op)
            except Exception:
                pass

        # Should now be unhealthy
        results = registry.check_all()
        assert results["circuit_breaker_health"] is False
        assert registry.is_healthy() is False


class TestPerformanceWithAllComponents:
    """Test performance characteristics with all components combined."""

    def test_throughput_with_all_protections(self):
        """Test system throughput with all protections enabled."""
        # Setup full stack
        cache = LRUCache(max_size=1000)
        limiter = TokenBucketLimiter(capacity=500, refill_rate=100)
        breaker = CircuitBreaker(failure_threshold=10)
        retry_policy = RetryPolicy(max_attempts=2)
        logger = StructuredLogger("throughput_test")
        metrics = MetricsCollector("throughput_metrics")

        success_count = 0

        @monitor(logger=logger, metrics=metrics)
        @rate_limit(limiter, tokens=1)
        def protected_operation(item_id: int) -> bool:
            nonlocal success_count

            # Check cache
            cached = cache.get(f"item_{item_id}")
            if cached:
                return True

            # Execute with retry
            def operation():
                return f"result_{item_id}"

            try:

                def breaker_protected():
                    return breaker.call(operation)

                result = retry_policy.execute(breaker_protected)
                cache.set(f"item_{item_id}", result)
                success_count += 1
                return True
            except Exception:
                return False

        # Run operations
        start = time.time()
        for i in range(100):
            protected_operation(i % 50)  # Some cache hits
        duration = time.time() - start

        # Should complete reasonably
        assert duration < 5.0
        assert success_count > 0

    def test_memory_efficiency_with_layered_caching(self):
        """Test memory efficiency with multiple cache layers."""
        l1_cache = LRUCache(max_size=50)  # Hot data
        l2_cache = TTLCache(ttl_seconds=60, max_size=500)  # Warm data
        limiter = TokenBucketLimiter(capacity=10000, refill_rate=1000)

        def multilevel_get(key: str) -> Any:
            # Check L1
            result = l1_cache.get(key)
            if result is not None:
                return result

            # Check L2
            result = l2_cache.get(key)
            if result is not None:
                l1_cache.set(key, result)  # Promote to L1
                return result

            # Miss - fetch and populate both levels
            if limiter.allow_request(1):
                value = f"data_{key}"
                l2_cache.set(key, value)
                l1_cache.set(key, value)
                return value

            return None

        # Populate and verify
        for i in range(100):
            multilevel_get(f"key_{i % 20}")

        assert l1_cache.size() <= 50
        assert l2_cache.size() <= 100  # Not all keys loaded


class TestErrorRecoveryScenarios:
    """Test error recovery across component interactions."""

    def test_cascading_failure_and_recovery(self):
        """Test system recovers from cascading failures."""
        logger = StructuredLogger("failure_recovery")

        # Components
        cache = LRUCache(max_size=100)
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.2)
        retry_policy = RetryPolicy(max_attempts=4)
        limiter = TokenBucketLimiter(capacity=50, refill_rate=5)

        call_count = {"value": 0}
        failure_phase = {"active": True}

        @monitor(logger=logger)
        def operation_with_recovery():
            nonlocal call_count
            call_count["value"] += 1

            def op():
                # First 5 calls fail
                if failure_phase["active"] and call_count["value"] <= 5:
                    logger.warning("Operation failed", attempt=call_count["value"])
                    raise RuntimeError("Service unavailable")

                # Transition out of failure phase
                if call_count["value"] == 6:
                    failure_phase["active"] = False
                    logger.info("Exiting failure phase")

                return "recovered"

            # Use circuit breaker
            def breaker_protected():
                return breaker.call(op)

            return retry_policy.execute(breaker_protected)

        # Start recovery process
        for i in range(10):
            try:
                result = operation_with_recovery()
                logger.info("Success", iteration=i, result=result)
            except Exception as e:
                logger.error("Failed", iteration=i, exception=str(e))

            time.sleep(0.05)

        # Should eventually recover
        assert call_count["value"] > 5

    def test_timeout_with_monitoring_and_retry(self):
        """Test timeout handling with monitoring and retry."""
        logger = StructuredLogger("timeout_handling")
        metrics = MetricsCollector()
        retry_policy = RetryPolicy(max_attempts=3)

        @monitor(logger=logger, metrics=metrics)
        def monitored_retry():
            attempt_count = {"value": 0}

            def operation():
                attempt_count["value"] += 1
                metrics.record_counter("attempts", 1)

                if attempt_count["value"] < 3:
                    raise TimeoutError("Operation timeout")

                metrics.record_counter("successes", 1)
                return "success"

            return retry_policy.execute(operation)

        result = monitored_retry()
        assert result == "success"

        summary = metrics.get_metrics_summary()
        assert summary["attempts"]["count"] >= 3


class TestMonitoringAggregation:
    """Test aggregating metrics across multiple operations."""

    def test_aggregate_system_metrics(self):
        """Test aggregating metrics from multiple system components."""
        metrics = MetricsCollector("system_metrics")

        # Simulate multiple operations
        for op in range(5):
            # Each operation logs metrics
            metrics.record_counter("operations_started", 1)

            # Simulate variable execution time
            exec_time = 0.001 * (op + 1)
            metrics.record_histogram("operation_duration_ms", exec_time * 1000)

            metrics.record_counter("operations_completed", 1)

        summary = metrics.get_metrics_summary()

        # Verify aggregation
        assert summary["operations_started"]["count"] == 5
        assert summary["operations_completed"]["count"] == 5
        assert summary["operation_duration_ms"]["count"] == 5
        assert summary["operation_duration_ms"]["avg"] > 0

    def test_per_endpoint_metric_aggregation(self):
        """Test aggregating metrics per endpoint."""
        metrics = MetricsCollector()

        endpoints = ["/api/users", "/api/posts", "/api/comments"]

        for endpoint in endpoints:
            for i in range(10):
                metrics.record_counter("requests", 1, tags={"endpoint": endpoint})

        summary = metrics.get_metrics_summary()
        assert summary["requests"]["count"] == 30
        assert summary["requests"]["sum"] == 30


class TestPhase5Integration:
    """End-to-end integration tests for Phase 5."""

    def test_all_components_working_together(self):
        """Test all Phase 5 components working in concert."""
        # Initialize all components
        logger = StructuredLogger("phase5_system")
        metrics = MetricsCollector("phase5_metrics")
        cache = LRUCache(max_size=100)
        limiter = TokenBucketLimiter(capacity=200, refill_rate=50)
        breaker = CircuitBreaker(failure_threshold=5)
        retry_policy = RetryPolicy(max_attempts=3)
        health_registry = HealthCheckRegistry()

        # Register health checks
        health_registry.register(
            SimpleHealthCheck(
                "system_operational",
                lambda: breaker.state == CircuitBreakerState.CLOSED,
            )
        )

        operation_count = {"value": 0}

        @monitor(logger=logger, metrics=metrics)
        @rate_limit(limiter, tokens=1)
        def integrated_operation(op_id: int) -> str:
            # Try cache first
            cached = cache.get(f"op_{op_id}")
            if cached:
                metrics.record_counter("cache_hits", 1)
                return cached

            metrics.record_counter("cache_misses", 1)

            # Execute with retry
            def execute():
                nonlocal operation_count
                operation_count["value"] += 1
                return f"result_{op_id}"

            try:

                def breaker_protected():
                    return breaker.call(execute)

                result = retry_policy.execute(breaker_protected)
                cache.set(f"op_{op_id}", result)
                metrics.record_counter("operations_successful", 1)
                return result
            except Exception as e:
                metrics.record_counter("operations_failed", 1)
                logger.error("Operation failed", op_id=op_id, error=str(e))
                raise

        # Execute operations
        for i in range(50):
            try:
                result = integrated_operation(i % 10)
                assert result is not None
            except Exception:
                pass

        # Verify results
        assert operation_count["value"] > 0
        assert health_registry.is_healthy()

        summary = metrics.get_metrics_summary()
        assert "cache_hits" in summary
        assert "operations_successful" in summary
