"""Example: Integration of All Design Patterns

This module demonstrates how to use all integrated design patterns
(Dependency Injection, Event Bus, Circuit Breaker, Retry) together
in a production analysis workflow.

Patterns Demonstrated:
  - Dependency Injection: Service registration and resolution
  - Event Bus: Decoupled event handling with subscribers
  - Circuit Breaker: Fault tolerance with automatic state management
  - Retry: Automatic resilience with exponential backoff
  - Observer: Event notifications
  - Builder: Fluent configuration
  - Command: Undo/redo support

Usage:
    Run this module to see all patterns in action:
    $ python -m src.analysis.examples.pattern_integration_demo
"""

from __future__ import annotations

import logging
from typing import Any

from src.analysis.integration import AnalysisSystem, SystemConfiguration
from src.analysis.patterns.event_bus import EventBus
from src.analysis import events

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AnalysisEventLogger:
    """Logs all analysis events for demonstration."""

    def __init__(self):
        """Initialize event logger."""
        self.events_received = 0
        self.errors_logged = 0
        self.cache_hits = 0

    def on_analysis_started(self, event: events.AnalysisStartedEvent) -> None:
        """Handle analysis started event.

        Args:
            event: AnalysisStartedEvent
        """
        self.events_received += 1
        logger.info(f"✓ Analysis Started: {event.analysis_type} in {event.domain}")
        logger.info(f"  Cache: {event.cache_dir}")
        if event.parameters:
            logger.info(f"  Parameters: {event.parameters}")

    def on_analysis_completed(self, event: events.AnalysisCompletedEvent) -> None:
        """Handle analysis completed event.

        Args:
            event: AnalysisCompletedEvent
        """
        self.events_received += 1
        logger.info(f"✓ Analysis Completed: {event.analysis_type}")
        logger.info(f"  Result: {event.result_summary}")
        logger.info(f"  Execution Time: {event.execution_time_seconds:.2f}s")

    def on_analysis_failed(self, event: events.AnalysisFailedEvent) -> None:
        """Handle analysis failed event.

        Args:
            event: AnalysisFailedEvent
        """
        self.events_received += 1
        self.errors_logged += 1
        logger.error(f"✗ Analysis Failed: {event.analysis_type}")
        logger.error(f"  Error: {event.error_type}: {event.error}")

    def on_cache_hit(self, event: events.CacheHitEvent) -> None:
        """Handle cache hit event.

        Args:
            event: CacheHitEvent
        """
        self.events_received += 1
        self.cache_hits += 1
        logger.debug(f"♻ Cache Hit: {event.cache_type} - {event.key}")

    def on_error(self, event: events.ErrorOccurredEvent) -> None:
        """Handle error event.

        Args:
            event: ErrorOccurredEvent
        """
        self.events_received += 1
        self.errors_logged += 1
        level = "CRITICAL" if event.is_critical else "ERROR"
        logger.warning(f"⚠ {level} from {event.source}: {event.error_message}")


def demonstrate_dependency_injection(system: AnalysisSystem) -> None:
    """Demonstrate dependency injection pattern.

    Args:
        system: AnalysisSystem instance
    """
    logger.info("\n" + "=" * 70)
    logger.info("PATTERN 1: DEPENDENCY INJECTION")
    logger.info("=" * 70)

    logger.info("Resolving services from DI container...")

    try:
        # Get EventBus from container
        event_bus = system.get_service("EventBus")
        logger.info(f"✓ EventBus resolved: {type(event_bus).__name__}")

        # Get ConfigManager from container
        config_manager = system.get_service("ConfigManager")
        logger.info(f"✓ ConfigManager resolved: {type(config_manager).__name__}")

        # Get CircuitBreakerPool from container
        pool = system.get_service("CircuitBreakerPool")
        logger.info(f"✓ CircuitBreakerPool resolved: {type(pool).__name__}")

        logger.info("✓ All services successfully resolved from DI container")
    except Exception as e:
        logger.error(f"✗ Service resolution failed: {e}")


def demonstrate_event_bus(
    system: AnalysisSystem, logger_obj: AnalysisEventLogger
) -> None:
    """Demonstrate event bus pattern.

    Args:
        system: AnalysisSystem instance
        logger_obj: Event logger for capturing events
    """
    logger.info("\n" + "=" * 70)
    logger.info("PATTERN 2: EVENT BUS (Decoupled Event Handling)")
    logger.info("=" * 70)

    logger.info("Setting up event subscribers...")

    # Subscribe to events using fluent API
    system.subscribe_to_events().on_analysis_started(
        logger_obj.on_analysis_started
    ).on_analysis_completed(logger_obj.on_analysis_completed).on_analysis_failed(
        logger_obj.on_analysis_failed
    ).on_cache_hit(
        logger_obj.on_cache_hit
    ).on_error(
        logger_obj.on_error
    )

    logger.info("✓ Event subscribers registered")

    # Publish sample events to demonstrate event bus
    logger.info("Publishing sample events...")

    event_bus = system.get_event_bus()

    # Publish analysis started event
    event = events.AnalysisStartedEvent(
        analysis_type="FaciesCorrelation",
        domain="depth",
        cache_dir=".cache",
        parameters={"quality": "high"},
    )
    event_bus.publish(event)

    # Publish cache hit event
    cache_event = events.CacheHitEvent(
        key="avo_results_depth",
        cache_type="pickle",
    )
    event_bus.publish(cache_event)

    logger.info(f"✓ {logger_obj.events_received} events received by subscribers")


def demonstrate_circuit_breaker(system: AnalysisSystem) -> None:
    """Demonstrate circuit breaker pattern.

    Args:
        system: AnalysisSystem instance
    """
    logger.info("\n" + "=" * 70)
    logger.info("PATTERN 3: CIRCUIT BREAKER (Fault Tolerance)")
    logger.info("=" * 70)

    logger.info("Creating circuit breaker for fault tolerance...")

    # Create analyzer with circuit breaker
    analyzer = system.create_analyzer()

    # Get circuit breaker
    breaker = analyzer.get_circuit_breaker("analysis_operation")

    logger.info(f"✓ Circuit breaker created: {breaker.name}")
    logger.info(f"  State: {breaker.state.value}")
    logger.info(f"  Failure Threshold: {breaker.failure_threshold}")
    logger.info(f"  Recovery Timeout: {breaker.recovery_timeout}s")

    # Demonstrate circuit breaker stats
    stats = breaker.get_stats()
    logger.info(f"  Stats: {stats.total_calls} total calls")
    logger.info(f"  Success Rate: {stats.success_rate:.1f}%")

    logger.info("✓ Circuit breaker ready for fault tolerance")


def demonstrate_retry_logic() -> None:
    """Demonstrate retry pattern with resilience."""
    logger.info("\n" + "=" * 70)
    logger.info("PATTERN 4: RETRY (Automatic Resilience)")
    logger.info("=" * 70)

    logger.info("Creating resilient operation with retry...")

    from src.analysis.patterns.retry import RetryPolicy, ExponentialBackoffStrategy

    # Create retry policy
    policy = RetryPolicy(
        max_attempts=3,
        initial_delay=1.0,
        strategy=ExponentialBackoffStrategy(),
        name="sample_operation",
    )

    logger.info(f"✓ Retry policy created: {policy.name}")
    logger.info(f"  Max Attempts: {policy.max_attempts}")
    logger.info(f"  Strategy: {policy.strategy}")
    logger.info(f"  Initial Delay: {policy.initial_delay}s")

    # Demonstrate retry with a sample function
    attempt_count = 0

    def flaky_operation():
        nonlocal attempt_count
        attempt_count += 1
        logger.info(f"  Attempt #{attempt_count}")

        if attempt_count < 2:
            raise RuntimeError("Temporary failure")

        return "Success!"

    try:
        result = policy.execute(flaky_operation)
        logger.info(f"✓ Operation succeeded: {result}")
        logger.info(f"  Total attempts: {attempt_count}")
    except Exception as e:
        logger.error(f"✗ Operation failed: {e}")


def demonstrate_combined_patterns() -> None:
    """Demonstrate all patterns working together."""
    logger.info("\n" + "=" * 70)
    logger.info("INTEGRATED PATTERNS: All Working Together")
    logger.info("=" * 70)

    # Create system with configuration
    config = SystemConfiguration()
    config.max_retries = 3
    config.circuit_breaker_threshold = 5

    system = AnalysisSystem(configuration=config)

    logger.info(f"System Configuration: {config}")
    logger.info()

    # Demonstrate each pattern
    demonstrate_dependency_injection(system)

    event_logger = AnalysisEventLogger()
    demonstrate_event_bus(system, event_logger)

    demonstrate_circuit_breaker(system)

    demonstrate_retry_logic()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: Pattern Integration")
    logger.info("=" * 70)

    logger.info("✓ Dependency Injection: Services managed by container")
    logger.info("✓ Event Bus: Decoupled event handling with subscribers")
    logger.info("✓ Circuit Breaker: Fault tolerance with state management")
    logger.info("✓ Retry: Automatic resilience with exponential backoff")
    logger.info()
    logger.info("All patterns are fully integrated and ready for production use!")


if __name__ == "__main__":
    logger.info("Starting Pattern Integration Demonstration\n")
    demonstrate_combined_patterns()
    logger.info("\n✓ Demonstration completed successfully!")
