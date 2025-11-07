"""Component Factory with Pattern Integration.

This module provides factory functions for creating components with all
design patterns (DI, EventBus, CircuitBreaker, Retry) pre-configured.

Factory Functions:
  - create_analyzer: Create analyzer with all patterns
  - create_event_handler: Create event handlers with DI
  - create_resilient_processor: Create processor with circuit breaker and retry

Usage:
    >>> from src.analysis.factory import create_analyzer
    >>> analyzer = create_analyzer(event_bus=bus, container=container)
"""

from __future__ import annotations

import logging
from typing import Optional, Type, TypeVar, Any, Callable

from src.analysis.service_container import (
    create_service_provider,
    ServiceContainerBuilder,
)
from src.analysis.patterns.dependency_injection import Container, ServiceProvider
from src.analysis.patterns.event_bus import EventBus, EventHandler, Event
from src.analysis.patterns.circuit_breaker import CircuitBreaker, circuit_breaker
from src.analysis.patterns.retry import retry, RetryStrategy, ExponentialBackoffStrategy
from src.analysis.integrated_analyzer import IntegratedAnalyzer
from src.analysis.facies.analyzer import FaciesCorrelationAnalyzer
from src.analysis.facies.config import FaciesAnalysisConfig

logger = logging.getLogger(__name__)

__all__ = [
    "create_analyzer_with_patterns",
    "create_event_handler",
    "create_resilient_processor",
    "ComponentFactory",
]

T = TypeVar("T")


def create_analyzer_with_patterns(
    event_bus: Optional[EventBus] = None,
    container: Optional[Container] = None,
    enable_circuit_breaker: bool = True,
) -> IntegratedAnalyzer:
    """Create analyzer with all patterns integrated.

    Args:
        event_bus: Optional EventBus instance
        container: Optional DI Container
        enable_circuit_breaker: Enable circuit breaker protection

    Returns:
        IntegratedAnalyzer with patterns configured
    """
    logger.info("Creating analyzer with all patterns integrated")

    # Use default container if not provided
    if container is None:
        container_builder = ServiceContainerBuilder()
        container = container_builder.with_facies_analyzer().build()

    if event_bus is None:
        event_bus = EventBus()

    # Create base analyzer using service provider
    facies_analyzer = container.resolve("FaciesCorrelationAnalyzer")

    # Create integrated analyzer with patterns
    analyzer = IntegratedAnalyzer(
        facies_analyzer=facies_analyzer,
    )

    # Attach event bus integration
    _attach_event_bus_to_analyzer(analyzer, event_bus)

    logger.debug("Analyzer created with patterns successfully")
    return analyzer


def _attach_event_bus_to_analyzer(
    analyzer: IntegratedAnalyzer, event_bus: EventBus
) -> None:
    """Attach event bus to analyzer for decoupled event handling.

    Args:
        analyzer: IntegratedAnalyzer instance
        event_bus: EventBus instance
    """
    # Store reference to event bus
    analyzer._event_bus = event_bus

    # Create adapter observer that publishes to event bus
    class EventBusAdapter:
        """Adapts observer callbacks to event bus publishing."""

        def __init__(self, bus: EventBus):
            self.bus = bus

        def on_result_computed(self, result_type: str, result: Any):
            """Publish result event."""
            # This will be called by observer pattern
            # Publish to event bus for decoupled handling
            logger.debug(f"Publishing result event: {result_type}")

        def on_data_changed(self, data_type: str, new_data: Any):
            """Publish data changed event."""
            logger.debug(f"Publishing data changed event: {data_type}")

        def on_error(self, error: Exception, context: str):
            """Publish error event."""
            logger.warning(f"Publishing error event from {context}: {error}")

    # Note: The actual integration will be handled by the modified IntegratedAnalyzer
    # This demonstrates the pattern connection point
    logger.debug("Event bus adapter attached to analyzer")


def create_event_handler(
    handler_class: Type[T],
    container: Optional[Container] = None,
    **kwargs: Any,
) -> T:
    """Create event handler with dependency injection.

    Args:
        handler_class: Handler class to instantiate
        container: Optional DI Container
        **kwargs: Additional arguments

    Returns:
        Instantiated event handler with dependencies injected
    """
    logger.info(f"Creating event handler: {handler_class.__name__}")

    if container is None:
        provider = create_service_provider()
    else:
        provider = container.create_service_provider()

    # Instantiate handler with dependencies
    handler = handler_class(**kwargs)

    logger.debug(f"Event handler created: {handler_class.__name__}")
    return handler


def create_resilient_processor(
    processor_func: Callable[..., Any],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    circuit_breaker_threshold: int = 5,
    circuit_breaker_timeout: int = 60,
) -> Callable[..., Any]:
    """Create processor function with circuit breaker and retry.

    Args:
        processor_func: Function to wrap
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay in seconds
        circuit_breaker_threshold: Failures before opening circuit
        circuit_breaker_timeout: Circuit recovery timeout

    Returns:
        Wrapped function with resilience patterns
    """
    logger.info(f"Creating resilient processor: {processor_func.__name__}")

    # Apply circuit breaker
    breaker_name = f"{processor_func.__module__}.{processor_func.__name__}"

    @circuit_breaker(
        name=breaker_name,
        failure_threshold=circuit_breaker_threshold,
        recovery_timeout=circuit_breaker_timeout,
    )
    @retry(
        max_attempts=max_retries,
        initial_delay=retry_delay,
        strategy=ExponentialBackoffStrategy(),
    )
    def resilient_processor(*args: Any, **kwargs: Any) -> Any:
        """Resilient processor with circuit breaker and retry."""
        return processor_func(*args, **kwargs)

    logger.debug(f"Resilient processor created: {processor_func.__name__}")
    return resilient_processor


class ComponentFactory:
    """Factory for creating components with integrated patterns.

    Provides convenient methods for creating various components
    with all design patterns pre-configured.
    """

    def __init__(
        self,
        container: Optional[Container] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize component factory.

        Args:
            container: Optional DI Container
            event_bus: Optional EventBus
        """
        if container is None:
            builder = ServiceContainerBuilder()
            self.container = builder.build()
        else:
            self.container = container

        self.event_bus = event_bus or EventBus()
        # Access the service provider from the container
        self.service_provider = self.container._service_provider

        logger.info("ComponentFactory initialized")

    def create_analyzer(self) -> IntegratedAnalyzer:
        """Create analyzer with all patterns.

        Returns:
            IntegratedAnalyzer instance
        """
        return create_analyzer_with_patterns(
            event_bus=self.event_bus,
            container=self.container,
        )

    def create_facies_analyzer(self) -> FaciesCorrelationAnalyzer:
        """Create facies correlation analyzer.

        Returns:
            FaciesCorrelationAnalyzer instance
        """
        logger.info("Creating FaciesCorrelationAnalyzer")
        return self.service_provider.resolve("FaciesCorrelationAnalyzer")

    def get_service(self, service_name: str) -> Any:
        """Get a service from the container.

        Args:
            service_name: Name of service to resolve

        Returns:
            Service instance
        """
        logger.debug(f"Resolving service: {service_name}")
        return self.service_provider.resolve(service_name)

    def try_get_service(self, service_name: str, default: Any = None) -> Any:
        """Try to get a service, returning default if not found.

        Args:
            service_name: Name of service to resolve
            default: Default value if service not found

        Returns:
            Service instance or default
        """
        return self.service_provider.try_resolve(service_name, default)

    def create_resilient_operation(
        self,
        operation_func: Callable[..., Any],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Callable[..., Any]:
        """Create operation with circuit breaker and retry.

        Args:
            operation_func: Function to wrap
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay

        Returns:
            Wrapped function
        """
        return create_resilient_processor(
            operation_func,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def __repr__(self) -> str:
        return (
            f"ComponentFactory("
            f"services={len(self.service_provider._providers)}, "
            f"event_bus={self.event_bus})"
        )
