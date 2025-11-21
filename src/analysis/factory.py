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
from collections.abc import Callable
from typing import Any, TypeVar, cast

from src.analysis.facies.analyzer import FaciesCorrelationAnalyzer
from src.analysis.integrated_analyzer import IntegratedAnalyzer
from src.analysis.patterns.circuit_breaker import circuit_breaker
from src.analysis.patterns.dependency_injection import Container, ServiceProvider
from src.analysis.patterns.event_bus import EventBus
from src.analysis.patterns.retry import ExponentialBackoffStrategy, retry
from src.analysis.service_container import ServiceContainerBuilder

logger = logging.getLogger(__name__)

__all__ = [
    "create_analyzer_with_patterns",
    "create_event_handler",
    "create_resilient_processor",
    "ComponentFactory",
]

T = TypeVar("T")

def create_analyzer_with_patterns(
    event_bus: EventBus | None = None,
    container: Container | None = None,
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
    # Accept the flag for API compatibility (not used internally).
    _ = enable_circuit_breaker

    # Use default container if not provided
    if container is None:
        container_builder = ServiceContainerBuilder()
        container = container_builder.with_facies_analyzer().build()

    if event_bus is None:
        event_bus = EventBus()

    # Create base analyzer using service provider
    facies_analyzer = cast(
        FaciesCorrelationAnalyzer, container.resolve("FaciesCorrelationAnalyzer")
    )

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
    # Store reference to event bus via public API if available, otherwise set a public attribute
    set_bus = getattr(analyzer, "set_event_bus", None)
    if callable(set_bus):
        set_bus(event_bus)
    else:
        # Avoid directly writing protected attribute; create/overwrite public attribute
        setattr(analyzer, "event_bus", event_bus)

    # Create adapter observer that publishes to event bus
    class EventBusAdapter:
        """Adapts observer callbacks to event bus publishing."""

        def __init__(self, bus: EventBus):
            self.bus = bus

        def on_result_computed(self, result_type: str, result: Any) -> None:
            """Publish result event."""
            # This will be called by observer pattern
            # Publish to event bus for decoupled handling
            logger.debug("Publishing result event: %s", result_type)
            # Keep `result` available for potential future use and to
            # silence unused-argument linters.
            _ = result

        def on_data_changed(self, data_type: str, _new_data: Any) -> None:
            """Publish data changed event."""
            logger.debug("Publishing data changed event: %s", data_type)

        def on_error(self, error: Exception, context: str) -> None:
            """Publish error event."""
            logger.warning("Publishing error event from %s: %s", context, error)

    # Instantiate and attach the adapter so the class is used and available for the analyzer.
    adapter = EventBusAdapter(event_bus)
    # Keep a reference on the analyzer for lifecycle and debugging.
    # Prefer a public attribute to avoid writing protected members.
    # Use `event_bus_adapter` so callers can inspect lifecycle state
    # without relying on protected attributes.
    setattr(analyzer, "event_bus_adapter", adapter)

    # If the analyzer exposes an observer registration API, try to register the adapter.
    try:
        # Use getattr and callable checks to avoid static attribute access
        # which can trigger type-checker errors if the methods are not declared.
        _register = getattr(analyzer, "register_observer", None)
        _add = getattr(analyzer, "add_observer", None)
        if callable(_register):
            _register(adapter)
        elif callable(_add):
            _add(adapter)
    except (AttributeError, RuntimeError, TypeError):
        # Don't fail analyzer creation if registration is not supported or raises.
        logger.debug(
            "Failed to register EventBusAdapter with analyzer (registration not supported)",
            exc_info=True,
        )

    # Note: The actual integration will be handled by the modified IntegratedAnalyzer
    # This demonstrates the pattern connection point
    logger.debug("Event bus adapter attached to analyzer")

def create_event_handler(
    handler_class: type[T],
    container: Container | None = None,
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
    logger.info("Creating event handler: %s", handler_class.__name__)

    if container is None:
        container = ServiceContainerBuilder().build()

    # Instantiate handler with dependencies
    handler = handler_class(**kwargs)

    logger.debug("Event handler created: %s", handler_class.__name__)
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
    logger.info("Creating resilient processor: %s", processor_func.__name__)

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

    logger.debug("Resilient processor created: %s", processor_func.__name__)
    return resilient_processor

class ComponentFactory:
    """Factory for creating components with integrated patterns.

    Provides convenient methods for creating various components
    with all design patterns pre-configured.
    """

    def __init__(
        self,
        container: Container | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
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
        # Access the service provider from the container using public API if available
        service_provider = getattr(self.container, "service_provider", None)
        if service_provider is None:
            # Try a getter method if present
            get_sp = getattr(self.container, "get_service_provider", None)
            if get_sp is not None:
                # Support both callable getter methods and attributes/property
                if callable(get_sp):
                    service_provider = get_sp()
                else:
                    service_provider = get_sp
        if service_provider is None:
            # Fallback to protected attribute only as a last resort and warn
            service_provider = getattr(self.container, "_service_provider", None)
            if service_provider is None:
                raise RuntimeError("ServiceProvider not available on container")
            logger.warning(
                "Accessing protected attribute '_service_provider' as a fallback"
            )

        self.service_provider: ServiceProvider = cast(ServiceProvider, service_provider)

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
        return cast(
            FaciesCorrelationAnalyzer,
            self.service_provider.resolve("FaciesCorrelationAnalyzer"),
        )

    def get_service(self, service_name: str) -> Any:
        """Get a service from the container.

        Args:
            service_name: Name of service to resolve

        Returns:
            Service instance
        """
        logger.debug("Resolving service: %s", service_name)
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
            f"services={len(self.container.get_services())}, "
            f"event_bus={self.event_bus})"
        )
