"""Pattern Integration Module

This module demonstrates how to integrate all design patterns (DI, EventBus,
CircuitBreaker, Retry) into a cohesive analysis framework.

It provides high-level APIs that abstract away the complexity of individual
patterns while maintaining their benefits.

Usage:
    >>> from src.analysis.integration import AnalysisSystem
    >>> system = AnalysisSystem()
    >>> analyzer = system.create_analyzer()
    >>> system.subscribe_to_analysis_events(on_complete=my_handler)
    >>> result = analyzer.run_with_command(cache_dir=".cache", domain="depth")
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import TracebackType
from typing import Any

from src.analysis import events
from src.analysis.factory import ComponentFactory, create_analyzer_with_patterns
from src.analysis.integrated_analyzer import IntegratedAnalyzer
from src.analysis.patterns.circuit_breaker import reset_all_circuit_breakers
from src.analysis.patterns.dependency_injection import Container
from src.analysis.patterns.event_bus import Event, EventBus, EventHandler
from src.analysis.service_container import ServiceContainerBuilder

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisSystem",
    "SystemConfiguration",
    "EventSubscriber",
]

class EventSubscriber:
    """Helper for subscribing to analysis events."""

    def __init__(self, event_bus: EventBus):
        """Initialize event subscriber.

        Args:
            event_bus: EventBus instance
        """
        self.event_bus = event_bus
        self._subscriptions: list[Any] = []

    def _wrap_handler(self, fn: Callable[[Any], None]) -> EventHandler:
        """Wrap a simple callable into an EventHandler instance."""

        class _FuncHandler(EventHandler):
            def __init__(self, f: Callable[[Any], None]) -> None:
                self._f = f

            def handle(self, event: Event) -> None:
                self._f(event)

        return _FuncHandler(fn)

    # Integration glue — the module composes many patterns; small helper classes
    # are concise by design to avoid unnecessary indirection.

    def on_analysis_started(
        self, handler: Callable[[events.AnalysisStartedEvent], None]
    ) -> EventSubscriber:
        """Subscribe to analysis started events.

        Args:
            handler: Callback function

        Returns:
            Self for chaining
        """

        handler_obj = self._wrap_handler(handler)
        sub = self.event_bus.subscribe(events.AnalysisStartedEvent, handler_obj)
        self._subscriptions.append(sub)
        logger.debug("Subscribed to analysis started events")
        return self

    def on_analysis_completed(
        self, handler: Callable[[events.AnalysisCompletedEvent], None]
    ) -> EventSubscriber:
        """Subscribe to analysis completed events.

        Args:
            handler: Callback function

        Returns:
            Self for chaining
        """

        handler_obj = self._wrap_handler(handler)
        sub = self.event_bus.subscribe(events.AnalysisCompletedEvent, handler_obj)
        self._subscriptions.append(sub)
        logger.debug("Subscribed to analysis completed events")
        return self

    def on_analysis_failed(
        self, handler: Callable[[events.AnalysisFailedEvent], None]
    ) -> EventSubscriber:
        """Subscribe to analysis failed events.

        Args:
            handler: Callback function

        Returns:
            Self for chaining
        """

        handler_obj = self._wrap_handler(handler)
        sub = self.event_bus.subscribe(events.AnalysisFailedEvent, handler_obj)
        self._subscriptions.append(sub)
        logger.debug("Subscribed to analysis failed events")
        return self

    def on_error(
        self, handler: Callable[[events.ErrorOccurredEvent], None]
    ) -> EventSubscriber:
        """Subscribe to error events.

        Args:
            handler: Callback function

        Returns:
            Self for chaining
        """

        handler_obj = self._wrap_handler(handler)
        sub = self.event_bus.subscribe(events.ErrorOccurredEvent, handler_obj)
        self._subscriptions.append(sub)
        logger.debug("Subscribed to error events")
        return self

    def on_cache_hit(
        self, handler: Callable[[events.CacheHitEvent], None]
    ) -> EventSubscriber:
        """Subscribe to cache hit events.

        Args:
            handler: Callback function

        Returns:
            Self for chaining
        """

        handler_obj = self._wrap_handler(handler)
        sub = self.event_bus.subscribe(events.CacheHitEvent, handler_obj)
        self._subscriptions.append(sub)
        logger.debug("Subscribed to cache hit events")
        return self

    def on_processor_started(
        self, handler: Callable[[events.ProcessorExecutionStartedEvent], None]
    ) -> EventSubscriber:
        """Subscribe to processor execution started events.

        Args:
            handler: Callback function

        Returns:
            Self for chaining
        """

        handler_obj = self._wrap_handler(handler)
        sub = self.event_bus.subscribe(
            events.ProcessorExecutionStartedEvent, handler_obj
        )
        self._subscriptions.append(sub)
        logger.debug("Subscribed to processor execution started events")
        return self

class SystemConfiguration:
    """Configuration for the analysis system.

    Controls behavior of all integrated patterns.
    """

    # This class is intentionally a compact data holder for configuration
    # values used across the integrated analysis system. It is acceptable
    # to have several instance attributes and few public methods here.

    def __init__(self) -> None:
        """Initialize configuration."""
        self.max_retries: int = 3
        self.retry_delay: float = 1.0
        self.circuit_breaker_threshold: int = 5
        self.circuit_breaker_timeout: int = 60
        self.enable_event_bus: bool = True
        self.enable_circuit_breaker: bool = True
        self.enable_retry: bool = True
        self.enable_dependency_injection: bool = True

    def __repr__(self) -> str:
        return (
            f"SystemConfiguration("
            f"retries={self.max_retries}, "
            f"circuit_breaker={self.enable_circuit_breaker}, "
            f"retry={self.enable_retry}, "
            f"di={self.enable_dependency_injection})"
        )

class AnalysisSystem:
    """Integrated analysis system with all patterns enabled.

    Provides a high-level API for creating and configuring analysis
    components with all design patterns pre-integrated.

    Integrated Patterns:
      - Dependency Injection: Service resolution
      - Event Bus: Decoupled event handling
      - Circuit Breaker: Fault tolerance
      - Retry: Automatic resilience
      - Observer: Event notifications
      - Builder: Fluent configuration
      - Command: Undo/redo support

    Example:
        >>> system = AnalysisSystem()
        >>> system.configure(max_retries=5)
        >>> system.subscribe_to_events().on_analysis_completed(my_handler)
        >>> analyzer = system.create_analyzer()
        >>> result = analyzer.run_with_command(cache_dir=".cache", domain="depth")
    """

    def __init__(self, configuration: SystemConfiguration | None = None) -> None:
        """Initialize analysis system.

        Args:
            configuration: Optional SystemConfiguration
        """
        self.config = configuration or SystemConfiguration()
        self._container = self._create_container()
        self._factory = ComponentFactory(
            container=self._container,
            event_bus=EventBus(),
        )
        self._event_subscriber = EventSubscriber(self._factory.event_bus)

        logger.info("AnalysisSystem initialized: %s", self.config)

    def _create_container(self) -> Container:
        """Create DI container with configured settings.

        Returns:
            Configured Container
        """
        builder = ServiceContainerBuilder()

        if self.config.enable_dependency_injection:
            builder.with_event_bus()
            builder.with_circuit_breaker_pool()
            builder.with_config_manager()
            builder.with_facies_analyzer()

        return builder.build()

    def configure(
        self,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        circuit_breaker_threshold: int | None = None,
        circuit_breaker_timeout: int | None = None,
        enable_event_bus: bool | None = None,
        enable_circuit_breaker: bool | None = None,
        enable_retry: bool | None = None,
    ) -> AnalysisSystem:
        """Configure system settings.

        Args:
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay
            circuit_breaker_threshold: Failure threshold
            circuit_breaker_timeout: Recovery timeout
            enable_event_bus: Enable event bus
            enable_circuit_breaker: Enable circuit breaker
            enable_retry: Enable retry

        Returns:
            Self for chaining
        """
        if max_retries is not None:
            self.config.max_retries = max_retries

        if retry_delay is not None:
            self.config.retry_delay = retry_delay

        if circuit_breaker_threshold is not None:
            self.config.circuit_breaker_threshold = circuit_breaker_threshold

        if circuit_breaker_timeout is not None:
            self.config.circuit_breaker_timeout = circuit_breaker_timeout

        if enable_event_bus is not None:
            self.config.enable_event_bus = enable_event_bus

        if enable_circuit_breaker is not None:
            self.config.enable_circuit_breaker = enable_circuit_breaker

        if enable_retry is not None:
            self.config.enable_retry = enable_retry

        logger.info("System configured: %s", self.config)
        return self

    def create_analyzer(self) -> IntegratedAnalyzer:
        """Create integrated analyzer with all patterns.

        Returns:
            IntegratedAnalyzer instance
        """
        logger.info("Creating analyzer with all patterns integrated")

        analyzer = create_analyzer_with_patterns(
            event_bus=self._factory.event_bus,
            container=self._container,
            enable_circuit_breaker=self.config.enable_circuit_breaker,
        )

        # Set up DI and event bus integration
        analyzer.set_event_bus(self._factory.event_bus)

        logger.debug("Analyzer created successfully")
        return analyzer

    def subscribe_to_events(self) -> EventSubscriber:
        """Get event subscriber for fluent subscription API.

        Returns:
            EventSubscriber instance
        """
        return self._event_subscriber

    def get_service(self, service_name: str) -> Any:
        """Get a service from the DI container.

        Args:
            service_name: Service name

        Returns:
            Service instance
        """
        return self._container.resolve(service_name)

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        logger.info("Resetting all circuit breakers")
        reset_all_circuit_breakers()

    def get_event_bus(self) -> EventBus:
        """Get the system's event bus.

        Returns:
            EventBus instance
        """
        return self._factory.event_bus

    def get_factory(self) -> ComponentFactory:
        """Get the component factory.

        Returns:
            ComponentFactory instance
        """
        return self._factory

    def __enter__(self) -> AnalysisSystem:
        """Enter context manager."""
        logger.debug("Entering AnalysisSystem context")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        logger.debug("Exiting AnalysisSystem context")
        if exc_type is not None:
            logger.error(
                "Context exit with exception: %s: %s", exc_type.__name__, exc_val
            )

    def __repr__(self) -> str:
        return f"AnalysisSystem({self.config})"
