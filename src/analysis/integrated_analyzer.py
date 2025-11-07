"""Integrated Analysis Framework combining Observer, Builder, and Command patterns.

This module provides enhanced analysis capabilities by integrating multiple
design patterns (Observer, Builder, Command, DependencyInjection, EventBus,
CircuitBreaker, Retry) into a cohesive framework.

Integration Points:
  - Observer Pattern: Event notifications for analysis lifecycle
  - Builder Pattern: Fluent configuration API
  - Command Pattern: Undo/redo and operation history
  - Dependency Injection: Service resolution and loose coupling
  - Event Bus: Decoupled event handling
  - Circuit Breaker: Fault tolerance
  - Retry: Automatic resilience

Example:
    >>> from src.analysis.integrated_analyzer import IntegratedAnalyzer
    >>> from src.analysis.patterns.observer import ProgressObserver, LoggingObserver
    >>> from src.analysis.patterns.event_bus import EventBus
    >>>
    >>> # Create analyzer with builder
    >>> analyzer = IntegratedAnalyzer()
    >>>
    >>> # Attach observers for event notifications
    >>> progress = ProgressObserver()
    >>> logger_obs = LoggingObserver()
    >>> analyzer.attach(progress)
    >>> analyzer.attach(logger_obs)
    >>>
    >>> # Create event bus for decoupled event handling
    >>> event_bus = EventBus()
    >>> analyzer.set_event_bus(event_bus)
    >>>
    >>> # Run analysis with command support
    >>> result = analyzer.run_with_command(cache_dir=".cache", domain="depth")
    >>>
    >>> # Access command history
    >>> analyzer.undo()
    >>> analyzer.redo()
    >>> print(analyzer.command_history)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List, Type, Callable
from dataclasses import dataclass, field

from src.analysis.facies.analyzer import FaciesCorrelationAnalyzer
from src.analysis.facies.config import FaciesAnalysisConfig
from src.analysis.patterns.observer import (
    Observable,
    AnalysisObserver,
    AnalysisEvent,
    EventType,
)
from src.analysis.patterns.builder import FaciesAnalyzerBuilder, AnalysisBuilderBase
from src.analysis.patterns.command import (
    CommandQueue,
    RunAnalysisCommand,
    AnalysisCommand,
)
from src.analysis.patterns.event_bus import EventBus, Event, EventHandler
from src.analysis.patterns.dependency_injection import Container, ServiceProvider
from src.analysis.patterns.circuit_breaker import CircuitBreaker, circuit_breaker
from src.analysis.patterns.retry import retry
from src.analysis import events

logger = logging.getLogger(__name__)

__all__ = [
    "IntegratedAnalyzer",
    "AnalysisContext",
    "AnalysisOperation",
]


@dataclass
class AnalysisContext:
    """Context information for an analysis operation.

    Captures all parameters needed to recreate an analysis operation
    for undo/redo support.
    """

    cache_dir: str
    domain: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "cache_dir": self.cache_dir,
            "domain": self.domain,
            "parameters": self.parameters.copy(),
        }


class AnalysisOperation(AnalysisCommand):
    """Operation that encapsulates a single analysis run.

    Used by IntegratedAnalyzer to manage analysis operations
    with undo/redo support.
    """

    def __init__(
        self,
        analyzer: IntegratedAnalyzer,
        context: AnalysisContext,
    ):
        """Initialize analysis operation.

        Args:
            analyzer: IntegratedAnalyzer instance
            context: Analysis context with parameters
        """
        super().__init__()
        self.analyzer = analyzer
        self.context = context
        self.result = None
        self.previous_results = None

    def execute(self) -> Any:
        """Execute the analysis operation.

        Returns:
            Analysis result
        """
        logger.info(f"Executing analysis: {self.description}")

        # Notify observers of operation start
        self.analyzer._notify_event(
            EventType.COMPUTATION_STARTED,
            {"context": self.context.to_dict()},
        )

        try:
            # Save previous results for undo
            if hasattr(self.analyzer._facies_analyzer, "_results"):
                self.previous_results = self.analyzer._facies_analyzer._results

            # Run the underlying analyzer
            self.result = self.analyzer._facies_analyzer.run(
                cache_dir=self.context.cache_dir,
                domain=self.context.domain,
                **self.context.parameters,
            )

            self.executed = True

            # Notify observers of result
            self.analyzer._notify_event(
                EventType.RESULT_COMPUTED,
                {"result": str(self.result), "context": self.context.to_dict()},
            )

            logger.debug(f"Analysis execution succeeded: {self.description}")
            return self.result

        except Exception as e:
            logger.error(f"Analysis execution failed: {e}")

            # Notify observers of error
            self.analyzer._notify_event(
                EventType.ERROR_OCCURRED,
                {"error": str(e), "context": self.context.to_dict()},
            )

            raise

    def undo(self) -> bool:
        """Undo the analysis operation.

        Returns:
            True if undo successful
        """
        logger.info(f"Undoing analysis: {self.description}")

        try:
            # Restore previous results
            if self.previous_results is not None and hasattr(
                self.analyzer._facies_analyzer, "_results"
            ):
                self.analyzer._facies_analyzer._results = self.previous_results

            self.result = None
            self.executed = False

            self.analyzer._notify_event(
                EventType.DATA_CHANGED,
                {"context": self.context.to_dict()},
            )

            logger.debug(f"Analysis undo succeeded: {self.description}")
            return True

        except Exception as e:
            logger.error(f"Analysis undo failed: {e}")
            return False

    def redo(self) -> Any:
        """Redo the analysis operation.

        Returns:
            Analysis result
        """
        logger.info(f"Redoing analysis: {self.description}")
        return self.execute()

    @property
    def description(self) -> str:
        """Get operation description."""
        return (
            f"Analysis(domain={self.context.domain}, "
            f"cache_dir={self.context.cache_dir})"
        )


class IntegratedAnalyzer(Observable):
    """Enhanced FaciesCorrelationAnalyzer with integrated design patterns.

    Combines multiple design patterns to provide:
    - Event notifications for all analysis lifecycle events
    - Fluent builder API for analyzer configuration
    - Command history with undo/redo support
    - Dependency injection for loose coupling
    - Event bus for decoupled event handling
    - Circuit breaker for fault tolerance
    - Retry logic for resilience

    This class wraps FaciesCorrelationAnalyzer to add design pattern support.
    """

    def __init__(
        self,
        config: Optional[FaciesAnalysisConfig] = None,
        facies_analyzer: Optional[FaciesCorrelationAnalyzer] = None,
        max_command_history: int = 100,
        event_bus: Optional[EventBus] = None,
        container: Optional[Container] = None,
        service_provider: Optional[ServiceProvider] = None,
    ):
        """Initialize integrated analyzer.

        Args:
            config: Optional FaciesAnalysisConfig
            facies_analyzer: Optional pre-configured FaciesCorrelationAnalyzer
            max_command_history: Maximum commands in history
            event_bus: Optional EventBus for decoupled event handling
            container: Optional DI Container
            service_provider: Optional ServiceProvider for service resolution
        """
        super().__init__()

        self._config = config or FaciesAnalysisConfig()
        self._facies_analyzer = facies_analyzer or FaciesCorrelationAnalyzer(
            config=self._config
        )
        self._command_queue = CommandQueue(max_history=max_command_history)

        # Pattern integrations
        self._event_bus = event_bus or EventBus()
        self._container = container
        self._service_provider = service_provider
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        logger.info("IntegratedAnalyzer initialized with all pattern support")

    @classmethod
    def from_builder(cls) -> IntegratedAnalyzer:
        """Create integrated analyzer using fluent builder pattern.

        Returns:
            IntegratedAnalyzer instance configured via builder
        """
        logger.info("Creating IntegratedAnalyzer from builder")

        # Use the FaciesAnalyzerBuilder to configure the underlying analyzer
        facies_builder = FaciesAnalyzerBuilder()
        facies_analyzer = facies_builder.build()

        return cls(facies_analyzer=facies_analyzer)

    def run(
        self,
        cache_dir: str,
        domain: str,
        **parameters: Any,
    ) -> Any:
        """Run analysis without command support (direct execution).

        Args:
            cache_dir: Cache directory path
            domain: Analysis domain
            **parameters: Additional analysis parameters

        Returns:
            Analysis result
        """
        logger.info(f"Running analysis directly (no command history)")

        self._notify_event(
            EventType.COMPUTATION_STARTED,
            {"cache_dir": cache_dir, "domain": domain},
        )

        try:
            result = self._facies_analyzer.run(
                cache_dir=cache_dir,
                domain=domain,
                **parameters,
            )

            self._notify_event(
                EventType.RESULT_COMPUTED,
                {"domain": domain},
            )

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            self._notify_event(
                EventType.ERROR_OCCURRED,
                {"error": str(e)},
            )
            raise

    def run_with_command(
        self,
        cache_dir: str,
        domain: str,
        **parameters: Any,
    ) -> Any:
        """Run analysis with command support (enables undo/redo).

        This method creates a command for the analysis, executes it,
        and stores it in the command history for undo/redo support.

        Args:
            cache_dir: Cache directory path
            domain: Analysis domain
            **parameters: Additional analysis parameters

        Returns:
            Analysis result
        """
        logger.info(f"Running analysis with command support")

        # Create context and command
        context = AnalysisContext(
            cache_dir=cache_dir,
            domain=domain,
            parameters=parameters,
        )

        command = AnalysisOperation(self, context)

        # Execute command (stored in history)
        return self._command_queue.execute(command)

    def undo(self) -> bool:
        """Undo the last analysis command.

        Returns:
            True if undo successful
        """
        logger.info("Requesting undo operation")

        if not self._command_queue.can_undo:
            logger.warning("Nothing to undo")
            self._notify_event(
                EventType.ERROR_OCCURRED,
                {"error": "Nothing to undo"},
            )
            return False

        success = self._command_queue.undo()

        if success:
            self._notify_event(
                EventType.DATA_CHANGED,
                {},
            )

        return success

    def redo(self) -> bool:
        """Redo the last undone analysis command.

        Returns:
            True if redo successful
        """
        logger.info("Requesting redo operation")

        if not self._command_queue.can_redo:
            logger.warning("Nothing to redo")
            self._notify_event(
                EventType.ERROR_OCCURRED,
                {"error": "Nothing to redo"},
            )
            return False

        success = self._command_queue.redo()

        if success:
            self._notify_event(
                EventType.DATA_CHANGED,
                {},
            )

        return success

    def _notify_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ):
        """Notify observers of an analysis event.

        Args:
            event_type: Type of event
            data: Event data/payload
            context: Additional context
        """
        event = AnalysisEvent(
            event_type=event_type,
            source=self.__class__.__name__,
            data=data,
            context=context,
        )

        self.notify_observers(event)

    @property
    def command_history(self) -> str:
        """Get formatted command history summary.

        Returns:
            Command history as formatted string
        """
        return self._command_queue.get_history_summary()

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._command_queue.can_undo

    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._command_queue.can_redo

    @property
    def observer_count(self) -> int:
        """Get number of attached observers."""
        return len(self._observers)

    def clear_history(self):
        """Clear all command history."""
        self._command_queue.clear()
        logger.info("Command history cleared")

    def set_event_bus(self, event_bus: EventBus) -> IntegratedAnalyzer:
        """Set the event bus for decoupled event handling.

        Args:
            event_bus: EventBus instance

        Returns:
            Self for chaining
        """
        self._event_bus = event_bus
        logger.info("Event bus set for analyzer")
        return self

    def subscribe_to_event(
        self,
        event_type: str,
        handler: Callable[[Event], None],
    ) -> Any:
        """Subscribe to events from the event bus.

        Args:
            event_type: Type of event to subscribe to
            handler: Handler function to call on event

        Returns:
            Subscription handle
        """
        logger.debug(f"Subscribing to event type: {event_type}")
        return self._event_bus.subscribe(event_type, handler)

    def publish_event(self, event: Event) -> None:
        """Publish an event to the event bus.

        Args:
            event: Event to publish
        """
        logger.debug(f"Publishing event: {type(event).__name__}")
        self._event_bus.publish(event)

    def get_service(self, service_name: str) -> Any:
        """Get a service from the dependency injection container.

        Args:
            service_name: Name of service to resolve

        Returns:
            Service instance

        Raises:
            ResolutionError: If service not found
        """
        if self._service_provider is None:
            raise RuntimeError("Service provider not configured")

        logger.debug(f"Resolving service: {service_name}")
        return self._service_provider.resolve(service_name)

    def try_get_service(self, service_name: str, default: Any = None) -> Any:
        """Try to get a service from the DI container.

        Args:
            service_name: Name of service to resolve
            default: Default value if service not found

        Returns:
            Service instance or default
        """
        if self._service_provider is None:
            logger.warning("Service provider not configured")
            return default

        logger.debug(f"Trying to resolve service: {service_name}")
        return self._service_provider.try_resolve(service_name, default)

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a specific operation.

        Args:
            name: Name of the circuit breaker

        Returns:
            CircuitBreaker instance
        """
        if name not in self._circuit_breakers:
            logger.debug(f"Creating circuit breaker: {name}")
            self._circuit_breakers[name] = CircuitBreaker(name=name)

        return self._circuit_breakers[name]

    def clear_circuit_breakers(self) -> None:
        """Clear all circuit breakers."""
        self._circuit_breakers.clear()
        logger.info("Circuit breakers cleared")

    def __enter__(self) -> IntegratedAnalyzer:
        """Enter context manager."""
        logger.debug("Entering IntegratedAnalyzer context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        logger.debug("Exiting IntegratedAnalyzer context")
        if exc_type is not None:
            logger.error(f"Context exit with exception: {exc_type.__name__}: {exc_val}")

    def __repr__(self) -> str:
        return (
            f"IntegratedAnalyzer("
            f"observers={self.observer_count}, "
            f"history_position={self._command_queue.current_index + 1}/"
            f"{len(self._command_queue.history)}, "
            f"circuit_breakers={len(self._circuit_breakers)})"
        )
