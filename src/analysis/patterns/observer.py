"""Observer Pattern Implementation for Event-Driven Analysis

This module provides an event-driven architecture for seismic analysis,
allowing components to publish results and notify interested observers
without tight coupling.

Patterns Used:
  - Observer: Decoupled event notification
  - Mixin: Observable behavior for any class

Example:
    >>> from src.analysis.patterns.observer import Observable, AnalysisObserver
    >>>
    >>> class ResultPrinter(AnalysisObserver):
    ...     def on_result_computed(self, result_type, result):
    ...         print(f"Result: {result_type} = {result}")
    ...
    ...     def on_data_changed(self, data_type, new_data):
    ...         print(f"Data changed: {data_type}")
    ...
    ...     def on_error(self, error, context):
    ...         print(f"Error in {context}: {error}")
    >>>
    >>> analyzer = FaciesCorrelationAnalyzer()
    >>> analyzer.attach(ResultPrinter())
    >>> analyzer.run(data)  # Automatically notifies observer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisObserver",
    "Observable",
    "EventType",
    "AnalysisEvent",
    "ProgressObserver",
    "LoggingObserver",
]


class EventType:
    """Event type constants for analysis notifications"""

    # Computation events
    COMPUTATION_STARTED = "computation_started"
    COMPUTATION_COMPLETED = "computation_completed"
    COMPUTATION_FAILED = "computation_failed"

    # Data events
    DATA_LOADED = "data_loaded"
    DATA_CHANGED = "data_changed"
    DATA_VALIDATED = "data_validated"

    # Result events
    RESULT_COMPUTED = "result_computed"
    RESULT_CACHED = "result_cached"
    RESULT_EXPORTED = "result_exported"

    # Progress events
    PROGRESS_UPDATE = "progress_update"
    PROGRESS_COMPLETE = "progress_complete"

    # Error events
    ERROR_OCCURRED = "error_occurred"
    WARNING_ISSUED = "warning_issued"


class AnalysisEvent:
    """Encapsulates an analysis event with metadata"""

    def __init__(
        self,
        event_type: str,
        source: str,
        data: Any = None,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an analysis event.

        Args:
            event_type: Type of event (from EventType constants)
            source: Component that triggered the event
            data: Event data/payload
            timestamp: Event timestamp (auto-generated if not provided)
            context: Additional context information
        """
        import time

        self.event_type = event_type
        self.source = source
        self.data = data
        self.timestamp = timestamp or time.time()
        self.context = context or {}

    def __repr__(self) -> str:
        return (
            f"AnalysisEvent(type={self.event_type!r}, "
            f"source={self.source!r}, timestamp={self.timestamp})"
        )


class AnalysisObserver(ABC):
    """Abstract base class for analysis event observers.

    Observers can be attached to Observable components to receive
    notifications about analysis events without tight coupling.
    """

    @abstractmethod
    def on_result_computed(self, event: AnalysisEvent):
        """Called when an analysis result is computed.

        Args:
            event: AnalysisEvent with result data
        """
        pass

    @abstractmethod
    def on_data_changed(self, event: AnalysisEvent):
        """Called when input data changes.

        Args:
            event: AnalysisEvent with new data
        """
        pass

    @abstractmethod
    def on_error(self, event: AnalysisEvent):
        """Called when an error occurs during analysis.

        Args:
            event: AnalysisEvent with error information
        """
        pass

    def on_progress(self, event: AnalysisEvent):
        """Called to report progress updates (optional).

        Args:
            event: AnalysisEvent with progress data
        """
        pass


class Observable:
    """Mixin to add observable behavior to any analysis component.

    Allows attaching observers that receive notifications about
    analysis events, enabling decoupled event-driven architecture.

    Example:
        >>> class Analyzer(Observable):
        ...     def run(self, data):
        ...         self._notify(EventType.COMPUTATION_STARTED, self.__class__.__name__)
        ...         result = self._compute(data)
        ...         self._notify(EventType.RESULT_COMPUTED, result)
        ...         return result
        ...
        ...     def _notify(self, event_type: str, data: Any):
        ...         event = AnalysisEvent(event_type, self.__class__.__name__, data)
        ...         self.notify_observers(event)
    """

    def __init__(self):
        """Initialize observable with empty observer list"""
        self._observers: List[AnalysisObserver] = []

    def attach(self, observer: AnalysisObserver) -> AnalysisObserver:
        """Attach an observer to receive notifications.

        Args:
            observer: Observer instance to attach

        Returns:
            The observer (for chaining)
        """
        if observer not in self._observers:
            self._observers.append(observer)
            logger.debug(
                f"Attached observer {observer.__class__.__name__} to "
                f"{self.__class__.__name__}"
            )
        return observer

    def detach(self, observer: AnalysisObserver) -> bool:
        """Detach an observer from notifications.

        Args:
            observer: Observer instance to detach

        Returns:
            True if observer was attached and detached, False otherwise
        """
        if observer in self._observers:
            self._observers.remove(observer)
            logger.debug(
                f"Detached observer {observer.__class__.__name__} from "
                f"{self.__class__.__name__}"
            )
            return True
        return False

    def notify_observers(self, event: AnalysisEvent):
        """Notify all observers of an event.

        Args:
            event: AnalysisEvent to broadcast to all observers
        """
        logger.debug(f"Notifying {len(self._observers)} observers: {event}")

        for observer in self._observers:
            try:
                if event.event_type == EventType.RESULT_COMPUTED:
                    observer.on_result_computed(event)
                elif event.event_type in (
                    EventType.DATA_CHANGED,
                    EventType.DATA_LOADED,
                ):
                    observer.on_data_changed(event)
                elif event.event_type in (
                    EventType.ERROR_OCCURRED,
                    EventType.WARNING_ISSUED,
                ):
                    observer.on_error(event)
                elif event.event_type == EventType.PROGRESS_UPDATE:
                    observer.on_progress(event)
            except Exception as e:
                logger.error(
                    f"Error notifying observer {observer.__class__.__name__}: {e}",
                    exc_info=True,
                )

    def clear_observers(self):
        """Remove all attached observers"""
        self._observers.clear()
        logger.debug(f"Cleared all observers from {self.__class__.__name__}")

    @property
    def observer_count(self) -> int:
        """Get the number of attached observers"""
        return len(self._observers)


class ProgressObserver(AnalysisObserver):
    """Observer that tracks and reports progress updates.

    Useful for long-running analyses to show progress to users.
    """

    def __init__(self, name: str = "Analysis"):
        """Initialize progress observer.

        Args:
            name: Name for this analysis (for display)
        """
        self.name = name
        self.progress = 0.0
        self.message = ""

    def on_result_computed(self, event: AnalysisEvent):
        """Handle result computed event"""
        logger.info(f"{self.name}: Result computed - {event.source}")

    def on_data_changed(self, event: AnalysisEvent):
        """Handle data changed event"""
        logger.info(f"{self.name}: Data changed - {event.source}")

    def on_error(self, event: AnalysisEvent):
        """Handle error event"""
        error = event.data
        logger.error(f"{self.name}: Error in {event.source} - {error}")

    def on_progress(self, event: AnalysisEvent):
        """Handle progress update event"""
        if "progress" in event.context:
            self.progress = event.context["progress"]
        if "message" in event.context:
            self.message = event.context["message"]

        logger.info(f"{self.name}: {self.message} ({self.progress*100:.1f}%)")


class LoggingObserver(AnalysisObserver):
    """Observer that logs all analysis events.

    Useful for debugging and tracing analysis execution flow.
    """

    def __init__(self, level: int = logging.INFO):
        """Initialize logging observer.

        Args:
            level: Logging level for events (default: INFO)
        """
        self.level = level
        self.logger = logging.getLogger(self.__class__.__name__)

    def on_result_computed(self, event: AnalysisEvent):
        """Log result computed event"""
        self.logger.log(
            self.level,
            f"Result computed by {event.source}: {type(event.data).__name__}",
        )

    def on_data_changed(self, event: AnalysisEvent):
        """Log data changed event"""
        self.logger.log(
            self.level, f"Data changed in {event.source}: {type(event.data).__name__}"
        )

    def on_error(self, event: AnalysisEvent):
        """Log error event"""
        self.logger.error(
            f"Error in {event.source}: {event.data}",
            exc_info=isinstance(event.data, Exception),
        )

    def on_progress(self, event: AnalysisEvent):
        """Log progress event"""
        msg = event.context.get("message", "Progress update")
        progress = event.context.get("progress", 0)
        self.logger.log(self.level, f"Progress: {msg} ({progress*100:.1f}%)")
