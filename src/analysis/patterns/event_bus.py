"""Event Bus System for Decoupled Event-Driven Architecture

This module provides a comprehensive event bus system enabling loose coupling
between components through event publishing and subscription patterns.

Patterns Used:
  - Pub/Sub: Publish-subscribe messaging
  - Event: Domain events for system notifications
  - Observer: Event handlers observe bus events
  - Mediator: Bus mediates between publishers and subscribers

Example:
    >>> from src.analysis.patterns.event_bus import EventBus, Event, EventHandler
    >>>
    >>> # Define custom event
    >>> class AnalysisCompleteEvent(Event):
    ...     def __init__(self, result_id: str, status: str):
    ...         super().__init__()
    ...         self.result_id = result_id
    ...         self.status = status
    >>>
    >>> # Define event handler
    >>> class NotificationHandler(EventHandler):
    ...     def handle(self, event: Event):
    ...         print(f"Analysis {event.result_id} completed: {event.status}")
    >>>
    >>> # Use event bus
    >>> bus = EventBus()
    >>> bus.subscribe(AnalysisCompleteEvent, NotificationHandler())
    >>> bus.publish(AnalysisCompleteEvent("task_1", "success"))
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Empty, Full, Queue
from threading import Lock, Thread
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "Event",
    "EventHandler",
    "EventBus",
    "AsyncEventBus",
    "EventDispatcher",
    "EventFilter",
    "EventPriority",
    "SubscriptionHandle",
]

class EventPriority(Enum):
    """Event processing priority levels."""

    CRITICAL = 0  # Process immediately
    HIGH = 1  # Process before normal
    NORMAL = 2  # Standard priority
    LOW = 3  # Process after normal
    DEFERRED = 4  # Process when idle

@dataclass
class Event:
    """Base class for all domain events."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    source: str = field(default="unknown")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.event_id[:8]}..., "
            f"source={self.source})"
        )

class EventHandler(ABC):
    """Abstract base class for event handlers."""

    @abstractmethod
    def handle(self, event: Event) -> None:
        """Handle an event.

        Args:
            event: Event to handle
        """

    @property
    def handler_id(self) -> str:
        """Unique handler identifier.

        Returns:
            Handler ID
        """
        return f"{self.__class__.__name__}_{id(self)}"

class EventFilter:
    """Filters events based on criteria."""

    def __init__(self, **criteria: Any) -> None:
        """Initialize filter with criteria.

        Args:
            **criteria: Filter criteria as kwargs
        """
        self.criteria: dict[str, Any] = criteria

    def matches(self, event: Event) -> bool:
        """Check if event matches filter criteria.

        Args:
            event: Event to check

        Returns:
            True if event matches, False otherwise
        """
        for key, value in self.criteria.items():
            if not hasattr(event, key):
                return False
            if getattr(event, key) != value:
                return False
        return True

class SubscriptionHandle:
    """Handle for managing event subscriptions."""

    def __init__(
        self, bus: EventBus, event_type: type[Event], handler: EventHandler
    ) -> None:
        """Initialize subscription handle.

        Args:
            bus: Event bus instance
            event_type: Event type subscribed to
            handler: Event handler
        """
        self.bus = bus
        self.event_type = event_type
        self.handler = handler

    def unsubscribe(self) -> bool:
        """Unsubscribe from events.

        Returns:
            True if unsubscribed, False if not found
        """
        return self.bus.unsubscribe(self.event_type, self.handler)

class EventBus:
    """Synchronous event bus for event-driven architecture."""

    def __init__(self) -> None:
        """Initialize event bus."""
        self._handlers: dict[
            type[Event], list[tuple[EventHandler, EventFilter | None, EventPriority]]
        ] = {}
        self._lock = Lock()
        self._event_history: list[Event] = []
        self._max_history = 1000
        self._middleware: list[Callable[[Event], bool]] = []
        logger.info("EventBus initialized (synchronous)")

    def subscribe(
        self,
        event_type: type[Event],
        handler: EventHandler,
        filter_fn: EventFilter | None = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> SubscriptionHandle:
        """Subscribe to an event type.

        Args:
            event_type: Event class to subscribe to
            handler: Handler to call on events
            filter_fn: Optional filter for selective handling
            priority: Handler execution priority

        Returns:
            Subscription handle for managing subscription
        """
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []

            self._handlers[event_type].append((handler, filter_fn, priority))

            # Sort by priority
            self._handlers[event_type].sort(key=lambda x: x[2].value)

        logger.debug("Subscribed %s to %s", handler.handler_id, event_type.__name__)

        return SubscriptionHandle(self, event_type, handler)

    def unsubscribe(
        self,
        event_type: type[Event],
        handler: EventHandler,
    ) -> bool:
        """Unsubscribe from an event type.

        Args:
            event_type: Event class
            handler: Handler to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if event_type not in self._handlers:
                return False

            original_count = len(self._handlers[event_type])
            self._handlers[event_type] = [
                (h, f, p) for h, f, p in self._handlers[event_type] if h is not handler
            ]

            if not self._handlers[event_type]:
                del self._handlers[event_type]

            removed = len(self._handlers.get(event_type, [])) < original_count

        if removed:
            logger.debug(
                "Unsubscribed %s from %s", handler.handler_id, event_type.__name__
            )

        return removed

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        # Run middleware
        for middleware in self._middleware:
            if not middleware(event):
                logger.debug("Event blocked by middleware: %s", event)
                return

        # Store in history
        self._add_to_history(event)

        # Get handlers (copy to avoid lock during iteration)
        with self._lock:
            handlers = self._handlers.get(type(event), []).copy()

        if not handlers:
            logger.debug("No handlers for event: %s", type(event).__name__)
            return

        # Call handlers
        for handler, filter_fn, _ in handlers:
            try:
                if filter_fn is None or filter_fn.matches(event):
                    handler.handle(event)
                    logger.debug(
                        "Handled %s by %s", type(event).__name__, handler.handler_id
                    )
            except Exception as e:
                # Handler exceptions should not stop the event bus; log and
                # continue to allow other handlers to receive the event.
                logger.error(
                    "Error handling %s in %s: %s",
                    type(event).__name__,
                    handler.handler_id,
                    e,
                )

    def add_middleware(self, middleware: Callable[[Event], bool]) -> EventBus:
        """Add middleware to filter/modify events.

        Args:
            middleware: Function that returns True to allow event, False to block

        Returns:
            Self for chaining
        """
        self._middleware.append(middleware)
        logger.debug("Added event middleware")
        return self

    def _add_to_history(self, event: Event) -> None:
        """Add event to history.

        Args:
            event: Event to record
        """
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history :]

    def get_history(self, event_type: type[Event] | None = None) -> list[Event]:
        """Get event history.

        Args:
            event_type: Optional filter by event type

        Returns:
            List of historical events
        """
        with self._lock:
            history = self._event_history.copy()

        if event_type:
            history = [e for e in history if isinstance(e, event_type)]

        return history

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()
        logger.info("Event history cleared")

    def __repr__(self) -> str:
        return (
            f"EventBus("
            f"handlers={len(self._handlers)}, "
            f"history={len(self._event_history)})"
        )

class AsyncEventBus(EventBus):
    """Asynchronous event bus for non-blocking event processing."""

    def __init__(self, worker_threads: int = 2) -> None:
        """Initialize async event bus.

        Args:
            worker_threads: Number of worker threads
        """
        super().__init__()
        self._queue: Queue[Event | None] = Queue()
        self._workers: list[Thread] = []
        self._running: bool = True
        self._stop_event: bool = False

        # Start worker threads
        for _ in range(worker_threads):
            worker = Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)

        logger.info("AsyncEventBus initialized with %s workers", worker_threads)

    def publish(self, event: Event) -> None:
        """Publish event asynchronously.

        Args:
            event: Event to publish
        """
        if not self._running:
            logger.warning("EventBus not running, dropping event")
            return

        self._queue.put(event)
        logger.debug("Queued event: %s", type(event).__name__)

    def _worker_loop(self) -> None:
        """Worker thread loop for processing events."""
        while self._running and not self._stop_event:
            try:
                event: Event | None = self._queue.get(timeout=0.5)
                if event is None:  # Sentinel value to stop
                    break
                super().publish(event)
                self._queue.task_done()
            except Empty:
                continue
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                if self._running:
                    logger.debug("Worker loop: %s", e)

    def stop(self) -> None:
        """Stop the event bus and worker threads."""
        self._running = False
        self._stop_event = True

        # Send sentinel values to wake up workers
        for _ in self._workers:
            try:
                self._queue.put(None, timeout=1)
            except Full:
                pass

        # Wait for workers to finish
        for worker in self._workers:
            if worker.is_alive():
                worker.join(timeout=2)

        logger.info("AsyncEventBus stopped")

    def __enter__(self) -> AsyncEventBus:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.stop()

class EventDispatcher:
    """Dispatches events with customizable routing."""

    def __init__(self, bus: EventBus):
        """Initialize dispatcher.

        Args:
            bus: Event bus instance
        """
        self.bus = bus
        self._routes: dict[type[Event], Callable[[Event], None]] = {}

    def map_event(
        self,
        event_type: type[Event],
        handler: Callable[[Event], None],
    ) -> EventDispatcher:
        """Map an event type to a handler function.

        Args:
            event_type: Event class
            handler: Handler function

        Returns:
            Self for chaining
        """
        self._routes[event_type] = handler
        return self

    def dispatch(self, event: Event) -> None:
        """Dispatch event using route mapping.

        Args:
            event: Event to dispatch
        """
        event_type = type(event)

        if event_type in self._routes:
            handler = self._routes[event_type]
            try:
                handler(event)
            except Exception as e:
                # Route handlers are application-level; swallow and log
                # unexpected errors to avoid crashing dispatch.
                logger.error("Error dispatching %s: %s", event_type.__name__, e)
        else:
            logger.warning("No route for event: %s", event_type.__name__)
