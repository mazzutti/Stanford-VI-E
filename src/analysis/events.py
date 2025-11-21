"""Application-specific events for the event bus integration.

This module defines all events that can be published through the EventBus,
providing a centralized location for event definitions used across the
analysis framework.

Event Categories:
  - Analysis Events: Analysis lifecycle and results
  - Cache Events: Cache operations and invalidations
  - Processor Events: Processor execution and completion
  - Error Events: Error conditions and failures
  - Configuration Events: Configuration changes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, cast

from src.analysis.patterns.event_bus import Event

__all__ = [
    "AnalysisEventType",
    "CacheEventType",
    "ProcessorEventType",
    "ConfigurationEventType",
    "ErrorEventType",
    "AnalysisStartedEvent",
    "AnalysisCompletedEvent",
    "AnalysisFailedEvent",
    "CacheHitEvent",
    "CacheMissEvent",
    "CacheInvalidatedEvent",
    "ProcessorExecutionStartedEvent",
    "ProcessorExecutionCompletedEvent",
    "ProcessorExecutionFailedEvent",
    "ConfigurationChangedEvent",
    "ErrorOccurredEvent",
]

# Events are small dataclasses used by the EventBus; keep them compact

class AnalysisEventType(Enum):
    """Analysis event types."""

    STARTED = "analysis.started"
    COMPLETED = "analysis.completed"
    FAILED = "analysis.failed"
    PAUSED = "analysis.paused"
    RESUMED = "analysis.resumed"
    PROGRESS = "analysis.progress"

class CacheEventType(Enum):
    """Cache event types."""

    HIT = "cache.hit"
    MISS = "cache.miss"
    INVALIDATED = "cache.invalidated"
    CLEARED = "cache.cleared"

class ProcessorEventType(Enum):
    """Processor event types."""

    EXECUTION_STARTED = "processor.execution_started"
    EXECUTION_COMPLETED = "processor.execution_completed"
    EXECUTION_FAILED = "processor.execution_failed"

class ConfigurationEventType(Enum):
    """Configuration event types."""

    CHANGED = "configuration.changed"
    RELOADED = "configuration.reloaded"
    VALIDATED = "configuration.validated"

class ErrorEventType(Enum):
    """Error event types."""

    OCCURRED = "error.occurred"
    RECOVERED = "error.recovered"
    CRITICAL = "error.critical"

@dataclass
class AnalysisStartedEvent(Event):
    """Event fired when analysis starts."""

    analysis_type: str = ""
    domain: str = ""
    cache_dir: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Initialize event metadata."""
        self.event_type = AnalysisEventType.STARTED.value
        self.priority = 50

@dataclass
class AnalysisCompletedEvent(Event):
    """Event fired when analysis completes successfully."""

    analysis_type: str = ""
    domain: str = ""
    result_summary: str = ""
    execution_time_seconds: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Initialize event metadata."""
        self.event_type = AnalysisEventType.COMPLETED.value
        self.priority = 50

@dataclass
class AnalysisFailedEvent(Event):
    """Event fired when analysis fails.

    This dataclass intentionally contains several fields used by the
    event-bus framework; suppress the instance-attribute count warning.
    """

    analysis_type: str = ""
    domain: str = ""
    error: str = ""
    error_type: str = ""
    traceback: str | None = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Initialize event metadata."""
        self.event_type = AnalysisEventType.FAILED.value
        self.priority = 100

@dataclass
class CacheHitEvent(Event):
    """Event fired on cache hit."""

    key: str = ""
    cache_type: str = ""
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Initialize event metadata."""
        self.event_type = CacheEventType.HIT.value
        self.priority = 20

@dataclass
class CacheMissEvent(Event):
    """Event fired on cache miss."""

    key: str = ""
    cache_type: str = ""
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Initialize event metadata."""
        self.event_type = CacheEventType.MISS.value
        self.priority = 20

@dataclass
class CacheInvalidatedEvent(Event):
    """Event fired when cache is invalidated."""

    keys: list[str] = field(default_factory=lambda: cast(list[str], []))
    cache_type: str = ""
    reason: str = ""
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Initialize event metadata."""
        self.event_type = CacheEventType.INVALIDATED.value
        self.priority = 60

@dataclass
class ProcessorExecutionStartedEvent(Event):
    """Event fired when processor execution starts."""

    processor_name: str = ""
    processor_type: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Initialize event metadata."""
        self.event_type = ProcessorEventType.EXECUTION_STARTED.value
        self.priority = 30

@dataclass
class ProcessorExecutionCompletedEvent(Event):
    """Event fired when processor execution completes."""

    processor_name: str = ""
    processor_type: str = ""
    result_type: str = ""
    execution_time_seconds: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Initialize event metadata."""
        self.event_type = ProcessorEventType.EXECUTION_COMPLETED.value
        self.priority = 30

@dataclass
class ProcessorExecutionFailedEvent(Event):
    """Event fired when processor execution fails."""

    processor_name: str = ""
    processor_type: str = ""
    error: str = ""
    error_type: str = ""
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Initialize event metadata."""
        self.event_type = ProcessorEventType.EXECUTION_FAILED.value
        self.priority = 90

@dataclass
class ConfigurationChangedEvent(Event):
    """Event fired when configuration changes."""

    config_name: str = ""
    changed_keys: list[str] = field(default_factory=lambda: cast(list[str], []))
    old_values: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))
    new_values: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Initialize event metadata."""
        self.event_type = ConfigurationEventType.CHANGED.value
        self.priority = 40

@dataclass
class ErrorOccurredEvent(Event):
    # This dataclass intentionally contains several fields used by the
    # event-bus framework; suppress the instance-attribute count warning.

    """Event fired when an error occurs."""

    error_message: str = ""
    error_type: str = ""
    source: str = ""
    is_critical: bool = False
    context: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Initialize event metadata."""
        self.event_type = (
            ErrorEventType.CRITICAL.value
            if self.is_critical
            else ErrorEventType.OCCURRED.value
        )
        self.priority = 100 if self.is_critical else 80
