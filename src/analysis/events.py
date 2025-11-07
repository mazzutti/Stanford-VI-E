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
from typing import Any, Dict, Optional
from enum import Enum
from datetime import datetime

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
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
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
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize event metadata."""
        self.event_type = AnalysisEventType.COMPLETED.value
        self.priority = 50


@dataclass
class AnalysisFailedEvent(Event):
    """Event fired when analysis fails."""

    analysis_type: str = ""
    domain: str = ""
    error: str = ""
    error_type: str = ""
    traceback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize event metadata."""
        self.event_type = AnalysisEventType.FAILED.value
        self.priority = 100


@dataclass
class CacheHitEvent(Event):
    """Event fired on cache hit."""

    key: str = ""
    cache_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize event metadata."""
        self.event_type = CacheEventType.HIT.value
        self.priority = 20


@dataclass
class CacheMissEvent(Event):
    """Event fired on cache miss."""

    key: str = ""
    cache_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize event metadata."""
        self.event_type = CacheEventType.MISS.value
        self.priority = 20


@dataclass
class CacheInvalidatedEvent(Event):
    """Event fired when cache is invalidated."""

    keys: list = field(default_factory=list)
    cache_type: str = ""
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize event metadata."""
        self.event_type = CacheEventType.INVALIDATED.value
        self.priority = 60


@dataclass
class ProcessorExecutionStartedEvent(Event):
    """Event fired when processor execution starts."""

    processor_name: str = ""
    processor_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
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
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
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
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize event metadata."""
        self.event_type = ProcessorEventType.EXECUTION_FAILED.value
        self.priority = 90


@dataclass
class ConfigurationChangedEvent(Event):
    """Event fired when configuration changes."""

    config_name: str = ""
    changed_keys: list = field(default_factory=list)
    old_values: Dict[str, Any] = field(default_factory=dict)
    new_values: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize event metadata."""
        self.event_type = ConfigurationEventType.CHANGED.value
        self.priority = 40


@dataclass
class ErrorOccurredEvent(Event):
    """Event fired when an error occurs."""

    error_message: str = ""
    error_type: str = ""
    source: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    is_critical: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize event metadata."""
        self.event_type = (
            ErrorEventType.CRITICAL.value
            if self.is_critical
            else ErrorEventType.OCCURRED.value
        )
        self.priority = 100 if self.is_critical else 80
