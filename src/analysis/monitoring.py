"""
Logging & Monitoring Implementation

Provides structured logging, metrics collection, and comprehensive monitoring
for observability and diagnostics.

Key Features:
- Structured JSON logging for easy parsing
- Metrics collection and aggregation
- Performance monitoring with timers
- Health checks and system status
- Integration with standard Python logging
- Thread-safe operations
- Configurable log levels and formatting

Usage:
    # Structured logging
    logger = StructuredLogger("my_service")
    logger.info("Operation started", operation_id="op123", user_id="user456")

    # Metrics collection
    metrics = MetricsCollector()
    metrics.record_counter("requests", 1)
    metrics.record_histogram("response_time", 0.45)

    # Performance monitoring
    with PerformanceMonitor("database_query") as monitor:
        result = query_database()
    print(monitor.get_metrics())
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, Type, Literal, cast
from types import TracebackType
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from threading import RLock
from functools import wraps
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Standard log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"  # Timing measurements


@dataclass
class LogEvent:
    """Structured log event."""

    timestamp: str
    level: str
    message: str
    service: str
    context: Dict[str, Any] = field(default_factory=lambda: {})
    exception: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)

    def __str__(self) -> str:
        return f"[{self.timestamp}] {self.level}: {self.message}"


@dataclass
class MetricValue:
    """A single metric value."""

    name: str
    type: MetricType
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=lambda: cast(Dict[str, str], {}))

    def __str__(self) -> str:
        tags_str = ", ".join(f"{k}={v}" for k, v in self.tags.items())
        return f"{self.name}({self.type.value}): {self.value}" + (
            f" [{tags_str}]" if tags_str else ""
        )


@dataclass
class PerformanceMetrics:
    """Performance metrics for timed operations."""

    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=lambda: cast(Dict[str, Any], {}))

    @property
    def is_complete(self) -> bool:
        """Check if operation has completed."""
        return self.end_time is not None

    def complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark operation as complete."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        duration_ms = (self.duration * 1000) if self.duration else "N/A"
        return f"{status} {self.name}: {duration_ms:.2f}ms"


class StructuredLogger:
    """
    Structured logging with JSON output for better machine readability.
    """

    def __init__(
        self,
        service_name: str,
        log_level: LogLevel = LogLevel.INFO,
        include_context: bool = True,
    ) -> None:
        """
        Initialize structured logger.

        Args:
            service_name: Name of the service
            log_level: Minimum log level
            include_context: Whether to include context in logs
        """
        self.service_name = service_name
        self.log_level = log_level
        self.include_context = include_context
        self._lock = RLock()
        self._context_stack: List[Dict[str, Any]] = []

    def _get_current_context(self) -> Dict[str, Any]:
        """Get merged context from stack."""
        merged: Dict[str, Any] = {}
        for ctx in self._context_stack:
            # ctx is a Dict[str, Any], annotating merged makes update type-safe
            merged.update(ctx)
        return merged

    def _log(
        self,
        level: LogLevel,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs: Any,
    ) -> LogEvent:
        """Internal logging method."""
        event = LogEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.value,
            message=message,
            service=self.service_name,
            context=kwargs if self.include_context else {},
            exception=str(exception) if exception else None,
        )

        # Log to standard logger
        log_func = getattr(logger, level.value.lower())
        log_func(event.to_json())

        return event

    def debug(self, message: str, **kwargs: Any) -> LogEvent:
        """Log debug message."""
        return self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> LogEvent:
        """Log info message."""
        return self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> LogEvent:
        """Log warning message."""
        return self._log(LogLevel.WARNING, message, **kwargs)

    def error(
        self, message: str, exception: Optional[Exception] = None, **kwargs: Any
    ) -> LogEvent:
        """Log error message."""
        return self._log(LogLevel.ERROR, message, exception, **kwargs)

    def critical(
        self, message: str, exception: Optional[Exception] = None, **kwargs: Any
    ) -> LogEvent:
        """Log critical message."""
        return self._log(LogLevel.CRITICAL, message, exception, **kwargs)

    def push_context(self, **context: Any) -> None:
        """Push context onto stack."""
        with self._lock:
            self._context_stack.append(context)

    def pop_context(self) -> None:
        """Pop context from stack."""
        with self._lock:
            if self._context_stack:
                self._context_stack.pop()

    def __enter__(self) -> "StructuredLogger":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> None:
        """Context manager exit."""
        pass


class MetricsCollector:
    """
    Collects and aggregates metrics for monitoring.
    """

    def __init__(self, name: str = "default") -> None:
        """
        Initialize metrics collector.

        Args:
            name: Name of the metrics collector
        """
        self.name = name
        self._metrics: Dict[str, List[MetricValue]] = {}
        self._lock = RLock()

    def record_counter(
        self,
        metric_name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a counter metric (monotonically increasing).

        Args:
            metric_name: Name of the metric
            value: Value to add (default: 1)
            tags: Optional tags for grouping
        """
        with self._lock:
            metric = MetricValue(
                name=metric_name,
                type=MetricType.COUNTER,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
            )
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []
            self._metrics[metric_name].append(metric)

    def record_gauge(
        self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a gauge metric (point-in-time value).

        Args:
            metric_name: Name of the metric
            value: Current value
            tags: Optional tags for grouping
        """
        with self._lock:
            metric = MetricValue(
                name=metric_name,
                type=MetricType.GAUGE,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
            )
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []
            self._metrics[metric_name].append(metric)

    def record_histogram(
        self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a histogram metric (distribution of values).

        Args:
            metric_name: Name of the metric
            value: Value to record
            tags: Optional tags for grouping
        """
        with self._lock:
            metric = MetricValue(
                name=metric_name,
                type=MetricType.HISTOGRAM,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
            )
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []
            self._metrics[metric_name].append(metric)

    def record_timer(
        self, metric_name: str, duration: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a timer metric (duration in seconds).

        Args:
            metric_name: Name of the metric
            duration: Duration in seconds
            tags: Optional tags for grouping
        """
        with self._lock:
            metric = MetricValue(
                name=metric_name,
                type=MetricType.TIMER,
                value=duration,
                timestamp=time.time(),
                tags=tags or {},
            )
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []
            self._metrics[metric_name].append(metric)

    def get_metric(self, name: str) -> Optional[MetricValue]:
        """Get the latest value of a metric."""
        with self._lock:
            values = self._metrics.get(name, [])
            return values[-1] if values else None

    def get_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all metrics."""
        with self._lock:
            summary: Dict[str, Dict[str, Any]] = {}
            for name, values in self._metrics.items():
                if not values:
                    continue

                # Get latest value
                latest = values[-1]
                metric_values = [float(v.value) for v in values]

                summary[name] = {
                    "type": latest.type.value,
                    "latest": float(latest.value),
                    "count": len(values),
                    "sum": float(sum(metric_values)),
                    "min": float(min(metric_values)) if metric_values else 0.0,
                    "max": float(max(metric_values)) if metric_values else 0.0,
                    "avg": (
                        float(statistics.mean(metric_values)) if metric_values else 0.0
                    ),
                    "tags": latest.tags,
                }

            return summary

    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()


class PerformanceMonitor:
    """
    Context manager for monitoring performance of operations.
    """

    def __init__(self, name: str, logger: Optional[StructuredLogger] = None) -> None:
        """
        Initialize performance monitor.

        Args:
            name: Name of the operation
            logger: Optional structured logger for output
        """
        self.name = name
        self.logger = logger
        self.metrics = PerformanceMetrics(name, time.time())

    def __enter__(self) -> "PerformanceMonitor":
        """Enter context - operation started."""
        if self.logger:
            self.logger.debug(f"Operation started: {self.name}")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        """Exit context - operation completed."""
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self.metrics.complete(success=success, error=error)

        if self.logger:
            if success:
                self.logger.info(
                    f"Operation completed: {self.name}",
                    duration_ms=(
                        self.metrics.duration * 1000 if self.metrics.duration else 0
                    ),
                )
            else:
                self.logger.error(
                    f"Operation failed: {self.name}",
                    error=error,
                    duration_ms=(
                        self.metrics.duration * 1000 if self.metrics.duration else 0
                    ),
                )

        return False  # Don't suppress exceptions

    def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        return self.metrics


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    @abstractmethod
    def check(self) -> bool:
        """
        Perform health check.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the health check."""
        pass


class SimpleHealthCheck(HealthCheck):
    """Simple callable-based health check."""

    def __init__(self, name: str, check_func: Callable[[], bool]) -> None:
        """
        Initialize health check.

        Args:
            name: Name of the check
            check_func: Callable that returns True if healthy
        """
        self._name = name
        self._check_func = check_func

    def check(self) -> bool:
        """Perform health check."""
        try:
            return self._check_func()
        except Exception:
            return False

    @property
    def name(self) -> str:
        """Name of the health check."""
        return self._name


class HealthCheckRegistry:
    """
    Registry for health checks.
    """

    def __init__(self) -> None:
        """Initialize health check registry."""
        self._checks: Dict[str, HealthCheck] = {}
        self._lock = RLock()

    def register(self, check: HealthCheck) -> None:
        """
        Register a health check.

        Args:
            check: Health check instance
        """
        with self._lock:
            self._checks[check.name] = check

    def unregister(self, name: str) -> None:
        """
        Unregister a health check.

        Args:
            name: Name of the check
        """
        with self._lock:
            if name in self._checks:
                del self._checks[name]

    def check_all(self) -> Dict[str, bool]:
        """
        Run all health checks.

        Returns:
            Dict mapping check names to health status
        """
        with self._lock:
            results: Dict[str, bool] = {}
            for name, check in self._checks.items():
                results[name] = check.check()
            return results

    def is_healthy(self) -> bool:
        """
        Check if all health checks pass.

        Returns:
            True if all checks pass, False otherwise
        """
        return all(self.check_all().values())


def monitor(
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[MetricsCollector] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for monitoring function execution.

    Args:
        logger: Optional structured logger
        metrics: Optional metrics collector

    Returns:
        Decorated function with monitoring

    Example:
        @monitor(logger=my_logger, metrics=my_metrics)
        def important_function():
            pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            monitor_obj = PerformanceMonitor(
                f"{func.__module__}.{func.__name__}", logger=logger
            )

            with monitor_obj:
                result = func(*args, **kwargs)

            # Record metrics if collector provided
            if metrics and monitor_obj.metrics.duration:
                metrics.record_timer(
                    f"{func.__name__}_duration", monitor_obj.metrics.duration
                )

            return result

        return wrapper

    return decorator
