"""Composable mixins for processor behavior and state management.

This module provides reusable mixin classes that implement common processor
behavior through composition instead of deep inheritance hierarchies.

Advantages:
- Eliminates deep inheritance chains (300+ lines of boilerplate)
- Enables multiple independent concerns (caching, logging, validation, state)
- Easier to test and maintain individual behaviors
- More flexible than inheritance-based design
- Follows composition over inheritance principle

Key Mixins:
- LoggingMixin: Automatic logging of processor lifecycle and execution
- CachingMixin: Transparent caching of processor results
- ValidationMixin: Input/output validation with error handling
- StateTrackingMixin: Track processor execution state and history
- ErrorHandlingMixin: Consistent error handling and recovery patterns
- MetricsMixin: Track execution metrics (timing, memory, errors)

Example Usage:
    >>> class MyProcessor(LoggingMixin, ValidationMixin, Processor):
    ...     def process(self, data):
    ...         self.log_info(f"Processing {len(data)} items")
    ...         validated = self.validate_input(data)
    ...         return self.transform(validated)

"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Dict, List, TypeVar, cast
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

__all__ = [
    "ProcessorState",
    "LoggingMixin",
    "CachingMixin",
    "ValidationMixin",
    "StateTrackingMixin",
    "ErrorHandlingMixin",
    "MetricsMixin",
    "ProcessorMixinManager",
]

T = TypeVar("T")


class ProcessorState(Enum):
    """Enumeration of processor execution states."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class ExecutionMetrics:
    """Tracks execution metrics for a processor operation.

    Attributes
    ----------
    start_time : float
        Unix timestamp when execution started.
    end_time : Optional[float]
        Unix timestamp when execution ended (None if still running).
    duration : float
        Total execution duration in seconds.
    memory_bytes : int
        Approximate memory used during execution.
    error_count : int
        Number of errors encountered.
    cache_hits : int
        Number of cache hits.
    cache_misses : int
        Number of cache misses.
    """

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: float = 0.0
    memory_bytes: int = 0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.end_time is not None

    def finalize(self) -> None:
        """Mark execution as complete and calculate duration."""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time


@dataclass
class ExecutionRecord:
    """Records execution history for state tracking.

    Attributes
    ----------
    state : ProcessorState
        State of the execution (running, succeeded, failed, etc).
    timestamp : datetime
        When the execution occurred.
    metrics : ExecutionMetrics
        Metrics for the execution.
    error_message : Optional[str]
        Error message if state is FAILED.
    """

    state: ProcessorState
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    error_message: Optional[str] = None


class LoggingMixin:
    """Provides automatic logging capability to processors.

    Adds convenience methods for logging processor lifecycle events.
    Integrates with Python's logging module for consistent output.

    Example
    -------
    >>> class MyProcessor(LoggingMixin):
    ...     def process(self, data):
    ...         self.log_debug(f"Processing {len(data)} items")
    ...         self.log_info("Processing started")
    ...         try:
    ...             result = self.compute(data)
    ...             self.log_info("Processing completed")
    ...             return result
    ...         except Exception as e:
    ...             self.log_error(f"Processing failed: {e}")
    ...             raise
    """

    def log_debug(self, message: str) -> None:
        """Log a debug-level message."""
        logger.debug(f"[{self.__class__.__name__}] {message}")

    def log_info(self, message: str) -> None:
        """Log an info-level message."""
        logger.info(f"[{self.__class__.__name__}] {message}")

    def log_warning(self, message: str) -> None:
        """Log a warning-level message."""
        logger.warning(f"[{self.__class__.__name__}] {message}")

    def log_error(self, message: str) -> None:
        """Log an error-level message."""
        logger.error(f"[{self.__class__.__name__}] {message}")

    def log_critical(self, message: str) -> None:
        """Log a critical-level message."""
        logger.critical(f"[{self.__class__.__name__}] {message}")


class CachingMixin:
    """Provides transparent result caching to processors.

    Automatically caches processor results based on input arguments.
    Supports cache invalidation and statistics tracking.

    Attributes
    ----------
    _cache : Dict[str, Any]
        Internal cache store mapping arguments to results.
    _cache_enabled : bool
        Whether caching is currently enabled.

    Example
    -------
    >>> class CachedProcessor(CachingMixin):
    ...     def process(self, data):
    ...         return self.cached_call(data, self._compute)
    ...
    ...     def _compute(self, data):
    ...         return expensive_operation(data)
    """

    def __init__(self) -> None:
        """Initialize caching infrastructure."""
        self._cache: Dict[str, Any] = {}
        self._cache_enabled: bool = True

    def enable_cache(self) -> None:
        """Enable result caching."""
        self._cache_enabled = True

    def disable_cache(self) -> None:
        """Disable result caching."""
        self._cache_enabled = False

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Get the number of cached results."""
        return len(self._cache)

    def _make_cache_key(self, *args: Any, **kwargs: Any) -> str:
        """Create a cache key from arguments.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        str
            Cache key as string representation of args/kwargs.
        """
        args_str = ",".join(str(arg) for arg in args)
        kwargs_str = ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{args_str}|{kwargs_str}"

    def cached_call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with caching.

        Parameters
        ----------
        func : Callable
            Function to execute (cached if enabled).
        *args
            Positional arguments to pass to function.
        **kwargs
            Keyword arguments to pass to function.

        Returns
        -------
        T
            Result from cache or function execution.
        """
        if not self._cache_enabled:
            return func(*args, **kwargs)

        cache_key = self._make_cache_key(*args, **kwargs)
        if cache_key in self._cache:
            return cast(T, self._cache[cache_key])

        result = func(*args, **kwargs)
        self._cache[cache_key] = result
        return result


class ValidationMixin:
    """Provides input/output validation capability to processors.

    Enables automatic validation of processor inputs and outputs.
    Supports custom validators and error handling.

    Example
    -------
    >>> class ValidatingProcessor(ValidationMixin):
    ...     def process(self, data):
    ...         self.validate_input(data)
    ...         result = self.compute(data)
    ...         self.validate_output(result)
    ...         return result
    """

    def validate_input(
        self, data: Any, predicate: Optional[Callable[[Any], bool]] = None
    ) -> Any:
        """Validate processor input.

        Parameters
        ----------
        data : Any
            Input data to validate.
        predicate : Optional[Callable]
            Validation predicate. If None, only checks for non-None.

        Returns
        -------
        Any
            The validated data.

        Raises
        ------
        ValueError
            If validation fails.
        """
        if data is None:
            raise ValueError(f"{self.__class__.__name__}: input cannot be None")

        if predicate and not predicate(data):
            raise ValueError(f"{self.__class__.__name__}: input validation failed")

        return data

    def validate_output(
        self, data: Any, predicate: Optional[Callable[[Any], bool]] = None
    ) -> Any:
        """Validate processor output.

        Parameters
        ----------
        data : Any
            Output data to validate.
        predicate : Optional[Callable]
            Validation predicate. If None, only checks for non-None.

        Returns
        -------
        Any
            The validated data.

        Raises
        ------
        ValueError
            If validation fails.
        """
        if data is None:
            raise ValueError(f"{self.__class__.__name__}: output cannot be None")

        if predicate and not predicate(data):
            raise ValueError(f"{self.__class__.__name__}: output validation failed")

        return data


class StateTrackingMixin:
    """Tracks processor execution state and history.

    Maintains a history of executions with state transitions and metrics.
    Useful for debugging, auditing, and monitoring processor behavior.

    Attributes
    ----------
    _state : ProcessorState
        Current execution state.
    _execution_history : List[ExecutionRecord]
        History of all executions.
    _max_history_size : int
        Maximum number of records to keep (default: 100).

    Example
    -------
    >>> class TrackedProcessor(StateTrackingMixin):
    ...     def process(self, data):
    ...         self.set_state(ProcessorState.RUNNING)
    ...         try:
    ...             result = self.compute(data)
    ...             self.set_state(ProcessorState.SUCCEEDED)
    ...             return result
    ...         except Exception as e:
    ...             self.set_state(ProcessorState.FAILED, str(e))
    ...             raise
    """

    def __init__(self, max_history_size: int = 100) -> None:
        """Initialize state tracking.

        Parameters
        ----------
        max_history_size : int
            Maximum number of execution records to keep.
        """
        self._state: ProcessorState = ProcessorState.INITIALIZED
        self._execution_history: List[ExecutionRecord] = []
        self._max_history_size: int = max_history_size
        self._current_metrics: Optional[ExecutionMetrics] = None
        self._last_metrics: Optional[ExecutionMetrics] = None

    @property
    def state(self) -> ProcessorState:
        """Get current processor state."""
        return self._state

    def set_state(
        self, state: ProcessorState, error_message: Optional[str] = None
    ) -> None:
        """Set processor state.

        Parameters
        ----------
        state : ProcessorState
            New state.
        error_message : Optional[str]
            Error message if state is FAILED.
        """
        self._state = state
        if self._current_metrics:
            self._current_metrics.finalize()

            record = ExecutionRecord(
                state=state,
                metrics=self._current_metrics,
                error_message=error_message,
            )
            self._execution_history.append(record)
            self._last_metrics = self._current_metrics  # Store for later retrieval

            # Keep history size bounded
            if len(self._execution_history) > self._max_history_size:
                self._execution_history.pop(0)

            self._current_metrics = None

    def start_metrics(self) -> None:
        """Start tracking execution metrics."""
        self._current_metrics = ExecutionMetrics()

    def get_state_history(self) -> List[ExecutionRecord]:
        """Get execution history.

        Returns
        -------
        List[ExecutionRecord]
            All recorded executions.
        """
        return list(self._execution_history)

    def get_last_execution(self) -> Optional[ExecutionRecord]:
        """Get the most recent execution record.

        Returns
        -------
        Optional[ExecutionRecord]
            Last execution record or None if no executions.
        """
        return self._execution_history[-1] if self._execution_history else None


class ErrorHandlingMixin:
    """Provides consistent error handling and recovery patterns.

    Enables uniform error handling, retries, and recovery strategies.
    Useful for building resilient processors.

    Attributes
    ----------
    _error_handlers : Dict[type, Callable]
        Mapping of exception types to handler functions.
    _max_retries : int
        Maximum number of retry attempts.

    Example
    -------
    >>> class ResilientProcessor(ErrorHandlingMixin):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.register_error_handler(ValueError, self._handle_value_error)
    ...
    ...     def _handle_value_error(self, error: ValueError) -> Any:
    ...         logger.warning(f"Value error: {error}")
    ...         return None  # or recovery value
    """

    def __init__(self, max_retries: int = 3) -> None:
        """Initialize error handling.

        Parameters
        ----------
        max_retries : int
            Maximum number of retry attempts.
        """
        self._error_handlers: Dict[type, Callable[[Exception], Any]] = {}
        self._max_retries: int = max_retries

    def register_error_handler(
        self, error_type: type, handler: Callable[[Exception], Any]
    ) -> None:
        """Register a handler for a specific exception type.

        Parameters
        ----------
        error_type : type
            Exception class to handle.
        handler : Callable
            Handler function that receives the exception.
        """
        self._error_handlers[error_type] = handler

    def handle_error(self, error: Exception) -> Any:
        """Handle an exception using registered handler.

        Parameters
        ----------
        error : Exception
            The exception to handle.

        Returns
        -------
        Any
            Result from handler or None if no handler registered.

        Raises
        ------
        Exception
            If no handler is registered for the exception type.
        """
        error_type = type(error)
        if error_type in self._error_handlers:
            return self._error_handlers[error_type](error)

        # Check for parent class handlers
        for registered_type, handler in self._error_handlers.items():
            if isinstance(error, registered_type):
                return handler(error)

        # No handler found
        raise error

    def retry(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with automatic retry on failure.

        Parameters
        ----------
        func : Callable
            Function to execute with retries.
        *args
            Positional arguments to function.
        **kwargs
            Keyword arguments to function.

        Returns
        -------
        T
            Result from function.

        Raises
        ------
        Exception
            If all retry attempts fail.
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self._max_retries}: {e}"
                    )

        if last_error is not None:
            raise last_error
        raise RuntimeError("Retry exhausted without error")


class MetricsMixin:
    """Tracks execution metrics and performance data.

    Collects timing, memory, error, and cache statistics.
    Useful for monitoring and profiling processor performance.

    Attributes
    ----------
    _metrics : ExecutionMetrics
        Current execution metrics.

    Example
    -------
    >>> class MetricsProcessor(MetricsMixin):
    ...     def process(self, data):
    ...         with self.track_metrics():
    ...             return self.compute(data)
    """

    def __init__(self) -> None:
        """Initialize metrics tracking."""
        self._metrics: Optional[ExecutionMetrics] = None

    def track_metrics(self) -> "_MetricsContext":
        """Context manager for tracking execution metrics.

        Usage
        -----
        >>> processor = MetricsProcessor()
        >>> with processor.track_metrics():
        ...     result = processor.compute(data)
        >>> print(processor.get_metrics().duration)
        """
        return _MetricsContext(self)

    def start_metrics(self) -> None:
        """Start a new metrics collection."""
        # If StateTrackingMixin is also being used, delegate to it
        if hasattr(self, "_current_metrics"):
            self._current_metrics = ExecutionMetrics()
        else:
            self._metrics = ExecutionMetrics()

    def end_metrics(self) -> ExecutionMetrics:
        """End metrics collection and return the metrics.

        Returns
        -------
        ExecutionMetrics
            Collected metrics.
        """
        if self._metrics:
            self._metrics.finalize()
            return self._metrics
        return ExecutionMetrics()

    def get_metrics(self) -> Optional[ExecutionMetrics]:
        """Get current metrics.

        Returns
        -------
        Optional[ExecutionMetrics]
            Current metrics or None if not tracking.
        """
        # If StateTrackingMixin is being used, get from there using getattr to satisfy type checkers
        current = getattr(self, "_current_metrics", None)
        if current is not None:
            return cast(Optional[ExecutionMetrics], current)

        last = getattr(self, "_last_metrics", None)
        if last is not None:
            return cast(Optional[ExecutionMetrics], last)

        return self._metrics

    def record_error(self) -> None:
        """Record an error in metrics."""
        metrics = self.get_metrics()
        if metrics:
            metrics.error_count += 1

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        metrics = self.get_metrics()
        if metrics:
            metrics.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        metrics = self.get_metrics()
        if metrics:
            metrics.cache_misses += 1


class _MetricsContext:
    """Context manager for metrics tracking."""

    def __init__(self, mixin: MetricsMixin) -> None:
        """Initialize metrics context.

        Parameters
        ----------
        mixin : MetricsMixin
            The mixin instance to track.
        """
        self.mixin = mixin

    def __enter__(self) -> ExecutionMetrics:
        """Enter context, start metrics."""
        self.mixin.start_metrics()
        metrics = self.mixin.get_metrics()
        if metrics is None:
            raise RuntimeError("Metrics could not be initialized by start_metrics.")
        return metrics

    def __exit__(self, exc_type: Any, exc_val: Any, _exc_tb: Any) -> None:
        """Exit context, end metrics."""
        self.mixin.end_metrics()


class ProcessorMixinManager:
    """Manager for composable processor mixins.

    Helps coordinate and configure multiple mixins on a processor instance.
    Useful for enabling/disabling specific behaviors at runtime.

    Example
    -------
    >>> class MyProcessor(LoggingMixin, CachingMixin, ValidationMixin):
    ...     pass
    ...
    >>> processor = MyProcessor()
    >>> manager = ProcessorMixinManager(processor)
    >>> manager.enable_mixin(CachingMixin)
    >>> manager.disable_mixin(LoggingMixin)
    """

    def __init__(self, processor: Any) -> None:
        """Initialize mixin manager.

        Parameters
        ----------
        processor : Any
            Processor instance with mixins.
        """
        self.processor = processor
        self._enabled_mixins: Dict[str, bool] = {}

    def enable_mixin(self, mixin_class: type) -> None:
        """Enable a mixin on the processor.

        Parameters
        ----------
        mixin_class : type
            The mixin class to enable.
        """
        mixin_name = mixin_class.__name__
        if isinstance(self.processor, mixin_class):
            self._enabled_mixins[mixin_name] = True
        else:
            logger.warning(f"Processor does not have {mixin_name}")

    def disable_mixin(self, mixin_class: type) -> None:
        """Disable a mixin on the processor.

        Parameters
        ----------
        mixin_class : type
            The mixin class to disable.
        """
        mixin_name = mixin_class.__name__
        self._enabled_mixins[mixin_name] = False

    def is_mixin_enabled(self, mixin_class: type) -> bool:
        """Check if a mixin is enabled.

        Parameters
        ----------
        mixin_class : type
            The mixin class to check.

        Returns
        -------
        bool
            True if mixin is enabled.
        """
        mixin_name = mixin_class.__name__
        return self._enabled_mixins.get(mixin_name, False)

    def get_enabled_mixins(self) -> List[str]:
        """Get list of enabled mixin names.

        Returns
        -------
        List[str]
            Names of enabled mixins.
        """
        return [name for name, enabled in self._enabled_mixins.items() if enabled]
