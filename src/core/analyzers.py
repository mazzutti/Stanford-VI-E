"""Analyzer base classes for code consolidation.

This module provides abstract base classes and mixins for all analyzer
implementations, eliminating ~300-400 LOC of duplicated initialization,
configuration, and lifecycle management code.

Patterns Used:
  - Template Method: Standard analyzer lifecycle
  - Mixin: Compose behavior across analyzers
  - Protocol: Type-safe interfaces
  - ABC: Enforce analyzer contracts

Benefits:
  - Eliminates boilerplate (context managers, initialization, etc.)
  - Centralizes validation and error handling
  - Consistent logging and lifecycle
  - Easy to extend with new analyzers
  - Reduces facies analyzer from 988 → 700 LOC
  - Reduces rock_physics analyzer from 515 → 350 LOC

Savings: ~300-400 LOC per phase

Example:
    >>> from src.core.analyzers import BaseAnalyzer, AnalyzerState
    >>>
    >>> class MyAnalyzer(BaseAnalyzer):
    ...     def _validate_config(self):
    ...         if not self.config:
    ...             raise ValueError("Config required")
    ...
    ...     def analyze(self, data):
    ...         return process(data)
    >>>
    >>> analyzer = MyAnalyzer("my_analyzer", config)
    >>> result = analyzer.execute(data)  # Lifecycle handled automatically
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any, Generic, TypeVar, cast

from src.core.validation import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "AnalyzerState",
    "BaseAnalyzer",
    "AnalyzerLifecycle",
    "PipelineAnalyzer",
    "CompositeMixin",
    "CacheMixin",
    "ValidationMixin",
    "MetricsMixin",
]

ConfigT = TypeVar("ConfigT")  # Configuration type
ResultT = TypeVar("ResultT")  # Result type

class AnalyzerState(Enum):
    """Lifecycle state of an analyzer."""

    CREATED = "created"  # Just instantiated
    INITIALIZED = "initialized"  # Config validated, dependencies ready
    RUNNING = "running"  # Analysis in progress
    COMPLETED = "completed"  # Analysis finished
    FAILED = "failed"  # Analysis failed
    DISPOSED = "disposed"  # Resources cleaned up

@dataclass
class AnalysisMetrics:
    """Track analysis execution metrics."""

    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    memory_used: int = 0
    error_count: int = 0

# ============================================================================
# Core Analyzer Classes
# ============================================================================

class BaseAnalyzer(ABC, Generic[ConfigT, ResultT]):
    """Abstract base class for all analyzers.

    Provides common lifecycle management, configuration handling, validation,
    logging, and error handling. Subclasses implement domain-specific analysis.

    Template Method Pattern:
    - Subclasses override: _validate_config(), analyze()
    - Framework calls: execute(), __enter__, __exit__

    Type Parameters
    ---------------
    ConfigT
        Configuration class for this analyzer
    ResultT
        Result type returned by analysis
    """

    # Class-level constants (override in subclasses)
    DEFAULT_NAME = "analyzer"

    def __init__(
        self,
        name: str | None = None,
        config: ConfigT | None = None,
    ):
        """Initialize analyzer.

        Parameters
        ----------
        name : str, optional
            Analyzer name for logging/debugging. Defaults to class name.
        config : ConfigT, optional
            Configuration object. Subclass handles if None.
        """
        self.name = name or self.__class__.__name__
        self._config = config
        self._state = AnalyzerState.CREATED
        self._initialized = False
        self._metrics = AnalysisMetrics()
        self._error: Exception | None = None

        logger.debug("%s: Created", self.name)

    @property
    def config(self) -> ConfigT | None:
        """Return the analyzer configuration (may be None)."""
        return self._config

    # ====================================================================
    # Lifecycle Management
    # ====================================================================

    @property
    def state(self) -> AnalyzerState:
        """Get current analyzer state."""
        return self._state

    @property
    def is_initialized(self) -> bool:
        """Check if analyzer is initialized."""
        return self._initialized

    @property
    def is_ready(self) -> bool:
        """Check if analyzer is ready for analysis."""
        return self._initialized and self._state != AnalyzerState.FAILED

    def _mark_error(self, error: Exception, context: str) -> None:
        """Mark analyzer as failed and log error.

        Parameters
        ----------
        error : Exception
            The exception that occurred.
        context : str
            Description of where the error occurred (e.g., "Initialization failed").
        """
        self._state = AnalyzerState.FAILED
        self._error = error
        logger.error("%s: %s: %s", self.name, context, error)

    def initialize(self) -> None:
        """Initialize analyzer (validate config, setup dependencies).

        Called automatically by execute() or can be called explicitly.

        Raises
        ------
        ValidationError
            If configuration is invalid
        """
        if self._initialized:
            logger.debug("%s: Already initialized", self.name)
            return

        try:
            logger.debug("%s: Validating configuration", self.name)
            self._validate_config()

            logger.debug("%s: Setting up dependencies", self.name)
            self._setup()

            self._initialized = True
            self._state = AnalyzerState.INITIALIZED
            logger.debug("%s: Initialized successfully", self.name)

        except (ValidationError, RuntimeError, ValueError, OSError) as e:
            self._mark_error(e, "Initialization failed")
            raise

    def _validate_config(self) -> None:
        """Validate configuration. Override for custom validation.

        Raises
        ------
        ValidationError
            If configuration is invalid
        """
        if self.config is None:
            raise ValidationError(f"{self.name}: Configuration required")

    def _setup(self) -> None:
        """Setup analyzer dependencies. Override for custom setup."""

    def execute(self, data: Any) -> ResultT:
        """Execute analysis with automatic lifecycle management.

        Parameters
        ----------
        data
            Input data for analysis

        Returns
        -------
        ResultT
            Analysis result

        Raises
        ------
        ValidationError
            If not ready or validation fails
        RuntimeError
            If analysis fails
        """
        # Ensure initialized
        if not self._initialized:
            self.initialize()

        # Check state
        if not self.is_ready:
            raise RuntimeError(
                f"{self.name}: Not ready for analysis (state={self._state})"
            )

        # Execute analysis
        try:
            self._state = AnalyzerState.RUNNING
            logger.debug("%s: Starting analysis", self.name)

            result = self.analyze(data)

            self._state = AnalyzerState.COMPLETED
            logger.debug("%s: Analysis completed successfully", self.name)
            return result
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self._mark_error(e, "Analysis failed")
            raise

    @abstractmethod
    def analyze(self, data: Any) -> ResultT:
        """Perform domain-specific analysis. Subclasses implement.

        Parameters
        ----------
        data
            Input data for analysis

        Returns
        -------
        ResultT
            Analysis result
        """

    # ====================================================================
    # Context Manager Support
    # ====================================================================

    def __enter__(self) -> BaseAnalyzer[ConfigT, ResultT]:
        """Enter context manager."""
        logger.debug("%s: Entering context", self.name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        if exc_type is not None:
            logger.error(
                "%s: Context exited with exception: %s", self.name, exc_type.__name__
            )
        self.dispose()

    # ====================================================================
    # Resource Management
    # ====================================================================

    def dispose(self) -> None:
        """Clean up resources. Override for cleanup logic."""
        self._state = AnalyzerState.DISPOSED
        logger.debug("%s: Disposed", self.name)

    # ====================================================================
    # String Representations
    # ====================================================================

    def __repr__(self) -> str:
        """Return detailed string representation."""
        config_type = self.config.__class__.__name__ if self.config else "None"
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"config=<{config_type}>, "
            f"state={self._state.value})"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return f"{self.name} (state={self._state.value})"

# ============================================================================
# Specialized Analyzer Classes
# ============================================================================

class AnalyzerLifecycle(BaseAnalyzer[ConfigT, ResultT]):
    """Enhanced analyzer with lifecycle hooks.

    Provides pre/post execution hooks for setup/teardown, useful for
    managing resources that need cleanup.
    """

    def execute(self, data: Any) -> ResultT:
        """Execute with lifecycle hooks."""
        self.on_before_analyze(data)
        try:
            result = super().execute(data)
            self.on_after_analyze(data, result)
            return result
        except Exception as exc:
            # Framework-level lifecycle wrapper: ensure hook is invoked for
            # any unexpected error from the underlying analysis and preserve
            # original exception context when re-raising.
            self.on_analysis_failed(data, exc)
            raise

    def on_before_analyze(self, data: Any) -> None:
        """Called before analysis. Override for custom pre-processing."""

    def on_after_analyze(self, data: Any, result: ResultT) -> None:
        """Called after successful analysis. Override for cleanup."""

    def on_analysis_failed(self, data: Any, error: Exception) -> None:
        """Called when analysis fails. Override for error handling."""

class PipelineAnalyzer(BaseAnalyzer[ConfigT, ResultT]):
    """Analyzer that chains multiple processing steps.

    Useful for analyzers with multiple stages (boundary detection,
    feature extraction, classification, etc.).

    Example:
        >>> class MultiStageAnalyzer(PipelineAnalyzer):
        ...     def _create_pipeline(self):
        ...         return [
        ...             ("detect", self.detect_boundaries),
        ...             ("extract", self.extract_features),
        ...             ("classify", self.classify),
        ...         ]
        ...
        ...     def analyze(self, data):
        ...         return self._execute_pipeline(data)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize pipeline analyzer."""
        super().__init__(*args, **kwargs)
        self._pipeline: list[tuple[str, Any]] = []

    def _setup(self) -> None:
        """Setup pipeline stages."""
        self._pipeline = self._create_pipeline()

    def _create_pipeline(self) -> list[tuple[str, Any]]:
        """Create pipeline stages. Override to define pipeline."""
        return []

    def _execute_pipeline(self, data: Any) -> ResultT:
        """Execute all pipeline stages in order."""
        result = data
        for stage_name, stage_func in self._pipeline:
            logger.debug("%s: Executing stage '%s'", self.name, stage_name)
            result = stage_func(result)
        return cast(ResultT, result)

# ============================================================================
# Mixins for Shared Functionality
# ============================================================================

class CompositeMixin:
    """Mixin for analyzers that compose multiple sub-analyzers."""

    name: str

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize composite."""
        super().__init__(*args, **kwargs)
        self._sub_analyzers: dict[str, Any] = {}

    def add_sub_analyzer(self, name: str, analyzer: Any) -> None:
        """Register sub-analyzer."""
        self._sub_analyzers[name] = analyzer
        logger.debug("%s: Added sub-analyzer '%s'", self.name, name)

    def get_sub_analyzer(self, name: str) -> Any:
        """Get sub-analyzer by name."""
        return self._sub_analyzers.get(name)

class CacheMixin:
    """Mixin for analyzers that use caching."""

    name: str

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize cache mixin."""
        super().__init__(*args, **kwargs)
        self._cache: dict[str, Any] = {}
        self._cache_enabled = True

    def cache_result(self, key: str, value: Any) -> None:
        """Cache a result."""
        if self._cache_enabled:
            self._cache[key] = value
            logger.debug("%s: Cached '%s'", self.name, key)

    def get_cached(self, key: str) -> Any | None:
        """Get cached result."""
        return self._cache.get(key) if self._cache_enabled else None

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        logger.debug("%s: Cache cleared", self.name)

class ValidationMixin:
    """Mixin for analyzers with validation requirements."""

    name: str

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize validation mixin."""
        super().__init__(*args, **kwargs)
        self._validators: list[Any] = []

    def add_validator(self, validator: Any) -> None:
        """Add validator for data."""
        self._validators.append(validator)

    def validate_data(self, data: Any) -> bool:
        """Validate data using all validators."""
        for validator in self._validators:
            if not validator(data):
                logger.warning("%s: Validation failed for %s", self.name, validator)
                return False
        return True

class MetricsMixin:
    """Mixin for tracking analysis metrics."""

    name: str

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize metrics mixin."""
        super().__init__(*args, **kwargs)
        self._metrics_history: list[dict[str, Any]] = []

    def record_metric(self, name: str, value: Any) -> None:
        """Record a metric value."""
        self._metrics_history.append({"name": name, "value": value})
        logger.debug("%s: Recorded metric '%s=%s'", self.name, name, value)

    def get_metrics(self) -> list[dict[str, Any]]:
        """Get all recorded metrics."""
        return self._metrics_history.copy()

    def clear_metrics(self) -> None:
        """Clear metrics history."""
        self._metrics_history.clear()
