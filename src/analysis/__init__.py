"""Analysis pipelines package.

This package contains multi-step pipelines for seismic and rock-physics
analysis. Invoke with e.g. `python -m src.analysis.seismograms`.

Modules:
    - exceptions: Structured exception hierarchy for error handling
    - validators: Reusable validation utilities
    - cache.extractors: Data extraction strategies
    - common: AnalysisCommon singleton for common utilities
"""

import logging

from .exceptions import (
    AnalysisException,
    CacheError,
    CacheLoadingError,
    CacheSelectionError,
    CacheExtractionError,
    ValidationError,
    DomainError,
    ProcessingError,
    ConfigurationError,
    BuilderValidationError,
    BuilderFrozenError,
    ExceptionContextBuilder,
    ComputationError,
    AlignmentError,
    DetectionError,
    ExtractionError,
    InterpolationError,
    StateError,
)
from .validators import (
    Validator,
    RangeValidator,
    CountValidator,
    QuantileValidator,
    ValidatorStrategy,
    CompositeValidator,
)
from .common import AnalysisCommon
from .factories import TypeValidator
from .base import AnalyzerInterface, AnalysisConfig
from .strategies import (
    ArrayStatisticsStrategy,
    StandardArrayStatistics,
    RobustArrayStatistics,
)
from .mixins import (
    SingletonMixin,
    ValidatableMixin,
    ConfigurableMixin,
    StateTrackingMixin,
)
from .results import (
    Result as GenericResult,
    ResultMetadata,
    ResultData,
    wrap_result,
    create_metadata,
)
from .builder import (
    AnalysisBuilder,
    build_facies_analyzer,
    build_rock_physics_analyzer,
)
from .validator_chain import (
    ValidatorChain,
    ValidatorComposite,
    not_none,
    positive,
    negative,
    in_range,
    length_between,
    matches_type,
    is_callable,
)
from .config_builder import (
    ConfigBuilder,
    build_config,
    config_with_defaults,
)
from .processor_mixins import (
    ProcessorState,
    LoggingMixin,
    CachingMixin,
    ValidationMixin,
    StateTrackingMixin,
    ErrorHandlingMixin,
    MetricsMixin,
    ProcessorMixinManager,
    ExecutionMetrics,
    ExecutionRecord,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Modules
    "common",
    "rock_physics",
    "base",
    "strategies",
    "types",
    "domain",
    "pipelines",
    "processors",
    # Singleton
    "AnalysisCommon",
    # Base classes
    "AnalyzerInterface",
    "AnalysisConfig",
    # Strategies
    "ArrayStatisticsStrategy",
    "StandardArrayStatistics",
    "RobustArrayStatistics",
    # Exceptions
    "AnalysisException",
    "CacheError",
    "CacheLoadingError",
    "CacheSelectionError",
    "CacheExtractionError",
    "ValidationError",
    "DomainError",
    "ProcessingError",
    "ConfigurationError",
    "BuilderValidationError",
    "BuilderFrozenError",
    "ExceptionContextBuilder",
    "ComputationError",
    "AlignmentError",
    "DetectionError",
    "ExtractionError",
    "InterpolationError",
    "StateError",
    # Validators
    "Validator",
    "RangeValidator",
    "CountValidator",
    "QuantileValidator",
    "ValidatorStrategy",
    "CompositeValidator",
    # Mixins
    "SingletonMixin",
    "ValidatableMixin",
    "ConfigurableMixin",
    "StateTrackingMixin",
    # OOP utilities
    "TypeValidator",
    # Generic result wrapper
    "GenericResult",
    "ResultMetadata",
    "ResultData",
    "wrap_result",
    "create_metadata",
    # Fluent analyzer builder
    "AnalysisBuilder",
    "build_facies_analyzer",
    "build_rock_physics_analyzer",
    # Validator chain composition
    "ValidatorChain",
    "ValidatorComposite",
    "not_none",
    "positive",
    "negative",
    "in_range",
    "length_between",
    "matches_type",
    "is_callable",
    # Configuration builder
    "ConfigBuilder",
    "build_config",
    "config_with_defaults",
    # Processor mixins
    "ProcessorState",
    "LoggingMixin",
    "CachingMixin",
    "ValidationMixin",
    "StateTrackingMixin",
    "ErrorHandlingMixin",
    "MetricsMixin",
    "ProcessorMixinManager",
    "ExecutionMetrics",
    "ExecutionRecord",
]
