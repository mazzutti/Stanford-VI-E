"""Factory methods and builders for analyzer construction.

This subpackage provides factory methods and builder patterns for creating
pre-configured FaciesCorrelationAnalyzer instances with full dependency injection.

Main Components
---------------
- AnalyzerFactory: Namespace class with static factory methods
- AnalyzerBuilder: Fluent builder for custom configuration
- BuilderValidationError: Raised when builder validation fails
- BuilderFrozenError: Raised when modifying a frozen builder

Quick Start
-----------
>>> from src.analysis.factories import AnalyzerFactory
>>> analyzer = AnalyzerFactory.create_default()
>>> custom_analyzer = (AnalyzerFactory.builder()
...     .with_config(config)
...     .with_plotter(plotter)
...     .build())
"""

from src.analysis.factories.analyzer_factory import AnalyzerFactory
from src.analysis.factories.builder import AnalyzerBuilder
from src.analysis.factories.validators import (
    BuilderValidationError,
    BuilderFrozenError,
    validate_type,
)

__all__ = [
    "AnalyzerFactory",
    "AnalyzerBuilder",
    "BuilderValidationError",
    "BuilderFrozenError",
    "validate_type",
]
