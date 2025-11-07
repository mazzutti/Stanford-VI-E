"""Factory methods and builders for analyzer construction.

This subpackage provides factory methods and builder patterns for creating
pre-configured FaciesCorrelationAnalyzer instances with full dependency injection.

Main Components
---------------
- AnalyzerFactory: Namespace class with static factory methods
- AnalyzerBuilder: Fluent builder for custom configuration
- TypeValidator: Object-oriented type validation

Note: BuilderValidationError and BuilderFrozenError have been moved to
src.analysis.exceptions for centralized exception handling.

Quick Start
-----------
>>> from src.analysis.factories import AnalyzerFactory, TypeValidator
>>> analyzer = AnalyzerFactory.create_default()
>>> custom_analyzer = (AnalyzerFactory.builder()
...     .with_config(config)
...     .with_plotter(plotter)
...     .build())
"""

from src.analysis.factories.analyzer_factory import AnalyzerFactory
from src.analysis.factories.builder import AnalyzerBuilder
from src.analysis.factories.conversion_factory import ConversionStrategyFactory
from src.analysis.factories.service_factory import ProcessorServiceFactory
from src.analysis.factories.validators import TypeValidator

__all__ = [
    "AnalyzerFactory",
    "AnalyzerBuilder",
    "TypeValidator",
    "ProcessorServiceFactory",
    "ConversionStrategyFactory",
]
