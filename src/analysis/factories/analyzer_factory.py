"""Factory methods for common analyzer configurations.

This module provides the AnalyzerFactory class with convenient static methods
for creating pre-configured analyzers for common use cases without manual
builder setup.

AnalyzerFactory is a namespace class (similar to a module with static methods).
It cannot and should not be instantiated. Use its static methods directly.

Example
-------
>>> analyzer = AnalyzerFactory.create_default()
>>> builder = AnalyzerFactory.builder()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.analysis.factories.builder import AnalyzerBuilder
from src.analysis.models import FaciesCorrelationConfig
from src.analysis.processors import (
    BoundaryDetector,
    CubeAligner,
    GradientCorrelationCalculator,
    InterfaceReflectionAnalyzer,
    FaciesDiscriminationCalculator,
)
from src.analysis.domain import DomainHandlerFactory

if TYPE_CHECKING:
    from src.analysis.facies.analyzer import FaciesCorrelationAnalyzer

logger = logging.getLogger(__name__)


class AnalyzerFactory:
    """Factory methods for common analyzer configurations.

    Provides convenient static methods for creating pre-configured analyzers
    for common use cases without manual builder setup.

    This is a namespace class (similar to a module with static methods).
    It cannot and should not be instantiated. Use its static methods directly:

    Example
    -------
    >>> analyzer = AnalyzerFactory.create_default()
    >>> builder = AnalyzerFactory.builder()
    """

    def __init__(self) -> None:
        """Prevent instantiation of factory namespace class.

        Raises
        ------
        TypeError
            Always raised to prevent instantiation.
        """
        raise TypeError(
            f"{self.__class__.__name__} is a namespace class and cannot be instantiated. "
            "Use its static methods directly: AnalyzerFactory.create_default(), etc."
        )

    @staticmethod
    def create_default() -> FaciesCorrelationAnalyzer:
        """Create an analyzer with default configuration.

        All processors are lazily initialized with default values.

        Returns
        -------
        FaciesCorrelationAnalyzer
            Analyzer with standard setup.

        Example
        -------
        >>> analyzer = AnalyzerFactory.create_default()
        """
        logger.info("Creating default FaciesCorrelationAnalyzer")
        return AnalyzerBuilder().build()

    @staticmethod
    def create_for_testing() -> FaciesCorrelationAnalyzer:
        """Create an analyzer configured for unit testing.

        All processors are lazily initialized with default values.

        Returns
        -------
        FaciesCorrelationAnalyzer
            Analyzer with testing-friendly defaults.

        Example
        -------
        >>> analyzer = AnalyzerFactory.create_for_testing()
        >>> # Use in unit tests
        """
        logger.info("Creating FaciesCorrelationAnalyzer for testing")
        config = FaciesCorrelationConfig()
        return AnalyzerBuilder().with_config(config).build()

    @staticmethod
    def builder() -> AnalyzerBuilder:
        """Create a new builder for custom configuration.

        The returned builder uses lazy initialization, so processors
        are only created during build() if not explicitly configured.

        Returns
        -------
        AnalyzerBuilder
            Fluent builder for step-by-step configuration.

        Example
        -------
        >>> builder = AnalyzerFactory.builder()
        >>> analyzer = (builder
        ...     .with_config(config)
        ...     .with_plotter(plotter)
        ...     .freeze()
        ...     .build())
        """
        return AnalyzerBuilder()

    @staticmethod
    def preset_debug() -> AnalyzerBuilder:
        """Create a builder configured for debugging with verbose logging.

        Useful for troubleshooting and development. Enables all debug logging
        and uses default configuration for easy testing.

        Returns
        -------
        AnalyzerBuilder
            Builder configured for debugging.

        Example
        -------
        >>> import logging
        >>> analyzer = AnalyzerFactory.preset_debug().build()
        """
        logger.info("Creating debug preset analyzer")
        AnalyzerBuilder.set_log_level(logging.DEBUG)
        return AnalyzerFactory.builder().with_config(FaciesCorrelationConfig())

    @staticmethod
    def preset_production() -> AnalyzerBuilder:
        """Create a builder configured for production use.

        Optimized for performance with minimal logging. Processors are
        lazily initialized on demand.

        Returns
        -------
        AnalyzerBuilder
            Builder configured for production.

        Example
        -------
        >>> analyzer = AnalyzerFactory.preset_production().build()
        """
        logger.info("Creating production preset analyzer")
        AnalyzerBuilder.set_log_level(logging.WARNING)
        return AnalyzerFactory.builder()

    @staticmethod
    def preset_minimal() -> AnalyzerBuilder:
        """Create a builder with minimal configuration (smallest memory footprint).

        All components are lazily initialized only when needed. Good for
        memory-constrained environments.

        Returns
        -------
        AnalyzerBuilder
            Builder with lazy initialization for all components.

        Example
        -------
        >>> analyzer = AnalyzerFactory.preset_minimal().build()
        """
        logger.info("Creating minimal preset analyzer")
        return AnalyzerFactory.builder()

    @staticmethod
    def preset_full() -> AnalyzerBuilder:
        """Create a builder with all components eagerly configured.

        Useful for benchmarking or ensuring all dependencies are available
        upfront. Freeze the builder to prevent accidental changes.

        Returns
        -------
        AnalyzerBuilder
            Builder ready to build with full configuration.

        Example
        -------
        >>> analyzer = (AnalyzerFactory.preset_full()
        ...     .with_config(config)
        ...     .freeze()
        ...     .build())
        """
        logger.info("Creating full preset analyzer")
        config = FaciesCorrelationConfig()
        return (
            AnalyzerFactory.builder()
            .with_config(config)
            .with_boundary_detector(BoundaryDetector())
            .with_cube_aligner(CubeAligner())
            .with_gradient_calculator(GradientCorrelationCalculator())
            .with_interface_analyzer(InterfaceReflectionAnalyzer())
            .with_facies_discriminator(FaciesDiscriminationCalculator())
            .with_domain_handler_factory(DomainHandlerFactory())
        )
