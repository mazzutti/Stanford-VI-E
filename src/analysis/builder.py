"""Fluent builder pattern for simplified analyzer configuration.

This module provides a unified builder pattern that simplifies the creation
and configuration of analyzers across the analysis module. Eliminates boilerplate
for initialization, dependency injection, and configuration setup.

Design Pattern:
    - Builder: Fluent API for constructing analyzers
    - Method chaining: Each builder method returns self for fluent interface
    - Immutable intermediates: Each step creates a new builder state
    - Late binding: Configuration applied only when build() called

Benefits:
    - Eliminates repetitive __init__ patterns across analyzers
    - Cleaner, more readable analyzer instantiation code
    - Consistent dependency injection across module
    - Easier to test with mock dependencies

Example:
    >>> analyzer = (AnalysisBuilder("facies")
    ...     .with_resampler_factory(custom_resampler)
    ...     .with_cache_loader(custom_loader)
    ...     .with_config(config)
    ...     .build())
"""

# This module intentionally performs local (call-time) imports inside
# builder/factory methods to avoid import-time side-effects and cycles.
# Silence the related pylint checks for this module (keep import-outside-toplevel).
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

if TYPE_CHECKING:
    from src.analysis.base import AnalyzerInterface
    from src.analysis.rock_physics import RockPhysicsAnalyzer

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisBuilder",
    "Buildable",
]

T = TypeVar("T")  # legacy/placeholder TypeVar (unused for builder)
# R is the concrete analyzer return type used by build()
R = TypeVar("R")

class Buildable(ABC, Generic[T]):
    """Protocol for buildable objects.

    Objects that can be constructed through a builder pattern.
    """

    @abstractmethod
    def build(self) -> T:
        """Construct and return the built object.

        Returns
        -------
        T
            Constructed object with all configured settings

        Raises
        ------
        ValueError
            If required configuration is missing
        """
        raise NotImplementedError()

    # Builder helpers are intentionally compact; they provide fluent wiring
    # and are expected to remain minimal.

@dataclass
class AnalysisBuilder:
    """Fluent builder for analyzer instantiation and configuration.

    Provides a clean, chainable API for creating analyzers with proper
    dependency injection, eliminating boilerplate initialization code.

    Attributes
    ----------
    domain : str
        Analysis domain identifier (e.g., 'facies', 'rock_physics')
    dependencies : dict
        Configured dependencies for the analyzer
    config : optional
        Analyzer configuration object

    Example
    -------
    >>> from src.analysis import AnalysisBuilder
    >>> analyzer = (AnalysisBuilder("facies")
    ...     .with_resampler(my_resampler)
    ...     .with_cache_loader(my_loader)
    ...     .with_config(my_config)
    ...     .build_facies_analyzer())

    >>> # For custom analyzers
    >>> custom = (AnalysisBuilder("custom")
    ...     .with_dependency("factory", my_factory)
    ...     .with_dependency("loader", my_loader)
    ...     .build(AnalyzerClass))
    """

    domain: str
    dependencies: dict[str, Any] = field(
        default_factory=lambda: cast(dict[str, Any], {})
    )
    config: Any | None = None
    _metadata: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))

    def with_dependency(self, name: str, dependency: Any) -> AnalysisBuilder:
        """Register a dependency for the analyzer.

        Dependencies are injected into the analyzer during build().

        Parameters
        ----------
        name : str
            Dependency identifier (e.g., 'resampler_factory', 'cache_loader')
        dependency : Any
            Dependency object to inject

        Returns
        -------
        AnalysisBuilder
            Self for method chaining

        Example
        -------
        >>> builder.with_dependency("factory", MyFactory())
        """
        self.dependencies[name] = dependency
        logger.debug("Registered dependency '%s' for domain '%s'", name, self.domain)
        return self

    def with_resampler(self, resampler: Any) -> AnalysisBuilder:
        """Configure resampler dependency (convenience method).

        Parameters
        ----------
        resampler : Any
            Resampler instance or factory

        Returns
        -------
        AnalysisBuilder
            Self for method chaining
        """
        return self.with_dependency("resampler_factory", resampler)

    def with_cache_loader(self, loader: Any) -> AnalysisBuilder:
        """Configure cache loader dependency (convenience method).

        Parameters
        ----------
        loader : Any
            Cache loader instance

        Returns
        -------
        AnalysisBuilder
            Self for method chaining
        """
        return self.with_dependency("cache_loader", loader)

    def with_plotter(self, plotter: Any) -> AnalysisBuilder:
        """Configure plotter dependency (convenience method).

        Parameters
        ----------
        plotter : Any
            Plotter instance

        Returns
        -------
        AnalysisBuilder
            Self for method chaining
        """
        return self.with_dependency("plotter", plotter)

    def with_config(self, config: Any) -> AnalysisBuilder:
        """Set the analyzer configuration.

        Configuration is applied to the analyzer during build().

        Parameters
        ----------
        config : Any
            Configuration object specific to the analyzer

        Returns
        -------
        AnalysisBuilder
            Self for method chaining
        """
        self.config = config
        logger.debug(
            "Set configuration %s for domain '%s'", type(config).__name__, self.domain
        )
        return self

    def with_metadata(self, key: str, value: Any) -> AnalysisBuilder:
        """Add arbitrary metadata (useful for debugging/logging).

        Parameters
        ----------
        key : str
            Metadata key
        value : Any
            Metadata value

        Returns
        -------
        AnalysisBuilder
            Self for method chaining
        """
        self._metadata[key] = value
        return self

    def build(self, analyzer_class: type[Any]) -> Any:
        """Build analyzer with registered dependencies and config.

        Instantiates the analyzer class with all configured dependencies
        and applies the configuration if set.

        Parameters
        ----------
        analyzer_class : Type[T]
            Analyzer class to instantiate

        Returns
        -------
        T
            Configured analyzer instance

        Raises
        ------
        TypeError
            If analyzer_class constructor doesn't accept the configured dependencies
        ValueError
            If required dependencies are missing

        Example
        -------
        >>> from src.analysis.facies import FaciesCorrelationAnalyzer
        >>> analyzer = (AnalysisBuilder("facies")
        ...     .with_resampler(my_resampler)
        ...     .build(FaciesCorrelationAnalyzer))
        """
        try:
            # Try to instantiate with keyword arguments
            instance = analyzer_class(**self.dependencies)

            # Apply configuration if set and analyzer supports it
            if self.config is not None and hasattr(instance, "configure"):
                cfg = getattr(instance, "configure")
                if callable(cfg):
                    cfg(self.config)
                    logger.debug("Applied configuration to %s", analyzer_class.__name__)

            # Store metadata in instance for traceability
            if self._metadata and hasattr(instance, "_builder_metadata"):
                setattr(instance, "_builder_metadata", self._metadata)

            logger.info(
                "Built analyzer '%s' of type %s with %d dependencies",
                self.domain,
                analyzer_class.__name__,
                len(self.dependencies),
            )
            return instance

        except TypeError as e:
            logger.error(
                "Failed to build %s: %s. Available dependencies: %s",
                analyzer_class.__name__,
                e,
                list(self.dependencies.keys()),
            )
            raise ValueError(
                f"Cannot instantiate {analyzer_class.__name__} with configured "
                f"dependencies. Missing required parameter? Error: {e}"
            ) from e

    def build_facies_analyzer(self) -> AnalyzerInterface[Any, Any]:
        """Build a FaciesCorrelationAnalyzer (convenience method).

        Returns
        -------
        AnalyzerInterface
            Configured FaciesCorrelationAnalyzer instance
        """
        # Lazy import to avoid import-time cycles when constructing analyzers

        from src.analysis.facies.analyzer import FaciesCorrelationAnalyzer

        # (previously disabled import-outside-toplevel here; now removed as redundant)

        return cast(
            "AnalyzerInterface[Any, Any]", self.build(FaciesCorrelationAnalyzer)
        )

    def build_rock_physics_analyzer(self) -> RockPhysicsAnalyzer:
        """Build a RockPhysicsAnalyzer (convenience method).

        Returns
        -------
        RockPhysicsAnalyzer
            Configured RockPhysicsAnalyzer instance
        """
        # Lazy import to avoid import-time cycles when constructing analyzers

        from src.analysis.rock_physics import RockPhysicsAnalyzer

        # (previously disabled import-outside-toplevel here; now removed as redundant)

        return cast("RockPhysicsAnalyzer", self.build(RockPhysicsAnalyzer))

    def clone(self) -> AnalysisBuilder:
        """Create a copy of this builder for reuse/modification.

        Returns
        -------
        AnalysisBuilder
            New builder with same configuration

        Example
        -------
        >>> builder1 = AnalysisBuilder("facies").with_config(config1)
        >>> builder2 = builder1.clone().with_config(config2)
        """
        return AnalysisBuilder(
            domain=self.domain,
            dependencies=self.dependencies.copy(),
            config=self.config,
            _metadata=self._metadata.copy(),
        )

    def summary(self) -> str:
        """Get human-readable summary of builder state.

        Returns
        -------
        str
            Summary of configured dependencies, config, and metadata
        """
        config_name = type(self.config).__name__ if self.config else "Not configured"
        return (
            f"AnalysisBuilder(domain={self.domain!r}, "
            f"dependencies={len(self.dependencies)}, "
            f"config={config_name})"
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return self.summary()

    def __str__(self) -> str:
        """Return human-readable string."""
        return self.summary()

# Convenience factory functions for common builder patterns
def build_facies_analyzer(**kwargs: Any) -> AnalyzerInterface[Any, Any]:
    """Create a FaciesCorrelationAnalyzer quickly.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments passed to AnalysisBuilder.with_dependency()
        Common keys: resampler_factory, cache_loader, plotter, config

    Returns
    -------
    AnalyzerInterface
        Configured FaciesCorrelationAnalyzer

    Example
    -------
    >>> analyzer = build_facies_analyzer(
    ...     resampler_factory=my_resampler,
    ...     config=my_config
    ...
    >>> custom = (AnalysisBuilder("custom")
    ...     .with_dependency("factory", my_factory)
    ...     .with_dependency("loader", my_loader)
    ...     .build(AnalyzerClass))
    """

    # Build a default AnalysisBuilder for facies and apply any provided
    # keyword dependencies (mirrors `build_rock_physics_analyzer` behavior).
    builder = AnalysisBuilder("facies")
    for key, value in kwargs.items():
        if key == "config":
            builder = builder.with_config(value)
        else:
            builder = builder.with_dependency(key, value)

    return builder.build_facies_analyzer()

def build_rock_physics_analyzer(**kwargs: Any) -> RockPhysicsAnalyzer:
    """Create a RockPhysicsAnalyzer quickly.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments passed to AnalysisBuilder.with_dependency()

    Returns
    -------
    RockPhysicsAnalyzer
        Configured RockPhysicsAnalyzer
    """
    builder = AnalysisBuilder("rock_physics")
    for key, value in kwargs.items():
        if key == "config":
            builder = builder.with_config(value)
        else:
            builder = builder.with_dependency(key, value)
    return builder.build_rock_physics_analyzer()
