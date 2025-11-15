"""Consolidated factory pattern implementation.

This module unifies builder, factory, and component factory implementations
across the analysis module, reducing duplication while maintaining full
extensibility through OOP principles.

Patterns:
  - Builder: Fluent API for complex object construction
  - Factory: Centralized object creation
  - Strategy: Pluggable creation strategies
  - Composite: Combine multiple builders

Savings:
  - Consolidates builder.py, factory.py, patterns/builder.py, factories/
  - Eliminates ~500 LOC of duplicate factory logic
  - Provides single source of truth for object creation

Example:
    >>> # Fluent builder
    >>> analyzer = (FluentBuilder("facies")
    ...     .with_config(config)
    ...     .with_cache_loader(loader)
    ...     .build())
    >>>
    >>> # Service factory
    >>> factory = ServiceFactory()
    >>> analyzer = factory.create("FaciesAnalyzer", config=config)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    cast,
)
from collections.abc import Callable

if TYPE_CHECKING:
    from src.analysis.models.config import FaciesCorrelationConfig

logger = logging.getLogger(__name__)

__all__ = [
    "FluentBuilder",
    "BuildableFactory",
    "ServiceFactory",
    "AnalyzerFactory",
    "ComponentBuilder",
]

T = TypeVar("T")  # Generic type for builder output


# ============================================================================
# Buildable Protocol & Base Classes
# ============================================================================


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
        ...


# ============================================================================
# Fluent Builder (Generic)
# ============================================================================


@dataclass
class FluentBuilder(Generic[T]):
    """Generic fluent builder for complex objects.

    Provides method chaining API for building objects step-by-step.
    Replaces duplicate builder implementations across codebase.

    Example:
        >>> builder = FluentBuilder("MyComponent")
        >>> obj = (builder
        ...     .with_config(my_config)
        ...     .with_dependency(my_dep)
        ...     .build())
    """

    name: str = "component"
    _components: dict[str, Any] = field(
        default_factory=lambda: cast(dict[str, Any], {})
    )
    _config: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))
    _validators: list[Callable[[dict[str, Any]], bool]] = field(
        default_factory=lambda: cast(list[Callable[[dict[str, Any]], bool]], [])
    )

    def with_config(self, config: Any) -> FluentBuilder[T]:
        """Set component configuration.

        Args:
            config: Configuration object or dict

        Returns:
            Self for chaining
        """
        self._config["main"] = config
        logger.debug(f"{self.name}: Set configuration")
        return self

    def with_component(self, name: str, component: Any) -> FluentBuilder[T]:
        """Add named component dependency.

        Args:
            name: Component identifier
            component: Component instance

        Returns:
            Self for chaining
        """
        if not component:
            raise ValueError(f"Component '{name}' cannot be None")
        self._components[name] = component
        logger.debug(f"{self.name}: Added component '{name}'")
        return self

    def with_dependencies(self, **deps: Any) -> FluentBuilder[T]:
        """Add multiple component dependencies.

        Args:
            **deps: Keyword arguments mapping names to components

        Returns:
            Self for chaining
        """
        for name, comp in deps.items():
            self.with_component(name, comp)
        return self

    def with_validator(
        self, validator: Callable[[dict[str, Any]], bool]
    ) -> FluentBuilder[T]:
        """Add configuration validator.

        Args:
            validator: Validation function

        Returns:
            Self for chaining
        """
        self._validators.append(validator)
        return self

    def validate(self) -> bool:
        """Run all validators on configuration.

        Returns:
            True if all validators pass

        Raises:
            ValueError: If validation fails
        """
        for validator in self._validators:
            if not validator(self._config):
                raise ValueError(f"Validation failed: {validator.__name__}")
        return True

    def reset(self) -> FluentBuilder[T]:
        """Reset builder to initial state.

        Returns:
            Self for chaining
        """
        self._components.clear()
        self._config.clear()
        logger.debug(f"{self.name}: Reset")
        return self

    @abstractmethod
    def build(self) -> T:
        """Build and return configured component."""
        raise NotImplementedError("Subclasses must implement build()")


# ============================================================================
# Factory Implementations
# ============================================================================


class BuildableFactory(ABC, Generic[T]):
    """Base factory for creating buildable objects.

    Consolidates factory creation logic with builder pattern support.
    """

    def __init__(self) -> None:
        """Initialize factory."""
        self._builders: dict[str, type[FluentBuilder[Any]]] = {}
        self._creators: dict[str, Callable[..., T]] = {}

    def register_builder(
        self,
        name: str,
        builder_class: type[FluentBuilder[Any]],
    ) -> None:
        """Register a builder class.

        Args:
            name: Builder identifier
            builder_class: Builder class to register
        """
        self._builders[name] = builder_class
        logger.debug(f"Registered builder: {name}")

    def register_creator(
        self,
        name: str,
        creator: Callable[..., T],
    ) -> None:
        """Register a creation function.

        Args:
            name: Creator identifier
            creator: Creation function
        """
        self._creators[name] = creator
        logger.debug(f"Registered creator: {name}")

    def create_with_builder(
        self,
        builder_name: str,
        **builder_args: Any,
    ) -> Any:
        """Create object using registered builder.

        Args:
            builder_name: Name of registered builder
            **builder_args: Arguments passed to builder.build()

        Returns:
            Created object

        Raises:
            KeyError: If builder not registered
        """
        if builder_name not in self._builders:
            raise KeyError(f"Builder not found: {builder_name}")

        builder = self._builders[builder_name]()
        return builder.build(**builder_args)

    def create_with_creator(
        self,
        creator_name: str,
        **creator_args: Any,
    ) -> T:
        """Create object using registered creator function.

        Args:
            creator_name: Name of registered creator
            **creator_args: Arguments for creator function

        Returns:
            Created object

        Raises:
            KeyError: If creator not registered
        """
        if creator_name not in self._creators:
            raise KeyError(f"Creator not found: {creator_name}")

        return self._creators[creator_name](**creator_args)


class ServiceFactory:
    """Factory for creating named services/components.

    Provides registry-based object creation with support for both
    builder and direct creation patterns.
    """

    def __init__(self) -> None:
        """Initialize service factory."""
        self._services: dict[str, Callable[..., Any]] = {}
        self._instances: dict[str, Any] = {}  # For singletons
        self._singletons: set[str] = set()  # Track which services are singletons

    def register(
        self,
        name: str,
        creator: Callable[..., Any],
        singleton: bool = False,
    ) -> None:
        """Register a service creator.

        Args:
            name: Service name
            creator: Function that creates service
            singleton: If True, only create once and reuse
        """
        self._services[name] = creator
        if singleton:
            self._singletons.add(name)
        logger.debug(f"Registered service: {name} (singleton={singleton})")

    def create(self, name: str, **kwargs: Any) -> Any:
        """Create or retrieve a service.

        Args:
            name: Service name
            **kwargs: Arguments for creator function

        Returns:
            Service instance

        Raises:
            KeyError: If service not registered
        """
        if name not in self._services:
            raise KeyError(f"Service not found: {name}")

        if name in self._singletons:
            if name not in self._instances:
                self._instances[name] = self._services[name](**kwargs)
            return self._instances[name]

        return self._services[name](**kwargs)

    def clear_singletons(self) -> None:
        """Clear all singleton instances."""
        self._instances.clear()
        logger.debug("Cleared singleton instances")


class AnalyzerFactory(BuildableFactory[Any]):
    """Specialized factory for creating analyzers.

    Consolidates analyzer creation logic with support for
    multiple analyzer types and configurations.
    """

    def create_facies_analyzer(
        self,
        config: FaciesCorrelationConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create facies correlation analyzer.

        Args:
            config: Analysis configuration
            **kwargs: Additional creation arguments

        Returns:
            FaciesCorrelationAnalyzer instance
        """
        from src.analysis.facies.analyzer import FaciesCorrelationAnalyzer

        if config is None:
            from src.analysis.facies.config import FaciesAnalysisConfig

            # FaciesAnalysisConfig and FaciesCorrelationConfig types differ; cast to Any
            config = cast(Any, FaciesAnalysisConfig())

        return FaciesCorrelationAnalyzer(config=config)

    def create_rock_physics_analyzer(
        self,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create rock physics analyzer.

        Args:
            config: Analysis configuration
            **kwargs: Additional creation arguments

        Returns:
            RockPhysicsAnalyzer instance
        """
        from src.analysis.rock_physics.analyzer import RockPhysicsAnalyzer

        return RockPhysicsAnalyzer(config=config) if config else RockPhysicsAnalyzer()


# ============================================================================
# Helper Functions
# ============================================================================


def create_analyzer(
    analyzer_type: str = "facies",
    config: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Convenience function to create analyzer with minimal setup.

    Args:
        analyzer_type: Type of analyzer ('facies', 'rock_physics')
        config: Analysis configuration
        **kwargs: Additional creation arguments

    Returns:
        Configured analyzer instance

    Example:
        >>> analyzer = create_analyzer("facies", config=my_config)
    """
    factory = AnalyzerFactory()

    if analyzer_type == "facies":
        return factory.create_facies_analyzer(config=config, **kwargs)
    elif analyzer_type == "rock_physics":
        return factory.create_rock_physics_analyzer(config=config, **kwargs)
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")


# ============================================================================
# Component Builder Base Class
# ============================================================================


class ComponentBuilder(Buildable[T], ABC):
    """Abstract component builder combining builder and buildable patterns.

    Provides template method pattern for standard build workflow.
    """

    def __init__(self, name: str = "component"):
        """Initialize builder.

        Args:
            name: Component name for logging
        """
        self.name = name
        self._config: dict[str, Any] = {}
        self._dependencies: dict[str, Any] = {}

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration is complete. Override in subclasses."""
        pass

    @abstractmethod
    def _create_component(self) -> T:
        """Create the component. Override in subclasses."""
        pass

    def build(self) -> T:
        """Template method for building component.

        Steps:
            1. Validate configuration
            2. Create component
            3. Log creation

        Returns:
            Built component
        """
        self._validate_config()
        component = self._create_component()
        logger.debug(f"Built component: {self.name}")
        return component
