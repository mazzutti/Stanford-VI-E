"""Service Container for Dependency Injection.

This module configures the dependency injection container with all services,
enabling loose coupling and easy testing through constructor injection.

Services Registered:
  - Analyzers (FaciesCorrelationAnalyzer, RockPhysicsAnalyzer)
  - Configuration managers (ConfigManager)
  - Cache managers
  - Event buses
  - Circuit breakers
  - Processors

Usage:
    >>> from src.analysis.service_container import create_container, get_service_provider
    >>> container = create_container()
    >>> provider = container.create_service_provider()
    >>> analyzer = provider.resolve('FaciesCorrelationAnalyzer')
"""

from __future__ import annotations

import logging
from typing import Optional

from src.analysis.patterns.dependency_injection import (
    Container,
    ContainerBuilder,
)
from src.analysis.patterns.dependency_injection import ServiceProvider
from src.analysis.patterns.event_bus import EventBus
from src.analysis.patterns.circuit_breaker import CircuitBreakerPool
from src.analysis.config_manager import ConfigManager
from src.analysis.facies.analyzer import FaciesCorrelationAnalyzer

logger = logging.getLogger(__name__)

__all__ = [
    "create_container",
    "create_service_provider",
    "ServiceContainerBuilder",
    "get_default_container",
]

# Global container instance
_default_container: Optional[Container] = None


def create_container() -> Container:
    """Create and configure the service container.

    Returns:
        Configured Container instance
    """
    logger.info("Creating service container")

    builder = ContainerBuilder()

    # Register event bus as singleton
    builder.register_singleton(
        "EventBus",
        EventBus,
        factory=lambda: EventBus(),
    )

    # Register circuit breaker pool as singleton
    builder.register_singleton(
        "CircuitBreakerPool",
        CircuitBreakerPool,
        factory=lambda: CircuitBreakerPool(),
    )

    # Register configuration manager as singleton
    builder.register_singleton(
        "ConfigManager",
        ConfigManager,
        factory=lambda: ConfigManager(),
    )

    # Register FaciesCorrelationAnalyzer as transient (no dependencies required)
    builder.register_transient(
        "FaciesCorrelationAnalyzer",
        FaciesCorrelationAnalyzer,
        factory=lambda: FaciesCorrelationAnalyzer(),
    )

    # Register IntegratedAnalyzer as transient
    # This will be done in a separate module to avoid circular imports
    logger.debug("Service container created successfully")

    return builder.build()


def create_service_provider(container: Optional[Container] = None) -> ServiceProvider:
    """Create a service provider from a container.

    Args:
        container: Optional container (creates default if not provided)

    Returns:
        ServiceProvider instance
    """
    if container is None:
        container = create_container()

    logger.debug("Creating service provider")
    # Access the container's service provider instance directly
    return container.service_provider


def get_default_container() -> Container:
    """Get or create the default global container.

    Returns:
        Global Container instance
    """
    global _default_container

    if _default_container is None:
        _default_container = create_container()
        logger.info("Default container created")

    return _default_container


def reset_default_container() -> None:
    """Reset the default global container."""
    global _default_container
    _default_container = None
    logger.info("Default container reset")


class ServiceContainerBuilder:
    """Helper class for building customized service containers.

    Allows for easy extension and customization of the DI container.
    """

    def __init__(self) -> None:
        """Initialize builder."""
        self.builder = ContainerBuilder()
        self._event_bus: Optional[EventBus] = None
        self._circuit_breaker_pool: Optional[CircuitBreakerPool] = None
        self._config_manager: Optional[ConfigManager] = None

    def with_event_bus(
        self, event_bus: Optional[EventBus] = None
    ) -> ServiceContainerBuilder:
        """Configure event bus.

        Args:
            event_bus: Optional EventBus instance (creates new if not provided)

        Returns:
            Self for chaining
        """
        if event_bus is None:
            event_bus = EventBus()

        self._event_bus = event_bus
        self.builder.register_singleton(
            "EventBus",
            EventBus,
            factory=lambda: event_bus,
        )
        logger.debug("EventBus registered")
        return self

    def with_circuit_breaker_pool(
        self, pool: Optional[CircuitBreakerPool] = None
    ) -> ServiceContainerBuilder:
        """Configure circuit breaker pool.

        Args:
            pool: Optional CircuitBreakerPool (creates new if not provided)

        Returns:
            Self for chaining
        """
        if pool is None:
            pool = CircuitBreakerPool()

        self._circuit_breaker_pool = pool
        self.builder.register_singleton(
            "CircuitBreakerPool",
            CircuitBreakerPool,
            factory=lambda: pool,
        )
        logger.debug("CircuitBreakerPool registered")
        return self

    def with_config_manager(
        self, config_manager: Optional[ConfigManager] = None
    ) -> ServiceContainerBuilder:
        """Configure config manager.

        Args:
            config_manager: Optional ConfigManager (creates new if not provided)

        Returns:
            Self for chaining
        """
        if config_manager is None:
            config_manager = ConfigManager()

        self._config_manager = config_manager
        self.builder.register_singleton(
            "ConfigManager",
            ConfigManager,
            factory=lambda: config_manager,
        )
        logger.debug("ConfigManager registered")
        return self

    def with_facies_analyzer(self) -> ServiceContainerBuilder:
        """Register facies analyzer dependencies.

        Returns:
            Self for chaining
        """
        # Ensure dependencies are registered
        if self._config_manager is None:
            self.with_config_manager()

        self.builder.register_transient(
            "FaciesCorrelationAnalyzer",
            FaciesCorrelationAnalyzer,
            factory=lambda: FaciesCorrelationAnalyzer(),
        )

        logger.debug("FaciesCorrelationAnalyzer registered")
        return self

    def build(self) -> Container:
        """Build the container.

        Returns:
            Configured Container instance
        """
        # Ensure default services are registered if not explicitly configured
        if self._event_bus is None:
            self.with_event_bus()

        if self._circuit_breaker_pool is None:
            self.with_circuit_breaker_pool()

        if self._config_manager is None:
            self.with_config_manager()

        logger.info("Service container built successfully")
        return self.builder.build()

    def build_provider(self) -> ServiceProvider:
        """Build and return service provider.

        Returns:
            ServiceProvider instance
        """
        container = self.build()
        # Access the container's service provider instance directly
        return container.service_provider
