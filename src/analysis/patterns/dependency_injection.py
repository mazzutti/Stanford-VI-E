"""Dependency Injection Framework for Advanced Component Management

This module provides a comprehensive dependency injection container system
enabling loose coupling, improved testability, and flexible component configuration.

Patterns Used:
  - Dependency Injection: Invert control of object creation
  - Factory: Create instances with configuration
  - Registry: Central component management
  - Lifecycle Management: Handle singleton/transient/scoped instances

Example:
    >>> from src.analysis.patterns.dependency_injection import Container
    >>>
    >>> # Build container configuration
    >>> container = Container()
    >>> container.register_singleton("analyzer", FaciesCorrelationAnalyzer)
    >>> container.register_transient("processor", DataProcessor)
    >>>
    >>> # Resolve services
    >>> analyzer = container.resolve("analyzer")
    >>> processor = container.resolve("processor")
    >>>
    >>> # With fluent builder
    >>> container = (ContainerBuilder()
    ...     .register_singleton("analyzer", FaciesCorrelationAnalyzer)
    ...     .register_transient("processor", DataProcessor)
    ...     .build())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Generic
from enum import Enum
import logging
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)

__all__ = [
    "Lifecycle",
    "ServiceDescriptor",
    "Container",
    "ContainerBuilder",
    "ServiceProvider",
    "LifecycleManager",
    "RegistrationError",
    "ResolutionError",
]

T = TypeVar("T")


class Lifecycle(Enum):
    """Service lifecycle management strategies."""

    TRANSIENT = "transient"  # New instance every time
    SINGLETON = "singleton"  # Single instance, reused
    SCOPED = "scoped"  # Single instance per scope


class RegistrationError(Exception):
    """Raised when service registration fails."""

    pass


class ResolutionError(Exception):
    """Raised when service resolution fails."""

    pass


class ServiceDescriptor:
    """Describes a registered service with lifecycle and factory information."""

    def __init__(
        self,
        name: str,
        service_type: Type,
        implementation: Any,
        lifecycle: Lifecycle,
        factory: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
    ):
        """Initialize service descriptor.

        Args:
            name: Service name/key
            service_type: Service interface or base type
            implementation: Concrete implementation
            lifecycle: Lifecycle strategy (TRANSIENT, SINGLETON, SCOPED)
            factory: Optional factory function to create instances
            dependencies: Optional list of dependency names
        """
        self.name = name
        self.service_type = service_type
        self.implementation = implementation
        self.lifecycle = lifecycle
        self.factory = factory
        self.dependencies = dependencies or []
        self.instance = None  # For singleton caching
        self.creation_time = datetime.now()

    def __repr__(self) -> str:
        return (
            f"ServiceDescriptor("
            f"name={self.name!r}, "
            f"type={self.service_type.__name__}, "
            f"lifecycle={self.lifecycle.value})"
        )


class LifecycleManager:
    """Manages component lifecycle and instance caching."""

    def __init__(self):
        """Initialize lifecycle manager."""
        self._singletons: Dict[str, Any] = {}
        self._scopes: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def get_instance(
        self,
        descriptor: ServiceDescriptor,
        factory: Callable,
        scope_id: Optional[str] = None,
    ) -> Any:
        """Get or create instance based on lifecycle.

        Args:
            descriptor: Service descriptor
            factory: Factory function to create instances
            scope_id: Optional scope identifier

        Returns:
            Service instance
        """
        if descriptor.lifecycle == Lifecycle.TRANSIENT:
            return factory()

        elif descriptor.lifecycle == Lifecycle.SINGLETON:
            with self._lock:
                if descriptor.name not in self._singletons:
                    self._singletons[descriptor.name] = factory()
                return self._singletons[descriptor.name]

        elif descriptor.lifecycle == Lifecycle.SCOPED:
            if scope_id is None:
                raise ResolutionError(
                    f"Scoped service {descriptor.name!r} requires scope_id"
                )

            with self._lock:
                if scope_id not in self._scopes:
                    self._scopes[scope_id] = {}

                scope = self._scopes[scope_id]
                if descriptor.name not in scope:
                    scope[descriptor.name] = factory()
                return scope[descriptor.name]

        raise ResolutionError(f"Unknown lifecycle: {descriptor.lifecycle}")

    def clear_singletons(self):
        """Clear all singleton instances."""
        with self._lock:
            self._singletons.clear()
        logger.info("Singleton instances cleared")

    def clear_scope(self, scope_id: str):
        """Clear instances for a specific scope.

        Args:
            scope_id: Scope identifier
        """
        with self._lock:
            self._scopes.pop(scope_id, None)
        logger.debug(f"Scope {scope_id!r} cleared")


class ServiceProvider:
    """Provides access to registered services."""

    def __init__(
        self,
        descriptors: Dict[str, ServiceDescriptor],
        lifecycle_manager: LifecycleManager,
    ):
        """Initialize service provider.

        Args:
            descriptors: Dictionary of registered service descriptors
            lifecycle_manager: Lifecycle manager for instance management
        """
        self._descriptors = descriptors
        self._lifecycle_manager = lifecycle_manager
        self._resolution_stack: List[str] = []

    def resolve(
        self,
        service_name: str,
        scope_id: Optional[str] = None,
    ) -> Any:
        """Resolve a service by name with dependency injection.

        Args:
            service_name: Name of service to resolve
            scope_id: Optional scope identifier

        Returns:
            Service instance with dependencies injected

        Raises:
            ResolutionError: If service not found or circular dependency detected
        """
        # Check for circular dependencies
        if service_name in self._resolution_stack:
            cycle = " -> ".join(self._resolution_stack + [service_name])
            raise ResolutionError(f"Circular dependency detected: {cycle}")

        # Get service descriptor
        if service_name not in self._descriptors:
            raise ResolutionError(f"Service not registered: {service_name!r}")

        descriptor = self._descriptors[service_name]

        # Resolve dependencies
        self._resolution_stack.append(service_name)
        try:
            kwargs = {}
            for dep_name in descriptor.dependencies:
                kwargs[dep_name] = self.resolve(dep_name, scope_id)

            # Create factory
            if descriptor.factory:
                factory = lambda: descriptor.factory(**kwargs)
            else:
                factory = lambda: descriptor.implementation(**kwargs)

            # Get instance from lifecycle manager
            instance = self._lifecycle_manager.get_instance(
                descriptor,
                factory,
                scope_id,
            )

            logger.debug(f"Resolved service: {service_name!r}")
            return instance

        finally:
            self._resolution_stack.pop()

    def try_resolve(
        self,
        service_name: str,
        default: Any = None,
        scope_id: Optional[str] = None,
    ) -> Any:
        """Try to resolve a service, returning default if not found.

        Args:
            service_name: Name of service to resolve
            default: Default value if service not found
            scope_id: Optional scope identifier

        Returns:
            Service instance or default value
        """
        try:
            return self.resolve(service_name, scope_id)
        except ResolutionError:
            return default


class Container:
    """Dependency injection container for service registration and resolution."""

    def __init__(self):
        """Initialize container."""
        self._descriptors: Dict[str, ServiceDescriptor] = {}
        self._lifecycle_manager = LifecycleManager()
        self._service_provider = ServiceProvider(
            self._descriptors,
            self._lifecycle_manager,
        )
        logger.info("Dependency Injection Container initialized")

    def register_singleton(
        self,
        name: str,
        implementation: Type[T],
        factory: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
    ) -> Container:
        """Register a singleton service.

        Args:
            name: Service name
            implementation: Implementation class or instance
            factory: Optional factory function
            dependencies: Optional list of dependency names

        Returns:
            Self for chaining
        """
        descriptor = ServiceDescriptor(
            name=name,
            service_type=implementation,
            implementation=implementation,
            lifecycle=Lifecycle.SINGLETON,
            factory=factory,
            dependencies=dependencies,
        )
        self._descriptors[name] = descriptor
        logger.debug(f"Registered singleton: {name!r}")
        return self

    def register_transient(
        self,
        name: str,
        implementation: Type[T],
        factory: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
    ) -> Container:
        """Register a transient service (new instance each time).

        Args:
            name: Service name
            implementation: Implementation class
            factory: Optional factory function
            dependencies: Optional list of dependency names

        Returns:
            Self for chaining
        """
        descriptor = ServiceDescriptor(
            name=name,
            service_type=implementation,
            implementation=implementation,
            lifecycle=Lifecycle.TRANSIENT,
            factory=factory,
            dependencies=dependencies,
        )
        self._descriptors[name] = descriptor
        logger.debug(f"Registered transient: {name!r}")
        return self

    def register_scoped(
        self,
        name: str,
        implementation: Type[T],
        factory: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
    ) -> Container:
        """Register a scoped service (single instance per scope).

        Args:
            name: Service name
            implementation: Implementation class
            factory: Optional factory function
            dependencies: Optional list of dependency names

        Returns:
            Self for chaining
        """
        descriptor = ServiceDescriptor(
            name=name,
            service_type=implementation,
            implementation=implementation,
            lifecycle=Lifecycle.SCOPED,
            factory=factory,
            dependencies=dependencies,
        )
        self._descriptors[name] = descriptor
        logger.debug(f"Registered scoped: {name!r}")
        return self

    def resolve(self, service_name: str, scope_id: Optional[str] = None) -> Any:
        """Resolve a service by name.

        Args:
            service_name: Name of service to resolve
            scope_id: Optional scope identifier

        Returns:
            Service instance
        """
        return self._service_provider.resolve(service_name, scope_id)

    def try_resolve(
        self,
        service_name: str,
        default: Any = None,
        scope_id: Optional[str] = None,
    ) -> Any:
        """Try to resolve a service, returning default if not found.

        Args:
            service_name: Service name
            default: Default value
            scope_id: Optional scope identifier

        Returns:
            Service instance or default
        """
        return self._service_provider.try_resolve(service_name, default, scope_id)

    def is_registered(self, service_name: str) -> bool:
        """Check if a service is registered.

        Args:
            service_name: Service name

        Returns:
            True if registered, False otherwise
        """
        return service_name in self._descriptors

    def get_services(self) -> Dict[str, ServiceDescriptor]:
        """Get all registered services.

        Returns:
            Dictionary of service descriptors
        """
        return self._descriptors.copy()

    def clear(self):
        """Clear all registrations and cached instances."""
        self._descriptors.clear()
        self._lifecycle_manager.clear_singletons()
        logger.info("Container cleared")

    def __repr__(self) -> str:
        return (
            f"Container("
            f"services={len(self._descriptors)}, "
            f"singletons={len(self._lifecycle_manager._singletons)})"
        )


class ContainerBuilder:
    """Fluent builder for Container configuration."""

    def __init__(self):
        """Initialize container builder."""
        self._container = Container()

    def register_singleton(
        self,
        name: str,
        implementation: Type[T],
        factory: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
    ) -> ContainerBuilder:
        """Register singleton service.

        Args:
            name: Service name
            implementation: Implementation class
            factory: Optional factory function
            dependencies: Optional list of dependencies

        Returns:
            Self for chaining
        """
        self._container.register_singleton(
            name,
            implementation,
            factory,
            dependencies,
        )
        return self

    def register_transient(
        self,
        name: str,
        implementation: Type[T],
        factory: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
    ) -> ContainerBuilder:
        """Register transient service.

        Args:
            name: Service name
            implementation: Implementation class
            factory: Optional factory function
            dependencies: Optional list of dependencies

        Returns:
            Self for chaining
        """
        self._container.register_transient(
            name,
            implementation,
            factory,
            dependencies,
        )
        return self

    def register_scoped(
        self,
        name: str,
        implementation: Type[T],
        factory: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
    ) -> ContainerBuilder:
        """Register scoped service.

        Args:
            name: Service name
            implementation: Implementation class
            factory: Optional factory function
            dependencies: Optional list of dependencies

        Returns:
            Self for chaining
        """
        self._container.register_scoped(
            name,
            implementation,
            factory,
            dependencies,
        )
        return self

    def build(self) -> Container:
        """Build and return configured container.

        Returns:
            Configured Container instance
        """
        logger.info(
            f"Built container with {len(self._container._descriptors)} "
            f"registered services"
        )
        return self._container
