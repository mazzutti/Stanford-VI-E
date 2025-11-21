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

import logging
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from functools import partial
from threading import Lock
from typing import Any, TypeVar

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

class ResolutionError(Exception):
    """Raised when service resolution fails."""

class ServiceDescriptor:
    """Describes a registered service with lifecycle and factory information."""

    def __init__(
        self,
        name: str,
        service_type: type[Any],
        implementation: Any,
        lifecycle: Lifecycle,
        factory: Callable[..., Any] | None = None,
        dependencies: list[str] | None = None,
    ) -> None:
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

    def __init__(self) -> None:
        """Initialize lifecycle manager."""
        self._singletons: dict[str, Any] = {}
        self._scopes: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def get_instance(
        self,
        descriptor: ServiceDescriptor,
        factory: Callable[[], Any],
        scope_id: str | None = None,
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

    def clear_singletons(self) -> None:
        """Clear all singleton instances."""
        with self._lock:
            self._singletons.clear()
        logger.info("Singleton instances cleared")

    def singleton_count(self) -> int:
        """Return number of registered singleton instances."""
        with self._lock:
            return len(self._singletons)

    def clear_scope(self, scope_id: str) -> None:
        """Clear instances for a specific scope.

        Args:
            scope_id: Scope identifier
        """
        with self._lock:
            self._scopes.pop(scope_id, None)
        logger.debug("Scope %r cleared", scope_id)

class ServiceProvider:
    """Provides access to registered services."""

    def __init__(
        self,
        descriptors: dict[str, ServiceDescriptor],
        lifecycle_manager: LifecycleManager,
    ) -> None:
        """Initialize service provider.

        Args:
            descriptors: Dictionary of registered service descriptors
            lifecycle_manager: Lifecycle manager for instance management
        """
        self._descriptors = descriptors
        self._lifecycle_manager = lifecycle_manager
        self._resolution_stack: list[str] = []

    def resolve(
        self,
        service_name: str,
        scope_id: str | None = None,
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
            kwargs: dict[str, Any] = {}
            for dep_name in descriptor.dependencies:
                kwargs[dep_name] = self.resolve(dep_name, scope_id)

            # Create factory as a zero-argument callable using functools.partial
            if descriptor.factory is not None:
                fact = descriptor.factory
                factory = partial(fact, **kwargs)
            else:
                impl = descriptor.implementation
                factory = partial(impl, **kwargs)

            # Get instance from lifecycle manager
            instance = self._lifecycle_manager.get_instance(
                descriptor,
                factory,
                scope_id,
            )

            logger.debug("Resolved service: %r", service_name)
            return instance

        finally:
            self._resolution_stack.pop()

    def try_resolve(
        self,
        service_name: str,
        default: Any = None,
        scope_id: str | None = None,
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

    def __init__(self) -> None:
        """Initialize container."""
        self._descriptors: dict[str, ServiceDescriptor] = {}
        self._lifecycle_manager = LifecycleManager()
        self._service_provider = ServiceProvider(
            self._descriptors,
            self._lifecycle_manager,
        )
        logger.info("Dependency Injection Container initialized")

    def clear(self) -> None:
        """Clear all registrations and cached instances."""
        self._descriptors.clear()
        self._lifecycle_manager.clear_singletons()
        logger.info("Container cleared")

    def register_singleton(
        self,
        name: str,
        implementation: type[T],
        factory: Callable[..., Any] | None = None,
        dependencies: list[str] | None = None,
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
        logger.debug("Registered singleton: %r", name)
        return self

    def register_transient(
        self,
        name: str,
        implementation: type[T],
        factory: Callable[..., Any] | None = None,
        dependencies: list[str] | None = None,
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
        logger.debug("Registered transient: %r", name)
        return self

    def register_scoped(
        self,
        name: str,
        implementation: type[T],
        factory: Callable[..., Any] | None = None,
        dependencies: list[str] | None = None,
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
        logger.debug("Registered scoped: %r", name)
        return self

    def resolve(self, service_name: str, scope_id: str | None = None) -> Any:
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
        scope_id: str | None = None,
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

    def get_services(self) -> dict[str, ServiceDescriptor]:
        """Get all registered services.

        Returns:
            Dictionary of service descriptors
        """
        return self._descriptors.copy()

    @property
    def service_provider(self) -> ServiceProvider:
        """Return the underlying ServiceProvider instance.

        This provides read-only access to the container's service resolver
        for advanced use-cases (testing, inspection, or custom resolution
        flows).
        """
        return self._service_provider

    # Note: `clear` implemented above at construction time; keep __repr__ below.

    def __repr__(self) -> str:
        # Use public accessors to avoid referencing protected attributes
        services = len(self.get_services())
        singletons = self._lifecycle_manager.singleton_count()
        return f"Container(services={services}, singletons={singletons})"

class ContainerBuilder:
    """Fluent builder for Container configuration."""

    def __init__(self) -> None:
        """Initialize container builder."""
        self._container = Container()

    def register_singleton(
        self,
        name: str,
        implementation: type[T],
        factory: Callable[..., T] | None = None,
        dependencies: list[str] | None = None,
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
        implementation: type[T],
        factory: Callable[..., T] | None = None,
        dependencies: list[str] | None = None,
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
        implementation: type[T],
        factory: Callable[..., T] | None = None,
        dependencies: list[str] | None = None,
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
            f"Built container with {len(self._container.get_services())} "
            f"registered services"
        )
        return self._container
