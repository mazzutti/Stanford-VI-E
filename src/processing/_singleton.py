"""Unified singleton and service management.

Provides a clean, reusable pattern for all singletons and services in the
processing module, centralizing lazy initialization and dependency injection.
"""

from __future__ import annotations

from typing import TypeVar, Generic, Optional, Callable

T = TypeVar("T")


class SingletonFactory(Generic[T]):
    """Generic singleton factory for lazy initialization and dependency injection.

    Supports both default lazy-initialized singletons and explicit injection.
    """

    def __init__(self, factory_fn: Callable[[], T]) -> None:
        """Initialize the factory with a callable that creates instances."""
        self._factory_fn = factory_fn
        self._instance: Optional[T] = None

    def get(self, instance: Optional[T] = None) -> T:
        """Get singleton instance, optionally providing one for injection.

        Args:
            instance: Optional instance to use instead of singleton

        Returns:
            Provided instance or lazily-created singleton
        """
        if instance is not None:
            return instance
        if self._instance is None:
            self._instance = self._factory_fn()
        return self._instance

    def reset(self) -> None:
        """Reset singleton for testing or reinitialization."""
        self._instance = None


class ServiceRegistry:
    """Central registry for all services and singletons in the module.

    Simplifies dependency injection and provides a single point of control
    for service lifecycle management.
    """

    def __init__(self) -> None:
        self._services: dict[str, object] = {}
        self._factories: dict[str, SingletonFactory[object]] = {}

    def register_singleton(self, name: str, factory_fn: Callable[[], T]) -> None:
        """Register a singleton factory.

        Args:
            name: Unique name for the service
            factory_fn: Callable that creates instances

        Raises:
            KeyError: If service with that name already exists
        """
        if name in self._factories or name in self._services:
            raise KeyError(f"Service '{name}' already registered")
        self._factories[name] = SingletonFactory(factory_fn)

    def get_service(self, name: str, instance: object | None = None) -> object:
        """Get a service by name, optionally providing an override instance.

        Args:
            name: Service name
            instance: Optional override instance

        Returns:
            The service instance

        Raises:
            KeyError: If service not found
        """
        if instance is not None:
            return instance
        if name in self._services:
            return self._services[name]
        if name in self._factories:
            return self._factories[name].get()
        raise KeyError(f"Service '{name}' not registered")

    def reset_all(self) -> None:
        """Reset all singletons. Useful for testing."""
        for factory in self._factories.values():
            factory.reset()
        self._services.clear()

    def reset_service(self, name: str) -> None:
        """Reset a specific service."""
        if name in self._factories:
            self._factories[name].reset()
        elif name in self._services:
            del self._services[name]


# Global service registry instance
_global_registry = ServiceRegistry()
