"""Generic factory pattern implementation with registration support.

Eliminates duplicate factory code across CacheLoaderFactory, ExtractorFactory,
ResamplerFactory, etc. by providing a reusable base class.
"""

from typing import TypeVar, Generic, Dict, Callable, Type, Any, Optional

__all__ = ["GenericFactory"]

T = TypeVar("T")


class GenericFactory(Generic[T]):
    """Base factory with registration pattern.

    Provides a reusable factory implementation that can be used by any
    component that needs to create instances based on string identifiers.

    Example:
        >>> class CacheLoaderFactory(GenericFactory['CacheLoader']):
        ...     pass
        >>>
        >>> factory = CacheLoaderFactory()
        >>>
        >>> @factory.register("default")
        >>> def create_default(cache_size: int = 100):
        ...     return CacheLoader(cache_size=cache_size)
        >>>
        >>> loader = factory.create("default", cache_size=200)
    """

    def __init__(self):
        """Initialize factory with empty registries."""
        self._builders: Dict[str, Callable[..., T]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        builder: Optional[Callable[..., T]] = None,
        default_config: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Register a builder with optional default config.

        Can be used as a decorator or called directly.

        Args:
            name: Unique identifier for this builder
            builder: Builder function (if not using as decorator)
            default_config: Default configuration for this builder

        Returns:
            The builder (for decorator usage)

        Example:
            >>> @factory.register("custom")
            >>> def create_custom():
            ...     return MyClass()
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            self._builders[name] = func
            if default_config:
                self._configs[name] = default_config
            return func

        if builder is not None:
            return decorator(builder)
        return decorator

    def register_class(self, name: str, cls: Type[T], **default_kwargs: Any) -> None:
        """Register a class with default initialization args.

        Args:
            name: Unique identifier for this class
            cls: Class to register
            **default_kwargs: Default keyword arguments for initialization

        Example:
            >>> factory.register_class("mmap", CacheLoader, cache_size=0)
        """
        self.register(name, lambda **kw: cls(**{**default_kwargs, **kw}))

    def create(self, name: str, **kwargs: Any) -> T:
        """Create instance using registered builder.

        Args:
            name: Identifier of the builder to use
            **kwargs: Arguments to pass to the builder

        Returns:
            Created instance

        Raises:
            ValueError: If builder name is not registered

        Example:
            >>> loader = factory.create("default", cache_size=100)
        """
        if name not in self._builders:
            available = ", ".join(self._builders.keys())
            raise ValueError(f"Unknown builder: {name}. Available: {available}")

        config = {**self._configs.get(name, {}), **kwargs}
        return self._builders[name](**config)

    def has(self, name: str) -> bool:
        """Check if a builder is registered.

        Args:
            name: Builder identifier

        Returns:
            True if builder is registered
        """
        return name in self._builders

    def list_builders(self) -> list[str]:
        """Get list of registered builder names.

        Returns:
            List of builder identifiers
        """
        return list(self._builders.keys())

    def unregister(self, name: str) -> None:
        """Remove a registered builder.

        Args:
            name: Builder identifier to remove
        """
        self._builders.pop(name, None)
        self._configs.pop(name, None)
