"""Processor registry for plugin architecture and dynamic processor management.

This module provides ProcessorRegistry which enables:
- Runtime processor discovery and instantiation
- Processor metadata and versioning
- Domain-specific processor sets
- Processor composition and chaining
- Mock processor injection for testing

Pattern: Registry Pattern + Factory Pattern
- Central registry for processor creation
- Metadata-driven processor discovery
- Tag-based filtering for processors
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
)
from dataclasses import dataclass, field
import logging

__all__ = [
    "ProcessorMetadata",
    "ProcessorRegistry",
    "get_default_processor_registry",
    "register_processor",
    "create_processor",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Generic processor type


@dataclass
class ProcessorMetadata:
    """Metadata about a registered processor.

    Attributes
    ----------
    name : str
        Unique identifier for the processor.
    domain : str
        Domain/category this processor belongs to (e.g., 'facies', 'avo').
    version : str
        Version of the processor (e.g., '1.0', '2.1.0').
    tags : List[str]
        Keywords describing processor capabilities.
    description : str
        Human-readable description of what processor does.
    dependencies : List[str]
        Names of other processors this one depends on.
    """

    name: str
    domain: str = "default"
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    description: str = ""
    dependencies: List[str] = field(default_factory=list)

    def matches_tags(self, required_tags: List[str]) -> bool:
        """Check if this processor has all required tags.

        Parameters
        ----------
        required_tags : List[str]
            Tags that must be present.

        Returns
        -------
        bool
            True if processor has all required tags.
        """
        return all(tag in self.tags for tag in required_tags)


class ProcessorRegistry:
    """Central registry for processor creation and management.

    Manages processor factories, metadata, and provides methods to:
    - Register new processors
    - Create processor instances
    - Query available processors
    - List processors by domain/tags

    Examples
    --------
    Register processors:

    >>> registry = ProcessorRegistry()
    >>> registry.register(
    ...     name="boundary_detector_v1",
    ...     factory=lambda: BoundaryDetector(),
    ...     domain="facies",
    ...     tags=["boundary", "detection"],
    ... )

    Create processor instances:

    >>> detector = registry.create("boundary_detector_v1")

    List processors:

    >>> facies_processors = registry.list_processors(domain="facies")
    >>> detection_processors = registry.list_processors(tags=["detection"])
    """

    def __init__(self) -> None:
        """Initialize an empty processor registry."""
        self._processors: Dict[str, Callable[[], Any]] = {}
        self._metadata: Dict[str, ProcessorMetadata] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        domain: str = "default",
        version: str = "1.0",
        tags: Optional[List[str]] = None,
        description: str = "",
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a processor factory.

        Parameters
        ----------
        name : str
            Unique identifier for this processor.
        factory : Callable[[], Any]
            Callable that creates processor instances (no-arg constructor).
        domain : str, optional
            Domain/category of processor (default: "default").
        version : str, optional
            Version identifier (default: "1.0").
        tags : List[str], optional
            Keywords describing processor (default: empty list).
        description : str, optional
            Human-readable description (default: empty).
        dependencies : List[str], optional
            Names of other required processors (default: empty list).

        Raises
        ------
        ValueError
            If processor with same name already registered.
        TypeError
            If factory is not callable.
        """
        if not callable(factory):
            raise TypeError(f"factory must be callable, got {type(factory)}")

        if name in self._processors:
            raise ValueError(f"Processor '{name}' already registered")

        self._processors[name] = factory
        self._metadata[name] = ProcessorMetadata(
            name=name,
            domain=domain,
            version=version,
            tags=tags or [],
            description=description,
            dependencies=dependencies or [],
        )
        logger.debug(f"Registered processor: {name} ({domain}/{version})")

    def unregister(self, name: str) -> bool:
        """Unregister a processor.

        Parameters
        ----------
        name : str
            Name of processor to unregister.

        Returns
        -------
        bool
            True if processor was registered and removed, False otherwise.
        """
        if name in self._processors:
            del self._processors[name]
            del self._metadata[name]
            logger.debug(f"Unregistered processor: {name}")
            return True
        return False

    def create(self, name: str) -> Any:
        """Create a processor instance by name.

        Parameters
        ----------
        name : str
            Name of processor to create.

        Returns
        -------
        Any
            New instance of the processor.

        Raises
        ------
        ValueError
            If processor name not found in registry.

        Examples
        --------
        >>> detector = registry.create("boundary_detector_v1")
        """
        if name not in self._processors:
            available = list(self._processors.keys())
            raise ValueError(f"Unknown processor '{name}'. Available: {available}")

        try:
            instance = self._processors[name]()
            logger.debug(f"Created processor instance: {name}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create processor '{name}': {e}")
            raise

    def create_all(self, names: List[str]) -> Dict[str, Any]:
        """Create multiple processor instances.

        Parameters
        ----------
        names : List[str]
            Names of processors to create.

        Returns
        -------
        Dict[str, Any]
            Mapping of processor names to instances.

        Raises
        ------
        ValueError
            If any processor name not found.
        """
        return {name: self.create(name) for name in names}

    def list_processors(
        self,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
    ) -> List[str]:
        """List registered processor names with optional filtering.

        Parameters
        ----------
        domain : str, optional
            Filter by domain (e.g., 'facies', 'avo').
        tags : List[str], optional
            Filter by all required tags (AND logic).
        version : str, optional
            Filter by version.

        Returns
        -------
        List[str]
            Names of processors matching all filters.

        Examples
        --------
        >>> facies_processors = registry.list_processors(domain="facies")
        >>> detection = registry.list_processors(tags=["detection", "boundary"])
        """
        results = []
        for name, meta in self._metadata.items():
            if domain and meta.domain != domain:
                continue
            if tags and not meta.matches_tags(tags):
                continue
            if version and meta.version != version:
                continue
            results.append(name)
        return results

    def get_metadata(self, name: str) -> ProcessorMetadata:
        """Get metadata about a processor.

        Parameters
        ----------
        name : str
            Name of processor.

        Returns
        -------
        ProcessorMetadata
            Metadata describing the processor.

        Raises
        ------
        ValueError
            If processor not found.
        """
        if name not in self._metadata:
            raise ValueError(f"Unknown processor: {name}")
        return self._metadata[name]

    def has(self, name: str) -> bool:
        """Check if a processor is registered.

        Parameters
        ----------
        name : str
            Name of processor to check.

        Returns
        -------
        bool
            True if processor is registered.
        """
        return name in self._processors

    def get_all_metadata(self) -> Dict[str, ProcessorMetadata]:
        """Get metadata for all registered processors.

        Returns
        -------
        Dict[str, ProcessorMetadata]
            Mapping of processor names to their metadata.
        """
        return dict(self._metadata)

    def __repr__(self) -> str:
        """Return string representation showing registry state.

        Returns
        -------
        str
            Representation showing count and domains of registered processors.
        """
        count = len(self._processors)
        domains = set(m.domain for m in self._metadata.values())
        return f"ProcessorRegistry({count} processors in domains: {domains})"


# Global default processor registry
_default_registry: Optional[ProcessorRegistry] = None


def get_default_processor_registry() -> ProcessorRegistry:
    """Get or create the global default processor registry.

    Returns
    -------
    ProcessorRegistry
        The shared default registry for the application.

    Examples
    --------
    >>> registry = get_default_processor_registry()
    >>> registry.register("my_processor", factory=MyProcessor)
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ProcessorRegistry()
    return _default_registry


def register_processor(
    name: str,
    factory: Callable[[], Any],
    *,
    domain: str = "default",
    **kwargs: Any,
) -> None:
    """Register a processor in the default registry.

    Convenience function for registering processors globally.

    Parameters
    ----------
    name : str
        Unique processor identifier.
    factory : Callable[[], Any]
        Callable that creates processor instances.
    domain : str, optional
        Processor domain (default: "default").
    **kwargs
        Additional metadata (version, tags, description, dependencies).
    """
    get_default_processor_registry().register(name, factory, domain=domain, **kwargs)


def create_processor(name: str) -> Any:
    """Create a processor from the default registry.

    Convenience function for creating processors.

    Parameters
    ----------
    name : str
        Name of processor to create.

    Returns
    -------
    Any
        New processor instance.

    Raises
    ------
    ValueError
        If processor not found.
    """
    return get_default_processor_registry().create(name)
