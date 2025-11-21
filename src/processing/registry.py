"""Service registry for dependency injection and singleton management.


Centralizes creation and access to all processing services using a class-based
factory pattern instead of functional decorators.


This is the single unified entry point for all services (Phase 1 refactoring).
All module-level singleton functions are consolidated here.


Usage:
    registry = ServiceRegistry.get_default()
    resampler = registry.get_resampler_service()
    backends = registry.get_backend_manager()
    cache = registry.get_resample_cache()
    metrics = registry.get_backend_metrics()
    hub = registry.get_manager_hub()


Example:
    >>> from src.processing import ServiceRegistry
    >>> registry = ServiceRegistry.get_default()
    >>> service = registry.get_resampler_service()
    >>> result = service.resample(data, plan)
"""

# This module intentionally uses lazy, call-time imports within service
# accessor methods to avoid heavy import-time dependencies and circular
# references. Add a per-file pylint disable for these import patterns.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.io.grid import GridSpec
    from src.processing.avo.validator import AVOValidator
    from src.processing.managers import ManagerHub
    from src.processing.metrics import BackendMetrics
    from src.processing.resampling._cache import ResamplePlanCache
    from src.processing.resampling.backends._manager import BackendManager
    from src.processing.resampling.service import ResamplerService


__all__ = ["ServiceRegistry", "get_registry", "reset_registry"]


logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Centralized OOP service registry for dependency injection.

    **PHASE 1 REFACTORING**: Single unified entry point for all services.

    This consolidates all module-level singleton functions into a single registry:
    - get_resampler_factory() → registry.get_resampler_service()
    - get_resample_plan_cache() → registry.get_resample_cache()
    - get_backend_manager() → registry.get_backend_manager()
    - get_global_metrics() → registry.get_backend_metrics()
    - get_registry() → registry (use ServiceRegistry.get_default())

    Manages singleton creation and initialization of all processing services
    with proper dependency injection and clear initialization order.

    Benefits:
    - Single source of truth for service configuration
    - Easier testing with dependency injection
    - Clear initialization order and dependencies
    - Cleaner API with method-based access
    - Better state management
    - No hidden global state

    Attributes:
        _instances: Cache of singleton service instances
        _logger: Logger instance
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the registry.

        Args:
            logger: Optional logger instance
        """
        self._instances: dict[
            str,
            ResamplerService
            | ResamplePlanCache
            | BackendManager
            | BackendMetrics
            | ManagerHub
            | AVOValidator,
        ] = {}
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    def _get_or_create(
        self,
        key: str,
        factory: Callable[
            [],
            ResamplerService
            | ResamplePlanCache
            | BackendManager
            | BackendMetrics
            | ManagerHub
            | AVOValidator,
        ],
    ) -> (
        ResamplerService
        | ResamplePlanCache
        | BackendManager
        | BackendMetrics
        | ManagerHub
        | AVOValidator
    ):
        """Get cached instance or create new one via factory.

        Args:
            key: Service identifier
            factory: Callable that creates the service

        Returns:
            Service instance (singleton)
        """
        if key not in self._instances:
            self._instances[key] = factory()
            self._logger.debug("Created service: %s", key)
        return self._instances[key]

    def get_resampler_service(
        self, grid_spec: GridSpec | None = None
    ) -> ResamplerService:
        """Get or create ResamplerService singleton.

        The ResamplerService wraps depth/time resampling with caching and metrics.
        Provides the primary public interface for resampling operations.

        Args:
            grid_spec: Optional GridSpec configuration. If None, uses default.

        Returns:
            ResamplerService instance (singleton)

        Raises:
            ImportError: If required dependencies are not available
        """
        # Lazy imports to avoid heavy import-time dependencies

        from src.io.grid import GridSpec as GridSpecClass
        from src.processing.resampling.service import ResamplerService

        # (previously disabled import-outside-toplevel here; now removed as redundant)

        def factory() -> ResamplerService:
            grid = grid_spec or GridSpecClass((512, 512, 512))
            return ResamplerService(grid_spec=grid)

        result = self._get_or_create("resampler_service", factory)
        assert isinstance(result, ResamplerService)
        return result

    def get_backend_manager(self) -> BackendManager:
        """Get or create BackendManager singleton.

        Manages registration and selection of resampling backends.
        Replaces: get_backend_manager(), register_backend(), list_backends(), etc.

        Returns:
            BackendManager instance (singleton)

        Raises:
            ImportError: If required dependencies are not available
        """

        from src.processing.resampling.backends._manager import BackendManager

        # (previously disabled import-outside-toplevel here; now removed as redundant)

        def factory() -> BackendManager:
            return BackendManager()

        result = self._get_or_create("backend_manager", factory)
        assert isinstance(result, BackendManager)
        return result

    def get_resample_cache(self) -> ResamplePlanCache:
        """Get or create ResamplePlanCache singleton.

        Caches resampling plans to avoid recomputation.
        Replaces: get_resample_plan_cache(), get_plan(), set_cache(), etc.

        Returns:
            ResamplePlanCache instance (singleton)

        Raises:
            ImportError: If required dependencies are not available
        """

        from src.processing.resampling._cache import ResamplePlanCache

        # (previously disabled import-outside-toplevel here; now removed as redundant)

        def factory() -> ResamplePlanCache:
            # Import here to avoid circular dependencies
            return ResamplePlanCache(maxsize=16)

        result = self._get_or_create("resample_cache", factory)
        assert isinstance(result, ResamplePlanCache)
        return result

    def get_backend_metrics(self) -> BackendMetrics:
        """Get or create BackendMetrics singleton.

        Tracks backend selection and runtime metrics for performance analysis.
        Replaces: get_global_metrics()

        Returns:
            BackendMetrics instance (singleton)

        Raises:
            ImportError: If required dependencies are not available
        """

        from src.processing.metrics import BackendMetrics

        # (previously disabled import-outside-toplevel here; now removed as redundant)

        def factory() -> BackendMetrics:
            return BackendMetrics()

        result = self._get_or_create("backend_metrics", factory)
        assert isinstance(result, BackendMetrics)
        return result

    def get_manager_hub(self) -> ManagerHub:
        """Get or create ManagerHub singleton.

        The ManagerHub provides unified access to cache, file, and process managers.

        Returns:
            ManagerHub instance (singleton)

        Raises:
            ImportError: If required dependencies are not available
        """

        from src.processing.managers import ManagerHub

        # (previously disabled import-outside-toplevel here; now removed as redundant)

        def factory() -> ManagerHub:
            return ManagerHub()

        result = self._get_or_create("manager_hub", factory)
        assert isinstance(result, ManagerHub)
        return result

    def get_avo_validator(self, max_angle: float = 30.0) -> AVOValidator:
        """Get or create AVOValidator singleton.

        The AVOValidator performs AVO analysis and linearization checks.

        Args:
            max_angle: Maximum angle in degrees for AVO linearization check.
                      Default is 30.0.

        Returns:
            AVOValidator instance (singleton)

        Raises:
            ImportError: If required dependencies are not available
        """

        from src.processing.avo.validator import AVOValidator

        # (previously disabled import-outside-toplevel here; now removed as redundant)

        def factory() -> AVOValidator:
            return AVOValidator(max_angle=max_angle)

        result = self._get_or_create("avo_validator", factory)
        assert isinstance(result, AVOValidator)
        return result

    def get_rock_physics_model(self) -> type:
        """Get RockPhysicsModel class (factory, not singleton).

        Returns the class itself for direct instantiation with custom parameters.

        Returns:
            RockPhysicsModel class

        Raises:
            ImportError: If required dependencies are not available
        """

        from src.processing.rock_physics.model import RockPhysicsModel

        return RockPhysicsModel

    def clear(self) -> None:
        """Clear all service instances (useful for testing).

        After calling this, subsequent calls will recreate services.
        """
        self._instances.clear()
        self._logger.info("Cleared all service instances")


def get_registry() -> ServiceRegistry:
    """Get the global service registry singleton.

    Lazily initializes the registry on first call. Stored as a function
    attribute to avoid using the ``global`` statement while preserving
    module-level singleton semantics.
    """
    inst = getattr(get_registry, "_instance", None)
    if inst is None:
        inst = ServiceRegistry()
        setattr(get_registry, "_instance", inst)
    return inst


def reset_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    inst = getattr(get_registry, "_instance", None)
    if inst is not None:
        inst.clear()
