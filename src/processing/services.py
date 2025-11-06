"""Unified service factory and dependency injection configuration.

This module provides a single consolidated interface for accessing all processing services.
It implements the service locator pattern with lazy singleton factories for all major
services in the processing module.

Services available:
    - ResamplerService: Depth/time resampling with caching
    - ManagerHub: Unified resource management interface
    - RockPhysicsService: Rock physics model management
    - MaterialsService: Material property models
    - AVOService: AVO analysis and validation

Usage:
    from src.processing.services import get_resampler_service, get_manager_hub

    resampler = get_resampler_service()
    hub = get_manager_hub()
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional
import logging

from src.io.grid import GridSpec
from src.processing.avo.validator import AVOValidator
from src.processing.managers.processor import ManagerHub
from src.processing.resampling.service import ResamplerService
from src.processing.rock_physics.model import RockPhysicsModel

__all__ = [
    "get_resampler_service",
    "get_manager_hub",
    "get_avo_validator",
    "get_rock_physics_service",
    "ServiceRegistry",
]

logger = logging.getLogger(__name__)


# ============================================================================
# Resampler Service
# ============================================================================


@lru_cache(maxsize=1)
def get_resampler_service(grid_spec: Optional[GridSpec] = None) -> "ResamplerService":
    """Get or create the resampling service singleton.

    Args:
        grid_spec: Optional GridSpec for resampler initialization.
                  Defaults to GridSpec() if not provided.

    Returns:
        Singleton ResamplerService instance

    Example:
        >>> service = get_resampler_service()
        >>> data_time, dt = service.depth_to_time(data_depth, vp_depth)
    """
    from src.processing.resampling.service import ResamplerService

    if grid_spec is None:
        from src.io.grid import GridSpec

        grid_spec = GridSpec((512, 512, 512))

    return ResamplerService(grid_spec=grid_spec)


# ============================================================================
# Manager Hub
# ============================================================================


@lru_cache(maxsize=1)
def get_manager_hub() -> "ManagerHub":
    """Get or create the manager hub singleton.

    The ManagerHub provides unified access to all resource managers:
    - CacheManager: Cache directory operations
    - FileManager: File operations
    - ProcessManager: Process management

    Returns:
        Singleton ManagerHub instance

    Example:
        >>> hub = get_manager_hub()
        >>> hub.cache.clear()
        >>> hub.summarize()
    """
    from src.processing.managers import ManagerHub

    return ManagerHub()


# ============================================================================
# AVO Validator Service
# ============================================================================


@lru_cache(maxsize=1)
def get_avo_validator(max_angle: float = 30.0) -> "AVOValidator":
    """Get or create an AVO validator singleton.

    Args:
        max_angle: Maximum angle in degrees for AVO linearization check.
                  Default is 30.0.

    Returns:
        Singleton AVOValidator instance

    Example:
        >>> validator = get_avo_validator()
        >>> report = validator.validate(vp, vs, rho)
        >>> report.print_summary()
    """
    from src.processing.avo.validator import AVOValidator

    return AVOValidator(max_angle=max_angle)


# ============================================================================
# Rock Physics Service
# ============================================================================


@lru_cache(maxsize=1)
def get_rock_physics_service() -> "RockPhysicsService":
    """Get or create the rock physics service singleton.

    Provides high-level operations for rock physics models and caching.

    Returns:
        Singleton RockPhysicsService instance

    Example:
        >>> service = get_rock_physics_service()
        >>> model = service.create_model(vp, vs, rho, grid_spec)
    """
    return RockPhysicsService()


# ============================================================================
# Service Registry
# ============================================================================


class ServiceRegistry:
    """Central registry for all processing services.

    Provides access to all major processing services through a unified interface.
    Services are lazily initialized and cached as singletons.

    Attributes:
        resampler: Lazy property for ResamplerService
        managers: Lazy property for ManagerHub
        avo: Lazy property for AVOValidator
        rock_physics: Lazy property for RockPhysicsService

    Example:
        >>> registry = ServiceRegistry()
        >>> registry.managers.cache.clear()
        >>> report = registry.avo.validate(vp, vs, rho)
    """

    def __init__(self):
        """Initialize service registry."""
        self._resampler = None
        self._managers = None
        self._avo = None
        self._rock_physics = None

    @property
    def resampler(self) -> "ResamplerService":
        """Get resampler service (lazy singleton)."""
        if self._resampler is None:
            self._resampler = get_resampler_service()
        return self._resampler

    @property
    def managers(self) -> "ManagerHub":
        """Get manager hub (lazy singleton)."""
        if self._managers is None:
            self._managers = get_manager_hub()
        return self._managers

    @property
    def avo(self) -> "AVOValidator":
        """Get AVO validator (lazy singleton)."""
        if self._avo is None:
            self._avo = get_avo_validator()
        return self._avo

    @property
    def rock_physics(self) -> "RockPhysicsService":
        """Get rock physics service (lazy singleton)."""
        if self._rock_physics is None:
            self._rock_physics = get_rock_physics_service()
        return self._rock_physics

    def reset(self) -> None:
        """Reset all cached services (useful for testing)."""
        self._resampler = None
        self._managers = None
        self._avo = None
        self._rock_physics = None
        logger.info("Service registry cleared")


# ============================================================================
# Placeholder Service Implementations
# ============================================================================


class RockPhysicsService:
    """High-level service for rock physics model operations.

    Provides convenient factory and utility methods for creating and managing
    rock physics models.
    """

    def __init__(self):
        """Initialize rock physics service."""
        self.logger = logging.getLogger(__name__)

    def create_model(
        self,
        vp=None,
        vs=None,
        rho=None,
        facies=None,
        grid_spec=None,
        disk_cache=None,
    ) -> "RockPhysicsModel":
        """Create a rock physics model.

        Args:
            vp: P-wave velocity array (optional)
            vs: S-wave velocity array (optional)
            rho: Density array (optional)
            facies: Facies classification (optional)
            grid_spec: GridSpec instance
            disk_cache: Optional DiskCache instance
        Returns:
            RockPhysicsModel instance
        """
        from src.processing.rock_physics.model import RockPhysicsModel

        return RockPhysicsModel(
            vp=vp,
            vs=vs,
            rho=rho,
            facies=facies,
            grid_spec=grid_spec or GridSpec(),
            disk_cache=disk_cache,
        )

    def create_from_dict(self, props: dict, grid_spec: Optional[GridSpec] = None):
        """Create a rock physics model from a dictionary.

        Args:
            props: Dictionary with keys 'vp', 'vs', 'rho', 'facies' (optional)
            grid_spec: Optional GridSpec instance
        Returns:
            RockPhysicsModel instance
        """
        from src.processing.rock_physics.model import RockPhysicsModel

        if grid_spec is None:
            grid_spec = GridSpec()

        return RockPhysicsModel.from_props(props, grid_spec)


# ============================================================================
# Global Service Registry Singleton
# ============================================================================

_global_registry: Optional[ServiceRegistry] = None


def get_global_registry() -> ServiceRegistry:
    """Get the global service registry singleton.

    Returns:
        Global ServiceRegistry instance

    Example:
        >>> registry = get_global_registry()
        >>> cache_mgr = registry.managers.cache
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
        logger.debug("Created global service registry")
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global service registry (primarily for testing).

    WARNING: This clears all cached service instances. Use with caution
    in production code.
    """
    global _global_registry
    if _global_registry is not None:
        _global_registry.reset()
    _global_registry = None
    logger.warning("Global service registry reset")
