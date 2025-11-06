"""Unified service factory and dependency injection configuration.

This module provides factory functions for accessing all processing services.
It implements the service locator pattern with lazy singleton factories for all major
services in the processing module.

Services available:
    - ResamplerService: Depth/time resampling with caching
    - ManagerHub: Unified resource management interface
    - RockPhysicsModel: Rock physics model management
    - MaterialsService: Material property models
    - AVOValidator: AVO analysis and validation

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

__all__ = [
    "get_resampler_service",
    "get_manager_hub",
    "get_avo_validator",
    "get_rock_physics_service",
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
# Rock Physics Model Factory
# ============================================================================


@lru_cache(maxsize=1)
def get_rock_physics_service():
    """Get or create a rock physics model factory.

    Returns the RockPhysicsModel class directly for instantiation.

    Returns:
        RockPhysicsModel class

    Example:
        >>> from src.processing.rock_physics.model import RockPhysicsModel
        >>> model = RockPhysicsModel(vp=vp_array, vs=vs_array, rho=rho_array)
    """
    from src.processing.rock_physics.model import RockPhysicsModel

    return RockPhysicsModel
