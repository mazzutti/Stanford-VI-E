"""Processing utilities package.


High-level processing operations for seismic data transformation, including:
- Depth/time resampling (ResamplerService)
- Process management (ManagerHub, ProcessManager, CacheManager, FileManager)
- AVO analysis and validation
- Rock physics models with caching
- Material property models with unit handling


PHASE 1 REFACTORING: Single unified entry point via ServiceRegistry.


The module is organized into subpackages:
    core: Abstract base classes and core interfaces
    managers: Resource management (cache, files, processes)
    materials: Material property models (velocity, density, etc.)
    rock_physics: Rock physics data models and caching
    avo: AVO analysis and validation
    resampling: Depth/time resampling (backends, plans, service)


Service Registry (OOP Approach - Recommended):
    Use ServiceRegistry for dependency injection and service management:

        from src.processing import ServiceRegistry

        registry = ServiceRegistry.get_default()
        resampler = registry.get_resampler_service()
        backends = registry.get_backend_manager()
        cache = registry.get_resample_cache()
        metrics = registry.get_backend_metrics()
        hub = registry.get_manager_hub()
        validator = registry.get_avo_validator()

    Or for convenience:

        service = ResamplerService()  # Uses default registry internally


Key classes:
    ServiceRegistry: OOP service registry for dependency injection (SINGLE ENTRY POINT)
    ResamplerService: High-level resampling with caching and metrics
    BackendManager: Manages resampling backend selection
    ResamplePlanCache: Caches resampling plans
    BackendMetrics: Performance metrics collector
    ManagerHub: Unified interface for all resource managers
    ProcessManager: Process management facade (simplified API)
    CacheManager: Cache directory operations
    FileManager: File operations
    RockPhysicsModel: Rock physics properties (vp, vs, rho, facies)
    MaterialModel: Base class for unit-aware property models
    VelocityModel: P-wave velocity model
    VsModel: S-wave velocity model
    DensityModel: Density property model
    AVOValidator: AVO linearization validator
"""

import logging

# AVO analysis
from src.processing.avo.validator import AVOValidator, AVOValidityReport

# Core abstractions and exceptions
from src.processing.core import (
    CacheError,
    ConfigurationError,
    Manager,
    MaterialProperty,
    ProcessingError,
    Processor,
    Resampler,
    ResamplingError,
    ValidationError,
    Validator,
)

# Managers (simplified API)
from src.processing.managers import (
    BaseManager,
    CacheManager,
    FileManager,
    ManagerHub,
    ProcessManager,
)

# Materials
from src.processing.materials import DensityModel, MaterialModel, VsModel
from src.processing.materials.velocity import VelocityModel
from src.processing.metrics import BackendMetrics

# PHASE 1: New unified OOP service registry (single entry point)
from src.processing.registry import ServiceRegistry, get_registry, reset_registry
from src.processing.resampling._cache import ResamplePlanCache

# Resampling components (for advanced usage)
from src.processing.resampling.backends._manager import BackendManager

# High-level services
from src.processing.resampling.service import ResamplerService
from src.processing.rock_physics.cache import ModelCache as RockPhysicsModelCache

# Rock physics
from src.processing.rock_physics.model import RockPhysicsModel

logger = logging.getLogger(__name__)


__all__ = [
    # Core abstractions
    "Processor",
    "Manager",
    "Resampler",
    "MaterialProperty",
    "Validator",
    # Core exceptions
    "ProcessingError",
    "ResamplingError",
    "ValidationError",
    "CacheError",
    "ConfigurationError",
    # PHASE 1: Unified service registry (RECOMMENDED ENTRY POINT)
    "ServiceRegistry",
    "get_registry",
    "reset_registry",
    # High-level services
    "ResamplerService",
    "BackendManager",
    "ResamplePlanCache",
    "BackendMetrics",
    # Managers
    "BaseManager",
    "CacheManager",
    "FileManager",
    "ProcessManager",
    "ManagerHub",
    # Rock physics
    "RockPhysicsModel",
    "RockPhysicsModelCache",
    # Materials
    "MaterialModel",
    "VelocityModel",
    "VsModel",
    "DensityModel",
    # AVO
    "AVOValidator",
    "AVOValidityReport",
]
