"""Processing utilities package.

High-level processing operations for seismic data transformation, including:
- Depth/time resampling (ResamplerService)
- Process management (ManagerHub, ProcessManager, CacheManager, FileManager)
- AVO analysis and validation
- Rock physics models with caching
- Material property models with unit handling

The module is organized into subpackages:
    core: Abstract base classes and core interfaces
    managers: Resource management (cache, files, processes)
    materials: Material property models (velocity, density, etc.)
    rock_physics: Rock physics data models and caching
    avo: AVO analysis and validation
    resampling: Depth/time resampling (backends, plans, service)

Service Registry (OOP Approach):
    Use ServiceRegistry for dependency injection and service management:
    
        from src.processing.registry import get_registry
        
        registry = get_registry()
        resampler = registry.resampler_service()
        hub = registry.manager_hub()
        validator = registry.avo_validator()

Key classes:
    ServiceRegistry: OOP service registry for dependency injection
    ResamplerService: High-level resampling with caching and metrics
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

# Core abstractions and exceptions
from src.processing.core import (
    Processor,
    Manager,
    Resampler,
    MaterialProperty,
    Validator,
    ProcessingError,
    ResamplingError,
    ValidationError,
    CacheError,
    ConfigurationError,
)

# High-level services
from src.processing.resampling.service import ResamplerService

# Managers (simplified API)
from src.processing.managers import (
    BaseManager,
    CacheManager,
    FileManager,
    ProcessManager,
    ManagerHub,
)

# Rock physics
from src.processing.rock_physics.model import RockPhysicsModel
from src.processing.rock_physics.cache import ModelCache as RockPhysicsModelCache

# Materials
from src.processing.materials import (
    MaterialModel,
    VsModel,
    DensityModel,
)
from src.processing.materials.velocity import VelocityModel

# AVO analysis
from src.processing.avo.validator import (
    AVOValidator,
    AVOValidityReport,
)

# New OOP service registry
from src.processing.registry import (
    ServiceRegistry,
    get_registry,
)

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
    # High-level APIs
    "ResamplerService",
    # OOP Service Registry
    "ServiceRegistry",
    "get_registry",
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
    # Subpackages
    "core",
    "managers",
    "materials",
    "rock_physics",
    "avo",
    "resampling",
    "registry",
]
