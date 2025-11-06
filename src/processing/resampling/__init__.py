"""Resampling package - Depth <-> Time resampling utilities.

This package provides a unified interface for resampling seismic data between
depth and time domains. It includes:

- Resampling kernels and plans
- Backend management for different resampling algorithms
- Caching for performance optimization
- High-level service interface

Main Components:
    - Resampler: Core resampling functionality
    - ResamplerService: High-level service interface
    - ResamplePlan: Resampling plan representation
    - ResampleCache: LRU cache for plans
    - Backend system: Pluggable resampling backends
"""

from .resampler import (
    DepthTimeResampler,
    set_backend_verbose,
    is_backend_verbose,
)

from .service import (
    ResamplerService,
)

from .plan import (
    ResamplePlan,
)

from .cache import (
    ResamplePlanCache,
    get_resample_plan_cache,
)

from .backends import (
    ResamplerBackend,
    BackendError,
    BackendManager,
    get_backend_manager,
    register_backend,
    list_backends,
    get_best_backend,
)

__all__ = [
    # Core
    "DepthTimeResampler",
    "ResamplerService",
    # Plans and caching
    "ResamplePlan",
    "ResamplePlanCache",
    "get_resample_plan_cache",
    # Backends
    "ResamplerBackend",
    "BackendError",
    "BackendManager",
    "get_backend_manager",
    "register_backend",
    "list_backends",
    "get_best_backend",
    # Utilities
    "set_backend_verbose",
    "is_backend_verbose",
]
