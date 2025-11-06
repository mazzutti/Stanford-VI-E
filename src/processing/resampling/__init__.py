"""Resampling package - Depth <-> Time resampling utilities.

PHASE 2 REFACTORING: Internal modules now use _ prefix to mark them as internal.

Public API: ResamplerService (primary interface), plus supporting types.
Internal: _resampler, _cache, _plan, _kernels (implementation details).

This package provides a unified interface for resampling seismic data between
depth and time domains. It includes:

- Resampling kernels and plans (internal)
- Backend management for different resampling algorithms
- Caching for performance optimization (internal)
- High-level service interface (PUBLIC)

Main Components:
    - ResamplerService: High-level service interface (PRIMARY PUBLIC API)
    - ResamplePlan: Resampling plan representation (internal)
    - ResampleCache: LRU cache for plans (internal)
    - Backend system: Pluggable resampling backends

RECOMMENDED USAGE:
    from src.processing import ResamplerService

    service = ResamplerService()
    result = service.resample(data, plan)
"""

from .service import (
    ResamplerService,
)

from .backends import (
    ResamplerBackend,
    BackendError,
    BackendManager,
)

# Public API only - no backward compatibility imports
__all__ = [
    # Primary public API
    "ResamplerService",
    # Backend support (public)
    "ResamplerBackend",
    "BackendError",
    "BackendManager",
]
