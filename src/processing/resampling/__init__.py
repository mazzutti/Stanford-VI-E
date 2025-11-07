"""Resampling package - Depth <-> Time resampling utilities.

This package provides a unified interface for resampling seismic data between
depth and time domains. It includes:

- Resampling kernels and plans (internal)
- Backend management for different resampling algorithms
- Caching for performance optimization (internal)
- High-level service interface (public)

Main Components:
    - ResamplerService: High-level service interface (primary public API)
    - ResamplePlan: Resampling plan representation (internal)
    - ResampleCache: LRU cache for plans (internal)
    - Backend system: Pluggable resampling backends

Recommended usage:
    from src.processing import ResamplerService

    service = ResamplerService()
    result = service.resample(data, plan)
"""

from .service import ResamplerService
from .backends import ResamplerBackend, BackendError, BackendManager

__all__ = [
    # Primary public API
    "ResamplerService",
    # Backend support (public)
    "ResamplerBackend",
    "BackendError",
    "BackendManager",
]
