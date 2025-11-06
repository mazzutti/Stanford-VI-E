"""Resampling backends - Pluggable backend implementations.

This package provides the backend system for resampling algorithms, including:

- Abstract base classes for backend implementations
- Concrete backend implementations
- Backend manager for registration and selection

Main Components:
    - ResamplerBackend: Abstract base class for backends
    - BackendResult: Result from backend execution
    - BackendError: Backend-specific exceptions
    - BackendManager: Manages backend registration and selection
"""

from .base import (
    ResamplerBackend,
    BackendResult,
    BackendError,
)

from .manager import (
    BackendManager,
    get_backend_manager,
    register_backend,
    list_backends,
    get_best_backend,
    set_backend_verbose,
    is_backend_verbose,
)

__all__ = [
    # Base classes and types
    "ResamplerBackend",
    "BackendResult",
    "BackendError",
    # Manager
    "BackendManager",
    "get_backend_manager",
    "register_backend",
    "list_backends",
    "get_best_backend",
    "set_backend_verbose",
    "is_backend_verbose",
]
