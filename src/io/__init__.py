"""I/O utilities for data loading and caching.

This package provides streamlined APIs for:
- Data Loading: Loading Stanford VI-E dataset files
- Caching: Disk and in-memory cache storage with TTL/size-based pruning
- Configuration: Grid specifications and cache policies

Clean architecture with clear separation of concerns:
- storage.py: DiskStore, MemoryStore implementations
- pruning.py: PruneStrategy, Pruner for cache maintenance
- loader.py: DatasetManager, GslibLoader for data loading
- grid.py: GridSpec for dataset metadata
- config.py: CachePolicy for configuration
- exceptions.py: Domain-specific exceptions
- disk_cache.py: DiskCache high-level interface
"""

import logging

# Core abstractions
from src.io.backends import CacheStore, DefaultFileSystemOps, FileSystemOps
from src.io.config import CachePolicy

# Exceptions
from src.io.exceptions import (
    CacheError,
    CachePruneError,
    CacheValidationError,
    DataLoaderError,
    FileLocatorError,
    GridError,
    GSLibError,
    IOBaseError,
)
from src.io.file_locator import FileLocator

# Grid and configuration
from src.io.grid import GridSpec
from src.io.gslib_reader import GSLibConfig, GSLibReader

# Data loading
from src.io.loader import DatasetManager, GslibLoader

# Pruning utilities
from src.io.pruning import (
    Pruner,
    PruneResult,
    PruneStrategy,
    should_expire_by_size,
    should_expire_by_ttl,
)

# Storage implementations
from src.io.storage import DiskStore, MemoryStore

logger = logging.getLogger(__name__)

# Public API - what users should import
__all__ = [
    # Core abstractions
    "CacheStore",
    "FileSystemOps",
    "DefaultFileSystemOps",
    # Storage
    "DiskStore",
    "MemoryStore",
    # Pruning
    "PruneStrategy",
    "Pruner",
    "PruneResult",
    "should_expire_by_ttl",
    "should_expire_by_size",
    # Data loading
    "DatasetManager",
    "GslibLoader",
    "GSLibReader",
    "GSLibConfig",
    "FileLocator",
    # Grid & config
    "GridSpec",
    "CachePolicy",
    # Exceptions
    "IOBaseError",
    "CacheError",
    "CacheValidationError",
    "CachePruneError",
    "DataLoaderError",
    "GSLibError",
    "GridError",
    "FileLocatorError",
]
