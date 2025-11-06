"""Cache loading and management subpackage.

This subpackage provides efficient loading and caching of AVO (Amplitude Versus Offset)
data files with support for both NPZ (compressed) and NPY (uncompressed) formats.

Public API:
    - CacheLoader: Main class for loading and caching AVO data
    - CacheLoaderFactory: Factory for creating configured CacheLoader instances
    - CacheConfig: Named tuple for cache configuration
    - ArrayExtractor: Abstract base for data extraction
    - NpzExtractor: Extraction from NPZ archives
    - NpyExtractor: Extraction from NPY files
    - ExtractorFactory: Factory for creating extractors

All classes are re-exported from the subpackage root for convenient importing.

Example:
    >>> from src.analysis.cache import CacheLoader, CacheLoaderFactory, ExtractorFactory
    >>> loader = CacheLoaderFactory.create_default(cache_size=100)
    >>> data = loader.load_full_stack("/path/to/cache/avo_acoustic.npz")
    >>>
    >>> # Or use extractors directly
    >>> extractor = ExtractorFactory.for_path("data.npz")
    >>> data = extractor.extract(archive)
"""

from .loader import (
    CacheLoader,
    CacheLoaderFactory,
    CacheConfig,
)
from .extractors import (
    ArrayExtractor,
    NpzExtractor,
    NpyExtractor,
    ExtractorFactory,
)

__all__ = [
    # Loader
    "CacheLoader",
    "CacheLoaderFactory",
    "CacheConfig",
    # Extractors
    "ArrayExtractor",
    "NpzExtractor",
    "NpyExtractor",
    "ExtractorFactory",
]
