"""AVO Cache Loading with LRU Caching Support.

This module provides efficient loading and caching of AVO (Amplitude Versus Offset)
data files from the cache directory. It supports both NPZ (compressed) and NPY
(uncompressed) file formats with optional in-memory caching via LRU cache.

Core Components:
    - CacheLoader: Main class for loading and caching AVO data with memory
      mapping support to preserve efficient data access patterns.
    - CacheLoaderFactory: Factory class for creating CacheLoader instances
      with flexible configuration options.

Features:
    - LRU caching with optional sharding for parallel access
    - Memory-mapped file support (preserves mmap for large datasets)
    - Configurable file selection strategies
    - NPZ archive extraction with customizable strategies
    - Comprehensive error handling and logging
    - Full dependency injection support for testing

Example Usage:
    >>> from src.analysis.cache_loader import CacheLoaderFactory
    >>> loader = CacheLoaderFactory.create_default(cache_size=100)
    >>> data = loader.load_full_stack("/path/to/cache", mmap_mode="r")
"""

from pathlib import Path
from typing import Any, NamedTuple, cast
from collections.abc import Callable
from types import TracebackType
from os import PathLike
from numpy.typing import NDArray
from numpy.lib.npyio import NpzFile

import numpy as np
import logging

from src.analysis.types.protocols import (
    CacheProtocol,
    SelectorProtocol,
    ArchiveExtractorProtocol,
)
from src.utils.lru import LRUCache, ShardedLRUCache
from src.core.generic_factory import GenericFactory

logger = logging.getLogger(__name__)

# Public API
__all__ = ["CacheLoader", "CacheLoaderFactory", "CacheConfig"]

# Constants for file naming and archive handling
_FILE_PREFIX = "avo_"
_FULL_STACK_KEY = "full_stack"
_NPZ_EXTENSION = ".npz"
_NPY_EXTENSION = ".npy"


class CacheConfig(NamedTuple):
    """Configuration for CacheLoader caching behavior.

    Attributes
    ----------
    cache_size : int
        Size of the LRU cache. 0 means caching is disabled.
    archive_extractor : ArchiveExtractorProtocol | None
        Custom NPZ extraction strategy, or None for default.
    selector : SelectorProtocol | None
        Custom file selection strategy, or None for default.
    cache : CacheProtocol[NDArray[np.float64]] | None
        External cache instance, or None to create default.
    np_load : Callable
        NumPy load function (for testing/mocking).
    """

    cache_size: int
    archive_extractor: ArchiveExtractorProtocol | None = None
    selector: SelectorProtocol | None = None
    cache: CacheProtocol[NDArray[np.float64]] | None = None
    np_load: Callable[..., NDArray[Any] | NpzFile] = np.load


class CacheLoader:
    """Load and cache AVO data files with LRU caching and memory mapping support.

    This class provides efficient loading of AVO cache files (NPZ or NPY format)
    with optional in-memory LRU caching. It preserves memory-mapped access for
    large datasets while caching non-memmap arrays as float64 copies.

    Attributes
    ----------
    _selector : SelectorProtocol | None
        Custom file selection callable. If None, uses default selection logic.
    _np_load : Callable
        NumPy load function (can be mocked for testing).
    _archive_extractor : ArchiveExtractorProtocol | None
        Custom NPZ archive extraction callable.
    _cache : CacheProtocol | None
        LRU cache instance for storing loaded arrays.
    _cache_size : int
        Maximum size of the cache.

    Methods
    -------
    select_cache_file(cache_dir, domain, *, prefer_latest=True, allow_npy=True)
        Find and select a cache file for a given domain.
    load_full_stack(path, *, mmap_mode=None, raise_on_error=False)
        Main entry point: load AVO data with automatic caching.
    cache_enabled : bool
        Check if caching is currently active.
    cache_maxsize : int
        Get the configured cache maximum size.
    cache_info()
        Get cache statistics (hits, misses, etc).
    cache_keys()
        Get all keys currently in the cache.
    cache_clear()
        Clear all entries from the cache.

    Notes
    -----
    - Memory-mapped arrays are never cached to preserve their efficiency benefits
    - Only non-memmap arrays are converted to float64 and cached
    - Cache lookups only occur when mmap_mode is None
    - Supports sharded LRU caching for improved concurrency
    """

    @staticmethod
    def _build_cache_filename(domain: str, extension: str = _NPZ_EXTENSION) -> str:
        """Build standard cache filename (e.g., 'avo_acoustic.npz')."""
        if not domain:
            raise ValueError("domain must be a non-empty string")
        return f"{_FILE_PREFIX}{domain}{extension}"

    @staticmethod
    def default_selector(cache_dir: str, domain: str) -> str | None:
        """Locate AVO cache file (tries .npz then .npy)."""
        if not domain:
            raise ValueError("domain must be a non-empty string")

        p = Path(cache_dir)
        candidate = p / CacheLoader._build_cache_filename(domain, _NPZ_EXTENSION)
        if candidate.exists():
            return str(candidate)
        candidate2 = p / CacheLoader._build_cache_filename(domain, _NPY_EXTENSION)
        if candidate2.exists():
            return str(candidate2)
        return None

    @staticmethod
    def _extract_array_from_npz(archive: NpzFile) -> NDArray[np.float64] | None:
        """Extract array from NPZ archive.

        Extraction strategy:
        1. If ``'full_stack'`` key exists, returns that array
        2. Otherwise, returns the first array in the archive
        3. Returns ``None`` on any error and logs the exception

        This is the shared extraction logic used by both the default extractor
        and the instance method.

        Parameters
        ----------
        archive : NpzFile
            Loaded NPZ archive object.

        Returns
        -------
        NDArray[np.float64] | None
            Extracted array as float64, or None if extraction failed.

        Raises
        ------
        Exception
            Internal exceptions are caught and logged, not raised.

        Notes
        -----
        The ``'full_stack'`` key is a convention in this project for storing
        the main data array within NPZ files.

        Examples
        --------
        >>> import numpy as np
        >>> # Create a sample NPZ archive
        >>> data = {'full_stack': np.array([[1, 2], [3, 4]])}
        >>> # In practice, you would load an NPZ file:
        >>> # archive = np.load("data.npz")
        >>> # result = CacheLoader._extract_array_from_npz(archive)
        """
        try:
            if _FULL_STACK_KEY in archive:
                return np.asarray(archive[_FULL_STACK_KEY])
            files = getattr(archive, "files", [])
            if files:
                return np.asarray(archive[files[0]])
        except Exception:
            logger.exception("Failed to extract array from NPZ archive")
        return None

    @staticmethod
    def default_archive_extractor(archive: NpzFile) -> NDArray[np.float64] | None:
        """Extract array from NPZ (tries 'full_stack' key, then first array)."""
        return CacheLoader._extract_array_from_npz(archive)

    def __init__(
        self,
        selector: SelectorProtocol | None = None,
        np_load: Callable[..., NDArray[Any] | NpzFile] = np.load,
        *,
        cache: CacheProtocol[NDArray[np.float64]] | None = None,
        cache_size: int = 0,
        archive_extractor: ArchiveExtractorProtocol | None = None,
    ) -> None:
        """Initialize with optional caching and custom strategies.

        Raises
        ------
        ValueError
            If cache_size is negative.

        Notes
        -----
        If both cache and cache_size are provided, the provided cache instance
        takes precedence and cache_size is ignored. To enable caching without
        providing a cache instance, set cache_size > 0.
        """
        if cache_size < 0:
            raise ValueError(f"cache_size must be non-negative, got {cache_size}")

        self._selector = selector
        self._np_load = np_load
        self._archive_extractor = archive_extractor
        self._cache_size = int(cache_size)

        # If the caller injected a cache instance, use it. Otherwise, create
        # a default LRUCache only when cache_size > 0.
        if cache is not None:
            self._cache: CacheProtocol[NDArray[np.float64]] | None = cache
        else:
            self._cache = None
            if self._cache_size > 0:
                self._cache = LRUCache[NDArray[np.float64]](self._cache_size)

    def __enter__(self) -> "CacheLoader":
        """Enter context manager for CacheLoader.

        Returns
        -------
        CacheLoader
            Returns self for use in with statements.

        Examples
        --------
        >>> with CacheLoaderFactory.create_default(cache_size=100) as loader:
        ...     data = loader.load_full_stack("/path/to/cache/avo_acoustic.npz")
        ...     # Cache is automatically cleared when exiting the context
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """
        Exit context manager for CacheLoader.

                Clears the cache when exiting the context, ensuring clean resource
                cleanup. If an exception occurred, it is not suppressed.

                Parameters
                ----------
                exc_type : type | None
                    The exception type if an exception occurred, None otherwise.
                exc_val : Exception | None
                    The exception instance if an exception occurred, None otherwise.
                _exc_tb : TracebackType | None
                    The traceback if an exception occurred, None otherwise.

                Returns
                -------
                None
                    Does not suppress exceptions.
        """
        self.cache_clear()

    def __repr__(self) -> str:
        """Return a detailed string representation of CacheLoader.

        Returns
        -------
        str
            String representation showing cache configuration and status.

        Examples
        --------
        >>> loader = CacheLoaderFactory.create_default(cache_size=100)
        >>> repr(loader)
        'CacheLoader(cache_enabled=True, cache_maxsize=100, cache_size=0)'
        """
        cache_info = self.cache_info()
        currsize = cache_info.get("currsize", 0) if cache_info else 0
        return (
            f"CacheLoader(cache_enabled={self.cache_enabled}, "
            f"cache_maxsize={self.cache_maxsize}, cache_size={currsize})"
        )

    def __str__(self) -> str:
        """Return a user-friendly string representation of CacheLoader.

        Returns
        -------
        str
            Human-readable description of the loader configuration.

        Examples
        --------
        >>> loader = CacheLoaderFactory.create_default(cache_size=100)
        >>> str(loader)
        'CacheLoader with cache enabled (100 slots, 0 items cached)'
        """
        if not self.cache_enabled:
            return "CacheLoader with cache disabled"
        cache_info = self.cache_info()
        currsize = cache_info.get("currsize", 0) if cache_info else 0
        return (
            f"CacheLoader with cache enabled "
            f"({self.cache_maxsize} slots, {currsize} items cached)"
        )

    def select_cache_file(
        self,
        cache_dir: str | PathLike[str],
        domain: str,
        *,
        prefer_latest: bool = True,
        allow_npy: bool = True,
    ) -> str | None:
        """Select cache file (tries custom selector, then .npz/.npy, then glob)."""
        if not domain:
            raise ValueError("domain must be a non-empty string")

        cache_path = Path(cache_dir)
        logger.debug(
            f"Searching for cache file: domain={domain}, cache_dir={cache_path}"
        )

        # Chain of responsibility pattern for file selection
        for strategy in self._get_selection_strategies(
            cache_path, domain, prefer_latest, allow_npy
        ):
            result = strategy()
            if result:
                return result

        logger.warning(f"No cache file found for domain '{domain}' in {cache_path}")
        return None

    def _get_selection_strategies(
        self, cache_path: Path, domain: str, prefer_latest: bool, allow_npy: bool
    ) -> list[Callable[[], str | None]]:
        """Get ordered list of selection strategies to try."""
        strategies = [
            lambda: self._try_custom_selector(cache_path, domain),
            lambda: self._try_standard_files(cache_path, domain, allow_npy),
        ]
        if prefer_latest:
            strategies.append(
                lambda: self._try_latest_match(cache_path, domain, allow_npy)
            )
        return strategies

    def _try_custom_selector(self, cache_path: Path, domain: str) -> str | None:
        """Try custom selector if configured."""
        if self._selector is None:
            return None
        try:
            result = self._selector(str(cache_path), domain)
            if result:
                logger.debug(f"Custom selector found cache file: {result}")
            return result
        except Exception:
            logger.exception("Injected selector raised an exception")
            return None

    def _try_standard_files(
        self, cache_path: Path, domain: str, allow_npy: bool
    ) -> str | None:
        """Try standard cache file naming conventions."""
        for ext in [_NPZ_EXTENSION] + ([_NPY_EXTENSION] if allow_npy else []):
            candidate = cache_path / self._build_cache_filename(domain, ext)
            if candidate.exists():
                logger.debug(f"Found {ext.upper()} cache file: {candidate}")
                return str(candidate)
        return None

    def _try_latest_match(
        self, cache_path: Path, domain: str, allow_npy: bool
    ) -> str | None:
        """Find latest matching cache file by glob pattern."""
        if not cache_path.is_dir():
            return None
        matches = self._find_matching_cache_files(cache_path, domain, allow_npy)
        if matches:
            newest = max(matches, key=lambda p: p.stat().st_mtime)
            logger.debug(f"Found latest matching cache file: {newest}")
            return str(newest)
        return None

    def _find_matching_cache_files(
        self, cache_path: Path, domain: str, allow_npy: bool = True
    ) -> list[Path]:
        """Find all matching cache files for a domain via globbing.

        Parameters
        ----------
        cache_path : Path
            Directory to search.
        domain : str
            Domain identifier. Must be non-empty.
        allow_npy : bool, default=True
            If True, also search for .npy files."""
        try:
            return self._glob_and_filter(cache_path, domain, allow_npy)
        except Exception:
            logger.debug("Error globbing cache directory %s", cache_path, exc_info=True)
            return []

    def _glob_and_filter(
        self, cache_path: Path, domain: str, allow_npy: bool
    ) -> list[Path]:
        """Glob for matching files and filter existing ones."""
        patterns = [f"{_FILE_PREFIX}*{domain}*{_NPZ_EXTENSION}"]
        if allow_npy:
            patterns.append(f"{_FILE_PREFIX}*{domain}*{_NPY_EXTENSION}")

        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(p for p in cache_path.glob(pattern) if p.exists())
        return matches

    def _call_loader(
        self, path: Path, *, mmap_mode: str | None = None
    ) -> NDArray[Any] | NpzFile:
        """Load file using numpy loader with optional memory mapping."""
        mmap_str = (
            f"with mmap_mode='{mmap_mode}'" if mmap_mode else "without memory mapping"
        )
        logger.debug(f"Loading file {path.name} {mmap_str}")
        if mmap_mode is not None:
            return self._np_load(str(path), mmap_mode=mmap_mode, allow_pickle=False)
        return self._np_load(str(path), allow_pickle=False)

    def _extract_array_from_archive(
        self, archive: NpzFile
    ) -> NDArray[np.float64] | None:
        """
        Extract array from NPZ archive.

                Uses the internal extraction logic: prioritizes 'full_stack' key,
                then returns the first available array.

                Parameters
                ----------
                archive : NpzFile
                    Loaded NPZ archive object.

                Returns
                -------
                NDArray[np.float64] | None
                    Extracted array as float64, or None on error.

                Notes
                -----
                This method delegates to the shared extraction logic.
                Exceptions are logged but not raised.
        """
        return self._extract_array_from_npz(archive)

    def _as_float64(self, arr: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Convert array to float64."""
        return np.asarray(arr).astype(np.float64)

    def _process_loaded_data(self, loaded: Any, p: Path) -> NDArray[Any] | None:
        """Process loaded data based on type (memmap, NPZ, or regular array)."""
        if isinstance(loaded, np.memmap):
            logger.debug("Loaded memory-mapped array from %s", p.name)
            return cast(NDArray[Any], loaded)

        if isinstance(loaded, NpzFile):
            return self._extract_from_npz(loaded, p)

        logger.debug(f"Loaded and converted array from {p.name} to float64")
        return np.asarray(loaded)

    def _extract_from_npz(self, npz_file: Any, p: Path) -> NDArray[np.float64] | None:
        """Extract array from NPZ archive using configured or default extractor."""
        with npz_file as archive:
            # Try custom extractor first
            if self._archive_extractor is not None:
                result = self._try_custom_extractor(self._archive_extractor, archive, p)
                if result is not None:
                    return result

            # Fall back to default extraction
            result = self._extract_array_from_archive(archive)
            if result is not None:
                logger.debug(f"Extracted array from NPZ {p.name}")
            return result

    def _try_custom_extractor(
        self, extractor: ArchiveExtractorProtocol, archive: NpzFile, p: Path
    ) -> NDArray[np.float64] | None:
        """Try custom archive extractor with error handling."""
        try:
            result = extractor(archive)
            if result is not None:
                logger.debug(
                    f"Extracted array from NPZ {p.name} using custom extractor"
                )
            return result
        except Exception as e:
            logger.exception(
                f"Custom archive_extractor failed for {p.name}: {type(e).__name__}"
            )
            return None

    def _load_uncached(
        self,
        p: Path,
        *,
        mmap_mode: str | None = None,
        raise_on_error: bool = False,
    ) -> NDArray[Any] | None:
        """Load a file without consulting the cache.

        Handles both single-array files (.npy) and multi-array archives (.npz).
        Preserves memory-mapped access when requested. Converts other arrays to
        float64.

        Parameters
        ----------
        p : Path
            File path to load.
        mmap_mode : str | None
            Memory mapping mode. None disables mmap.
        raise_on_error : bool, default=False
            If True, raises exceptions. If False, logs and returns None.

        Returns
        -------
        NDArray | None
            Loaded array, or None on error.

        Raises
        ------
        OSError
            If raise_on_error=True and file cannot be accessed.
        ValueError
            If raise_on_error=True and data cannot be extracted.

        Notes
        -----
        - Memory-mapped arrays are returned as-is (not converted to float64)
        - Non-memmap arrays are converted to float64
        - Exceptions are logged either way
        - NPZ files are automatically detected and extracted

        Examples
        --------
        Load an NPY file (returns float64 array):
        >>> loader = CacheLoaderFactory.create_default()
        >>> arr = loader._load_uncached(Path("data.npy"))

        Load with memory mapping (bypasses conversion):
        >>> arr = loader._load_uncached(Path("data.npy"), mmap_mode="r")
        """
        try:
            loaded = self._call_loader(p, mmap_mode=mmap_mode)
            return self._process_loaded_data(loaded, p)
        except OSError as e:
            logger.error(f"File access error loading {p.name}: {e}")
            if raise_on_error:
                raise
            return None
        except (ValueError, TypeError) as e:
            logger.error(f"Data format error in {p.name}: {e}")
            if raise_on_error:
                raise
            return None
        except Exception as e:
            logger.exception(f"Unexpected error loading {p.name}: {type(e).__name__}")
            if raise_on_error:
                raise
            return None

    @property
    def cache_enabled(self) -> bool:
        """Check if caching is active.

        Returns
        -------
        bool
            True if a cache instance is configured, False otherwise.
        """
        return self._cache is not None

    @property
    def cache_maxsize(self) -> int:
        """Get the configured cache maximum size.

        Returns
        -------
        int
            Maximum number of items the cache can hold.
        """
        return self._cache_size

    def cache_info(self) -> dict[str, int | float | bool] | None:
        """
        Get cache statistics.

                Returns information about cache hits, misses, and size if available.

                Returns
                -------
                dict | None
                    Cache statistics if cache is enabled, None otherwise.

                See Also
                --------
                cache_enabled : Check if caching is active
        """
        if self._cache is None:
            return None
        return self._cache.info()

    def cache_keys(self) -> list[str]:
        """Get all keys currently in the cache.

        Returns
        -------
            list[str]
                List of cache keys (file paths). Empty list if cache is disabled.
        """
        if self._cache is None:
            return []
        return self._cache.keys()

    def cache_clear(self) -> None:
        """Clear all entries from the cache.

        No-op if cache is not enabled.
        """
        if self._cache is None:
            return
        self._cache.clear()

    def cache_status(self) -> dict[str, bool | int | float]:
        """Get a user-friendly cache status report.

        Returns comprehensive information about cache configuration and performance
        including enabled status, capacity, current usage, hit rate, and key count.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'enabled' (bool): Whether caching is active
            - 'maxsize' (int): Maximum cache capacity
            - 'currsize' (int): Current number of items cached
            - 'hits' (int): Number of cache hits
            - 'misses' (int): Number of cache misses
            - 'hit_rate' (float): Hit rate as percentage (0-100), or 0 if no accesses
            - 'num_keys' (int): Number of unique keys currently cached

        Examples
        --------
        >>> loader = CacheLoaderFactory.create_default(cache_size=100)
        >>> status = loader.cache_status()
        >>> print(f"Cache enabled: {status['enabled']}")
        >>> print(f"Hit rate: {status['hit_rate']:.1f}%")
        >>> print(f"Usage: {status['currsize']}/{status['maxsize']}")
        """
        if not self.cache_enabled:
            return {
                "enabled": False,
                "maxsize": 0,
                "currsize": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "num_keys": 0,
            }

        cache_info = self.cache_info()
        hits = cache_info.get("hits", 0) if cache_info else 0
        misses = cache_info.get("misses", 0) if cache_info else 0
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0.0

        return {
            "enabled": True,
            "maxsize": self.cache_maxsize,
            "currsize": cache_info.get("currsize", 0) if cache_info else 0,
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "num_keys": len(self.cache_keys()),
        }

    def load_full_stack(
        self,
        path: str | PathLike[str] | None,
        *,
        mmap_mode: str | None = None,
        raise_on_error: bool = False,
    ) -> NDArray[Any] | None:
        """
        Load AVO full stack array from cache file.

                Main entry point for loading data. Automatically handles caching:
                - Returns cached copy if available (when mmap_mode is None)
                - Caches non-memmap arrays as float64
                - Preserves memmap access for large files

                Parameters
                ----------
                path : str | PathLike[str] | None
                    File path to load. If None or empty, returns None.
                mmap_mode : str | None
                    Memory mapping mode. None disables mmap. Common values:
                    - 'r': read-only
                    - 'r+': read-write
                    - 'c': copy-on-write
                raise_on_error : bool, default=False
                    If True, raises exceptions. If False, logs and returns None.

                Returns
                -------
                NDArray | None
                    Loaded array, or None if file not found or loading failed.

                Examples
                --------
                Load with caching (default):
                >>> loader = CacheLoaderFactory.create_default(cache_size=100)
                >>> data = loader.load_full_stack("/path/to/cache/avo_acoustic.npz")

                Load with memory mapping (bypasses cache):
                >>> data = loader.load_full_stack(
                ...     "/path/to/cache/avo_acoustic.npz",
                ...     mmap_mode="r"
                ... )

                Notes
                -----
                - Cache lookup only occurs when mmap_mode is None
                - Memory-mapped arrays are never cached
                - Only non-memmap arrays are cached as float64 copies
                - File existence is checked before loading
        """
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            logger.debug("Cache file does not exist: %s", path)
            return None

        # Cache the string representation to avoid repeated conversions
        cache_key = str(p)

        try:
            if mmap_mode is None and self._cache is not None:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    logger.debug("Cache hit for %s", cache_key)
                    return cached

            arr = self._load_uncached(
                p, mmap_mode=mmap_mode, raise_on_error=raise_on_error
            )

            if (
                arr is not None
                and self._cache is not None
                and not isinstance(arr, np.memmap)
            ):
                try:
                    # Only cache if arr is a floating point array
                    if hasattr(arr, "dtype") and np.issubdtype(arr.dtype, np.floating):
                        self._cache.set(
                            cache_key,
                            self._as_float64(cast(NDArray[np.floating[Any]], arr)),
                        )
                except Exception:
                    logger.warning(
                        "Failed to cache loaded array for %s", path, exc_info=True
                    )

            return arr
        except Exception:
            logger.exception("Failed to load cache file: %s", path)
            if raise_on_error:
                raise
        return None


class CacheLoaderFactory(GenericFactory["CacheLoader"]):
    """Factory for creating CacheLoader instances with various configurations.

    Uses the generic factory pattern to eliminate boilerplate code.

    Examples
    --------
    Create with custom settings:
    >>> factory = CacheLoaderFactory()
    >>> loader = factory.create("custom", cache_size=100, shards=4)

    Create with defaults:
    >>> loader = factory.create("default", cache_size=100)
    """

    def __init__(self) -> None:
        """Initialize factory with standard builders."""
        super().__init__()
        self._register_standard_builders()

    def _register_standard_builders(self) -> None:
        """Register standard cache loader builders."""

        @self.register("custom")
        def create_custom(
            cache_size: int = 0,
            shards: int = 1,
            cache: CacheProtocol[NDArray[np.float64]] | None = None,
            selector: SelectorProtocol | None = None,
            archive_extractor: ArchiveExtractorProtocol | None = None,
        ) -> CacheLoader:
            """Create a CacheLoader with custom configuration."""
            if cache is None and cache_size > 0:
                if shards > 1:
                    cache = ShardedLRUCache[NDArray[np.float64]](
                        maxsize=cache_size, shards=shards
                    )
                else:
                    cache = LRUCache[NDArray[np.float64]](cache_size)

            return CacheLoader(
                selector=selector,
                cache=cache,
                cache_size=cache_size,
                archive_extractor=archive_extractor,
            )

        @self.register("default")
        def create_default(
            cache_size: int = 0,
            shards: int = 4,
            selector: SelectorProtocol | None = None,
        ) -> CacheLoader:
            """Create a CacheLoader with sensible defaults."""
            cache = None
            if cache_size > 0:
                cache = ShardedLRUCache[NDArray[np.float64]](
                    maxsize=cache_size, shards=shards
                )
            return CacheLoader(
                selector=(selector or CacheLoader.default_selector),
                cache=cache,
                cache_size=cache_size,
                archive_extractor=CacheLoader.default_archive_extractor,
            )

        del create_custom, create_default

    # Backward compatibility static methods
    def create(self, name: str, **kwargs: Any) -> CacheLoader:
        """Create instance using registered builder (instance method).

        This implements the `GenericFactory.create` signature so static type
        checkers consider this an override. Legacy class-level convenience
        callables are attached to the class after its definition to preserve
        the historical `CacheLoaderFactory.create(...)` usage.
        """
        return super().create(name, **kwargs)

    def _create_custom(self, **kwargs: Any) -> CacheLoader:
        """Internal method that calls the registered builder."""
        return super().create("custom", **kwargs)

    def _create_default(self, **kwargs: Any) -> CacheLoader:
        """Internal method that calls the registered builder."""
        return super().create("default", **kwargs)

    @classmethod
    def create_custom(cls, **kwargs: Any) -> CacheLoader:
        """Class-level convenience wrapper for creating a 'custom' loader."""
        factory = cls()
        return factory._create_custom(**kwargs)

    @classmethod
    def create_default(cls, **kwargs: Any) -> CacheLoader:
        """Class-level convenience wrapper for creating a 'default' loader."""
        factory = cls()
        return factory._create_default(**kwargs)
