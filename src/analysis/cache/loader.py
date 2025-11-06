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
from typing import Any, Callable, Optional, Union, List, NamedTuple, cast
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
    archive_extractor : Optional[ArchiveExtractorProtocol]
        Custom NPZ extraction strategy, or None for default.
    selector : Optional[SelectorProtocol]
        Custom file selection strategy, or None for default.
    cache : Optional[CacheProtocol[NDArray[np.float64]]]
        External cache instance, or None to create default.
    np_load : Callable
        NumPy load function (for testing/mocking).
    """

    cache_size: int
    archive_extractor: Optional[ArchiveExtractorProtocol] = None
    selector: Optional[SelectorProtocol] = None
    cache: Optional[CacheProtocol[NDArray[np.float64]]] = None
    np_load: Callable[..., Union[NDArray[np.float64], NpzFile]] = np.load


class CacheLoader:
    """Load and cache AVO data files with LRU caching and memory mapping support.

    This class provides efficient loading of AVO cache files (NPZ or NPY format)
    with optional in-memory LRU caching. It preserves memory-mapped access for
    large datasets while caching non-memmap arrays as float64 copies.

    Attributes
    ----------
    _selector : Optional[SelectorProtocol]
        Custom file selection callable. If None, uses default selection logic.
    _np_load : Callable
        NumPy load function (can be mocked for testing).
    _archive_extractor : Optional[ArchiveExtractorProtocol]
        Custom NPZ archive extraction callable.
    _cache : Optional[CacheProtocol]
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
        """Build standard cache filename for a domain.

        Parameters
        ----------
        domain : str
            Domain identifier (e.g., 'acoustic', 'elastic'). Must be non-empty.
        extension : str, default=".npz"
            File extension to use.

        Returns
        -------
        str
            Formatted filename (e.g., "avo_acoustic.npz").

        Raises
        ------
        ValueError
            If domain is empty or None.
        """
        if not domain:
            raise ValueError("domain must be a non-empty string")
        return f"{_FILE_PREFIX}{domain}{extension}"

    @staticmethod
    def default_selector(cache_dir: str, domain: str) -> Optional[str]:
        """Locate AVO cache file for a given domain.

        Searches for cache files in the following priority order:
        1. avo_{domain}.npz (compressed NPZ format)
        2. avo_{domain}.npy (uncompressed NPY format)

        This is the default file selection strategy used when no custom
        selector is provided.

        Parameters
        ----------
        cache_dir : str
            Directory path containing cache files.
        domain : str
            Domain identifier (e.g., 'elastic', 'acoustic'). Must be non-empty.

        Returns
        -------
        Optional[str]
            Full path to the selected cache file, or None if not found.

        Raises
        ------
        ValueError
            If domain is empty or None.
        OSError
            If cache_dir is not accessible or doesn't exist.

        Examples
        --------
        >>> path = CacheLoader.default_selector("/data/cache", "elastic")
        >>> if path:
        ...     print(f"Found cache: {path}")
        ... else:
        ...     print("No cache file found")
        """
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
    def _extract_array_from_npz(archive: NpzFile) -> Optional[NDArray[np.float64]]:
        """Extract array from NPZ archive.

        Extraction strategy:
        1. If 'full_stack' key exists, returns that array
        2. Otherwise, returns the first array in the archive
        3. Returns None on any error and logs the exception

        This is the shared extraction logic used by both the default extractor
        and the instance method.

        Parameters
        ----------
        archive : NpzFile
            Loaded NPZ archive object.

        Returns
        -------
        Optional[NDArray[np.float64]]
            Extracted array as float64, or None if extraction failed.

        Raises
        ------
        Exception
            Internal exceptions are caught and logged, not raised.

        Notes
        -----
        The 'full_stack' key is a convention in this project for storing
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
    def default_archive_extractor(archive: NpzFile) -> Optional[NDArray[np.float64]]:
        """Extract array from NPZ archive using sensible defaults.

        Extraction strategy:
        1. If 'full_stack' key exists, returns that array
        2. Otherwise, returns the first array in the archive
        3. Returns None on any error and logs the exception

        This is the default extraction strategy used when no custom
        extractor is provided.

        Parameters
        ----------
        archive : NpzFile
            Loaded NPZ archive object.

        Returns
        -------
        Optional[NDArray[np.float64]]
            Extracted array as float64, or None if extraction failed.

        Notes
        -----
        The 'full_stack' key is a convention in this project for storing
        the main data array within NPZ files.

        Examples
        --------
        >>> import numpy as np
        >>> archive = np.load("data.npz")
        >>> data = CacheLoader.default_archive_extractor(archive)
        """
        return CacheLoader._extract_array_from_npz(archive)

    def __init__(
        self,
        selector: Optional[SelectorProtocol] = None,
        np_load: Callable[..., Union[NDArray[np.float64], NpzFile]] = np.load,
        *,
        cache: Optional[CacheProtocol[NDArray[np.float64]]] = None,
        cache_size: int = 0,
        archive_extractor: Optional[ArchiveExtractorProtocol] = None,
    ) -> None:
        """Initialize CacheLoader with optional caching and custom strategies.

        Parameters
        ----------
        selector : Optional[SelectorProtocol]
            Custom file selection callable. If None, uses default_selector.
        np_load : Callable
            NumPy load function (default: np.load). Can be mocked for testing.
        cache : Optional[CacheProtocol]
            Externally provided cache instance. If provided, cache_size is ignored.
        cache_size : int, default=0
            Size of the LRU cache. If 0 and cache is None, caching is disabled.
            Must be non-negative.
        archive_extractor : Optional[ArchiveExtractorProtocol]
            Custom NPZ extraction callable. If None, uses default_archive_extractor.

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
            self._cache: Optional[CacheProtocol[NDArray[np.float64]]] = cache
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
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit context manager for CacheLoader.

        Clears the cache when exiting the context, ensuring clean resource
        cleanup. If an exception occurred, it is not suppressed.

        Parameters
        ----------
        exc_type : Optional[type]
            The exception type if an exception occurred, None otherwise.
        exc_val : Optional[Exception]
            The exception instance if an exception occurred, None otherwise.
        exc_tb : Optional[TracebackType]
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
        cache_dir: Union[str, PathLike[str]],
        domain: str,
        *,
        prefer_latest: bool = True,
        allow_npy: bool = True,
    ) -> Optional[str]:
        """Select a cache file for the given domain.

        Uses the configured selector first. If that fails or returns None,
        falls back to default selection logic with optional "latest file"
        matching via globbing.

        Selection order:
        1. Try custom selector (if configured)
        2. Look for avo_{domain}.npz
        3. Look for avo_{domain}.npy (if allow_npy=True)
        4. If prefer_latest=True, glob for matching patterns and return newest

        Parameters
        ----------
        cache_dir : Union[str, PathLike[str]]
            Directory containing cache files.
        domain : str
            Domain identifier (e.g., 'acoustic', 'elastic').
        prefer_latest : bool, default=True
            If True and no standard file found, search for latest matching
            file by modification time.
        allow_npy : bool, default=True
            If True, also search for .npy files as fallback.

        Returns
        -------
        Optional[str]
            Full path to selected cache file, or None if not found.

        Raises
        ------
        ValueError
            If domain is empty or None.
        OSError
            If cache_dir is not accessible.

        Notes
        -----
        Exceptions from the custom selector are logged but do not raise.
        Globbing only occurs if prefer_latest=True and no standard file exists.

        Examples
        --------
        Basic file selection:
        >>> loader = CacheLoaderFactory.create_default()
        >>> path = loader.select_cache_file("/data/cache", "acoustic")

        With options:
        >>> path = loader.select_cache_file(
        ...     "/data/cache",
        ...     "elastic",
        ...     prefer_latest=True,
        ...     allow_npy=False
        ... )
        """
        if not domain:
            raise ValueError("domain must be a non-empty string")

        cache_path = Path(cache_dir)
        logger.debug(
            f"Searching for cache file: domain={domain}, cache_dir={cache_path}"
        )

        if self._selector is not None:
            try:
                result = self._selector(str(cache_path), domain)
                if result:
                    logger.debug(f"Custom selector found cache file: {result}")
                    return result
                logger.debug("Custom selector returned None")
            except Exception:
                logger.exception("Injected selector raised an exception")

        candidate_npz = cache_path / self._build_cache_filename(domain, _NPZ_EXTENSION)
        if candidate_npz.exists():
            logger.debug(f"Found NPZ cache file: {candidate_npz}")
            return str(candidate_npz)
        if allow_npy:
            candidate_npy = cache_path / self._build_cache_filename(
                domain, _NPY_EXTENSION
            )
            if candidate_npy.exists():
                logger.debug(f"Found NPY cache file: {candidate_npy}")
                return str(candidate_npy)

        if prefer_latest and cache_path.is_dir():
            existing_matches = self._find_matching_cache_files(
                cache_path, domain, allow_npy
            )
            if existing_matches:
                newest = max(existing_matches, key=lambda p: p.stat().st_mtime)
                logger.debug(f"Found latest matching cache file: {newest}")
                return str(newest)

        logger.warning(f"No cache file found for domain '{domain}' in {cache_path}")

        return None

    def _find_matching_cache_files(
        self, cache_path: Path, domain: str, allow_npy: bool = True
    ) -> List[Path]:
        """Find all matching cache files for a domain via globbing.

        Parameters
        ----------
        cache_path : Path
            Directory to search.
        domain : str
            Domain identifier. Must be non-empty.
        allow_npy : bool, default=True
            If True, also search for .npy files.

        Returns
        -------
        List[Path]
            List of existing matching files, sorted by modification time
            (newest first).

        Notes
        -----
        Uses globbing patterns to find files matching avo_*domain*.ext format.
        All results are filtered to ensure they still exist on disk.

        Examples
        --------
        >>> loader = CacheLoaderFactory.create_default()
        >>> matches = loader._find_matching_cache_files(
        ...     Path("/data/cache"),
        ...     "acoustic",
        ...     allow_npy=True
        ... )
        >>> if matches:
        ...     latest = matches[0]  # Already sorted by mtime
        """
        try:
            # Collect all matching patterns
            matches = [
                p
                for glob_result in [
                    cache_path.glob(f"{_FILE_PREFIX}*{domain}*{_NPZ_EXTENSION}"),
                    (
                        cache_path.glob(f"{_FILE_PREFIX}*{domain}*{_NPY_EXTENSION}")
                        if allow_npy
                        else []
                    ),
                ]
                for p in glob_result
                if p.exists()
            ]
            return matches
        except Exception:
            logger.debug("Error globbing cache directory %s", cache_path, exc_info=True)
            return []

    def _call_loader(
        self, path: Path, *, mmap_mode: Optional[str] = None
    ) -> Union[NDArray[np.float64], NpzFile]:
        """Load a file using the configured numpy loader.

        Attempts to load with allow_pickle=False for security. Falls back to
        loading without this parameter for compatibility with older NumPy versions.

        Parameters
        ----------
        path : Path
            File path to load.
        mmap_mode : Optional[str]
            Memory mapping mode ('r', 'r+', 'w+', 'c'). None disables mmap.

        Returns
        -------
        Union[NDArray, NpzFile]
            Loaded data or archive object.

        Raises
        ------
        Various numpy exceptions if file cannot be loaded.
        """
        mmap_str = (
            f"with mmap_mode='{mmap_mode}'" if mmap_mode else "without memory mapping"
        )
        logger.debug(f"Loading file {path.name} {mmap_str}")
        try:
            if mmap_mode is not None:
                return self._np_load(str(path), mmap_mode=mmap_mode, allow_pickle=False)
            return self._np_load(str(path), allow_pickle=False)
        except TypeError:
            logger.debug(
                "NumPy version doesn't support allow_pickle, retrying without it"
            )
            return self._np_load(str(path))

    def _extract_array_from_archive(
        self, archive: NpzFile
    ) -> Optional[NDArray[np.float64]]:
        """Extract array from NPZ archive.

        Uses the internal extraction logic: prioritizes 'full_stack' key,
        then returns the first available array.

        Parameters
        ----------
        archive : NpzFile
            Loaded NPZ archive object.

        Returns
        -------
        Optional[NDArray[np.float64]]
            Extracted array as float64, or None on error.

        Notes
        -----
        This method delegates to the shared extraction logic.
        Exceptions are logged but not raised.
        """
        return self._extract_array_from_npz(archive)

    def _as_float64(self, arr: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Convert array to float64 type.

        Parameters
        ----------
        arr : NDArray
            Input array of any compatible type.

        Returns
        -------
        NDArray[np.float64]
            Array converted to float64.
        """
        return np.asarray(arr).astype(np.float64)

    def _load_uncached(
        self,
        p: Path,
        *,
        mmap_mode: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> Union[
        NDArray[np.float64], np.ndarray[tuple[int, ...], np.dtype[np.generic]], None
    ]:
        """Load a file without consulting the cache.

        Handles both single-array files (.npy) and multi-array archives (.npz).
        Preserves memory-mapped access when requested. Converts other arrays to
        float64.

        Parameters
        ----------
        p : Path
            File path to load.
        mmap_mode : Optional[str]
            Memory mapping mode. None disables mmap.
        raise_on_error : bool, default=False
            If True, raises exceptions. If False, logs and returns None.

        Returns
        -------
        Optional[NDArray]
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

            # Return memmap as-is to preserve efficiency
            if isinstance(loaded, np.memmap):
                logger.debug(f"Loaded memory-mapped array from {p.name}")
                return loaded

            # Handle NPZ archives
            if isinstance(loaded, NpzFile):
                with loaded as archive:
                    if self._archive_extractor is not None:
                        try:
                            result = self._archive_extractor(archive)
                            if result is not None:
                                logger.debug(
                                    f"Extracted array from NPZ {p.name} using custom extractor"
                                )
                            return result
                        except Exception as e:
                            logger.exception(
                                f"Custom archive_extractor failed for {p.name}: {type(e).__name__}"
                            )
                    result = self._extract_array_from_archive(archive)
                    if result is not None:
                        logger.debug(f"Extracted array from NPZ {p.name}")
                    return result

            # For other formats (NPY, etc.), convert to float64
            logger.debug(f"Loaded and converted array from {p.name} to float64")
            return np.asarray(loaded)
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

    def cache_info(self) -> Optional[dict[str, int | float | bool]]:
        """Get cache statistics.

        Returns information about cache hits, misses, and size if available.

        Returns
        -------
        Optional[dict]
            Cache statistics if cache is enabled, None otherwise.

        See Also
        --------
        cache_enabled : Check if caching is active
        """
        if self._cache is None:
            return None
        return self._cache.info()

    def cache_keys(self) -> List[str]:
        """Get all keys currently in the cache.

        Returns
        -------
        List[str]
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
        path: Optional[Union[str, PathLike[str]]],
        *,
        mmap_mode: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> Union[
        NDArray[np.float64], np.ndarray[tuple[int, ...], np.dtype[np.generic]], None
    ]:
        """Load AVO full stack array from cache file.

        Main entry point for loading data. Automatically handles caching:
        - Returns cached copy if available (when mmap_mode is None)
        - Caches non-memmap arrays as float64
        - Preserves memmap access for large files

        Parameters
        ----------
        path : Optional[Union[str, PathLike[str]]]
            File path to load. If None or empty, returns None.
        mmap_mode : Optional[str]
            Memory mapping mode. None disables mmap. Common values:
            - 'r': read-only
            - 'r+': read-write
            - 'c': copy-on-write
        raise_on_error : bool, default=False
            If True, raises exceptions. If False, logs and returns None.

        Returns
        -------
        Optional[NDArray]
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
                    if isinstance(arr, np.ndarray) and np.issubdtype(
                        arr.dtype, np.floating
                    ):
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


class CacheLoaderFactory:
    """Factory for creating CacheLoader instances with various configurations.

    Provides factory methods for creating properly configured CacheLoader instances
    with flexible configuration options. This factory pattern ensures proper cache
    and loader initialization while remaining fully testable through dependency
    injection.

    Methods
    -------
    create(*, cache_size=0, shards=1, cache=None, selector=None, archive_extractor=None)
        Create a CacheLoader with custom configuration parameters.
    create_default(cache_size=0, shards=4, selector=None)
        Create a CacheLoader with sensible defaults for typical usage.

    Examples
    --------
    Create with custom settings:
    >>> loader = CacheLoaderFactory.create(
    ...     cache_size=100,
    ...     shards=4,
    ...     selector=custom_selector_func
    ... )

    Create with defaults:
    >>> loader = CacheLoaderFactory.create_default(cache_size=100)
    """

    @staticmethod
    def create(
        *,
        cache_size: int = 0,
        shards: int = 1,
        cache: Optional[CacheProtocol[NDArray[np.float64]]] = None,
        selector: Optional[SelectorProtocol] = None,
        archive_extractor: Optional[ArchiveExtractorProtocol] = None,
    ) -> CacheLoader:
        """Factory to create a configured CacheLoader.

        Parameters
        ----------
        cache_size : int, default=0
            Maximum number of items for a default in-memory LRU cache.
            If zero, no default cache is created.
        shards : int, default=1
            Number of shards for a sharded LRU (only used when creating
            the default cache).
        cache : CacheProtocol, optional
            An optional, caller-provided cache instance (must implement the
            CacheProtocol). If provided, it will be used as-is and the factory
            will not create a default cache regardless of ``cache_size``.
        selector : SelectorProtocol, optional
            Optional selector callable used by the returned CacheLoader.
        archive_extractor : ArchiveExtractorProtocol, optional
            Optional callable used to extract arrays from NPZ archives.

        Returns
        -------
        CacheLoader
            A configured CacheLoader instance.

        Notes
        -----
        The factory honors an injected ``cache`` instance. If ``cache`` is
        None, a default LRU or ShardedLRU is constructed only when
        ``cache_size > 0``.
        """
        # If a cache instance was explicitly provided, honor it. Otherwise,
        # create a default cache only when cache_size > 0.
        if cache is None:
            if cache_size > 0:
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

    @staticmethod
    def create_default(
        cache_size: int = 0,
        shards: int = 4,
        selector: Optional[SelectorProtocol] = None,
    ) -> CacheLoader:
        """Create a CacheLoader with sensible defaults.

        Parameters
        ----------
        cache_size : int, default=0
            Maximum number of items for the LRU cache.
        shards : int, default=4
            Number of shards for the ShardedLRU cache.
        selector : SelectorProtocol, optional
            Optional custom selector. Uses CacheLoader.default_selector if None.

        Returns
        -------
        CacheLoader
            A CacheLoader with default selector and archive extractor.
        """
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
