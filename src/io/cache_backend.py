"""Shared cache backend primitives for TTL and size-based pruning.

This module provides reusable, low-level cache management primitives that
can be composed into higher-level cache implementations. It abstracts the
common patterns of TTL-based expiration and size-based pruning used by
both CacheLoader (in-memory LRU) and DiskCache (disk-backed NPZ storage).

Design Principles:
    - Pure functions where possible (no side effects)
    - Composition over inheritance (use in existing classes)
    - Dependency injection for file system operations (testability)
    - Comprehensive error handling (best-effort operations)
    - Type-safe with full annotations

Example Usage:
    >>> from pathlib import Path
    >>> from src.io.cache_backend import PruneStrategy, should_expire_by_ttl
    >>> import time
    >>>
    >>> # Check if a file has expired by TTL
    >>> cache_dir = Path(".cache")
    >>> now = time.time()
    >>> for file in cache_dir.glob("*.npz"):
    ...     if should_expire_by_ttl(file, ttl_seconds=3600, now=now):
    ...         file.unlink()  # Remove expired files
    >>>
    >>> # Or use the pruning strategy
    >>> strategy = PruneStrategy.by_size_then_ttl(
    ...     max_bytes=10*1024**3, ttl_seconds=86400
    ... )
    >>> files_to_remove = strategy.select_for_removal(cache_dir)
    >>> for f in files_to_remove:
    ...     f.unlink()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "should_expire_by_ttl",
    "should_expire_by_size",
    "PruneStrategy",
    "TTLAndSizePruner",
]


def should_expire_by_ttl(
    file_path: Path,
    ttl_seconds: Optional[int],
    now: Optional[float] = None,
) -> bool:
    """Check if a file should be considered expired by TTL.

    Parameters
    ----------
    file_path : Path
        Path to the file to check.
    ttl_seconds : Optional[int]
        TTL in seconds, or None to disable TTL checking.
    now : Optional[float]
        Current time (seconds since epoch). If None, uses time.time().
        Useful for testing with fixed time values.

    Returns
    -------
    bool
        True if the file has exceeded TTL and should be removed, False otherwise.
        Returns False if ttl_seconds is None (TTL disabled).

    Examples
    --------
    >>> from pathlib import Path
    >>> import time
    >>> cache_file = Path("cache.npz")
    >>> # File expires after 1 hour
    >>> if should_expire_by_ttl(cache_file, ttl_seconds=3600):
    ...     cache_file.unlink()
    """
    if ttl_seconds is None:
        return False

    if now is None:
        now = time.time()

    try:
        mtime = file_path.stat().st_mtime
        age_seconds = now - mtime
        return age_seconds > ttl_seconds
    except (OSError, ValueError):
        # File may have been deleted or stat failed
        return False


def should_expire_by_size(
    files: Sequence[Path],
    max_cache_bytes: int,
    get_size: Optional[Callable[[Path], int]] = None,
) -> bool:
    """Check if total cache size exceeds limit.

    Parameters
    ----------
    files : Sequence[Path]
        Collection of file paths to check.
    max_cache_bytes : int
        Maximum allowed total size in bytes.
    get_size : Optional[Callable]
        Function to get file size. Defaults to Path.stat().st_size.
        Useful for testing or custom size calculations.

    Returns
    -------
    bool
        True if total size exceeds max_cache_bytes, False otherwise.

    Examples
    --------
    >>> from pathlib import Path
    >>> cache_files = list(Path(".cache").glob("*.npz"))
    >>> if should_expire_by_size(cache_files, max_cache_bytes=1_000_000_000):
    ...     # Need to prune cache
    ...     print("Cache too large!")
    """
    if get_size is None:

        def get_size(p: Path) -> int:
            try:
                return p.stat().st_size
            except (OSError, ValueError):
                return 0

    try:
        total = sum(get_size(f) for f in files)
        return total > max_cache_bytes
    except (OSError, ValueError):
        # Calculation error, assume we don't need to prune
        return False


@dataclass
class PruneStrategy:
    """Strategy for selecting files to remove during cache pruning.

    This class defines how files are selected for removal based on various
    criteria (TTL, modification time, size). It's designed to be composed
    with actual file deletion logic (which must be implemented separately
    for better testability and error handling).

    Attributes
    ----------
    ttl_seconds : Optional[int]
        TTL in seconds for cache entries. Files older than this are
        marked for removal first.
    max_cache_bytes : int
        Maximum total cache size. If exceeded, oldest files are removed
        until size is under this threshold.
    glob_pattern : str
        Glob pattern for finding cache files (e.g., "*.npz").
        Defaults to "*.npz".

    Methods
    -------
    select_for_removal(cache_dir: Path) -> List[Path]
        Returns list of file paths that should be removed to satisfy
        both TTL and size constraints.
    """

    ttl_seconds: Optional[int]
    max_cache_bytes: int
    glob_pattern: str = "*.npz"

    @classmethod
    def by_size_only(cls, max_cache_bytes: int) -> PruneStrategy:
        """Create strategy that prunes only by size (no TTL).

        Parameters
        ----------
        max_cache_bytes : int
            Maximum total cache size in bytes.

        Returns
        -------
        PruneStrategy
            Strategy with TTL disabled.
        """
        return cls(ttl_seconds=None, max_cache_bytes=max_cache_bytes)

    @classmethod
    def by_ttl_only(cls, ttl_seconds: int) -> PruneStrategy:
        """Create strategy that prunes only by TTL (no size limit).

        Parameters
        ----------
        ttl_seconds : int
            TTL in seconds for cache entries.

        Returns
        -------
        PruneStrategy
            Strategy with no size limit (max_cache_bytes=inf).
        """
        return cls(ttl_seconds=ttl_seconds, max_cache_bytes=int(1e18))

    @classmethod
    def by_size_then_ttl(
        cls,
        max_cache_bytes: int,
        ttl_seconds: Optional[int] = None,
    ) -> PruneStrategy:
        """Create strategy that applies size and TTL constraints.

        This is the most common strategy: first expire files by TTL,
        then remove oldest files until size is under the limit.

        Parameters
        ----------
        max_cache_bytes : int
            Maximum total cache size in bytes.
        ttl_seconds : Optional[int]
            Optional TTL in seconds.

        Returns
        -------
        PruneStrategy
            Strategy with both size and TTL constraints.
        """
        return cls(ttl_seconds=ttl_seconds, max_cache_bytes=max_cache_bytes)

    def select_for_removal(
        self,
        cache_dir: Path,
        now: Optional[float] = None,
        get_size: Optional[Callable[[Path], int]] = None,
    ) -> List[Path]:
        """Select files to remove to satisfy TTL and size constraints.

        Selection logic:
        1. Remove all files that have exceeded TTL (if TTL is enabled)
        2. Recalculate total size
        3. If size > max_cache_bytes, remove oldest files (by mtime) until under limit

        Parameters
        ----------
        cache_dir : Path
            Directory containing cache files.
        now : Optional[float]
            Current time for TTL checking. Defaults to time.time().
        get_size : Optional[Callable]
            Function to get file size. Defaults to Path.stat().st_size.

        Returns
        -------
        List[Path]
            List of file paths that should be removed.

        Examples
        --------
        >>> strategy = PruneStrategy.by_size_then_ttl(
        ...     max_cache_bytes=1_000_000_000,
        ...     ttl_seconds=86400
        ... )
        >>> cache_dir = Path(".cache")
        >>> files_to_remove = strategy.select_for_removal(cache_dir)
        >>> for f in files_to_remove:
        ...     f.unlink()
        """
        if now is None:
            now = time.time()

        if get_size is None:

            def get_size(p: Path) -> int:
                try:
                    return p.stat().st_size
                except (OSError, ValueError):
                    return 0

        to_remove: List[Path] = []

        try:
            # Get all cache files
            files = list(cache_dir.glob(self.glob_pattern))
            if not files:
                return to_remove

            # Phase 1: Expire files by TTL
            if self.ttl_seconds is not None:
                for f in files:
                    if should_expire_by_ttl(f, self.ttl_seconds, now):
                        to_remove.append(f)

            # Calculate remaining files and total size
            remaining = [f for f in files if f not in to_remove]
            total_size = sum(get_size(f) for f in remaining)

            # Phase 2: Remove oldest files if size exceeds limit
            if total_size > self.max_cache_bytes:
                # Sort by modification time (oldest first)
                remaining_sorted = sorted(remaining, key=lambda p: self._get_mtime(p))

                for f in remaining_sorted:
                    if total_size <= self.max_cache_bytes:
                        break
                    size = get_size(f)
                    to_remove.append(f)
                    total_size -= size

        except (OSError, ValueError) as e:
            logger.debug(f"Error during cache pruning analysis: {e}")
            # Return what we found so far
            pass

        return to_remove

    @staticmethod
    def _get_mtime(path: Path) -> float:
        """Get modification time for a path (safe wrapper).

        Returns sys.maxsize if stat fails, so file sorts to end.
        """
        try:
            return path.stat().st_mtime
        except (OSError, ValueError):
            import sys

            return sys.maxsize


class TTLAndSizePruner:
    """Utility class for applying pruning strategy to a cache directory.

    This class wraps PruneStrategy and provides a convenient interface
    for actually removing files from disk. It handles all error cases
    gracefully and logs pruning actions.

    Attributes
    ----------
    strategy : PruneStrategy
        The pruning strategy to use.
    logger_obj : logging.Logger
        Logger for debug/info messages.

    Methods
    -------
    prune(cache_dir: Path) -> PruneResult
        Apply pruning strategy and remove files, returning statistics.
    """

    def __init__(
        self,
        strategy: PruneStrategy,
        logger_obj: Optional[logging.Logger] = None,
    ):
        """Initialize the pruner.

        Parameters
        ----------
        strategy : PruneStrategy
            Pruning strategy to use.
        logger_obj : Optional[logging.Logger]
            Logger for debug messages. Defaults to module logger.
        """
        self.strategy = strategy
        self.logger_obj = logger_obj or logger

    def prune(self, cache_dir: Path) -> "PruneResult":
        """Apply pruning strategy and remove expired/oversized files.

        This method:
        1. Calls strategy.select_for_removal() to get files to remove
        2. Attempts to delete each file
        3. Returns statistics (removed count, freed bytes, errors)

        All errors are caught and logged; this method never raises.

        Parameters
        ----------
        cache_dir : Path
            Cache directory to prune.

        Returns
        -------
        PruneResult
            Statistics about the pruning operation.

        Examples
        --------
        >>> strategy = PruneStrategy.by_size_then_ttl(
        ...     max_cache_bytes=1_000_000_000,
        ...     ttl_seconds=86400
        ... )
        >>> pruner = TTLAndSizePruner(strategy)
        >>> result = pruner.prune(Path(".cache"))
        >>> print(f"Removed {result.count} files, freed {result.bytes_freed} bytes")
        """
        result = PruneResult(count=0, bytes_freed=0, errors=0)

        try:
            to_remove = self.strategy.select_for_removal(cache_dir)
            for path in to_remove:
                try:
                    size = path.stat().st_size
                    path.unlink()
                    result.count += 1
                    result.bytes_freed += size
                    self.logger_obj.debug(f"Pruned cache file: {path.name}")
                except (OSError, ValueError) as e:
                    result.errors += 1
                    self.logger_obj.debug(f"Failed to prune {path.name}: {e}")
        except (OSError, ValueError) as e:
            self.logger_obj.debug(f"Error during cache pruning: {e}")
            result.errors = 1

        return result


@dataclass
class PruneResult:
    """Results from a pruning operation.

    Attributes
    ----------
    count : int
        Number of files successfully removed.
    bytes_freed : int
        Total bytes freed by removing files.
    errors : int
        Number of errors encountered during pruning.
    """

    count: int
    bytes_freed: int
    errors: int

    @property
    def success(self) -> bool:
        """Check if pruning was entirely successful (no errors)."""
        return self.errors == 0

    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "✓" if self.success else "⚠"
        mb_freed = self.bytes_freed / (1024**2)
        return (
            f"{status} Pruned {self.count} files, freed {mb_freed:.1f} MB"
            f"{f', {self.errors} errors' if self.errors > 0 else ''}"
        )
