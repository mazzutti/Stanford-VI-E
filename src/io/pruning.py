"""Pruning layer for cache management.

Consolidates TTL and size-based pruning logic. Provides strategies and
utilities for maintaining cache within size and time constraints.

Design:
- PruneStrategy: Defines which files to remove
- Pruner: Executes pruning and removes files
- Pure functions for testability
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "should_expire_by_ttl",
    "should_expire_by_size",
    "PruneStrategy",
    "Pruner",
    "PruneResult",
]

logger = logging.getLogger(__name__)

# Small pruning helpers and strategy classes intentionally expose a
# compact public surface; silence too-few-public-methods for clarity.


def should_expire_by_ttl(
    file_path: Path,
    ttl_seconds: int | None,
    now: float | None = None,
) -> bool:
    """Check if a file should be considered expired by TTL.

    Parameters
    ----------
    file_path : Path
        Path to the file to check.
    ttl_seconds : int | None
        TTL in seconds, or None to disable TTL checking.
    now : float | None
        Current time (seconds since epoch). If None, uses time.time().

    Returns
    -------
    bool
        True if the file has exceeded TTL, False otherwise.

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
        return False


def should_expire_by_size(
    files: Sequence[Path],
    max_cache_bytes: int,
    get_size: Callable[[Path], int] | None = None,
) -> bool:
    """Check if total cache size exceeds limit.

    Parameters
    ----------
    files : Sequence[Path]
        Collection of file paths to check.
    max_cache_bytes : int
        Maximum allowed total size in bytes.
    get_size : Callable | None
        Function to get file size. Defaults to Path.stat().st_size.

    Returns
    -------
    bool
        True if total size exceeds max_cache_bytes.

    """
    if get_size is None:

        def _default_get_size(p: Path) -> int:
            try:
                return p.stat().st_size
            except (OSError, ValueError):
                return 0

        get_size = _default_get_size

    try:
        total = sum(get_size(f) for f in files)
        return total > max_cache_bytes
    except (OSError, ValueError):
        return False


@dataclass
class PruneStrategy:
    """Strategy for selecting files to remove during cache pruning.

    Defines how files are selected for removal based on TTL and size constraints.

    Attributes
    ----------
    ttl_seconds : int | None
        TTL in seconds for cache entries.
    max_cache_bytes : int
        Maximum total cache size.
    glob_pattern : str
        Glob pattern for finding cache files.

    """

    ttl_seconds: int | None
    max_cache_bytes: int
    glob_pattern: str = "*.npz"

    @classmethod
    def by_size_only(cls, max_cache_bytes: int) -> PruneStrategy:
        """Create strategy that prunes only by size (no TTL)."""
        return cls(ttl_seconds=None, max_cache_bytes=max_cache_bytes)

    @classmethod
    def by_ttl_only(cls, ttl_seconds: int) -> PruneStrategy:
        """Create strategy that prunes only by TTL (no size limit)."""
        return cls(ttl_seconds=ttl_seconds, max_cache_bytes=int(1e18))

    @classmethod
    def by_size_then_ttl(
        cls,
        max_cache_bytes: int,
        ttl_seconds: int | None = None,
    ) -> PruneStrategy:
        """Create strategy with combined size and TTL constraints."""
        return cls(ttl_seconds=ttl_seconds, max_cache_bytes=max_cache_bytes)

    def select_for_removal(
        self,
        cache_dir: Path,
        now: float | None = None,
        get_size: Callable[[Path], int] | None = None,
    ) -> list[Path]:
        """Select files to remove to satisfy constraints.

        Selection logic:
        1. Remove files that have exceeded TTL (if enabled)
        2. If size > max_cache_bytes, remove oldest files (by mtime)

        Parameters
        ----------
        cache_dir : Path
            Directory containing cache files.
        now : float | None
            Current time for TTL checking.
        get_size : Callable | None
            Function to get file size.

        Returns
        -------
        list[Path]
            List of file paths to remove.

        """
        if now is None:
            now = time.time()

        if get_size is None:

            def _default_get_size(p: Path) -> int:
                try:
                    return p.stat().st_size
                except (OSError, ValueError):
                    return 0

            get_size = _default_get_size

        to_remove: list[Path] = []

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
                remaining_sorted = sorted(remaining, key=self._get_mtime)

                for f in remaining_sorted:
                    if total_size <= self.max_cache_bytes:
                        break
                    size = get_size(f)
                    to_remove.append(f)
                    total_size -= size

        except (OSError, ValueError) as e:
            logger.debug("Error during cache pruning analysis: %s", e)

        return to_remove

    @staticmethod
    def _get_mtime(path: Path) -> float:
        """Get modification time for a path (safe wrapper)."""
        try:
            return path.stat().st_mtime
        except (OSError, ValueError):
            return sys.maxsize


@dataclass
class PruneResult:
    """Results from a pruning operation.

    Attributes
    ----------
    count : int
        Number of files successfully removed.
    bytes_freed : int
        Total bytes freed.
    errors : int
        Number of errors encountered.
    """

    count: int
    bytes_freed: int
    errors: int

    @property
    def success(self) -> bool:
        """Check if pruning was entirely successful."""
        return self.errors == 0

    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "✓" if self.success else "⚠"
        mb_freed = self.bytes_freed / (1024**2)
        msg = f"{status} Pruned {self.count} files, freed {mb_freed:.1f} MB"
        if self.errors > 0:
            msg += f", {self.errors} errors"
        return msg


class Pruner:
    """Executes cache pruning strategy and removes files.

    Wraps PruneStrategy and handles actual file deletion with error handling.

    Attributes
    ----------
    strategy : PruneStrategy
        The pruning strategy to use.
    logger_obj : logging.Logger
        Logger for debug messages.
    """

    def __init__(
        self,
        strategy: PruneStrategy,
        logger_obj: logging.Logger | None = None,
    ):
        """Initialize the pruner.

        Parameters
        ----------
        strategy : PruneStrategy
            Pruning strategy to use.
        logger_obj : logging.Logger | None
            Logger instance.

        """
        self.strategy = strategy
        self.logger_obj = logger_obj or logger

    def prune(self, cache_dir: Path) -> PruneResult:
        """Apply pruning strategy and remove files.

        Calls strategy.select_for_removal() and attempts to delete each file,
        returning statistics about the operation. All errors are caught and logged.

        Parameters
        ----------
        cache_dir : Path
            Cache directory to prune.

        Returns
        -------
        PruneResult
            Statistics about the pruning operation.
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
                    self.logger_obj.debug("Pruned cache file: %s", path.name)
                except (OSError, ValueError) as e:
                    result.errors += 1
                    self.logger_obj.debug("Failed to prune %s: %s", path.name, e)
        except (OSError, ValueError) as e:
            self.logger_obj.debug("Error during cache pruning: %s", e)
            result.errors = 1

        return result
