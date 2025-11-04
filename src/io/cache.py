"""Cache utilities.

Helpers to list and save cache files used by the project.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from src.utils.facades import LazyObjectProxy


@dataclass
class CacheEntry:
    """Represents a single cache file and basic metadata.

    Attributes:
        key: logical key prefix (e.g., 'avo')
        path: absolute Path to the cache file
        mtime: modification time (seconds since epoch)
        size_bytes: file size in bytes
    """

    key: str
    path: Path
    mtime: float
    size_bytes: int
    config_hash: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    valid: Optional[bool] = None

    @classmethod
    def from_path(cls, p: Union[str, os.PathLike]) -> "CacheEntry":
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(p)
        stat = p.stat()
        name = p.name
        key = name.split("_")[0] if "_" in name else name
        # try to extract a config/hash part (20 hex chars or longer) from filename
        import re

        m = re.search(r"([0-9a-f]{20,})", name)
        config_hash = m.group(1) if m else None

        # attempt to read NPZ metadata (non-fatal)
        config = None
        valid = None
        try:
            import numpy as _np

            with _np.load(p, allow_pickle=True) as npz:
                valid = True
                if "config" in npz:
                    cfg = npz["config"]
                    cfg_dict = None
                    try:
                        cfg_dict = dict(cfg)
                    except Exception:
                        try:
                            cfg_dict = cfg.item()
                        except Exception:
                            cfg_dict = None
                    config = cfg_dict
        except Exception:
            # if file can't be read, mark as invalid but still return metadata
            valid = False

        return cls(
            key=key,
            path=p,
            mtime=stat.st_mtime,
            size_bytes=stat.st_size,
            config_hash=config_hash,
            config=config,
            valid=valid,
        )

    @classmethod
    def from_path_shallow(cls, p: Union[str, os.PathLike]) -> "CacheEntry":
        """Create CacheEntry without attempting to read file contents (fast)."""
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(p)
        stat = p.stat()
        name = p.name
        key = name.split("_")[0] if "_" in name else name
        import re

        m = re.search(r"([0-9a-f]{20,})", name)
        config_hash = m.group(1) if m else None
        return cls(
            key=key,
            path=p,
            mtime=stat.st_mtime,
            size_bytes=stat.st_size,
            config_hash=config_hash,
        )

    def __repr__(self) -> str:
        return (
            f"CacheEntry(key={self.key!r}, path={str(self.path)!r}, "
            f"mtime={self.mtime:.0f}, size_bytes={self.size_bytes}, valid={self.valid})"
        )

    def to_dict(self) -> Dict[str, Union[str, int, float, None]]:
        return {
            "key": self.key,
            "path": str(self.path),
            "mtime": self.mtime,
            "size_bytes": self.size_bytes,
            "config_hash": self.config_hash,
            "config": self.config,
            "valid": self.valid,
        }


logger = logging.getLogger(__name__)


class CacheFileIdentifier:
    """Helper class for identifying cache files matching patterns."""

    def __init__(self, cache_dir: str, logger_obj: Optional[logging.Logger] = None):
        """Initialize identifier.

        Parameters
        ----------
        cache_dir : str
            Cache directory to scan.
        logger_obj : Optional[logging.Logger]
            Logger for messages.
        """
        self.cache_dir = Path(cache_dir)
        self.logger = logger_obj or logging.getLogger(__name__)

    def find_old_cache_files(self) -> List[str]:
        """Find old cache files (avo_* without _time_ or _depth_ suffixes).

        Returns
        -------
        List[str]
            Paths to old cache files.
        """
        if not self.cache_dir.exists():
            return []

        old_files: List[str] = []
        for file_path in self.cache_dir.glob("*.npz"):
            filename = file_path.name
            if filename.startswith("avo_") and not (
                "_time_" in filename or "_depth_" in filename
            ):
                old_files.append(str(file_path))

        return old_files


class CacheFileCleanup:
    """Helper class for cleaning up cache files."""

    def __init__(self, logger_obj: Optional[logging.Logger] = None):
        """Initialize cleanup helper.

        Parameters
        ----------
        logger_obj : Optional[logging.Logger]
            Logger for cleanup messages.
        """
        self.logger = logger_obj or logging.getLogger(__name__)

    def cleanup_files(
        self, files: List[str], dry_run: bool = False
    ) -> tuple[int, float]:
        """Remove cache files with optional dry-run mode.

        Parameters
        ----------
        files : List[str]
            Paths to files to remove.
        dry_run : bool
            If True, report what would be removed without deleting.

        Returns
        -------
        tuple[int, float]
            (removed_count, total_size_mb)
        """
        if not files:
            self.logger.info("✓ No files to remove")
            return 0, 0.0

        total_size_bytes = sum(os.path.getsize(f) for f in files)
        total_size_mb = total_size_bytes / (1024**2)

        if dry_run:
            self.logger.info(
                "DRY RUN: Would remove %d files (%.1f MB)",
                len(files),
                total_size_mb,
            )
            return 0, 0.0

        removed_count = 0
        for file_path in files:
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as e:
                self.logger.warning("Error removing %s: %s", file_path, e)

        self.logger.info(
            "✓ Removed %d/%d files (%.1f MB freed)",
            removed_count,
            len(files),
            total_size_mb,
        )

        return removed_count, total_size_mb


class CacheManager:
    """Object-oriented wrapper around cache utilities.

    Provides functionality to list, inspect, and clean up cache files.
    Uses helper classes (CacheFileIdentifier, CacheFileCleanup) for better
    separation of concerns.
    """

    def __init__(
        self, cache_dir: str = ".cache", logger: Optional[logging.Logger] = None
    ):
        """Initialize CacheManager.

        Parameters
        ----------
        cache_dir : str
            Cache directory path.
        logger : Optional[logging.Logger]
            Logger instance.
        """
        self.cache_dir = cache_dir
        self.logger = logger or logging.getLogger(__name__)
        self._identifier = CacheFileIdentifier(cache_dir, self.logger)
        self._cleanup = CacheFileCleanup(self.logger)

    # Use select_latest_cache_entries() to obtain grouped CacheEntry objects

    def select_latest_cache_entries(
        self, skip_inspect: bool = False
    ) -> Dict[str, List[CacheEntry]]:
        """Return grouped CacheEntry objects for files under the cache dir.

        Groups are keyed by the filename prefix (text before first underscore).
        """
        groups: Dict[str, List[CacheEntry]] = {}
        p = Path(self.cache_dir)
        if not p.exists():
            return groups
        for fn in sorted(p.iterdir()):
            try:
                if skip_inspect:
                    entry = CacheEntry.from_path_shallow(fn)
                else:
                    entry = CacheEntry.from_path(fn)
            except Exception:
                # skip unreadable files
                continue
            groups.setdefault(entry.key, []).append(entry)
        return groups

    def save_npz(self, fn: Union[str, os.PathLike], data: Dict[str, Any]) -> None:
        """Save a compressed npz file ensuring parent directory exists."""
        import numpy as _np

        p = Path(fn)
        p.parent.mkdir(parents=True, exist_ok=True)
        _np.savez_compressed(p, **data)

    def resolve_latest_paths(
        self, keys: Optional[List[str]] = None, prefer_depth_ei: bool = False
    ) -> Dict[str, Optional[str]]:
        # use entries without inspecting NPZ contents for speed
        groups = self.select_latest_cache_entries(skip_inspect=True)
        resolved: Dict[str, Optional[str]] = {}

        keys = keys or []
        for key in keys:
            base = key.split("_")[0]
            candidates = groups.get(base)
            if not candidates:
                resolved[key] = None
                continue

            # candidates are CacheEntry objects
            if key.endswith("_depth"):
                pick = next(
                    (f for f in reversed(candidates) if "_depth" in f.path.name),
                    candidates[-1],
                )
            elif key.endswith("_time"):
                pick = next(
                    (f for f in reversed(candidates) if "_time" in f.path.name),
                    candidates[-1],
                )
            else:
                pick = candidates[-1]

            resolved[key] = str(pick.path)

        # Do not synthesize implicit keys; callers should request explicit
        # filenames such as 'avo_time_<hash>.npz' or 'avo_depth_<hash>.npz'.

        return resolved

    def identify_old_cache_files(self) -> List[str]:
        """Identify old cache files.

        Returns
        -------
        List[str]
            Paths to old cache files.
        """
        return self._identifier.find_old_cache_files()

    def cleanup_old_cache(self, dry_run: bool = False) -> tuple[int, float]:
        """Remove old cache files.

        Parameters
        ----------
        dry_run : bool
            If True, report what would be removed without deleting.

        Returns
        -------
        tuple[int, float]
            (removed_count, total_size_mb)
        """
        old_files = self.identify_old_cache_files()
        if not old_files:
            self.logger.info("✓ No old cache files found in '%s'", self.cache_dir)
        return self._cleanup.cleanup_files(old_files, dry_run=dry_run)

    def run(self, dry_run: bool = False, verbose: bool = False) -> tuple[int, float]:
        """Programmatic entrypoint for cache cleanup.

        Keeps the same behavior as the previous CLI helper. Returns a tuple
        (removed_count, total_size_mb).
        """
        # Configure minimal logging if requested
        if verbose:
            try:
                import logging as _logging

                _logging.basicConfig(
                    level=_logging.DEBUG, format="[%(levelname)s] %(message)s"
                )
            except Exception:
                pass

        self.logger.info("%s", "=" * 70)
        self.logger.info("CACHE CLEANUP UTILITY")
        self.logger.info("%s", "=" * 70)
        self.logger.info("Cache directory: %s", self.cache_dir)
        self.logger.info("Mode: %s", "DRY RUN" if dry_run else "DELETE")
        self.logger.info("%s", "=" * 70)

        removed, size_mb = self.cleanup_old_cache(dry_run)

        if not dry_run and removed > 0:
            self.logger.info("%s", "\n" + "=" * 70)
            self.logger.info("CLEANUP COMPLETE")
            self.logger.info("%s", "=" * 70)

        return removed, size_mb


# Default, module-level convenience singleton for simple scripts and callers
DEFAULT_CACHE_DIR = ".cache"


# Default cache_manager proxy
cache_manager = LazyObjectProxy(lambda: CacheManager(cache_dir=DEFAULT_CACHE_DIR))


def cache_for_dir(cache_dir: str | None) -> CacheManager:
    """Return a CacheManager instance for `cache_dir`.

    If the requested directory matches the module default, returns the
    shared `cache_manager` singleton. Otherwise creates a lightweight
    temporary `CacheManager` for that directory.

    Parameters
    ----------
    cache_dir : str | None
        Cache directory path, or None for default.

    Returns
    -------
    CacheManager
        Cache manager instance (shared or temporary).
    """
    if cache_dir is None or cache_dir == DEFAULT_CACHE_DIR:
        return cache_manager
    return CacheManager(cache_dir=str(cache_dir))


def get_default_cache(cache_dir: str | None = None) -> CacheManager:
    """Return a CacheManager instance.

    When `cache_dir` is None, returns the shared module-level proxy.
    Otherwise returns a CacheManager instance configured for that directory.

    Parameters
    ----------
    cache_dir : str | None
        Cache directory path, or None for default.

    Returns
    -------
    CacheManager
        Cache manager instance (shared or temporary).
    """
    if cache_dir is None or cache_dir == DEFAULT_CACHE_DIR:
        return cache_manager
    return CacheManager(cache_dir=str(cache_dir))


__all__ = [
    "CacheEntry",
    "CacheFileIdentifier",
    "CacheFileCleanup",
    "CacheManager",
    "cache_for_dir",
    "get_default_cache",
    "DEFAULT_CACHE_DIR",
]
