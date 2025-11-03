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


class CacheManager:
    """Object-oriented wrapper around cache utilities.

    Provides the same functionality as the old module-level helpers but in a
    cohesive class that can be instantiated with a default cache directory and
    injected logger (useful for testing).
    """

    def __init__(
        self, cache_dir: str = ".cache", logger: Optional[logging.Logger] = None
    ):
        self.cache_dir = cache_dir
        self.logger = logger or logging.getLogger(__name__)

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
        p = Path(self.cache_dir)
        if not p.exists():
            return []

        all_npz = list(p.glob("*.npz"))
        old_files: List[str] = []

        for file_path in all_npz:
            filename = file_path.name

            if filename.startswith("avo_") and not (
                "_time_" in filename or "_depth_" in filename
            ):
                old_files.append(str(file_path))

        return old_files

    def cleanup_old_cache(self, dry_run: bool = False) -> tuple[int, float]:
        """Remove old cache files identified by `identify_old_cache_files`.

        Returns a tuple (removed_count, total_size_mb).
        """
        old_files = self.identify_old_cache_files()
        if not old_files:
            self.logger.info("✓ No old cache files found in '%s'", self.cache_dir)
            return 0, 0.0

        total_size_bytes = 0
        for file_path in old_files:
            total_size_bytes += os.path.getsize(file_path)

        total_size_mb = total_size_bytes / (1024**2)

        if dry_run:
            self.logger.info(
                "DRY RUN: Would remove %d files (%.1f MB)",
                len(old_files),
                total_size_mb,
            )
            return 0, 0.0

        removed_count = 0
        for file_path in old_files:
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as e:
                self.logger.warning("Error removing %s: %s", file_path, e)

        self.logger.info(
            "✓ Removed %d/%d files (%.1f MB freed)",
            removed_count,
            len(old_files),
            total_size_mb,
        )

        return removed_count, total_size_mb

    def main(self, dry_run: bool = False, verbose: bool = False) -> tuple[int, float]:
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


__all__ = ["CacheEntry", "CacheManager"]

# Default, module-level convenience singleton for simple scripts and callers
DEFAULT_CACHE_DIR = ".cache"


# Default cache_manager proxy
cache_manager = LazyObjectProxy(lambda: CacheManager(cache_dir=DEFAULT_CACHE_DIR))


def cache_for_dir(cache_dir: str | None):
    """Return a CacheManager instance for `cache_dir`.

    If the requested directory matches the module default, returns the
    shared `cache_manager` singleton. Otherwise creates a lightweight
    temporary `CacheManager` for that directory.
    """
    if cache_dir is None or cache_dir == DEFAULT_CACHE_DIR:
        return cache_manager
    return CacheManager(cache_dir=str(cache_dir))


def _impl_cache_for_dir(cache_dir: str | None):
    """Canonical implementation for cache_for_dir providing a single entrypoint.

    Keeps the lazy `cache_manager` behaviour for the default directory and
    returns a temporary `CacheManager` for custom directories.
    """
    # Backwards-compatible canonical implementation kept for tests; prefer
    # calling `cache_for_dir(...)` above which includes the same logic.
    if cache_dir is None or cache_dir == DEFAULT_CACHE_DIR:
        return cache_manager
    return CacheManager(cache_dir=str(cache_dir))

    # Note: module-level helpers have been replaced by `CacheManager` instances
    # or the `cache_for_dir(...)` helper which returns either the shared proxy
    # or a temporary instance.


__all__.extend(["CacheEntry", "CacheManager", "cache_for_dir", "DEFAULT_CACHE_DIR"])


def get_default_cache(cache_dir: str | None = None):
    """Return the module default `cache_manager` when `cache_dir` is None,
    otherwise return a `CacheManager` instance configured for `cache_dir`.

    This mirrors `get_default_disk_cache` in `src.io.disk_cache` and gives
    callers a single helper to obtain either the shared lazy singleton or a
    temporary instance for custom directories.
    """
    if cache_dir is None or cache_dir == DEFAULT_CACHE_DIR:
        return cache_manager
    return CacheManager(cache_dir=str(cache_dir))


def _impl_get_default_cache(cache_dir: str | None = None):
    if cache_dir is None or cache_dir == DEFAULT_CACHE_DIR:
        return cache_manager
    return CacheManager(cache_dir=str(cache_dir))


__all__.append("get_default_cache")


def get_cache_manager(cache_dir: str | None = None):
    """Return a CacheManager instance for `cache_dir`.

    When `cache_dir` is None the shared module-level proxy is returned.
    """
    # Return shared proxy when using the default cache dir, otherwise return a
    # temporary CacheManager instance for the provided directory.
    if cache_dir is None or cache_dir == DEFAULT_CACHE_DIR:
        return cache_manager
    return CacheManager(cache_dir=str(cache_dir))


def _impl_get_cache_manager(cache_dir: str | None = None):
    return _impl_get_default_cache(cache_dir)


__all__.append("get_cache_manager")
