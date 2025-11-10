"""Storage layer for cache implementations.

This module provides clean storage abstractions for both disk and in-memory caching.
Implements the CacheStore interface with concrete implementations.

Design:
- DiskStore: Persistent storage using NPZ format
- MemoryStore: Lightweight in-memory storage for testing/development
- Both implement the CacheStore interface for interchangeability
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, TypeVar
import numpy as np

from src.io.backends import CacheStore

T = TypeVar("T")  # Type variable for generic cached values

__all__ = ["DiskStore", "MemoryStore"]

logger = logging.getLogger(__name__)


def _hash_for_obj(obj: dict[str, str | int | float | bool] | bytes | bytearray) -> str:
    """Create a SHA1 hex digest for JSON-serializable objects or raw bytes."""
    if isinstance(obj, (bytes, bytearray)):
        data = bytes(obj)
    else:
        try:
            data = json.dumps(obj, sort_keys=True, default=str).encode("utf8")
        except Exception:
            data = str(obj).encode("utf8")
    return hashlib.sha1(data).hexdigest()


class DiskStore(CacheStore[dict[str, str | int | float | bool] | bytes]):
    """Persistent disk-based cache storage using NPZ format.

    Stores serialized objects as compressed NPZ files with content-addressable
    naming based on SHA1 hashes. Supports TTL and size-based pruning.

    Attributes
    ----------
    cache_dir : Path
        Directory for storing cache files.
    logger : logging.Logger
        Logger instance for debug/info messages.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        logger_obj: Optional[logging.Logger] = None,
    ):
        """Initialize disk cache store.

        Parameters
        ----------
        cache_dir : str | Path
            Directory for cache files.
        logger_obj : Optional[logging.Logger]
            Logger instance.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger_obj or logger

    def make_key(self, prefix: str, meta: dict[str, str | int | float | bool]) -> str:
        """Create a cache key from prefix and metadata.

        Parameters
        ----------
        prefix : str
            Key prefix (e.g., 'avo', 'rockphysics').
        meta : dict[str, str | int | float | bool]
            Metadata dictionary to hash for uniqueness.

        Returns
        -------
        str
            Cache key in format '{prefix}_{hash}'.
        """
        h = _hash_for_obj(meta)
        return f"{prefix}_{h}"

    def get_path_for_key(self, key: str) -> Optional[Path]:
        """Find cache file path for a given key.

        Parameters
        ----------
        key : str
            Cache key prefix to search for.

        Returns
        -------
        Optional[Path]
            Path to cache file if found, None otherwise.
        """
        if not self.cache_dir.exists():
            return None

        for p in sorted(self.cache_dir.iterdir()):
            if p.name.startswith(key) and p.suffix == ".npz":
                return p

        return None

    def get(self, key: str) -> Optional[dict[str, str | int | float | bool] | bytes]:
        """Retrieve item from cache.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        Optional[dict[str, str | int | float | bool] | bytes]
            Cached value or None if not found.
        """
        path = self.get_path_for_key(key)
        if not path or not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=True) as npz:
                return dict(npz)
        except Exception as e:
            self.logger.debug(f"Error loading cache from {path}: {e}")
            return None

    def set(self, key: str, value: dict[str, str | int | float | bool] | bytes) -> None:
        """Store item in cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : dict[str, str | int | float | bool] | bytes
            Value to cache.
        """
        if not isinstance(value, dict):
            raise ValueError("DiskStore requires dict-like values")
        short = key.split("_")[-1][:20]
        path = self.cache_dir / f"{key}_{short}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(path, **value)
            self.logger.debug(f"Saved cache to {path}")
        except Exception as e:
            self.logger.debug(f"Error storing key '{key}': {e}")

    def has(self, key: str) -> bool:
        """Check if key exists in cache.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if key exists, False otherwise.
        """
        try:
            path = self.get_path_for_key(key)
            return path is not None and path.exists()
        except Exception as e:
            self.logger.debug(f"Error checking key '{key}': {e}")
            return False

    def total_size_bytes(self) -> int:
        """Get total size of all cache files.

        Returns
        -------
        int
            Total size in bytes.
        """
        if not self.cache_dir.exists():
            return 0

        try:
            return sum(f.stat().st_size for f in self.cache_dir.glob("*.npz"))
        except Exception as e:
            self.logger.debug(f"Error calculating cache size: {e}")
            return 0

    def entry_count(self) -> int:
        """Get number of cache entries.

        Returns
        -------
        int
            Number of NPZ files in cache directory.
        """
        if not self.cache_dir.exists():
            return 0

        try:
            return len(list(self.cache_dir.glob("*.npz")))
        except Exception as e:
            self.logger.debug(f"Error counting cache entries: {e}")
            return 0

    def list_entries(self) -> list[dict[str, str | int | float]]:
        """List metadata for all cache entries.

        Returns
        -------
        list[dict[str, str | int | float]]
            List of dicts with 'name', 'size', 'mtime' keys.
        """
        if not self.cache_dir.exists():
            return []

        entries: list[dict[str, str | int | float]] = []
        try:
            for path in sorted(self.cache_dir.glob("*.npz")):
                try:
                    stat = path.stat()
                    entry: dict[str, str | int | float] = {
                        "name": path.name,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    }
                    entries.append(entry)
                except Exception as e:
                    self.logger.debug(f"Error stat'ing {path}: {e}")
        except Exception as e:
            self.logger.debug(f"Error listing cache entries: {e}")

        return entries

    def delete(self, key: str) -> bool:
        """Delete cache entry by key.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        path = self.get_path_for_key(key)
        if not path or not path.exists():
            return False
        try:
            path.unlink()
            self.logger.debug(f"Deleted cache file: {path.name}")
            return True
        except Exception as e:
            self.logger.debug(f"Error deleting cache file {path}: {e}")
            return 0

    def clear(self) -> None:
        """Clear all cache entries."""
        if not self.cache_dir.exists():
            return
        try:
            for path in self.cache_dir.glob("*.npz"):
                try:
                    path.unlink()
                except Exception as e:
                    self.logger.debug(f"Error deleting {path}: {e}")
            self.logger.debug(f"Cleared cache directory: {self.cache_dir}")
        except Exception as e:
            self.logger.debug(f"Error clearing cache directory: {e}")


class MemoryStore(CacheStore[dict[str, str | int | float | bool] | bytes]):
    """In-memory cache storage for testing and lightweight use.

    Stores objects directly in a dictionary with no persistence.
    Useful for unit tests or temporary caching without disk I/O.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance.
    _store : dict[str, dict[str, str | int | float | bool] | bytes]
        Internal storage dictionary.
    """

    def __init__(self, logger_obj: Optional[logging.Logger] = None):
        """Initialize in-memory cache store.

        Parameters
        ----------
        logger_obj : Optional[logging.Logger]
            Logger instance.
        """
        self.logger = logger_obj or logger
        self._store: dict[str, dict[str, str | int | float | bool] | bytes] = {}

    def _get_impl(
        self, key: str
    ) -> Optional[dict[str, str | int | float | bool] | bytes]:
        """Retrieve object from memory.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        Optional[dict[str, str | int | float | bool] | bytes]
            Cached value or None if not found.
        """
        result = self._store.get(key)
        if result is not None:
            self.logger.debug(f"Memory cache hit: {key}")
        return result

    def get(self, key: str) -> Optional[dict[str, str | int | float | bool] | bytes]:
        """Retrieve item from cache.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        Optional[dict[str, str | int | float | bool] | bytes]
            Cached value or None if not found.
        """
        try:
            return self._get_impl(key)
        except Exception as e:
            self.logger.debug(f"Error retrieving key '{key}': {e}")
            return None

    def _set_impl(
        self, key: str, value: dict[str, str | int | float | bool] | bytes
    ) -> None:
        """Store object in memory.

        Parameters
        ----------
        key : str
            Cache key.
        value : dict[str, str | int | float | bool] | bytes
            Value to cache.
        """
        self._store[key] = value
        self.logger.debug(f"Memory cache set: {key}")

    def set(self, key: str, value: dict[str, str | int | float | bool] | bytes) -> None:
        """Store item in cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : dict[str, str | int | float | bool] | bytes
            Value to cache.
        """
        try:
            self._set_impl(key, value)
        except Exception as e:
            self.logger.debug(f"Error storing key '{key}': {e}")

    def _has_impl(self, key: str) -> bool:
        """Check if key exists in memory.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if key exists.
        """
        return key in self._store

    def has(self, key: str) -> bool:
        """Check if key exists in cache.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if key exists, False otherwise.
        """
        try:
            return self._has_impl(key)
        except Exception as e:
            self.logger.debug(f"Error checking key '{key}': {e}")
            return False

    def _delete_impl(self, key: str) -> bool:
        """Delete object from memory.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        if key in self._store:
            del self._store[key]
            self.logger.debug(f"Memory cache delete: {key}")
            return True
        return False

    def delete(self, key: str) -> bool:
        """Delete cache entry.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        try:
            return self._delete_impl(key)
        except Exception as e:
            self.logger.debug(f"Error deleting key '{key}': {e}")
            return False

    def _clear_impl(self) -> None:
        """Clear all entries from memory."""
        self._store.clear()
        self.logger.debug("Memory cache cleared")

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self._clear_impl()
        except Exception as e:
            self.logger.debug(f"Error clearing cache: {e}")

    def size(self) -> int:
        """Get number of entries in memory cache.

        Returns
        -------
        int
            Number of cached items.
        """
        return len(self._store)

    def keys(self) -> list[str]:
        """Get all cache keys.

        Returns
        -------
        list[str]
            List of cache keys.
        """
        return list(self._store.keys())
