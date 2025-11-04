"""Abstract base classes and protocols for cache backends.

This module defines the interfaces for cache implementations, enabling
easy composition and testing through dependency injection.

Key abstractions:
- FileSystemOps: Protocol for injecting different file system behaviors
- CacheStore: Interface that all cache backends must implement
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Protocol
import logging

__all__ = ["FileSystemOps", "CacheStore"]


class FileSystemOps(Protocol):
    """Protocol for file system operations used by cache implementations.

    This protocol allows injection of different file system behaviors
    (e.g., real filesystem, in-memory mock, etc.) for better testability.

    Methods
    -------
    read_file(path: Path) -> bytes
        Read file contents as bytes.
    write_file(path: Path, data: bytes) -> None
        Write file contents.
    delete_file(path: Path) -> None
        Delete a file.
    file_exists(path: Path) -> bool
        Check if file exists.
    get_file_size(path: Path) -> int
        Get file size in bytes.
    get_file_mtime(path: Path) -> float
        Get file modification time (seconds since epoch).
    list_files(directory: Path, pattern: str) -> list[Path]
        List files matching pattern in directory.
    """

    def read_file(self, path: Path) -> bytes:
        """Read file contents as bytes."""
        ...

    def write_file(self, path: Path, data: bytes) -> None:
        """Write file contents."""
        ...

    def delete_file(self, path: Path) -> None:
        """Delete a file."""
        ...

    def file_exists(self, path: Path) -> bool:
        """Check if file exists."""
        ...

    def get_file_size(self, path: Path) -> int:
        """Get file size in bytes."""
        ...

    def get_file_mtime(self, path: Path) -> float:
        """Get file modification time (seconds since epoch)."""
        ...

    def list_files(self, directory: Path, pattern: str) -> list[Path]:
        """List files matching pattern in directory."""
        ...


class CacheStore(ABC):
    """Abstract base class for cache storage implementations.

    Defines the common interface that all cache backends (in-memory,
    disk-backed, etc.) must implement for get/set/delete operations.

    This is a Protocol-like interface for all cache implementations.
    Subclasses should implement the get/set/has/delete/clear contract.

    Methods
    -------
    get(key: str) -> Optional[Any]
        Retrieve item from cache.
    set(key: str, value: Any) -> None
        Store item in cache.
    has(key: str) -> bool
        Check if key exists in cache.
    delete(key: str) -> bool
        Delete cache entry.
    clear() -> None
        Clear all cache entries.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache.

        Parameters
        ----------
        key : str
            Cache key or prefix.

        Returns
        -------
        Optional[Any]
            Cached value or None if not found or expired.
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Store item in cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache.
        """
        pass

    @abstractmethod
    def has(self, key: str) -> bool:
        """Check if key exists in cache.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if key exists and is valid, False otherwise.
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cache entry by key.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if deleted, False if not found or error.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass


class DefaultFileSystemOps:
    """Default implementation of FileSystemOps using real filesystem."""

    def read_file(self, path: Path) -> bytes:
        """Read file contents as bytes."""
        return path.read_bytes()

    def write_file(self, path: Path, data: bytes) -> None:
        """Write file contents."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def delete_file(self, path: Path) -> None:
        """Delete a file."""
        path.unlink(missing_ok=True)

    def file_exists(self, path: Path) -> bool:
        """Check if file exists."""
        return path.exists()

    def get_file_size(self, path: Path) -> int:
        """Get file size in bytes."""
        try:
            return path.stat().st_size
        except (OSError, ValueError):
            return 0

    def get_file_mtime(self, path: Path) -> float:
        """Get file modification time (seconds since epoch)."""
        try:
            return path.stat().st_mtime
        except (OSError, ValueError):
            import sys

            return sys.maxsize

    def list_files(self, directory: Path, pattern: str) -> list[Path]:
        """List files matching pattern in directory."""
        if not directory.exists():
            return []
        return list(directory.glob(pattern))


__all__ = [
    "FileSystemOps",
    "CacheStore",
    "DefaultFileSystemOps",
]
