"""Custom exceptions for I/O and cache operations.

This module defines domain-specific exceptions that provide clearer error
messages and enable targeted exception handling in client code.
"""


class IOError(Exception):
    """Base exception for all I/O-related errors in the src.io module."""

    pass


class CacheError(IOError):
    """Base exception for cache-related errors."""

    pass


class CacheValidationError(CacheError):
    """Raised when cache data fails validation or integrity checks."""

    pass


class CachePruneError(CacheError):
    """Raised when cache pruning operations fail."""

    pass


class DataLoaderError(IOError):
    """Base exception for data loading errors."""

    pass


class GSLibError(DataLoaderError):
    """Raised when GSLIB file reading fails."""

    pass


class GridError(IOError):
    """Raised when grid specification is invalid."""

    pass


class FileLocatorError(DataLoaderError):
    """Raised when required data files cannot be located."""

    pass


__all__ = [
    "IOError",
    "CacheError",
    "CacheValidationError",
    "CachePruneError",
    "DataLoaderError",
    "GSLibError",
    "GridError",
    "FileLocatorError",
]
