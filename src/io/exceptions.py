"""Custom exceptions for I/O and cache operations.

This module defines domain-specific exceptions that provide clearer error
messages and enable targeted exception handling in client code.
"""


class IOBaseError(Exception):
    """Base exception for all I/O-related errors in the src.io module."""


class CacheError(IOBaseError):
    """Base exception for cache-related errors."""


class CacheValidationError(CacheError):
    """Raised when cache data fails validation or integrity checks."""


class CachePruneError(CacheError):
    """Raised when cache pruning operations fail."""


class DataLoaderError(IOBaseError):
    """Base exception for data loading errors."""


class GSLibError(DataLoaderError):
    """Raised when GSLIB file reading fails."""


class GridError(IOBaseError):
    """Raised when grid specification is invalid."""


class FileLocatorError(DataLoaderError):
    """Raised when required data files cannot be located."""


__all__ = [
    "IOBaseError",
    "CacheError",
    "CacheValidationError",
    "CachePruneError",
    "DataLoaderError",
    "GSLibError",
    "GridError",
    "FileLocatorError",
]
