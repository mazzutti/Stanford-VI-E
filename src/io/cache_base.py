"""Base cache class with common operations.

This module provides BaseCache, an abstract class that unifies common cache
operations (get/set/has/delete/clear) across different storage backends.

Design follows the Template Method pattern:
- Subclasses implement _get_impl, _set_impl, _has_impl, _delete_impl, _clear_impl
- Common logic for key handling and error recovery is in BaseCache
- Promotes code reuse and consistent behavior across cache types
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

__all__ = ["BaseCache"]

logger = logging.getLogger(__name__)


class BaseCache(ABC):
    """Abstract base class for cache storage implementations.

    Implements common cache operations using Template Method pattern.
    Subclasses must implement:
        - _get_impl(key: str) -> Optional[Any]
        - _set_impl(key: str, value: Any) -> None
        - _has_impl(key: str) -> bool
        - _delete_impl(key: str) -> bool
        - _clear_impl() -> None

    This design:
    - Centralizes logging and error handling
    - Ensures consistent behavior across backends
    - Reduces code duplication between DiskCache and CacheManager
    - Makes it easy to add new storage backends
    """

    def __init__(self, logger_obj: Optional[logging.Logger] = None):
        """Initialize base cache.

        Parameters
        ----------
        logger_obj : Optional[logging.Logger]
            Logger instance. Defaults to module logger.
        """
        self.logger = logger_obj or logger

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
        try:
            return self._get_impl(key)
        except Exception as e:
            self.logger.debug(f"Error retrieving key '{key}': {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        """Store item in cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache.
        """
        try:
            self._set_impl(key, value)
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
            True if key exists and is valid, False otherwise.
        """
        try:
            return self._has_impl(key)
        except Exception as e:
            self.logger.debug(f"Error checking key '{key}': {e}")
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
            True if deleted, False if not found or error occurred.
        """
        try:
            return self._delete_impl(key)
        except Exception as e:
            self.logger.debug(f"Error deleting key '{key}': {e}")
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self._clear_impl()
        except Exception as e:
            self.logger.debug(f"Error clearing cache: {e}")

    @abstractmethod
    def _get_impl(self, key: str) -> Optional[Any]:
        """Implementation of get operation. Must be implemented by subclass.

        Parameters
        ----------
        key : str
            Cache key or prefix.

        Returns
        -------
        Optional[Any]
            Cached value or None if not found.
        """
        pass

    @abstractmethod
    def _set_impl(self, key: str, value: Any) -> None:
        """Implementation of set operation. Must be implemented by subclass.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache.
        """
        pass

    @abstractmethod
    def _has_impl(self, key: str) -> bool:
        """Implementation of has operation. Must be implemented by subclass.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if key exists, False otherwise.
        """
        pass

    @abstractmethod
    def _delete_impl(self, key: str) -> bool:
        """Implementation of delete operation. Must be implemented by subclass.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def _clear_impl(self) -> None:
        """Implementation of clear operation. Must be implemented by subclass."""
        pass
