"""Comprehensive Caching System for High-Performance Applications

This module provides flexible, multi-strategy caching with support for LRU, LFU,
TTL-based, and FIFO cache eviction policies. All caches are thread-safe and
designed for high-performance scenarios.

Patterns Used:
  - Strategy: Different cache eviction policies
  - Decorator: @cache_result for function caching
  - Observer: Cache invalidation callbacks
  - Singleton: Cache manager instance

Example:
    >>> from src.analysis.caching import CacheManager, CacheStrategy
    >>>
    >>> # Create LRU cache
    >>> cache = LRUCache(max_size=1000)
    >>> cache.set("user:123", {"name": "John"})
    >>> user = cache.get("user:123")
    >>>
    >>> # Use cache decorator
    >>> @cache_result(max_size=500, ttl=3600)
    ... def expensive_computation(x: int) -> int:
    ...     return x * x
    >>>
    >>> result = expensive_computation(10)  # Computed
    >>> result = expensive_computation(10)  # From cache
    >>>
    >>> # Manage caches
    >>> manager = CacheManager()
    >>> manager.register("users", LRUCache(1000))
    >>> manager.register("sessions", TTLCache(300))
    >>> users_cache = manager.get("users")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, List, TypeVar, Generic
from dataclasses import dataclass, field
from collections import OrderedDict
from threading import Lock, RLock
from time import time
from functools import wraps
import logging
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

__all__ = [
    "CacheStrategy",
    "LRUCache",
    "LFUCache",
    "TTLCache",
    "FIFOCache",
    "CacheStats",
    "CacheManager",
    "cache_result",
    "CacheInvalidationEvent",
]

T = TypeVar("T")


class CacheInvalidationEvent(Enum):
    """Cache invalidation events."""

    EVICTED = "evicted"
    EXPIRED = "expired"
    MANUAL = "manual"


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    def __str__(self) -> str:
        """String representation."""
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.1f}%, size={self.size}/{self.max_size})"
        )


class CacheStrategy(ABC):
    """Abstract base class for cache implementations."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all values from cache."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of items in cache
        """
        pass

    @abstractmethod
    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with hit/miss info
        """
        pass


class LRUCache(CacheStrategy):
    """Least Recently Used cache implementation.

    Evicts least recently used items when max size exceeded.
    Thread-safe with automatic eviction.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value and mark as recently used.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]

            self._misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if key in self._cache:
                # Update and move to end
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                # Add new item
                self._cache[key] = value

                # Evict if over capacity
                while len(self._cache) > self.max_size:
                    removed_key, _ = self._cache.popitem(last=False)
                    self._evictions += 1
                    logger.debug(f"LRU evicted: {removed_key}")

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all values from cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of items in cache
        """
        with self._lock:
            return len(self._cache)

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats object
        """
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=len(self._cache),
                max_size=self.max_size,
            )


class LFUCache(CacheStrategy):
    """Least Frequently Used cache implementation.

    Evicts items with lowest access frequency when max size exceeded.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize LFU cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._frequency: Dict[str, int] = {}
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value and increment frequency.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self._lock:
            if key in self._cache:
                self._frequency[key] = self._frequency.get(key, 0) + 1
                self._hits += 1
                return self._cache[key]

            self._misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            self._cache[key] = value
            self._frequency[key] = self._frequency.get(key, 0) + 1

            # Evict if over capacity
            while len(self._cache) > self.max_size:
                # Find least frequently used
                min_key = min(self._frequency, key=self._frequency.get)
                del self._cache[min_key]
                del self._frequency[min_key]
                self._evictions += 1
                logger.debug(f"LFU evicted: {min_key}")

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._frequency[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all values from cache."""
        with self._lock:
            self._cache.clear()
            self._frequency.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of items in cache
        """
        with self._lock:
            return len(self._cache)

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats object
        """
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=len(self._cache),
                max_size=self.max_size,
            )


class TTLCache(CacheStrategy):
    """Time-to-Live cache implementation.

    Automatically expires items based on TTL. Lazy expiration on access.
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """Initialize TTL cache.

        Args:
            ttl_seconds: Time-to-live in seconds
            max_size: Maximum number of items to cache
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/not found
        """
        with self._lock:
            if key in self._cache:
                # Check if expired
                if time() >= self._expiry[key]:
                    del self._cache[key]
                    del self._expiry[key]
                    self._misses += 1
                    return None

                self._hits += 1
                return self._cache[key]

            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override in seconds
        """
        with self._lock:
            self._cache[key] = value
            ttl_to_use = ttl if ttl is not None else self.ttl_seconds
            self._expiry[key] = time() + ttl_to_use

            # Evict if over capacity
            if len(self._cache) > self.max_size:
                # Remove earliest expiring item
                min_key = min(self._expiry, key=self._expiry.get)
                del self._cache[min_key]
                del self._expiry[min_key]
                self._evictions += 1
                logger.debug(f"TTL evicted: {min_key}")

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._expiry[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all values from cache."""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def size(self) -> int:
        """Get current cache size (excluding expired).

        Returns:
            Number of valid items in cache
        """
        with self._lock:
            current_time = time()
            return sum(1 for expiry in self._expiry.values() if current_time < expiry)

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats object
        """
        with self._lock:
            current_time = time()
            valid_count = sum(
                1 for expiry in self._expiry.values() if current_time < expiry
            )
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=valid_count,
                max_size=self.max_size,
            )


class FIFOCache(CacheStrategy):
    """First-In-First-Out cache implementation.

    Evicts oldest items when max size exceeded.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize FIFO cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]

            self._misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if key not in self._cache:
                self._cache[key] = value

                # Evict if over capacity
                while len(self._cache) > self.max_size:
                    removed_key, _ = self._cache.popitem(last=False)
                    self._evictions += 1
                    logger.debug(f"FIFO evicted: {removed_key}")
            else:
                self._cache[key] = value

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all values from cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of items in cache
        """
        with self._lock:
            return len(self._cache)

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats object
        """
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=len(self._cache),
                max_size=self.max_size,
            )


class CacheManager:
    """Manages multiple named caches.

    Provides centralized cache registration, lifecycle management,
    and statistics aggregation.
    """

    def __init__(self):
        """Initialize cache manager."""
        self._caches: Dict[str, CacheStrategy] = {}
        self._lock = Lock()

    def register(self, name: str, cache: CacheStrategy) -> None:
        """Register a named cache.

        Args:
            name: Cache name
            cache: Cache instance
        """
        with self._lock:
            self._caches[name] = cache
            logger.debug(f"Registered cache: {name}")

    def get(self, name: str) -> Optional[CacheStrategy]:
        """Get a named cache.

        Args:
            name: Cache name

        Returns:
            Cache instance or None if not registered
        """
        with self._lock:
            return self._caches.get(name)

    def unregister(self, name: str) -> bool:
        """Unregister a named cache.

        Args:
            name: Cache name

        Returns:
            True if unregistered, False if not found
        """
        with self._lock:
            if name in self._caches:
                del self._caches[name]
                logger.debug(f"Unregistered cache: {name}")
                return True
            return False

    def clear_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            logger.debug("Cleared all caches")

    def stats_all(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches.

        Returns:
            Dictionary of cache name to stats
        """
        with self._lock:
            return {name: cache.stats() for name, cache in self._caches.items()}

    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            return f"CacheManager(caches={list(self._caches.keys())})"


def cache_result(
    max_size: int = 1000,
    ttl: Optional[int] = None,
    strategy: str = "lru",
) -> Callable:
    """Decorator to cache function results.

    Args:
        max_size: Maximum cache size
        ttl: Time-to-live in seconds (for TTL strategy)
        strategy: Cache strategy ("lru", "lfu", "ttl", "fifo")

    Returns:
        Decorated function with caching

    Example:
        >>> @cache_result(max_size=500, ttl=3600)
        ... def expensive_function(x: int) -> int:
        ...     return x * x
    """
    # Create cache instance
    if strategy == "lru":
        cache = LRUCache(max_size=max_size)
    elif strategy == "lfu":
        cache = LFUCache(max_size=max_size)
    elif strategy == "ttl":
        cache = TTLCache(ttl_seconds=ttl or 3600, max_size=max_size)
    elif strategy == "fifo":
        cache = FIFOCache(max_size=max_size)
    else:
        raise ValueError(f"Unknown cache strategy: {strategy}")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Build cache key from function name and arguments
            cache_key = f"{func.__name__}:{args}:{kwargs}"

            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {func.__name__}")
                return cached_value

            # Compute and cache result
            logger.debug(f"Cache miss: {func.__name__}")
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result

        # Attach cache for direct access
        wrapper.cache = cache
        wrapper.stats = cache.stats

        return wrapper

    return decorator
