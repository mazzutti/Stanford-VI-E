"""Lightweight in-memory LRU cache utilities.

Provides a simple LRUCache and a sharded variant for reduced contention
under concurrency. These helpers are intentionally small and have few
dependencies so they can be used in performance-sensitive code paths.
"""

import threading
from collections import OrderedDict
from typing import Generic, TypeVar

T = TypeVar("T")

class LRUCache(Generic[T]):
    """A tiny LRU cache wrapper around OrderedDict.

    Stores values under string keys. When maxsize is 0 or None, caching is disabled.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize = int(maxsize)
        self._data: OrderedDict[str, T] = OrderedDict()
        # re-entrant lock to allow thread-safe get/set/clear operations
        self._lock = threading.RLock()

    def get(self, key: str) -> T | None:
        """Return value for `key` or `None` if not present.

        Marks the key as recently used when present.
        """
        with self._lock:
            v = self._data.get(key)
            if v is not None:
                # mark recently used
                try:
                    self._data.move_to_end(key)
                except (KeyError, AttributeError, TypeError):
                    pass
            return v

    def set(self, key: str, value: T) -> None:
        """Store `value` under `key`, evicting oldest items when full."""
        with self._lock:
            self._data[key] = value
            try:
                while self.maxsize > 0 and len(self._data) > self.maxsize:
                    self._data.popitem(last=False)
            except (KeyError, IndexError):
                pass

    def keys(self) -> list[str]:
        """Return current cache keys in LRU order (least-recently-used first)."""
        with self._lock:
            return list(self._data.keys())

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._data.clear()

    def info(self) -> dict[str, int]:
        """Return `maxsize` and current size of the cache."""
        with self._lock:
            return {"maxsize": self.maxsize, "currsize": len(self._data)}

class ShardedLRUCache(Generic[T]):
    """A simple sharded LRU cache that composes multiple LRUCache shards.

    Each shard is an independent LRUCache. Keys are assigned to shards by
    hashing the key. This reduces contention under high concurrency.
    """

    def __init__(self, maxsize: int = 0, shards: int = 1) -> None:
        self.shards = max(1, int(shards))
        # Distribute maxsize evenly across shards (floor division)
        per_shard = max(0, int(maxsize) // self.shards) if maxsize > 0 else 0
        # If maxsize > 0 but per_shard == 0, give each shard at least 1
        if maxsize > 0 and per_shard == 0:
            per_shard = 1
        self._shards: list[LRUCache[T]] = [
            LRUCache[T](per_shard) for _ in range(self.shards)
        ]

    def _pick(self, key: str) -> LRUCache[T]:
        """Select the shard responsible for `key`."""
        return self._shards[hash(key) % self.shards]

    def get(self, key: str) -> T | None:
        """Return value for `key` from the appropriate shard."""
        return self._pick(key).get(key)

    def set(self, key: str, value: T) -> None:
        """Store `value` under `key` in the appropriate shard."""
        self._pick(key).set(key, value)

    def keys(self) -> list[str]:
        """Aggregate keys from all shards (order is shard-local LRU order)."""
        ks: list[str] = []
        for s in self._shards:
            ks.extend(s.keys())
        return ks

    def clear(self) -> None:
        """Clear all shards' caches."""
        for s in self._shards:
            s.clear()

    def info(self) -> dict[str, int]:
        """Return aggregated `maxsize` and `currsize` across shards."""
        total = sum(s.info().get("currsize", 0) for s in self._shards)
        # report maxsize as sum of shard maxsizes
        max_total = sum(s.info().get("maxsize", 0) for s in self._shards)
        return {"maxsize": max_total, "currsize": total}

# LRU utilities are intentionally small and dependency-free; they are
# designed for embedded use in performance-sensitive code paths.
