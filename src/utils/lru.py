from collections import OrderedDict
from typing import List, Optional, Generic, TypeVar
import threading


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

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            v = self._data.get(key)
            if v is not None:
                # mark recently used
                try:
                    self._data.move_to_end(key)
                except Exception:
                    pass
            return v

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._data[key] = value
            try:
                while self.maxsize > 0 and len(self._data) > self.maxsize:
                    self._data.popitem(last=False)
            except Exception:
                pass

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def info(self) -> dict[str, int]:
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
        self._shards: List[LRUCache[T]] = [
            LRUCache[T](per_shard) for _ in range(self.shards)
        ]

    def _pick(self, key: str) -> LRUCache[T]:
        return self._shards[hash(key) % self.shards]

    def get(self, key: str) -> Optional[T]:
        return self._pick(key).get(key)

    def set(self, key: str, value: T) -> None:
        self._pick(key).set(key, value)

    def keys(self) -> List[str]:
        # aggregate keys from all shards; order is shard-local LRU order
        ks: List[str] = []
        for s in self._shards:
            ks.extend(s.keys())
        return ks

    def clear(self) -> None:
        for s in self._shards:
            s.clear()

    def info(self) -> dict[str, int]:
        total = sum(s.info().get("currsize", 0) for s in self._shards)
        # report maxsize as sum of shard maxsizes
        max_total = sum(s.info().get("maxsize", 0) for s in self._shards)
        return {"maxsize": max_total, "currsize": total}
