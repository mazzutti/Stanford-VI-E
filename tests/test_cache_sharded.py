# mypy: ignore-errors
import threading
import numpy as np

from src.analysis.cache import CacheLoaderFactory, CacheLoader
from src.utils.lru import ShardedLRUCache


def test_factory_creates_sharded_cache():
    loader = CacheLoaderFactory.create_default(cache_size=20, shards=4)
    assert isinstance(loader, CacheLoader)
    assert loader.cache_enabled
    info = loader.cache_info()
    assert info["maxsize"] == 20


def test_inject_custom_cache():
    # inject a tiny test cache that records set/get calls
    events = []

    class DummyCache:
        def __init__(self):
            self._storage = {}

        def get(self, k):
            events.append(("get", k))
            return self._storage.get(k)

        def set(self, k, v):
            events.append(("set", k, v))
            self._storage[k] = v

        def keys(self):
            return list(self._storage.keys())

        def clear(self):
            self._storage.clear()

        def info(self):
            return {"maxsize": 0, "currsize": len(self._storage)}

    dummy = DummyCache()
    loader = CacheLoader(selector=None, cache=dummy)
    # simulate a cache store via loader internals
    loader._cache.set("x", np.array([1.0]))
    assert (
        "set",
        "x",
    ) in [(e[0], e[1]) for e in events]


def test_sharded_cache_concurrency_smoke():
    cache = ShardedLRUCache[int](maxsize=200, shards=8)

    def writer(start):
        for i in range(start, start + 500):
            cache.set(f"k{i}", i)

    threads = [threading.Thread(target=writer, args=(s * 500,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # basic sanity
    assert cache.info()["currsize"] <= 200


def test_sharded_cache_concurrency():
    c = ShardedLRUCache[int](maxsize=100, shards=4)

    def writer(start):
        for i in range(start, start + 200):
            c.set(f"k{i}", i)

    threads = [threading.Thread(target=writer, args=(s * 200,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(c.keys()) <= 100


def test_cacheloader_injection_and_extractor(tmp_path):
    # create a fake npz file with an array
    arr = np.arange(12.0).reshape(3, 4)
    npz_path = tmp_path / "avo_test.npz"
    np.savez(npz_path, full_stack=arr)

    # fake cache that records set/get calls
    class SimpleCache:
        def __init__(self):
            self._storage = {}

        def get(self, k):
            return self._storage.get(k)

        def set(self, k, v):
            self._storage[k] = v

        def keys(self):
            return list(self._storage.keys())

        def clear(self):
            self._storage.clear()

        def info(self):
            return {"maxsize": 0, "currsize": len(self._storage)}

    cache = SimpleCache()

    def extractor(archive):
        return np.asarray(archive["full_stack"]) * 2.0

    # Construct CacheLoader manually with injected cache
    loader = CacheLoader(
        selector=None, cache=cache, cache_size=0, archive_extractor=extractor
    )

    result = loader.load_full_stack(str(npz_path))
    assert result is not None
    # extractor doubled values
    assert np.allclose(result, arr * 2.0)
    # injected cache should be populated because the caller supplied it
    # even though cache_size==0 (injected caches are honored)
    assert cache.keys() != []
