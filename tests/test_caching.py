"""Comprehensive tests for Phase 5.1: Caching Layer

Tests cover all cache strategies, statistics, thread safety, and invalidation.
"""

import time
from threading import Thread
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from src.analysis.caching import (CacheManager, CacheStats, FIFOCache,
                                  LFUCache, LRUCache, TTLCache, cache_result)

# ============================================================================
# LRU CACHE TESTS
# ============================================================================


class TestLRUCache:
    """Tests for LRU cache implementation."""

    def test_lru_basic_set_get(self):
        """Test basic set and get operations."""
        cache = LRUCache(max_size=2)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None

    def test_lru_eviction(self):
        """Test LRU eviction when max size exceeded."""
        cache = LRUCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_lru_recently_used_not_evicted(self):
        """Test that recently used items are not evicted."""
        cache = LRUCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Access key1 (mark as recently used)
        cache.set("key3", "value3")  # Should evict key2

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    def test_lru_delete(self):
        """Test delete operation."""
        cache = LRUCache(max_size=10)

        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("key2") is False

    def test_lru_clear(self):
        """Test clear operation."""
        cache = LRUCache(max_size=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.size() == 0
        assert cache.get("key1") is None

    def test_lru_size(self):
        """Test size tracking."""
        cache = LRUCache(max_size=10)

        assert cache.size() == 0
        cache.set("key1", "value1")
        assert cache.size() == 1
        cache.set("key2", "value2")
        assert cache.size() == 2

    def test_lru_stats(self):
        """Test statistics tracking."""
        cache = LRUCache(max_size=10)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 50.0


# ============================================================================
# LFU CACHE TESTS
# ============================================================================


class TestLFUCache:
    """Tests for LFU cache implementation."""

    def test_lfu_basic_set_get(self):
        """Test basic set and get operations."""
        cache = LFUCache(max_size=2)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_lfu_eviction_least_frequent(self):
        """Test that least frequently used items are evicted."""
        cache = LFUCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key2")  # key2 more frequent
        cache.get("key2")
        cache.set("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_lfu_stats(self):
        """Test statistics tracking."""
        cache = LFUCache(max_size=10)

        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("key1")
        cache.get("key2")  # Miss

        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 1


# ============================================================================
# TTL CACHE TESTS
# ============================================================================


@pytest.mark.slow
class TestTTLCache:
    """Tests for TTL cache implementation."""

    def test_ttl_basic_set_get(self):
        """Test basic set and get operations."""
        cache = TTLCache(ttl_seconds=10)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    @freeze_time("2025-01-01 12:00:00")
    def test_ttl_expiration(self):
        """Test that expired items are removed."""
        cache = TTLCache(ttl_seconds=1)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Move time forward by 1.1 seconds
        with freeze_time("2025-01-01 12:00:01.1"):
            assert cache.get("key1") is None

    @freeze_time("2025-01-01 12:00:00")
    def test_ttl_custom_ttl(self):
        """Test custom TTL per item."""
        cache = TTLCache(ttl_seconds=10)

        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"

        # Move time forward by 1.1 seconds
        with freeze_time("2025-01-01 12:00:01.1"):
            assert cache.get("key1") is None

    @freeze_time("2025-01-01 12:00:00")
    def test_ttl_stats(self):
        """Test statistics with expiration."""
        cache = TTLCache(ttl_seconds=1, max_size=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.size() == 2

        # Move time forward by 1.1 seconds
        with freeze_time("2025-01-01 12:00:01.1"):
            # Size should only count non-expired
            assert cache.size() == 0


# ============================================================================
# FIFO CACHE TESTS
# ============================================================================


class TestFIFOCache:
    """Tests for FIFO cache implementation."""

    def test_fifo_basic_set_get(self):
        """Test basic set and get operations."""
        cache = FIFOCache(max_size=2)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_fifo_eviction_oldest(self):
        """Test that oldest items are evicted first."""
        cache = FIFOCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Access doesn't affect order
        cache.set("key3", "value3")  # Should evict key1 (oldest)

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_fifo_stats(self):
        """Test statistics tracking."""
        cache = FIFOCache(max_size=10)

        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("key2")  # Miss

        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1


# ============================================================================
# CACHE MANAGER TESTS
# ============================================================================


class TestCacheManager:
    """Tests for cache manager."""

    def test_cache_manager_register_get(self):
        """Test cache registration and retrieval."""
        manager = CacheManager()
        cache = LRUCache(100)

        manager.register("users", cache)
        retrieved = manager.get("users")

        assert retrieved is cache

    def test_cache_manager_multiple_caches(self):
        """Test managing multiple caches."""
        manager = CacheManager()

        manager.register("lru", LRUCache(100))
        manager.register("ttl", TTLCache(300))

        assert manager.get("lru") is not None
        assert manager.get("ttl") is not None
        assert manager.get("unknown") is None

    def test_cache_manager_unregister(self):
        """Test unregistering caches."""
        manager = CacheManager()
        cache = LRUCache(100)

        manager.register("users", cache)
        assert manager.unregister("users") is True
        assert manager.get("users") is None
        assert manager.unregister("users") is False

    def test_cache_manager_clear_all(self):
        """Test clearing all caches."""
        manager = CacheManager()
        cache1 = LRUCache(100)
        cache2 = LRUCache(100)

        manager.register("cache1", cache1)
        manager.register("cache2", cache2)

        cache1.set("key1", "value1")
        cache2.set("key2", "value2")

        manager.clear_all()

        assert cache1.size() == 0
        assert cache2.size() == 0

    def test_cache_manager_stats_all(self):
        """Test getting all statistics."""
        manager = CacheManager()
        cache1 = LRUCache(100)
        cache2 = LRUCache(100)

        manager.register("cache1", cache1)
        manager.register("cache2", cache2)

        cache1.set("key1", "value1")
        cache2.set("key2", "value2")

        stats = manager.stats_all()

        assert "cache1" in stats
        assert "cache2" in stats
        assert stats["cache1"].size == 1
        assert stats["cache2"].size == 1


# ============================================================================
# CACHE DECORATOR TESTS
# ============================================================================


class TestCacheDecorator:
    """Tests for @cache_result decorator."""

    def test_cache_decorator_caches_result(self):
        """Test that decorator caches function results."""
        call_count = 0

        @cache_result(max_size=100)
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * x

        # First call - computed
        result1 = expensive_function(5)
        assert result1 == 25
        assert call_count == 1

        # Second call - from cache
        result2 = expensive_function(5)
        assert result2 == 25
        assert call_count == 1  # Not incremented

    def test_cache_decorator_different_args(self):
        """Test decorator with different arguments."""
        call_count = 0

        @cache_result(max_size=100)
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * x

        result1 = expensive_function(5)
        result2 = expensive_function(10)

        assert result1 == 25
        assert result2 == 100
        assert call_count == 2  # Both computed

    def test_cache_decorator_lru_strategy(self):
        """Test decorator with LRU strategy."""

        @cache_result(max_size=2, strategy="lru")
        def func(x: int) -> int:
            return x * x

        func(1)
        func(2)
        func(1)  # Make 1 recently used
        func(3)  # Should evict 2

        # Cache stats should show 1 eviction
        stats = func.stats()
        assert stats.evictions == 1

    @freeze_time("2025-01-01 12:00:00")
    def test_cache_decorator_ttl_strategy(self):
        """Test decorator with TTL strategy."""

        @cache_result(max_size=100, ttl=1, strategy="ttl")
        def func(x: int) -> int:
            return x * x

        result1 = func(5)
        assert result1 == 25

        # Move time forward past TTL
        with freeze_time("2025-01-01 12:00:01.1"):
            # Should recompute after TTL
            result2 = func(5)
            assert result2 == 25


# ============================================================================
# THREAD SAFETY TESTS
# ============================================================================


class TestCacheThreadSafety:
    """Tests for thread safety of caches."""

    def test_lru_thread_safe(self):
        """Test LRU cache with concurrent access."""
        cache = LRUCache(max_size=1000)
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    cache.set(key, f"value_{i}")
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.size() == 1000

    def test_ttl_thread_safe(self):
        """Test TTL cache with concurrent access."""
        cache = TTLCache(ttl_seconds=10, max_size=1000)
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    cache.set(key, f"value_{i}")
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# CACHE STATS TESTS
# ============================================================================


class TestCacheStats:
    """Tests for cache statistics."""

    def test_cache_stats_hit_rate(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=10, misses=90, evictions=5, size=100, max_size=1000)

        assert stats.hit_rate == 10.0

    def test_cache_stats_string_representation(self):
        """Test string representation."""
        stats = CacheStats(hits=10, misses=90, evictions=5, size=100, max_size=1000)

        stats_str = str(stats)
        assert "hits=10" in stats_str
        assert "misses=90" in stats_str
        assert "hit_rate=" in stats_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
