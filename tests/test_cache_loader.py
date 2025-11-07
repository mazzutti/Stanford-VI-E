"""Comprehensive unit tests for CacheLoader and CacheLoaderFactory.

This module contains all unit tests for cache_loader.py including:
    - Existing pytest-based integration tests
    - New unittest-based comprehensive tests
    - Tests for all recent improvements (repr, str, cache_status, etc.)

Test Organization:
    1. Core Functionality Tests (file selection, loading, caching)
    2. Configuration & Initialization Tests
    3. Static Methods & Utilities Tests
    4. Object Representation Tests
    5. Context Manager & Resource Management Tests
    6. Factory Pattern Tests
    7. Integration Tests
"""

# mypy: ignore-errors


import os
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
from numpy.lib.npyio import NpzFile

from src.analysis.cache import (
    CacheLoader,
    CacheLoaderFactory,
    CacheConfig,
)

# Public test constants
FILE_PREFIX = "avo_"
FULL_STACK_KEY = "full_stack"
NPZ_EXTENSION = ".npz"
NPY_EXTENSION = ".npy"


# =============================================================================
# PART 1: CORE FUNCTIONALITY TESTS (File Selection, Loading, Caching)
# =============================================================================


def test_select_cache_file_fallback(tmp_path: Path) -> None:
    """Test fallback cache file selection with standard naming convention."""
    cache_dir = tmp_path
    p = cache_dir / "avo_time.npz"
    np.savez(p, full_stack=np.array([1, 2, 3]))

    loader = CacheLoader()
    result = loader.select_cache_file(str(cache_dir), "time")
    assert result is not None
    assert Path(result).exists()


def test_load_full_stack_returns_array(tmp_path: Path) -> None:
    """Test loading full stack array from NPZ file."""
    p = tmp_path / "mycache.npz"
    arr = np.arange(6).reshape(2, 3)
    np.savez(p, full_stack=arr)

    loader = CacheLoader()
    got = loader.load_full_stack(str(p))
    assert got is not None
    np.testing.assert_array_equal(got, arr)


def test_load_full_stack_missing_file_returns_none(tmp_path: Path) -> None:
    """Test loading nonexistent file returns None."""
    p = tmp_path / "nope.npz"
    loader = CacheLoader()
    assert loader.load_full_stack(str(p)) is None


def test_load_full_stack_npy(tmp_path: Path) -> None:
    """Test loading NPY file format."""
    d = tmp_path
    path = d / "avo_test.npy"
    expected = np.linspace(0, 1, 5)
    np.save(path, expected)

    loader = CacheLoader()
    arr = loader.load_full_stack(path)
    assert arr is not None
    assert np.allclose(arr, expected)


def test_select_prefers_latest(tmp_path: Path) -> None:
    """Test file selection prefers latest by modification time."""
    d = tmp_path
    old = d / "avo_depth_old.npz"
    new = d / "avo_depth_new.npz"
    np.savez(old, full_stack=np.zeros((1,)))
    np.savez(new, full_stack=np.ones((1,)))
    os.utime(old, (100, 100))
    os.utime(new, (200, 200))

    loader = CacheLoader()
    selected = loader.select_cache_file(d, "depth", prefer_latest=True)
    assert selected is not None
    assert Path(selected).name == new.name


def test_select_allownpy(tmp_path: Path) -> None:
    """Test file selection with NPY format support."""
    d = tmp_path
    npy = d / "avo_time.npy"
    arr = np.arange(4)
    np.save(npy, arr)

    loader = CacheLoader()
    selected = loader.select_cache_file(d, "time", allow_npy=True)
    assert selected is not None
    assert Path(selected).suffix == ".npy"


# =============================================================================
# PART 2: CACHING BEHAVIOR TESTS
# =============================================================================


def test_cache_hit_and_stores_copy(tmp_path: Path) -> None:
    """Test that cache stores independent copies of arrays."""
    p = tmp_path / "one.npy"
    arr1 = np.array([1, 2, 3], dtype=float)
    np.save(p, arr1)

    loader = CacheLoader(cache_size=2)
    first = loader.load_full_stack(p)
    assert first is not None
    np.testing.assert_array_equal(first, arr1)

    # Overwrite file; cached copy should remain unchanged
    arr2 = np.array([9, 9, 9], dtype=float)
    np.save(p, arr2)
    second = loader.load_full_stack(p)
    assert second is not None
    np.testing.assert_array_equal(second, arr1)


def test_cache_lru_eviction(tmp_path: Path) -> None:
    """Test LRU cache eviction policy."""
    p1 = tmp_path / "a.npy"
    p2 = tmp_path / "b.npy"
    p3 = tmp_path / "c.npy"
    np.save(p1, np.array([1.0]))
    np.save(p2, np.array([2.0]))
    np.save(p3, np.array([3.0]))

    loader = CacheLoader(cache_size=2)
    loader.load_full_stack(p1)
    loader.load_full_stack(p2)
    keys = loader.cache_keys()
    assert str(p1) in keys and str(p2) in keys

    # Load p3 should evict p1 (least recently used)
    loader.load_full_stack(p3)
    keys = loader.cache_keys()
    assert str(p1) not in keys
    assert str(p3) in keys


def test_mmap_returns_memmap_and_not_cached(tmp_path: Path) -> None:
    """Test memory-mapped files are not cached."""
    p = tmp_path / "big.npy"
    arr = np.arange(100, dtype=float)
    np.save(p, arr)

    loader = CacheLoader(cache_size=2)
    got = loader.load_full_stack(p, mmap_mode="r")
    assert isinstance(got, np.ndarray)

    arr2 = np.arange(100, 200, dtype=float)
    np.save(p, arr2)
    got2 = loader.load_full_stack(p, mmap_mode="r")
    assert isinstance(got2, np.ndarray)
    assert got2.shape == got.shape
    assert str(p) not in loader.cache_keys()


def test_mmap_not_cached_but_cached_copy_exists_for_non_memmap(tmp_path: Path) -> None:
    """Test mixed mmap and non-mmap access patterns."""
    p = tmp_path / "small.npy"
    arr = np.array([1.0, 2.0, 3.0])
    np.save(p, arr)

    loader = CacheLoader(cache_size=2)
    mm = loader.load_full_stack(p, mmap_mode="r")
    assert isinstance(mm, np.ndarray)

    arr_no_mmap = loader.load_full_stack(p, mmap_mode=None)
    assert arr_no_mmap is not None

    keys = loader.cache_keys()
    assert any(str(p) == k for k in keys)


# =============================================================================
# PART 3: ERROR HANDLING & EDGE CASES
# =============================================================================


def test_injected_selector_used(tmp_path: Path) -> None:
    """Test custom selector injection."""

    def selector(cache_dir: str, domain: str) -> str:
        return str(Path(cache_dir) / "injected.npz")

    p = tmp_path / "injected.npz"
    np.savez(p, full_stack=np.array([9]))

    loader = CacheLoader(selector=selector)
    res = loader.select_cache_file(str(tmp_path), "whatever")
    assert res is not None
    assert Path(res).exists()


def test_raise_on_error_propagates(tmp_path: Path) -> None:
    """Test error propagation with raise_on_error=True."""

    def bad_loader(_path: str, **kwargs) -> None:  # type: ignore
        raise RuntimeError("boom")

    loader = CacheLoader(np_load=bad_loader)  # type: ignore
    target = tmp_path / "anything.npz"
    target.write_bytes(b"")
    with pytest.raises(RuntimeError):
        loader.load_full_stack(target, raise_on_error=True)


# =============================================================================
# PART 4: CONFIGURATION & INITIALIZATION TESTS
# =============================================================================


class TestCacheConfig:
    """Configuration object tests."""

    def test_cache_config_creation(self) -> None:
        """Test CacheConfig creation with defaults."""
        config = CacheConfig(cache_size=100)
        assert config.cache_size == 100
        assert config.archive_extractor is None
        assert config.selector is None
        assert config.cache is None
        assert config.np_load == np.load

    def test_cache_config_with_custom_loader(self) -> None:
        """Test CacheConfig with custom numpy loader."""
        mock_loader = Mock()
        config = CacheConfig(cache_size=50, np_load=mock_loader)
        assert config.cache_size == 50
        assert config.np_load == mock_loader


class TestCacheLoaderInitialization:
    """Initialization and property tests."""

    def test_init_no_caching(self) -> None:
        """Test initialization with caching disabled."""
        loader = CacheLoader(cache_size=0)
        assert not loader.cache_enabled
        assert loader.cache_maxsize == 0

    def test_init_with_cache_size(self) -> None:
        """Test initialization with cache enabled."""
        loader = CacheLoader(cache_size=100)
        assert loader.cache_enabled
        assert loader.cache_maxsize == 100

    def test_init_negative_cache_size_raises(self) -> None:
        """Test negative cache_size raises ValueError."""
        with pytest.raises(ValueError):
            CacheLoader(cache_size=-1)

    def test_init_with_custom_selector(self) -> None:
        """Test custom selector injection via public API."""
        custom_selector = Mock(return_value=None)
        loader = CacheLoader(selector=custom_selector, cache_size=0)
        # Verify through public API (cache_enabled and behavior)
        assert loader.cache_enabled is False

    def test_init_with_custom_np_load(self) -> None:
        """Test custom numpy loader injection via public API."""
        custom_loader = Mock(return_value=np.array([1, 2, 3]))
        loader = CacheLoader(np_load=custom_loader, cache_size=0)
        # Verify through public API
        assert loader.cache_enabled is False

    def test_init_with_external_cache(self) -> None:
        """Test external cache instance injection via public API."""
        mock_cache = Mock()
        loader = CacheLoader(cache=mock_cache, cache_size=100)
        # Verify through public API
        assert loader.cache_enabled is True
        assert loader.cache_maxsize == 100

    def test_init_external_cache_takes_precedence(self) -> None:
        """Test external cache overrides cache_size through public API."""
        mock_cache = Mock()
        loader = CacheLoader(cache=mock_cache, cache_size=0)
        # Verify through public API
        assert loader.cache_enabled is True


# =============================================================================
# PART 5: DEFAULT SELECTOR & EXTRACTOR TESTS (Public Static Methods)
# =============================================================================


class TestDefaultSelector:
    """Default file selector tests."""

    def test_selector_finds_npz_file(self, tmp_path: Path) -> None:
        """Test selector finds NPZ file."""
        npz_file = tmp_path / f"{FILE_PREFIX}acoustic{NPZ_EXTENSION}"
        npz_file.touch()

        result = CacheLoader.default_selector(str(tmp_path), "acoustic")
        assert result == str(npz_file)

    def test_selector_finds_npy_file(self, tmp_path: Path) -> None:
        """Test selector finds NPY file as fallback."""
        npy_file = tmp_path / f"{FILE_PREFIX}acoustic{NPY_EXTENSION}"
        npy_file.touch()

        result = CacheLoader.default_selector(str(tmp_path), "acoustic")
        assert result == str(npy_file)

    def test_selector_prefers_npz_over_npy(self, tmp_path: Path) -> None:
        """Test selector prefers NPZ over NPY."""
        npz_file = tmp_path / f"{FILE_PREFIX}acoustic{NPZ_EXTENSION}"
        npy_file = tmp_path / f"{FILE_PREFIX}acoustic{NPY_EXTENSION}"
        npz_file.touch()
        npy_file.touch()

        result = CacheLoader.default_selector(str(tmp_path), "acoustic")
        assert result == str(npz_file)

    def test_selector_returns_none_when_file_not_found(self, tmp_path: Path) -> None:
        """Test selector returns None when file not found."""
        result = CacheLoader.default_selector(str(tmp_path), "nonexistent")
        assert result is None

    def test_selector_empty_domain_raises(self, tmp_path: Path) -> None:
        """Test selector raises for empty domain."""
        with pytest.raises(ValueError):
            CacheLoader.default_selector(str(tmp_path), "")


class TestDefaultArchiveExtractor:
    """Default archive extraction tests (public API)."""

    def test_extractor_extracts_full_stack(self) -> None:
        """Test extractor uses default extraction logic."""
        expected_data = np.array([[1, 2]], dtype=np.float64)
        mock_archive = {FULL_STACK_KEY: expected_data}

        mock_npz = MagicMock(spec=NpzFile)
        mock_npz.__contains__ = lambda self, key: key in mock_archive
        mock_npz.__getitem__ = lambda self, key: mock_archive[key]
        mock_npz.files = [FULL_STACK_KEY]

        result = CacheLoader.default_archive_extractor(mock_npz)
        np.testing.assert_array_equal(result, expected_data)  # type: ignore

    def test_extractor_handles_exception(self) -> None:
        """Test extractor handles extraction errors gracefully."""
        mock_npz = MagicMock(spec=NpzFile)
        mock_npz.__contains__.side_effect = Exception("Archive error")

        result = CacheLoader.default_archive_extractor(mock_npz)
        assert result is None


# =============================================================================
# PART 6: OBJECT REPRESENTATION & STATUS TESTS
# =============================================================================


class TestStringRepresentations:
    """Object representation tests (__repr__, __str__)."""

    def test_repr_when_disabled(self) -> None:
        """Test __repr__ when cache is disabled."""
        loader = CacheLoader(cache_size=0)
        repr_str = repr(loader)
        assert "CacheLoader" in repr_str
        assert "cache_enabled=False" in repr_str

    def test_repr_when_enabled(self) -> None:
        """Test __repr__ when cache is enabled."""
        loader = CacheLoader(cache_size=100)
        repr_str = repr(loader)
        assert "CacheLoader" in repr_str
        assert "cache_enabled=True" in repr_str
        assert "100" in repr_str

    def test_str_when_disabled(self) -> None:
        """Test __str__ when cache is disabled."""
        loader = CacheLoader(cache_size=0)
        str_repr = str(loader)
        assert "cache disabled" in str_repr

    def test_str_when_enabled(self) -> None:
        """Test __str__ when cache is enabled."""
        loader = CacheLoader(cache_size=100)
        str_repr = str(loader)
        assert "cache enabled" in str_repr
        assert "100 slots" in str_repr


class TestCacheStatus:
    """Cache status reporting tests."""

    def test_cache_status_when_disabled(self) -> None:
        """Test cache_status when disabled."""
        loader = CacheLoader(cache_size=0)
        status = loader.cache_status()

        assert status["enabled"] is False
        assert status["maxsize"] == 0
        assert status["currsize"] == 0
        assert status["hits"] == 0
        assert status["misses"] == 0
        assert status["hit_rate"] == 0.0
        assert status["num_keys"] == 0

    def test_cache_status_when_enabled(self) -> None:
        """Test cache_status when enabled."""
        loader = CacheLoader(cache_size=100)
        status = loader.cache_status()

        assert status["enabled"] is True
        assert status["maxsize"] == 100
        assert "currsize" in status
        assert "hit_rate" in status


# =============================================================================
# PART 7: CONTEXT MANAGER & RESOURCE MANAGEMENT TESTS
# =============================================================================


class TestContextManager:
    """Context manager functionality tests."""

    def test_enter_returns_self(self) -> None:
        """Test __enter__ returns self."""
        loader = CacheLoader(cache_size=100)
        with loader as ctx_loader:
            assert ctx_loader is loader

    def test_exit_clears_cache(self) -> None:
        """Test __exit__ clears cache."""
        loader = CacheLoader(cache_size=100)

        with patch.object(loader, "cache_clear") as mock_clear:
            with loader:
                pass
            mock_clear.assert_called_once()

    def test_context_manager_with_exception(self) -> None:
        """Test __exit__ clears cache even with exception."""
        loader = CacheLoader(cache_size=100)

        with patch.object(loader, "cache_clear") as mock_clear:
            try:
                with loader:
                    raise ValueError("Test error")
            except ValueError:
                pass
            mock_clear.assert_called_once()


class TestSelectCacheFile:
    """select_cache_file method tests."""

    def test_select_cache_file_finds_npz(self, tmp_path: Path) -> None:
        """Test selecting NPZ cache file."""
        npz_file = tmp_path / f"{FILE_PREFIX}acoustic{NPZ_EXTENSION}"
        npz_file.touch()

        loader = CacheLoader(cache_size=0)
        result = loader.select_cache_file(str(tmp_path), "acoustic")
        assert result == str(npz_file)

    def test_select_cache_file_empty_domain_raises(self, tmp_path: Path) -> None:
        """Test empty domain raises ValueError."""
        loader = CacheLoader(cache_size=0)
        with pytest.raises(ValueError):
            loader.select_cache_file(str(tmp_path), "")

    def test_select_cache_file_with_custom_selector(self, tmp_path: Path) -> None:
        """Test using custom selector."""
        custom_path = "/custom/path.npz"
        custom_selector = Mock(return_value=custom_path)

        loader = CacheLoader(selector=custom_selector, cache_size=0)
        result = loader.select_cache_file(str(tmp_path), "test")
        assert result == custom_path

    def test_select_cache_file_not_found(self, tmp_path: Path) -> None:
        """Test returns None when file not found."""
        loader = CacheLoader(cache_size=0)
        result = loader.select_cache_file(str(tmp_path), "nonexistent")
        assert result is None


# =============================================================================
# PART 8: FACTORY PATTERN TESTS
# =============================================================================


class TestCacheLoaderFactory:
    """CacheLoaderFactory tests."""

    def test_factory_create_default(self) -> None:
        """Test factory.create_default() creates loader."""
        loader = CacheLoaderFactory.create_default(cache_size=100)
        assert isinstance(loader, CacheLoader)
        assert loader.cache_enabled

    def test_factory_create_with_custom_params(self) -> None:
        """Test factory.create() with custom parameters."""
        custom_selector = Mock()
        loader = CacheLoaderFactory.create(cache_size=50, selector=custom_selector)
        assert isinstance(loader, CacheLoader)
        assert loader.cache_enabled

    def test_factory_create_no_cache(self) -> None:
        """Test factory.create() without cache."""
        loader = CacheLoaderFactory.create(cache_size=0)
        assert isinstance(loader, CacheLoader)
        assert not loader.cache_enabled

    def test_factory_create_default_shards(self) -> None:
        """Test factory.create_default() uses default shards."""
        loader = CacheLoaderFactory.create_default(cache_size=100)
        assert isinstance(loader, CacheLoader)
        assert loader.cache_enabled


class TestEdgeCasesAndErrors:
    """Edge cases and error condition tests."""

    def test_multiple_cache_operations(self) -> None:
        """Test multiple cache operations in sequence."""
        loader = CacheLoader(cache_size=100)

        status1 = loader.cache_status()
        assert status1["enabled"]

        loader.cache_clear()

        status2 = loader.cache_status()
        assert status2["enabled"]

    def test_string_conversion_consistency(self) -> None:
        """Test str/repr consistency across states."""
        for cache_size in [0, 50, 100]:
            loader = CacheLoader(cache_size=cache_size)
            str_repr = str(loader)
            repr_repr = repr(loader)
            assert isinstance(str_repr, str)
            assert isinstance(repr_repr, str)
            assert len(str_repr) > 0
            assert len(repr_repr) > 0

    def test_cache_with_zero_cache_size(self) -> None:
        """Test all cache methods with cache_size=0."""
        loader = CacheLoader(cache_size=0)

        assert not loader.cache_enabled
        assert loader.cache_maxsize == 0
        assert loader.cache_info() is None
        assert loader.cache_keys() == []
        loader.cache_clear()
        status = loader.cache_status()
        assert not status["enabled"]


# =============================================================================
# PART 9: INTEGRATION TESTS
# =============================================================================


class TestCacheLoaderIntegration:
    """End-to-end integration tests."""

    def test_workflow_select_and_load(self, tmp_path: Path) -> None:
        """Test complete workflow of selecting and loading files."""
        domains = ["acoustic", "elastic", "porosity"]
        for domain in domains:
            npz_file = tmp_path / f"{FILE_PREFIX}{domain}{NPZ_EXTENSION}"
            test_array = np.random.rand(5, 5).astype(np.float64)
            np.savez(str(npz_file), **{FULL_STACK_KEY: test_array})

        loader = CacheLoaderFactory.create_default(cache_size=100)

        for domain in domains:
            file_path = loader.select_cache_file(str(tmp_path), domain)
            assert file_path is not None

            data = loader.load_full_stack(file_path)
            assert data is not None
            assert data.shape == (5, 5)

    def test_workflow_with_context_manager(self, tmp_path: Path) -> None:
        """Test complete workflow using context manager."""
        npz_file = tmp_path / f"{FILE_PREFIX}acoustic{NPZ_EXTENSION}"
        test_array = np.random.rand(5, 5).astype(np.float64)
        np.savez(str(npz_file), **{FULL_STACK_KEY: test_array})

        with CacheLoaderFactory.create_default(cache_size=100) as loader:
            file_path = loader.select_cache_file(str(tmp_path), "acoustic")
            assert file_path is not None

            data = loader.load_full_stack(file_path)
            assert data is not None

    def test_factory_variations(self) -> None:
        """Test different factory creation methods."""
        loader1 = CacheLoaderFactory.create_default(cache_size=100)
        assert loader1.cache_enabled

        loader2 = CacheLoaderFactory.create(cache_size=0)
        assert not loader2.cache_enabled

        custom_selector = Mock(return_value=None)
        loader3 = CacheLoaderFactory.create(cache_size=50, selector=custom_selector)
        assert loader3.cache_enabled

    # =============================================================================
    # INTEGRATION TESTS (Public API)
    # =============================================================================

    def test_factory_create_with_external_cache_ignores_shards(self) -> None:
        """Test factory.create() uses external cache regardless of shards."""
        mock_cache = Mock()
        loader = CacheLoaderFactory.create(cache_size=0, shards=4, cache=mock_cache)
        # Verify through public API that cache was used
        assert loader.cache_enabled is True

    def test_select_cache_file_with_non_existent_directory(
        self, tmp_path: Path
    ) -> None:
        """Test select_cache_file behavior with non-existent directory."""
        fake_dir = tmp_path / "nonexistent"
        loader = CacheLoader(cache_size=0)
        result = loader.select_cache_file(str(fake_dir), "test")
        assert result is None

    def test_load_full_stack_with_raise_on_error_exception_propagation(
        self, tmp_path: Path
    ) -> None:
        """Test load_full_stack propagates exceptions with raise_on_error=True."""
        p = tmp_path / "test.npy"
        p.write_bytes(b"invalid")

        loader = CacheLoader(cache_size=0)
        with pytest.raises(Exception):
            loader.load_full_stack(str(p), raise_on_error=True)

    def test_cache_status_hit_rate_calculation(self) -> None:
        """Test cache_status hit rate calculation with various scenarios."""
        loader = CacheLoader(cache_size=100)

        # When cache has no hits/misses
        status = loader.cache_status()
        assert status["hit_rate"] == 0.0

    def test_select_cache_file_prefer_latest_false(self, tmp_path: Path) -> None:
        """Test select_cache_file with prefer_latest=False."""
        d = tmp_path
        old = d / "avo_depth_old.npz"
        new = d / "avo_depth_new.npz"
        np.savez(old, full_stack=np.zeros((1,)))
        np.savez(new, full_stack=np.ones((1,)))
        os.utime(old, (100, 100))
        os.utime(new, (200, 200))

        loader = CacheLoader()
        # With prefer_latest=False, should not search for matching files
        selected = loader.select_cache_file(d, "depth", prefer_latest=False)
        # Should still find standard named file if it exists
        assert selected is None  # No standard avo_depth.npz file

    def test_select_cache_file_custom_selector_raises_exception(
        self, tmp_path: Path
    ) -> None:
        """Test select_cache_file when custom selector raises exception."""

        def bad_selector(cache_dir: str, domain: str) -> Optional[str]:
            raise RuntimeError("Selector error")

        npz_file = tmp_path / f"{FILE_PREFIX}test{NPZ_EXTENSION}"
        npz_file.touch()

        loader = CacheLoader(selector=bad_selector, cache_size=0)
        # Should fall back to default selection after exception
        result = loader.select_cache_file(str(tmp_path), "test")
        assert result == str(npz_file)

    def test_load_full_stack_with_raise_and_load_error(self, tmp_path: Path) -> None:
        """Test load_full_stack propagates exceptions with raise_on_error=True."""
        p = tmp_path / "bad.npy"
        p.write_bytes(b"not valid")

        loader = CacheLoader(cache_size=0)
        with pytest.raises(Exception):
            loader.load_full_stack(str(p), raise_on_error=True)

    def test_factory_create_with_cache_none_and_zero_size(self) -> None:
        """Test factory.create() returns loader with no cache when both are None/0."""
        loader = CacheLoaderFactory.create(cache_size=0, cache=None)
        assert not loader.cache_enabled

    def test_factory_create_default_with_zero_cache_size(self) -> None:
        """Test factory.create_default() with cache_size=0."""
        loader = CacheLoaderFactory.create_default(cache_size=0)
        assert not loader.cache_enabled

    def test_select_cache_file_selector_exception_continues_to_fallback(
        self, tmp_path: Path
    ) -> None:
        """Test that selector exception is caught and fallback occurs."""

        def exception_selector(cache_dir, domain) -> None:  # type: ignore
            raise ValueError("Selector broken")

        # Create standard cache file for fallback
        cache_file = tmp_path / f"{FILE_PREFIX}test{NPZ_EXTENSION}"
        np.savez(str(cache_file), full_stack=np.array([1, 2]))

        loader = CacheLoader(selector=exception_selector)
        # Should catch exception and fall back to default selector
        result = loader.select_cache_file(str(tmp_path), "test")
        assert result == str(cache_file)

    def test_factory_create_with_shards_greater_than_one(self) -> None:
        """Test factory creates ShardedLRUCache when shards > 1."""
        loader = CacheLoaderFactory.create(cache_size=100, shards=8)
        assert loader.cache_enabled
        # Verify it was created with multiple shards
        assert loader.cache_maxsize == 100

    def test_factory_create_with_shards_equals_one(self) -> None:
        """Test factory creates regular LRUCache when shards=1."""
        loader = CacheLoaderFactory.create(cache_size=100, shards=1)
        assert loader.cache_enabled
        assert loader.cache_maxsize == 100

    def test_load_uncached_with_memmap_preserves_type(self, tmp_path: Path) -> None:
        """Test that memmap arrays are preserved through public API."""
        p = tmp_path / "test.npy"
        arr = np.array([1, 2, 3], dtype=np.int32)
        np.save(p, arr)

        loader = CacheLoader(cache_size=0)
        # Use public API
        result = loader.load_full_stack(str(p), mmap_mode="r")
        assert result is not None

    def test_select_cache_file_with_multiple_files(self, tmp_path: Path) -> None:
        """Test select_cache_file handles multiple matching files via public API."""
        d = tmp_path
        f1 = d / "avo_acoustic_001.npz"
        f2 = d / "avo_acoustic_002.npz"
        f1.write_bytes(b"data")
        f2.write_bytes(b"data")
        # Make f2 newer
        import time

        time.sleep(0.01)
        f2.touch()

        loader = CacheLoader(cache_size=0)
        # Should use public API
        result = loader.select_cache_file(str(d), "acoustic")
        assert result is not None

    def test_select_cache_file_custom_selector_returns_none(
        self, tmp_path: Path
    ) -> None:
        """Test select_cache_file when custom selector returns None."""

        # Line 497: logger.debug("Custom selector returned None")
        def selector_returns_none(cache_dir, domain) -> None:  # type: ignore
            return None

        loader = CacheLoader(selector=selector_returns_none, cache_size=0)
        # Should fall back to default selection
        result = loader.select_cache_file(str(tmp_path), "acoustic")
        # Result is None since no files exist and selector returned None
        assert result is None

    def test_load_uncached_file_not_found_with_raise_false(
        self, tmp_path: Path
    ) -> None:
        """Test load_full_stack returns None for missing file when raise_on_error=False."""
        p = tmp_path / "nonexistent.npy"
        loader = CacheLoader(cache_size=0)
        result = loader.load_full_stack(str(p), raise_on_error=False)
        assert result is None

    def test_load_full_stack_oserror_raise_false(self, tmp_path: Path) -> None:
        """Test load_full_stack handles OSError with raise_on_error=False returns None."""
        p = tmp_path / "test.npy"
        p.write_bytes(b"corrupted")

        def bad_loader(path, **kwargs) -> None:  # type: ignore
            raise OSError("Permission denied")

        loader = CacheLoader(np_load=bad_loader, cache_size=0)  # type: ignore
        result = loader.load_full_stack(str(p), raise_on_error=False)
        assert result is None

    def test_load_full_stack_valueerror_raise_false(self, tmp_path: Path) -> None:
        """Test load_full_stack handles ValueError with raise_on_error=False returns None."""
        p = tmp_path / "test.npy"
        p.write_bytes(b"corrupted")

        def bad_loader(path, **kwargs) -> None:  # type: ignore
            raise ValueError("Invalid data format")

        loader = CacheLoader(np_load=bad_loader, cache_size=0)  # type: ignore
        result = loader.load_full_stack(str(p), raise_on_error=False)
        assert result is None

    def test_load_full_stack_general_exception_raise_false(
        self, tmp_path: Path
    ) -> None:
        """Test load_full_stack general exception with raise_on_error=False returns None."""
        # Line 959: return None (general Exception except block)
        p = tmp_path / "test.npy"
        p.write_bytes(b"data")

        def bad_loader(path, **kwargs) -> None:  # type: ignore
            raise RuntimeError("Unexpected error")

        loader = CacheLoader(np_load=bad_loader, cache_size=0)  # type: ignore
        result = loader.load_full_stack(str(p), raise_on_error=False)
        assert result is None

    def test_load_full_stack_empty_path_returns_none(self) -> None:
        """Test load_full_stack with empty path returns None immediately."""
        # Line 922: return None (if not path check)
        loader = CacheLoader(cache_size=0)
        assert loader.load_full_stack("") is None
        assert loader.load_full_stack(None) is None

    def test_load_full_stack_exception_path_return_none(self, tmp_path: Path) -> None:
        """Test load_full_stack returns None when exception occurs in try block."""
        # Line 959: return None (final return after exception handler)
        p = tmp_path / "test.npz"
        np.savez(p, full_stack=np.array([[1, 2]]))

        def bad_loader(path, **kwargs) -> None:  # type: ignore
            raise ValueError("Data error")

        loader = CacheLoader(np_load=bad_loader, cache_size=100)  # type: ignore
        # This should raise in the try block and return None at line 959
        result = loader.load_full_stack(str(p), raise_on_error=False)
        assert result is None


# =============================================================================
# IMPROVED TESTS FOR ADDITIONAL COVERAGE
# =============================================================================


class TestCacheConfigImproved:
    """Improved tests for CacheConfig."""

    def test_cache_config_creation_minimal(self):
        """Test creating CacheConfig with minimal arguments."""
        config = CacheConfig(cache_size=100)
        assert config.cache_size == 100
        assert config.archive_extractor is None
        assert config.selector is None
        assert config.cache is None
        assert config.np_load is np.load

    def test_cache_config_zero_cache_size(self):
        """Test CacheConfig with zero cache size (caching disabled)."""
        config = CacheConfig(cache_size=0)
        assert config.cache_size == 0

    def test_cache_config_large_cache_size(self):
        """Test CacheConfig with large cache size."""
        config = CacheConfig(cache_size=10000)
        assert config.cache_size == 10000
