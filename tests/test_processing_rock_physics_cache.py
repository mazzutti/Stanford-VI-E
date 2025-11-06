"""Tests for src.processing.rock_physics.cache module.

Tests for ModelCache class for managing rock physics model derived attributes.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.processing.rock_physics.cache import ModelCache


class TestModelCacheInitialization:
    """Test ModelCache initialization."""

    def test_init_without_disk_cache(self):
        """Test initialization without disk cache."""
        cache = ModelCache()

        assert cache.disk_cache is None
        assert cache._derived_cache is None
        assert cache._refl_cache is None

    def test_init_with_disk_cache(self):
        """Test initialization with disk cache."""
        disk_cache = Mock()
        cache = ModelCache(disk_cache=disk_cache)

        assert cache.disk_cache is disk_cache
        assert cache._derived_cache is None
        assert cache._refl_cache is None

    def test_init_with_real_disk_cache_mock(self):
        """Test initialization with realistic disk cache mock."""
        disk_cache = MagicMock()
        disk_cache.get = Mock(return_value=None)
        disk_cache.set = Mock()

        cache = ModelCache(disk_cache=disk_cache)
        assert cache.disk_cache is disk_cache


class TestModelCacheDerivedMethods:
    """Test derived attribute cache methods."""

    def test_get_derived_initial_none(self):
        """Test get_derived returns None initially."""
        cache = ModelCache()

        result = cache.get_derived()
        assert result is None

    def test_set_derived_and_get(self):
        """Test set_derived and get_derived roundtrip."""
        cache = ModelCache()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        cache.set_derived(data)
        result = cache.get_derived()

        assert np.array_equal(result, data)

    def test_set_derived_replaces_previous(self):
        """Test set_derived replaces previous value."""
        cache = ModelCache()
        data1 = np.array([1.0, 2.0])
        data2 = np.array([3.0, 4.0])

        cache.set_derived(data1)
        cache.set_derived(data2)
        result = cache.get_derived()

        np.testing.assert_array_equal(result, data2)

    def test_set_derived_with_different_shapes(self):
        """Test set_derived with various array shapes."""
        cache = ModelCache()

        # 1D array
        cache.set_derived(np.array([1.0, 2.0, 3.0]))
        assert cache.get_derived().ndim == 1

        # 2D array
        cache.set_derived(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert cache.get_derived().ndim == 2

        # 3D array
        cache.set_derived(np.ones((2, 3, 4)))
        assert cache.get_derived().ndim == 3

    def test_set_derived_with_different_dtypes(self):
        """Test set_derived with different data types."""
        cache = ModelCache()

        # Float32
        cache.set_derived(np.array([1.0, 2.0], dtype=np.float32))
        assert cache.get_derived().dtype == np.float32

        # Float64
        cache.set_derived(np.array([1.0, 2.0], dtype=np.float64))
        assert cache.get_derived().dtype == np.float64

        # Integer
        cache.set_derived(np.array([1, 2, 3], dtype=np.int32))
        assert cache.get_derived().dtype == np.int32

    def test_derived_cache_multiple_sets(self):
        """Test multiple set_derived calls."""
        cache = ModelCache()

        for i in range(5):
            data = np.ones((2, 3)) * i
            cache.set_derived(data)
            assert np.all(cache.get_derived() == i)


class TestModelCacheReflectivityMethods:
    """Test reflectivity cache methods."""

    def test_get_reflectivity_initial_none(self):
        """Test get_reflectivity returns None initially."""
        cache = ModelCache()

        result = cache.get_reflectivity()
        assert result is None

    def test_set_reflectivity_and_get(self):
        """Test set_reflectivity and get_reflectivity roundtrip."""
        cache = ModelCache()
        data = np.array([[0.1, 0.2], [0.3, 0.4]])

        cache.set_reflectivity(data)
        result = cache.get_reflectivity()

        assert np.array_equal(result, data)

    def test_set_reflectivity_replaces_previous(self):
        """Test set_reflectivity replaces previous value."""
        cache = ModelCache()
        data1 = np.array([0.1, 0.2])
        data2 = np.array([0.3, 0.4])

        cache.set_reflectivity(data1)
        cache.set_reflectivity(data2)
        result = cache.get_reflectivity()

        np.testing.assert_array_equal(result, data2)

    def test_set_reflectivity_with_different_shapes(self):
        """Test set_reflectivity with various array shapes."""
        cache = ModelCache()

        # 1D array
        cache.set_reflectivity(np.array([0.1, 0.2, 0.3]))
        assert cache.get_reflectivity().ndim == 1

        # 2D array
        cache.set_reflectivity(np.array([[0.1, 0.2], [0.3, 0.4]]))
        assert cache.get_reflectivity().ndim == 2

        # 3D array
        cache.set_reflectivity(np.ones((2, 3, 4)) * 0.1)
        assert cache.get_reflectivity().ndim == 3

    def test_reflectivity_independent_from_derived(self):
        """Test that reflectivity and derived caches are independent."""
        cache = ModelCache()
        derived = np.array([1.0, 2.0])
        refl = np.array([0.1, 0.2])

        cache.set_derived(derived)
        cache.set_reflectivity(refl)

        np.testing.assert_array_equal(cache.get_derived(), derived)
        np.testing.assert_array_equal(cache.get_reflectivity(), refl)

    def test_reflectivity_multiple_sets(self):
        """Test multiple set_reflectivity calls."""
        cache = ModelCache()

        for i in range(5):
            data = np.ones((2, 3)) * (0.1 * i)
            cache.set_reflectivity(data)
            expected = np.ones((2, 3)) * (0.1 * i)
            np.testing.assert_array_almost_equal(cache.get_reflectivity(), expected)


class TestModelCacheInvalidation:
    """Test cache invalidation methods."""

    def test_invalidate_clears_derived(self):
        """Test invalidate clears derived cache."""
        cache = ModelCache()
        cache.set_derived(np.array([1.0, 2.0]))

        cache.invalidate()

        assert cache.get_derived() is None

    def test_invalidate_clears_reflectivity(self):
        """Test invalidate clears reflectivity cache."""
        cache = ModelCache()
        cache.set_reflectivity(np.array([0.1, 0.2]))

        cache.invalidate()

        assert cache.get_reflectivity() is None

    def test_invalidate_clears_both_caches(self):
        """Test invalidate clears both derived and reflectivity."""
        cache = ModelCache()
        cache.set_derived(np.array([1.0, 2.0]))
        cache.set_reflectivity(np.array([0.1, 0.2]))

        cache.invalidate()

        assert cache.get_derived() is None
        assert cache.get_reflectivity() is None

    def test_invalidate_multiple_times(self):
        """Test calling invalidate multiple times."""
        cache = ModelCache()
        cache.set_derived(np.array([1.0, 2.0]))
        cache.set_reflectivity(np.array([0.1, 0.2]))

        # First invalidation
        cache.invalidate()
        assert cache.get_derived() is None
        assert cache.get_reflectivity() is None

        # Second invalidation (should not error)
        cache.invalidate()
        assert cache.get_derived() is None
        assert cache.get_reflectivity() is None

    def test_invalidate_then_set_again(self):
        """Test setting cache after invalidation."""
        cache = ModelCache()
        data1 = np.array([1.0, 2.0])

        cache.set_derived(data1)
        cache.invalidate()

        data2 = np.array([3.0, 4.0])
        cache.set_derived(data2)

        np.testing.assert_array_equal(cache.get_derived(), data2)


class TestModelCacheWorkflows:
    """Test typical ModelCache workflows."""

    def test_workflow_cache_and_invalidate(self):
        """Test workflow: set -> get -> invalidate."""
        cache = ModelCache()
        derived = np.array([[1.0, 2.0], [3.0, 4.0]])
        refl = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Set both caches
        cache.set_derived(derived)
        cache.set_reflectivity(refl)

        # Verify
        np.testing.assert_array_equal(cache.get_derived(), derived)
        np.testing.assert_array_equal(cache.get_reflectivity(), refl)

        # Invalidate
        cache.invalidate()

        # Verify invalidated
        assert cache.get_derived() is None
        assert cache.get_reflectivity() is None

    def test_workflow_partial_invalidation_workaround(self):
        """Test partial cache management (via repeated sets)."""
        cache = ModelCache()

        # Set both
        cache.set_derived(np.array([1.0]))
        cache.set_reflectivity(np.array([0.1]))

        # Update only derived
        new_derived = np.array([2.0])
        cache.set_derived(new_derived)

        # Derived updated, reflectivity unchanged
        np.testing.assert_array_equal(cache.get_derived(), new_derived)
        np.testing.assert_array_equal(cache.get_reflectivity(), [0.1])

    def test_workflow_with_disk_cache_mock(self):
        """Test workflow with disk cache."""
        disk_cache = MagicMock()
        cache = ModelCache(disk_cache=disk_cache)

        data = np.array([1.0, 2.0])
        cache.set_derived(data)

        # Cache should have disk_cache reference
        assert cache.disk_cache is disk_cache

    def test_large_array_caching(self):
        """Test caching large arrays."""
        cache = ModelCache()

        # Create large array (100 x 100 x 100)
        large_data = np.random.rand(100, 100, 100)
        cache.set_derived(large_data)

        # Verify shape and values
        cached = cache.get_derived()
        assert cached.shape == (100, 100, 100)
        np.testing.assert_array_equal(cached, large_data)

    def test_small_array_caching(self):
        """Test caching very small arrays."""
        cache = ModelCache()

        # Single element
        cache.set_derived(np.array([42.0]))
        assert cache.get_derived()[0] == 42.0

        # Empty array
        cache.set_derived(np.array([]))
        assert len(cache.get_derived()) == 0

    def test_special_float_values(self):
        """Test caching special float values."""
        cache = ModelCache()

        # Array with special values
        data = np.array([0.0, np.inf, -np.inf, 1.0])
        cache.set_derived(data)

        cached = cache.get_derived()
        assert cached[0] == 0.0
        assert np.isinf(cached[1])
        assert np.isinf(cached[2])
        assert cached[3] == 1.0

    def test_cache_with_nan_values(self):
        """Test caching arrays with NaN values."""
        cache = ModelCache()

        data = np.array([1.0, np.nan, 3.0])
        cache.set_derived(data)

        cached = cache.get_derived()
        assert cached[0] == 1.0
        assert np.isnan(cached[1])
        assert cached[2] == 3.0


class TestModelCacheEdgeCases:
    """Test edge cases and error conditions."""

    def test_get_before_set(self):
        """Test getting cache that was never set."""
        cache = ModelCache()

        assert cache.get_derived() is None
        assert cache.get_reflectivity() is None

    def test_cache_state_persistence(self):
        """Test cache state persists across operations."""
        cache = ModelCache()
        data = np.array([1.0, 2.0, 3.0])

        cache.set_derived(data)

        # Multiple gets should return same reference
        ref1 = cache.get_derived()
        ref2 = cache.get_derived()

        assert ref1 is ref2

    def test_cache_different_types(self):
        """Test caching different numeric types."""
        cache = ModelCache()

        # Complex numbers
        cache.set_derived(np.array([1 + 2j, 3 + 4j]))
        cached = cache.get_derived()
        assert cached.dtype == np.complex128

    def test_readonly_array_caching(self):
        """Test caching read-only arrays."""
        cache = ModelCache()

        # Create read-only array
        data = np.array([1.0, 2.0])
        data.flags.writeable = False

        cache.set_derived(data)
        cached = cache.get_derived()
        assert cached is data

    def test_fortran_order_array(self):
        """Test caching Fortran-order arrays."""
        cache = ModelCache()

        # Fortran order
        data = np.array([[1, 2], [3, 4]], order="F")
        cache.set_derived(data)

        cached = cache.get_derived()
        assert cached is data

    def test_memory_layout_preservation(self):
        """Test that cache preserves memory layout."""
        cache = ModelCache()

        c_order = np.array([[1, 2], [3, 4]], order="C")
        cache.set_derived(c_order)

        cached = cache.get_derived()
        np.testing.assert_array_equal(cached, c_order)


class TestModelCacheWithDiskCache:
    """Test ModelCache interaction with disk cache."""

    def test_disk_cache_reference_stored(self):
        """Test disk cache reference is stored."""
        disk_cache = Mock()
        cache = ModelCache(disk_cache=disk_cache)

        assert cache.disk_cache is disk_cache

    def test_disk_cache_none_default(self):
        """Test disk cache defaults to None."""
        cache = ModelCache()

        assert cache.disk_cache is None

    def test_disk_cache_not_called_on_memory_cache(self):
        """Test disk cache is not called by memory operations."""
        disk_cache = Mock()
        cache = ModelCache(disk_cache=disk_cache)

        # These operations only use memory cache
        cache.set_derived(np.array([1.0]))
        cache.get_derived()

        # Disk cache should not be called
        disk_cache.set.assert_not_called()
        disk_cache.get.assert_not_called()
