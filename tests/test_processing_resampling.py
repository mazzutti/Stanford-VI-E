"""Tests for the resampling module backends and services.

Tests backend implementations, backend manager, caching, and service layer.
Focuses on testable components without unit registry issues.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.processing.resampling.backends._base import BackendError, BackendResult
from src.processing.resampling.backends._implementations import (
    VectorizedBackend,
    BatchedInterpolatorBackend,
    _register_default_backends,
)
from src.processing.resampling.backends._manager import BackendManager
from src.processing.resampling.service import ResamplerService
from src.processing.resampling._cache import (
    ResamplePlanCache,
    get_resample_plan_cache,
    set_resample_plan_cache,
)
from src.io.grid import GridSpec


class TestBackendVerboseLogging:
    """Test backend verbose logging."""

    def test_set_backend_verbose_true(self):
        """Test enabling verbose logging."""
        from src.processing.resampling._resampler import (
            set_backend_verbose,
            is_backend_verbose,
        )

        set_backend_verbose(True)
        assert is_backend_verbose() is True

    def test_set_backend_verbose_false(self):
        """Test disabling verbose logging."""
        from src.processing.resampling._resampler import (
            set_backend_verbose,
            is_backend_verbose,
        )

        set_backend_verbose(False)
        assert is_backend_verbose() is False

    def test_set_backend_verbose_toggles(self):
        """Test toggling verbose logging multiple times."""
        from src.processing.resampling._resampler import (
            set_backend_verbose,
            is_backend_verbose,
        )

        for _ in range(3):
            set_backend_verbose(True)
            assert is_backend_verbose() is True
            set_backend_verbose(False)
            assert is_backend_verbose() is False


class TestVectorizedBackend:
    """Test VectorizedBackend class."""

    def test_backend_name(self):
        """Test backend name constant."""
        backend = VectorizedBackend()
        assert backend.name == "vectorized"
        assert isinstance(backend.name, str)

    def test_backend_supports_uniform(self):
        """Test backend supports uniform TWT plans."""
        backend = VectorizedBackend()
        plan = MagicMock()
        plan.uniform_twt = True
        assert backend.supports(plan) is True

    def test_backend_rejects_nonuniform(self):
        """Test backend rejects non-uniform plans."""
        backend = VectorizedBackend()
        plan = MagicMock()
        plan.uniform_twt = False
        assert backend.supports(plan) is False


class TestBatchedInterpolatorBackend:
    """Test BatchedInterpolatorBackend class."""

    def test_backend_name(self):
        """Test backend name."""
        backend = BatchedInterpolatorBackend()
        assert backend.name == "batched_interpolator"

    def test_backend_support_is_boolean(self):
        """Test backend support check returns boolean."""
        backend = BatchedInterpolatorBackend()
        plan = MagicMock()
        result = backend.supports(plan)
        assert isinstance(result, (bool, type(None))) or isinstance(result, bool)

    def test_depth_to_time_error_handling(self):
        """Test depth_to_time error when BatchedInterpolator unavailable."""
        with patch(
            "src.processing.resampling.backends._implementations.BatchedInterpolator",
            None,
        ):
            backend = BatchedInterpolatorBackend()
            plan = MagicMock()
            data = np.zeros((2, 2, 20))
            vp = np.full((2, 2, 20), 3500.0)

            with pytest.raises(BackendError):
                backend.depth_to_time(data, vp, plan)


class TestBackendManager:
    """Test BackendManager."""

    def test_manager_can_be_instantiated(self):
        """Test creating BackendManager."""
        manager = BackendManager()
        assert manager is not None

    def test_manager_register_backend(self):
        """Test registering backends."""
        manager = BackendManager()
        backend = MagicMock()
        manager.register("test", backend)
        # Should not raise

    def test_register_default_backends_succeeds(self):
        """Test registering default backends."""
        # Should not raise any exception
        _register_default_backends()


class TestResamplerService:
    """Test ResamplerService."""

    def test_service_requires_grid_spec(self):
        """Test service requires grid_spec parameter."""
        with pytest.raises(TypeError):
            ResamplerService()  # type: ignore

    def test_service_with_grid_spec(self):
        """Test service creation with grid spec."""
        grid_spec = GridSpec.from_dimensions(nx=2, ny=2, nz=10)
        service = ResamplerService(grid_spec=grid_spec)
        assert service is not None
        assert service.grid_spec == grid_spec

    def test_service_has_expected_methods(self):
        """Test service has expected public methods."""
        grid_spec = GridSpec.from_dimensions(nx=2, ny=2, nz=10)
        service = ResamplerService(grid_spec=grid_spec)
        assert hasattr(service, "depth_to_time")
        assert hasattr(service, "time_to_depth")
        assert callable(service.depth_to_time)
        assert callable(service.time_to_depth)

    def test_service_has_cache(self):
        """Test service has cache."""
        grid_spec = GridSpec.from_dimensions(nx=2, ny=2, nz=10)
        service = ResamplerService(grid_spec=grid_spec)
        assert hasattr(service, "cache")
        assert service.cache is not None


class TestResamplePlanCache:
    """Test ResamplePlanCache implementation."""

    def test_cache_instantiation(self):
        """Test creating cache."""
        cache = ResamplePlanCache()
        assert cache is not None
        assert hasattr(cache, "_store")

    def test_cache_default_maxsize(self):
        """Test cache has default maxsize."""
        cache = ResamplePlanCache()
        assert hasattr(cache, "maxsize")
        assert cache.maxsize >= 1

    def test_cache_custom_maxsize(self):
        """Test cache with custom maxsize."""
        cache = ResamplePlanCache(maxsize=32)
        assert cache.maxsize == 32

    def test_cache_hash_velocity(self):
        """Test velocity hashing."""
        cache = ResamplePlanCache()
        vp1 = np.full((2, 2, 5), 3500.0)
        vp2 = np.full((2, 2, 5), 3500.0)
        vp3 = np.full((2, 2, 5), 4000.0)

        hash1 = cache._hash_vp(vp1)
        hash2 = cache._hash_vp(vp2)
        hash3 = cache._hash_vp(vp3)

        assert hash1 == hash2
        assert hash1 != hash3
        assert isinstance(hash1, str)

    def test_cache_make_key(self):
        """Test key creation."""
        cache = ResamplePlanCache()
        grid_spec = GridSpec.from_dimensions(nx=2, ny=2, nz=5)
        vp = np.full((2, 2, 5), 3500.0)
        key = cache._make_key(grid_spec, vp, None, None)
        assert key is not None

    def test_cache_key_consistency(self):
        """Test key consistency."""
        cache = ResamplePlanCache()
        grid_spec = GridSpec.from_dimensions(nx=2, ny=2, nz=5)
        vp = np.full((2, 2, 5), 3500.0)

        key1 = cache._make_key(grid_spec, vp, None, None)
        key2 = cache._make_key(grid_spec, vp, None, None)
        assert key1 == key2

    def test_cache_key_differs_with_dt(self):
        """Test keys differ with different target_dt."""
        cache = ResamplePlanCache()
        grid_spec = GridSpec.from_dimensions(nx=2, ny=2, nz=5)
        vp = np.full((2, 2, 5), 3500.0)

        key1 = cache._make_key(grid_spec, vp, target_dt=0.001, target_nt=None)
        key2 = cache._make_key(grid_spec, vp, target_dt=0.002, target_nt=None)
        assert key1 != key2

    def test_cache_key_differs_with_vp(self):
        """Test keys differ with different velocity."""
        cache = ResamplePlanCache()
        grid_spec = GridSpec.from_dimensions(nx=2, ny=2, nz=5)
        vp1 = np.full((2, 2, 5), 3500.0)
        vp2 = np.full((2, 2, 5), 4000.0)

        key1 = cache._make_key(grid_spec, vp1, None, None)
        key2 = cache._make_key(grid_spec, vp2, None, None)
        assert key1 != key2


class TestCacheGlobalFunctions:
    """Test cache module-level functions."""

    def test_get_resample_plan_cache(self):
        """Test getting default cache."""
        cache = get_resample_plan_cache()
        assert cache is not None
        assert isinstance(cache, ResamplePlanCache)

    def test_get_cache_returns_singleton(self):
        """Test cache is singleton."""
        cache1 = get_resample_plan_cache()
        cache2 = get_resample_plan_cache()
        assert cache1 is cache2

    def test_set_resample_plan_cache(self):
        """Test setting custom cache."""
        new_cache = ResamplePlanCache(maxsize=8)
        set_resample_plan_cache(new_cache)
        retrieved = get_resample_plan_cache()
        # Should return the new cache
        assert retrieved is new_cache


class TestBackendResult:
    """Test BackendResult dataclass."""

    def test_result_with_array_only(self):
        """Test result with only array."""
        arr = np.zeros((2, 2, 10))
        result = BackendResult(array=arr)
        assert result.array is arr
        assert result.dt is None

    def test_result_with_dt(self):
        """Test result with dt parameter."""
        arr = np.zeros((2, 2, 10))
        result = BackendResult(array=arr, dt=0.004)
        assert result.array is arr
        assert result.dt == 0.004

    def test_result_preserves_array_properties(self):
        """Test result preserves array shape and dtype."""
        arr = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
        result = BackendResult(array=arr)
        assert result.array.shape == (1, 2, 2)
        assert result.array.dtype == np.float32

    def test_result_with_int_array(self):
        """Test result with integer array."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = BackendResult(array=arr, dt=None)
        assert result.array.dtype == np.int32


class TestBackendError:
    """Test BackendError exception."""

    def test_error_instantiation(self):
        """Test creating BackendError."""
        error = BackendError("test error")
        assert str(error) == "test error"

    def test_error_is_exception(self):
        """Test BackendError is Exception subclass."""
        error = BackendError("msg")
        assert isinstance(error, Exception)

    def test_error_with_message(self):
        """Test error message preservation."""
        msg = "Custom backend error message"
        error = BackendError(msg)
        assert msg in str(error)


class TestBackendIntegration:
    """Integration tests for backends."""

    def test_vectorized_and_batched_backends_exist(self):
        """Test both backend implementations exist."""
        v_backend = VectorizedBackend()
        b_backend = BatchedInterpolatorBackend()
        assert v_backend.name != b_backend.name
        assert v_backend.name == "vectorized"
        assert b_backend.name == "batched_interpolator"

    def test_backends_implement_support_method(self):
        """Test all backends implement support method."""
        backends = [VectorizedBackend(), BatchedInterpolatorBackend()]
        plan = MagicMock()
        plan.uniform_twt = True

        for backend in backends:
            result = backend.supports(plan)
            assert isinstance(result, (bool, type(None)))

    def test_manager_can_register_multiple_backends(self):
        """Test manager can register multiple backends."""
        manager = BackendManager()
        backend1 = VectorizedBackend()
        backend2 = BatchedInterpolatorBackend()

        manager.register(backend1.name, backend1)
        manager.register(backend2.name, backend2)
        # Should not raise


class TestResamplerServiceGridSpecs:
    """Test ResamplerService with various grid specs."""

    def test_service_with_minimal_grid(self):
        """Test service with minimal grid."""
        spec = GridSpec.from_dimensions(nx=1, ny=1, nz=1)
        service = ResamplerService(grid_spec=spec)
        assert service.grid_spec.nx == 1
        assert service.grid_spec.ny == 1
        assert service.grid_spec.nz == 1

    def test_service_with_large_grid(self):
        """Test service with larger grid."""
        spec = GridSpec.from_dimensions(nx=100, ny=100, nz=500)
        service = ResamplerService(grid_spec=spec)
        assert service.grid_spec.nx == 100
        assert service.grid_spec.ny == 100
        assert service.grid_spec.nz == 500

    def test_service_with_custom_spacing(self):
        """Test service with custom spacing."""
        spec = GridSpec.from_dimensions(nx=10, ny=10, nz=50, dz=2.5, dt=0.002)
        service = ResamplerService(grid_spec=spec)
        assert service.grid_spec.dz == 2.5
        assert service.grid_spec.dt == 0.002
