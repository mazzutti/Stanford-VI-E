from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.processing._backend_base import BackendError, ResamplerBackend, BackendResult
from src.processing.resample_plan import ResamplePlan
from src.processing.backend_manager import (
    get_backend_manager,
    register_backend as manager_register_backend,
    list_backends as manager_list_backends,
    get_best_backend as manager_get_best_backend,
    set_backend_verbose as manager_set_backend_verbose,
    is_backend_verbose as manager_is_backend_verbose,
)

try:
    from src.processing.interpolator import BatchedInterpolator
except Exception:  # pragma: no cover - optional import
    BatchedInterpolator = None


_REGISTRY: Dict[str, ResamplerBackend] = {}

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "BackendsRegistry",
    "backends_registry",
    "get_backends_registry",
]


# Thin OO facade for backend registry
class BackendsRegistry:
    """Facade around the backend manager helpers.

    This thin registry delegates to the manager-level helpers imported above
    (which in turn forward to the BackendManager singleton). Providing an
    object facade makes it easier to inject or mock registry behavior in
    tests or higher-level code.
    """

    def register_backend(self, name: str, impl: ResamplerBackend) -> None:
        manager_register_backend(name, impl)

    def get_backend(self, name: str) -> ResamplerBackend:
        # Use the BackendManager singleton to retrieve the backend by name
        return get_backend_manager().get(name)

    def list_backends(self) -> List[str]:
        return manager_list_backends()

    def get_best_backend(self, plan: ResamplePlan) -> Optional[ResamplerBackend]:
        return manager_get_best_backend(plan)

    def set_backend_verbose(self, on: bool) -> None:
        manager_set_backend_verbose(on)

    def is_backend_verbose(self) -> bool:
        return manager_is_backend_verbose()


# Module-level lazy proxy for BackendsRegistry
from src.utils.facades import LazyObjectProxy


# Module-level lazy proxy using shared LazyObjectProxy
backends_registry = LazyObjectProxy(lambda: BackendsRegistry())
__all__.extend(["BackendsRegistry", "backends_registry"])


def get_backends_registry(config: dict | None = None):
    if config is None:
        return backends_registry
    return BackendsRegistry()


__all__.append("get_backends_registry")


# The module now exposes the canonical BackendsRegistry facade and its
# module-level lazy proxy. Callers should access `backends_registry` or
# request a configured registry via `get_backends_registry`.


class VectorizedBackend:
    """Backend that handles uniform TWT (fast vectorized path).

    This backend will only advertise support when ResamplePlan indicates a
    uniform time axis for all traces.
    """

    name = "vectorized"

    def supports(self, plan: ResamplePlan) -> bool:
        return plan.uniform_twt

    def depth_to_time(
        self, data: np.ndarray, vp: np.ndarray, plan: ResamplePlan, **kwargs
    ) -> BackendResult:
        # For uniform_twt, we can call the resampler's fast path directly.
        from src.processing.resampler import resampler_factory

        resampler = resampler_factory.get_resampler(plan.grid_spec)
        out, dt = resampler.depth_to_time_cube(data, vp, plan=plan)
        return BackendResult(array=out, dt=dt)

    def time_to_depth(
        self, data: np.ndarray, vp: np.ndarray, plan: ResamplePlan, **kwargs
    ) -> BackendResult:
        from src.processing.resampler import resampler_factory

        resampler = resampler_factory.get_resampler(plan.grid_spec)
        out = resampler.time_to_depth_cube(data, vp, plan=plan)
        # time_to_depth returns array or Quantity; ensure we pass raw array
        if hasattr(out, "array"):
            arr = out.array
        else:
            arr = out
        return BackendResult(array=arr)


class BatchedInterpolatorBackend:
    """Backend that uses the BatchedInterpolator (CPU fallback).

    Advertises support for non-uniform plans when BatchedInterpolator is
    available.
    """

    name = "batched_interpolator"

    def supports(self, plan: ResamplePlan) -> bool:
        return BatchedInterpolator is not None

    def depth_to_time(
        self, data: np.ndarray, vp: np.ndarray, plan: ResamplePlan, **kwargs
    ) -> BackendResult:
        if BatchedInterpolator is None:
            raise BackendError("BatchedInterpolator not available")
        # Prepare padded arrays and delegate
        twt_padded = plan.twt_padded()
        depth_padded_flat = plan.prepare_depth_padded_flat(data)
        bi = BatchedInterpolator()

        out = bi.interpolate(twt_padded, depth_padded_flat)
        # BatchedInterpolator returns shape (nt, ntr) -> reshape to (ni,nj,nt)
        ni, nj, nz = plan.ni, plan.nj, plan.nz
        nt = plan.nt
        out_arr = out.reshape(nt, ni, nj).transpose(1, 2, 0)
        return BackendResult(array=out_arr, dt=plan.dt)

    def time_to_depth(
        self, data: np.ndarray, vp: np.ndarray, plan: ResamplePlan, **kwargs
    ) -> BackendResult:
        # Not implemented in this simple backend; fall back to resampler
        from src.processing.resampler import resampler_factory

        resampler = resampler_factory.get_resampler(plan.grid_spec)
        out = resampler.time_to_depth_cube(data, vp, plan=plan)
        if hasattr(out, "array"):
            arr = out.array
        else:
            arr = out
        return BackendResult(array=arr)


# Register default backends
try:
    backends_registry.register_backend(VectorizedBackend.name, VectorizedBackend())
except Exception:
    pass

try:
    backends_registry.register_backend(
        BatchedInterpolatorBackend.name, BatchedInterpolatorBackend()
    )
except Exception:
    pass
