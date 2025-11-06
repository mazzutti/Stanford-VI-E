from __future__ import annotations

from typing import Any, Optional, Union, Type
import numpy as np

import logging

from src.processing.resampling.backends._base import (
    BackendError,
    BackendResult,
)
from src.processing.resampling._plan import ResamplePlan
from src.processing.resampling.backends._manager import BackendManager


try:
    from src.processing.interpolator import BatchedInterpolator as _BatchedInterpolator
    BatchedInterpolator: Optional[Type[Any]] = _BatchedInterpolator
except Exception:  # pragma: no cover - optional import
    BatchedInterpolator = None


logger = logging.getLogger(__name__)


class VectorizedBackend:
    """Backend that handles uniform TWT (fast vectorized path).

    This backend will only advertise support when ResamplePlan indicates a
    uniform time axis for all traces.
    """

    name = "vectorized"

    def supports(self, plan: ResamplePlan) -> bool:
        return plan.uniform_twt

    def depth_to_time(
        self, data: np.ndarray, vp: np.ndarray, plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
        # For uniform_twt, we can call the resampler's fast path directly.
        from src.processing.resampling._resampler import resampler_factory

        resampler = resampler_factory.get_resampler(plan.grid_spec)
        out, dt = resampler.depth_to_time_cube(data, vp, plan=plan)
        return BackendResult(array=out, dt=dt)

    def time_to_depth(
        self, data: np.ndarray, vp: np.ndarray, plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
        from src.processing.resampling._resampler import resampler_factory

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
        self, data: np.ndarray, vp: np.ndarray, plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
        if BatchedInterpolator is None:
            raise BackendError("BatchedInterpolator not available")
        # Prepare padded arrays and delegate
        twt_padded = plan.twt_padded()
        depth_padded_flat = plan.prepare_depth_padded_flat(data)
        bi = BatchedInterpolator()

        out = bi.interpolate(twt_padded, depth_padded_flat)
        # BatchedInterpolator returns shape (nt, ntr) -> reshape to (ni,nj,nt)
        ni, nj = plan.ni, plan.nj
        nt = plan.nt
        out_arr = out.reshape(nt, ni, nj).transpose(1, 2, 0)
        return BackendResult(array=out_arr, dt=plan.dt)

    def time_to_depth(
        self, data: np.ndarray, vp: np.ndarray, plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
        # Not implemented in this simple backend; fall back to resampler
        from src.processing.resampling._resampler import resampler_factory

        resampler = resampler_factory.get_resampler(plan.grid_spec)
        out = resampler.time_to_depth_cube(data, vp, plan=plan)
        if hasattr(out, "array"):
            arr = out.array
        else:
            arr = out
        return BackendResult(array=arr)


# Register default backends
def _register_default_backends() -> None:
    """Register the default backend implementations with the global BackendManager."""
    manager = BackendManager()

    try:
        manager.register(VectorizedBackend.name, VectorizedBackend())
    except Exception:
        pass

    try:
        manager.register(BatchedInterpolatorBackend.name, BatchedInterpolatorBackend())
    except Exception:
        pass


_register_default_backends()
