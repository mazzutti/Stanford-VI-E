from __future__ import annotations


from typing import Optional, TYPE_CHECKING, Any


from numpy.typing import NDArray
import logging


from src.processing.resampling.backends._base import (
    BackendResult,
)
from src.processing.resampling._plan import ResamplePlan
from src.processing.resampling.backends._manager import BackendManager
from src.utils.quantity import Quantity


if TYPE_CHECKING:
    pass


# Try to import BatchedInterpolator at runtime
try:
    from src.processing.interpolator import BatchedInterpolator as _BatchedInterpolator

    BatchedInterpolator_runtime: Optional[type] = _BatchedInterpolator
except Exception:  # pragma: no cover - optional import
    BatchedInterpolator_runtime = None


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
        self, data: NDArray[Any], vp: NDArray[Any], plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
        # For uniform_twt, we can call the resampler's fast path directly.
        from src.processing.resampling._resampler import resampler_factory

        resampler = resampler_factory.get_resampler(plan.grid_spec)
        out, dt = resampler.depth_to_time_cube(data, vp, plan=plan)
        # Ensure we extract the array if it's a Quantity
        if isinstance(out, Quantity):
            arr = out.array
        else:
            arr = out
        return BackendResult(array=arr, dt=dt)

    def time_to_depth(
        self, data: NDArray[Any], vp: NDArray[Any], plan: ResamplePlan, **kwargs: Any
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
        return BatchedInterpolator_runtime is not None

    def depth_to_time(
        self, data: NDArray[Any], vp: NDArray[Any], plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
        # Prepare padded arrays and delegate
        twt_padded = plan.twt_padded()
        depth_padded_flat = plan.prepare_depth_padded_flat(data)
        if BatchedInterpolator_runtime is None:
            raise RuntimeError("BatchedInterpolator not available")
        bi = BatchedInterpolator_runtime()

        out = bi.interpolate(twt_padded, depth_padded_flat)
        # BatchedInterpolator returns shape (nt, ntr) -> reshape to (ni,nj,nt)
        ni, nj = plan.ni, plan.nj
        nt = plan.nt
        out_arr = out.reshape(nt, ni, nj).transpose(1, 2, 0)
        return BackendResult(array=out_arr, dt=plan.dt)

    def time_to_depth(
        self, data: NDArray[Any], vp: NDArray[Any], plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
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
