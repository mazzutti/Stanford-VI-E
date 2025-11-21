"""Concrete resampling backend implementations.

Provides a couple of default backend implementations (vectorized and
batched interpolator) and a helper to register them into the
application's backend manager. Some backends optionally import heavy
helpers at runtime (e.g. `BatchedInterpolator`) and therefore perform
deferred imports.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.processing.resampling._plan import ResamplePlan
from src.processing.resampling.backends._base import BackendError, BackendResult
from src.processing.resampling.backends._manager import BackendManager
from src.utils.quantity import Quantity

# Try to import BatchedInterpolator at runtime
try:
    from src.processing.interpolator import BatchedInterpolator as _BatchedInterpolator

    # Name is intentionally snake_case and holds a runtime class reference.
    # Place the pylint disable on the assignment line so pylint recognises
    # this is an allowed exception to the naming convention.

    batched_interpolator_runtime: type | None = _BatchedInterpolator
    # pylint: enable=invalid-name
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional import
    batched_interpolator_runtime = None

# Backwards-compatible alias: some tests expect `BatchedInterpolator_runtime`
# (historic casing). Provide an alias so monkeypatch/patch calls succeed.
# Keep both names pointing to the same value.
BatchedInterpolator_runtime = batched_interpolator_runtime

logger = logging.getLogger(__name__)

# The backend implementations may import resampler factories or optional
# interpolator helpers at runtime. These imports are optional and are
# intentionally deferred; disable import-order warnings to reduce noise.

# Some runtime helper names intentionally use compact snake_case or
# non-Pascal identifiers for clarity in optional runtime bindings
# (e.g., `batched_interpolator_runtime`). Relax naming checks here.

class VectorizedBackend:
    """Backend that handles uniform TWT (fast vectorized path).

    This backend will only advertise support when ResamplePlan indicates a
    uniform time axis for all traces.
    """

    name = "vectorized"

    def supports(self, plan: ResamplePlan) -> bool:
        """Return True for plans that use a uniform two-way travel time."""
        return plan.uniform_twt

    def depth_to_time(
        self, data: NDArray[Any], vp: NDArray[Any], plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
        """Depth->time fast path for uniform TWT plans (vectorized)."""
        # For uniform_twt, we can call the resampler's fast path directly.
        from src.processing.resampling._resampler import (
            resampler_factory,
        )

        resampler = resampler_factory.get_resampler(plan.grid_spec)
        out, dt = resampler.depth_to_time_cube(data, vp, plan=plan)
        # Ensure we extract the array if it's a Quantity-like object with an
        # `.array` attribute and convert to an ndarray at runtime. Converting
        # with `np.asarray` both preserves runtime behavior and satisfies the
        # static type `NDArray[Any]` expected by `BackendResult`.
        arr_like = out.array if isinstance(out, Quantity) else out
        arr = np.asarray(arr_like)
        # Accept and ignore extra kwargs for Protocol compatibility.
        _ = kwargs
        return BackendResult(array=arr, dt=dt)

    def time_to_depth(
        self, data: NDArray[Any], vp: NDArray[Any], plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
        """Time->depth fast path for uniform TWT plans (vectorized)."""
        from src.processing.resampling._resampler import (
            resampler_factory,
        )

        resampler = resampler_factory.get_resampler(plan.grid_spec)
        out = resampler.time_to_depth_cube(data, vp, plan=plan)
        # time_to_depth returns array or Quantity; ensure we pass a numpy
        # ndarray into BackendResult by using `np.asarray`.
        arr_like = out.array if isinstance(out, Quantity) else out
        arr = np.asarray(arr_like)
        # Accept and ignore extra kwargs for Protocol compatibility.
        _ = kwargs
        return BackendResult(array=arr)

class BatchedInterpolatorBackend:
    """Backend that uses the BatchedInterpolator (CPU fallback).

    Advertises support for non-uniform plans when BatchedInterpolator is
    available.
    """

    name = "batched_interpolator"

    def supports(self, plan: ResamplePlan) -> bool:
        """Return True when the optional `BatchedInterpolator` runtime is available."""
        # `plan` parameter not required for this backend's availability check;
        # reference it to satisfy static analysis without changing behavior.
        _ = plan
        return BatchedInterpolator_runtime is not None

    def depth_to_time(
        self, data: NDArray[Any], vp: NDArray[Any], plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
        """Depth->time using batched interpolator fallback implementation."""
        # Prepare padded arrays and delegate
        twt_padded = plan.twt_padded()
        depth_padded_flat = plan.prepare_depth_padded_flat(data)
        if BatchedInterpolator_runtime is None:
            raise BackendError("BatchedInterpolator not available")
        # Instantiate with the target time axis from the plan so the
        # interpolator knows the time samples to interpolate onto.
        bi = BatchedInterpolator_runtime(plan.time_axis)

        out = bi.interpolate(twt_padded, depth_padded_flat)
        # BatchedInterpolator returns shape (nt, ntr) -> reshape to (ni,nj,nt)
        ni, nj = plan.ni, plan.nj
        nt = plan.nt
        out_arr = out.reshape(nt, ni, nj).transpose(1, 2, 0)
        # `vp` is accepted for API compatibility but not used by this backend;
        # reference to silence unused-argument warnings. Also accept and
        # ignore any extra kwargs to preserve Protocol compatibility.
        _ = vp
        _ = kwargs
        return BackendResult(array=out_arr, dt=plan.dt)

    def time_to_depth(
        self, data: NDArray[Any], vp: NDArray[Any], plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
        """Time->depth implementation that delegates to existing resampler."""
        from src.processing.resampling._resampler import (
            resampler_factory,
        )

        resampler = resampler_factory.get_resampler(plan.grid_spec)
        out = resampler.time_to_depth_cube(data, vp, plan=plan)
        arr_like = out.array if isinstance(out, Quantity) else out
        arr = np.asarray(arr_like)
        # No additional kwargs are consumed by this backend; accept and ignore
        # them to remain Protocol-compatible.
        _ = kwargs
        return BackendResult(array=arr)

# Register default backends

def register_default_backends(manager: BackendManager | None = None) -> None:
    """Register the default backend implementations.

    If `manager` is provided, register into that manager. Otherwise attempt
    to obtain the application's global backend manager via the
    `ServiceRegistry` (`src.processing.get_registry().get_backend_manager()`).

    This function is intentionally idempotent and safe to call multiple times.
    It must be invoked by application bootstrap instead of relying on
    import-time side-effects.
    """

    if manager is None:
        try:
            # Import lazily to avoid import-time cycles. Keep a focused inline
            # suppression for the call-time import so pylint C0415 is quiet.
            from src.processing import (
                get_registry,
            )

            manager = get_registry().get_backend_manager()
        except (ImportError, ModuleNotFoundError):
            # If the registry module isn't available, fall back to a local
            # BackendManager instance as a best-effort (no global side-effects).
            manager = BackendManager()

    try:
        manager.register(VectorizedBackend.name, VectorizedBackend())
    except (KeyError, RuntimeError, TypeError):
        pass

    try:
        manager.register(BatchedInterpolatorBackend.name, BatchedInterpolatorBackend())
    except (KeyError, RuntimeError, TypeError):
        pass

# Backwards-compatible alias used by older code/tests that imported the
# internal name `_register_default_backends`.
_register_default_backends = register_default_backends
