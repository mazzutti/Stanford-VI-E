"""Depth <-> Time resampling utilities.

Provides a testable Resampler that centralizes depth/time conversions
using a `GridSpec` object.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from src.analysis.decorators import log_execution, time_operation
from src.io.grid import GridSpec
from src.processing.interpolator import BatchedInterpolator
from src.processing.resampling._interpolation import linear_interpolate_value
from src.processing.resampling._plan import ResamplePlan
from src.processing.resampling.backends._manager import BackendManager
from src.utils.quantity import Quantity, to_ndarray
from src.utils.units import UnitRegistry

__all__ = ["DepthTimeResampler", "set_backend_verbose", "is_backend_verbose"]

# Module logger
logger = logging.getLogger(__name__)

# Many functions in this module are performance-sensitive resampling
# routines and intentionally contain large argument lists, branching
# and local variables. Disable related pylint complexity checks here
# with a focused justification so that other genuine issues remain
# visible in the codebase.

# Module-level lazy proxies are defined later in this file

# Enable extra backend debug logging if the environment flag is set.
# Use a mutable container to avoid `global` in setter function.
_backend_state: dict[str, bool] = {
    "verbose": os.environ.get("RESAMPLE_BACKEND_VERBOSE", "0").lower()
    in (
        "1",
        "true",
        "yes",
    )
}
if _backend_state.get("verbose"):
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger(__name__).debug(
        "RESAMPLE_BACKEND_VERBOSE enabled: backend debug logs ON"
    )

# Numba-optimized generic resampling kernels. These are defined at
# module level to avoid nesting compiled functions inside methods which
# increases function complexity and interferes static analysis.
# Numba kernels are naturally verbose; allow larger local-variable counts.

@njit(parallel=True)
def _nearest_resample_numba_jitted(
    twt_ir: NDArray[Any],
    data: NDArray[Any],
    t_axis: NDArray[Any],
    out_arr: NDArray[Any],
) -> None:
    """Numba-jitted nearest-neighbor resampling kernel.

    This performs an in-place nearest-neighbor resample of `data`
    from irregular two-way travel-time traces `twt_ir` onto the
    regular time axis `t_axis`, writing results into `out_arr`.
    """
    ni_, nj_, nz_ = data.shape
    nt_ = t_axis.shape[0]
    for ii in prange(ni_):
        for jj in range(nj_):
            twt = twt_ir[ii, jj]
            prop = data[ii, jj]
            k = 0
            for ti in range(nt_):
                t = t_axis[ti]
                while k < nz_ and twt[k] <= t:
                    k += 1
                if k == 0:
                    out_arr[ii, jj, ti] = prop[0]
                elif k >= nz_:
                    out_arr[ii, jj, ti] = prop[nz_ - 1]
                else:
                    if abs(t - twt[k - 1]) <= abs(twt[k] - t):
                        out_arr[ii, jj, ti] = prop[k - 1]
                    else:
                        out_arr[ii, jj, ti] = prop[k]

# Numba kernels are naturally verbose; allow larger local-variable counts.

@njit(parallel=True)
def _linear_resample_numba_jitted(
    twt_ir: NDArray[Any],
    data: NDArray[Any],
    t_axis: NDArray[Any],
    out_arr: NDArray[Any],
) -> None:
    """Numba-jitted linear interpolation resampling kernel.

    Performs an in-place linear interpolation of `data` using
    irregular TWT traces `twt_ir` onto `t_axis`, storing results
    in `out_arr`.
    """
    ni_, nj_, nz_ = data.shape
    nt_ = t_axis.shape[0]
    for ii in prange(ni_):
        for jj in range(nj_):
            twt = twt_ir[ii, jj]
            trace = data[ii, jj]
            k = 0
            for ti in range(nt_):
                t = t_axis[ti]
                while k < nz_ and twt[k] <= t:
                    k += 1
                if k == 0:
                    out_arr[ii, jj, ti] = trace[0]
                elif k >= nz_:
                    out_arr[ii, jj, ti] = trace[nz_ - 1]
                else:
                    t0 = twt[k - 1]
                    t1 = twt[k]
                    v0 = trace[k - 1]
                    v1 = trace[k]
                    out_arr[ii, jj, ti] = linear_interpolate_value(t, t0, t1, v0, v1)

def set_backend_verbose(on: bool) -> None:
    """Programmatically enable/disable backend verbose logging for this module.

    When enabled the module logger is set to DEBUG and additional plan
    metadata is emitted when a backend is selected.
    """
    _backend_state["verbose"] = bool(on)
    if _backend_state["verbose"]:
        logger.setLevel(logging.DEBUG)
        logger.debug("Backend verbose logging enabled via set_backend_verbose(True)")
    else:
        # Revert to INFO level by default
        logger.setLevel(logging.INFO)
        logger.info("Backend verbose logging disabled via set_backend_verbose(False)")

def is_backend_verbose() -> bool:
    """Return whether backend verbose logging is enabled for this module."""
    return bool(_backend_state.get("verbose"))

@dataclass
class DepthTimeResampler:
    """Utility for converting between depth-sampled cubes and time-sampled cubes.

    Methods accept 3D arrays with shape (ni, nj, nk) for depth or (ni, nj, nt)
    for time. The resampler uses per-trace velocity (vp) in depth domain to
    compute two-way travel time (TWT) traces per column and performs
    interpolation using SciPy's interp1d.
    """

    grid_spec: GridSpec
    backend_manager: BackendManager | None = None

    def compute_one_way_time(self, vp_trace: NDArray[Any] | Quantity) -> NDArray[Any]:
        """Compute one-way cumulative time for a single vertical trace.

        Args:
            vp_trace: 1D array of P-wave velocities in depth (m/s), length nz

        Returns:
            1D array of one-way travel times at each depth sample (seconds)
        """
        # Accept Quantity-wrapped trace
        if isinstance(vp_trace, Quantity):
            vp_val = vp_trace.array
        else:
            vp_val = np.asarray(vp_trace)

        if vp_val.ndim != 1:
            raise ValueError("vp_trace must be 1D")

        # Ensure vp_trace is in m/s
        vp_converted, _ = UnitRegistry.ensure_m_per_s(vp_val, copy_on_convert=True)
        vp_arr = np.asarray(vp_converted, dtype=float)

        if np.any(vp_arr <= 0) or np.isnan(vp_arr).any():
            raise ValueError("vp_trace contains non-positive or NaN values")

        # Ensure dz is in meters (GridSpec.dz is expected in meters). If grid_spec.dz
        # appears to be small (e.g., < 0.1) treat as kilometers and convert.
        dz_val, _ = UnitRegistry.ensure_meters(self.grid_spec.dz)

        slowness = 1.0 / vp_arr
        one_way = np.cumsum(slowness * dz_val)
        return one_way

    @log_execution
    @time_operation("compute_one_way_times", threshold_ms=100)
    def compute_one_way_times(self, vp_arr: NDArray[Any] | Quantity) -> NDArray[Any]:
        """Vectorized computation of one-way cumulative time for all traces.

        Args:
            vp_arr: 3D array of shape (ni, nj, nz) or a Quantity wrapping that

        Returns:
            3D array (ni, nj, nz) of one-way cumulative time (seconds)
        """
        # Normalize Quantity/ndarray to ndarray
        vp_val = to_ndarray(vp_arr)

        if vp_val.ndim != 3:
            raise ValueError("vp_arr must be a 3D array (ni, nj, nz)")

        # Ensure vp_arr is in m/s
        vp_converted, _ = UnitRegistry.ensure_m_per_s(vp_val, copy_on_convert=True)
        vp_float = np.asarray(vp_converted, dtype=float)

        if np.any(vp_float <= 0) or np.isnan(vp_float).any():
            raise ValueError("vp_arr contains non-positive or NaN values")

        dz_val, _ = UnitRegistry.ensure_meters(self.grid_spec.dz)

        slowness = 1.0 / vp_float
        # cumulative sum along vertical axis (nz)
        one_way = np.cumsum(slowness * dz_val, axis=2)
        return one_way

    @log_execution
    @time_operation("depth_to_time_cube", threshold_ms=500)
    def depth_to_time_cube(
        self,
        data_depth: NDArray[Any] | Quantity,
        vp_depth: NDArray[Any] | Quantity,
        target_dt: float | None = None,
        target_nt: int | None = None,
        plan: ResamplePlan | None = None,
    ) -> tuple[NDArray[Any] | Quantity, float]:
        """Resample depth-sampled `data_depth` into a regularly sampled time cube.

        Args:
            data_depth: (ni, nj, nz) depth-sampled property (float or int)
            vp_depth: (ni, nj, nz) P-wave velocity in m/s
            target_dt: optional time sampling interval; if None uses grid_spec.dt

        Returns:
        time_axis: NDArray[Any] = cast(NDArray[Any], plan.time_axis)
        """
        # Normalize inputs and remember original units when present
        if isinstance(data_depth, Quantity):
            data_unit = data_depth.unit
            data_arr = to_ndarray(data_depth)
            data_was_quantity = True
        else:
            data_unit = None
            data_arr = to_ndarray(data_depth)
            data_was_quantity = False

        if isinstance(vp_depth, Quantity):
            vp_arr = to_ndarray(vp_depth)
        else:
            vp_arr = to_ndarray(vp_depth)

        if data_arr.shape != vp_arr.shape:
            raise ValueError("data_depth and vp_depth must have the same shape")

        ni, nj, _ = data_arr.shape
        # Use provided ResamplePlan if available to avoid recomputing.
        if plan is None:
            plan = ResamplePlan.create(
                self.grid_spec,
                vp_depth=vp_arr,
                target_dt=target_dt,
                target_nt=target_nt,
            )

        # Try delegated backend first (keeps legacy behavior and avoids
        # importing the global registry into this low-level module).
        backend_result = self._try_backend_depth_to_time(
            data_arr=data_arr,
            vp_arr=vp_arr,
            plan=plan,
            data_was_quantity=data_was_quantity,
            data_unit=data_unit,
        )
        if backend_result is not None:
            return backend_result
        # Annotate time_axis so the type checker knows this is an ndarray
        # (prevents "partially unknown" diagnostics when passed into
        # numba-jitted functions and BatchedInterpolator).
        time_axis: NDArray[Any] = plan.time_axis
        out = np.zeros((ni, nj, plan.nt), dtype=data_arr.dtype)

        # If numba is available and enabled via env, use a JIT-compiled linear
        # resampler that performs per-trace interpolation in parallel. This
        # avoids Python overhead for large grids. Nearest-neighbor categorical
        # resampling still uses the fallback interp1d for correctness.
        # Numba is a required dependency, so use optimized compiled path
        use_numba = os.environ.get("RESAMPLE_USE_NUMBA", "1") == "1"

        # Fast vectorized path: if all TWT traces are identical across the
        # spatial grid (common when using a flat layered model), we can
        # perform a single broadcasted interpolation using SciPy's interp1d
        # with axis=0 which is implemented in C and is significantly faster
        # than per-trace Python loops.
        # twt/depth padded arrays are handled in `_handle_uniform_twt`
        # Handle uniform-TWT fast path via helper to reduce method complexity.
        if plan.uniform_twt:
            res = self._handle_uniform_twt(
                data_arr=data_arr,
                time_axis=time_axis,
                plan=plan,
                data_was_quantity=data_was_quantity,
                data_unit=data_unit,
            )
            if res is not None:
                return res

        if use_numba:
            # Use module-level jitted kernels for performance and to keep
            # the method body small.
            if np.issubdtype(data_arr.dtype, np.integer):
                _nearest_resample_numba_jitted(
                    plan.one_way * 2.0, data_arr, time_axis, out
                )
            else:
                _linear_resample_numba_jitted(
                    plan.one_way * 2.0, data_arr, time_axis, out
                )
        else:
            out = self._depth_to_time_fallback(
                data_arr=data_arr, time_axis=time_axis, plan=plan, out=out
            )

        # Wrap output as Quantity if data_depth was Quantity
        if data_was_quantity:
            return Quantity(out, cast(str, data_unit)), plan.dt
        return out, plan.dt

    def _depth_to_time_fallback(
        self,
        data_arr: NDArray[Any],
        time_axis: NDArray[Any],
        plan: ResamplePlan,
        out: NDArray[Any],
    ) -> NDArray[Any]:
        """Fallback non-numba resampling path extracted from
        `depth_to_time_cube` to reduce method complexity.

        This implements the blocked processing strategy that uses
        `BatchedInterpolator` for homogeneous blocks and `interp1d`
        for mixed blocks.
        """
        # compute sizes locally to avoid burdening the caller with many
        # temporaries (keeps caller simpler and reduces local variable counts)
        ni, nj, nz = data_arr.shape
        nt = plan.nt
        ntr = ni * nj
        # flatten spatial dims -> traces as columns
        data_flat = data_arr.transpose(2, 0, 1).reshape(nz, -1)
        twt_flat = (2.0 * plan.one_way).transpose(2, 0, 1).reshape(nz, -1)

        interp = BatchedInterpolator(time_axis=time_axis)
        bs = interp.block_size
        for start in range(0, ntr, bs):
            end = min(start + bs, ntr)
            self._process_resample_block(
                data_flat=data_flat,
                twt_flat=twt_flat,
                start=start,
                end=end,
                nz=nz,
                nt=nt,
                time_axis=time_axis,
                out=out,
                interp=interp,
                data_dtype=data_arr.dtype,
            )

        return out

    def _try_backend_depth_to_time(
        self,
        data_arr: NDArray[Any],
        vp_arr: NDArray[Any],
        plan: ResamplePlan,
        data_was_quantity: bool,
        data_unit: Any,
    ) -> tuple[NDArray[Any] | Quantity, float] | None:
        """Attempt to run a delegated backend resampler, returning the
        `(out, dt)` tuple when a backend is available and succeeded, or
        `None` when no backend was selected.
        """
        if self.backend_manager is not None:
            backend = self.backend_manager.get_best(plan)
        else:
            backend = None
        if backend is None:
            return None

        logger.info(
            "Depth->Time: using backend '%s' (uniform_twt=%s)",
            getattr(backend, "name", repr(backend)),
            getattr(plan, "uniform_twt", None),
        )
        logger.debug(
            "Depth->Time: plan summary dt=%s nt=%s ni=%s nj=%s nz=%s",
            getattr(plan, "dt", None),
            getattr(plan, "nt", None),
            getattr(plan, "ni", None),
            getattr(plan, "nj", None),
            getattr(plan, "nz", None),
        )
        try:
            out_backend = backend.depth_to_time(data_arr, vp_arr, plan=plan)
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
            logger.exception(
                "Depth->Time: backend '%s' raised an exception",
                getattr(backend, "name", repr(backend)),
            )
            raise

        out = out_backend.array
        dt = out_backend.dt if out_backend.dt is not None else plan.dt
        if data_was_quantity:
            return Quantity(out, cast(str, data_unit)), dt
        return out, dt

    def _handle_uniform_twt(
        self,
        data_arr: NDArray[Any],
        time_axis: NDArray[Any],
        plan: ResamplePlan,
        data_was_quantity: bool,
        data_unit: Any,
    ) -> tuple[NDArray[Any] | Quantity, float] | None:
        """Handle the uniform-TWT fast path used by `depth_to_time_cube`.

        Returns a tuple `(out, dt)` (with `out` possibly a `Quantity`) when
        handled, otherwise `None`.
        """
        twt_padded = plan.twt_padded()
        depth_padded_flat = plan.prepare_depth_padded_flat(data_arr)

        interp = BatchedInterpolator(time_axis=time_axis, kind="linear")
        if np.issubdtype(data_arr.dtype, np.integer):
            res_vec = interp.nearest(twt_padded, depth_padded_flat)
        else:
            res_vec = interp.interpolate(twt_padded, depth_padded_flat)

        ni, nj, _ = data_arr.shape
        nt = plan.nt
        out = res_vec.reshape(nt, ni, nj).transpose(1, 2, 0)
        dt = plan.dt
        if data_was_quantity:
            return Quantity(out, cast(str, data_unit)), dt
        return out, dt

    def _process_resample_block(
        self,
        data_flat: NDArray[Any],
        twt_flat: NDArray[Any],
        start: int,
        end: int,
        nz: int,
        nt: int,
        time_axis: NDArray[Any],
        out: NDArray[Any],
        interp: BatchedInterpolator,
        data_dtype: Any,
    ) -> None:
        """Process a single block for the non-numba resampler fallback.

        Extracted from `_depth_to_time_fallback` to reduce local
        variable counts in that method.
        """
        data_block = data_flat[:, start:end]  # shape (nz, nblock)
        twt_block = twt_flat[:, start:end]  # shape (nz, nblock)

        # Check if all columns in this block have identical twt
        # Compare to first column
        if np.allclose(twt_block, np.broadcast_to(twt_block[:, 0:1], twt_block.shape)):
            twt_padded, depth_padded_flat = self._build_padded_for_block(
                data_block=data_block, twt_block=twt_block, nz=nz
            )
            if np.issubdtype(data_dtype, np.integer):
                res = interp.nearest(twt_padded, depth_padded_flat)
            else:
                res = interp.interpolate(twt_padded, depth_padded_flat)
            # res shape (nt, nblock)
            out.reshape(nt, -1)[:, start:end] = res
        else:
            # Mixed twt in block: fall back to per-trace interp1d for this block
            self._process_resample_block_per_trace(
                twt_block=twt_block,
                data_block=data_block,
                start=start,
                end=end,
                nt=nt,
                time_axis=time_axis,
                out=out,
                data_dtype=data_dtype,
            )

    def _build_padded_for_block(
        self, data_block: NDArray[Any], twt_block: NDArray[Any], nz: int
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Build padded twt and depth arrays for a homogeneous block.

        Returns (twt_padded, depth_padded_flat) where `twt_padded` has a
        leading 0.0 and `depth_padded_flat` is shaped (nz+1, nblock).
        """
        twt_padded = np.concatenate([[0.0], twt_block[:, 0]])
        depth_padded = np.concatenate([data_block[0:1, :], data_block], axis=0)
        depth_padded_flat = depth_padded.reshape(nz + 1, -1)
        return twt_padded, depth_padded_flat

    def _process_resample_block_per_trace(
        self,
        twt_block: NDArray[Any],
        data_block: NDArray[Any],
        start: int,
        end: int,
        nt: int,
        time_axis: NDArray[Any],
        out: NDArray[Any],
        data_dtype: Any,
    ) -> None:
        """Per-trace fallback processing for a block (uses interp1d).

        Pulled out to reduce local variable counts in `_process_resample_block`.
        """
        for col in range(start, end):
            col_idx = col - start
            twt_trace = twt_block[:, col_idx]
            twt_padded = np.concatenate([[0.0], twt_trace])
            depth_trace = data_block[:, col_idx]
            depth_padded = np.concatenate([[depth_trace[0]], depth_trace])
            interp_fn = interp1d(
                twt_padded,
                depth_padded,
                kind=("nearest" if np.issubdtype(data_dtype, np.integer) else "linear"),
                bounds_error=False,
                fill_value=0.0,
            )
            out.reshape(nt, -1)[:, col] = interp_fn(time_axis)

    def time_to_depth_cube(
        self,
        seismogram_time: NDArray[Any] | Quantity,
        vp_depth: NDArray[Any] | Quantity,
        plan: ResamplePlan | None = None,
    ) -> NDArray[Any] | Quantity:
        """Convert a time-sampled seismogram to depth-sampled cube using vp_depth.

        Args:
            seismogram_time: (ni, nj, nt)
            vp_depth: (ni, nj, nz)

        Returns:
            seismogram_depth: (ni, nj, nz)
        """
        # Normalize inputs and remember units if present
        if isinstance(seismogram_time, Quantity):
            seis_unit = seismogram_time.unit
            seis_arr = to_ndarray(seismogram_time)
            seis_was_quantity = True
        else:
            seis_unit = None
            seis_arr = to_ndarray(seismogram_time)
            seis_was_quantity = False

        if isinstance(vp_depth, Quantity):
            vp_arr = to_ndarray(vp_depth)
        else:
            vp_arr = to_ndarray(vp_depth)

        ni_t, nj_t, nt = seis_arr.shape
        ni, nj, nz = vp_arr.shape
        if (ni_t, nj_t) != (ni, nj):
            raise ValueError(
                "spatial dimensions of seismogram_time and vp_depth must match"
            )

        dt = self.grid_spec.dt
        # Annotate and cast time_axis to NDArray[Any] to avoid Unknown union types
        time_axis: NDArray[Any] = cast(NDArray[Any], np.arange(nt) * dt)

        out = np.zeros((ni, nj, nz), dtype=seis_arr.dtype)

        # Use provided ResamplePlan if available to avoid recomputing TWT.
        if plan is None:
            plan = ResamplePlan.create(self.grid_spec, vp_arr)

        # Try delegated backend first to keep legacy hooking points.
        backend_result = self._try_backend_time_to_depth(
            seis_arr=seis_arr,
            vp_arr=vp_arr,
            plan=plan,
            seis_was_quantity=seis_was_quantity,
            seis_unit=seis_unit,
        )
        if backend_result is not None:
            return backend_result

        # TWT array used by both branches
        twt_arr = plan.twt_arr

        # Handle uniform twt fast path via helper to simplify this method
        uniform_res = self._handle_uniform_twt_time_to_depth(
            seis_arr=seis_arr, plan=plan, time_axis=time_axis
        )
        if uniform_res is not None:
            out = uniform_res
        else:
            self._time_to_depth_per_cell_time_to_depth(
                out=out,
                seis_arr=seis_arr,
                twt_arr=twt_arr,
                time_axis=time_axis,
            )
        if seis_was_quantity:
            return Quantity(out, cast(str, seis_unit))
        return out

    def _time_to_depth_per_cell_time_to_depth(
        self,
        out: NDArray[Any],
        seis_arr: NDArray[Any],
        twt_arr: NDArray[Any],
        time_axis: NDArray[Any],
    ) -> None:
        """Per-cell interpolation for `time_to_depth_cube` non-uniform path."""
        ni, nj, _ = out.shape
        for i in range(ni):
            for j in range(nj):
                twt_trace = twt_arr[i, j, :]
                out[i, j, :] = np.interp(
                    twt_trace, time_axis, seis_arr[i, j, :], left=0.0, right=0.0
                )

    def _try_backend_time_to_depth(
        self,
        seis_arr: NDArray[Any],
        vp_arr: NDArray[Any],
        plan: ResamplePlan,
        seis_was_quantity: bool,
        seis_unit: Any,
    ) -> NDArray[Any] | Quantity | None:
        """Attempt to delegate time->depth resampling to an external backend.

        Returns the resampled array (or Quantity) if a backend is selected
        and succeeded, otherwise `None`.
        """
        if self.backend_manager is not None:
            backend = self.backend_manager.get_best(plan)
        else:
            backend = None
        if backend is None:
            return None

        logger.info(
            "Time->Depth: using backend '%s' (uniform_twt=%s)",
            getattr(backend, "name", repr(backend)),
            getattr(plan, "uniform_twt", None),
        )
        logger.debug(
            "Time->Depth: plan summary dt=%s nt=%s ni=%s nj=%s nz=%s",
            getattr(plan, "dt", None),
            getattr(plan, "nt", None),
            getattr(plan, "ni", None),
            getattr(plan, "nj", None),
            getattr(plan, "nz", None),
        )
        try:
            out_backend = backend.time_to_depth(seis_arr, vp_arr, plan=plan)
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
            logger.exception(
                "Time->Depth: backend '%s' raised an exception",
                getattr(backend, "name", repr(backend)),
            )
            raise
        out = out_backend.array
        if seis_was_quantity:
            return Quantity(out, cast(str, seis_unit))
        return out

    def _handle_uniform_twt_time_to_depth(
        self, seis_arr: NDArray[Any], plan: ResamplePlan, time_axis: NDArray[Any]
    ) -> NDArray[Any] | None:
        """Fast path for uniform TWT in time->depth conversion.

        Returns the resampled `out` array when uniform TWT applies, or
        `None` otherwise.
        """
        twt_arr = plan.twt_arr
        if not plan.uniform_twt:
            return None

        # common twt positions for depth samples
        twt_common = twt_arr[0, 0, :]
        _, _, nt = seis_arr.shape
        seis_flat = seis_arr.transpose(2, 0, 1).reshape(nt, -1)
        interp = BatchedInterpolator(time_axis=twt_common, kind="linear")
        # if seis is integer/categorical, prefer nearest
        if np.issubdtype(seis_arr.dtype, np.integer):
            res = interp.nearest(time_axis, seis_flat)
        else:
            res = interp.interpolate(time_axis, seis_flat)

        ni, nj, nz = plan.ni, plan.nj, plan.nz
        # res shape (nz, ntr) -> reshape to (ni, nj, nz)
        return res.reshape(nz, ni, nj).transpose(1, 2, 0)

    def compute_twt_for_trace(
        self, vp_trace: NDArray[Any] | Quantity
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Return TWT (padded) and depth axis (padded) for a single trace.

        Returns twt_trace (with a leading 0) and a depth trace (also padded)
        so they can be passed to interpolation helpers.
        """
        dz = self.grid_spec.dz
        # Accept Quantity-wrapped vp_trace
        if isinstance(vp_trace, Quantity):
            vp_val = vp_trace.array
        else:
            vp_val = np.asarray(vp_trace)

        one_way = self.compute_one_way_time(vp_val)
        twt_trace = 2 * one_way
        twt_padded = np.concatenate([[0.0], twt_trace])
        depth_axis = np.arange(len(vp_val)) * dz
        depth_padded = np.concatenate([[0.0], depth_axis + dz])
        return twt_padded, depth_padded

    def resample_time_cube(
        self,
        data_time: NDArray[Any],
        src_time_axis: NDArray[Any],
        target_time_axis: NDArray[Any],
        kind: str = "linear",
        progress_every: int | None = 30,
        prefix: str = "",
    ) -> NDArray[Any]:
        """Resample a time-sampled cube from src_time_axis to target_time_axis.

        Mirrors the old utility function signature.
        """
        # Keep parameters for API compatibility; explicitly mark unused
        # to satisfy linters.
        del progress_every  # intentionally unused
        del prefix  # intentionally unused

        ni, nj, nt_src = data_time.shape
        nt_tgt = len(target_time_axis)
        out = np.zeros((ni, nj, nt_tgt), dtype=data_time.dtype)

        # Vectorized / batched interpolation across traces via BatchedInterpolator.
        # Flatten spatial dims so we interpolate all traces with a single
        # blocked/interpolated call which is much faster than nested Python loops.
        src_flat = data_time.transpose(2, 0, 1).reshape(nt_src, -1)

        interp = BatchedInterpolator(time_axis=target_time_axis, kind=kind)
        if kind == "nearest" or np.issubdtype(data_time.dtype, np.integer):
            res_vec = interp.nearest(src_time_axis, src_flat)
        else:
            res_vec = interp.interpolate(src_time_axis, src_flat)

        # res_vec shape (nt_tgt, ntr) -> reshape back to (ni, nj, nt_tgt)
        out = res_vec.reshape(len(target_time_axis), ni, nj).transpose(1, 2, 0)
        return out

    def depth_to_time_from_twt(
        self,
        data_depth: NDArray[Any],
        twt_irregular: NDArray[Any],
        time_axis: NDArray[Any],
        is_categorical: bool = False,
        progress_every: int | None = 30,
        prefix: str = "",
        plan: ResamplePlan | None = None,
    ) -> NDArray[Any]:
        """Convert a depth-sampled property cube to regular time using an
        irregular TWT cube (twt_irregular).
        """
        # `prefix` is kept for API compatibility; mark as intentionally
        # unused to avoid lint warnings.
        del prefix  # intentionally unused

        ni, nj, nz = data_depth.shape
        # reference nz to satisfy static analysis tools
        assert nz >= 0
        nt = len(time_axis)
        data_time = np.zeros((ni, nj, nt), dtype=data_depth.dtype)

        # Numba is a required dependency, so use optimized compiled path
        use_numba = os.environ.get("RESAMPLE_USE_NUMBA", "1") == "1"

        if use_numba:
            # Use shared module-level jitted kernels for from-twt resampling.
            if is_categorical or np.issubdtype(data_depth.dtype, np.integer):
                _nearest_resample_numba_jitted(
                    twt_irregular, data_depth, time_axis, data_time
                )
            else:
                _linear_resample_numba_jitted(
                    twt_irregular, data_depth, time_axis, data_time
                )
        else:
            data_time = self._depth_to_time_from_twt_fallback(
                data_depth=data_depth,
                twt_irregular=twt_irregular,
                time_axis=time_axis,
                is_categorical=is_categorical,
                progress_every=progress_every,
                plan=plan,
            )

        return data_time

    def _depth_to_time_from_twt_fallback(
        self,
        data_depth: NDArray[Any],
        twt_irregular: NDArray[Any],
        time_axis: NDArray[Any],
        is_categorical: bool,
        progress_every: int | None,
        plan: ResamplePlan | None,
    ) -> NDArray[Any]:
        """Fallback non-numba path for `depth_to_time_from_twt` extracted
        to reduce method complexity. Preserves the original blocked and
        per-trace interpolation behaviour.
        """
        ni, nj, _ = data_depth.shape
        nt = len(time_axis)
        data_time = np.zeros((ni, nj, nt), dtype=data_depth.dtype)

        # If a ResamplePlan is provided and matches this twt_irregular, use it
        if plan is not None:
            twt_arr = plan.twt_arr
        else:
            twt_arr = twt_irregular

        uniform_twt = np.allclose(
            twt_arr, np.broadcast_to(twt_arr[0, 0, :], twt_arr.shape)
        )
        if uniform_twt:
            data_time = self._handle_uniform_twt_from_twt(
                data_depth=data_depth,
                twt_arr=twt_arr,
                time_axis=time_axis,
                is_categorical=is_categorical,
            )
        else:
            # Delegate per-cell time->depth interpolation to helper
            self._time_to_depth_per_cell(
                data_time=data_time,
                data_depth=data_depth,
                twt_irregular=twt_irregular,
                time_axis=time_axis,
                is_categorical=is_categorical,
                progress_every=progress_every,
            )

        return data_time

    def _time_to_depth_per_cell(
        self,
        data_time: NDArray[Any],
        data_depth: NDArray[Any],
        twt_irregular: NDArray[Any],
        time_axis: NDArray[Any],
        is_categorical: bool,
        progress_every: int | None,
    ) -> None:
        """Per-cell fallback interpolation for `depth_to_time_from_twt`.

        Extracted to reduce local variables and statements in the parent
        method while preserving behavior.
        """
        ni, nj, _ = data_depth.shape
        for i in range(ni):
            if progress_every and i % progress_every == 0:
                pass
            for j in range(nj):
                twt_trace = twt_irregular[i, j, :]
                twt_padded = np.concatenate([[0.0], twt_trace])
                data_trace = data_depth[i, j, :]
                data_padded = np.concatenate([[data_trace[0]], data_trace])

                kind = "nearest" if is_categorical else "linear"
                interp_func = interp1d(
                    twt_padded,
                    data_padded,
                    kind=kind,
                    bounds_error=False,
                    fill_value=0.0,
                )
                data_time[i, j, :] = interp_func(time_axis)

    def _handle_uniform_twt_from_twt(
        self,
        data_depth: NDArray[Any],
        twt_arr: NDArray[Any],
        time_axis: NDArray[Any],
        is_categorical: bool,
    ) -> NDArray[Any]:
        """Uniform-twt fast path for `depth_to_time_from_twt`.

        Returns a resampled `data_time` array for the uniform-twt case.
        """
        ni, nj, nz = data_depth.shape
        nt = len(time_axis)
        twt_padded = np.concatenate([[0.0], twt_arr[0, 0, :]])
        depth_padded = np.concatenate([data_depth[:, :, 0:1], data_depth], axis=2)
        depth_padded_flat = depth_padded.transpose(2, 0, 1).reshape(nz + 1, -1)

        interp = BatchedInterpolator(
            time_axis=time_axis, kind=("nearest" if is_categorical else "linear")
        )
        if is_categorical or np.issubdtype(data_depth.dtype, np.integer):
            res_vec = interp.nearest(twt_padded, depth_padded_flat)
        else:
            res_vec = interp.interpolate(twt_padded, depth_padded_flat)

        return res_vec.reshape(nt, ni, nj).transpose(1, 2, 0)

# Thin factory to provide DepthTimeResampler instances per GridSpec.

class ResamplerFactory:
    """Factory that returns cached DepthTimeResampler instances keyed by
    grid_spec (shape, dz, dt). This avoids repeated construction when many
    modules request a resampler for the same grid."""

    def __init__(self) -> None:
        self._cache: dict[tuple[tuple[int, ...], float, float], DepthTimeResampler] = {}

    def get_resampler(self, grid_spec: GridSpec) -> DepthTimeResampler:
        """Return a cached `DepthTimeResampler` for `grid_spec`.

        Creates and caches a new `DepthTimeResampler` on first request for
        a given grid specification, keyed by `(shape, dz, dt)`.
        """
        key: tuple[tuple[int, ...], float, float] = (
            tuple(grid_spec.shape),
            float(grid_spec.dz),
            float(grid_spec.dt),
        )
        if key not in self._cache:
            self._cache[key] = DepthTimeResampler(grid_spec=grid_spec)
        return self._cache[key]

__all__.extend(["ResamplerFactory"])

# Module-level singleton instance for convenient access

def _create_resampler_factory() -> ResamplerFactory:
    """Factory function to create ResamplerFactory singleton."""
    return ResamplerFactory()

resampler_factory: ResamplerFactory = _create_resampler_factory()
__all__.append("resampler_factory")
