"""Depth <-> Time resampling helpers.

Provides a small, testable Resampler that centralizes depth/time conversions
using a `GridSpec` object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy.interpolate import interp1d
from src.processing.interpolator import BatchedInterpolator
from src.processing.resample_plan import ResamplePlan
from src.utils.compat import _NUMBA_AVAILABLE, njit, prange
import logging
import os

__all__ = ["DepthTimeResampler", "set_backend_verbose", "is_backend_verbose"]

# Module logger
logger = logging.getLogger(__name__)


# Enable extra backend debug logging if the environment flag is set.
# Set RESAMPLE_BACKEND_VERBOSE=1 (or 'true') to enable DEBUG-level logs from
# this module which will include plan metadata when a backend is selected.
_BACKEND_VERBOSE = os.environ.get("RESAMPLE_BACKEND_VERBOSE", "0").lower() in (
    "1",
    "true",
    "yes",
)
if _BACKEND_VERBOSE:
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger(__name__).debug(
        "RESAMPLE_BACKEND_VERBOSE enabled: backend debug logs ON"
    )


def set_backend_verbose(on: bool) -> None:
    """Programmatically enable/disable backend verbose logging for this module.

    When enabled the module logger is set to DEBUG and additional plan
    metadata is emitted when a backend is selected.
    """
    global _BACKEND_VERBOSE
    _BACKEND_VERBOSE = bool(on)
    logger = logging.getLogger(__name__)
    if _BACKEND_VERBOSE:
        logger.setLevel(logging.DEBUG)
        logger.debug("Backend verbose logging enabled via set_backend_verbose(True)")
    else:
        # Revert to INFO level by default
        logger.setLevel(logging.INFO)
        logger.info("Backend verbose logging disabled via set_backend_verbose(False)")


def is_backend_verbose() -> bool:
    """Return whether backend verbose logging is enabled for this module."""
    return bool(_BACKEND_VERBOSE)


from src.processing._backend_base import BackendResult, BackendError

from src.io.grid import GridSpec
from src.utils.units import UnitRegistry
from src.utils.quantity import Quantity


@dataclass
class DepthTimeResampler:
    """Utility for converting between depth-sampled cubes and time-sampled cubes.

    Methods accept 3D arrays with shape (ni, nj, nk) for depth or (ni, nj, nt)
    for time. The resampler uses per-trace velocity (vp) in depth domain to
    compute two-way travel time (TWT) traces per column and performs
    interpolation using SciPy's interp1d.
    """

    grid_spec: GridSpec

    def compute_one_way_time(self, vp_trace: np.ndarray | Quantity) -> np.ndarray:
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

    def compute_one_way_times(self, vp_arr: np.ndarray | Quantity) -> np.ndarray:
        """Vectorized computation of one-way cumulative time for all traces.

        Args:
            vp_arr: 3D array of shape (ni, nj, nz) or a Quantity wrapping that

        Returns:
            3D array (ni, nj, nz) of one-way cumulative time (seconds)
        """
        # Unwrap Quantity if needed
        if isinstance(vp_arr, Quantity):
            vp_val = vp_arr.array
        else:
            vp_val = np.asarray(vp_arr)

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

    def depth_to_time_cube(
        self,
        data_depth: np.ndarray | Quantity,
        vp_depth: np.ndarray | Quantity,
        target_dt: Optional[float] = None,
        target_nt: Optional[int] = None,
        plan: "ResamplePlan" | None = None,
    ) -> Tuple[np.ndarray, float]:
        """Resample depth-sampled `data_depth` into a regularly sampled time cube.

        Args:
            data_depth: (ni, nj, nz) depth-sampled property (float or int)
            vp_depth: (ni, nj, nz) P-wave velocity in m/s
            target_dt: optional time sampling interval; if None uses grid_spec.dt

        Returns:
            (data_time, dt) where data_time has shape (ni, nj, nt)
        """
        # Unwrap Quantity inputs if provided
        data_was_quantity = isinstance(data_depth, Quantity)
        vp_was_quantity = isinstance(vp_depth, Quantity)

        data_arr = data_depth.array if data_was_quantity else np.asarray(data_depth)
        vp_arr = vp_depth.array if vp_was_quantity else np.asarray(vp_depth)

        if data_arr.shape != vp_arr.shape:
            raise ValueError("data_depth and vp_depth must have the same shape")

        ni, nj, nz = data_arr.shape
        # Use provided ResamplePlan if available to avoid recomputing.
        if plan is None:
            plan = ResamplePlan.create(
                self.grid_spec,
                vp_depth=vp_arr,
                target_dt=target_dt,
                target_nt=target_nt,
            )

        # Try to consult a pluggable backend for potential optimized paths.
        from src.processing.backend_manager import get_backend_manager

        backend = get_backend_manager().get_best(plan)
        logger = logging.getLogger(__name__)
        if backend is not None:
            logger.info(
                "Depth->Time: using backend '%s' (uniform_twt=%s)",
                getattr(backend, "name", repr(backend)),
                getattr(plan, "uniform_twt", None),
            )
            # Extra plan metadata at DEBUG level when requested
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
            except Exception as exc:  # log and re-raise
                logger.exception(
                    "Depth->Time: backend '%s' raised an exception",
                    getattr(backend, "name", repr(backend)),
                )
                raise
            # Enforce strict BackendResult return type from backends
            if not isinstance(out_backend, BackendResult):
                logger.error(
                    "Depth->Time: backend '%s' returned non-BackendResult",
                    getattr(backend, "name", repr(backend)),
                )
                raise BackendError(
                    f"backend {getattr(backend, 'name', repr(backend))} returned non-BackendResult"
                )
            out = out_backend.array
            dt = out_backend.dt if out_backend.dt is not None else plan.dt
            if data_was_quantity:
                return Quantity(out, data_depth.unit), dt
            return out, dt
        time_axis = plan.time_axis
        nt = plan.nt
        dt = plan.dt
        out = np.zeros((ni, nj, nt), dtype=data_arr.dtype)

        # If numba is available and enabled via env, use a JIT-compiled linear
        # resampler that performs per-trace interpolation in parallel. This
        # avoids Python overhead for large grids. Nearest-neighbor categorical
        # resampling still uses the fallback interp1d for correctness.
        use_numba = (
            _NUMBA_AVAILABLE and os.environ.get("RESAMPLE_USE_NUMBA", "1") == "1"
        )

        # Fast vectorized path: if all TWT traces are identical across the
        # spatial grid (common when using a flat layered model), we can
        # perform a single broadcasted interpolation using SciPy's interp1d
        # with axis=0 which is implemented in C and is significantly faster
        # than per-trace Python loops.
        twt_padded = plan.twt_padded()
        depth_padded_flat = plan.prepare_depth_padded_flat(data_arr)

        if plan.uniform_twt and not np.issubdtype(data_arr.dtype, np.integer):
            interp = BatchedInterpolator(time_axis=time_axis, kind="linear")
            res_vec = interp.interpolate(twt_padded, depth_padded_flat)
            out = res_vec.reshape(nt, ni, nj).transpose(1, 2, 0)
            if data_was_quantity:
                return Quantity(out, data_depth.unit), dt
            return out, dt

        if plan.uniform_twt:
            interp = BatchedInterpolator(time_axis=time_axis, kind="linear")
            if np.issubdtype(data_arr.dtype, np.integer):
                res_vec = interp.nearest(twt_padded, depth_padded_flat)
            else:
                res_vec = interp.interpolate(twt_padded, depth_padded_flat)

            out = res_vec.reshape(nt, ni, nj).transpose(1, 2, 0)
            if data_was_quantity:
                return Quantity(out, data_depth.unit), dt
            return out, dt

        if use_numba:
            # Implement compiled resamplers using numba. Choose nearest for
            # integer/categorical data, linear otherwise.
            if np.issubdtype(data_arr.dtype, np.integer):

                @njit(parallel=True)
                def _nearest_resample_numba(twt_ir, data, t_axis, out_arr):
                    ni_, nj_, nz_ = data.shape
                    nt_ = t_axis.shape[0]
                    for ii in prange(ni_):
                        for jj in range(nj_):
                            twt = twt_ir[ii, jj]
                            prop = data[ii, jj]
                            k = 0
                            # k is the number of twt samples <= t (padded with t=0)
                            for ti in range(nt_):
                                t = t_axis[ti]
                                while k < nz_ and twt[k] <= t:
                                    k += 1
                                if k == 0:
                                    # before first depth sample -> choose first
                                    out_arr[ii, jj, ti] = prop[0]
                                elif k >= nz_:
                                    # after last sample -> choose last
                                    out_arr[ii, jj, ti] = prop[nz_ - 1]
                                else:
                                    # compare neighbors k-1 and k
                                    if abs(t - twt[k - 1]) <= abs(twt[k] - t):
                                        out_arr[ii, jj, ti] = prop[k - 1]
                                    else:
                                        out_arr[ii, jj, ti] = prop[k]

                _nearest_resample_numba(plan.one_way * 2.0, data_arr, time_axis, out)
            else:

                @njit(parallel=True)
                def _linear_resample_numba(twt_ir, data, t_axis, out_arr):
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
                                    # interpolate between k-1 and k
                                    t0 = twt[k - 1]
                                    t1 = twt[k]
                                    v0 = trace[k - 1]
                                    v1 = trace[k]
                                    if t1 == t0:
                                        out_arr[ii, jj, ti] = v0
                                    else:
                                        frac = (t - t0) / (t1 - t0)
                                        out_arr[ii, jj, ti] = (
                                            v0 * (1 - frac) + v1 * frac
                                        )

                _linear_resample_numba(plan.one_way * 2.0, data_arr, time_axis, out)
        else:
            # Fallback: process traces in blocks and use BatchedInterpolator
            # when a block has identical TWT traces (common in layered models).
            ntr = ni * nj
            # flatten spatial dims -> traces as columns
            data_flat = data_arr.transpose(2, 0, 1).reshape(nz, -1)
            twt_flat = (2.0 * plan.one_way).transpose(2, 0, 1).reshape(nz, -1)

            interp = BatchedInterpolator(time_axis=time_axis)
            bs = interp.block_size
            for start in range(0, ntr, bs):
                end = min(start + bs, ntr)
                data_block = data_flat[:, start:end]  # shape (nz, nblock)
                twt_block = twt_flat[:, start:end]  # shape (nz, nblock)

                # Check if all columns in this block have identical twt
                # Compare to first column
                if np.allclose(
                    twt_block, np.broadcast_to(twt_block[:, 0:1], twt_block.shape)
                ):
                    twt_padded = np.concatenate([[0.0], twt_block[:, 0]])
                    depth_padded = np.concatenate(
                        [data_block[0:1, :], data_block], axis=0
                    )
                    depth_padded_flat = depth_padded.reshape(nz + 1, -1)
                    if np.issubdtype(data_arr.dtype, np.integer):
                        res = interp.nearest(twt_padded, depth_padded_flat)
                    else:
                        res = interp.interpolate(twt_padded, depth_padded_flat)
                    # res shape (nt, nblock)
                    out.reshape(nt, -1)[:, start:end] = res
                else:
                    # Mixed twt in block: fall back to per-trace interp1d for this block
                    for col in range(start, end):
                        col_idx = col - start
                        twt_trace = twt_block[:, col_idx]
                        twt_padded = np.concatenate([[0.0], twt_trace])
                        depth_trace = data_block[:, col_idx]
                        depth_padded = np.concatenate([[depth_trace[0]], depth_trace])
                        interp_fn = interp1d(
                            twt_padded,
                            depth_padded,
                            kind=(
                                "nearest"
                                if np.issubdtype(data_arr.dtype, np.integer)
                                else "linear"
                            ),
                            bounds_error=False,
                            fill_value=0.0,
                        )
                        out.reshape(nt, -1)[:, col] = interp_fn(time_axis)

            # reshape out already filled via views
            out = out

        # Wrap output as Quantity if data_depth was Quantity
        if data_was_quantity:
            return Quantity(out, data_depth.unit), dt
        return out, dt

    def time_to_depth_cube(
        self,
        seismogram_time: np.ndarray | Quantity,
        vp_depth: np.ndarray | Quantity,
        plan: "ResamplePlan" | None = None,
    ) -> np.ndarray | Quantity:
        """Convert a time-sampled seismogram to depth-sampled cube using vp_depth.

        Args:
            seismogram_time: (ni, nj, nt)
            vp_depth: (ni, nj, nz)

        Returns:
            seismogram_depth: (ni, nj, nz)
        """
        # Unwrap
        seis_was_quantity = isinstance(seismogram_time, Quantity)
        seis_arr = (
            seismogram_time.array if seis_was_quantity else np.asarray(seismogram_time)
        )
        vp_arr = (
            vp_depth.array if isinstance(vp_depth, Quantity) else np.asarray(vp_depth)
        )

        ni_t, nj_t, nt = seis_arr.shape
        ni, nj, nz = vp_arr.shape
        if (ni_t, nj_t) != (ni, nj):
            raise ValueError(
                "spatial dimensions of seismogram_time and vp_depth must match"
            )

        dt = self.grid_spec.dt
        time_axis = np.arange(nt) * dt

        out = np.zeros((ni, nj, nz), dtype=seis_arr.dtype)

        # Use provided ResamplePlan if available to avoid recomputing TWT.
        if plan is None:
            plan = ResamplePlan.create(self.grid_spec, vp_arr)

        # Try backend first (consult the BackendManager singleton)
        from src.processing.backend_manager import get_backend_manager

        backend = get_backend_manager().get_best(plan)
        logger = logging.getLogger(__name__)
        if backend is not None:
            logger.info(
                "Time->Depth: using backend '%s' (uniform_twt=%s)",
                getattr(backend, "name", repr(backend)),
                getattr(plan, "uniform_twt", None),
            )
            # Extra plan metadata at DEBUG level when requested
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
            except Exception:
                logger.exception(
                    "Time->Depth: backend '%s' raised an exception",
                    getattr(backend, "name", repr(backend)),
                )
                raise
            if not isinstance(out_backend, BackendResult):
                logger.error(
                    "Time->Depth: backend '%s' returned non-BackendResult",
                    getattr(backend, "name", repr(backend)),
                )
                raise BackendError(
                    f"backend {getattr(backend, 'name', repr(backend))} returned non-BackendResult"
                )
            out = out_backend.array
            if seis_was_quantity:
                return Quantity(out, seismogram_time.unit)
            return out

        twt_arr = plan.twt_arr
        if plan.uniform_twt:
            # common twt positions for depth samples
            twt_common = twt_arr[0, 0, :]
            ntr = ni * nj
            seis_flat = seis_arr.transpose(2, 0, 1).reshape(nt, -1)
            interp = BatchedInterpolator(time_axis=twt_common, kind="linear")
            # if seis is integer/categorical, prefer nearest
            if np.issubdtype(seis_arr.dtype, np.integer):
                res = interp.nearest(time_axis, seis_flat)
            else:
                res = interp.interpolate(time_axis, seis_flat)

            # res shape (nz, ntr) -> reshape to (ni, nj, nz)
            out = res.reshape(nz, ni, nj).transpose(1, 2, 0)
        else:
            for i in range(ni):
                for j in range(nj):
                    twt_trace = twt_arr[i, j, :]

                    # Directly sample the time-domain seismogram at the TWT corresponding
                    # to each depth sample. Use np.interp which expects xp increasing
                    # (time_axis) and fp of same length (seismogram samples).
                    out[i, j, :] = np.interp(
                        twt_trace, time_axis, seis_arr[i, j, :], left=0.0, right=0.0
                    )
        if seis_was_quantity:
            return Quantity(out, seismogram_time.unit)
        return out

    def compute_twt_for_trace(
        self, vp_trace: np.ndarray | Quantity
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compatibility helper: return TWT (padded) and depth axis (padded).

        Returns the same shapes as the old `compute_twt_for_trace` helper used
        elsewhere in the codebase: twt_trace (with a leading 0) and a depth
        trace (also padded) so they can be passed to interp functions.
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
        data_time: np.ndarray,
        src_time_axis: np.ndarray,
        target_time_axis: np.ndarray,
        kind: str = "linear",
        progress_every: Optional[int] = 30,
        prefix: str = "",
    ) -> np.ndarray:
        """Resample a time-sampled cube from src_time_axis to target_time_axis.

        Mirrors the old utility function signature.
        """
        ni, nj, nt_src = data_time.shape
        nt_tgt = len(target_time_axis)
        out = np.zeros((ni, nj, nt_tgt), dtype=data_time.dtype)

        # Vectorized / batched interpolation across traces via BatchedInterpolator.
        # Flatten spatial dims so we interpolate all traces with a single
        # blocked/interpolated call which is much faster than nested Python loops.
        nt_src = data_time.shape[2]
        ntr = ni * nj
        # depth_padded_flat equivalent: rows are source-time samples, cols are traces
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
        data_depth: np.ndarray,
        twt_irregular: np.ndarray,
        time_axis: np.ndarray,
        is_categorical: bool = False,
        progress_every: Optional[int] = 30,
        prefix: str = "",
        plan: "ResamplePlan" | None = None,
    ) -> np.ndarray:
        """Convert a depth-sampled property cube to regular time using an
        irregular TWT cube (twt_irregular).
        """
        ni, nj, nz = data_depth.shape
        nt = len(time_axis)
        data_time = np.zeros((ni, nj, nt), dtype=data_depth.dtype)

        use_numba = (
            _NUMBA_AVAILABLE and os.environ.get("RESAMPLE_USE_NUMBA", "1") == "1"
        )

        if use_numba:
            if is_categorical or np.issubdtype(data_depth.dtype, np.integer):

                @njit(parallel=True)
                def _nearest_from_twt(twt_ir, data_d, t_axis, out_a):
                    ni_, nj_, nz_ = data_d.shape
                    nt_ = t_axis.shape[0]
                    for ii in prange(ni_):
                        for jj in range(nj_):
                            twt = twt_ir[ii, jj]
                            prop = data_d[ii, jj]
                            k = 0
                            for ti in range(nt_):
                                t = t_axis[ti]
                                while k < nz_ and twt[k] <= t:
                                    k += 1
                                if k == 0:
                                    out_a[ii, jj, ti] = prop[0]
                                elif k >= nz_:
                                    out_a[ii, jj, ti] = prop[nz_ - 1]
                                else:
                                    if abs(t - twt[k - 1]) <= abs(twt[k] - t):
                                        out_a[ii, jj, ti] = prop[k - 1]
                                    else:
                                        out_a[ii, jj, ti] = prop[k]

                _nearest_from_twt(twt_irregular, data_depth, time_axis, data_time)
            else:

                @njit(parallel=True)
                def _linear_resample_from_twt(twt_ir, data_d, t_axis, out_a):
                    ni_, nj_, nz_ = data_d.shape
                    nt_ = t_axis.shape[0]
                    for ii in prange(ni_):
                        for jj in range(nj_):
                            twt = twt_ir[ii, jj]
                            trace = data_d[ii, jj]
                            k = 0
                            for ti in range(nt_):
                                t = t_axis[ti]
                                while k < nz_ and twt[k] <= t:
                                    k += 1
                                if k == 0:
                                    out_a[ii, jj, ti] = trace[0]
                                elif k >= nz_:
                                    out_a[ii, jj, ti] = trace[nz_ - 1]
                                else:
                                    t0 = twt[k - 1]
                                    t1 = twt[k]
                                    v0 = trace[k - 1]
                                    v1 = trace[k]
                                    if t1 == t0:
                                        out_a[ii, jj, ti] = v0
                                    else:
                                        frac = (t - t0) / (t1 - t0)
                                        out_a[ii, jj, ti] = v0 * (1 - frac) + v1 * frac

                _linear_resample_from_twt(
                    twt_irregular, data_depth, time_axis, data_time
                )
        else:
            # If a ResamplePlan is provided and matches this twt_irregular, use it
            if plan is not None:
                twt_arr = plan.twt_arr
            else:
                twt_arr = twt_irregular

            uniform_twt = np.allclose(
                twt_arr, np.broadcast_to(twt_arr[0, 0, :], twt_arr.shape)
            )

            if uniform_twt:
                twt_padded = np.concatenate([[0.0], twt_arr[0, 0, :]])
                depth_padded = np.concatenate(
                    [data_depth[:, :, 0:1], data_depth], axis=2
                )
                depth_padded_flat = depth_padded.transpose(2, 0, 1).reshape(nz + 1, -1)

                interp = BatchedInterpolator(
                    time_axis=time_axis,
                    kind=("nearest" if is_categorical else "linear"),
                )
                if is_categorical or np.issubdtype(data_depth.dtype, np.integer):
                    res_vec = interp.nearest(twt_padded, depth_padded_flat)
                else:
                    res_vec = interp.interpolate(twt_padded, depth_padded_flat)

                data_time = res_vec.reshape(nt, ni, nj).transpose(1, 2, 0)
            else:
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

        return data_time


# Thin factory to provide DepthTimeResampler instances per GridSpec.
class ResamplerFactory:
    """Factory that returns cached DepthTimeResampler instances keyed by
    grid_spec (shape, dz, dt). This avoids repeated construction when many
    modules request a resampler for the same grid."""

    def __init__(self):
        self._cache = {}

    def get_resampler(self, grid_spec: GridSpec) -> DepthTimeResampler:
        key = (tuple(grid_spec.shape), float(grid_spec.dz), float(grid_spec.dt))
        if key not in self._cache:
            self._cache[key] = DepthTimeResampler(grid_spec=grid_spec)
        return self._cache[key]


# Module-level singleton factory for callers to obtain resamplers
from src.utils.facades import LazyObjectProxy


# Module-level singleton factory for callers to obtain resamplers
resampler_factory = LazyObjectProxy(lambda: ResamplerFactory())

__all__.extend(["ResamplerFactory", "resampler_factory"])


# We prefer callers use the ResamplerFactory facade or the ResamplerService
# module-level lazy proxies. The thin top-level delegate wrappers were
# retained for backward compatibility but are removed here to reduce
# duplicated surface area. Callers should use `resampler_factory` or
# `get_resampler_service()` instead.
__all__.extend([
    "ResamplerFactory",
    "resampler_factory",
    "ResamplerService",
    "resampler_service",
    "get_resampler_service",
    "get_resampler_factory",
])


# --- Simplified OO facade -------------------------------------------------
class ResamplerService:
    """A thin OO facade that forwards common resampling helpers to the
    ResamplerFactory. This provides a convenient single object to call from
    client code while keeping the original top-level functions available.
    """

    def get_resampler(self, grid_spec: GridSpec) -> DepthTimeResampler:
        return resampler_factory.get_resampler(grid_spec)

    def compute_one_way_time(
        self, grid_spec: GridSpec, vp_trace: np.ndarray | Quantity
    ):
        return resampler_factory.get_resampler(grid_spec).compute_one_way_time(
            vp_trace
        )

    def compute_one_way_times(self, grid_spec: GridSpec, vp_arr: np.ndarray | Quantity):
        return resampler_factory.get_resampler(grid_spec).compute_one_way_times(
            vp_arr
        )

    def depth_to_time_cube(
        self,
        grid_spec: GridSpec,
        data_depth,
        vp_depth,
        target_dt=None,
        target_nt=None,
        plan: ResamplePlan | None = None,
    ):
        return resampler_factory.get_resampler(grid_spec).depth_to_time_cube(
            data_depth, vp_depth, target_dt=target_dt, target_nt=target_nt, plan=plan
        )

    def time_to_depth_cube(
        self,
        grid_spec: GridSpec,
        seismogram_time,
        vp_depth,
        plan: ResamplePlan | None = None,
    ):
        return resampler_factory.get_resampler(grid_spec).time_to_depth_cube(
            seismogram_time, vp_depth, plan=plan
        )

    def resample_time_cube(
        self,
        grid_spec: GridSpec,
        data_time,
        src_time_axis,
        target_time_axis,
        kind: str = "linear",
        progress_every: Optional[int] = 30,
        prefix: str = "",
    ):
        return resampler_factory.get_resampler(grid_spec).resample_time_cube(
            data_time,
            src_time_axis,
            target_time_axis,
            kind=kind,
            progress_every=progress_every,
            prefix=prefix,
        )

    def depth_to_time_from_twt(
        self,
        grid_spec: GridSpec,
        data_depth,
        twt_irregular,
        time_axis,
        is_categorical: bool = False,
        progress_every: Optional[int] = 30,
        prefix: str = "",
        plan: ResamplePlan | None = None,
    ):
        return resampler_factory.get_resampler(grid_spec).depth_to_time_from_twt(
            data_depth,
            twt_irregular,
            time_axis,
            is_categorical=is_categorical,
            progress_every=progress_every,
            prefix=prefix,
            plan=plan,
        )

    def compute_twt_for_trace(
        self, grid_spec: GridSpec, vp_trace: np.ndarray | Quantity
    ):
        return resampler_factory.get_resampler(grid_spec).compute_twt_for_trace(
            vp_trace
        )


from src.utils.facades import LazyObjectProxy


# Use the generic LazyObjectProxy to reduce local boilerplate
resampler_service: ResamplerService = LazyObjectProxy(lambda: ResamplerService())


def get_resampler_service(
    service: ResamplerService | None = None,
) -> "ResamplerService":
    """Return the provided ResamplerService or the module-level lazy singleton.

    This mirrors the common get_* pattern used across the codebase and makes
    dependency injection in tests and clients straightforward.
    """
    return service if service is not None else resampler_service


__all__.extend(["ResamplerService", "resampler_service", "get_resampler_service"])


def get_resampler_factory(config: dict | None = None):
    """Return the module-level `resampler_factory` when `config` is None.

    If `config` is provided, return a fresh `ResamplerFactory` instance.
    This mirrors the repo-wide `get_*` pattern to make dependency injection
    and testing easier.
    """
    if config is None:
        return resampler_factory
    return ResamplerFactory()


__all__.append("get_resampler_factory")
