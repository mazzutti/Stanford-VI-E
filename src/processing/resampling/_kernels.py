"""Optimized resampling kernels using Numba JIT compilation.


Centralizes numba-compiled functions for depth-to-time and time-to-depth
resampling, reducing complexity in the main DepthTimeResampler class.


Functions are compiled once and cached by numba for subsequent calls.
"""

from typing import Any
from numpy.typing import NDArray

from numba import njit, prange

from src.processing.resampling._interpolation import linear_interpolate_value


# ============================================================================
# Depth-to-Time Resampling Kernels
# ============================================================================


@njit(parallel=True)
def resample_depth_to_time_nearest(
    twt_irregular: NDArray[Any],
    data_depth: NDArray[Any],
    time_axis: NDArray[Any],
    out_array: NDArray[Any],
) -> None:
    """Nearest-neighbor resampling from depth to time domain (in-place).

    For integer/categorical data. Performs nearest-neighbor interpolation
    in parallel across all traces.

    Args:
        twt_irregular: (ni, nj, nz) two-way travel times at each depth
        data_depth: (ni, nj, nz) depth-domain data
        time_axis: (nt,) time sampling axis
        out_array: (ni, nj, nt) output array to fill
    """
    ni, nj, nz = data_depth.shape
    nt = time_axis.shape[0]

    for ii in prange(ni):
        for jj in range(nj):
            twt = twt_irregular[ii, jj]
            prop = data_depth[ii, jj]
            k = 0

            for ti in range(nt):
                t = time_axis[ti]
                # Find first depth sample where twt >= t
                while k < nz and twt[k] <= t:
                    k += 1

                if k == 0:
                    out_array[ii, jj, ti] = prop[0]
                elif k >= nz:
                    out_array[ii, jj, ti] = prop[nz - 1]
                else:
                    # Compare neighbors and pick closest
                    if abs(t - twt[k - 1]) <= abs(twt[k] - t):
                        out_array[ii, jj, ti] = prop[k - 1]
                    else:
                        out_array[ii, jj, ti] = prop[k]


@njit(parallel=True)
def resample_depth_to_time_linear(
    twt_irregular: NDArray[Any],
    data_depth: NDArray[Any],
    time_axis: NDArray[Any],
    out_array: NDArray[Any],
) -> None:
    """Linear interpolation from depth to time domain (in-place).

    For floating-point data. Performs linear interpolation in parallel
    across all traces.

    Args:
        twt_irregular: (ni, nj, nz) two-way travel times at each depth
        data_depth: (ni, nj, nz) depth-domain data
        time_axis: (nt,) time sampling axis
        out_array: (ni, nj, nt) output array to fill
    """
    ni, nj, nz = data_depth.shape
    nt = time_axis.shape[0]

    for ii in prange(ni):
        for jj in range(nj):
            twt = twt_irregular[ii, jj]
            trace = data_depth[ii, jj]
            k = 0

            for ti in range(nt):
                t = time_axis[ti]
                # Find first depth sample where twt >= t
                while k < nz and twt[k] <= t:
                    k += 1

                if k == 0:
                    out_array[ii, jj, ti] = trace[0]
                elif k >= nz:
                    out_array[ii, jj, ti] = trace[nz - 1]
                else:
                    # Linear interpolation between k-1 and k
                    t0 = twt[k - 1]
                    t1 = twt[k]
                    v0 = trace[k - 1]
                    v1 = trace[k]
                    out_array[ii, jj, ti] = linear_interpolate_value(t, t0, t1, v0, v1)


# ============================================================================
# Time-to-Depth Resampling Kernels (Irregular Time-Domain Input)
# ============================================================================


@njit(parallel=True)
def resample_depth_to_time_from_irregular_nearest(
    twt_irregular: NDArray[Any],
    data_depth: NDArray[Any],
    time_axis: NDArray[Any],
    out_array: NDArray[Any],
) -> None:
    """Nearest-neighbor resampling with irregular TWT (in-place).

    For integer/categorical data. Used when TWT varies per trace.

    Args:
        twt_irregular: (ni, nj, nz) two-way travel times
        data_depth: (ni, nj, nz) depth-domain data
        time_axis: (nt,) time sampling axis
        out_array: (ni, nj, nt) output array to fill
    """
    ni, nj, nz = data_depth.shape
    nt = time_axis.shape[0]

    for ii in prange(ni):
        for jj in range(nj):
            twt = twt_irregular[ii, jj]
            prop = data_depth[ii, jj]
            k = 0

            for ti in range(nt):
                t = time_axis[ti]
                while k < nz and twt[k] <= t:
                    k += 1

                if k == 0:
                    out_array[ii, jj, ti] = prop[0]
                elif k >= nz:
                    out_array[ii, jj, ti] = prop[nz - 1]
                else:
                    if abs(t - twt[k - 1]) <= abs(twt[k] - t):
                        out_array[ii, jj, ti] = prop[k - 1]
                    else:
                        out_array[ii, jj, ti] = prop[k]


@njit(parallel=True)
def resample_depth_to_time_from_irregular_linear(
    twt_irregular: NDArray[Any],
    data_depth: NDArray[Any],
    time_axis: NDArray[Any],
    out_array: NDArray[Any],
) -> None:
    """Linear interpolation with irregular TWT (in-place).

    For floating-point data. Used when TWT varies per trace.

    Args:
        twt_irregular: (ni, nj, nz) two-way travel times
        data_depth: (ni, nj, nz) depth-domain data
        time_axis: (nt,) time sampling axis
        out_array: (ni, nj, nt) output array to fill
    """
    ni, nj, nz = data_depth.shape
    nt = time_axis.shape[0]

    for ii in prange(ni):
        for jj in range(nj):
            twt = twt_irregular[ii, jj]
            trace = data_depth[ii, jj]
            k = 0

            for ti in range(nt):
                t = time_axis[ti]
                while k < nz and twt[k] <= t:
                    k += 1

                if k == 0:
                    out_array[ii, jj, ti] = trace[0]
                elif k >= nz:
                    out_array[ii, jj, ti] = trace[nz - 1]
                else:
                    t0 = twt[k - 1]
                    t1 = twt[k]
                    v0 = trace[k - 1]
                    v1 = trace[k]

                    if t1 == t0:
                        out_array[ii, jj, ti] = v0
                    else:
                        frac = (t - t0) / (t1 - t0)
                        out_array[ii, jj, ti] = v0 * (1 - frac) + v1 * frac
