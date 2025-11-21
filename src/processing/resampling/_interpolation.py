"""Shared interpolation utilities for resampling operations.

This module provides common interpolation functions used across different
resampling implementations to avoid code duplication.
"""

from typing import Any

from numba import njit
from numpy.typing import NDArray

@njit
def linear_interpolate_value(
    t: float, t0: float, t1: float, v0: float, v1: float
) -> float:
    """Perform linear interpolation between two values.

    Given a time point `t` between `t0` and `t1`, interpolate the value
    between `v0` (at t0) and `v1` (at t1).

    Args:
        t: Target time point for interpolation
        t0: Time of first sample
        t1: Time of second sample
        v0: Value at t0
        v1: Value at t1

    Returns:
        Interpolated value at time t
    """
    if t1 == t0:
        return v0
    frac = (t - t0) / (t1 - t0)
    return v0 * (1 - frac) + v1 * frac

@njit
def interpolate_trace_at_time(
    t: float,
    k: int,
    twt: NDArray[Any],
    trace: NDArray[Any],
) -> tuple[float, int]:
    """Interpolate trace value at a given time point.

    Performs binary search to find the depth sample containing the time point,
    then linearly interpolates the trace value at that time.

    Args:
        t: Target time for interpolation
        k: Current depth index (will be updated)
        twt: Two-way travel time array
        trace: Trace data array

    Returns:
        Tuple of (interpolated_value, updated_k_index)
    """
    nz = len(twt)

    # Advance k while twt[k] <= t
    while k < nz and twt[k] <= t:
        k += 1

    if k == 0:
        return trace[0], k
    if k >= nz:
        return trace[nz - 1], k
    # Linear interpolation between k-1 and k
    t0 = twt[k - 1]
    t1 = twt[k]
    v0 = trace[k - 1]
    v1 = trace[k]

    value = linear_interpolate_value(t, t0, t1, v0, v1)
    return value, k
