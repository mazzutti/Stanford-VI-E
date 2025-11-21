"""Shared helper to build Plotly Surface traces for 3D volumes.

This helper centralizes the trace construction logic so multiple
plotter modules can reuse the same implementation and avoid duplicated
code paths that trigger R0801 (duplicate-code) findings.
"""

# This module intentionally uses short, mathematical variable names and
# helper functions with several parameters to match existing call sites
# and make the surface construction explicit. Silence naming and
# argument-count warnings that would otherwise be noisy for this
# focused plotting helper.

from __future__ import annotations

from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from src.plotting.helpers.colorbar import make_plotly_colorbar
from src.plotting.helpers.configs import TraceConfig

def _build_mesh_ranges(
    shape: tuple[int, int, int], k_scale_val: float
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    ni_val, nj_val, nk_val = shape
    i_r: NDArray[np.int64] = np.arange(ni_val, dtype=np.int64)
    j_r: NDArray[np.int64] = np.arange(nj_val, dtype=np.int64)
    k_r: NDArray[np.float64] = np.arange(nk_val, dtype=np.float64) * float(k_scale_val)
    return i_r, j_r, k_r

def _make_inline_trace(
    arr_local: NDArray[Any],
    inline_i: int,
    j_r: NDArray[np.int64],
    k_r: NDArray[np.float64],
    colorscale_local: Any,
    cmin_local: float | None,
    cmax_local: float | None,
) -> go.Surface:
    # Some callers pass a precomputed `k_r` mesh; keep parameter signatures
    # stable for compatibility with calling code. Silence false-positive

    J_inline_local, K_inline_local = np.meshgrid(j_r, k_r, indexing="ij")
    I_inline_local = cast(
        Any, np.full_like(J_inline_local, float(inline_i), dtype=float)
    )
    inline_data_local = arr_local[inline_i, :, :]
    return go.Surface(
        x=I_inline_local,
        y=J_inline_local,
        z=K_inline_local,
        surfacecolor=cast(Any, inline_data_local),
        colorscale=colorscale_local,
        cmin=cmin_local,
        cmax=cmax_local,
        showscale=False,
        name=f"Inline {inline_i}",
    )

def _make_crossline_trace(
    arr_local: NDArray[Any],
    cross_j: int,
    i_r: NDArray[np.int64],
    k_r: NDArray[np.float64],
    colorscale_local: Any,
    cmin_local: float | None,
    cmax_local: float | None,
) -> go.Surface:
    # Keep signature stable for compatibility with callers; suppress lint.

    I_cross_local, K_cross_local = np.meshgrid(i_r, k_r, indexing="ij")
    J_cross_local = cast(Any, np.full_like(I_cross_local, float(cross_j), dtype=float))
    cross_data_local = arr_local[:, cross_j, :]
    return go.Surface(
        x=I_cross_local,
        y=J_cross_local,
        z=K_cross_local,
        surfacecolor=cast(Any, cross_data_local),
        colorscale=colorscale_local,
        cmin=cmin_local,
        cmax=cmax_local,
        showscale=False,
        name=f"Crossline {cross_j}",
    )

def _make_depth_trace(
    arr_local: NDArray[Any],
    depth_k: int,
    i_r: NDArray[np.int64],
    j_r: NDArray[np.int64],
    k_r: NDArray[np.float64],
    colorscale_local: Any,
    cmin_local: float | None,
    cmax_local: float | None,
    show_colorbar_local: bool,
    k_unit_local: str | None,
    colorbar_len_local: float,
    k_scale_local: float,
) -> go.Surface:
    # Keep signature stable for compatibility with callers; suppress lint.

    I_z_local, J_z_local = np.meshgrid(i_r, j_r, indexing="ij")
    return go.Surface(
        x=cast(Any, I_z_local),
        y=cast(Any, J_z_local),
        z=cast(
            Any,
            np.full_like(I_z_local, float(depth_k) * float(k_scale_local), dtype=float),
        ),
        surfacecolor=cast(Any, arr_local[:, :, depth_k]),
        colorscale=colorscale_local,
        cmin=cmin_local,
        cmax=cmax_local,
        showscale=show_colorbar_local,
        name="Depth slice",
        colorbar=make_plotly_colorbar(
            k_unit_local, show_colorbar_local, colorbar_len_local, for_inline=False
        ),
    )

def make_plotly_surface_traces(
    arr: NDArray[Any],
    inline_idx: int,
    crossline_idx: int,
    depth_idx: int,
    k_scale: float = 1.0,
    colorscale_to_use: Any = "RdBu",
    cmin: float | None = None,
    cmax: float | None = None,
    show_colorbar: bool = False,
    k_unit: str | None = None,
    colorbar_len: float = 0.75,
) -> list[go.Surface]:
    """Create three Plotly Surface traces for the provided 3D array.

    Parameters mirror the previous implementations found in the codebase.

    Returns
    -------
    list[go.Surface]
        [inline_trace, crossline_trace, depth_trace]
    """
    # Build mesh ranges and construct traces via focused module-level helpers.
    i_range, j_range, k_range = _build_mesh_ranges(arr.shape, k_scale)

    traces_out: list[go.Surface] = []
    traces_out.append(
        _make_inline_trace(
            arr, inline_idx, j_range, k_range, colorscale_to_use, cmin, cmax
        )
    )
    traces_out.append(
        _make_crossline_trace(
            arr, crossline_idx, i_range, k_range, colorscale_to_use, cmin, cmax
        )
    )
    traces_out.append(
        _make_depth_trace(
            arr,
            depth_idx,
            i_range,
            j_range,
            k_range,
            colorscale_to_use,
            cmin,
            cmax,
            show_colorbar,
            k_unit,
            colorbar_len,
            k_scale,
        )
    )

    return traces_out

def make_plotly_surface_traces_from_config(
    arr: NDArray[Any],
    inline_idx: int,
    crossline_idx: int,
    depth_idx: int,
    config: TraceConfig,
) -> list[go.Surface]:
    """Compatibility wrapper that accepts a `TraceConfig` dataclass.

    This reduces long argument lists in callers by allowing construction of
    a small config object and a single-argument pass-through.
    """
    return make_plotly_surface_traces(
        arr,
        inline_idx,
        crossline_idx,
        depth_idx,
        k_scale=config.k_scale,
        colorscale_to_use=config.colorscale_to_use,
        cmin=config.cmin,
        cmax=config.cmax,
        show_colorbar=config.show_colorbar,
        k_unit=config.k_unit,
        colorbar_len=config.colorbar_len,
    )
