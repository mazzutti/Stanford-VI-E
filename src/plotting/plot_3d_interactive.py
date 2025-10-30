"""Interactive 3D plotting (Plotly) moved into src.plotting package."""

import logging
import numpy as np
from numpy.typing import ArrayLike
import plotly.graph_objects as go

__all__ = ["create_3d_volume_plotly", "main"]

logger = logging.getLogger(__name__)


class PlotlyVisualization:
    def create_3d_volume_plotly(
        self,
        cube: ArrayLike,
        slice_indices,
        title,
        k_scale=1.0,
        k_label="K",
        k_unit="",
        colorscale="RdBu",
        is_categorical=False,
        is_impedance=False,
        show_colorbar=True,
    ):
        """Create Plotly Surface traces for three orthogonal slices."""
        arr = np.asarray(cube)
        ni, nj, nk = arr.shape
        idx_i, idx_j, idx_k = slice_indices

        if is_categorical:
            colorscale = [
                [0, "rgb(31, 119, 180)"],
                [0.33, "rgb(255, 127, 14)"],
                [0.67, "rgb(44, 160, 44)"],
                [1, "rgb(214, 39, 40)"],
            ]
            cmin = 0
            cmax = 3
        elif is_impedance:
            slice_inline = arr[idx_i, :, :]
            slice_crossline = arr[:, idx_j, :]
            slice_k = arr[:, :, idx_k]

            all_slices = np.concatenate(
                [slice_inline.ravel(), slice_crossline.ravel(), slice_k.ravel()]
            )
            cmin = float(np.percentile(all_slices, 2))
            cmax = float(np.percentile(all_slices, 98))
            if cmax == cmin:
                cmax = cmin + 1.0
            colorscale = "Turbo"
        else:
            slice_inline = arr[idx_i, :, :]
            slice_crossline = arr[:, idx_j, :]
            slice_k = arr[:, :, idx_k]

            p_inline = np.percentile(np.abs(slice_inline), 99.5)
            p_crossline = np.percentile(np.abs(slice_crossline), 99.5)
            p_k = np.percentile(np.abs(slice_k), 99.5)

            vmax = max(p_inline, p_crossline, p_k)
            cmax = float(vmax)
            cmin = -cmax
            if cmax == 0:
                cmax = 1.0
                cmin = -1.0
            colorscale = "RdBu_r"

        traces = []

        j_range = np.arange(nj)
        k_range = np.arange(nk) * k_scale
        J_inline, K_inline = np.meshgrid(j_range, k_range)
        I_inline = np.full_like(J_inline, idx_i, dtype=float)
        inline_data = arr[idx_i, :, :].T  # Shape: (nk, nj)

        trace_inline = go.Surface(
            x=I_inline,
            y=J_inline,
            z=K_inline,
            surfacecolor=inline_data,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            name=f"Inline {idx_i}",
        )
        traces.append(trace_inline)

        i_range = np.arange(ni)
        I_cross, K_cross = np.meshgrid(i_range, k_range)
        J_cross = np.full_like(I_cross, idx_j, dtype=float)
        cross_data = arr[:, idx_j, :].T

        trace_cross = go.Surface(
            x=I_cross,
            y=J_cross,
            z=K_cross,
            surfacecolor=cross_data,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            name=f"Crossline {idx_j}",
        )
        traces.append(trace_cross)

        I_z, J_z = np.meshgrid(i_range, j_range)
        K_z = np.full_like(I_z, idx_k * k_scale, dtype=float)
        z_data = arr[:, :, idx_k].T

        trace_z = go.Surface(
            x=I_z,
            y=J_z,
            z=K_z,
            surfacecolor=z_data,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=show_colorbar,
            name=f"{k_label} slice",
        )
        traces.append(trace_z)

        return traces


def main(argv=None):
    """Minimal CLI wrapper for interactive 3D plotting.

    This wrapper is intentionally small: it exposes a canonical entrypoint
    so other tooling (like `src.__main__`) can delegate to it.
    """
    # Import lazily to avoid heavy deps at module import time
    from src.plotting.helpers.plot import prepare_plotting_args, default_plot_config

    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Generate interactive 3D seismic visualization"
    )
    parser.add_argument(
        "--cache-dir", default=".cache", help="Directory for cache files"
    )
    parser.add_argument(
        "--domain",
        choices=["depth", "time"],
        default="depth",
        help="Domain for processing/visualization",
    )
    parser.add_argument(
        "--no-multiangle", action="store_true", help="Disable multi-angle EI processing"
    )
    parser.add_argument(
        "--backend", default=None, help="Optional matplotlib backend override"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

    prepare_plotting_args(args)

    plot_cfg = default_plot_config()
    # Apply any CLI-specified matplotlib backend and initialize plotting
    if args.backend:
        plot_cfg.backend = args.backend
    plot_cfg.apply_backend()

    cache_dir = args.cache_dir
    keys = [
        "avo_depth" if args.domain != "time" else "avo_time",
        "ai_depth" if args.domain != "time" else "ai_time",
        "ei_depth",
        "ei_time",
    ]

    from src.io.cache import cache_for_dir

    resolved = cache_for_dir(cache_dir).resolve_latest_paths(
        keys=keys, prefer_depth_ei=(args.domain != "time")
    )

    avo_fn = resolved.get(keys[0])
    ai_fn = resolved.get(keys[1])
    ei_fn = resolved.get("ei")

    if not (avo_fn and ai_fn and ei_fn):
        raise SystemExit("Missing required cache files for 3D interactive plotting")

    return resolved


from src.utils.facades import LazyObjectProxy


# Module-level lazy proxy for Plotly visualization facade
plotly_visualization = LazyObjectProxy(lambda: PlotlyVisualization())


def get_plotly_visualization(config: dict | None = None):
    if config is None:
        return plotly_visualization
    return PlotlyVisualization()


# Backwards-compatible top-level wrapper
def create_3d_volume_plotly(
    cube: ArrayLike,
    slice_indices,
    title,
    k_scale=1.0,
    k_label="K",
    k_unit="",
    colorscale="RdBu",
    is_categorical=False,
    is_impedance=False,
    show_colorbar=True,
):
    return plotly_visualization.create_3d_volume_plotly(
        cube,
        slice_indices,
        title,
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        colorscale=colorscale,
        is_categorical=is_categorical,
        is_impedance=is_impedance,
        show_colorbar=show_colorbar,
    )
