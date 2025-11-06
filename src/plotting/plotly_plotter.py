"""Interactive 3D plotting with Plotly.

Provides PlotlyPlotter class for creating interactive 3D visualizations.
Replaces plot_3d_interactive.py with cleaner OOP interface.
"""

import logging
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go

from src.plotting.helpers.base import BasePlotter

logger = logging.getLogger(__name__)


class PlotlyPlotter(BasePlotter):
    """Plotter for interactive 3D visualizations using Plotly.

    Creates interactive Surface traces for visualizing seismic data
    and other 3D volumes.
    """

    def create_3d_volume(
        self,
        cube: NDArray[np.floating[Any]],
        slice_indices: Tuple[int, int, int],
        title: str = "",
        k_scale: float = 1.0,
        k_label: str = "K",
        k_unit: str = "",
        colorscale: str = "RdBu",
        is_categorical: bool = False,
        show_colorbar: bool = True,
    ) -> List[go.Surface]:
        """Create Plotly Surface traces for three orthogonal slices.

        Args:
            cube: 3D data array (I, J, K)
            slice_indices: Tuple of (idx_i, idx_j, idx_k)
            title: Plot title
            k_scale: Vertical scale factor
            k_label: Label for vertical axis
            k_unit: Unit for vertical axis
            colorscale: Plotly colorscale name
            is_categorical: Whether data is categorical
            show_colorbar: Whether to show colorbar

        Returns:
            List of Plotly Surface traces
        """
        arr = np.asarray(cube)
        ni, nj, nk = arr.shape
        idx_i, idx_j, idx_k = slice_indices

        # Determine color scale and limits
        cmin: float | int
        cmax: float | int
        colorscale_to_use: str | List[List[float | str]]
        if is_categorical:
            colorscale_to_use = [
                [0, "rgb(31, 119, 180)"],
                [0.33, "rgb(255, 127, 14)"],
                [0.67, "rgb(44, 160, 44)"],
                [1, "rgb(214, 39, 40)"],
            ]
            cmin = 0
            cmax = 3
        else:
            colorscale_to_use = colorscale
            slice_inline = arr[idx_i, :, :]
            slice_crossline = arr[:, idx_j, :]
            slice_k = arr[:, :, idx_k]

            p_inline = np.percentile(np.abs(slice_inline), 99.5)
            p_crossline = np.percentile(np.abs(slice_crossline), 99.5)
            p_k = np.percentile(np.abs(slice_k), 99.5)

            vmax: float = max(p_inline, p_crossline, p_k)
            cmax = float(vmax)
            cmin = -cmax
            if cmax == 0:
                cmax = 1.0
                cmin = -1.0

        traces = []

        # Inline slice (constant I)
        j_range = np.arange(nj)
        k_range = np.arange(nk) * k_scale
        J_inline, K_inline = np.meshgrid(j_range, k_range)
        I_inline = np.full_like(J_inline, idx_i, dtype=float)
        inline_data = arr[idx_i, :, :].T

        trace_inline = go.Surface(
            x=I_inline,
            y=J_inline,
            z=K_inline,
            surfacecolor=inline_data,
            colorscale=colorscale_to_use,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            name=f"Inline {idx_i}",
        )
        traces.append(trace_inline)

        # Crossline slice (constant J)
        i_range = np.arange(ni)
        I_cross, K_cross = np.meshgrid(i_range, k_range)
        J_cross = np.full_like(I_cross, idx_j, dtype=float)
        cross_data = arr[:, idx_j, :].T

        trace_cross = go.Surface(
            x=I_cross,
            y=J_cross,
            z=K_cross,
            surfacecolor=cross_data,
            colorscale=colorscale_to_use,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            name=f"Crossline {idx_j}",
        )
        traces.append(trace_cross)

        # Time/Depth slice (constant K)
        I_z, J_z = np.meshgrid(i_range, j_range)
        K_z = np.full_like(I_z, idx_k * k_scale, dtype=float)
        z_data = arr[:, :, idx_k].T

        trace_z = go.Surface(
            x=I_z,
            y=J_z,
            z=K_z,
            surfacecolor=z_data,
            colorscale=colorscale_to_use,
            cmin=cmin,
            cmax=cmax,
            showscale=show_colorbar,
            name=f"{k_label} slice",
            colorbar=dict(title="Value") if show_colorbar else None,
        )
        traces.append(trace_z)

        self._log_info(
            f"created 3d volume traces: indices=({idx_i}, {idx_j}, {idx_k}), "
            f"cmin={cmin}, cmax={cmax}"
        )

        return traces

    def create_figure(
        self,
        traces: List[go.Surface],
        title: str = "",
        width: int = 1000,
        height: int = 800,
    ) -> go.Figure:
        """Create an interactive Plotly figure from traces.

        Args:
            traces: List of Plotly traces
            title: Figure title
            width: Figure width in pixels
            height: Figure height in pixels

        Returns:
            Plotly Figure object
        """
        fig = go.Figure(data=traces)

        fig.update_layout(
            title=title,
            width=width,
            height=height,
            scene=dict(
                xaxis_title="Inline (I)",
                yaxis_title="Crossline (J)",
                zaxis_title="Time/Depth (K)",
            ),
        )

        return fig

    def show_figure(self, fig: go.Figure) -> None:
        """Display an interactive Plotly figure.

        Args:
            fig: Plotly Figure object
        """
        fig.show()
        self._log_info("displayed interactive figure")

    def save_figure(self, fig: go.Figure, filepath: str) -> None:
        """Save an interactive Plotly figure to HTML.

        Args:
            fig: Plotly Figure object
            filepath: Output HTML file path
        """
        fig.write_html(filepath)
        self._log_info(f"saved interactive figure to {filepath}")
