"""Small plotting mixins shared across plotter classes.

Currently provides a concrete implementation for `_imshow_with_colorbar`
so the implementation can be reused by multiple plotter subclasses.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from src.plotting.helpers.configs import TraceConfig
from src.plotting.helpers.traces import make_plotly_surface_traces_from_config


class ImshowWithColorbarMixin:
    """Provides a reusable `_imshow_with_colorbar` implementation.

    Use by inheriting from the mixin before `PropertyPlotter` so
    the concrete implementation satisfies the abstract method.
    """

    def _imshow_with_colorbar(
        self,
        ax: Axes,
        arr: NDArray[Any],
        title: str,
        xlabel: str,
        ylabel: str,
        cmap: str,
        vmin: float,
        vmax: float,
        colorbar_label: str,
    ) -> AxesImage:
        im: AxesImage = ax.imshow(
            arr,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cast(Any, plt).colorbar(im, ax=ax, label=colorbar_label)
        return im

    def _compute_slice_indices_and_bounds(
        self, data: NDArray[np.floating[Any]], title: str, units: str
    ) -> tuple[int, int, int, float, float, str]:
        ni, nj, nk = data.shape
        mid_i: int = ni // 2
        mid_j: int = nj // 2
        mid_k: int = nk // 2

        vmin, vmax = np.percentile(data, [2, 98])
        colorbar_label: str = f"{title}\n[{units}]"
        return mid_i, mid_j, mid_k, float(vmin), float(vmax), colorbar_label

    def _make_plotly_traces(
        self, data: NDArray[np.floating[Any]], cmap: str, units: str
    ) -> tuple[list[go.Surface], int, int, int]:
        ni, nj, nk = data.shape
        mid_i, mid_j, mid_k = ni // 2, nj // 2, nk // 2

        cfg = TraceConfig(
            k_scale=1.0,
            colorscale_to_use=cmap,
            cmin=None,
            cmax=None,
            show_colorbar=False,
            k_unit=units,
            colorbar_len=0.75,
        )

        traces: list[go.Surface] = make_plotly_surface_traces_from_config(
            data, mid_i, mid_j, mid_k, cfg
        )

        try:
            inline_trace = traces[0]
            inline_trace.update(
                showscale=True, colorbar={"title": units, "x": 1.02, "len": 0.75}
            )
        except (IndexError, AttributeError, TypeError):
            # Some plotly trace objects expose different attribute access
            # patterns depending on version/implementation. Try attribute
            # style fallback and ignore attribute/index errors.
            try:
                traces[0].showscale = True
                traces[0].colorbar = {"title": units, "x": 1.02, "len": 0.75}
            except (IndexError, AttributeError, TypeError):
                pass

        return traces, mid_i, mid_j, mid_k
