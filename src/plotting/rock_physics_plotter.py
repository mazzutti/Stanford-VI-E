"""Rock physics attribute plotting.

Provides RockPhysicsPlotter class for visualizing rock physics attributes.
Uses PlotConfig and ImageRenderer for clean visualization.
"""

import logging
from typing import Any, Literal, Optional, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar

from src.plotting.helpers.base import BasePlotter
from src.plotting.helpers.config import PlotConfig
from src.plotting.helpers.components import ImageRenderer

logger = logging.getLogger(__name__)


class RockPhysicsPlotter(BasePlotter):
    """Plotter for rock physics attributes.

    Visualizes rock physics derived attributes (Vp, Vs, density, impedances, etc.)
    in multiple slice orientations using PlotConfig and ImageRenderer.
    """

    def plot_attribute(
        self,
        ax: Axes,
        data: NDArray[np.floating[Any]],
        idx: int,
        slice_type: Literal["inline", "crossline", "depthslice"] = "inline",
        config: Optional[PlotConfig] = None,
    ) -> Tuple[AxesImage, Colorbar | None]:
        """Plot a rock physics attribute slice.

        Args:
            ax: Matplotlib axis
            data: 3D attribute data (I, J, K)
            idx: Slice index
            slice_type: Type of slice ('inline', 'crossline', 'depthslice')
            config: PlotConfig for styling (uses viridis default if None)

        Returns:
            Tuple of (image, colorbar)
        """
        config = config or PlotConfig.for_attributes("Attribute", cmap="viridis")

        if slice_type == "inline":
            slice_data = data[idx, :, :].T  # Transpose for correct orientation
            xlabel = "Crossline Index"
            ylabel = "Depth Index"
            title = config.title or f"Inline {idx}"

        elif slice_type == "crossline":
            slice_data = data[:, idx, :].T  # Transpose for correct orientation
            xlabel = "Inline Index"
            ylabel = "Depth Index"
            title = config.title or f"Crossline {idx}"

        else:  # depthslice
            slice_data = data[:, :, idx]  # No transpose for depth slice
            xlabel = "Inline Index"
            ylabel = "Crossline Index"
            title = config.title or f"Depth {idx}m"

        config = config.update(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )

        im, cbar = ImageRenderer.render(ax, slice_data, config)

        self._log_debug(f"plotted {slice_type}: idx={idx}, shape={slice_data.shape}")

        return im, cbar

    def plot_multiple_attributes(
        self,
        fig: Figure,
        attributes: Dict[str, NDArray[np.floating[Any]]],
        idx: int = 0,
        slice_type: Literal["inline", "crossline", "depthslice"] = "inline",
        cmap: str = "viridis",
    ) -> None:
        """Plot multiple attributes in a grid layout.

        Args:
            fig: Matplotlib figure
            attributes: Dict of {name: data_array}
            idx: Slice index
            slice_type: Type of slice
            cmap: Colormap name
        """
        n_attrs = len(attributes)
        n_cols = min(3, n_attrs)
        n_rows = (n_attrs + n_cols - 1) // n_cols

        fig.suptitle(
            f"Rock Physics Attributes ({slice_type.capitalize()} {idx})", fontsize=14
        )

        for ax_idx, (name, data) in enumerate(attributes.items(), 1):
            ax = fig.add_subplot(n_rows, n_cols, ax_idx)
            config = PlotConfig.for_attributes(name, cmap=cmap)
            self.plot_attribute(ax, data, idx, slice_type=slice_type, config=config)

        fig.tight_layout()
        self._log_info(f"plotted {n_attrs} attributes in {n_rows}x{n_cols} grid")
