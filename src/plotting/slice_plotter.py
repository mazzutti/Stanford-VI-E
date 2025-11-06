"""Unified 2D and 3D slice plotting.

Provides SlicePlotter class for both 2D and 3D orthogonal slice visualizations.
Uses PlotConfig for configuration and ImageRenderer for rendering.
"""

import logging
from typing import Optional, Tuple, Union, cast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from mpl_toolkits.mplot3d import Axes3D

from src.plotting.helpers.base import BasePlotter
from src.plotting.helpers.config import PlotConfig
from src.plotting.helpers.components import (
    ImageRenderer,
    DataNormalizer,
)

logger = logging.getLogger(__name__)


class SlicePlotter(BasePlotter):
    """Plotter for 2D and 3D orthogonal slice visualizations.

    Handles inline, crossline, and time/depth slices for seismic data.
    Supports both 2D and 3D visualizations with consistent styling.
    Uses composition with SliceExtractor and ImageRenderer components.
    """

    def plot_2d_slices(
        self,
        ax: Axes,
        cube: np.ndarray,
        slice_indices: Tuple[int, int, int],
        config: Optional[PlotConfig] = None,
    ) -> Tuple[AxesImage, Colorbar | None]:
        """Plot a single 2D slice of a 3D cube.

        Args:
            ax: Matplotlib axis to plot on
            cube: 3D data array (I, J, K)
            slice_indices: Tuple of (idx_i, idx_j, idx_k)
            config: PlotConfig for styling (uses default if None)

        Returns:
            Tuple of (image, colorbar)
        """
        config = config or PlotConfig.default()

        # Extract inline slice
        idx_i, _, _ = slice_indices
        slice_data = cube[idx_i, :, :]

        # Update config with slice information
        config = config.update(
            xlabel="Crossline (J)",
            ylabel="Depth Index (K)",
            title=config.title or f"Inline {idx_i}",
        )

        # Render with components
        im, cbar = ImageRenderer.render(ax, slice_data, config)

        self._log_debug(f"plotted inline slice: idx={idx_i}, shape={slice_data.shape}")

        return im, cbar

    def plot_crossline(
        self,
        ax: Axes,
        cube: np.ndarray,
        slice_indices: Tuple[int, int, int],
        config: Optional[PlotConfig] = None,
    ) -> Tuple[AxesImage, Colorbar | None]:
        """Plot a crossline slice.

        Args:
            ax: Matplotlib axis to plot on
            cube: 3D data array (I, J, K)
            slice_indices: Tuple of (idx_i, idx_j, idx_k)
            config: PlotConfig for styling

        Returns:
            Tuple of (image, colorbar)
        """
        config = config or PlotConfig.default()

        _, idx_j, _ = slice_indices
        slice_data = cube[:, idx_j, :]

        config = config.update(
            xlabel="Inline (I)",
            ylabel="Depth Index (K)",
            title=config.title or f"Crossline {idx_j}",
        )

        im, cbar = ImageRenderer.render(ax, slice_data, config)

        self._log_debug(
            f"plotted crossline slice: idx={idx_j}, shape={slice_data.shape}"
        )

        return im, cbar

    def plot_depthslice(
        self,
        ax: Axes,
        cube: np.ndarray,
        slice_indices: Tuple[int, int, int],
        config: Optional[PlotConfig] = None,
    ) -> Tuple[AxesImage, Colorbar | None]:
        """Plot a depth/time slice.

        Args:
            ax: Matplotlib axis to plot on
            cube: 3D data array (I, J, K)
            slice_indices: Tuple of (idx_i, idx_j, idx_k)
            config: PlotConfig for styling

        Returns:
            Tuple of (image, colorbar)
        """
        config = config or PlotConfig.default()

        _, _, idx_k = slice_indices
        slice_data = cube[:, :, idx_k]

        config = config.update(
            xlabel="Inline (I)",
            ylabel="Crossline (J)",
            title=config.title or f"Depth {idx_k}",
        )

        im, cbar = ImageRenderer.render(ax, slice_data, config)

        self._log_debug(f"plotted depth slice: idx={idx_k}, shape={slice_data.shape}")

        return im, cbar

    def plot_3d_slices(
        self,
        ax: Union[Axes, "Axes3D"],
        cube: np.ndarray,
        slice_indices: Tuple[int, int, int],
        config: Optional[PlotConfig] = None,
    ) -> Union[Axes, "Axes3D"]:
        """Plot three orthogonal slices as 3D surfaces.

        Args:
            ax: 3D matplotlib axis (Axes3D)
            cube: 3D data array (I, J, K)
            slice_indices: Tuple of (idx_i, idx_j, idx_k)
            config: PlotConfig for styling

        Returns:
            The 3D axis
        """
        config = config or PlotConfig.default()

        ni, nj, nk = cube.shape
        idx_i, idx_j, idx_k = slice_indices

        # Extract 2D slices
        slice_i = cube[idx_i, :, :]
        slice_j = cube[:, idx_j, :]
        slice_k = cube[:, :, idx_k]

        # Compute normalization
        vmin, vmax = DataNormalizer.compute_limits(
            np.concatenate([slice_i.flat, slice_j.flat, slice_k.flat]),
            percentile=config.percentile,
        )

        denom = vmax - vmin if vmax != vmin else 1.0
        cmap_fn = plt.get_cmap(config.cmap)

        # Create coordinate grids
        J, K = np.mgrid[0:nj, 0:nk]
        I_j, K_j = np.mgrid[0:ni, 0:nk]
        I_k, J_k = np.mgrid[0:ni, 0:nj]

        Xi = np.full_like(J, fill_value=idx_i, dtype=float)
        Xj = I_j
        Yj = np.full_like(I_j, fill_value=idx_j, dtype=float)
        Xk = I_k
        Yk = J_k
        Zk = np.full_like(I_k, fill_value=idx_k * config.k_scale, dtype=float)

        # Plot surfaces (cast to 3D axes for proper typing)
        ax3d = cast(Axes3D, ax)
        ax3d.plot_surface(
            Xi,
            J,
            K * config.k_scale,
            rstride=5,
            cstride=5,
            facecolors=cmap_fn(np.clip((slice_i - vmin) / denom, 0, 1)),
            shade=False,
        )

        ax3d.plot_surface(
            Xj,
            Yj,
            K_j * config.k_scale,
            rstride=5,
            cstride=5,
            facecolors=cmap_fn(np.clip((slice_j - vmin) / denom, 0, 1)),
            shade=False,
        )

        ax3d.plot_surface(
            Xk,
            Yk,
            Zk,
            rstride=5,
            cstride=5,
            facecolors=cmap_fn(np.clip((slice_k - vmin) / denom, 0, 1)),
            shade=False,
        )

        if config.title:
            ax3d.set_title(config.title, fontsize=config.fontsize_title, weight="bold")
        ax3d.set_xlabel("I-axis (Inline)")
        ax3d.set_ylabel("J-axis (Crossline)")
        ax3d.set_zlabel(
            f"{config.k_label} ({config.k_unit})" if config.k_unit else config.k_label
        )
        ax3d.invert_zaxis()

        self._log_debug(
            f"plotted 3d slices: indices=({idx_i}, {idx_j}, {idx_k}), "
            f"vmin={vmin:.2f}, vmax={vmax:.2f}"
        )

        return ax
