"""Reusable plotting components using composition.

Provides focused, single-responsibility classes for common plotting tasks:
- SliceExtractor: Extract slices from 3D cubes
- ColorbarManager: Handle colorbar computation and styling
- AxisStyler: Apply consistent axis styling
- DataNormalizer: Normalize and compute limits for data
"""

import logging
from typing import Any, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Colormap
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar

from .config import PlotConfig

logger = logging.getLogger(__name__)


class SliceExtractor:
    """Extract 2D slices from 3D data cubes.

    Handles all logic for extracting inline, crossline, and depth slices
    with proper axis labeling.
    """

    def __init__(self, shape: Tuple[int, int, int]):
        """Initialize with 3D data shape.

        Args:
            shape: Tuple of (ni, nj, nk) dimensions
        """
        self.ni, self.nj, self.nk = shape

    def extract_inline(self, cube: NDArray[np.floating[Any]], idx: int) -> Tuple[NDArray[np.floating[Any]], str, str]:
        """Extract inline slice at index idx.

        Args:
            cube: 3D data array (I, J, K)
            idx: Inline index

        Returns:
            Tuple of (slice_data, xlabel, ylabel)
        """
        return cube[idx, :, :], "Crossline (J)", "Depth Index (K)"

    def extract_crossline(
        self, cube: NDArray[np.floating[Any]], idx: int
    ) -> Tuple[NDArray[np.floating[Any]], str, str]:
        """Extract crossline slice at index idx.

        Args:
            cube: 3D data array (I, J, K)
            idx: Crossline index

        Returns:
            Tuple of (slice_data, xlabel, ylabel)
        """
        return cube[:, idx, :], "Inline (I)", "Depth Index (K)"

    def extract_depthslice(
        self, cube: NDArray[np.floating[Any]], idx: int
    ) -> Tuple[NDArray[np.floating[Any]], str, str]:
        """Extract depth/time slice at index idx.

        Args:
            cube: 3D data array (I, J, K)
            idx: Depth index

        Returns:
            Tuple of (slice_data, xlabel, ylabel)
        """
        return cube[:, :, idx], "Inline (I)", "Crossline (J)"

    def extract_by_orientation(
        self, cube: NDArray[np.floating[Any]], idx: int, orientation: str
    ) -> Tuple[NDArray[np.floating[Any]], str, str]:
        """Extract slice based on orientation string.

        Args:
            cube: 3D data array (I, J, K)
            idx: Slice index
            orientation: 'inline', 'crossline', or 'depthslice'

        Returns:
            Tuple of (slice_data, xlabel, ylabel)

        Raises:
            ValueError: If orientation not recognized
        """
        if orientation == "inline":
            return self.extract_inline(cube, idx)
        elif orientation == "crossline":
            return self.extract_crossline(cube, idx)
        elif orientation in ("depthslice", "timeslice"):
            return self.extract_depthslice(cube, idx)
        else:
            raise ValueError(f"Unknown orientation: {orientation}")


class DataNormalizer:
    """Compute data limits and normalize data for visualization."""

    @staticmethod
    def compute_limits(
        data: NDArray[np.floating[Any]],
        is_categorical: bool = False,
        percentile: float = 99.5,
    ) -> Tuple[float, float]:
        """Compute vmin/vmax for data visualization.

        Args:
            data: Data array to analyze
            is_categorical: If True, use categorical limits (0, n_categories)
            percentile: Percentile for limit computation (0-100)

        Returns:
            Tuple of (vmin, vmax)
        """
        if is_categorical:
            return 0.0, 3.0

        p = np.percentile(np.abs(data), percentile)
        vmax = float(p)
        vmin = -vmax
        if vmax == vmin:
            vmax = vmin + 1.0
        return vmin, vmax

    @staticmethod
    def get_discrete_colormap(n_colors: int = 4) -> ListedColormap:
        """Get a discrete colormap for categorical data.

        Args:
            n_colors: Number of colors (default: 4 for facies)

        Returns:
            ListedColormap with n_colors
        """
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 0.4, n_colors))
        return ListedColormap(colors)


class AxisStyler:
    """Apply consistent styling to matplotlib axes."""

    @staticmethod
    def style_axis(
        ax: Axes,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        fontsize_title: int = 12,
        fontsize_labels: int = 10,
        grid: bool = True,
        grid_alpha: float = 0.3,
    ) -> None:
        """Apply consistent styling to an axis.

        Args:
            ax: Matplotlib axis
            title: Title text
            xlabel: X-axis label
            ylabel: Y-axis label
            fontsize_title: Title font size
            fontsize_labels: Label font sizes
            grid: Whether to show grid
            grid_alpha: Grid alpha transparency
        """
        if title:
            ax.set_title(title, fontsize=fontsize_title, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize_labels)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize_labels)
        if grid:
            ax.grid(True, alpha=grid_alpha)

    @staticmethod
    def add_colorbar(im: AxesImage, ax: Axes, label: str = "Value") -> Colorbar:
        """Add colorbar to axis with consistent styling.

        Args:
            im: Mappable image object
            ax: Matplotlib axis
            label: Colorbar label

        Returns:
            Colorbar object
        """
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label, fontsize=10)
        return cbar


class ImageRenderer:
    """Render 2D arrays as images with colorbars."""

    @staticmethod
    def render(
        ax: Axes,
        data: NDArray[np.floating[Any]],
        config: PlotConfig,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        extent: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[AxesImage, Colorbar | None]:
        """Render 2D data as image with colorbar.

        Args:
            ax: Matplotlib axis
            data: 2D data array
            config: PlotConfig object
            vmin: Minimum colorbar value (computed if None)
            vmax: Maximum colorbar value (computed if None)
            extent: Extent for imshow

        Returns:
            Tuple of (image, colorbar)
        """
        if vmin is None or vmax is None:
            vmin, vmax = DataNormalizer.compute_limits(
                data, is_categorical=config.is_categorical, percentile=config.percentile
            )

        # Choose colormap and interpolation based on data type
        cmap: Colormap | str
        if config.is_categorical:
            cmap = DataNormalizer.get_discrete_colormap(4)
            interpolation = "nearest"
        else:
            cmap = config.cmap
            interpolation = "bilinear"

        # Render image
        ax.clear()
        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            interpolation=interpolation,
            extent=extent,
        )

        # Add colorbar if requested
        cbar = None
        if config.show_colorbar:
            cbar = AxisStyler.add_colorbar(im, ax, config.colorbar_label)

        # Style the axis
        AxisStyler.style_axis(
            ax,
            title=config.title,
            xlabel=config.xlabel,
            ylabel=config.ylabel,
            fontsize_title=config.fontsize_title,
            fontsize_labels=config.fontsize_labels,
            grid=config.grid,
            grid_alpha=config.grid_alpha,
        )

        return im, cbar
