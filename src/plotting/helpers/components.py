"""Reusable plotting components using composition.

Provides focused, single-responsibility classes for common plotting tasks:
- SliceExtractor: Extract slices from 3D cubes
- ColorbarManager: Handle colorbar computation and styling
- AxisStyler: Apply consistent axis styling
- DataNormalizer: Normalize and compute limits for data
"""

import logging
from typing import Any, cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from .config import PlotConfig

logger = logging.getLogger(__name__)

# This module contains small plotting helper functions and classes that
# intentionally expose several styling/configuration parameters. These
# argument-heavy helpers are by-design; disable the argument-count
# and positional-argument checks at module level to reduce noisy reports.


class SliceExtractor:
    """Extract 2D slices from 3D data cubes.

    Handles all logic for extracting inline, crossline, and depth slices
    with proper axis labeling.
    """

    def __init__(self, shape: tuple[int, int, int]):
        """Initialize with 3D data shape.

        Args:
            shape: Tuple of (ni, nj, nk) dimensions
        """
        self.ni, self.nj, self.nk = shape

    def extract_inline(
        self, cube: NDArray[np.floating[Any]], idx: int
    ) -> tuple[NDArray[np.floating[Any]], str, str]:
        """Extract inline slice at inline index idx.

        Args:
            cube: 3D data array (I, J, K)
            idx: Inline index (I axis) - fixes this position

        Returns:
            Tuple of (slice_data, xlabel, ylabel)
            Extracts ``cube[idx, :, :]`` giving shape (J, K).
            Returned array is not transposed: columns are depth (K) and
            rows are crossline (J). Thus the X axis corresponds to depth
            (K) and the Y axis corresponds to crossline (J).
            Shows crossline-depth plane at inline position ``idx``
        """
        return cube[idx, :, :], "Crossline (J)", "Depth Index (K)"

    def extract_crossline(
        self, cube: NDArray[np.floating[Any]], idx: int
    ) -> tuple[NDArray[np.floating[Any]], str, str]:
        """Extract crossline slice at crossline index idx.

        Args:
            cube: 3D data array (I, J, K)
            idx: Crossline index (J axis) - fixes this position

        Returns:
            Tuple of (slice_data, xlabel, ylabel)
            Extracts ``cube[:, idx, :]`` giving shape (I, K).
            Returned array: columns are depth (K) and
            rows are inline (I). Thus the X axis corresponds to depth (K)
            and the Y axis corresponds to inline (I).
            Shows inline-depth plane at crossline position ``idx``
        """
        return cube[:, idx, :], "Inline (I)", "Depth Index (K)"

    def extract_depthslice(
        self, cube: NDArray[np.floating[Any]], idx: int
    ) -> tuple[NDArray[np.floating[Any]], str, str]:
        """Extract depth/time slice at index idx.

        Args:
            cube: 3D data array (I, J, K)
            idx: Depth index (K axis)

        Returns:
            Tuple of (slice_data, xlabel, ylabel)
            Extracts ``cube[:, :, idx]`` giving shape (I, J).
            Returned array is not transposed: columns are crossline (J)
            and rows are inline (I). Thus the X axis corresponds to
            crossline (J) and the Y axis corresponds to inline (I).
            Displays the inline-crossline plane at depth position ``idx``
        """
        return cube[:, :, idx], "Inline (I)", "Crossline (J)"

    def extract_by_orientation(
        self, cube: NDArray[np.floating[Any]], idx: int, orientation: str
    ) -> tuple[NDArray[np.floating[Any]], str, str]:
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
        if orientation == "crossline":
            return self.extract_crossline(cube, idx)
        if orientation in ("depthslice", "timeslice"):
            return self.extract_depthslice(cube, idx)
        raise ValueError(f"Unknown orientation: {orientation}")


class DataNormalizer:
    """Compute data limits and normalize data for visualization."""

    @staticmethod
    def compute_limits(
        data: NDArray[np.floating[Any]],
        is_categorical: bool = False,
        percentile: float = 99.5,
    ) -> tuple[float, float]:
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
    def get_discrete_colormap(n_colors: int = 4) -> Colormap:
        """Get a discrete colormap for categorical data.

        Args:
            n_colors: Number of colors (default: 4 for facies)

        Returns:
            Colormap with n_colors
        """
        cmap = cm.get_cmap("tab10")
        colors = list(cmap(np.linspace(0, 0.4, n_colors)))
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
        # Cast to Any for member calls to avoid third-party stub noise
        ax_any = cast(Any, ax)
        if title:
            ax_any.set_title(title, fontsize=fontsize_title, fontweight="bold")
        if xlabel:
            ax_any.set_xlabel(xlabel, fontsize=fontsize_labels)
        if ylabel:
            ax_any.set_ylabel(ylabel, fontsize=fontsize_labels)
        if grid:
            ax_any.grid(True, alpha=grid_alpha)
        # The signature intentionally exposes several styling options
        # for convenience; suppress argument-count warnings for this
        # UI helper.

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
        plt_any = cast(Any, plt)
        cbar = plt_any.colorbar(im, ax=ax)
        # set_label may be partially unknown in stubs; call via Any
        try:
            cbar.set_label(label, fontsize=10)
        except AttributeError:
            # Some colorbar implementations may not expose set_label
            # in the type stubs; ignore attribute errors only.
            pass
        return cast(Colorbar, cbar)


class ImageRenderer:
    """Render 2D arrays as images with colorbars."""

    # This helper class intentionally exposes a small public surface;
    # silence the too-few-public-methods warning for this focused utility.

    # Also silence argument-count warnings for this rendering helper class.
    # Use symbolic message names instead of numeric codes so pylint
    # recognizes them across configs.

    @staticmethod
    def _determine_categories_and_ncolors(
        data: NDArray[np.floating[Any]], config: PlotConfig
    ) -> tuple[NDArray[Any], int, Colormap, BoundaryNorm]:
        """Determine unique categories, number of colors, colormap and norm.

        Extracted from `render` to keep the main method focused.
        """
        try:
            categories = np.unique(data.astype(int))
        except (ValueError, TypeError):
            categories = np.array([0])

        if config.n_categories is not None:
            n_colors = config.n_categories
        else:
            if categories.size:
                n_colors = int(categories.max()) + 1
            else:
                n_colors = 1

        cmap = DataNormalizer.get_discrete_colormap(n_colors)
        boundaries = np.arange(-0.5, n_colors + 0.5, 1.0)
        norm = BoundaryNorm(boundaries, ncolors=n_colors)
        return categories, n_colors, cmap, norm

    @staticmethod
    def _render_imshow(
        ax: Axes,
        data: NDArray[np.floating[Any]],
        cmap: Any,
        norm: BoundaryNorm | None,
        vmin: float | None,
        vmax: float | None,
        extent: tuple[float, float, float, float] | None,
        interpolation: str,
    ) -> AxesImage:
        """Render `imshow` with a compact parameter set.

        The helper accepts multiple parameters to keep `render` concise.
        """
        # This helper intentionally accepts several parameters; suppress
        # argument-count warnings locally.
        kwargs: dict[str, Any] = {
            "cmap": cmap,
            "origin": "upper",
            "interpolation": interpolation,
            "aspect": "auto",
        }
        if norm is not None:
            kwargs["norm"] = norm
        else:
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax
        if extent is not None:
            kwargs["extent"] = extent
        im = ax.imshow(data, **kwargs)
        return im

    @staticmethod
    def _build_category_legend(
        ax: Axes,
        cmap: Any,
        config: PlotConfig,
        n_colors: int,
        categories: NDArray[Any],
    ) -> None:
        """Build and attach a categorical legend to `ax` when appropriate."""
        if config.category_labels:
            labels_map = config.category_labels
        else:
            labels_map = {int(k): f"Facies {int(k)}" for k in categories}

        patches: list[Any] = []
        for cat in sorted(labels_map.keys()):
            if hasattr(cmap, "colors"):
                try:
                    colors = getattr(cmap, "colors")
                    color = colors[int(cat) % len(colors)]
                except (IndexError, ValueError, TypeError):
                    try:
                        color = cmap(int(cat) / max(1, n_colors - 1))
                    except (TypeError, ValueError):
                        color = (0.5, 0.5, 0.5)
            else:
                try:
                    color = cmap(int(cat) / max(1, n_colors - 1))
                except (TypeError, ValueError):
                    color = (0.5, 0.5, 0.5)

            patches.append(mpatches.Patch(color=color, label=labels_map[cat]))

        if patches:
            ax.legend(
                handles=patches,
                title=config.colorbar_label,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=8,
            )

    # The render helper intentionally exposes several optional arguments;
    # place the pylint disable immediately above the decorator so pylint
    # recognizes it for the following method.

    @staticmethod
    def render(
        ax: Axes,
        data: NDArray[np.floating[Any]],
        config: PlotConfig,
        vmin: float | None = None,
        vmax: float | None = None,
        extent: tuple[float, float, float, float] | None = None,
    ) -> tuple[AxesImage, Colorbar | None]:
        """Render 2D data as image with colorbar.

        Args:
            ax: Matplotlib axis
            data: 2D data array
            config: PlotConfig object

        Returns:
            Tuple of (image, colorbar)
        """
        # For categorical data we use a discrete ListedColormap and a
        # BoundaryNorm so values map to discrete colors. For continuous
        # data we fall back to previous behavior. Break the implementation
        # into small helpers to reduce branching and local counts.
        ax.clear()

        # The `render` function is a public helper with multiple optional
        # parameters (vmin/vmax/extent). Keep the signature stable; this is
        # a convenience entrypoint for callers. Disable the arg-count
        # warning here.

        # Determine colormap, normalization and imshow parameters
        # Pre-declare values to keep static analyzers happy about scoping
        categories: NDArray[Any] = np.array([0])
        n_colors: int = 1
        cmap_used: Any
        norm_used: BoundaryNorm | None = None

        if config.is_categorical:
            (
                categories,
                n_colors,
                cmap_used,
                norm_used,
            ) = ImageRenderer._determine_categories_and_ncolors(data, config)
            im = ImageRenderer._render_imshow(
                ax, data, cmap_used, norm_used, None, None, config.extent, "nearest"
            )
        else:
            # Determine vmin/vmax precedence: explicit args > config > computed
            arg_vmin = vmin
            arg_vmax = vmax
            cfg_vmin = config.vmin
            cfg_vmax = config.vmax

            if arg_vmin is None and arg_vmax is None:
                # fall back to config values if available
                arg_vmin = cfg_vmin
                arg_vmax = cfg_vmax

            if arg_vmin is None or arg_vmax is None:
                arg_vmin, arg_vmax = DataNormalizer.compute_limits(
                    data, is_categorical=False, percentile=config.percentile
                )

            cmap_used = config.cmap
            im = ImageRenderer._render_imshow(
                ax,
                data,
                cmap_used,
                None,
                arg_vmin,
                arg_vmax,
                extent or config.extent,
                "bilinear",
            )

        # Add colorbar if requested
        cbar = None
        if config.show_colorbar:
            cbar = AxisStyler.add_colorbar(im, ax, config.colorbar_label)

        # If categorical, also add a legend mapping category -> color when
        # category labels are available (or derive labels from integers).
        if config.is_categorical:
            ImageRenderer._build_category_legend(
                ax,
                cmap_used,
                config,
                n_colors,
                categories,
            )

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
