"""Unified 2D and 3D slice plotting.

Provides SlicePlotter class for both 2D and 3D orthogonal slice visualizations.
Uses PlotConfig for configuration and ImageRenderer for rendering.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray

from src.plotting.helpers.base import BasePlotter
from src.plotting.helpers.components import DataNormalizer, ImageRenderer
from src.plotting.helpers.config import PlotConfig

logger = logging.getLogger(__name__)


def plot_3d_slices_to_png(
    data: NDArray[np.floating[Any]],
    output_path: Any,  # Path or str
    title: str,
    units: str,
    cmap: str = "viridis",
    dpi: int = 1000,
) -> Any:
    # Allow higher arg counts for convenience in this plotting helper
    # (positional and argument count are by-design for users of the helper).

    """Generate a 3-slice PNG plot (inline, crossline, depth) for 3D data.

    This is a unified plotting function used by both rock physics attributes
    and original property visualization tools. Creates a figure with three
    orthogonal slices through the 3D volume with consistent sizing.

    Parameters
    ----------
    data : NDArray[np.floating[Any]]
        3D data array with shape (ni, nj, nk)
    output_path : Path or str
        Output file path for the PNG image
    title : str
        Main title for the figure
    units : str
        Units for the colorbar label
    cmap : str
        Matplotlib colormap name, default: viridis
    dpi : int
        Resolution in dots per inch, default: 1000

    Returns
    -------
    Path or str
        Path to the generated PNG file

    Examples
    --------
    >>> from pathlib import Path
    >>> import numpy as np
    >>> data = np.random.rand(150, 200, 200)
    >>> output = Path("docs/images/my_plot.png")
    >>> plot_3d_slices_to_png(
    ...     data, output, "P-wave Velocity", "km/s", cmap="viridis"
    ... )
    """
    # Create figure with 3 subplots (inline, crossline, depthslice)
    fig: Figure = plt.figure(figsize=(15, 5))
    # When subplots returns a sequence/iterable of Axes, treat it as a Sequence[Axes]
    axes = cast(Sequence[Axes], fig.subplots(1, 3))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Prepare slice indices and display limits
    mid_i, mid_j, mid_k, vmin, vmax, colorbar_label = _prepare_3d_slice_meta(
        data, title, units
    )

    # Render the three slices onto the axes (helper reduces local variables)
    _render_three_slices_to_axes(
        axes,
        data,
        mid_i,
        mid_j,
        mid_k,
        vmin,
        vmax,
        colorbar_label,
        cmap,
    )

    plt.tight_layout()

    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Saved 3D slice plot: %s", output_path)

    return output_path


def _prepare_3d_slice_meta(
    data: NDArray[np.floating[Any]], title: str, units: str
) -> tuple[int, int, int, float, float, str]:
    """Compute middle indices, display limits and colorbar label for 3D slice plotting.

    Extracted to reduce locals/statements inside the main plotting function.
    """
    ni, nj, nk = data.shape
    mid_i: int = ni // 2
    mid_j: int = nj // 2
    mid_k: int = nk // 2

    # Get data range for consistent colorbar (2nd and 98th percentile)
    vmin, vmax = np.percentile(data, [2, 98])

    # Create colorbar label with units
    colorbar_label: str = f"{title}\n[{units}]"
    return mid_i, mid_j, mid_k, vmin, vmax, colorbar_label


def _render_three_slices_to_axes(
    axes: Sequence[Axes],
    data: NDArray[np.floating[Any]],
    mid_i: int,
    mid_j: int,
    mid_k: int,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    cmap: str,
) -> None:
    """Render three orthogonal imshow slices onto the provided axes.

    Kept as a helper to reduce local variable counts in the caller.
    """
    # Helper accepts many args for clarity; suppress argument-count warnings.

    # axes is already typed as Sequence[Axes]

    ax0 = axes[0]
    im1 = ax0.imshow(
        data[mid_i, :, :],
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax0.set_title(f"Inline (i={mid_i})")
    ax0.set_xlabel("Crossline (j)")
    ax0.set_ylabel("Depth (k)")
    # use the axis' figure to attach a colorbar without needing outer-scope fig
    ax0.figure.colorbar(im1, ax=ax0, label=colorbar_label)

    ax1 = axes[1]
    im2 = ax1.imshow(
        data[:, mid_j, :],
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title(f"Crossline (j={mid_j})")
    ax1.set_xlabel("Inline (i)")
    ax1.set_ylabel("Depth (k)")
    ax1.figure.colorbar(im2, ax=ax1, label=colorbar_label)

    ax2 = axes[2]
    im3 = ax2.imshow(
        data[:, :, mid_k],
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title(f"Depth Slice (k={mid_k})")
    ax2.set_xlabel("Inline (i)")
    ax2.set_ylabel("Crossline (j)")
    ax2.figure.colorbar(im3, ax=ax2, label=colorbar_label)


def _plot_3d_surface(
    ax3d_any: Any,
    x: NDArray[Any],
    y: NDArray[Any],
    z: NDArray[Any],
    slice_data: NDArray[Any],
    vmin: float,
    denom: float,
    cmap_fn: Any,
    *,
    k_scale: float = 1.0,
) -> None:
    """Plot a single 3D surface on the provided 3D axis.

    Accepts `Z` unscaled and applies `k_scale` to allow reusing this helper
    for surfaces that require z-scaling.
    """
    facecolors = cmap_fn(np.clip((slice_data - vmin) / denom, 0, 1))
    ax3d_any.plot_surface(
        x,
        y,
        z * k_scale,
        rstride=5,
        cstride=5,
        facecolors=facecolors,
        shade=False,
    )

    # Use short variable names for grid coordinates (x, y, z) — lowercase
    # names satisfy the linting conventions while keeping the code concise.


class SlicePlotter(BasePlotter):
    """Plotter for 2D and 3D orthogonal slice visualizations.

    Handles inline, crossline, and time/depth slices for seismic data.
    Supports both 2D and 3D visualizations with consistent styling.
    Uses composition with SliceExtractor and ImageRenderer components.
    """

    def plot_2d_slices(
        self,
        ax: Axes,
        cube: NDArray[np.floating[Any]],
        slice_indices: tuple[int, int, int],
        config: PlotConfig | None = None,
    ) -> tuple[AxesImage, Colorbar | None]:
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
        slice_data: NDArray[Any] = cube[idx_i, :, :]
        # Ensure we pass a concrete ndarray to downstream renderers
        slice_data = np.asarray(slice_data)

        # Update config with slice information
        config = config.update(
            xlabel="Crossline (J)",
            ylabel="Depth Index (K)",
            title=config.title or f"Inline {idx_i}",
        )

        # Render with components
        im, cbar = ImageRenderer.render(ax, slice_data, config)

        self.logger.debug(
            "plotted inline slice: idx=%d, shape=%s", idx_i, slice_data.shape
        )

        return im, cbar

    def plot_crossline(
        self,
        ax: Axes,
        cube: NDArray[np.floating[Any]],
        slice_indices: tuple[int, int, int],
        config: PlotConfig | None = None,
    ) -> tuple[AxesImage, Colorbar | None]:
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
        slice_data: NDArray[Any] = cube[:, idx_j, :]
        # Ensure we pass a concrete ndarray to downstream renderers
        slice_data = np.asarray(slice_data)

        config = config.update(
            xlabel="Inline (I)",
            ylabel="Depth Index (K)",
            title=config.title or f"Crossline {idx_j}",
        )

        im, cbar = ImageRenderer.render(ax, slice_data, config)

        self.logger.debug(
            "plotted crossline slice: idx=%d, shape=%s", idx_j, slice_data.shape
        )

        return im, cbar

    def plot_depthslice(
        self,
        ax: Axes,
        cube: NDArray[np.floating[Any]],
        slice_indices: tuple[int, int, int],
        config: PlotConfig | None = None,
    ) -> tuple[AxesImage, Colorbar | None]:
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
        slice_data: NDArray[Any] = cube[:, :, idx_k]
        # Ensure we pass a concrete ndarray to downstream renderers
        slice_data = np.asarray(slice_data)

        config = config.update(
            xlabel="Inline (I)",
            ylabel="Crossline (J)",
            title=config.title or f"Depth {idx_k}",
        )

        im, cbar = ImageRenderer.render(ax, slice_data, config)

        self.logger.debug(
            "plotted depth slice: idx=%d, shape=%s", idx_k, slice_data.shape
        )

        return im, cbar

    def plot_3d_slices(
        self,
        ax: Axes | Axes3D,
        cube: NDArray[np.floating[Any]],
        slice_indices: tuple[int, int, int],
        config: PlotConfig | None = None,
    ) -> Axes | Axes3D:
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

        # Use compact local names to reduce total local variable count
        slices_meta = self._extract_and_normalize_slices(
            cube, slice_indices[0], slice_indices[1], slice_indices[2], config
        )

        grids = self._build_3d_grids(cube.shape, slice_indices, config)

        ax3d_any = cast(Any, ax)

        # grids is a tuple of three grid-tuples: ( (Xi,J,K), (Xj,Yj,K_j), (Xk,Yk,Zk) )
        _plot_3d_surface(
            ax3d_any,
            *grids[0],
            slices_meta[0],
            slices_meta[3],
            slices_meta[5],
            slices_meta[6],
            k_scale=config.k_scale,
        )

        _plot_3d_surface(
            ax3d_any,
            *grids[1],
            slices_meta[1],
            slices_meta[3],
            slices_meta[5],
            slices_meta[6],
            k_scale=config.k_scale,
        )

        _plot_3d_surface(
            ax3d_any,
            *grids[2],
            slices_meta[2],
            slices_meta[3],
            slices_meta[5],
            slices_meta[6],
            k_scale=1.0,
        )

        if config.title:
            ax3d_any.set_title(
                config.title, fontsize=config.fontsize_title, weight="bold"
            )
        ax3d_any.set_xlabel("I-axis (Inline)")
        ax3d_any.set_ylabel("J-axis (Crossline)")
        ax3d_any.set_zlabel(
            f"{config.k_label} ({config.k_unit})" if config.k_unit else config.k_label
        )
        ax3d_any.invert_zaxis()

        self.logger.debug(
            "plotted 3d slices: indices=(%d, %d, %d), vmin=%.2f, vmax=%.2f",
            slice_indices[0],
            slice_indices[1],
            slice_indices[2],
            slices_meta[3],
            slices_meta[4],
        )

        return ax

    def _extract_and_normalize_slices(
        self,
        cube: NDArray[np.floating[Any]],
        idx_i: int,
        idx_j: int,
        idx_k: int,
        config: PlotConfig,
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        float,
        float,
        float,
        Any,
    ]:
        """Extract inline/crossline/depth slices and compute vmin/vmax,
        denom and cmap function.

        """
        slice_i = np.asarray(cube[idx_i, :, :])
        slice_j = np.asarray(cube[:, idx_j, :])
        slice_k = np.asarray(cube[:, :, idx_k])

        vmin, vmax = DataNormalizer.compute_limits(
            np.concatenate([slice_i.flat, slice_j.flat, slice_k.flat]),
            percentile=config.percentile,
        )

        denom = vmax - vmin if vmax != vmin else 1.0
        plt_any = cast(Any, plt)
        cmap_fn = plt_any.get_cmap(config.cmap)
        return slice_i, slice_j, slice_k, vmin, vmax, denom, cmap_fn

    def _build_3d_grids(
        self,
        shape: tuple[int, int, int],
        slice_indices: tuple[int, int, int],
        config: PlotConfig,
    ) -> tuple[
        tuple[NDArray[Any], NDArray[Any], NDArray[Any]],
        tuple[NDArray[Any], NDArray[Any], NDArray[Any]],
        tuple[NDArray[Any], NDArray[Any], NDArray[Any]],
    ]:
        """Build and return grouped coordinate grids used for 3D surface plotting.

        Returns three grid-tuples: (Xi, J, K), (Xj, Yj, K_j), (Xk, Yk, Zk)
        """
        ni, nj, nk = shape
        idx_i, idx_j, idx_k = slice_indices

        J: NDArray[Any] = np.mgrid[0:nj, 0:nk][0]
        K: NDArray[Any] = np.mgrid[0:nj, 0:nk][1]
        I_j: NDArray[Any] = np.mgrid[0:ni, 0:nk][0]
        K_j: NDArray[Any] = np.mgrid[0:ni, 0:nk][1]
        I_k: NDArray[Any] = np.mgrid[0:ni, 0:nj][0]
        J_k: NDArray[Any] = np.mgrid[0:ni, 0:nj][1]

        Xi: NDArray[Any] = np.full_like(J, fill_value=idx_i, dtype=float)
        Xj: NDArray[Any] = I_j
        Yj: NDArray[Any] = np.full_like(I_j, fill_value=idx_j, dtype=float)
        Xk: NDArray[Any] = I_k
        Yk: NDArray[Any] = J_k
        Zk: NDArray[Any] = np.full_like(
            I_k, fill_value=idx_k * config.k_scale, dtype=float
        )

        return (Xi, J, K), (Xj, Yj, K_j), (Xk, Yk, Zk)
