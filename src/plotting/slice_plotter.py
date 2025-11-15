"""Unified 2D and 3D slice plotting.

Provides SlicePlotter class for both 2D and 3D orthogonal slice visualizations.
Uses PlotConfig for configuration and ImageRenderer for rendering.
"""

import logging
from typing import Any, cast
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure
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


def plot_3d_slices_to_png(
    data: NDArray[np.floating[Any]],
    output_path: Any,  # Path or str
    title: str,
    units: str,
    cmap: str = "viridis",
    dpi: int = 1000,
) -> Any:
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

    # Get middle indices for slicing
    ni, nj, nk = data.shape
    mid_i: int = ni // 2
    mid_j: int = nj // 2
    mid_k: int = nk // 2

    # Get data range for consistent colorbar (2nd and 98th percentile)
    vmin: float
    vmax: float
    vmin, vmax = np.percentile(data, [2, 98])

    # Create colorbar label with units
    colorbar_label: str = f"{title}\n[{units}]"

    # Inline slice (constant i)
    im1: AxesImage = axes[0].imshow(
        data[mid_i, :, :].T,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax0 = axes[0]
    ax0.set_title(f"Inline (i={mid_i})")
    ax0.set_xlabel("Crossline (j)")
    ax0.set_ylabel("Depth (k)")
    _ = fig.colorbar(im1, ax=ax0, label=colorbar_label)

    # Crossline slice (constant j)
    im2: AxesImage = axes[1].imshow(
        data[:, mid_j, :].T,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax1 = axes[1]
    ax1.set_title(f"Crossline (j={mid_j})")
    ax1.set_xlabel("Inline (i)")
    ax1.set_ylabel("Depth (k)")
    _ = fig.colorbar(im2, ax=ax1, label=colorbar_label)

    # Depth slice (constant k)
    im3: AxesImage = axes[2].imshow(
        data[:, :, mid_k].T,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax2 = axes[2]
    ax2.set_title(f"Depth Slice (k={mid_k})")
    ax2.set_xlabel("Inline (i)")
    ax2.set_ylabel("Crossline (j)")
    _ = fig.colorbar(im3, ax=ax2, label=colorbar_label)

    plt.tight_layout()

    # Save plot
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.debug(f"Saved 3D slice plot: {output_path}")

    return output_path


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

        self._log_debug(f"plotted inline slice: idx={idx_i}, shape={slice_data.shape}")

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

        self._log_debug(
            f"plotted crossline slice: idx={idx_j}, shape={slice_data.shape}"
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

        self._log_debug(f"plotted depth slice: idx={idx_k}, shape={slice_data.shape}")

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

        ni, nj, nk = cube.shape
        idx_i, idx_j, idx_k = slice_indices

        # Extract 2D slices
        slice_i = np.asarray(cube[idx_i, :, :])
        slice_j = np.asarray(cube[:, idx_j, :])
        slice_k = np.asarray(cube[:, :, idx_k])

        # Compute normalization
        vmin, vmax = DataNormalizer.compute_limits(
            np.concatenate([slice_i.flat, slice_j.flat, slice_k.flat]),
            percentile=config.percentile,
        )

        denom = vmax - vmin if vmax != vmin else 1.0
        plt_any = cast(Any, plt)
        cmap_fn = plt_any.get_cmap(config.cmap)

        # Create coordinate grids and annotate explicitly
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

        # Plot surfaces (cast to 3D axes for proper typing)
        # Avoid referencing Axes3D at runtime (it's TYPE_CHECKING-only); use Any
        ax3d_any = cast(Any, ax)
        ax3d_any.plot_surface(
            Xi,
            J,
            K * config.k_scale,
            rstride=5,
            cstride=5,
            facecolors=cmap_fn(np.clip((slice_i - vmin) / denom, 0, 1)),
            shade=False,
        )

        ax3d_any.plot_surface(
            Xj,
            Yj,
            K_j * config.k_scale,
            rstride=5,
            cstride=5,
            facecolors=cmap_fn(np.clip((slice_j - vmin) / denom, 0, 1)),
            shade=False,
        )

        ax3d_any.plot_surface(
            Xk,
            Yk,
            Zk,
            rstride=5,
            cstride=5,
            facecolors=cmap_fn(np.clip((slice_k - vmin) / denom, 0, 1)),
            shade=False,
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

        self._log_debug(
            f"plotted 3d slices: indices=({idx_i}, {idx_j}, {idx_k}), "
            f"vmin={vmin:.2f}, vmax={vmax:.2f}"
        )

        return ax
