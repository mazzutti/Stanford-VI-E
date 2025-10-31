"""Visualization helpers for 2D/3D plotting and display.

Provides thin facade methods for common visualization tasks used by the
plotting modules.
"""

# Lightweight imports must be at module top to satisfy linters (E402).
import logging

from src.plotting.helpers.plot import init_plotting
from src.utils.facades import LazyObjectProxy

logger = logging.getLogger(__name__)

# Configure matplotlib and get pyplot + numpy
plt, np = init_plotting(backend=None)

__all__ = ["plot_visualization", "get_plot_visualization"]


class PlotVisualization:
    """Thin facade grouping 2D/3D plotting helpers for visualization.

    Methods mirror the module-level functions so callers can use an
    instance-based API while the old function names remain available.
    """

    def plot_3d_slices(self, ax, cube, slice_indices, title, cmap="seismic"):
        ni, nj, nk = cube.shape
        idx_i, idx_j, idx_k = slice_indices

        # Extract the 2D data for each slice
        slice_i = cube[idx_i, :, :]
        slice_j = cube[:, idx_j, :]
        slice_k = cube[:, :, idx_k]

        # Create coordinate grids for each slice plane
        J, K = np.mgrid[0:nj, 0:nk]
        I_j, K_j = np.mgrid[0:ni, 0:nk]
        I_k, J_k = np.mgrid[0:ni, 0:nj]

        # Normalize colors for consistent display
        if cmap == "seismic":
            vmax = np.percentile(np.abs(cube), 98)
            vmin = -vmax
        else:
            vmin, vmax = np.min(cube), np.max(cube)

        # Ensure denom is non-zero for normalization
        denom = vmax - vmin
        if denom == 0:
            denom = 1.0

        # Build X/Y/Z grids with shapes matching each slice
        Xi = np.full_like(J, fill_value=idx_i, dtype=float)
        Xj = I_j
        Yj = np.full_like(I_j, fill_value=idx_j, dtype=float)
        Xk = I_k
        Yk = J_k
        Zk = np.full_like(I_k, fill_value=idx_k, dtype=float)

        # Plot each slice as a surface on the 3D axes. Use clipping when
        # mapping to colors
        cmap_fn = plt.get_cmap(cmap)

        ax.plot_surface(
            Xi,
            J,
            K,
            rstride=5,
            cstride=5,
            facecolors=cmap_fn(np.clip((slice_i - vmin) / denom, 0, 1)),
            shade=False,
        )

        ax.plot_surface(
            Xj,
            Yj,
            K_j,
            rstride=5,
            cstride=5,
            facecolors=cmap_fn(np.clip((slice_j - vmin) / denom, 0, 1)),
            shade=False,
        )

        ax.plot_surface(
            Xk,
            Yk,
            Zk,
            rstride=5,
            cstride=5,
            facecolors=cmap_fn(np.clip((slice_k - vmin) / denom, 0, 1)),
            shade=False,
        )

        ax.set_title(title)
        ax.set_xlabel("I-axis (Inline)")
        ax.set_ylabel("J-axis (Crossline)")
        ax.set_zlabel("K-axis (Time/Depth)")
        ax.invert_zaxis()

    def plot_2d_slices(self, ax, cube, slice_indices, title, cmap="seismic"):
        ni, nj, nk = cube.shape
        idx_i, idx_j, idx_k = slice_indices

        # Extract slices (works with memmap without loading entire array)
        slice_i = cube[idx_i, :, :]
        slice_j = cube[:, idx_j, :]
        slice_k = cube[:, :, idx_k]

        # Determine consistent vmin/vmax from the three slices only to avoid
        # forcing reads of the whole memory-mapped array
        if cmap == "seismic":
            p_i = np.percentile(np.abs(slice_i), 98)
            p_j = np.percentile(np.abs(slice_j), 98)
            p_k = np.percentile(np.abs(slice_k), 98)
            vmax = float(max(p_i, p_j, p_k))
            vmin = -vmax
        else:
            vmax = float(max(np.max(slice_i), np.max(slice_j), np.max(slice_k)))
            vmin = float(min(np.min(slice_i), np.min(slice_j), np.min(slice_k)))
        if vmax == vmin:
            vmax = vmin + 1.0

        # Plot three images side-by-side inside the provided Axes (clear first)
        ax.clear()
        ax.set_title(title)
        # We will use a simple layout: show i-slice, j-slice (vertical), k-slice
        # Create a mini-grid with imshow; treat axes as normal 2D
        from src.plotting.helpers.plot import imshow_with_labels

        imshow_with_labels(
            ax,
            slice_i,
            title,
            xlabel="J",
            k_label="K",
            k_unit="",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )


# Module-level lazy proxy for the visualization facade
plot_visualization = LazyObjectProxy(lambda: PlotVisualization())


def get_plot_visualization(config: dict | None = None):
    """Return the module-level `plot_visualization` proxy when `config` is None,
    otherwise return a new `PlotVisualization` instance. This keeps access
    patterns consistent with other helpers.
    """
    return _impl_get_plot_visualization(config)


def _impl_get_plot_visualization(config: dict | None = None):
    if config is None:
        return plot_visualization
    return PlotVisualization()
