# src/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import logging

# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Configure matplotlib to use standard fonts
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]


def plot_3d_slices(ax, cube, slice_indices, title, cmap="seismic"):
    """
    Plots three orthogonal slices of a 3D cube on a Matplotlib 3D axis.

    Args:
        ax (matplotlib.axes.Axes): The 3D subplot axis.
        cube (np.ndarray): The 3D data cube to plot.
        slice_indices (tuple): Indices (i, j, k) for the slices.
        title (str): The title for the subplot.
        cmap (str): The colormap to use.
    """
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

    # Plot each slice as a surface on the 3D axes. Use clipping when mapping to colors
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


def plot_2d_slices(ax, cube, slice_indices, title, cmap="seismic"):
    """Plot three orthogonal 2D slices using imshow (memory-light).

    This is intended as a low-memory alternative to the 3D surface plot.
    """
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
    ax.imshow(slice_i.T, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xlabel("J")
    ax.set_ylabel("K")
