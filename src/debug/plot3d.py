"""3D debug plotting utilities.

Matplotlib-based helpers for quick developer visualization of 3D
volumes. The functions are intentionally minimal and robust for
headless environments.

Functions:
- ``plot_volume``: render three orthogonal slices on a 3D axis.
- ``plot_voxels``: scatter voxels above a threshold.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

logger = logging.getLogger(__name__)

# Debug plotting helpers perform several lazy imports for optional plotting
# helpers and to avoid heavy dependencies during import. These deferred
# imports are intentional; silence import-order warnings at module level
# with a short justification. This module provides developer-facing
# visualization helpers that intentionally accept many small parameters
# and perform local orchestration; disable the related style checks.
# Also silence duplicate-code in this developer/debug-only module which
# intentionally mirrors small helper snippets present elsewhere in the
# codebase.
# pylint: disable=duplicate-code

__all__ = [
    "plot_volume",
]

# Type for categorical color information returned by the colormap helper.
# Expected to be either `None` for continuous data or a tuple
# `(categories, mapping, n_categories)`. The concrete implementation may
# use NumPy arrays for categories/mapping; use a permissive alias to avoid
# spurious type-checker errors when those are `ndarray`.
CategoricalInfo = tuple[Any, Any, int] | None


# map slice values to indices using mapping (fallback to 0)
def map_slice_to_indices(arr: np.ndarray, mapping: dict[float, int]) -> np.ndarray:
    """Map values in arr to integer indices using mapping.

    This avoids use of np.vectorize which can trigger static
    analyzer complaints about the callable signature.
    """
    out = np.zeros(arr.shape, dtype=int)
    # use isclose for float comparisons to be robust to FP
    for key, val in mapping.items():
        out[np.isclose(arr, key)] = val
    return out


def project_and_sort_faces(
    ax: Axes3D,
    faces: list[list[tuple[float, float, float]]],
    face_colors: list[tuple[float, float, float, float]],
) -> tuple[
    list[list[tuple[float, float, float]]], list[tuple[float, float, float, float]]
]:
    """Project face centers using the axes projection and return sorted faces
    and colors from back to front for correct alpha compositing.
    """
    proj_matrix = cast(np.ndarray, getattr(ax, "get_proj")())
    centers = [np.mean(np.asarray(f), axis=0) for f in faces]
    depths: list[float] = []
    for c in centers:
        vec = np.array([c[0], c[1], c[2], 1.0])
        proj = proj_matrix.dot(vec)
        depths.append(float(proj[2]))

    depths_arr = np.asarray(depths, dtype=float)
    order = np.argsort(depths_arr)
    sorted_faces = [faces[int(k)] for k in order]
    sorted_colors = [face_colors[int(k)] for k in order]
    return sorted_faces, sorted_colors


def make_and_add_poly_collection(
    ax: Axes3D,
    sorted_faces: list[list[tuple[float, float, float]]],
    sorted_colors: list[tuple[float, float, float, float]],
) -> None:
    """Create a Poly3DCollection from sorted faces/colors and add to `ax`."""
    poly = Poly3DCollection(
        sorted_faces, facecolors=sorted_colors, linewidths=0, antialiased=False
    )
    ax.add_collection3d(poly)


def make_colorbar(
    fig: Figure,
    ax: Axes3D,
    cmap_func: Callable[[np.ndarray], np.ndarray],
    norm: Normalize | None,
    categorical_info: CategoricalInfo,
) -> None:
    """Add a colorbar to `fig`/`ax` handling categorical and continuous cases.

    Kept as a small helper to reduce the statement/local count in
    `plot_volume`.
    """
    # Delegate creation of the ScalarMappable and any categorical metadata
    # to the plotting helpers module to reduce local imports and branching
    # inside this module.
    # Lazy import for plotting helper: colorbar creation
    from src.plotting.helpers.colors import (
        prepare_colorbar_mappable,
    )

    sm, bounds, ticks, ticklabels = prepare_colorbar_mappable(
        cmap_func, norm, categorical_info
    )
    sm.set_array([])
    if bounds is None:
        fig.colorbar(sm, ax=ax, shrink=0.6)
        return

    cb = fig.colorbar(sm, ax=ax, boundaries=bounds, ticks=ticks, shrink=0.6)
    if ticklabels is not None:
        # Type checkers may report an incompatible type for the
        # `set_ticklabels` parameter; cast to Any to satisfy static
        # analysis while preserving runtime behavior.
        cb.set_ticklabels(cast(Any, ticklabels))


def compute_slice_facecolors(
    slice_arr: np.ndarray,
    cmap_func: Callable[[np.ndarray], np.ndarray],
    norm: Normalize | None,
    alpha: float | None,
    categorical_info: CategoricalInfo,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-texel RGBA `colors` and per-face averaged `fc` for a slice.

    Returns `(colors, fc)` where `colors` is the RGBA array matching
    the slice shape and `fc` is the per-face RGBA array used for faces.
    """
    # Import locally to avoid top-level import cycles and to keep this helper
    # independent of module-level import ordering.
    # Lazy import for color mapping helper
    from src.plotting.helpers.colors import (
        slice_to_rgba,
    )

    colors = slice_to_rgba(slice_arr, cmap_func, norm, alpha, categorical_info)
    fc = (colors[:-1, :-1] + colors[1:, :-1] + colors[:-1, 1:] + colors[1:, 1:]) / 4.0
    return colors, fc


def set_reverse_crossline_ticks(
    ax: Axes3D, y: np.ndarray, reverse_crossline: bool
) -> None:
    """If requested, set Y-axis tick labels reversed to go max->min.

    Moved out of `plot_volume` to reduce locals/statements there.
    """
    if not reverse_crossline:
        return
    ymin = float(y.min())
    ymax = float(y.max())
    ticks = np.asarray(getattr(ax, "get_yticks")(), dtype=float)
    mapped = ymin + ymax - ticks
    if np.allclose(mapped, np.round(mapped)):
        labels = [str(int(round(v + 1))) for v in mapped]
    else:
        labels = [f"{v:.3f}".rstrip("0").rstrip(".") for v in mapped]
    # Ensure ticks are set explicitly before setting labels to avoid
    # a Matplotlib UserWarning about calling set_ticklabels without
    # a fixed number of ticks (this is headless/test-friendly).
    getattr(ax, "set_yticks")(ticks)
    getattr(ax, "set_yticklabels")(labels)


def _build_and_add_face_collection(
    ax: Axes3D,
    coords: Any,
    fc_xy: np.ndarray,
    fc_xz: np.ndarray,
    fc_yz: np.ndarray,
) -> None:
    """Collect quad faces from the mesh grids, compute per-face colors
    and add a Poly3DCollection to ``ax``.

    This encapsulates the face collection assembly to reduce the number
    of local variables inside ``plot_volume``.
    """
    # import mesh helper locally to avoid creating a top-level dependency
    # that could reintroduce import cycles when the package is imported.
    # Lazy import for mesh helper used when building face collections
    from src.plotting.helpers.mesh import (
        add_quads_to_lists,
    )

    faces: list[list[tuple[float, float, float]]] = []
    face_colors: list[tuple[float, float, float, float]] = []

    # collect quads and per-face colors using mesh helper (pass plane objects)
    add_quads_to_lists(faces, face_colors, coords.grids.xy, fc_xy)
    add_quads_to_lists(faces, face_colors, coords.grids.xz, fc_xz)
    add_quads_to_lists(faces, face_colors, coords.grids.yz, fc_yz)

    # project face centers and sort by depth for correct alpha compositing
    sorted_faces, sorted_colors = project_and_sort_faces(ax, faces, face_colors)
    make_and_add_poly_collection(ax, sorted_faces, sorted_colors)


def _prepare_coords_and_grids(
    shape: tuple[int, int, int],
    idxs: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Compute and return coords/grids and primary 1D coordinate arrays.

    Kept as a small wrapper so `plot_volume` does not directly import the
    mesh helper and accumulate locals/assignment statements.
    """
    # Lazy import for mesh grid computation
    from src.plotting.helpers.mesh import (
        compute_coords_and_grids,
    )

    coords = compute_coords_and_grids(shape, idxs, spacing)
    x = coords.coords.x
    y = coords.coords.y
    z = coords.coords.z
    return coords, x, y, z


def _prepare_figure_ax(
    figsize: tuple[float, float] = (9, 7)
) -> tuple[Figure, Axes3D, bool]:
    """Create figure and 3D axes and return them with a created_fig flag."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    return fig, cast(Axes3D, ax), True


def _compute_facecolors_for_slices(
    slices: tuple[np.ndarray, np.ndarray, np.ndarray],
    cmap_args: tuple[
        Callable[[np.ndarray], np.ndarray],
        Normalize | None,
        float | None,
        CategoricalInfo,
    ],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-face averaged RGBA arrays for the three slices."""
    slice_xy, slice_xz, slice_yz = slices
    cmap_func, norm, alpha, categorical_info = cmap_args
    _, fc_xy = compute_slice_facecolors(
        slice_xy, cmap_func, norm, alpha, categorical_info
    )
    _, fc_xz = compute_slice_facecolors(
        slice_xz, cmap_func, norm, alpha, categorical_info
    )
    _, fc_yz = compute_slice_facecolors(
        slice_yz, cmap_func, norm, alpha, categorical_info
    )
    return fc_xy, fc_xz, fc_yz


def select_slices_and_indices(
    volume: np.ndarray, indices: tuple[int, int, int] | None = None
) -> tuple[
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Return shape, selected indices and the three orthogonal slices.

    This centralizes the small piece of orchestration that selects the
    middle (or explicit) inline/crossline/depth indices and extracts the
    corresponding slices. Extracting it reduces the number of local
    variables in `plot_volume` and makes unit testing this logic easier.
    """
    if volume.ndim != 3:
        raise ValueError("volume must be a 3D numpy array")

    nx, ny, nz = volume.shape
    if indices is None:
        iz = nz // 2
        iy = ny // 2
        ix = nx // 2
    else:
        ix, iy, iz = indices

    slice_xy = volume[:, :, iz]
    slice_xz = volume[:, iy, :]
    slice_yz = volume[ix, :, :]

    return (nx, ny, nz), (ix, iy, iz), (slice_xy, slice_xz, slice_yz)


def axis_setup(
    ax: Axes3D,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    swap_xy: bool,
    reverse_crossline: bool,
) -> None:
    """Apply axis labels, autoscaling and optional view/tick adjustments.

    This consolidates several small steps from `plot_volume` to reduce
    the function's statement and local-variable count.
    """
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.auto_scale_xyz([x.min(), x.max()], [y.min(), y.max()], [z.min(), z.max()])

    if swap_xy:
        # Rotate view so X/Y appear swapped visually without modifying data.
        try:
            ax.view_init(azim=-45, roll=-180, elev=-45)
        except TypeError:
            # Some Matplotlib versions may not accept 'roll'; ignore safely.
            ax.view_init(azim=-45, elev=-45)

    # Reverse crossline labels if requested
    set_reverse_crossline_ticks(ax, y, reverse_crossline)


def save_and_show(save_path: str | None, show: bool, created_fig: bool) -> None:
    """Handle saving and optionally showing the created figure.

    Kept separate to reduce `plot_volume` local branching and make this
    behavior testable independently.
    """
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show and created_fig:
        # Avoid calling `plt.show()` in non-interactive/backends (e.g. Agg)
        # which emit a UserWarning in headless test environments.
        try:
            import matplotlib as mpl

            get_backend = getattr(mpl, "get_backend", None)
            backend = get_backend() if callable(get_backend) else ""
            backend = str(backend).lower()
            is_interactive = getattr(mpl, "is_interactive", None)
            interactive = is_interactive() if callable(is_interactive) else False
            if "agg" in backend or not interactive:
                return
        except (ImportError, AttributeError, TypeError):
            # If backend info isn't available for a known reason, skip show
            return

        plt.show()


def plot_volume(
    volume: np.ndarray,
    cmap: str = "viridis",
    alpha: float | None = 0.7,
    is_categorical: bool = False,
    show: bool = True,
    save_path: str | None = None,
    swap_xy: bool = True,
    reverse_crossline: bool = True,
) -> Axes3D | None:
    """Render three orthogonal slices on a single 3D axes.

    Uses a Poly3DCollection of sorted quad faces so translucency composes
    more predictably than raw plot_surface in Matplotlib 3D.
    """

    # selection/extraction of slice indices delegated to helper to reduce
    # the number of local variables and statements in this function.
    (nx, ny, nz), (ix, iy, iz), (slice_xy, slice_xz, slice_yz) = (
        select_slices_and_indices(volume)
    )

    # coordinates and color preparation delegated to helpers
    spacing = (1.0, 1.0, 1.0)
    # Lazy import for colormap and normalization helper
    from src.plotting.helpers.colors import (
        prepare_colormap_and_norm,
    )

    coords, x, y, z = _prepare_coords_and_grids((nx, ny, nz), (ix, iy, iz), spacing)

    # normalization and colormap
    cmap_func, norm, categorical_info = prepare_colormap_and_norm(
        volume, cmap, is_categorical
    )

    # create figure/axes
    fig, ax, created_fig = _prepare_figure_ax()

    # face colors (per-texel -> per-face) computed by helper
    fc_xy, fc_xz, fc_yz = _compute_facecolors_for_slices(
        (slice_xy, slice_xz, slice_yz), (cmap_func, norm, alpha, categorical_info)
    )

    # collect quads and per-face colors and add the collection to the axis
    _build_and_add_face_collection(ax, coords, fc_xy, fc_xz, fc_yz)

    # add colorbar
    # ax.get_figure may be missing from static type hints for Axes3D;
    # use getattr and cast to satisfy static analyzers and avoid
    # direct attribute access on Axes3D.
    fig = cast(Figure, getattr(ax, "get_figure")())
    make_colorbar(fig, ax, cmap_func, norm, categorical_info)

    # Apply axis labels, autoscaling, view orientation and tick adjustments
    axis_setup(ax, x, y, z, swap_xy, reverse_crossline)

    # Save and optionally show the generated figure
    save_and_show(save_path, show, created_fig)

    return ax
