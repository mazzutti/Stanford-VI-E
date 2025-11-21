"""Colorscale and bounds helpers for Plotly trace construction.

Centralizes matplotlib->plotly conversion and percentile-based bound
calculation so other modules can reuse the same logic.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from numpy.typing import NDArray


def _resolve_cmap(name: str, n_colors: int | None = None) -> Any:
    """Resolve a Matplotlib colormap by name in a case-insensitive way.

    Accepts an optional ``n_colors`` parameter to request a Listed/Indexed
    colormap suitable for categorical use. Raises ``LookupError`` when no
    matching colormap is found or the registry is unavailable.
    """
    registry = getattr(mpl, "colormaps", None)
    if registry is None:
        raise LookupError("matplotlib colormap registry not available")

    try:
        cmap = registry.get_cmap(name)
        if n_colors is None:
            return cmap
        # Prefer Colormap.resampled when available (Matplotlib >= 3.7)
        resampler = getattr(cmap, "resampled", None)
        if callable(resampler):
            return resampler(n_colors)
        # Fallback: build a ListedColormap from sampled colors
        colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
        return ListedColormap(colors)
    except (LookupError, KeyError, ValueError) as exc:
        # try lowercase and a registry scan for case-insensitive match
        try:
            cmap = registry.get_cmap(name.lower())
            if n_colors is None:
                return cmap
            resampler = getattr(cmap, "resampled", None)
            if callable(resampler):
                return resampler(n_colors)
            colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
            return ListedColormap(colors)
        except (LookupError, KeyError, ValueError):
            lname = name.lower()
            for k in registry:
                if k.lower() == lname:
                    if n_colors is None:
                        return registry.get_cmap(k)
                    cmap = registry.get_cmap(k)
                    resampler = getattr(cmap, "resampled", None)
                    if callable(resampler):
                        return resampler(n_colors)
                    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
                    return ListedColormap(colors)
            raise LookupError(f"colormap not found: {name}") from exc


def _mpl_to_plotly(name: str, samples: int = 256) -> list[list[float | str]]:
    """Convert a Matplotlib colormap name to a Plotly colorscale list."""
    cmap = _resolve_cmap(name)
    scalars = np.linspace(0.0, 1.0, samples)
    colors = [mcolors.to_hex(cmap(s)) for s in scalars]
    step = 1.0 / (len(colors) - 1)
    return [[i * step, colors[i]] for i in range(len(colors))]


def _compute_symmetric_bounds(
    arr: NDArray[np.floating[Any]], indices: tuple[int, int, int]
) -> tuple[float, float]:
    """Compute symmetric cmin/cmax from the 99.5th percentile across three slices."""
    inline_idx, crossline_idx, depth_idx = indices
    combined = np.concatenate(
        [
            np.ravel(np.abs(arr[inline_idx, :, :])),
            np.ravel(np.abs(arr[:, crossline_idx, :])),
            np.ravel(np.abs(arr[:, :, depth_idx])),
        ]
    )
    p = float(np.percentile(combined, 99.5))
    vmax = float(p) if p != 0.0 else 1.0
    return -vmax, vmax


def compute_plotly_colorscale_and_bounds(
    arr: NDArray[np.floating[Any]],
    indices: tuple[int, int, int],
    colorscale: str | list[list[float | str]] = "RdBu",
    is_categorical: bool = False,
) -> tuple[str | list[list[float | str]], float, float]:
    """Compute a Plotly-compatible colorscale and symmetric cmin/cmax bounds.

    Returns (colorscale_to_use, cmin, cmax).
    """
    if is_categorical:
        colorscale_to_use: str | list[list[float | str]] = [
            [0.0, "rgb(31,119,180)"],
            [0.33, "rgb(255,127,14)"],
            [0.67, "rgb(44,160,44)"],
            [1.0, "rgb(214,39,40)"],
        ]
        return colorscale_to_use, 0.0, 3.0

    colorscale_to_use = colorscale
    if isinstance(colorscale, str):
        colorscale_to_use = _mpl_to_plotly(colorscale)

    cmin, cmax = _compute_symmetric_bounds(arr, indices)
    return colorscale_to_use, cmin, cmax


def prepare_colormap_and_norm(
    volume: NDArray[np.floating[Any]], cmap: str, is_categorical: bool
) -> tuple[Any, Normalize | None, tuple[NDArray[Any], dict[float, int], int] | None]:
    """Return (cmap_func, norm, categorical_info).

    `categorical_info` is either None or (cats, mapping, ncat).
    """
    if is_categorical:
        cats = np.unique(volume[~np.isnan(volume)])
        if cats.size == 0:
            # Fallback to continuous normalization when no categories found
            norm = Normalize(
                vmin=float(np.nanmin(volume)), vmax=float(np.nanmax(volume))
            )
            cmap_func = _resolve_cmap(cmap)
            categorical_info = None
        else:
            ncat = int(cats.size)
            cmap_func = _resolve_cmap(cmap, ncat)
            mapping: dict[float, int] = {float(v): i for i, v in enumerate(cats)}
            categorical_info = (cats, mapping, ncat)
            norm = None
    else:
        norm = Normalize(vmin=float(np.nanmin(volume)), vmax=float(np.nanmax(volume)))
        cmap_func = _resolve_cmap(cmap)
        categorical_info = None

    return cmap_func, norm, categorical_info


def slice_to_rgba(
    slice_arr: NDArray[np.floating[Any]],
    cmap_func: Any,
    norm: Normalize | None,
    alpha: float | None,
    categorical_info: tuple[NDArray[Any], dict[float, int], int] | None,
) -> NDArray[np.floating[Any]]:
    """Convert a 2D slice to an RGBA array usable for face-color computation.

    Returns an array with shape (M, N, 4).
    """
    if categorical_info is None:
        assert norm is not None
        colors = cmap_func(norm(slice_arr))
        colors = cast(NDArray[np.floating[Any]], colors)
        if alpha is not None:
            colors[..., 3] = float(alpha)
        return colors

    _, mapping, ncat = categorical_info
    idx = np.zeros(slice_arr.shape, dtype=int)
    # Map values robustly using isclose matching to mapping keys
    for v, i in mapping.items():
        idx[np.isclose(slice_arr, v)] = i
    palette = cmap_func(np.linspace(0.0, 1.0, ncat))
    palette = cast(NDArray[Any], palette)
    colors = palette[idx]
    if alpha is not None:
        colors[..., 3] = float(alpha)
    # Palette indexing can produce an array typed as Any; cast to the
    # declared return type so static checkers understand the real shape.
    return cast(NDArray[np.floating[Any]], colors)


def prepare_colorbar_mappable(
    cmap_func: Any,
    norm: Normalize | None,
    categorical_info: tuple[NDArray[Any], dict[float, int], int] | None,
) -> tuple[Any, NDArray[Any] | None, NDArray[Any] | None, list[str] | None]:
    """Prepare a ScalarMappable and optional categorical metadata for colorbars.

    Returns a tuple ``(sm, bounds, ticks, ticklabels)`` where continuous cases
    return ``sm`` and the other values as ``None``. For categorical data the
    function produces a ``ListedColormap``-backed ScalarMappable, bounds array,
    tick positions and ticklabels.
    """
    if categorical_info is None:
        assert norm is not None
        sm = cm.ScalarMappable(cmap=cmap_func, norm=norm)
        return sm, None, None, None

    cats, _, ncat = categorical_info
    palette_arr = cast(
        NDArray[np.floating[Any]], cmap_func(np.linspace(0.0, 1.0, ncat))
    )
    palette = list(palette_arr)
    lc = ListedColormap(palette)
    bounds = np.arange(-0.5, ncat + 0.5, 1.0)
    bnorm = BoundaryNorm(bounds, ncolors=ncat)
    sm = cm.ScalarMappable(cmap=lc, norm=cast(Normalize, bnorm))
    ticks = np.arange(ncat)
    ticklabels = [str(v) for v in cats]
    return sm, bounds, ticks, ticklabels
