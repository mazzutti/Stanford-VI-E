"""2D slice plotting.

Implementation of 2D slice plotting helpers using plotting utilities from
`src.plotting.helpers`.
"""

import logging

from src.plotting.helpers.plot import init_plotting
from src.io.grid import GridSpec
from src.analysis.facies import FaciesCorrelationAnalyzer

# Initialize matplotlib and numpy for this module
plt, np = init_plotting(backend="Agg")

# Module logger
logger = logging.getLogger(__name__)

__all__ = ["convert_time_to_depth", "plot_with_units"]


def convert_time_to_depth(
    seismogram_time, vp_depth, grid_spec: GridSpec, is_categorical=False
):
    """Convert a time-domain seismogram to depth using the provided GridSpec.

    Uses the canonical class-based analyzer API directly.
    """
    analyzer = FaciesCorrelationAnalyzer()
    return analyzer.convert_time_to_depth(seismogram_time, vp_depth, grid_spec)


def plot_with_units(ax, cube, slice_idx, slice_orientation, title, **plot_kwargs):
    from src.plotting.helpers.plot import apply_plot_defaults

    plot_kwargs = apply_plot_defaults(plot_kwargs)
    k_scale = plot_kwargs["k_scale"]
    k_label = plot_kwargs["k_label"]
    k_unit = plot_kwargs["k_unit"]
    cmap = plot_kwargs["cmap"]
    is_categorical = plot_kwargs["is_categorical"]

    ni, nj, nk = cube.shape

    if slice_orientation == "inline":
        slice_data = cube[slice_idx, :, :]  # [J, K]
        xlabel = "Crossline (J)"
        title_with_slice = f"{title}\n(Inline I={slice_idx})"
    elif slice_orientation == "crossline":
        slice_data = cube[:, slice_idx, :]  # [I, K]
        xlabel = "Inline (I)"
        title_with_slice = f"{title}\n(Crossline J={slice_idx})"
    elif slice_orientation in ["timeslice", "depthslice"]:
        slice_data = cube[:, :, slice_idx]  # [I, J]
        xlabel = "Crossline (J)"
        slice_label = (
            f"{k_label}={slice_idx * k_scale:.3f}{k_unit}"
            if k_unit
            else f"{k_label}={slice_idx}"
        )
        title_with_slice = f"{title}\n({slice_label})"
        k_unit = ""  # Don't add units to ylabel for horizontal slices
    else:
        raise ValueError(f"Unknown slice orientation: {slice_orientation}")

    if is_categorical:
        vmin = 0
        vmax = 3  # Fixed to 4 facies (0, 1, 2, 3)
        from matplotlib.colors import ListedColormap

        colors = plt.cm.tab10(np.linspace(0, 0.4, 4))  # Get first 4 colors from tab10
        cmap_discrete = ListedColormap(colors)
    else:
        p_i = np.percentile(np.abs(slice_data), 99.5)
        vmax = float(p_i)
        vmin = -vmax
        if vmax == vmin:
            vmax = vmin + 1.0

    ax.clear()

    from src.plotting.helpers.plot import imshow_with_labels

    return imshow_with_labels(
        ax,
        slice_data,
        title_with_slice,
        xlabel=xlabel,
        k_label=k_label,
        k_unit=k_unit,
        cmap=cmap_discrete if is_categorical else cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest" if is_categorical else "bilinear",
        is_categorical=is_categorical,
        colorbar=True if not is_categorical else True,
        colorbar_label="Facies" if is_categorical else "Amplitude",
        fontsize_title=10,
        fontsize_labels=10,
    )
