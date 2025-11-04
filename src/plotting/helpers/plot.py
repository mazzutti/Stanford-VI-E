"""Plotting helpers.

Collection of plotting helper utilities used by the plotting subpackage.
"""

from typing import Optional
from dataclasses import dataclass
from typing import Dict, Tuple

from src.io.grid import GridSpec
import numpy as np
import logging


__all__ = [
    "setup_matplotlib",
    "init_plt",
    "init_plotting",
    "default_plot_config",
    "PlotConfig",
    "prepare_plotting_args",
    "apply_common_axis_style",
    "compute_boundary_alignment",
    "create_figure",
    "apply_plot_defaults",
    "create_figure_grid",
    "imshow_with_labels",
    "get_3d_surface_kwargs",
    "get_plot_helper",
]

# Module-level flags to control application of global plotting defaults.
PLOT_DEFAULTS_APPLIED: bool = False
PLOT_DEFAULTS_DISABLED: bool = False


@dataclass
class PlotConfig:
    data_path: str
    file_map: Dict[str, str]
    grid_spec: GridSpec
    backend: Optional[str] = "Agg"

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        return self.grid_spec.shape

    @property
    def dz(self) -> float:
        return self.grid_spec.dz

    @property
    def dt(self) -> float:
        return self.grid_spec.dt

    def as_grid_spec(self) -> GridSpec:
        return self.grid_spec

    @classmethod
    def default(cls) -> "PlotConfig":
        return cls(
            data_path=".",
            file_map={"vp": "P-wave Velocity", "facies": "Facies"},
            grid_spec=GridSpec((150, 200, 200), dz=1.0, dt=0.001),
            backend="Agg",
        )

    @classmethod
    def from_args(cls, args) -> "PlotConfig":
        pc = cls.default()
        pc.backend = getattr(args, "backend", pc.backend)
        return pc

    def apply_backend(self):
        setup_matplotlib(self.backend)


def configure_matplotlib_defaults() -> None:
    """Apply a small set of matplotlib defaults used across the project."""
    global PLOT_DEFAULTS_APPLIED

    if PLOT_DEFAULTS_DISABLED:
        return

    import logging
    import matplotlib.pyplot as plt

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "DejaVu Sans",
        "Arial",
        "Helvetica",
        "sans-serif",
    ]
    plt.rcParams["image.interpolation"] = "bilinear"
    plt.rcParams["image.resample"] = True
    plt.rcParams["image.composite_image"] = True

    PLOT_DEFAULTS_APPLIED = True


def get_3d_surface_kwargs(stride: int = 1, **kwargs):
    """Return a small kwargs dict suitable for Axes3D.plot_surface calls."""
    out = {}
    out["rstride"] = int(stride)
    out["cstride"] = int(stride)
    if "linewidth" in kwargs:
        out["linewidth"] = kwargs.get("linewidth")
    if "edgecolor" in kwargs:
        out["edgecolor"] = kwargs.get("edgecolor")
    out.setdefault("antialiased", True)
    out.setdefault("shade", False)
    return out


def apply_plot_defaults(plot_kwargs: dict) -> dict:
    """Ensure a minimal set of plotting keys exist and return a sanitized dict."""
    defaults = {
        "k_scale": 1.0,
        "k_label": "K",
        "k_unit": "",
        "cmap": "RdBu",
        "is_categorical": False,
        "show_colorbar": True,
    }
    out = dict(defaults)
    if plot_kwargs:
        out.update(plot_kwargs)
    return out


def setup_matplotlib(backend: Optional[str] = "Agg") -> None:
    import matplotlib

    if backend:
        try:
            matplotlib.use(backend)
        except Exception:
            pass
    configure_matplotlib_defaults()


def init_plt(backend: Optional[str] = "Agg"):
    setup_matplotlib(backend)
    import matplotlib.pyplot as plt

    return plt


def init_plotting(backend: Optional[str] = "Agg"):
    """Set backend, configure matplotlib and return (plt, np)."""
    import matplotlib

    if backend:
        try:
            matplotlib.use(backend)
        except Exception:
            pass
    configure_matplotlib_defaults()
    import matplotlib.pyplot as plt

    return plt, np


def default_plot_config() -> "PlotConfig":
    return PlotConfig.default()


def prepare_plotting_args(args):
    return PlotConfig.from_args(args)


def create_figure_grid(
    figsize=(12, 8), nrows=1, ncols=1, gridspec_kw=None, constrained_layout=True
):
    import matplotlib.pyplot as plt

    if gridspec_kw is None:
        gridspec_kw = {}
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        gridspec_kw=gridspec_kw,
        constrained_layout=constrained_layout,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])
    return fig, axes


def imshow_with_labels(
    ax,
    data,
    title: str = "",
    xlabel: str | None = None,
    k_label: str | None = None,
    k_unit: str | None = None,
    cmap=None,
    vmin=None,
    vmax=None,
    origin: str = "upper",
    interpolation: str = "bilinear",
    is_categorical: bool = False,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    fontsize_title: int = 12,
    fontsize_labels: int = 10,
):
    """Convenience wrapper to show a 2D image with axis labels and optional colorbar."""
    import matplotlib.pyplot as plt

    ax.clear()
    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        interpolation=interpolation,
    )
    if title:
        ax.set_title(title, fontsize=fontsize_title)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize_labels)
    if k_label is not None:
        label = k_label
        if k_unit:
            label = f"{k_label} ({k_unit})"
        ax.set_ylabel(label, fontsize=fontsize_labels)
    if colorbar:
        cb = plt.colorbar(im, ax=ax)
        if colorbar_label:
            cb.set_label(colorbar_label)
    return im


def apply_common_axis_style(ax, fontsize_title=12, fontsize_labels=10):
    try:
        ax.title.set_fontsize(fontsize_title)
    except Exception:
        pass
    try:
        ax.xaxis.label.set_size(fontsize_labels)
        ax.yaxis.label.set_size(fontsize_labels)
    except Exception:
        pass


def compute_boundary_alignment(seismic, facies, sigma: float = 1.5):
    from scipy.ndimage import gaussian_filter, sobel

    try:
        from src.plotting.plot_facies_overlay import detect_facies_boundaries
    except Exception:

        def detect_facies_boundaries(facies_slice, sigma=0.5):
            smoothed = gaussian_filter(facies_slice.astype(float), sigma=0.5)
            grad_x = sobel(smoothed, axis=0)
            grad_y = sobel(smoothed, axis=1)
            gradient_magnitude = (grad_x**2 + grad_y**2) ** 0.5
            return (gradient_magnitude > 0.1).astype(float)

    facies_boundaries = detect_facies_boundaries(facies)
    seismic_smooth = gaussian_filter(seismic, sigma=sigma)
    grad_j, grad_k = np.gradient(seismic_smooth)
    seismic_grad = np.sqrt(grad_j**2 + grad_k**2)
    seismic_grad = seismic_grad / (seismic_grad.max() + 1e-10)
    try:
        from src.processing.align import align_cubes

        seismic_grad, facies_boundaries = align_cubes(seismic_grad, facies_boundaries)
    except Exception:
        pass
    alignment = seismic_grad * facies_boundaries
    return alignment


def create_figure(figsize=None):
    import matplotlib.pyplot as plt

    if figsize is not None:
        return plt.figure(figsize=figsize)
    return plt.figure()


def get_plot_helper(config: dict | None = None):
    """Return a PlotConfig instance."""
    if config is None:
        return PlotConfig.default()
    return PlotConfig.default()


# Module logger
logger = logging.getLogger(__name__)
