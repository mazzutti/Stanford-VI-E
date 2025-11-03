"""Plotting helpers.

Collection of plotting helper utilities used by the plotting subpackage.
"""

from typing import Optional
from dataclasses import dataclass
from typing import Dict, Tuple

from src.io.grid import GridSpec
import numpy as np
import logging
from src.utils.facades import LazyObjectProxy


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
]

# Module-level flags to control application of global plotting defaults.
# Tests can set `PLOT_DEFAULTS_DISABLED = True` to prevent global rcParams
# from being applied; `PLOT_DEFAULTS_APPLIED` records whether defaults were
# applied during runtime.
PLOT_DEFAULTS_APPLIED: bool = False
PLOT_DEFAULTS_DISABLED: bool = False


# Object-oriented facade for plotting helpers
class PlotHelper:
    def __init__(self, config: Optional["PlotConfig"] = None):
        """Create a PlotHelper.

        Args:
            config: optional PlotConfig to use as defaults for this helper.
        """
        self.config = config or PlotConfig.default()

    # --- Instance (class-first) methods ---
    def get_3d_surface_kwargs(self, stride: int = 1, **kwargs):
        return _impl_get_3d_surface_kwargs(stride=stride, **kwargs)

    def apply_plot_defaults(self, plot_kwargs: dict) -> dict:
        return _impl_apply_plot_defaults(plot_kwargs)

    def setup_matplotlib(self, backend: Optional[str] = "Agg") -> None:
        return _impl_setup_matplotlib(backend=backend)

    def init_plt(self, backend: Optional[str] = "Agg"):
        return _impl_init_plt(backend=backend)

    def init_plotting(self, backend: Optional[str] = "Agg"):
        return _impl_init_plotting(backend=backend)

    def default_plot_config(self) -> "PlotConfig":
        return _impl_default_plot_config()

    def prepare_plotting_args(self, args):
        return _impl_prepare_plotting_args(args)

    # `select_cache_files` functionality was moved to
    # `src.analysis.cache_loader.select_cache_files` to centralize cache logic.

    def create_figure_grid(self, *args, **kwargs):
        return _impl_create_figure_grid(*args, **kwargs)

    def imshow_with_labels(self, *args, **kwargs):
        return _impl_imshow_with_labels(*args, **kwargs)

    def apply_common_axis_style(self, ax, fontsize_title=12, fontsize_labels=10):
        return _impl_apply_common_axis_style(
            ax, fontsize_title=fontsize_title, fontsize_labels=fontsize_labels
        )

    def compute_boundary_alignment(self, seismic, facies, sigma: float = 1.5):
        return _impl_compute_boundary_alignment(seismic, facies, sigma=sigma)

    def create_figure(self, figsize=None):
        return _impl_create_figure(figsize=figsize)

    # The PlotHelper intentionally exposes only stateful configuration.
    # Callers should use the module-level helper functions (for example,
    # `apply_plot_defaults`, `select_cache_files`, `compute_boundary_alignment`)
    # when they require the canonical implementations. This class remains a
    # lightweight container for a PlotConfig instance.


# Module-level singleton proxy for plotting helpers (lazy to avoid heavy imports at import time)
plot_helper = LazyObjectProxy(lambda: PlotHelper(config=PlotConfig.default()))


__all__.extend(["PlotHelper", "plot_helper"])


def get_plot_helper(config: dict | None = None):
    """Return the module-level `plot_helper` proxy when `config` is None,
    otherwise return a new PlotHelper instance configured from `config`.
    """
    # Return module-level proxy when no config supplied; otherwise construct
    # a fresh PlotHelper configured from the provided config.
    if config is None:
        return plot_helper
    pc = PlotConfig.default()
    return PlotHelper(config=pc)


__all__.append("get_plot_helper")


def get_3d_surface_kwargs(stride: int = 1, **kwargs):
    """Return a small kwargs dict suitable for Axes3D.plot_surface calls.

    Keeps defaults compact and centralizes any changes to surface plotting.
    """
    out = {}
    out["rstride"] = int(stride)
    out["cstride"] = int(stride)
    if "linewidth" in kwargs:
        out["linewidth"] = kwargs.get("linewidth")
    if "edgecolor" in kwargs:
        out["edgecolor"] = kwargs.get("edgecolor")
    # default antialiasing / shading options
    out.setdefault("antialiased", True)
    out.setdefault("shade", False)
    return out


def get_3d_surface_kwargs(stride: int = 1, **kwargs):
    # delegate to instance for class-first API (back-compat preserved)
    try:
        return plot_helper.get_3d_surface_kwargs(stride=stride, **kwargs)
    except Exception:
        return _impl_get_3d_surface_kwargs(stride=stride, **kwargs)


def _impl_get_3d_surface_kwargs(stride: int = 1, **kwargs):
    return get_3d_surface_kwargs(stride=stride, **kwargs)


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


def apply_plot_defaults(plot_kwargs: dict) -> dict:
    try:
        return plot_helper.apply_plot_defaults(plot_kwargs)
    except Exception:
        return _impl_apply_plot_defaults(plot_kwargs)


def _impl_apply_plot_defaults(plot_kwargs: dict) -> dict:
    return apply_plot_defaults(plot_kwargs)


def setup_matplotlib(backend: Optional[str] = "Agg") -> None:
    import matplotlib
    import logging

    if backend:
        matplotlib.use(backend)
    # centralize rcParams configuration in one helper so multiple codepaths
    # don't duplicate visual defaults
    # Apply centralized rcParams configuration
    configure_matplotlib_defaults()


def setup_matplotlib(backend: Optional[str] = "Agg") -> None:
    try:
        return plot_helper.setup_matplotlib(backend=backend)
    except Exception:
        return _impl_setup_matplotlib(backend=backend)


def _impl_setup_matplotlib(backend: Optional[str] = "Agg") -> None:
    return setup_matplotlib(backend=backend)


def init_plt(backend: Optional[str] = "Agg"):
    try:
        return plot_helper.init_plt(backend=backend)
    except Exception:
        return _impl_init_plt(backend=backend)


def _impl_init_plt(backend: Optional[str] = "Agg"):
    return init_plt(backend=backend)


def init_plotting(backend: Optional[str] = "Agg"):
    try:
        return plot_helper.init_plotting(backend=backend)
    except Exception:
        return _impl_init_plotting(backend=backend)


def _impl_init_plotting(backend: Optional[str] = "Agg"):
    # Concrete implementation: set backend, configure matplotlib and return (plt, np)
    import matplotlib
    import logging

    if backend:
        try:
            matplotlib.use(backend)
        except Exception:
            # If backend switch fails, continue and let matplotlib decide
            pass

    # Apply centralized rcParams configuration and return plt
    configure_matplotlib_defaults()
    import matplotlib.pyplot as plt

    return plt, np


def configure_matplotlib_defaults() -> None:
    """Apply a small set of matplotlib defaults used across the project.

    This helper allows non-plotting modules to opt-in to a consistent
    visual style without importing heavy plotting modules at import time.
    """
    global PLOT_DEFAULTS_APPLIED

    # Allow tests to disable global defaults.
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
    # image-related defaults used across plotting utilities
    plt.rcParams["image.interpolation"] = "bilinear"
    plt.rcParams["image.resample"] = True
    plt.rcParams["image.composite_image"] = True

    PLOT_DEFAULTS_APPLIED = True


def default_plot_config() -> "PlotConfig":
    try:
        return plot_helper.default_plot_config()
    except Exception:
        return _impl_default_plot_config()


def _impl_default_plot_config() -> "PlotConfig":
    return default_plot_config()


@dataclass
class PlotConfig:
    data_path: str
    file_map: Dict[str, str]
    grid_spec: GridSpec
    backend: Optional[str] = "Agg"

    # Convenience accessor properties
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
        """Return a GridSpec constructed from this PlotConfig."""
        # The PlotConfig stores a GridSpec directly; return it.
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
        try:
            setup_matplotlib(self.backend)
        except Exception:
            pass


def prepare_plotting_args(args):
    try:
        return plot_helper.prepare_plotting_args(args)
    except Exception:
        return _impl_prepare_plotting_args(args)


def _impl_prepare_plotting_args(args):
    return prepare_plotting_args(args)


# Cache selection is handled by `CacheLoader` in src.analysis.cache_loader


def _impl_create_figure_grid(
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


def create_figure_grid(
    figsize=(12, 8), nrows=1, ncols=1, gridspec_kw=None, constrained_layout=True
):
    try:
        return plot_helper.create_figure_grid(
            figsize=figsize,
            nrows=nrows,
            ncols=ncols,
            gridspec_kw=gridspec_kw,
            constrained_layout=constrained_layout,
        )
    except Exception:
        return _impl_create_figure_grid(
            figsize=figsize,
            nrows=nrows,
            ncols=ncols,
            gridspec_kw=gridspec_kw,
            constrained_layout=constrained_layout,
        )


def _impl_imshow_with_labels(
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
    """Convenience wrapper to show a 2D image with axis labels and optional colorbar.

    This helper centralizes consistent plotting style used across the plotting
    modules in the repository.
    """
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
        try:
            ax.set_title(title, fontsize=fontsize_title)
        except Exception:
            pass

    if xlabel is not None:
        try:
            ax.set_xlabel(xlabel, fontsize=fontsize_labels)
        except Exception:
            pass

    if k_label is not None:
        try:
            label = k_label
            if k_unit:
                label = f"{k_label} ({k_unit})"
            ax.set_ylabel(label, fontsize=fontsize_labels)
        except Exception:
            pass

    if colorbar:
        try:
            cb = plt.colorbar(im, ax=ax)
            if colorbar_label:
                try:
                    cb.set_label(colorbar_label)
                except Exception:
                    pass
        except Exception:
            pass

    return im


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
    try:
        return plot_helper.imshow_with_labels(
            ax,
            data,
            title=title,
            xlabel=xlabel,
            k_label=k_label,
            k_unit=k_unit,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin=origin,
            interpolation=interpolation,
            is_categorical=is_categorical,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            fontsize_title=fontsize_title,
            fontsize_labels=fontsize_labels,
        )
    except Exception:
        return _impl_imshow_with_labels(
            ax,
            data,
            title=title,
            xlabel=xlabel,
            k_label=k_label,
            k_unit=k_unit,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin=origin,
            interpolation=interpolation,
            is_categorical=is_categorical,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            fontsize_title=fontsize_title,
            fontsize_labels=fontsize_labels,
        )


def _impl_apply_common_axis_style(ax, fontsize_title=12, fontsize_labels=10):
    return apply_common_axis_style(
        ax, fontsize_title=fontsize_title, fontsize_labels=fontsize_labels
    )


def apply_common_axis_style(ax, fontsize_title=12, fontsize_labels=10):
    try:
        return plot_helper.apply_common_axis_style(
            ax, fontsize_title=fontsize_title, fontsize_labels=fontsize_labels
        )
    except Exception:
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
    from scipy.ndimage import gaussian_filter

    try:
        # plotting helper now lives under src.plotting
        from src.plotting.plot_facies_overlay import detect_facies_boundaries
    except Exception:

        def detect_facies_boundaries(facies_slice, sigma=0.5):
            from scipy.ndimage import gaussian_filter

            smoothed = gaussian_filter(facies_slice.astype(float), sigma=0.5)
            from scipy.ndimage import sobel

            grad_x = sobel(smoothed, axis=0)
            grad_y = sobel(smoothed, axis=1)
            gradient_magnitude = (grad_x**2 + grad_y**2) ** 0.5
            return (gradient_magnitude > 0.1).astype(float)

    try:
        return plot_helper.compute_boundary_alignment(seismic, facies, sigma=sigma)
    except Exception:
        facies_boundaries = detect_facies_boundaries(facies)

        seismic_smooth = gaussian_filter(seismic, sigma=sigma)
        import numpy as np

        grad_j, grad_k = np.gradient(seismic_smooth)
        seismic_grad = np.sqrt(grad_j**2 + grad_k**2)
        seismic_grad = seismic_grad / (seismic_grad.max() + 1e-10)

        from src.processing.align import align_cubes

        seismic_grad, facies_boundaries = align_cubes(seismic_grad, facies_boundaries)

        alignment = seismic_grad * facies_boundaries
        return alignment


def _impl_compute_boundary_alignment(seismic, facies, sigma: float = 1.5):
    return compute_boundary_alignment(seismic, facies, sigma=sigma)


def _impl_create_figure(figsize=None):
    import matplotlib.pyplot as plt

    if figsize is not None:
        return plt.figure(figsize=figsize)
    return plt.figure()


def create_figure(figsize=None):
    try:
        return plot_helper.create_figure(figsize=figsize)
    except Exception:
        return _impl_create_figure(figsize=figsize)


__all__.append("get_3d_surface_kwargs")

# Module logger
logger = logging.getLogger(__name__)
