"""Plotting helpers moved from src.utils.plot into src.plotting.helpers

This file is a nearly verbatim copy of `src/utils/plot.py` but re-homed under
`src.plotting.helpers` to reduce the size of `src.utils` and provide a more
logical package separation.
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
    "select_cache_files",
    "create_figure_grid",
    "imshow_with_labels",
    "apply_common_axis_style",
    "compute_boundary_alignment",
    "create_figure",
]


# Object-oriented facade for plotting helpers
class PlotHelper:
    def __init__(self, config: Optional["PlotConfig"] = None):
        """Create a PlotHelper.

        Args:
            config: optional PlotConfig to use as defaults for this helper.
        """
        self.config = config or PlotConfig.default()

    def setup_matplotlib(self, backend: Optional[str] = "Agg") -> None:
        return setup_matplotlib(backend=backend)

    def init_plt(self, backend: Optional[str] = "Agg"):
        return init_plt(backend=backend)

    def init_plotting(self, backend: Optional[str] = "Agg"):
        return init_plotting(backend=backend)

    def default_plot_config(self):
        # Return the instance config for stateful helpers; keep the top-level
        # factory function for backwards compatibility.
        return self.config

    def prepare_plotting_args(self, args):
        return prepare_plotting_args(args)

    def select_cache_files(self, cache_dir: str, domain: str):
        return select_cache_files(cache_dir, domain)

    def create_figure_grid(self, *args, **kwargs):
        # Delegate to module-level implementation but keep logic local so
        # callers using the instance get the same behaviour.
        return _impl_create_figure_grid(*args, **kwargs)

    def imshow_with_labels(self, *args, **kwargs):
        return _impl_imshow_with_labels(*args, **kwargs)

    def apply_common_axis_style(self, *args, **kwargs):
        return apply_common_axis_style(*args, **kwargs)

    def compute_boundary_alignment(self, *args, **kwargs):
        return compute_boundary_alignment(*args, **kwargs)

    def create_figure(self, *args, **kwargs):
        # simple wrapper matching the old create_figure convenience
        return _impl_create_figure(*args, **kwargs)


from src.utils.facades import LazyObjectProxy


# Module-level singleton for gradual migration (lazy proxy to avoid heavy imports at import time)
plot_helper = LazyObjectProxy(lambda: PlotHelper(config=PlotConfig.default()))


__all__.extend(["PlotHelper", "plot_helper"])


def get_plot_helper(config: dict | None = None):
    """Return the module-level `plot_helper` proxy when `config` is None,
    otherwise return a new PlotHelper instance configured from `config`.
    """
    if config is None:
        return plot_helper
    # minimal config mapping supported for now
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


def apply_plot_defaults(plot_kwargs: dict) -> dict:
    """Ensure a minimal set of plotting keys exist and return a sanitized dict.

    This keeps callers concise and backwards-compatible with older helpers.
    """
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
    import logging

    if backend:
        matplotlib.use(backend)

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    import matplotlib.pyplot as plt

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


def init_plt(backend: Optional[str] = "Agg"):
    setup_matplotlib(backend=backend)
    import matplotlib.pyplot as plt

    return plt


def init_plotting(backend: Optional[str] = "Agg"):
    plt = init_plt(backend=backend)
    import numpy as np

    return plt, np


def default_plot_config():
    return PlotConfig.default()


@dataclass
class PlotConfig:
    data_path: str
    file_map: Dict[str, str]
    grid_spec: GridSpec
    backend: Optional[str] = "Agg"

    # Backwards-compatible accessors
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
    args.use_multiangle = not getattr(args, "no_multiangle", False)
    if getattr(args, "domain", "depth") == "time" and args.use_multiangle:
        args.use_multiangle = False


def _find_latest(cache_dir: str, prefix: str):
    from src.io.cache import cache_for_dir

    groups = cache_for_dir(cache_dir).select_latest_cache_entries()
    candidates = []
    if prefix in groups:
        candidates = groups[prefix]
    else:
        for k, v in groups.items():
            if k.startswith(prefix.rstrip("_")):
                candidates.extend(v)

    if not candidates:
        return None

    # candidates are CacheEntry objects; prefer ones that include domain suffixes
    # pick the latest by mtime
    candidates_sorted = sorted(candidates, key=lambda e: e.mtime)
    chosen = candidates_sorted[-1]
    from pathlib import Path

    return str(Path(cache_dir) / chosen.path.name)


def select_cache_files(cache_dir: str, domain: str):
    # Note: we intentionally call select_latest_cache_entries inside helpers
    # such as _find_latest; no persistent 'groups' variable needed here.

    if domain == "time":
        avo_fn = _find_latest(cache_dir, "avo_time")
        ai_fn = _find_latest(cache_dir, "ai_time")
    else:
        avo_fn = _find_latest(cache_dir, "avo_depth")
        ai_fn = _find_latest(cache_dir, "ai_depth")

    ei_fn = None
    ei_data_key = None
    ei_type_str = ""
    ei_is_depth_domain = False

    if domain == "depth":
        ei_fn = _find_latest(cache_dir, "ei_depth") or _find_latest(
            cache_dir, "ei_time"
        )
        if ei_fn and "ei_depth" in ei_fn:
            ei_data_key = "ei_optimal"
            ei_type_str = "multi-angle depth-domain impedance (optimal stack)"
            ei_is_depth_domain = True
        else:
            ei_data_key = "ei_seismic"
            ei_type_str = "multi-angle time-domain seismogram"
            ei_is_depth_domain = False
    else:
        ei_fn = _find_latest(cache_dir, "ei_time") or _find_latest(
            cache_dir, "ei_depth"
        )
        if ei_fn and "ei_time" in ei_fn:
            ei_data_key = "ei_seismic"
            ei_type_str = "multi-angle time-domain seismogram"
            ei_is_depth_domain = False
        else:
            ei_data_key = "ei_optimal"
            ei_type_str = "multi-angle depth-domain impedance (optimal stack)"
            ei_is_depth_domain = True

    return avo_fn, ai_fn, ei_fn, ei_data_key, ei_type_str, ei_is_depth_domain


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
    return plot_helper.create_figure_grid(
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


def _impl_create_figure(figsize=None):
    import matplotlib.pyplot as plt

    if figsize is not None:
        return plt.figure(figsize=figsize)
    return plt.figure()


def create_figure(figsize=None):
    return plot_helper.create_figure(figsize=figsize)


__all__.append("get_3d_surface_kwargs")

# Module logger
logger = logging.getLogger(__name__)
