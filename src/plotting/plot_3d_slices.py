"""3D slice plotting.

Helpers for generating orthogonal 3D slice visualizations using matplotlib.
"""

import logging
from src.utils.facades import LazyObjectProxy

from src.plotting.helpers.plot import init_plotting
from typing import Optional
from src.analysis.types import CacheLoaderProtocol

# Prefer the new CacheLoader helper when available (backwards-compatible)
try:
    from src.analysis.cache import CacheLoader
except Exception:
    CacheLoader = None

__all__ = ["plot_3d_volume", "main"]

logger = logging.getLogger(__name__)

# Initialize matplotlib and numpy for this module
plt, np = init_plotting(backend="Agg")


def plot_3d_volume(ax, cube, slice_indices, title, **plot_kwargs):
    """Wrapper: normalize kwargs and delegate to the PlotVisualization
    facade. This keeps the top-level API while routing implementation via
    the OO facade."""
    # Call canonical implementation directly. Callers that require an
    # instance can still use `get_plot_3d_slices()` to obtain the proxy.
    return _impl_plot_3d_volume(ax, cube, slice_indices, title, **plot_kwargs)


def main(argv=None):
    """Canonical CLI wrapper for 3D slice plotting.

    This wrapper exposes a minimal entrypoint used by the centralized
    `src.__main__` delegators.
    """
    # Use the canonical CLI implementation.
    return _impl_main(argv)


class Plot3DSlices:
    """Facade for 3D slice plotting helpers.

    Accepts an optional `cache_loader` implementing `CacheLoaderProtocol` to
    locate and load cache files. This allows explicit dependency injection
    instead of constructing loaders ad-hoc.
    """

    def __init__(self, cache_loader: Optional[CacheLoaderProtocol] = None):
        self.cache_loader = cache_loader

    def plot_3d_volume(self, ax, cube, slice_indices, title, **plot_kwargs):
        return _impl_plot_3d_volume(ax, cube, slice_indices, title, **plot_kwargs)

    def main(self, argv=None):
        return _impl_main(argv, cache_loader=self.cache_loader)


# Module-level lazy proxy for the plotting facade
plot_3d_slices: Plot3DSlices = LazyObjectProxy(lambda: Plot3DSlices())


def get_plot_3d_slices(
    instance: Plot3DSlices | None = None,
    cache_loader: Optional[CacheLoaderProtocol] = None,
) -> Plot3DSlices:
    # Return provided instance, a new instance constructed with the provided
    # cache_loader, or the module-level lazy proxy.
    if instance is not None:
        return instance
    if cache_loader is not None:
        return Plot3DSlices(cache_loader=cache_loader)
    return plot_3d_slices


def _impl_main(argv=None, cache_loader: Optional[CacheLoaderProtocol] = None):
    import argparse
    from src.plotting.helpers.plot import (
        prepare_plotting_args,
        default_plot_config,
    )

    parser = argparse.ArgumentParser(
        description="Generate 3D orthogonal slice visualizations"
    )
    parser.add_argument(
        "--cache-dir", default=".cache", help="Directory for cache files"
    )
    parser.add_argument(
        "--domain",
        choices=["depth", "time"],
        default="depth",
        help="Domain for processing/visualization",
    )
    parser.add_argument(
        "--no-multiangle", action="store_true", help="Disable multi-angle processing"
    )
    parser.add_argument(
        "--backend", default=None, help="Optional matplotlib backend override"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

    prepare_plotting_args(args)

    plot_cfg = default_plot_config()
    # Honor any CLI backend override and apply the plotting backend.
    if args.backend:
        plot_cfg.backend = args.backend
    plot_cfg.apply_backend()

    cache_dir = args.cache_dir
    import os

    os.makedirs(cache_dir, exist_ok=True)

    # Prefer CacheLoader.select_cache_file if available, otherwise fall back
    # to the legacy select_cache_files helper. This keeps behavior stable
    # while enabling easier testing/mocking via CacheLoader.
    avo_fn = None
    if cache_loader is not None:
        try:
            avo_fn = cache_loader.select_cache_file(cache_dir, args.domain)
        except Exception:
            avo_fn = None
    else:
        # fallback to constructing a CacheLoader locally
        try:
            from src.analysis.cache import CacheLoader

            loader = CacheLoader()
            avo_fn = loader.select_cache_file(cache_dir, args.domain)
        except Exception:
            avo_fn = None

    return {"avo": avo_fn}


def _impl_plot_3d_volume(ax, cube, slice_indices, title, **plot_kwargs):
    from src.plotting.helpers.plot import apply_plot_defaults
    from src.plotting.helpers.visualization import plot_visualization

    plot_kwargs = apply_plot_defaults(plot_kwargs)
    cmap = plot_kwargs.get("cmap", "seismic")

    return plot_visualization.plot_3d_slices(ax, cube, slice_indices, title, cmap=cmap)
