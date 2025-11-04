"""3D slice plotting.

Helpers for generating orthogonal 3D slice visualizations using matplotlib.
"""

import logging
from src.utils.facades import LazyObjectProxy

from src.plotting.helpers.plot import init_plotting
from typing import Optional
from src.analysis.types.base import CacheLoaderProtocol
from src.analysis.cache import CacheLoader

__all__ = ["plot_3d_volume", "main"]

logger = logging.getLogger(__name__)

# Initialize matplotlib and numpy for this module
plt, np = init_plotting(backend="Agg")


def plot_3d_volume(ax, cube, slice_indices, title, **plot_kwargs):
    """Plot orthogonal 3D slices of a volume."""
    from src.plotting.helpers.plot import apply_plot_defaults
    from src.plotting.helpers.visualization import plot_visualization

    plot_kwargs = apply_plot_defaults(plot_kwargs)
    cmap = plot_kwargs.get("cmap", "seismic")

    return plot_visualization.plot_3d_slices(ax, cube, slice_indices, title, cmap=cmap)


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

    # Use CacheLoader to select appropriate cache file
    avo_fn = None
    if cache_loader is not None:
        avo_fn = cache_loader.select_cache_file(cache_dir, args.domain)
    else:
        # Construct a CacheLoader locally
        loader = CacheLoader()
        avo_fn = loader.select_cache_file(cache_dir, args.domain)

    return {"avo": avo_fn}


def main(argv=None):
    """CLI entrypoint for 3D slice plotting."""
    return _impl_main(argv)
