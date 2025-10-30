"""3D slice plotting (matplotlib) moved into src.plotting package."""

import logging

from src.plotting.helpers.plot import (
    init_plotting,
)

__all__ = ["plot_3d_volume", "main"]

logger = logging.getLogger(__name__)

# Initialize matplotlib and numpy for this module
plt, np = init_plotting(backend="Agg")


def plot_3d_volume(ax, cube, slice_indices, title, **plot_kwargs):
    """Backward-compatible wrapper: normalize kwargs and delegate to the
    PlotVisualization facade. This keeps the top-level API while routing
    implementation through the OO facade.
    """
    return plot_3d_slices.plot_3d_volume(ax, cube, slice_indices, title, **plot_kwargs)


def main(argv=None):
    """Canonical CLI wrapper for 3D slice plotting.

    This wrapper exposes a minimal entrypoint used by the centralized
    `src.__main__` delegators.
    """
    import argparse
    from src.plotting.helpers.plot import (
        plot_helper,
        prepare_plotting_args,
        default_plot_config,
    )

    import logging

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
        "--no-multiangle", action="store_true", help="Disable multi-angle EI processing"
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

    avo_fn, ai_fn, ei_fn, ei_data_key, ei_type_str, ei_is_depth_domain = (
        plot_helper.select_cache_files(cache_dir, args.domain)
    )

    return {
        "avo": avo_fn,
        "ai": ai_fn,
        "ei": ei_fn,
        "ei_key": ei_data_key,
        "ei_type": ei_type_str,
        "is_depth": ei_is_depth_domain,
    }


from src.utils.facades import LazyObjectProxy


class Plot3DSlices:
    """Facade for 3D slice plotting helpers."""

    def plot_3d_volume(self, ax, cube, slice_indices, title, **plot_kwargs):
        from src.plotting.helpers.plot import apply_plot_defaults
        from src.plotting.helpers.visualization import plot_visualization

        plot_kwargs = apply_plot_defaults(plot_kwargs)
        cmap = plot_kwargs.get("cmap", "seismic")

        return plot_visualization.plot_3d_slices(
            ax, cube, slice_indices, title, cmap=cmap
        )

    def main(self, argv=None):
        return _impl_main(argv)


# Module-level lazy proxy for gradual migration
plot_3d_slices: Plot3DSlices = LazyObjectProxy(lambda: Plot3DSlices())


def get_plot_3d_slices(instance: Plot3DSlices | None = None) -> Plot3DSlices:
    return instance if instance is not None else plot_3d_slices


def _impl_main(argv=None):
    import argparse
    from src.plotting.helpers.plot import (
        plot_helper,
        prepare_plotting_args,
        default_plot_config,
    )

    import logging

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
        "--no-multiangle", action="store_true", help="Disable multi-angle EI processing"
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

    avo_fn, ai_fn, ei_fn, ei_data_key, ei_type_str, ei_is_depth_domain = (
        plot_helper.select_cache_files(cache_dir, args.domain)
    )

    return {
        "avo": avo_fn,
        "ai": ai_fn,
        "ei": ei_fn,
        "ei_key": ei_data_key,
        "ei_type": ei_type_str,
        "is_depth": ei_is_depth_domain,
    }
