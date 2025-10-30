"""Rock physics attribute plotting moved into src.plotting."""

from src.plotting.helpers.plot import init_plotting, plot_helper

# Initialize plotting (matplotlib) for this module
plt, np = init_plotting(backend="Agg")

import logging

logger = logging.getLogger(__name__)

__all__ = ["plot_attribute", "main"]


# Note: plotting modules now prefer using `PlotConfig` / `GridSpec` from
# `src.plotting.helpers.plot.default_plot_config()` instead of module-level
# tuple constants. Obtain a `GridSpec` from `PlotConfig.grid_spec` for
# per-module defaults and avoid keeping separate GRID_SHAPE/DZ/DT tuples.


def plot_attribute(ax, data, idx, slice_type, title, cmap="viridis"):
    if slice_type == "inline":
        slice_data = data[idx, :, :]
        plot_helper.imshow_with_labels(
            ax,
            slice_data,
            f"{title} (Inline {idx})",
            xlabel="Crossline Index",
            k_label="Depth",
            k_unit="m",
            cmap=cmap,
            origin="upper",
            interpolation="bilinear",
        )
    elif slice_type == "crossline":
        slice_data = data[:, idx, :]
        plot_helper.imshow_with_labels(
            ax,
            slice_data,
            f"{title} (Crossline {idx})",
            xlabel="Inline Index",
            k_label="Depth",
            k_unit="m",
            cmap=cmap,
            origin="upper",
            interpolation="bilinear",
        )
    else:
        slice_data = data[:, :, idx]
        plot_helper.imshow_with_labels(
            ax,
            slice_data,
            f"{title} (Depth {idx}m)",
            xlabel="Inline Index",
            k_label="Crossline Index",
            k_unit="",
            cmap=cmap,
            origin="upper",
            interpolation="bilinear",
        )


def main(argv=None):
    """Minimal CLI wrapper to visualize rock physics attributes.

    This wrapper provides a canonical `main()` so `src.__main__` can delegate.
    """
    import argparse
    from src.plotting.helpers.plot import prepare_plotting_args, default_plot_config
    import logging

    parser = argparse.ArgumentParser(description="Visualize rock physics attributes")
    # Mirror ParserFactory.common_parser exactly
    parser.add_argument(
        "--domain",
        choices=["depth", "time"],
        default="depth",
        help="Domain for processing/visualization (default: depth)",
    )
    parser.add_argument(
        "--no-multiangle",
        action="store_true",
        help="Disable multi-angle EI processing and use single-angle fallback",
    )
    parser.add_argument(
        "--cache-dir", default=".cache", help="Directory for cache files"
    )
    parser.add_argument(
        "--backend", default=None, help="Optional matplotlib backend override"
    )
    # ParserFactory.start_plot_main configures logging from `--verbose` when present;
    # add a verbose flag for parity with CLI tooling.
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging across tools",
    )

    args = parser.parse_args(argv)

    # Configure basic logging consistent with ParserFactory.configure_logging
    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

    # Normalize plotting flags as the central helpers expect
    prepare_plotting_args(args)

    plot_cfg = default_plot_config()
    gs = plot_cfg.grid_spec
    DATA_PATH, FILE_MAP, grid_spec = plot_cfg.data_path, plot_cfg.file_map, gs

    cache_dir = args.cache_dir

    from src.io.cache import cache_for_dir

    groups = cache_for_dir(cache_dir).select_latest_cache_entries()
    hybrid_entries = groups.get("rock_physics_", []) or groups.get("rock_physics", [])

    if len(hybrid_entries) == 0:
        raise SystemExit("No rock physics cache file found")

    hybrid_fn = str(hybrid_entries[-1].path)
    logger = logging.getLogger(__name__)
    logger.info("Selected rock physics cache: %s", hybrid_fn)

    # For parity with ParserFactory.start_plot_main(), return args and a
    # Use GridSpec from PlotConfig rather than keeping separate tuple constants.
    from src.plotting.helpers.plot import compute_boundary_alignment

    return args, DATA_PATH, FILE_MAP, grid_spec, compute_boundary_alignment
