"""CLI tool implementations for seismic workflows.

This module contains all tool functions registered with @tool decorator,
including cache cleanup, plotting, analysis, and regeneration workflows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

# Third-party imports
import matplotlib.pyplot as _plt
import numpy as np

# First-party CLI helpers
from src.cli.parsers import ParserFactory, tool

# Many imports in CLI tools are intentionally performed at call-time to avoid
# heavy imports or import cycles when the CLI module is imported. Prefer
# adding a per-import suppression where that import is performed.

# The CLI performs some intentional late imports and re-exports to keep
# import-time lightweight and avoid circular dependencies. These are
# deliberate — silence import-order warnings at module level with a brief
# justification so pylint focuses on real issues.

logger = logging.getLogger(__name__)

# Shared small helpers to avoid duplication with tools_modeling
from ._tools_common import choose_html_path, save_npz  # noqa: E402

__all__ = [
    "cleanup_cache",
    "plot_3d_interactive",
    "plot_3d_slices",
    "plot_seismic_full_stack",
    "plot_rock_physics_attributes",
    "plot_original_properties",
    "analysis_rock_physics",
    "analyze_facies_correlation",
    "seismograms",
    "analysis_seismograms",
    "regenerate_seismograms",
    "regenerate_rock_physics",
    "rock_physics_attributes",
    "regenerate_all_3d_plots",
    "export_top_seismogram_layers",
    "export_top_facies_layers",
]

# Re-export plotting-related tools from the dedicated module to keep this
# file small and focused.
from .tools_plotting import (  # noqa: E402
    plot_3d_interactive,
    plot_3d_slices,
    plot_original_properties,
    plot_rock_physics_attributes,
    plot_seismic_full_stack,
    regenerate_all_3d_plots,
)


@tool
def cleanup_cache(
    cache_dir: str = ".cache", _dry_run: bool = False, verbose: bool = False
) -> tuple[int, float]:
    """Clean up old cache files (CLI tool).

    Parameters
    ----------
    cache_dir : str
        Path to cache directory
    _dry_run : bool
        If True, only report what would be cleaned
    verbose : bool
        Enable verbose logging

    Returns
    -------
    tuple[int, float]
        (number of files removed, MB freed)
    """
    ParserFactory.configure_logging(verbose)
    from src.io.pruning import (
        Pruner,
        PruneStrategy,
    )

    cache_path = Path(cache_dir)
    if cache_path.exists():
        strategy = PruneStrategy.by_size_only(max_cache_bytes=10 * 1024**3)
        pruner = Pruner(strategy)
        result = pruner.prune(cache_path)
        return result.count, result.bytes_freed / (1024**2)
    return 0, 0.0


# Modeling and analysis tools have been moved to `tools_modeling.py` to
# reduce the size of this facade module. Re-export the symbols so external
# callers can keep using the original names.
from .tools_modeling import (  # noqa: E402
    analysis_rock_physics,
    analysis_seismograms,
    analyze_facies_correlation,
    export_top_seismogram_layers,
    regenerate_seismograms,
    seismograms,
)


@tool
def export_top_facies_layers(
    cache_dir: str = ".cache",
    n_layers: int = 40 + 80,
    force_regeneration: bool = False,
    out: str | None = None,
    plot: bool = False,
    plot_out: str | None = None,
    matplotlib_only: bool = False,
) -> dict[str, str | tuple[int, int, int]]:
    """Extract the top N layers from a facies depth cache (CLI wrapper).

    Mirrors `export_top_seismogram_layers` behavior but for facies data.
    """
    ParserFactory.configure_logging(False)

    top = _get_top_layers(cache_dir, force_regeneration=force_regeneration)

    result: dict[str, str | tuple[int, int, int]] = {"shape": top.shape}

    if out:
        # If the caller asked for a specific output path, save there.
        result["saved"] = str(save_npz(Path(out), facies=top))
    else:
        # If no explicit output was provided, persist a canonical copy
        # in the cache so downstream tools can find it reliably.
        try:
            cache_path = Path(cache_dir)
            cache_file = cache_path / f"facies_top_layers_{n_layers}.npz"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            saved_cache = save_npz(cache_file, facies=top)
            result["cached"] = str(saved_cache)
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("Failed to save facies top layers to cache: %s", e)

    # Plotting behavior now delegates to FaciesTopLayersExtractor helpers
    if plot:

        out_dir = Path(plot_out).parent if plot_out else Path("docs/images")
        out_dir.mkdir(parents=True, exist_ok=True)

        if matplotlib_only:
            # Delegate to helper to keep top-level function small
            try:
                png = _plot_facies_matplotlib(top, n_layers, out_dir)
                if png:
                    result["png"] = str(png)
            except (ImportError, RuntimeError, OSError) as e:
                logger.warning("Matplotlib-only facies plotting failed: %s", e)
        else:
            try:
                html = _plot_facies_plotly(top, n_layers, out_dir, out, plot_out)
                if html:
                    result["html"] = str(html)
            except (ImportError, RuntimeError, OSError, ValueError) as e:
                logger.warning("Failed to create facies interactive plot: %s", e)

    return result


def _plot_facies_matplotlib(
    top: np.ndarray, n_layers: int, out_dir: Path
) -> Path | None:
    """Helper to create matplotlib PNG for top facies layers."""
    from src.plotting.helpers.components import (
        SliceExtractor,
    )
    from src.plotting.overlay_plotter import (
        OverlayPlotter,
    )

    png_path = out_dir / f"facies_top_layers_{n_layers}.png"

    fig, axes = _plt.subplots(1, 3, figsize=(15, 5))
    mid_i = top.shape[0] // 2

    extractor_se = SliceExtractor(shape=top.shape)
    op = OverlayPlotter()

    axes[0].set_title(f"Inline {mid_i}")
    op.plot_facies_only(axes[0], extractor_se.extract_inline(top, mid_i)[0])

    axes[1].set_title(f"Crossline {top.shape[1] // 2}")
    op.plot_facies_only(
        axes[1], extractor_se.extract_crossline(top, top.shape[1] // 2)[0]
    )

    axes[2].set_title(f"Depth slice {top.shape[2] // 2}")
    op.plot_facies_only(
        axes[2], extractor_se.extract_depthslice(top, top.shape[2] // 2)[0]
    )

    _plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    _plt.close(fig)
    return png_path


def _plot_facies_plotly(
    top: np.ndarray, n_layers: int, out_dir: Path, out: str | None, plot_out: str | None
) -> Path | None:
    """Helper to create Plotly HTML for top facies layers."""
    from src.plotting.plotly_plotter import (
        PlotlyPlotter,
    )

    plotter = PlotlyPlotter()
    html_path = choose_html_path(
        plot_out, out, out_dir, f"facies_top_layers_{n_layers}_depth.html"
    )

    # Inline creation and saving to reduce temporary locals
    plotter.save_figure(
        plotter.create_figure(
            plotter.create_3d_volume(
                top,
                (top.shape[0] // 2, top.shape[1] // 2, top.shape[2] // 2),
                is_categorical=True,
            ),
            title=f"Top {n_layers} layers",
        ),
        str(html_path),
    )
    return html_path


def _get_top_layers(cache_dir: str, force_regeneration: bool) -> np.ndarray:
    """Return the top two geological facies layers as a numpy array."""
    from src.gen.facies import DefaultCacheProvider as FaciesDefaultCacheProvider
    from src.gen.facies import FaciesTopLayersExtractor

    provider = FaciesDefaultCacheProvider(cache_dir=cache_dir)
    extractor = FaciesTopLayersExtractor.from_cache_or_generate(
        cache_provider=provider,
        cache_dir=cache_dir,
        generate_if_missing=True,
        force_regeneration=force_regeneration,
    )

    return np.asarray(extractor.extract_top_two_geological_layers())


@tool
def regenerate_rock_physics() -> bool:
    """Regenerate rock physics attributes without interactive steps.

    Returns
    -------
    bool
        True if successful
    """

    from src.analysis.common import (
        AnalysisCommon,
    )
    from src.analysis.io import HeaderPrinter
    from src.analysis.rock_physics import (
        RockPhysicsAnalyzer,
    )

    regen = AnalysisCommon.instance()

    long_desc = (
        "This pipeline clears caches, computes rock physics attributes and "
        "creates visualizations."
    )
    HeaderPrinter().print_analysis_header(
        "COMPLETE ROCK PHYSICS ANALYSIS PIPELINE",
        [
            "Compute ALL Attributes + Generate ALL Plots",
            long_desc,
        ],
    )

    cast(Any, regen).clear_cache()

    try:
        rpa = RockPhysicsAnalyzer()
        rpa.run(
            cache_dir=".cache",
            generate_plots=True,
            save_npz_only=False,
            angles_list=[0, 5, 10, 15, 20, 25],
        )
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Rock physics regeneration failed: %s", e)
        return False

    return True


@tool
def rock_physics_attributes(
    cache_dir: str = ".cache",
    generate_plots: bool = True,
    save_npz_only: bool = False,
    angles_list: list[int] | str | None = None,
    verbose: bool = False,
) -> Any:
    """Programmatic entry point for rock physics attribute computation.

    Parameters
    ----------
    cache_dir : str
        Cache directory path
    generate_plots : bool
        Whether to generate visualization plots
    save_npz_only : bool
        Save only NPZ files, skip plots and ranking
    angles_list : list[int] | str | None
        Angles to use for AVO (list or comma-separated string)
    verbose : bool
        Enable verbose logging

    Returns
    -------
    Any
        Analysis results
    """
    try:
        ParserFactory.configure_logging(verbose)
    except (RuntimeError, OSError):
        pass

    from src.analysis.rock_physics import (
        RockPhysicsAnalyzer,
    )

    try:
        if isinstance(angles_list, str):
            try:
                angles_list = [
                    int(x.strip()) for x in angles_list.split(",") if x.strip()
                ]
            except (ValueError, TypeError) as exc:
                raise SystemExit(
                    "Invalid --angles-list format; expected comma-separated ints"
                ) from exc

        rpa = RockPhysicsAnalyzer()
        return rpa.run(
            cache_dir=cache_dir,
            generate_plots=generate_plots,
            save_npz_only=save_npz_only,
            angles_list=angles_list,
            verbose=verbose,
        )
    except (RuntimeError, ImportError, ValueError, OSError) as exc:
        raise SystemExit(f"Rock physics delegator unavailable: {exc}") from exc


@tool
def resample_rock_physics_to_time(
    cache_dir: str = ".cache",
    verbose: bool = False,
) -> dict[str, Any]:
    """Resample rock physics attributes from depth domain to time domain.

    This tool loads depth-domain rock physics attributes and resamples them
    to the time domain using the P-wave velocity field. The time-domain
    attributes are saved to a separate cache file for plotting.

    Parameters
    ----------
    cache_dir : str
        Cache directory path, default: .cache
    verbose : bool
        Enable verbose logging, default: False

    Returns
    -------
    dict[str, Any]
        Result dictionary with keys:
        - success: boolean indicating success
        - input_file: source depth attributes file
        - output_file: destination time attributes file
        - attributes_resampled: list of attribute names
        - error: error message if failed (optional)
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    cache_path = Path(cache_dir)

    # Load depth domain rock physics attributes
    rp_file = cache_path / "rock_physics_attributes.npz"
    if not rp_file.exists():
        error_msg = f"Depth attributes file not found: {rp_file}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    logger.info("Loading depth-domain rock physics attributes from %s", rp_file)
    rp_data = np.load(rp_file, allow_pickle=True)

    # Load Vp and prepare resampler/plan via helpers
    try:
        grid_spec, vp_depth = _load_vp_depth()
    except (RuntimeError, ValueError, OSError) as e:
        error_msg = f"Could not load Vp: {e}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    try:
        resampler, plan = _get_resampler_and_plan(grid_spec, vp_depth)
    except (RuntimeError, ValueError, TypeError, OSError, ImportError) as e:
        error_msg = f"Could not prepare resampler: {e}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    logger.info("Resampling rock physics attributes to time domain...")
    # Delegate attribute resampling loop to helper to reduce top-level complexity
    resampled_attrs, resampled_names = _resample_attributes(
        rp_data, resampler, vp_depth, plan
    )

    # Save to new cache file
    output_file = cache_path / "rock_physics_attributes_time.npz"
    logger.info("Saving time-domain rock physics attributes to %s", output_file)
    save_npz(output_file, **resampled_attrs)

    logger.info("✓ Done! Resampled %d attributes to time domain", len(resampled_names))

    return {
        "success": True,
        "input_file": str(rp_file),
        "output_file": str(output_file),
        "attributes_resampled": resampled_names,
    }


def _resample_attributes(
    rp_data: Any, resampler: Any, vp_depth: np.ndarray, plan: Any
) -> tuple[dict[str, Any], list[str]]:
    """Resample listed rock-physics attributes to time domain.

    Extracted from `resample_rock_physics_to_time` to keep that function
    small and focused.
    """
    attributes_to_resample = [
        "lambda_rho",
        "mu_rho",
        "intercept",
        "gradient",
        "product",
        "scaled_gradient",
        "lambda_mu_ratio",
        "fluid_factor",
        "discrimination",
    ]

    resampled_attrs: dict[str, Any] = {}
    resampled_names: list[str] = []

    for attr_name in attributes_to_resample:
        if attr_name not in rp_data:
            continue

        attr_data = rp_data[attr_name]

        # Skip empty arrays or object arrays (discrimination)
        if (
            getattr(attr_data, "size", 0) == 0
            or getattr(attr_data, "dtype", None) is object
        ):
            logger.info("  Skipping %s (empty or object type)", attr_name)
            resampled_attrs[attr_name] = attr_data
            continue

        logger.info(
            "  Resampling %s: %s -> ... ", attr_name, getattr(attr_data, "shape", "?")
        )

        try:
            # Resample to time
            attr_time, _ = resampler.depth_to_time_cube(attr_data, vp_depth, plan=plan)
            logger.info("    → %s", getattr(attr_time, "shape", "?"))
            resampled_attrs[attr_name] = attr_time
            resampled_names.append(attr_name)
        except (RuntimeError, ValueError, IndexError) as e:
            logger.error("Failed to resample %s: %s", attr_name, e)
            continue

    return resampled_attrs, resampled_names


def _load_vp_depth() -> tuple[Any, np.ndarray]:
    """Load Vp property via DatasetManager and return GridSpec and Vp ndarray.

    Returns (grid_spec, vp_depth) or raises on failure.
    """
    # Lazy imports to keep module import lightweight
    from src.io.grid import GridSpec
    from src.io.loader import DatasetManager

    grid_spec = GridSpec(shape=(150, 200, 200), dz=1.0, dt=0.001)
    file_map = {"vp": "P-wave Velocity"}

    dm = DatasetManager.from_stanfordsix(".", file_map, grid_spec)
    vp_prop = dm.get_property("vp")
    if vp_prop is None:
        raise RuntimeError("Vp property not found in dataset manager")

    # Unwrap array-like wrappers (Quantity-like objects) safely
    from src.utils.quantity import to_ndarray

    vp_depth = to_ndarray(vp_prop)
    logger.info("Loaded Vp shape: %s", vp_depth.shape)
    return grid_spec, vp_depth


def _get_resampler_and_plan(grid_spec: Any, vp_depth: np.ndarray) -> tuple[Any, Any]:
    """Return (resampler, plan) for given grid_spec and vp_depth."""
    from src.processing.resampling._cache import (
        get_resample_plan_cache,
    )
    from src.processing.resampling._resampler import (
        resampler_factory,
    )

    resampler = resampler_factory.get_resampler(grid_spec)
    plan_cache = get_resample_plan_cache()
    plan = plan_cache.get_plan(grid_spec, vp_depth, target_dt=grid_spec.dt)
    return resampler, plan
