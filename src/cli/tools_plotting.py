"""Plotting-related CLI tools extracted from `tools.py`.

This module contains CLI `@tool` wrappers that perform plotting and
regeneration of interactive 3D visualizations. It is split out to reduce
complexity in the main `src.cli.tools` module.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

from src.cli.parsers import ParserFactory, tool

logger = logging.getLogger(__name__)

# CLI plotting helpers deliberately import plotting/analysis helpers at
# call-time to avoid heavy dependencies during import. Some CLI commands
# are long procedural scripts with many local temporaries; silence the
# related function-level stylistic warnings at module scope for brevity.
# Also allow call-time imports in this CLI module which intentionally
# perform lazy imports to keep the CLI lightweight on import.

# Note: call-site imports continue to use inline disables where needed.

@tool
def plot_3d_interactive(argv: list[str] | None = None) -> dict[str, str]:
    """Interactive 3D plotting using Plotly.

    Parameters
    ----------
    argv : list[str] | None
        Command line arguments

    Returns
    -------
    dict[str, str]
        Result with cache file path
    """
    from src.analysis.cache import (
        CacheLoader,
    )

    parser = argparse.ArgumentParser(
        description="Generate interactive 3D visualization"
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
    args = parser.parse_args(argv)

    loader = CacheLoader()
    avo_fn = loader.select_cache_file(args.cache_dir, args.domain)

    if not avo_fn:
        raise SystemExit(f"Missing cache file for {args.domain} domain")

    return {"cache_file": avo_fn}

@tool
def plot_3d_slices(argv: list[str] | None = None) -> dict[str, str]:
    """3D orthogonal slice visualization.

    Parameters
    ----------
    argv : list[str] | None
        Command line arguments

    Returns
    -------
    dict[str, str]
        Result with AVO file path
    """
    from src.analysis.cache import (
        CacheLoader,
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
    args = parser.parse_args(argv)

    loader = CacheLoader()
    avo_fn = loader.select_cache_file(args.cache_dir, args.domain)

    return {"avo": avo_fn or ""}

@tool
def plot_seismic_full_stack(
    domain: str = "time", cache_dir: str = ".cache", out_dir: str = "docs/images"
) -> None:
    """Generate interactive seismic full-stack 3D HTML for a domain.

    Parameters
    ----------
    domain : str
        'time' or 'depth'
    cache_dir : str
        Cache directory
    out_dir : str
        Output directory for HTML files
    """
    if domain not in ("time", "depth"):
        raise ValueError("domain must be 'time' or 'depth'")

    from src.plotting.seismic_plotter import (
        SeismicPlotter,
    )

    plotter = SeismicPlotter(cache_dir=cache_dir, out_dir=out_dir)
    generated = plotter.generate_from_caches(domain=domain)
    if generated:
        for p in generated:
            print(f"[INFO] Generated: {p}")
    else:
        print(f"[INFO] No caches found for domain: {domain}")

@tool
def plot_rock_physics_attributes(domain: str = "depth", verbose: bool = False) -> None:
    """Generate PNG plots for rock physics attributes.

    Creates individual plots for each attribute (Lambda-Rho, Mu-Rho, AVO Intercept,
    AVO Gradient) plus a comprehensive comparison plot showing all attributes.
    """
    # Import OOP plotter
    from src.plotting.property_plotter import (
        RockPhysicsPropertyPlotter,
    )

    matplotlib.use("Agg")  # Use non-interactive backend

    # Validate domain
    if domain not in ["depth", "time"]:
        raise ValueError(f"domain must be 'depth' or 'time', got '{domain}'")

    # Create plotter instance
    plotter = RockPhysicsPropertyPlotter(
        cache_dir=".cache",
        domain=domain,
        output_dir="docs/images",
        verbose=verbose,
    )

    # Generate individual attribute plots
    generated_files = plotter.generate_all_plots(file_prefix="rock_physics")

    # Generate multi-attribute comparison plot
    comparison_file = plotter.generate_comparison_plot()
    if comparison_file:
        generated_files.append(comparison_file)

    print(f"[INFO] Successfully generated {len(generated_files)} plot(s)")
    print("[INFO] Rock physics attribute plotting complete!")

@tool
def plot_original_properties(
    output_dir: str = "docs/images",
    data_dir: str = ".",
    plot_type: str = "2d",
    verbose: bool = False,
) -> dict[str, Any]:
    """Generate plots of original Stanford VI-E properties (Vp, Vs, Rho).

    This tool loads the original GSLIB data files and generates visualizations
    showing the spatial distribution of P-wave velocity, S-wave velocity,
    and density across the reservoir model.
    """
    # Import OOP plotter
    from src.plotting.property_plotter import (
        OriginalPropertyPlotter,
    )

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    try:
        # Create plotter instance
        plotter = OriginalPropertyPlotter(
            data_dir=data_dir,
            output_dir=output_dir,
            verbose=verbose,
        )

        generated_files = []

        if plot_type.lower() == "3d":
            # For 3D mode, we need to load data first
            plotter.load_data()
            # Generate 3D Plotly interactive plots
            generated_files = plotter.generate_3d_plotly_visualizations()
            # Get property names
            properties = plotter.get_properties()
        else:
            # Generate 2D matplotlib plots (includes data loading)
            generated_files = plotter.generate_all_plots(file_prefix="original")
            # Get property names
            properties = plotter.get_properties()

        logger.info(
            "✓ Generated %d original property plots (%s)",
            len(generated_files),
            plot_type,
        )

        return {
            "generated": generated_files,
            "count": len(generated_files),
            "properties": list(properties.keys()),
            "plot_type": plot_type,
        }

    except (RuntimeError, OSError, ValueError) as e:
        error_msg = f"Failed to generate original property plots: {e}"
        logger.error("%s", error_msg, exc_info=True)
        return {
            "error": error_msg,
            "generated": [],
            "count": 0,
            "properties": [],
        }

@tool
def regenerate_all_3d_plots(
    data_dir: str = ".",
    cache_dir: str = ".cache",
    output_dir: str = "docs/images",
    verbose: bool = False,
) -> dict[str, Any]:
    """Regenerate all 3D interactive plots (original properties, rock physics, seismic).

    This tool regenerates 8 interactive Plotly HTML visualizations:
    - 3 Original properties (Vp, Vs, Rho)
    - 3 Rock physics attributes (Lambda-Rho, Mu-Rho, Intercept, Gradient)
    - 2 Seismic full stack (Time-domain, Depth-domain)
    """
    if verbose:
        ParserFactory.configure_logging(True)

    try:
        from src.plotting.property_plotter import (
            OriginalPropertyPlotter,
            RockPhysicsPropertyPlotter,
        )
        from src.plotting.seismic_plotter import (
            SeismicPlotter,
        )

        logger.info("=" * 80)
        logger.info("REGENERATING ALL 3D INTERACTIVE PLOTS")
        logger.info("=" * 80)

        all_files: list[str] = []
        results: dict[str, Any] = {
            "original_properties": [],
            "rock_physics": [],
            "seismic": [],
        }

        # 1. Original Properties (3 plots: Vp, Vs, Rho)
        logger.info("\n[1/3] Regenerating ORIGINAL PROPERTIES 3D plots...")
        try:
            plotter = OriginalPropertyPlotter(
                data_dir=data_dir,
                output_dir=output_dir,
                verbose=verbose,
            )
            plotter.load_data()
            files_paths = plotter.generate_3d_plotly_visualizations()
            files = [str(p) for p in files_paths]
            results["original_properties"] = files
            all_files.extend(files)
            logger.info("✓ Generated %d original property plots:", len(files))
            for fp in files:
                logger.info("  - %s", Path(fp).name)
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Error generating original properties: %s", e, exc_info=True)
            return {
                "error": str(e),
                "original_properties": [],
                "rock_physics": [],
                "seismic": [],
                "total_count": 0,
            }

        # 2. Rock Physics Attributes (3 plots: Lambda-Rho, Mu-Rho, Intercept, Gradient)
        logger.info("\n[2/3] Regenerating ROCK PHYSICS ATTRIBUTES 3D plots...")
        try:
            plotter_rp = RockPhysicsPropertyPlotter(
                cache_dir=cache_dir,
                output_dir=output_dir,
                verbose=verbose,
            )
            plotter_rp.load_data()
            files_rp_paths = plotter_rp.generate_3d_plotly_visualizations()
            files_rp = [str(p) for p in files_rp_paths]
            results["rock_physics"] = files_rp
            all_files.extend(files_rp)
            logger.info("✓ Generated %d rock physics plots:", len(files_rp))
            for fp in files_rp:
                logger.info("  - %s", Path(fp).name)
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Error generating rock physics: %s", e, exc_info=True)
            return {
                "error": str(e),
                "original_properties": results["original_properties"],
                "rock_physics": [],
                "seismic": [],
                "total_count": len(all_files),
            }

        # 3. Seismic Full Stack (2 plots: Time-domain, Depth-domain)
        logger.info("\n[3/3] Regenerating SEISMIC FULL STACK 3D plots...")
        try:
            plotter_seismic = SeismicPlotter(
                cache_dir=cache_dir,
                out_dir=output_dir,
                verbose=verbose,
            )
            files_seismic_time = plotter_seismic.generate_from_caches(domain="time")
            files_seismic_depth = plotter_seismic.generate_from_caches(domain="depth")
            files_seismic_paths = files_seismic_time + files_seismic_depth
            files_seismic = [str(p) for p in files_seismic_paths]
            results["seismic"] = files_seismic
            all_files.extend(files_seismic)
            logger.info("✓ Generated %d seismic plots:", len(files_seismic))
            for fp in files_seismic:
                logger.info("  - %s", Path(fp).name)
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Error generating seismic plots: %s", e, exc_info=True)
            return {
                "error": str(e),
                "original_properties": results["original_properties"],
                "rock_physics": results["rock_physics"],
                "seismic": [],
                "total_count": len(all_files),
            }

        logger.info("\n%s", "=" * 80)
        logger.info("✓ ALL 3D PLOTS REGENERATED SUCCESSFULLY!")
        logger.info("=" * 80)

        # Verify all files were created
        plot_files = sorted(Path(output_dir).glob("*_3d.html"))
        logger.info("\nVerifying %d plot files in %s:", len(plot_files), output_dir)
        for p in plot_files:
            logger.info("  ✓ %s", p.name)

        results["total_count"] = len(all_files)
        results["verified_count"] = len(plot_files)

        return results

    except (RuntimeError, OSError, ValueError) as e:
        error_msg = f"Failed to regenerate 3D plots: {e}"
        logger.error("%s", error_msg, exc_info=True)
        return {
            "error": error_msg,
            "original_properties": [],
            "rock_physics": [],
            "seismic": [],
            "total_count": 0,
        }
