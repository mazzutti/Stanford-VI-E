"""CLI tool implementations for seismic workflows.

This module contains all tool functions registered with @tool decorator,
including cache cleanup, plotting, analysis, and regeneration workflows.
"""

from __future__ import annotations
    from src.gen.seismogram import (
        SeismogramTopLayersExtractor,
        DefaultCacheProvider as SeismogramDefaultCacheProvider,
    )
from src.cli.parsers import ParserFactory, tool

logger = logging.getLogger(__name__)

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
    try:
        ParserFactory.configure_logging(verbose)
        from src.gen.facies import (
            FaciesTopLayersExtractor,
            DefaultCacheProvider as FaciesDefaultCacheProvider,
        )
    from src.io.pruning import Pruner, PruneStrategy

    cache_path = Path(cache_dir)
    if cache_path.exists():
        strategy = PruneStrategy.by_size_only(max_cache_bytes=10 * 1024**3)
        pruner = Pruner(strategy)
        result = pruner.prune(cache_path)
        return result.count, result.bytes_freed / (1024**2)
    return 0, 0.0


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
    import argparse
    from src.analysis.cache import CacheLoader

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
    import argparse
    from src.analysis.cache import CacheLoader

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

    from src.plotting.seismic_plotter import SeismicPlotter

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

    Parameters
    ----------
    domain : str
        Domain for visualization. Either "depth" or "time".
        Default is "depth".
    verbose : bool
        Enable verbose logging. Default is False.

    Raises
    ------
    FileNotFoundError
        If cache files are missing
    ValueError
        If domain is not "depth" or "time"

    Examples
    --------
    >>> # Generate depth domain rock physics plots
    >>> plot_rock_physics_attributes(domain="depth")
    [INFO] Successfully generated 5 plot(s)

    >>> # Generate time domain plots with verbose logging
    >>> plot_rock_physics_attributes(domain="time", verbose=True)
    [DEBUG] Loading cache file: .cache/rock_physics_attributes_time.npz
    [INFO] Successfully generated 5 plot(s)
    """
    import matplotlib

    # Import OOP plotter
    from src.plotting.property_plotter import RockPhysicsPropertyPlotter

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
def analysis_rock_physics(
    _venv_python: str | None = None, cache_dir: str = ".cache", _prompt: bool = True
) -> bool:
    """Rock physics analysis pipeline with cache clearing and visualization.

    Parameters
    ----------
    _venv_python : str | None
        Virtual environment Python path (legacy)
    cache_dir : str
        Cache directory path
    _prompt : bool
        Whether to prompt user (legacy)

    Returns
    -------
    bool
        True if successful
    """
    from src.analysis.io import HeaderPrinter
    from src.analysis.common import AnalysisCommon

    analysis = AnalysisCommon.instance()
    long_desc = (
        "This pipeline clears caches, computes rock physics attributes and "
        "creates visualizations (AVO-focused)."
    )
    HeaderPrinter().print_analysis_header(
        "COMPLETE ROCK PHYSICS ANALYSIS PIPELINE",
        [
            "Compute ALL Attributes + Generate ALL Plots",
            long_desc,
        ],
    )

    analysis.clear_cache()

    try:
        from src.analysis.rock_physics import RockPhysicsAnalyzer

        rpa = RockPhysicsAnalyzer()
        rpa.run(
            cache_dir=cache_dir,
            generate_plots=True,
            save_npz_only=False,
            angles_list=[0, 5, 10, 15, 20, 25],
            verbose=False,
        )
    except Exception as e:
        logger.error("ERROR: Rock physics pipeline failed: %s", e)
        return False

    try:
        analysis.summarize_cache_files(cache_dir=cache_dir)
    except Exception:
        pass

    return True


@tool
def analyze_facies_correlation(
    cache_dir: str = ".cache",
    domain: str = "depth",
    _no_multiangle: bool = False,
    verbose: bool = False,
) -> Any:
    """Analyze facies correlation across domains.

    Parameters
    ----------
    cache_dir : str
        Cache directory path
    domain : str
        Domain for analysis ('depth' or 'time')
    _no_multiangle : bool
        Legacy parameter (unused)
    verbose : bool
        Enable verbose logging

    Returns
    -------
    Any
        Analysis results
    """
    from src.analysis.facies import FaciesCorrelationAnalyzer
    from src.analysis.domain.enum import Domain

    analyzer = FaciesCorrelationAnalyzer()
    domain_enum = Domain(domain)
    return analyzer.run(
        cache_dir=cache_dir,
        domain=domain_enum,
        verbose=verbose,
    )


@tool
def seismograms(
    cache_dir: str = ".cache",
    _venv_python: str | None = None,
    skip_cleanup: bool = False,
    verbose: bool = False,
) -> Any:
    """Seismogram modeling pipeline.

    Parameters
    ----------
    cache_dir : str
        Cache directory path
    _venv_python : str | None
        Virtual environment Python path (legacy)
    skip_cleanup : bool
        Skip cache cleanup
    verbose : bool
        Enable verbose logging

    Returns
    -------
    Any
        Pipeline results
    """
    from src.analysis.pipelines import SeismogramAnalyzer

    analyzer = SeismogramAnalyzer()
    return analyzer.run(
        cache_dir=cache_dir,
        skip_cleanup=skip_cleanup,
        verbose=verbose,
    )


@tool
def analysis_seismograms() -> bool:
    """Complete seismic modeling pipeline with analysis and visualization.

    Returns
    -------
    bool
        True if successful
    """
    from src.analysis.common import AnalysisCommon
    from src.analysis.pipelines import SeismogramAnalyzer
    from src.analysis.facies import FaciesCorrelationAnalyzer
    from src.analysis.domain.enum import Domain
    from src.analysis.cache import CacheLoader

    analysis = AnalysisCommon.instance()

    logger.info("%s", "=" * 70)
    logger.info("COMPLETE SEISMIC MODELING PIPELINE - DUAL DOMAIN")
    logger.info(
        "Regenerate ALL Data + Generate ALL Plots (DEPTH & TIME) + Open Everything"
    )
    logger.info("%s", "=" * 70)
    logger.info("")

    analysis.clear_cache()

    # Run seismic modeling
    try:
        _seis = SeismogramAnalyzer()
        _seis.run(cache_dir=".cache", skip_cleanup=True)
    except Exception as e:
        logger.error("Seismic modeling failed: %s", e)
        return False

    # Run facies correlation (depth)
    try:
        _fac = FaciesCorrelationAnalyzer()
        _fac.run(cache_dir=".cache", domain=Domain.DEPTH)
    except Exception as e:
        logger.warning("Facies depth analysis failed: %s", e)

    # Run facies correlation (time)
    try:
        _fac_time = FaciesCorrelationAnalyzer()
        _fac_time.run(cache_dir=".cache", domain=Domain.TIME)
    except Exception as e:
        logger.warning("Facies time analysis failed: %s", e)

    # Interactive 3D plots
    try:
        _loader = CacheLoader()
        _avo_fn = _loader.select_cache_file(".cache", "depth")
        if _avo_fn:
            logger.info("Generated 3D interactive plot for depth domain")
    except Exception as e:
        logger.warning("3D interactive plot (depth) failed: %s", e)

    try:
        _loader = CacheLoader()
        _avo_fn = _loader.select_cache_file(".cache", "time")
        if _avo_fn:
            logger.info("Generated 3D interactive plot for time domain")
    except Exception as e:
        logger.warning("3D interactive plot (time) failed: %s", e)

    return True


@tool
def regenerate_seismograms() -> bool:
    """Regenerate seismograms without interactive steps.

    Returns
    -------
    bool
        True if successful
    """
    from src.analysis.common import AnalysisCommon
    from src.analysis.pipelines import SeismogramAnalyzer

    regen = AnalysisCommon.instance()

    logger.info("%s", "=" * 70)
    logger.info("COMPLETE SEISMIC MODELING PIPELINE - DUAL DOMAIN")
    logger.info(
        "Regenerate ALL Data + Generate ALL Plots (DEPTH & TIME) + Open Everything"
    )
    logger.info("%s", "=" * 70)
    logger.info("")

    regen.clear_cache()
    try:
        _seis = SeismogramAnalyzer()
        _seis.run(cache_dir=".cache", skip_cleanup=True)
    except Exception as e:
        logger.error("Seismic modeling failed: %s", e)
        return False

    return True


@tool
def plot_seismograms(
    time_cache: str = ".cache/avo_time.npz",
    depth_cache: str = ".cache/avo_depth.npz",
    output_dir: str = "docs/images",
) -> bool:
    """Generate all seismogram PNG plots from cache files.

    Generates plots for both time and depth domains:
    - Full stack plots
    - Individual angle stack plots

    Parameters
    ----------
    time_cache : str
        Path to time domain cache file, default: .cache/avo_time.npz
    depth_cache : str
        Path to depth domain cache file, default: .cache/avo_depth.npz
    output_dir : str
        Output directory for PNG files, default: docs/images

    Returns
    -------
    bool
        True if successful

    Examples
    --------
    $ python -m src plot_seismograms
    $ python -m src plot_seismograms --time_cache .cache/avo_time.npz
    """
    from pathlib import Path

    from src.plotting import SeismogramPlotter

    logger.info("%s", "=" * 70)
    logger.info("SEISMOGRAM PLOT GENERATION")
    logger.info("%s", "=" * 70)

    cache_dir = Path(".cache")
    out_dir = Path(output_dir)

    # Find cache files (they may have hash suffixes)
    time_files_list = list(cache_dir.glob("avo_time*.npz"))
    depth_files_list = list(cache_dir.glob("avo_depth*.npz"))
    return True


@tool
def export_top_seismogram_layers(
    cache_dir: str = ".cache",
    n_layers: int = 2,
    force: bool = False,
    out: str | None = None,
    plot: bool = False,
    plot_out: str | None = None,
    geology: bool = False,
    save_to_cache: bool = False,
    matplotlib_only: bool = False,
) -> dict[str, str | tuple[int, int, int]]:
    """Extract the top N layers from the depth-domain seismogram cache.

    Parameters
    ----------
    cache_dir: str
        Directory where `.cache/avo_depth_*.npz` files live.
    n_layers: int
        Number of top layers to extract (default 2).
    force: bool
        If True, force regeneration of the depth-domain seismogram cache.
    out: str | None
        If provided, path to save the extracted top-layers as an NPZ.

    Returns
    -------
    dict
        Information about the saved file and the shape of the extracted cube.
    """
    ParserFactory.configure_logging(False)

    from pathlib import Path

    from src.gen.seismogram_extractor import (
        SeismogramTopLayersExtractor,
        DefaultCacheProvider,
    )

    provider = DefaultCacheProvider(cache_dir=cache_dir)
    extractor = SeismogramTopLayersExtractor.from_cache_or_generate(
        cache_provider=provider, cache_dir=cache_dir, generate_if_missing=True, force_generate=force
    )

    if geology:
        # Extract the top two geological layers (Stanford VI: 80 m + 40 m)
        top = extractor.extract_top_two_geological_layers()
    else:
        top = extractor.extract_top_layers(n_layers)

    result: dict[str, str | tuple[int, int, int]] = {}
    result["shape"] = top.shape
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        import numpy as _np

        _np.savez_compressed(p, top=top)
        result["saved"] = str(p)

    # Optionally save into cache directory for future reloads
    if save_to_cache:
        try:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            import time

            ts = int(time.time())
            cache_file = cache_path / f"seismogram_depth_top_layers_{ts}.npz"
            import numpy as _np

            _np.savez_compressed(cache_file, top=top)
            result["cache"] = str(cache_file)
        except Exception as e:
            logger.warning("Failed to save top layers to cache: %s", e)

    # Optionally create plots. If `matplotlib_only` is True, use SeismicPlotter (Matplotlib)
    if plot:
        if matplotlib_only:
            try:
                from src.plotting.seismic_plotter import SeismicPlotter

                out_dir = Path(plot_out) if plot_out else Path("docs/images")
                out_dir.mkdir(parents=True, exist_ok=True)
                png_path = out_dir / f"seismic_top_layers_matplotlib_{n_layers}.png"
                sp = SeismicPlotter(cache_dir=cache_dir, out_dir=str(out_dir))
                sp.plot_full_stack(top, output_path=png_path, domain="depth")
                result["png"] = str(png_path)
            except Exception as e:
                logger.warning("Matplotlib-only plotting failed: %s", e)
        else:
            try:
                from src.plotting.plotly_plotter import PlotlyPlotter

                # Determine output HTML path
                if plot_out:
                    html_path = Path(plot_out)
                elif out:
                    # If an NPZ path was provided, place HTML next to it
                    html_path = Path(out).with_suffix(".html")
                else:
                    # Default to docs/images to match project convention
                    html_path = Path("docs/images") / f"seismic_top_layers_{n_layers}_depth.html"

                html_path.parent.mkdir(parents=True, exist_ok=True)

                plotter = PlotlyPlotter()

                # Choose central slices for inline/crossline/depth (exact middle)
                ni, nj, nk = top.shape
                inline_idx = ni // 2
                crossline_idx = nj // 2
                depth_idx = nk // 2

                traces = plotter.create_3d_volume(
                    top, (inline_idx, crossline_idx, depth_idx), title=f"Top {n_layers} layers",
                )
                fig = plotter.create_figure(traces, title=f"Top {n_layers} layers")
                plotter.save_figure(fig, str(html_path))
                result["html"] = str(html_path)
                # Also attempt to export a static PNG similar to seismic_full_stack_depth.png
                try:
                    png_path = html_path.with_suffix(".png")
                    # write_image requires plotly[kaleido] or orca; attempt first
                    fig.write_image(str(png_path))
                    result["png"] = str(png_path)
                except Exception as e:
                    logger.info("Plotly PNG export failed: %s; falling back to Matplotlib renderer", e)
                    try:
                        # Use SeismicPlotter's Matplotlib renderer to create a comparable PNG
                        from src.plotting.seismic_plotter import SeismicPlotter

                        sp = SeismicPlotter(cache_dir=cache_dir, out_dir=str(html_path.parent))
                        # Use a descriptive filename matching project convention
                        png_path = html_path.with_name(f"seismic_full_stack_top_layers_depth.png")
                        sp.plot_full_stack(top, output_path=png_path, domain="depth")
                        result["png"] = str(png_path)
                    except Exception as e2:
                        logger.info("Matplotlib fallback PNG export failed: %s", e2)
            except Exception as e:
                logger.warning("Failed to create interactive plot: %s", e)

    return result


@tool
def export_top_facies_layers(
    cache_dir: str = ".cache",
    n_layers: int = 2,
    force: bool = False,
    out: str | None = None,
    plot: bool = False,
    plot_out: str | None = None,
    geology: bool = False,
    save_to_cache: bool = False,
    matplotlib_only: bool = False,
) -> dict[str, str | tuple[int, int, int]]:
    """Extract the top N layers from a facies depth cache (CLI wrapper).

    Mirrors `export_top_seismogram_layers` behavior but for facies data.
    """
    ParserFactory.configure_logging(False)

    from pathlib import Path

    from src.gen.facies_extractor import (
        FaciesTopLayersExtractor,
        DefaultCacheProvider as FaciesDefaultCacheProvider,
    )

    provider = FaciesDefaultCacheProvider(cache_dir=cache_dir)
    extractor = FaciesTopLayersExtractor.from_cache_or_generate(
        cache_provider=provider, cache_dir=cache_dir, generate_if_missing=True, force_generate=force
    )

    if geology:
        top = extractor.extract_top_two_geological_layers()
    else:
        top = extractor.extract_top_layers(n_layers)

    result: dict[str, str | tuple[int, int, int]] = {}
    result["shape"] = top.shape
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        import numpy as _np

        _np.savez_compressed(p, facies=top)
        result["saved"] = str(p)

    if save_to_cache:
        try:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            import time

            ts = int(time.time())
            cache_file = cache_path / f"facies_depth_top_layers_{ts}.npz"
            import numpy as _np

            _np.savez_compressed(cache_file, facies=top)
            result["cache"] = str(cache_file)
        except Exception as e:
            logger.warning("Failed to save facies top layers to cache: %s", e)

    # Plotting behavior mirrors seismogram tool (Matplotlib or Plotly)
    if plot:
        if matplotlib_only:
            try:
                from src.plotting.seismic_plotter import SeismicPlotter

                out_dir = Path(plot_out) if plot_out else Path("docs/images")
                out_dir.mkdir(parents=True, exist_ok=True)
                png_path = out_dir / f"facies_top_layers_matplotlib_{n_layers}.png"
                sp = SeismicPlotter(cache_dir=cache_dir, out_dir=str(out_dir))
                sp.plot_full_stack(top, output_path=png_path, domain="depth")
                result["png"] = str(png_path)
            except Exception as e:
                logger.warning("Matplotlib-only facies plotting failed: %s", e)
        else:
            try:
                from src.plotting.plotly_plotter import PlotlyPlotter

                if plot_out:
                    html_path = Path(plot_out)
                elif out:
                    html_path = Path(out).with_suffix(".html")
                else:
                    html_path = Path("docs/images") / f"facies_top_layers_{n_layers}_depth.html"

                html_path.parent.mkdir(parents=True, exist_ok=True)

                plotter = PlotlyPlotter()
                ni, nj, nk = top.shape
                inline_idx = ni // 2
                crossline_idx = nj // 2
                depth_idx = nk // 2

                traces = plotter.create_3d_volume(
                    top, (inline_idx, crossline_idx, depth_idx), title=f"Facies Top {n_layers} layers",
                )
                fig = plotter.create_figure(traces, title=f"Facies Top {n_layers} layers")
                plotter.save_figure(fig, str(html_path))
                result["html"] = str(html_path)
                try:
                    png_path = html_path.with_suffix(".png")
                    fig.write_image(str(png_path))
                    result["png"] = str(png_path)
                except Exception:
                    # fallback to Matplotlib
                    try:
                        from src.plotting.seismic_plotter import SeismicPlotter

                        sp = SeismicPlotter(cache_dir=cache_dir, out_dir=str(html_path.parent))
                        png_path = html_path.with_name(f"facies_full_stack_top_layers_depth.png")
                        sp.plot_full_stack(top, output_path=png_path, domain="depth")
                        result["png"] = str(png_path)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("Failed to create facies interactive plot: %s", e)

    return result

    if not time_files_list and not depth_files_list:
        logger.error("No cache files found. Run 'analysis_seismograms' first.")
        return False

    plotter = SeismogramPlotter(verbose=True)

    # Generate time domain plots
    if time_files_list:
        time_path = time_files_list[0]  # Use most recent
        logger.info(f"\nGenerating TIME DOMAIN seismogram plots...")
        logger.info(f"  Cache: {time_path}")
        logger.info(f"  Output: {out_dir}")
        time_files = plotter.plot_from_cache(time_path, out_dir, domain="time")
        logger.info(
            f"✓ Generated {len(time_files['angle_stacks']) + len(time_files['full_stack'])} time domain plot(s)\n"
        )
    else:
        logger.warning(f"Time domain cache not found in {cache_dir}")

    # Generate depth domain plots
    if depth_files_list:
        depth_path = depth_files_list[0]  # Use most recent
        logger.info(f"\nGenerating DEPTH DOMAIN seismogram plots...")
        logger.info(f"  Cache: {depth_path}")
        logger.info(f"  Output: {out_dir}")
        depth_files = plotter.plot_from_cache(depth_path, out_dir, domain="depth")
        logger.info(
            f"✓ Generated {len(depth_files['angle_stacks']) + len(depth_files['full_stack'])} depth domain plot(s)\n"
        )
    else:
        logger.warning(f"Depth domain cache not found in {cache_dir}")

    logger.info("%s", "=" * 70)
    logger.info("✓ SEISMOGRAM PLOTS GENERATED SUCCESSFULLY")
    logger.info("%s", "=" * 70)

    return True


@tool
def regenerate_rock_physics() -> bool:
    """Regenerate rock physics attributes without interactive steps.

    Returns
    -------
    bool
        True if successful
    """
    try:
        from src.analysis import regenerate_common as regen  # type: ignore
    except Exception:
        from src.analysis.common import AnalysisCommon

        regen = AnalysisCommon.instance()

    from src.analysis.io import HeaderPrinter
    from src.analysis.rock_physics import RockPhysicsAnalyzer

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

    regen.clear_cache()

    try:
        rpa = RockPhysicsAnalyzer()
        rpa.run(
            cache_dir=".cache",
            generate_plots=True,
            save_npz_only=False,
            angles_list=[0, 5, 10, 15, 20, 25],
        )
    except Exception as e:
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
    except Exception:
        pass

    from src.analysis.rock_physics import RockPhysicsAnalyzer

    try:
        if isinstance(angles_list, str):
            try:
                angles_list = [
                    int(x.strip()) for x in angles_list.split(",") if x.strip()
                ]
            except Exception:
                raise SystemExit(
                    "Invalid --angles-list format; expected comma-separated ints"
                )

        rpa = RockPhysicsAnalyzer()
        return rpa.run(
            cache_dir=cache_dir,
            generate_plots=generate_plots,
            save_npz_only=save_npz_only,
            angles_list=angles_list,
            verbose=verbose,
        )
    except Exception as exc:
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
    import numpy as np
    from pathlib import Path

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

    logger.info(f"Loading depth-domain rock physics attributes from {rp_file}")
    rp_data = np.load(rp_file, allow_pickle=True)

    # Load Vp from GSLIB file for resampling
    logger.info("Loading Vp from GSLIB file for resampling...")
    from src.io.loader import DatasetManager
    from src.io.grid import GridSpec

    grid_spec = GridSpec(shape=(150, 200, 200), dz=1.0, dt=0.001)

    file_map = {"vp": "P-wave Velocity"}

    try:
        dm = DatasetManager.from_stanfordsix(".", file_map, grid_spec)
        vp_prop = dm.get_property("vp")
        vp_depth = vp_prop.array if hasattr(vp_prop, "array") else vp_prop
        logger.info(f"Loaded Vp shape: {vp_depth.shape}")
    except Exception as e:
        error_msg = f"Could not load Vp: {e}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    # Initialize resampler
    from src.processing.resampling._resampler import resampler_factory
    from src.processing.resampling._cache import get_resample_plan_cache

    resampler = resampler_factory.get_resampler(grid_spec)
    plan_cache = get_resample_plan_cache()
    plan = plan_cache.get_plan(grid_spec, vp_depth, target_dt=grid_spec.dt)

    logger.info("Resampling rock physics attributes to time domain...")

    # Attributes to resample
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

    resampled_attrs = {}
    resampled_names = []

    for attr_name in attributes_to_resample:
        if attr_name not in rp_data:
            continue

        attr_data = rp_data[attr_name]

        # Skip empty arrays or object arrays (discrimination)
        if attr_data.size == 0 or attr_data.dtype == object:
            logger.info(f"  Skipping {attr_name} (empty or object type)")
            resampled_attrs[attr_name] = attr_data
            continue

        logger.info(f"  Resampling {attr_name}: {attr_data.shape} -> ... ")

        try:
            # Resample to time
            attr_time, dt = resampler.depth_to_time_cube(attr_data, vp_depth, plan=plan)
            logger.info(f"    → {attr_time.shape}")
            resampled_attrs[attr_name] = attr_time
            resampled_names.append(attr_name)
        except Exception as e:
            logger.error(f"Failed to resample {attr_name}: {e}")
            continue

    # Save to new cache file
    output_file = cache_path / "rock_physics_attributes_time.npz"
    logger.info(f"Saving time-domain rock physics attributes to {output_file}")
    np.savez_compressed(output_file, **resampled_attrs)

    logger.info(f"✓ Done! Resampled {len(resampled_names)} attributes to time domain")

    return {
        "success": True,
        "input_file": str(rp_file),
        "output_file": str(output_file),
        "attributes_resampled": resampled_names,
    }


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

    Parameters
    ----------
    output_dir : str
        Output directory for plot files, default: docs/images
    data_dir : str
        Root directory containing Stanford VI-E data folders, default: .
    plot_type : str
        Type of plot: '2d' for PNG slices (matplotlib) or '3d' for interactive
        HTML volume (Plotly), default: 2d
    verbose : bool
        Enable verbose logging, default: False

    Returns
    -------
    dict[str, Any]
        Result dictionary with keys:
        - generated: list of generated file paths
        - count: number of files generated
        - properties: list of property names plotted
        - plot_type: type of plots generated
        - error: error message if failed (optional)
    """
    # Import OOP plotter
    from src.plotting.property_plotter import OriginalPropertyPlotter

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
            f"✓ Generated {len(generated_files)} original property plots ({plot_type})"
        )

        return {
            "generated": generated_files,
            "count": len(generated_files),
            "properties": list(properties.keys()),
            "plot_type": plot_type,
        }

    except Exception as e:
        error_msg = f"Failed to generate original property plots: {e}"
        logger.error(error_msg, exc_info=True)
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

    All plots include aggressive zoom controls (2.5x sensitivity),
    persistent axis titles on resize, and fullscreen CSS.

    Parameters
    ----------
    data_dir : str
        Root directory containing Stanford VI-E property data, default: .
    cache_dir : str
        Cache directory for rock physics and seismic data, default: .cache
    output_dir : str
        Output directory for generated HTML files, default: docs/images
    verbose : bool
        Enable verbose logging, default: False

    Returns
    -------
    dict[str, Any]
        Dictionary with generation results including file counts and paths

    Examples
    --------
    $ python -m src regenerate_all_3d_plots
    $ python -m src regenerate_all_3d_plots --output-dir custom/output
    """
    from pathlib import Path

    if verbose:
        ParserFactory.configure_logging(True)

    try:
        from src.plotting.property_plotter import (
            OriginalPropertyPlotter,
            RockPhysicsPropertyPlotter,
        )
        from src.plotting.seismic_plotter import SeismicPlotter

        logger.info("=" * 80)
        logger.info("REGENERATING ALL 3D INTERACTIVE PLOTS")
        logger.info("=" * 80)

        all_files = []
        results = {
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
            files = plotter.generate_3d_plotly_visualizations()
            results["original_properties"] = files
            all_files.extend(files)
            logger.info(f"✓ Generated {len(files)} original property plots:")
            for f in files:
                logger.info(f"  - {Path(f).name}")
        except Exception as e:
            logger.error(f"Error generating original properties: {e}", exc_info=True)
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
            files_rp = plotter_rp.generate_3d_plotly_visualizations()
            results["rock_physics"] = files_rp
            all_files.extend(files_rp)
            logger.info(f"✓ Generated {len(files_rp)} rock physics plots:")
            for f in files_rp:
                logger.info(f"  - {Path(f).name}")
        except Exception as e:
            logger.error(f"Error generating rock physics: {e}", exc_info=True)
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
            files_seismic = files_seismic_time + files_seismic_depth
            results["seismic"] = [str(p) for p in files_seismic]
            all_files.extend(files_seismic)
            logger.info(f"✓ Generated {len(files_seismic)} seismic plots:")
            for f in files_seismic:
                logger.info(f"  - {Path(f).name}")
        except Exception as e:
            logger.error(f"Error generating seismic plots: {e}", exc_info=True)
            return {
                "error": str(e),
                "original_properties": results["original_properties"],
                "rock_physics": results["rock_physics"],
                "seismic": [],
                "total_count": len(all_files),
            }

        logger.info("\n" + "=" * 80)
        logger.info(f"✓ ALL 3D PLOTS REGENERATED SUCCESSFULLY!")
        logger.info("=" * 80)

        # Verify all files were created
        plot_files = sorted(Path(output_dir).glob("*_3d.html"))
        logger.info(f"\nVerifying {len(plot_files)} plot files in {output_dir}:")
        for f in plot_files:
            logger.info(f"  ✓ {f.name}")

        results["total_count"] = len(all_files)
        results["verified_count"] = len(plot_files)

        return results

    except Exception as e:
        error_msg = f"Failed to regenerate 3D plots: {e}"
        logger.error(error_msg, exc_info=True)
        return {
            "error": error_msg,
            "original_properties": [],
            "rock_physics": [],
            "seismic": [],
            "total_count": 0,
        }
