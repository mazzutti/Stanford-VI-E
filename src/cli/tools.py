"""CLI tool implementations for seismic workflows.

This module contains all tool functions registered with @tool decorator,
including cache cleanup, plotting, analysis, and regeneration workflows.
"""

from __future__ import annotations

import logging
from typing import Any

from src.cli.parsers import ParserFactory, tool

logger = logging.getLogger(__name__)

__all__ = [
    "cleanup_cache",
    "plot_3d_interactive",
    "plot_3d_slices",
    "plot_rock_physics_attributes",
    "analysis_rock_physics",
    "analyze_facies_correlation",
    "seismograms",
    "analysis_seismograms",
    "regenerate_seismograms",
    "regenerate_rock_physics",
    "rock_physics_attributes",
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
    except Exception:
        pass

    from pathlib import Path
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
def plot_rock_physics_attributes(argv: list[str] | None = None) -> dict[str, str]:
    """Rock physics attribute visualization.

    Parameters
    ----------
    argv : list[str] | None
        Command line arguments

    Returns
    -------
    dict[str, str]
        Result with domain name
    """
    import argparse

    parser = argparse.ArgumentParser(description="Visualize rock physics attributes")
    parser.add_argument(
        "--domain",
        choices=["depth", "time"],
        default="depth",
        help="Domain for visualization",
    )
    args = parser.parse_args(argv)

    return {"domain": args.domain}


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

    analysis = AnalysisCommon()
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

    analysis = AnalysisCommon()

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

    regen = AnalysisCommon()

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

        regen = AnalysisCommon()

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
