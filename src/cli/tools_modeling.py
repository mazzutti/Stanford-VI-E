"""Modeling and analysis CLI tools extracted from `tools.py`.

This module contains heavier modeling/analysis tool functions pulled out of
the main `tools.py` file to reduce per-file lint noise and improve
maintainability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.cli._tools_common import choose_html_path, save_npz, save_npz_with_timestamp
from src.cli.parsers import ParserFactory, tool

logger = logging.getLogger(__name__)

# Many CLI modeling helpers import heavy analysis/modeling modules inside
# command handlers to avoid import-time side effects and circular imports.
# Prefer adding a per-import suppression at the import site instead of a
# module-level disable.

# Public API exported when re-exporting from `src.cli.tools`
__all__ = [
    "analysis_rock_physics",
    "analyze_facies_correlation",
    "seismograms",
    "analysis_seismograms",
    "regenerate_seismograms",
    "export_top_seismogram_layers",
]


@tool
def analysis_rock_physics(
    _venv_python: str | None = None, cache_dir: str = ".cache", _prompt: bool = True
) -> bool:
    """Rock physics analysis pipeline with cache clearing and visualization.

    See original implementation in `src/cli/tools.py`.
    """
    # Local imports: avoid import-time side effects; keep scope explicit
    from src.analysis.common import (
        AnalysisCommon,
    )
    from src.analysis.io import HeaderPrinter

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
        # Lazy import used at call-time to avoid heavy import cost
        from src.analysis.rock_physics import (
            RockPhysicsAnalyzer,
        )

        rpa = RockPhysicsAnalyzer()
        rpa.run(
            cache_dir=cache_dir,
            generate_plots=True,
            save_npz_only=False,
            angles_list=[0, 5, 10, 15, 20, 25],
            verbose=False,
        )
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("ERROR: Rock physics pipeline failed: %s", e)
        return False

    try:
        analysis.summarize_cache_files(cache_dir=cache_dir)
    except OSError:
        pass

    return True


@tool
def analyze_facies_correlation(
    cache_dir: str = ".cache",
    domain: str = "depth",
    _no_multiangle: bool = False,
    verbose: bool = False,
) -> Any:
    """Analyze facies correlation across domains."""
    # Lazy imports for analysis components
    from src.analysis.domain.enum import (
        Domain,
    )
    from src.analysis.facies import (
        FaciesCorrelationAnalyzer,
    )

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
    """Seismogram modeling pipeline."""
    # Lazy import for pipeline analyzer
    from src.analysis.pipelines import (
        SeismogramAnalyzer,
    )

    analyzer = SeismogramAnalyzer()
    return analyzer.run(
        cache_dir=cache_dir,
        skip_cleanup=skip_cleanup,
        verbose=verbose,
    )


@tool
def analysis_seismograms() -> bool:
    """Complete seismic modeling pipeline with analysis and visualization."""
    # Local lazy imports for the complete seismogram analysis workflow
    from src.analysis.cache import (
        CacheLoader,
    )
    from src.analysis.common import (
        AnalysisCommon,
    )
    from src.analysis.domain.enum import (
        Domain,
    )
    from src.analysis.facies import (
        FaciesCorrelationAnalyzer,
    )
    from src.analysis.pipelines import (
        SeismogramAnalyzer,
    )

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
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Seismic modeling failed: %s", e)
        return False

    # Run facies correlation (depth)
    try:
        _fac = FaciesCorrelationAnalyzer()
        _fac.run(cache_dir=".cache", domain=Domain.DEPTH)
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning("Facies depth analysis failed: %s", e)

    # Run facies correlation (time)
    try:
        _fac_time = FaciesCorrelationAnalyzer()
        _fac_time.run(cache_dir=".cache", domain=Domain.TIME)
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning("Facies time analysis failed: %s", e)

    # Interactive 3D plots
    try:
        _loader = CacheLoader()
        _avo_fn = _loader.select_cache_file(".cache", "depth")
        if _avo_fn:
            logger.info("Generated 3D interactive plot for depth domain")
    except (OSError, RuntimeError) as e:
        logger.warning("3D interactive plot (depth) failed: %s", e)

    try:
        _loader = CacheLoader()
        _avo_fn = _loader.select_cache_file(".cache", "time")
        if _avo_fn:
            logger.info("Generated 3D interactive plot for time domain")
    except (OSError, RuntimeError) as e:
        logger.warning("3D interactive plot (time) failed: %s", e)

    return True


@tool
def regenerate_seismograms() -> bool:
    """Regenerate seismograms without interactive steps."""
    # Lazy imports for regeneration CLI
    from src.analysis.common import (
        AnalysisCommon,
    )
    from src.analysis.pipelines import (
        SeismogramAnalyzer,
    )

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
    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Seismic modeling failed: %s", e)
        return False

    return True


@tool
def export_top_seismogram_layers(
    cache_dir: str = ".cache",
    n_layers: int = 2,
    force_regeneration: bool = False,
    out: str | None = None,
    plot: bool = False,
    plot_out: str | None = None,
    save_to_cache: bool = False,
    matplotlib_only: bool = False,
) -> dict[str, str | tuple[int, int, int]]:
    """Extract the top N layers from the depth-domain seismogram cache.

    See original implementation in `src/cli/tools.py`.
    """
    ParserFactory.configure_logging(False)

    # Lazy imports for seismogram cache helpers
    from src.gen.seismogram import (
        DefaultCacheProvider,
        SeismogramTopLayersExtractor,
    )

    provider = DefaultCacheProvider(cache_dir=cache_dir)
    extractor = SeismogramTopLayersExtractor.from_cache_or_generate(
        cache_provider=provider,
        cache_dir=cache_dir,
        generate_if_missing=True,
        force_regeneration=force_regeneration,
    )

    top = extractor.extract_top_two_geological_layers()
    top = np.asarray(top)

    result: dict[str, str | tuple[int, int, int]] = {}
    result["shape"] = top.shape
    if out:
        p = Path(out)
        save_npz(p, top=top)
        result["saved"] = str(p)

    # Optionally save into cache directory for future reloads
    if save_to_cache:
        try:
            cache_file = save_npz_with_timestamp(
                cache_dir, "seismogram_depth_top_layers", top=top
            )
            result["cache"] = str(cache_file)
        except (OSError, ValueError, TypeError) as e:
            logger.warning("Failed to save top layers to cache: %s", e)

    # Optionally create plots. If `matplotlib_only` is True, use SeismicPlotter (Matplotlib)
    if plot:
        if matplotlib_only:
            try:
                # Lazy plotting import (matplotlib backend)
                from src.plotting.seismic_plotter import (
                    SeismicPlotter,
                )

                out_dir = Path(plot_out) if plot_out else Path("docs/images")
                out_dir.mkdir(parents=True, exist_ok=True)
                png_path = out_dir / f"seismic_top_layers_matplotlib_{n_layers}.png"
                sp = SeismicPlotter(cache_dir=cache_dir, out_dir=str(out_dir))
                sp.plot_full_stack(top, output_path=png_path, domain="depth")
                result["png"] = str(png_path)
            except (ImportError, RuntimeError, OSError) as e:
                logger.warning("Matplotlib-only plotting failed: %s", e)
        else:
            try:
                # Lazy plotting import (plotly backend)
                from src.plotting.plotly_plotter import (
                    PlotlyPlotter,
                )

                # Determine output HTML path
                html_path = choose_html_path(
                    plot_out,
                    out,
                    Path("docs/images"),
                    f"seismic_top_layers_{n_layers}_depth.html",
                )

                plotter = PlotlyPlotter()

                # Choose central slices for inline/crossline/depth (exact middle)
                ni, nj, nk = top.shape
                inline_idx = ni // 2
                crossline_idx = nj // 2
                depth_idx = nk // 2

                traces = plotter.create_3d_volume(
                    top,
                    (inline_idx, crossline_idx, depth_idx),
                )
                fig = plotter.create_figure(traces, title=f"Top {n_layers} layers")
                plotter.save_figure(fig, str(html_path))
                result["html"] = str(html_path)
            except (ImportError, RuntimeError, OSError, ValueError) as e:
                logger.warning("Failed to create interactive plot: %s", e)

    return result
