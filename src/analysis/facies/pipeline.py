"""Analysis pipeline orchestration for facies correlation.

This module encapsulates the workflow stages of the facies correlation
analysis pipeline, separating concerns and improving testability.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from src.analysis.domain.enum import Domain
from src.analysis.models import (
    AvoAnalysisResult,
    AvoResults,
    CacheLoadResult,
    DisplayCubesResult,
)
from src.analysis.processors.validators import PathValidator
from src.io.loader import DatasetManager
from src.plotting.helpers.config import PlotConfig
from src.processing.materials.velocity import VelocityModel

if TYPE_CHECKING:
    from src.analysis.facies.analyzer import FaciesCorrelationAnalyzer

# Some imports in this pipeline are intentionally deferred (e.g., cache
# loaders) to avoid heavy startup costs and circular imports. These late
# imports are intentional; disable import-order warnings so pylint focuses
# on actionable issues.

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """Orchestrator for facies correlation analysis workflow.

    Encapsulates the multi-stage analysis pipeline with clear separation
    between loading, processing, and result aggregation stages.

    Parameters
    ----------
    analyzer
        The FaciesCorrelationAnalyzer instance that provides processors
        and configuration.
    """

    # This orchestrator intentionally uses several local temporaries and
    # has a compact public surface; silence local stylistic warnings so
    # maintainers can focus on higher-risk issues.

    def __init__(self, analyzer: "FaciesCorrelationAnalyzer") -> None:
        """Initialize pipeline with analyzer dependencies."""
        self.analyzer = analyzer

    def execute(
        self,
        cache_dir: str,
        domain: Domain,
        plot_cfg: PlotConfig,
    ) -> Figure:
        """Execute the complete analysis pipeline.

        Parameters
        ----------
        cache_dir
            Directory containing AVO cache files.
        domain
            Analysis domain (DEPTH or TIME).
        plot_cfg
            Plot configuration with grid and data specifications.

        Returns
        -------
        matplotlib.figure.Figure
            Summary figure with analysis results.
        """
        # Stage 1: Load AVO cache
        cache_res = self._stage_load_cache(cache_dir, domain)
        avo = cache_res.avo

        # Stage 2: Load dataset
        # Extract necessary info from plot_cfg if it has these attributes,
        # otherwise they should be passed separately
        data_path_attr = getattr(plot_cfg, "data_path", None)
        file_map_attr = getattr(plot_cfg, "file_map", None)
        grid_spec_attr = getattr(plot_cfg, "grid_spec", None)

        if data_path_attr is None or file_map_attr is None or grid_spec_attr is None:
            # If PlotConfig doesn't have these, they need to be provided separately
            raise ValueError(
                "PlotConfig must provide data_path, file_map, and grid_spec "
                "either as attributes on plot_cfg or passed separately to execute"
            )

        # Normalize types for _stage_load_dataset
        if not isinstance(data_path_attr, Path):
            data_path = Path(data_path_attr)
        else:
            data_path = data_path_attr

        # Ensure file_map is a plain dict[str, str]
        file_map = dict(file_map_attr)

        grid_spec = grid_spec_attr

        dm = self._stage_load_dataset(data_path, file_map, grid_spec)

        # Stage 3: Align and prepare cubes
        display_res = self._stage_prepare_cubes(avo, dm, domain, plot_cfg)
        avo_display = display_res.avo_display
        facies_display = display_res.facies_display

        # Stage 4: Run analysis
        results = self._stage_run_analysis(avo_display, facies_display)

        # Stage 5: Create results and plot
        fig = self._stage_finalize(results, cache_dir, domain)

        return fig

    def _stage_load_cache(self, cache_dir: str, domain: Domain) -> CacheLoadResult:
        """Stage 1: Select and load the AVO cache file.

        The method prefers an injected ``cache_loader`` when provided. If
        not supplied it uses ``select_cache_files`` (injected or the
        helpers.select_cache_files fallback) to find a cache filename and
        then loads it using ``numpy.load``.

        Parameters
        ----------
        cache_dir
            Directory containing cache files.
        domain
            Analysis domain for cache selection.

        Returns
        -------
        CacheLoadResult
            Container with loaded AVO array and filename.
        """
        # Validate cache directory
        PathValidator.validate_cache_dir(cache_dir)

        # prefer injected CacheLoader-like helper
        cache_loader = getattr(self.analyzer, "cache_loader", None)
        if cache_loader is not None:
            avo_fn = cache_loader.select_cache_file(cache_dir, str(domain))
            if avo_fn is None:
                raise FileNotFoundError(f"No AVO cache file found in {cache_dir}")
            avo = cache_loader.load_full_stack(avo_fn)
            if avo is None:
                raise ValueError(f"Failed to load AVO data from {avo_fn}")
        # fall back to using CacheLoader directly
        # Deferred import: avoid circular dependency at module load time.
        from src.analysis.cache import (
            CacheLoader,
        )

        avo_fn = CacheLoader().select_cache_file(cache_dir, str(domain))

        if avo_fn is None:
            raise FileNotFoundError(f"No AVO cache file found in {cache_dir}")

        logger.info("Loading cache file:")
        logger.info("  Cache file: %s", Path(avo_fn).name)

        avo_cache = np.load(avo_fn)
        avo = avo_cache.get("full_stack")
        if avo is None:
            raise ValueError(f"No 'full_stack' key found in cache file: {avo_fn}")
        return CacheLoadResult(avo=avo, filename=avo_fn)

    def _stage_load_dataset(
        self,
        data_path: Path,
        file_map: dict[str, str],
        grid_spec: Any,
    ) -> DatasetManager:
        """Load dataset from disk.

        Parameters
        ----------
        data_path
            Root path to dataset files.
        file_map
            Mapping of property names to file paths.
        grid_spec
            Grid specification for the dataset.

        Returns
        -------
        DatasetManager
            Loaded dataset manager with all properties.
        """
        # Construct the canonical DatasetManager using the loader.
        return DatasetManager.from_stanfordsix(str(data_path), file_map, grid_spec)

    def _stage_prepare_cubes(
        self,
        avo: NDArray[np.float64],
        dm: DatasetManager,
        domain: Domain,
        _plot_cfg: PlotConfig,
    ) -> DisplayCubesResult:
        """Stage 3: Align cache with dataset and prepare display cubes.

        Parameters
        ----------
        avo
            Loaded AVO array.
        dm
            Loaded dataset manager.
        domain
            Analysis domain.
        _plot_cfg
            Plot configuration with grid specs. (unused in this stage)

        Returns
        -------
        DisplayCubesResult
            Aligned AVO and facies cubes for display.
        """
        logger.info("Loaded seismograms (%s domain):", domain.value)
        logger.info("  AVO shape: %s", getattr(avo, "shape", "unknown"))

        # Attempt to align the loaded AVO cache with the dataset grid
        try:
            aligned = dm.align_cache_array(avo)
            if aligned is not None:
                avo = aligned
                logger.info("Successfully aligned AVO cache to dataset grid")
            else:
                logger.warning(
                    "Loaded AVO cache could not be aligned to dataset grid; "
                    "proceeding with original array (may cause shape errors)."
                )
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logger.warning(
                "Error while aligning cache array to dataset grid: %s. "
                "Proceeding with original array.",
                e,
                exc_info=True,
            )

        # Use DatasetManager attributes directly.
        # Cast from generic array to int64 array for type safety
        facies_depth: NDArray[np.int64] = cast(NDArray[np.int64], dm.facies)
        if dm.vp is None:
            raise ValueError("vp not loaded from dataset")
        vm = VelocityModel(vp=dm.vp, grid_spec=dm.grid_spec)

        # Prepare display cubes for requested domain
        return self.analyzer.prepare_display_cubes(
            vm, facies_depth, avo, domain, dm.grid_spec
        )

    def _stage_run_analysis(
        self,
        avo_display: NDArray[np.float64],
        facies_display: NDArray[np.int64],
    ) -> AvoAnalysisResult:
        """Stage 4: Execute AVO analysis pipeline.

        Parameters
        ----------
        avo_display
            AVO array in the analysis domain.
        facies_display
            Facies array in the analysis domain.

        Returns
        -------
        AvoAnalysisResult
            Complete analysis results with all metrics.
        """
        logger.info("\nAVO Seismic-Facies Correlation\n")
        # This logic was moved from FaciesCorrelationAnalyzer._perform_avo_analysis
        # to keep the analysis sequence orchestration within the pipeline.
        analyzer = self.analyzer

        avo_gradient_corr = analyzer.calculate_gradient_correlation(
            avo_display, facies_display
        )
        avo_boundary_amps = analyzer.extract_boundary_amplitudes(
            avo_display, avo_gradient_corr.boundaries
        )
        avo_interface_result = analyzer.analyze_interface_reflections(
            avo_display, facies_display
        )
        facies_disc = analyzer.calculate_facies_discrimination(
            avo_display, facies_display
        )

        return AvoAnalysisResult(
            gradient_corr=avo_gradient_corr,
            boundary_amps=avo_boundary_amps,
            interface_summary=avo_interface_result.transitions_summary,
            interface_raw=avo_interface_result.interface_stats,
            facies_disc=facies_disc,
        )

    def _stage_finalize(
        self,
        results: AvoAnalysisResult,
        cache_dir: str,
        domain: Domain,
    ) -> Figure:
        """Stage 5: Create results object and generate plots.

        Parameters
        ----------
        results
            AVO analysis results.
        cache_dir
            Directory for cache artifacts.
        domain
            Analysis domain for plotting.

        Returns
        -------
        matplotlib.figure.Figure
            Summary figure with visualizations.
        """
        # This logic was moved from FaciesCorrelationAnalyzer._create_results_object
        # to keep the orchestration within the pipeline.
        facies_disc = results.facies_disc
        avo_results_obj = AvoResults(
            boundary_amps=results.boundary_amps,
            gradient_correlation=results.gradient_corr,
            separation_matrix=facies_disc.separation_matrix,
            facies_amplitudes=facies_disc.facies_amplitudes,
            interface_stats_summary=results.interface_summary,
        )

        # Persist the AVO results on the analyzer for external inspection
        # (e.g., undo/redo support in IntegratedAnalyzer).
        try:
            self.analyzer.last_avo_results = avo_results_obj
        except (AttributeError, RuntimeError, TypeError):
            # Non-fatal: analyzer may not support attribute assignment in
            # some injection/testing scenarios; continue regardless.
            logger.debug(
                "Analyzer does not accept last_avo_results assignment", exc_info=True
            )

        # Delegate plotting to the analyzer's plotter
        return self.analyzer.create_summary_plots(
            avo_results_obj, cache_dir, domain=domain
        )
