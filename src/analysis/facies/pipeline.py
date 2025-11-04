"""Analysis pipeline orchestration for facies correlation.

This module encapsulates the workflow stages of the facies correlation
analysis pipeline, separating concerns and improving testability.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure

from src.plotting.helpers.plot import PlotConfig
from src.processing.velocity import VelocityModel
from src.io import data_loader
from src.analysis.domain.enum import Domain
from src.io.data_loader import DatasetManager
from src.analysis.models import (
    CacheLoadResult,
    DisplayCubesResult,
    AvoAnalysisResult,
)
from src.analysis.processors.validators import PathValidator

if TYPE_CHECKING:
    from src.analysis.facies.analyzer import FaciesCorrelationAnalyzer

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
        dm = self._stage_load_dataset(plot_cfg)

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
        if self.analyzer._cache_loader is not None:
            avo_fn = self.analyzer._cache_loader.select_cache_file(
                cache_dir, str(domain)
            )
            if avo_fn is None:
                raise FileNotFoundError(f"No AVO cache file found in {cache_dir}")
            avo = self.analyzer._cache_loader.load_full_stack(avo_fn)
            if avo is None:
                raise ValueError(f"Failed to load AVO data from {avo_fn}")
        # fall back to using CacheLoader directly
        # Deferred import: avoid circular dependency at module load time.
        from src.analysis.cache import CacheLoader

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

    def _stage_load_dataset(self, plot_cfg: PlotConfig) -> DatasetManager:
        """Stage 2: Load dataset and velocity model.

        Parameters
        ----------
        plot_cfg
            Plot configuration with data path and file map.

        Returns
        -------
        DatasetManager
            Loaded dataset manager with all properties.
        """
        DATA_PATH, FILE_MAP, grid_spec = (
            plot_cfg.data_path,
            plot_cfg.file_map,
            plot_cfg.grid_spec,
        )
        # Construct the canonical DatasetManager using the loader.
        return data_loader.DatasetManager.from_stanfordsix(
            DATA_PATH, FILE_MAP, grid_spec
        )

    def _stage_prepare_cubes(
        self,
        avo: NDArray[np.float64],
        dm: DatasetManager,
        domain: Domain,
        plot_cfg: PlotConfig,
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
        plot_cfg
            Plot configuration with grid specs.

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
        except Exception as e:
            logger.warning(
                f"Error while aligning cache array to dataset grid: {e}. "
                "Proceeding with original array.",
                exc_info=True,
            )

        # Use DatasetManager attributes directly.
        # Cast from generic array to int64 array for type safety
        facies_depth: NDArray[np.int64] = cast(NDArray[np.int64], dm.facies)
        vm = VelocityModel.from_dataset(dm, vp_key="vp")

        # Prepare display cubes for requested domain
        return self.analyzer._prepare_display_cubes(
            vm, facies_depth, avo, domain, plot_cfg.grid_spec
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
        return self.analyzer._perform_avo_analysis(avo_display, facies_display)

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
        avo_results_obj = self.analyzer._create_results_object(results)
        return self.analyzer.create_summary_plots(
            avo_results_obj, cache_dir, domain=domain
        )
