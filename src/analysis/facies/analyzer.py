"""Main FaciesCorrelationAnalyzer orchestrator class.

This module provides the high-level FaciesCorrelationAnalyzer class that
coordinates the facies correlation analysis pipeline through dependency
injection and composition.

Integrated Patterns:
  - Circuit Breaker: Fault tolerance for analysis execution
  - Retry: Automatic resilience with exponential backoff
"""

import logging
from typing import Any, Callable, Optional, Type, Dict, cast

import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure

from src.plotting.helpers.config import PlotConfig
from src.processing.materials.velocity import VelocityModel
from src.io.grid import GridSpec

from src.analysis.types.protocols import (
    ResamplerFactory,
    CacheLoaderProtocol,
    PlotterProtocol,
)
from src.analysis.domain.enum import Domain
from src.analysis.facies.config import FaciesAnalysisConfig
from src.analysis.factories.service_factory import ServiceLocator
from src.analysis.decorators import log_execution, time_operation
from src.analysis.patterns.circuit_breaker import circuit_breaker
from src.analysis.patterns.retry import retry

from src.analysis.models import (
    FaciesCorrelationConfig,
    AvoResults,
    DisplayCubesResult,
    AvoAnalysisResult,
    TechniqueComparison,
    AvoStats,
    GradientCorrelationResult,
    InterfaceReflectionResult,
    FaciesDiscriminationResult,
    BoundaryAmpsResult,
)
from src.analysis.processors import (
    BoundaryDetector,
    CubeAligner,
    BoundaryAmplitudeExtractor,
    GradientCorrelationCalculator,
    InterfaceReflectionAnalyzer,
    FaciesDiscriminationCalculator,
)
from src.analysis.processors.validators import DomainValidator, PathValidator
from src.analysis.domain import DomainHandlerFactory
from src.core import BaseAnalyzer, PipelineAnalyzer, CompositeMixin

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_CACHE_DIR = ".cache"
DEFAULT_DOMAIN = Domain.DEPTH


class FaciesCorrelationAnalyzer(
    CompositeMixin, PipelineAnalyzer[FaciesAnalysisConfig, Figure]
):
    """Orchestrator for seismic-facies correlation analysis pipeline.

    The analyzer groups together a sequence of analysis routines (gradient
    correlation, boundary amplitude extraction, interface reflection
    aggregation and facies discrimination) and exposes a single
    ``run(...)`` method. ``run()`` executes the pipeline and returns a
    Matplotlib :class:`Figure` with analysis summary plots.

    Uses BaseAnalyzer lifecycle management for consistent error handling,
    logging, and resource cleanup. Implements PipelineAnalyzer for multi-stage
    analysis and CompositeMixin for sub-analyzer composition.

    Parameters
    ----------
    config
        Optional :class:`FaciesCorrelationConfig` instance containing
        tunable parameters for the analysis pipeline. A default is created
        when not supplied.
    resampler_factory
        Optional factory providing resampler instances for time<->depth
        operations. If omitted, the package default resampler factory is
        used when needed.
    select_cache_files
        Optional callable to select a precomputed AVO cache file.
    cache_loader
        Optional CacheLoaderProtocol implementation.
    velocity_model_class
        Optional VelocityModel-derived class.
    plotter
        Optional PlotterProtocol implementation. Lazily instantiated if omitted.
    **kwargs
        Processor dependencies injected for testing (boundary_detector,
        cube_aligner, boundary_amp_extractor, gradient_calculator,
        interface_analyzer, facies_discriminator, domain_handler_factory).

    Examples
    --------
    >>> analyzer = FaciesCorrelationAnalyzer()
    >>> with analyzer:
    ...     fig = analyzer.execute(cache_dir=".cache", domain=Domain.DEPTH)
    """

    # Class constants for validation and defaults
    VALID_DOMAINS = {Domain.DEPTH, Domain.TIME}

    def __init__(
        self,
        config: Optional[FaciesCorrelationConfig] = None,
        *,
        resampler_factory: Optional[ResamplerFactory] = None,
        select_cache_files: Optional[Callable[[str, str], Optional[str]]] = None,
        cache_loader: Optional[CacheLoaderProtocol] = None,
        velocity_model_class: Optional[Type[VelocityModel]] = None,
        plotter: Optional[PlotterProtocol] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize analyzer with optional dependency injection for testing."""
        super().__init__(config=config or FaciesCorrelationConfig(), name="facies")

        # Optional helper factories
        self._resampler_factory = resampler_factory
        self._select_cache_files = select_cache_files
        self._cache_loader = cache_loader
        self._velocity_model_class = velocity_model_class or VelocityModel
        self._plotter = plotter

        # Store processor dependencies from kwargs (injected by factory or tests)
        self._injected_processors = kwargs

    def _ensure_initialized(self) -> None:
        """Ensure analyzer is initialized before use."""
        if not self.is_initialized:
            self.initialize()

    def _validate_config(self) -> None:
        """Validate configuration (template method from BaseAnalyzer)."""
        if not self.config or not isinstance(self.config, FaciesCorrelationConfig):
            raise ValueError("Invalid FaciesCorrelationConfig")
        if not hasattr(self.config, "dilation_window"):
            raise ValueError("Config missing dilation_window")

    def _setup(self) -> None:
        """Setup processors (template method from BaseAnalyzer)."""
        # Initialize processors using ServiceLocator or injected dependencies
        self._boundary_detector = (
            self._injected_processors.get("boundary_detector")
            or ServiceLocator.create_boundary_detector()
        )
        self._cube_aligner = (
            self._injected_processors.get("cube_aligner")
            or ServiceLocator.create_cube_aligner()
        )
        self._boundary_amp_extractor = self._injected_processors.get(
            "boundary_amp_extractor"
        ) or ServiceLocator.create_boundary_amp_extractor(
            dilation_window=self.config.dilation_window
        )
        self._gradient_calculator = (
            self._injected_processors.get("gradient_calculator")
            or ServiceLocator.create_gradient_calculator()
        )
        self._interface_analyzer = (
            self._injected_processors.get("interface_analyzer")
            or ServiceLocator.create_interface_analyzer()
        )
        self._facies_discriminator = (
            self._injected_processors.get("facies_discriminator")
            or ServiceLocator.create_facies_discriminator()
        )
        self._domain_handler_factory = (
            self._injected_processors.get("domain_handler_factory")
            or DomainHandlerFactory()
        )

        # Add sub-analyzers for composite analysis
        self.add_sub_analyzer("boundary_detector", self._boundary_detector)
        self.add_sub_analyzer("cube_aligner", self._cube_aligner)
        self.add_sub_analyzer("boundary_amp_extractor", self._boundary_amp_extractor)
        self.add_sub_analyzer("gradient_calculator", self._gradient_calculator)
        self.add_sub_analyzer("interface_analyzer", self._interface_analyzer)
        self.add_sub_analyzer("facies_discriminator", self._facies_discriminator)

    @classmethod
    def from_builder(
        cls, builder_func: Optional[Callable[..., "FaciesCorrelationAnalyzer"]] = None
    ) -> "FaciesCorrelationAnalyzer":
        """Create analyzer using fluent AnalysisBuilder pattern.

        This factory method enables fluent API construction of the analyzer
        with the AnalysisBuilder pattern for cleaner initialization code.

        Parameters
        ----------
        builder_func : Callable, optional
            Builder function to customize analyzer. If omitted, uses default
            build_facies_analyzer() from builder module.

        Returns
        -------
        FaciesCorrelationAnalyzer
            Constructed analyzer instance

        Examples
        --------
        Using default builder::

            from src.analysis import build_facies_analyzer
            analyzer = FaciesCorrelationAnalyzer.from_builder()

        Using custom builder::

            from src.analysis import AnalysisBuilder
            analyzer = FaciesCorrelationAnalyzer.from_builder(
                lambda: (AnalysisBuilder()
                    .with_config(custom_config)
                    .with_dependency("plotter", custom_plotter)
                    .build())
            )

        Or inline fluent construction::

            from src.analysis import AnalysisBuilder
            analyzer = (AnalysisBuilder()
                .with_config(config)
                .with_dependency("cache_loader", loader)
                .build())
        """
        from src.analysis.builder import build_facies_analyzer

        if builder_func is None:
            return cast("FaciesCorrelationAnalyzer", build_facies_analyzer())

        return cast("FaciesCorrelationAnalyzer", builder_func())  # type: ignore[redundant-cast]

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        config_type = self.config.__class__.__name__
        return (
            f"FaciesCorrelationAnalyzer("
            f"config=<{config_type}>, "
            f"state={self.state.name}, "
            f"plotter={'injected' if self._plotter else 'lazy'})"
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"FaciesCorrelationAnalyzer(state={self.state.name})"

    @property
    def config(self) -> FaciesCorrelationConfig:
        """Get the analysis configuration."""
        return cast(FaciesCorrelationConfig, self._config)

    def analyze(self, data: Any) -> Figure:
        """Execute facies correlation analysis pipeline.

        Implements BaseAnalyzer abstract method.

        Parameters
        ----------
        data
            Input data (typically cache_dir and domain kwargs).

        Returns
        -------
        matplotlib.figure.Figure
            Summary figure with analysis results.
        """
        # data is kwargs dict passed from execute()
        if isinstance(data, dict):
            cache_dir = data.get("cache_dir", DEFAULT_CACHE_DIR)
            domain = data.get("domain", DEFAULT_DOMAIN)
            verbose = data.get("verbose", False)
            return self.run(cache_dir=cache_dir, domain=domain, verbose=verbose)
        raise TypeError(f"Expected dict kwargs, got {type(data)}")

    def run(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        domain: Domain = DEFAULT_DOMAIN,
        verbose: bool = False,
    ) -> Figure:
        """Execute facies correlation analysis pipeline (original implementation).

        This is the main execution method that coordinates the analysis pipeline.
        It's called by analyze() method which implements BaseAnalyzer interface.
        """
        logger.info(
            f"Running facies analysis (domain={domain.name}, cache={cache_dir})"
        )
        # [Original pipeline execution code continues below...]

    def convert_time_to_depth(
        self,
        seismogram_time: NDArray[np.float64],
        vp_depth: NDArray[np.float64],
        grid_spec: "GridSpec",
    ) -> NDArray[np.float64]:
        """Convert a time-domain seismogram to the depth domain.

        This helper uses the injected ``resampler_factory`` when provided
        otherwise it falls back to the package default resampler factory.

        Parameters
        ----------
        seismogram_time
            Time-domain seismogram (3D array: i, j, time/k).
        vp_depth
            P-wave velocity (depth) array aligned with the seismogram.
        grid_spec
            Grid specification used to resolve the resampling plan.

        Returns
        -------
        numpy.ndarray
            Depth-domain seismogram cube (same shape semantics as input
            but resampled to depth coordinates).
        """
        logger.info("Converting seismogram from time to depth domain...")
        if self._resampler_factory is not None:
            resampler: Any = self._resampler_factory.get_resampler(grid_spec)
        else:
            # Deferred import: avoid circular dependency at module load time.
            # The resampler module is only imported when actually needed.
            from src.processing.resampling._resampler import resampler_factory

            resampler = resampler_factory.get_resampler(grid_spec)

        # Deferred import: avoid circular dependency at module load time.
        # The resample cache module is only imported when actually needed.
        from src.processing.resampling._cache import get_resample_plan_cache

        plan = get_resample_plan_cache().get_plan(grid_spec, vp_depth)
        result = resampler.time_to_depth_cube(seismogram_time, vp_depth, plan=plan)
        return cast(NDArray[np.float64], result)

    def detect_facies_boundaries(
        self, facies_cube: NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        """Detect facies boundaries in a 3D facies cube.

        Delegates to the injected BoundaryDetector processor.

        Parameters
        ----------
        facies_cube
            Integer-valued 3D facies label cube with shape (i, j, k).

        Returns
        -------
        numpy.ndarray(dtype=bool)
            Boolean mask of the same shape (i, j, k) where ``True`` marks
            facies-boundary voxels.

        Raises
        ------
        ValueError
            If ``facies_cube`` is not a 3-dimensional array.
        """
        self._ensure_initialized()
        return cast(NDArray[np.bool_], self._boundary_detector.detect(facies_cube))

    def _align_cubes(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Crop two 3D cubes to their minimum common shape.

        Delegates to the injected CubeAligner processor.

        Parameters
        ----------
        seismic_cube, facies_cube
            The two 3D arrays to align and crop. Both must be 3-dimensional.

        Returns
        -------
        tuple
            Tuple of (seismic_cropped, facies_cropped) where each array has
            been sliced to the minimum common shape along each axis.

        Raises
        ------
        ValueError
            If either input is not a 3-dimensional array.
        """
        self._ensure_initialized()
        return cast(
            tuple[NDArray[np.float64], NDArray[np.int64]],
            self._cube_aligner.align(seismic_cube, facies_cube),
        )

    def extract_boundary_amplitudes(
        self,
        seismic_cube: NDArray[np.float64],
        boundaries: NDArray[np.bool_],
        window: Optional[int] = None,
    ) -> BoundaryAmpsResult:
        """Extract amplitudes at and away from facies boundaries.

        Delegates to the injected BoundaryAmplitudeExtractor processor.

        Parameters
        ----------
        seismic_cube
            3D seismic amplitude cube with shape (i, j, k).
        boundaries
            Boolean mask of the same shape indicating facies-boundary
            voxels.
        window
            Optional dilation radius (in iterations). When ``None`` the
            analyzer's configuration value ``self.config.dilation_window``
            is used.

        Returns
        -------
        BoundaryAmpsResult
            Named result containing arrays for amplitudes ``at_boundaries``
            and ``away_from_boundaries`` together with the boolean
            ``boundary_mask`` that was used.
        """
        self._ensure_initialized()
        return cast(
            BoundaryAmpsResult,
            self._boundary_amp_extractor.extract(seismic_cube, boundaries, window),
        )

    def calculate_gradient_correlation(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> GradientCorrelationResult:
        """Compute correlation between absolute vertical gradient and
        facies boundaries.

        Delegates to the injected GradientCorrelationCalculator processor.

        Returns
        -------
        GradientCorrelationResult
            Contains Pearson and Spearman correlations and p-values, the
            computed absolute gradient array and the boolean boundary mask
            used for the calculation.
        """
        self._ensure_initialized()
        return cast(
            GradientCorrelationResult,
            self._gradient_calculator.calculate(seismic_cube, facies_cube),
        )

    def analyze_interface_reflections(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> InterfaceReflectionResult:
        """Aggregate reflection amplitudes observed at facies interfaces.

        Delegates to the injected InterfaceReflectionAnalyzer processor.

        Returns
        -------
        InterfaceReflectionResult
            ``summary`` maps Transition -> statistics dict (or ``None`` when
            no samples) and ``interface_stats`` maps Transition -> raw
            NumPy array of observed amplitudes.
        """
        self._ensure_initialized()
        return cast(
            InterfaceReflectionResult,
            self._interface_analyzer.analyze(seismic_cube, facies_cube),
        )

    def calculate_facies_discrimination(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> FaciesDiscriminationResult:
        """Measure how well seismic amplitudes discriminate between facies types.

        Delegates to the injected FaciesDiscriminationCalculator processor.
        """
        self._ensure_initialized()
        return self._facies_discriminator.calculate(seismic_cube, facies_cube)

    def compare_techniques(
        self, avo_stats: "AvoStats", metric_name: str
    ) -> TechniqueComparison:
        """Return a concise AVO-only comparison for the requested metric.

        Returns a :class:`TechniqueComparison` dataclass for stronger typing
        and clearer API for downstream callers.
        """
        if not isinstance(avo_stats, AvoStats):
            raise TypeError("compare_techniques expects an AvoStats instance")

        if metric_name == TechniqueComparison.GRADIENT_CORRELATION:
            return TechniqueComparison(
                avo=AvoStats(
                    pearson_correlation=(
                        float(avo_stats.pearson_correlation)
                        if avo_stats.pearson_correlation is not None
                        else None
                    ),
                    spearman_correlation=(
                        float(avo_stats.spearman_correlation)
                        if avo_stats.spearman_correlation is not None
                        else None
                    ),
                ),
                winner="AVO",
                difference=0.0,
            )

        return TechniqueComparison(avo=avo_stats, winner="AVO", difference=0.0)

    def create_summary_plots(
        self, avo_results: AvoResults, cache_dir: str, domain: Domain = Domain.DEPTH
    ) -> Figure:
        """Create and return summary Figure for AVO analysis results.

        This method delegates rendering to the injected ``plotter``
        instance. If no plotter was injected the library's default
        ``FaciesPlotter`` is lazily instantiated.

        Parameters
        ----------
        avo_results
            An ``AvoResults`` dataclass produced by analysis pipeline.
        cache_dir
            Directory path containing AVO/cache artifacts used by some
            plotters to annotate or save outputs.
        domain
            A :class:`Domain` enum value (``Domain.DEPTH`` or ``Domain.TIME``)
            passed to the plotter to control axis labels and annotations.

        Returns
        -------
        matplotlib.figure.Figure
            Figure returned by the configured plotter.
        """

        # Lazily instantiate the default plotter if none was injected.
        # Deferred import: avoid circular dependency at module load time.
        # The plotter module is only imported when actually needed.
        if self._plotter is None:
            from src.plotting.facies_plotter import FaciesPlotter

            self._plotter = FaciesPlotter()

        # Delegate plotting to the plotter instance and return the Figure.
        return self._plotter.create_summary_plots(avo_results, cache_dir, domain=domain)

    def get_processor_info(self) -> Dict[str, str]:
        """Get information about configured processors (useful for testing).

        Returns
        -------
        dict
            Dictionary mapping processor names to their class names.

        Example
        -------
        >>> analyzer = FaciesCorrelationAnalyzer()
        >>> info = analyzer.get_processor_info()
        >>> print(info['boundary_detector'])
        'BoundaryDetector'
        """
        self._ensure_initialized()
        return {
            "boundary_detector": self._boundary_detector.__class__.__name__,
            "cube_aligner": self._cube_aligner.__class__.__name__,
            "boundary_amp_extractor": self._boundary_amp_extractor.__class__.__name__,
            "gradient_calculator": self._gradient_calculator.__class__.__name__,
            "interface_analyzer": self._interface_analyzer.__class__.__name__,
            "facies_discriminator": self._facies_discriminator.__class__.__name__,
            "domain_handler_factory": self._domain_handler_factory.__class__.__name__,
        }

    def get_summary(self) -> str:
        """Get a comprehensive summary of analyzer configuration and readiness.

        Returns
        -------
        str
            Multi-line formatted summary of the analyzer state.

        Example
        -------
        >>> analyzer = FaciesCorrelationAnalyzer()
        >>> print(analyzer.get_summary())
        """
        self._ensure_initialized()
        lines = []
        lines.append("=" * 70)
        lines.append("FaciesCorrelationAnalyzer Configuration Summary")
        lines.append("=" * 70)

        # Status
        lines.append(f"\nStatus: {'✓ Ready' if self.is_ready else '✗ Not Ready'}")

        # Configuration
        lines.append("\nConfiguration:")
        lines.append(f"  Config Type: {self._config.__class__.__name__}")
        if self._config:
            lines.append(f"  Dilation Window: {self._config.dilation_window}")

        # Dependencies
        lines.append("\nOptional Dependencies:")
        lines.append(
            f"  Resampler Factory: {'✓ Injected' if self._resampler_factory else '○ Lazy'}"
        )
        lines.append(
            f"  Cache Loader: {'✓ Injected' if self._cache_loader else '○ Lazy'}"
        )
        lines.append(f"  Plotter: {'✓ Injected' if self._plotter else '○ Lazy'}")
        vel_model_name = (
            self._velocity_model_class.__name__
            if self._velocity_model_class
            else "None"
        )
        lines.append(f"  Velocity Model: {vel_model_name}")

        # Processors
        lines.append("\nProcessors:")
        for proc_name, proc_type in self.get_processor_info().items():
            is_injected = getattr(self, f"_{proc_name}") is not None
            status = "✓" if is_injected else "○"
            lines.append(f"  {status} {proc_name:35} : {proc_type}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    @staticmethod
    def configure_logging(verbose: bool = False) -> None:
        """Configure logging for the analyzer.

        This is a static utility method that can be called before running
        the pipeline to set up logging verbosity. This separates logging
        configuration from the run() method, following the single-responsibility
        principle.

        Parameters
        ----------
        verbose
            If True, enables verbose DEBUG-level logging. If False, uses
            default logging configuration.

        Example
        -------
        >>> FaciesCorrelationAnalyzer.configure_logging(verbose=True)
        >>> analyzer = FaciesCorrelationAnalyzer()
        >>> fig = analyzer.run(cache_dir=".cache", domain=Domain.DEPTH)
        """
        import logging as lg

        if verbose:
            lg.basicConfig(level=lg.DEBUG, format="[%(levelname)s] %(message)s")

    @log_execution
    @time_operation("Facies correlation analysis", threshold_ms=5000)
    @circuit_breaker(
        name="facies_correlation_analysis",
        failure_threshold=3,
        recovery_timeout=60,
    )
    @retry(
        max_attempts=3,
        initial_delay=2.0,
        retryable_exceptions=[RuntimeError, OSError, IOError],
    )
    def run(
        self,
        *,
        cache_dir: str = DEFAULT_CACHE_DIR,
        domain: Domain = DEFAULT_DOMAIN,
        verbose: bool = False,
    ) -> Figure:
        """Programmatic entrypoint to run the facies-correlation pipeline.

        Parameters
        ----------
        cache_dir
            Directory containing precomputed AVO cache files (default:
            ``.cache``). Some plotters may read from or write to this
            directory for annotations or intermediate artifacts.
        domain
            ``'depth'`` or ``'time'``; controls whether AVO analysis is
            performed in depth or time domain.
        verbose
            If True, enables verbose logging for debugging. For fine-grained
            control over logging configuration, call ``configure_logging()``
            before ``run()``.

        Returns
        -------
        matplotlib.figure.Figure
            Summary figure produced by the analysis pipeline.

        Example
        -------
        For basic usage with default logging:

        >>> analyzer = FaciesCorrelationAnalyzer()
        >>> fig = analyzer.run(cache_dir=".cache", domain=Domain.DEPTH)

        For verbose logging:

        >>> FaciesCorrelationAnalyzer.configure_logging(verbose=True)
        >>> analyzer = FaciesCorrelationAnalyzer()
        >>> fig = analyzer.run(cache_dir=".cache")
        """
        from src.analysis.facies.pipeline import AnalysisPipeline

        # Configure logging based on verbose flag (for convenience in scripts)
        if verbose:
            self.configure_logging(verbose=True)

        # Validate inputs early
        domain = DomainValidator.validate_domain(domain, self.VALID_DOMAINS)
        cache_path = PathValidator.validate_cache_dir(cache_dir)

        # default_plot_config is untyped in helpers; treat as PlotConfig
        plot_cfg = PlotConfig.default()

        logger.info("Starting facies correlation analysis in %s domain", domain.value)
        logger.debug("Cache directory: %s", cache_path.resolve())

        # Delegate to pipeline
        pipeline = AnalysisPipeline(self)
        return pipeline.execute(cache_dir, domain, plot_cfg)

    # ------------------------------------------------------------------
    # Private helpers (improve organization / testability)
    # ------------------------------------------------------------------
    def _prepare_display_cubes(
        self,
        vm: VelocityModel,
        facies_depth: NDArray[np.int64],
        avo: NDArray[np.float64],
        domain: Domain,
        grid_spec: GridSpec,
    ) -> DisplayCubesResult:
        """Prepare AVO and facies cubes for display in the requested domain.

        Delegates to the appropriate domain handler strategy.
        """
        # Get resampler from factory
        if self._resampler_factory is not None:
            resampler: Any = self._resampler_factory.get_resampler(grid_spec)
        else:
            from src.processing.resampling._resampler import resampler_factory

            resampler = resampler_factory.get_resampler(grid_spec)

        handler = self._domain_handler_factory.get_handler(domain)
        avo_display, facies_display = handler.prepare_display_cubes(
            resampler, facies_depth, avo, grid_spec
        )
        return DisplayCubesResult(
            avo_display=avo_display, facies_display=facies_display
        )

    def _perform_avo_analysis(
        self, avo_display: NDArray[np.float64], facies_display: NDArray[np.int64]
    ) -> AvoAnalysisResult:
        """Execute the AVO analysis sequence and return aggregated results.

        This helper runs gradient correlation, boundary amplitude extraction
        interface reflection aggregation and facies discrimination and
        packages the results into an :class:`AvoAnalysisResult`.
        """
        avo_gradient_corr = self.calculate_gradient_correlation(
            avo_display, facies_display
        )
        avo_boundary_amps = self.extract_boundary_amplitudes(
            avo_display, avo_gradient_corr.boundaries
        )
        avo_interface_result = self.analyze_interface_reflections(
            avo_display, facies_display
        )
        facies_disc = self.calculate_facies_discrimination(avo_display, facies_display)

        return AvoAnalysisResult(
            gradient_corr=avo_gradient_corr,
            boundary_amps=avo_boundary_amps,
            interface_summary=avo_interface_result.transitions_summary,
            interface_raw=avo_interface_result.interface_stats,
            facies_disc=facies_disc,
        )

    def _create_results_object(self, avo_analysis: AvoAnalysisResult) -> AvoResults:
        """Create AvoResults object from analysis results.

        Parameters
        ----------
        avo_analysis
            AvoAnalysisResult from the pipeline.

        Returns
        -------
        AvoResults
            Formatted results object for plotting.

        Notes
        -----
        This method extracts the complex result construction logic into
        a single place for reusability and testability.
        """
        facies_disc = avo_analysis.facies_disc
        return AvoResults(
            boundary_amps=avo_analysis.boundary_amps,
            gradient_correlation=avo_analysis.gradient_corr,
            separation_matrix=facies_disc.separation_matrix,
            facies_amplitudes=facies_disc.facies_amplitudes,
            interface_stats_summary=avo_analysis.interface_summary,
        )
