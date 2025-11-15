"""Main FaciesCorrelationAnalyzer orchestrator class.

This module provides the high-level FaciesCorrelationAnalyzer class that
coordinates the facies correlation analysis pipeline through dependency
injection and composition.

Integrated Patterns:
  - Circuit Breaker: Fault tolerance for analysis execution
  - Retry: Automatic resilience with exponential backoff
"""

import logging
from typing import Any, cast
from collections.abc import Callable

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
from src.analysis.factories.service_factory import ServiceLocator
from src.analysis.decorators import log_execution, time_operation
from src.analysis.patterns.circuit_breaker import circuit_breaker
from src.analysis.patterns.retry import retry

from src.analysis.models import (
    FaciesCorrelationConfig,
    AvoResults,
    DisplayCubesResult,
    TechniqueComparison,
    AvoStats,
    GradientCorrelationResult,
    InterfaceReflectionResult,
    FaciesDiscriminationResult,
    BoundaryAmpsResult,
)
from src.analysis.processors.validators import DomainValidator, PathValidator
from src.analysis.domain import DomainHandlerFactory
from src.core import PipelineAnalyzer, CompositeMixin

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_CACHE_DIR = ".cache"
DEFAULT_DOMAIN = Domain.DEPTH


class FaciesCorrelationAnalyzer(
    CompositeMixin, PipelineAnalyzer[FaciesCorrelationConfig, Figure]
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
        config: FaciesCorrelationConfig | None = None,
        *,
        resampler_factory: ResamplerFactory | None = None,
        select_cache_files: Callable[[str, str], str | None] | None = None,
        cache_loader: CacheLoaderProtocol | None = None,
        velocity_model_class: type[VelocityModel] | None = None,
        plotter: PlotterProtocol | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize analyzer with optional dependency injection for testing."""
        super().__init__(config=config or FaciesCorrelationConfig(), name="facies")

        # Optional helper factories
        self._resampler_factory = resampler_factory
        self._select_cache_files = select_cache_files
        self._cache_loader = cache_loader
        # Expose injected cache_loader publicly for pipeline compatibility
        self.cache_loader = cache_loader
        self._velocity_model_class = velocity_model_class or VelocityModel
        self._plotter = plotter

        # Store processor dependencies from kwargs (injected by factory or tests)
        self._injected_processors = kwargs
        # Public slot to keep the last computed AVO results when available.
        # Populated by the AnalysisPipeline at finalize stage so external
        # orchestrators (e.g., IntegratedAnalyzer) can inspect/restore results.
        self.last_avo_results: AvoResults | None = None

    def _ensure_initialized(self) -> None:
        """Ensure analyzer is initialized before use."""
        if not self.is_initialized:
            self.initialize()

    def _get_or_create(self, name: str, factory: Callable[[], Any]) -> Any:
        """Get injected processor or create default."""
        return self._injected_processors.get(name) or factory()

    def _validate_config(self) -> None:
        """Validate configuration (template method from BaseAnalyzer)."""
        # The type of self.config is guaranteed by the class definition and __init__.
        # The type of dilation_window is `int`, so we only need to validate its value.
        if self.config.dilation_window < 0:
            raise ValueError(
                f"Config 'dilation_window' must be a non-negative integer, but got {self.config.dilation_window!r}"
            )

    def _setup(self) -> None:
        """Setup processors (template method from BaseAnalyzer)."""
        # Initialize all processors
        self._boundary_detector = self._get_or_create(
            "boundary_detector", ServiceLocator.create_boundary_detector
        )
        self._cube_aligner = self._get_or_create(
            "cube_aligner", ServiceLocator.create_cube_aligner
        )
        self._boundary_amp_extractor = self._get_or_create(
            "boundary_amp_extractor",
            lambda: ServiceLocator.create_boundary_amp_extractor(
                dilation_window=self.config.dilation_window
            ),
        )
        self._gradient_calculator = self._get_or_create(
            "gradient_calculator", ServiceLocator.create_gradient_calculator
        )
        self._interface_analyzer = self._get_or_create(
            "interface_analyzer", ServiceLocator.create_interface_analyzer
        )
        self._facies_discriminator = self._get_or_create(
            "facies_discriminator", ServiceLocator.create_facies_discriminator
        )
        self._domain_handler_factory = self._get_or_create(
            "domain_handler_factory", DomainHandlerFactory
        )

        # Register all as sub-analyzers
        for name in [
            "boundary_detector",
            "cube_aligner",
            "boundary_amp_extractor",
            "gradient_calculator",
            "interface_analyzer",
        ]:
            self.add_sub_analyzer(name, getattr(self, f"_{name}"))
        self.add_sub_analyzer("facies_discriminator", self._facies_discriminator)

    @classmethod
    def from_builder(
        cls, builder_func: Callable[..., "FaciesCorrelationAnalyzer"] | None = None
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

        return builder_func()

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
        """Execute facies correlation analysis pipeline (BaseAnalyzer interface)."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict kwargs, got {type(data)}")

        # Cast to a more specific dict type to resolve Pylance warnings
        run_kwargs = cast(dict[str, Any], data)
        return cast(
            Figure,
            self.run(
                cache_dir=run_kwargs.get("cache_dir", DEFAULT_CACHE_DIR),
                domain=run_kwargs.get("domain", DEFAULT_DOMAIN),
                verbose=run_kwargs.get("verbose", False),
            ),
        )

    # NOTE: The concrete `run` implementation with decorators is defined
    # later in this class. This placeholder was removed to avoid a duplicate
    # definition that confused static analysis.

    def convert_time_to_depth(
        self,
        seismogram_time: NDArray[np.float64],
        vp_depth: NDArray[np.float64],
        grid_spec: "GridSpec",
    ) -> NDArray[np.float64]:
        """Convert time-domain seismogram to depth domain."""
        logger.info("Converting seismogram from time to depth domain...")
        resampler = self._get_resampler(grid_spec)
        plan = self._get_resample_plan(grid_spec, vp_depth)
        result = resampler.time_to_depth_cube(seismogram_time, vp_depth, plan=plan)
        return cast(NDArray[np.float64], result)

    def _get_resampler(self, grid_spec: GridSpec) -> Any:
        """Get resampler instance (injected or default)."""
        if self._resampler_factory is not None:
            return self._resampler_factory.get_resampler(grid_spec)
        from src.processing.resampling._resampler import resampler_factory

        return resampler_factory.get_resampler(grid_spec)

    def _get_resample_plan(
        self, grid_spec: GridSpec, vp_depth: NDArray[np.float64]
    ) -> Any:
        """Get resample plan from cache."""
        from src.processing.resampling._cache import get_resample_plan_cache

        return get_resample_plan_cache().get_plan(grid_spec, vp_depth)

    def detect_facies_boundaries(
        self, facies_cube: NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        """Detect facies boundaries in a 3D cube."""
        self._ensure_initialized()
        return cast(NDArray[np.bool_], self._boundary_detector(facies_cube))

    def _align_cubes(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Crop cubes to minimum common shape."""
        self._ensure_initialized()
        return cast(
            tuple[NDArray[np.float64], NDArray[np.int64]],
            self._cube_aligner(seismic_cube, facies_cube),
        )

    def extract_boundary_amplitudes(
        self,
        seismic_cube: NDArray[np.float64],
        boundaries: NDArray[np.bool_],
        window: int | None = None,
    ) -> BoundaryAmpsResult:
        """Extract amplitudes at and away from facies boundaries."""
        self._ensure_initialized()
        return cast(
            BoundaryAmpsResult,
            self._boundary_amp_extractor(seismic_cube, boundaries, window),
        )

    def calculate_gradient_correlation(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> GradientCorrelationResult:
        """Compute correlation between gradient and facies boundaries."""
        self._ensure_initialized()
        return cast(
            GradientCorrelationResult,
            self._gradient_calculator(seismic_cube, facies_cube),
        )

    def analyze_interface_reflections(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> InterfaceReflectionResult:
        """Aggregate reflection amplitudes at facies interfaces."""
        self._ensure_initialized()
        return cast(
            InterfaceReflectionResult,
            self._interface_analyzer(seismic_cube, facies_cube),
        )

    def calculate_facies_discrimination(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> FaciesDiscriminationResult:
        """Measure how well amplitudes discriminate between facies."""
        self._ensure_initialized()
        return cast(
            FaciesDiscriminationResult,
            self._facies_discriminator(seismic_cube, facies_cube),
        )

    def compare_techniques(
        self, avo_stats: Any, metric_name: str
    ) -> TechniqueComparison:
        """Return a concise AVO-only comparison for the requested metric.

        Returns a :class:`TechniqueComparison` dataclass for stronger typing
        and clearer API for downstream callers.
        """

        # Ensure correct input type
        from src.analysis.models import AvoStats as _AvoStats

        if not isinstance(avo_stats, _AvoStats):
            raise TypeError("Expected AvoStats instance for avo_stats")

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
                    extras={},
                ),
                winner="AVO",
                difference=0.0,
            )

        return TechniqueComparison(avo=avo_stats, winner="AVO", difference=0.0)

    def create_summary_plots(
        self, avo_results: AvoResults, cache_dir: str, domain: Domain = Domain.DEPTH
    ) -> Figure:
        """Create summary Figure for AVO analysis results."""
        if self._plotter is None:
            from src.plotting.facies_plotter import FaciesPlotter

            self._plotter = FaciesPlotter()
        return self._plotter.create_summary_plots(avo_results, cache_dir, domain=domain)

    def get_processor_info(self) -> dict[str, str]:
        """Get configured processor class names."""
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
        """Get a comprehensive summary of analyzer configuration and readiness."""
        self._ensure_initialized()
        return "\n".join(
            [
                "=" * 70,
                "FaciesCorrelationAnalyzer Configuration Summary",
                "=" * 70,
                self._get_status_summary(),
                self._get_config_summary(),
                self._get_dependencies_summary(),
                self._get_processors_summary(),
                "=" * 70,
            ]
        )

    def _get_status_summary(self) -> str:
        """Get status line."""
        return f"\nStatus: {'✓ Ready' if self.is_ready else '✗ Not Ready'}"

    def _get_config_summary(self) -> str:
        """Get configuration summary."""
        lines = [
            "\nConfiguration:",
            f"  Config Type: {self._config.__class__.__name__}",
        ]
        if self._config:
            lines.append(f"  Dilation Window: {self._config.dilation_window}")
        return "\n".join(lines)

    def _get_dependencies_summary(self) -> str:
        """Get dependencies summary."""
        vel_model = (
            self._velocity_model_class.__name__
            if self._velocity_model_class
            else "None"
        )
        return "\n".join(
            [
                "\nOptional Dependencies:",
                f"  Resampler Factory: {'✓ Injected' if self._resampler_factory else '○ Lazy'}",
                f"  Cache Loader: {'✓ Injected' if self._cache_loader else '○ Lazy'}",
                f"  Plotter: {'✓ Injected' if self._plotter else '○ Lazy'}",
                f"  Velocity Model: {vel_model}",
            ]
        )

    def _get_processors_summary(self) -> str:
        """Get processors summary."""
        lines = ["\nProcessors:"]
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

        # Validate inputs early
        domain = DomainValidator.validate_domain(domain, self.VALID_DOMAINS)
        cache_path = PathValidator.validate_cache_dir(cache_dir)

        # Configure logging based on verbose flag (for convenience in scripts)
        if verbose:
            self.configure_logging(verbose=True)

        # The plot_cfg contains grid_spec, but data_path and file_map
        # are properties of the dataset itself, not the plot configuration.
        plot_cfg = PlotConfig.default()
        if not hasattr(plot_cfg, "grid_spec"):
            raise AttributeError("Default PlotConfig is missing 'grid_spec'.")

        logger.info("Starting facies correlation analysis in %s domain", domain.value)
        logger.debug("Cache directory: %s", cache_path.resolve())

        # Delegate to pipeline and let it handle dataset loading internally
        pipeline = AnalysisPipeline(self)
        return pipeline.execute(str(cache_path), domain, plot_cfg)

    def prepare_display_cubes(
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
        resampler = self._get_resampler(grid_spec)
        handler = self._domain_handler_factory.get_handler(domain)
        avo_display, facies_display = handler.prepare_display_cubes(
            resampler, facies_depth, avo, grid_spec
        )
        return DisplayCubesResult(
            avo_display=avo_display, facies_display=facies_display
        )
