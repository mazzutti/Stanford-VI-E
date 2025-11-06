"""Builder pattern implementation for FaciesCorrelationAnalyzer configuration.

This module provides the AnalyzerBuilder class for fluent, step-by-step
construction of FaciesCorrelationAnalyzer with full dependency injection.

Features:
- Lazy initialization of processors
- Type-safe processor registry
- Context manager support for transient configuration
- Builder state freezing for immutability
- State snapshots for serialization
"""

from __future__ import annotations

import logging
from copy import copy as shallow_copy
from typing import (
    Optional,
    Callable,
    Type,
    TYPE_CHECKING,
    Dict,
    Tuple,
    cast,
    Generator,
)
from contextlib import contextmanager

from src.processing.materials.velocity import VelocityModel
from src.analysis.types.base import (
    ResamplerFactory,
    CacheLoaderProtocol,
    PlotterProtocol,
)
from src.analysis.models import FaciesCorrelationConfig
from src.analysis.processors import (
    BoundaryDetector,
    CubeAligner,
    BoundaryAmplitudeExtractor,
    GradientCorrelationCalculator,
    InterfaceReflectionAnalyzer,
    FaciesDiscriminationCalculator,
)
from src.analysis.domain import DomainHandlerFactory
from src.analysis.factories.validators import TypeValidator
from src.analysis.exceptions import BuilderValidationError, BuilderFrozenError

if TYPE_CHECKING:
    from src.analysis.facies import FaciesCorrelationAnalyzer

logger = logging.getLogger(__name__)


class AnalyzerBuilder:
    """Builder for FaciesCorrelationAnalyzer with fluent configuration."""

    # Static factory methods for processors
    @staticmethod
    def _create_boundary_detector() -> BoundaryDetector:
        """Create a BoundaryDetector instance."""
        return BoundaryDetector()

    @staticmethod
    def _create_cube_aligner() -> CubeAligner:
        """Create a CubeAligner instance."""
        return CubeAligner()

    @staticmethod
    def _create_gradient_calculator() -> GradientCorrelationCalculator:
        """Create a GradientCorrelationCalculator instance."""
        return GradientCorrelationCalculator()

    @staticmethod
    def _create_interface_analyzer() -> InterfaceReflectionAnalyzer:
        """Create an InterfaceReflectionAnalyzer instance."""
        return InterfaceReflectionAnalyzer()

    @staticmethod
    def _create_facies_discriminator() -> FaciesDiscriminationCalculator:
        """Create a FaciesDiscriminationCalculator instance."""
        return FaciesDiscriminationCalculator()

    @staticmethod
    def _create_domain_handler_factory() -> DomainHandlerFactory:
        """Create a DomainHandlerFactory instance."""
        return DomainHandlerFactory()

    @staticmethod
    def _create_boundary_amp_extractor(
        config: FaciesCorrelationConfig,
    ) -> BoundaryAmplitudeExtractor:
        """Create a BoundaryAmplitudeExtractor instance."""
        return BoundaryAmplitudeExtractor(dilation_window=config.dilation_window)

    def __init__(self) -> None:
        """Initialize a new builder with no components initialized."""
        # Configuration dependencies
        self._resampler_factory: Optional[ResamplerFactory] = None
        self._select_cache_files: Optional[Callable[[str, str], Optional[str]]] = None
        self._cache_loader: Optional[CacheLoaderProtocol] = None
        self._velocity_model_class: Optional[Type[VelocityModel]] = None
        self._plotter: Optional[PlotterProtocol] = None
        self._config: Optional[FaciesCorrelationConfig] = None

        # Processor attributes (lazy-initialized)
        self._boundary_detector: Optional[BoundaryDetector] = None
        self._cube_aligner: Optional[CubeAligner] = None
        self._boundary_amp_extractor: Optional[BoundaryAmplitudeExtractor] = None
        self._gradient_calculator: Optional[GradientCorrelationCalculator] = None
        self._interface_analyzer: Optional[InterfaceReflectionAnalyzer] = None
        self._facies_discriminator: Optional[FaciesDiscriminationCalculator] = None
        self._domain_handler_factory: Optional[DomainHandlerFactory] = None

        # Processors stored in type-safe registry (lazy-initialized)
        self._processor_registry: Dict[
            str, Tuple[Type[object], Callable[..., object], Optional[object]]
        ] = {
            "boundary_detector": (
                BoundaryDetector,
                self._create_boundary_detector,
                None,
            ),
            "cube_aligner": (CubeAligner, self._create_cube_aligner, None),
            "gradient_calculator": (
                GradientCorrelationCalculator,
                self._create_gradient_calculator,
                None,
            ),
            "interface_analyzer": (
                InterfaceReflectionAnalyzer,
                self._create_interface_analyzer,
                None,
            ),
            "facies_discriminator": (
                FaciesDiscriminationCalculator,
                self._create_facies_discriminator,
                None,
            ),
            "domain_handler_factory": (
                DomainHandlerFactory,
                self._create_domain_handler_factory,
                None,
            ),
            "boundary_amp_extractor": (
                BoundaryAmplitudeExtractor,
                self._create_boundary_amp_extractor,
                None,
            ),
        }
        self._is_frozen: bool = False

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent modification after freezing.

        Raises
        ------
        BuilderFrozenError
            If builder is frozen.
        """
        if hasattr(self, "_is_frozen") and self._is_frozen:
            raise BuilderFrozenError(
                "Cannot modify frozen builder. Call unfreeze() first."
            )
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        """Return detailed string representation of builder state.

        Returns
        -------
        str
            String representation showing configuration and processor state.
        """
        config_items = []
        if self._resampler_factory is not None:
            config_items.append("resampler_factory=<set>")
        if self._select_cache_files is not None:
            config_items.append("cache_file_selector=<callable>")
        if self._cache_loader is not None:
            config_items.append("cache_loader=<set>")
        if self._velocity_model_class is not None:
            config_items.append(f"velocity_model={self._velocity_model_class.__name__}")
        if self._plotter is not None:
            config_items.append("plotter=<set>")
        if self._config is not None:
            config_items.append("config=<FaciesCorrelationConfig>")

        custom_processors = [
            k for k, (_, _, v) in self._processor_registry.items() if v is not None
        ]
        if custom_processors:
            config_items.append(f"custom_processors={custom_processors}")

        frozen_status = "frozen" if self._is_frozen else "mutable"

        config_str = ", ".join(config_items) if config_items else "empty"
        return f"AnalyzerBuilder({config_str}) [{frozen_status}]"

    def __eq__(self, other: object) -> bool:
        """Compare builder state with another builder.

        Two builders are equal if they have the same configuration and
        processor state.

        Parameters
        ----------
        other
            Object to compare with.

        Returns
        -------
        bool
            True if builders have equivalent state.
        """
        if not isinstance(other, AnalyzerBuilder):
            return False

        return (
            self._resampler_factory == other._resampler_factory
            and self._select_cache_files == other._select_cache_files
            and self._cache_loader == other._cache_loader
            and self._velocity_model_class == other._velocity_model_class
            and self._plotter == other._plotter
            and self._config == other._config
            and self.is_frozen() == other.is_frozen()
        )

    def __hash__(self) -> int:
        """Return hash of builder configuration for use in sets/dicts.

        Note: Builder state includes mutable components, so hash may
        only be stable for frozen builders.

        Returns
        -------
        int
            Hash of immutable builder state.
        """
        return hash(
            (
                id(self._resampler_factory),
                id(self._select_cache_files),
                id(self._cache_loader),
                id(self._velocity_model_class),
                id(self._plotter),
                id(self._config),
                self.is_frozen(),
            )
        )

    def _set_dependency(self, name: str, value: object) -> "AnalyzerBuilder":
        """Generic setter for reducing boilerplate with frozen check and type validation.

        Performs type validation using a centralized helper that properly handles
        both regular types and Protocol types with duck-typing fallback.

        Parameters
        ----------
        name
            The attribute name (without leading underscore).
        value
            The value to set.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.

        Raises
        ------
        BuilderFrozenError
            If builder is frozen.
        TypeError
            If value type doesn't match expected type.
        """
        # Type validation mapping
        type_map = {
            "resampler_factory": ResamplerFactory,
            "select_cache_files": Callable,
            "cache_loader": CacheLoaderProtocol,
            "velocity_model_class": type,
            "plotter": PlotterProtocol,
            "config": FaciesCorrelationConfig,
            "boundary_detector": BoundaryDetector,
            "cube_aligner": CubeAligner,
            "boundary_amp_extractor": BoundaryAmplitudeExtractor,
            "gradient_calculator": GradientCorrelationCalculator,
            "interface_analyzer": InterfaceReflectionAnalyzer,
            "facies_discriminator": FaciesDiscriminationCalculator,
            "domain_handler_factory": DomainHandlerFactory,
        }

        # Perform type validation using centralized helper
        if name in type_map:
            expected_type = type_map[name]
            TypeValidator.validate(value, expected_type, name)

        old_value = getattr(self, f"_{name}", None)
        setattr(self, f"_{name}", value)

        # Log changes
        if old_value != value:
            old_type = type(old_value).__name__ if old_value else "None"
            new_type = type(value).__name__ if value else "None"
            logger.debug(f"Updated {name}: {old_type} -> {new_type}")

        return self

    def freeze(self) -> "AnalyzerBuilder":
        """Freeze builder to prevent accidental modifications.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        object.__setattr__(self, "_is_frozen", True)
        logger.debug("Builder frozen")
        return self

    def unfreeze(self) -> "AnalyzerBuilder":
        """Unfreeze builder to allow modifications.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        object.__setattr__(self, "_is_frozen", False)
        logger.debug("Builder unfrozen")
        return self

    def is_frozen(self) -> bool:
        """Check if builder is frozen.

        Returns
        -------
        bool
            True if frozen, False otherwise.
        """
        return self._is_frozen

    @staticmethod
    def set_log_level(level: int) -> None:
        """Configure logging verbosity for builder initialization.

        Parameters
        ----------
        level
            Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING).

        Example
        -------
        >>> import logging
        >>> AnalyzerBuilder.set_log_level(logging.DEBUG)  # Verbose logging
        >>> analyzer = AnalyzerFactory.create_default()
        """
        logger.setLevel(level)

    def with_resampler_factory(self, factory: ResamplerFactory) -> "AnalyzerBuilder":
        """Configure a resampler factory.

        Parameters
        ----------
        factory
            ResamplerFactory instance for time<->depth operations.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        return self._set_dependency("resampler_factory", factory)

    def with_cache_file_selector(
        self, selector: Callable[[str, str], Optional[str]]
    ) -> "AnalyzerBuilder":
        """Configure a cache file selector function.

        Parameters
        ----------
        selector
            Callable(cache_dir, domain) -> Optional[cache_filename].

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        return self._set_dependency("select_cache_files", selector)

    def with_cache_loader(self, loader: CacheLoaderProtocol) -> "AnalyzerBuilder":
        """Configure a cache loader.

        Parameters
        ----------
        loader
            Object implementing CacheLoaderProtocol.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        return self._set_dependency("cache_loader", loader)

    def with_velocity_model_class(
        self, model_class: Type[VelocityModel]
    ) -> "AnalyzerBuilder":
        """Configure a velocity model class.

        Parameters
        ----------
        model_class
            Class deriving from VelocityModel.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        return self._set_dependency("velocity_model_class", model_class)

    def with_plotter(self, plotter: PlotterProtocol) -> "AnalyzerBuilder":
        """Configure a plotter.

        Parameters
        ----------
        plotter
            Object implementing PlotterProtocol.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        return self._set_dependency("plotter", plotter)

    def with_config(self, config: FaciesCorrelationConfig) -> "AnalyzerBuilder":
        """Configure analysis parameters.

        Parameters
        ----------
        config
            FaciesCorrelationConfig instance.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.

        Example
        -------
        >>> config = FaciesCorrelationConfig()
        >>> analyzer = (AnalyzerFactory.builder()
        ...     .with_config(config)
        ...     .build())
        """
        return self._set_dependency("config", config)

    def with_boundary_detector(self, detector: BoundaryDetector) -> "AnalyzerBuilder":
        """Configure a custom boundary detector processor.

        Parameters
        ----------
        detector
            BoundaryDetector instance.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        return self._set_dependency("boundary_detector", detector)

    def with_cube_aligner(self, aligner: CubeAligner) -> "AnalyzerBuilder":
        """Configure a custom cube aligner processor.

        Parameters
        ----------
        aligner
            CubeAligner instance.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        return self._set_dependency("cube_aligner", aligner)

    def with_boundary_amp_extractor(
        self, extractor: BoundaryAmplitudeExtractor
    ) -> "AnalyzerBuilder":
        """Configure a custom boundary amplitude extractor processor.

        Parameters
        ----------
        extractor
            BoundaryAmplitudeExtractor instance.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        return self._set_dependency("boundary_amp_extractor", extractor)

    def with_gradient_calculator(
        self, calculator: GradientCorrelationCalculator
    ) -> "AnalyzerBuilder":
        """Configure a custom gradient correlation calculator processor.

        Parameters
        ----------
        calculator
            GradientCorrelationCalculator instance.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        return self._set_dependency("gradient_calculator", calculator)

    def with_interface_analyzer(
        self, analyzer: InterfaceReflectionAnalyzer
    ) -> "AnalyzerBuilder":
        """Configure a custom interface reflection analyzer processor.

        Parameters
        ----------
        analyzer
            InterfaceReflectionAnalyzer instance.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        return self._set_dependency("interface_analyzer", analyzer)

    def with_facies_discriminator(
        self, discriminator: FaciesDiscriminationCalculator
    ) -> "AnalyzerBuilder":
        """Configure a custom facies discrimination calculator processor.

        Parameters
        ----------
        discriminator
            FaciesDiscriminationCalculator instance.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        return self._set_dependency("facies_discriminator", discriminator)

    def with_domain_handler_factory(
        self, factory: DomainHandlerFactory
    ) -> "AnalyzerBuilder":
        """Configure a custom domain handler factory.

        Parameters
        ----------
        factory
            DomainHandlerFactory instance.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        return self._set_dependency("domain_handler_factory", factory)

    def with_processors(self, **processors: object) -> "AnalyzerBuilder":
        """Configure multiple processors at once (batch configuration).

        Supports flexible naming: processor names can be specified with or
        without the "with_" prefix and with or without trailing "_processor".

        Parameters
        ----------
        **processors
            Keyword arguments mapping processor names to instances.
            Examples:
            - boundary_detector=BoundaryDetector()
            - cube_aligner=CubeAligner()
            - gradient_calculator=GradientCorrelationCalculator()

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.

        Raises
        ------
        ValueError
            If an unknown processor name is provided.

        Example
        -------
        >>> builder = AnalyzerFactory.builder()
        >>> analyzer = (builder
        ...     .with_processors(
        ...         boundary_detector=custom_detector,
        ...         cube_aligner=custom_aligner,
        ...         gradient_calculator=custom_calc
        ...     )
        ...     .build())
        """
        valid_processors = set(self._processor_registry.keys())

        for name, instance in processors.items():
            # Normalize processor name
            normalized_name = name.replace("_processor", "")

            if normalized_name not in valid_processors:
                raise ValueError(
                    f"Unknown processor '{name}'. Valid processors: {valid_processors}"
                )

            self._set_dependency(normalized_name, instance)
            logger.debug(f"Batch-configured {normalized_name}")

        return self

    @contextmanager
    def transient_config(
        self, **config_overrides: object
    ) -> Generator["AnalyzerBuilder", None, None]:
        """Context manager for temporary configuration overrides.

        Useful for testing or temporary customization without permanently
        modifying the builder state.

        Parameters
        ----------
        **config_overrides
            Temporary configuration to apply during the context.

        Yields
        ------
        AnalyzerBuilder
            Self with temporary configuration applied.

        Example
        -------
        >>> builder = AnalyzerBuilder()
        >>> with builder.transient_config(config=test_config) as temp_builder:
        ...     analyzer = temp_builder.build()
        """
        # Save current state
        original_state = shallow_copy(self)

        # Apply temporary overrides
        for key, value in config_overrides.items():
            if key.startswith("with_"):
                key = key[5:]
            self._set_dependency(key, value)

        logger.debug(f"Applied transient config: {list(config_overrides.keys())}")

        try:
            yield self
        finally:
            # Restore original state
            for attr_name in (
                "_resampler_factory",
                "_select_cache_files",
                "_cache_loader",
                "_velocity_model_class",
                "_plotter",
                "_config",
            ):
                object.__setattr__(self, attr_name, getattr(original_state, attr_name))
            logger.debug("Restored builder state after transient config")

    def _initialize_processors(self) -> None:
        """Initialize all processors using factory functions with lazy initialization.

        Uses type-safe registry to manage processor initialization with factory
        functions for consistent creation patterns.
        """
        # Initialize boundary amplitude extractor (needs config)
        if self._boundary_amp_extractor is None:
            config = self._config or FaciesCorrelationConfig()
            self._boundary_amp_extractor = self._create_boundary_amp_extractor(config)
            logger.debug("Lazy-initialized BoundaryAmplitudeExtractor")

        # Initialize other processors using factory functions from registry
        for processor_name, (
            proc_type,
            factory_func,
            _,
        ) in self._processor_registry.items():
            if processor_name == "boundary_amp_extractor":
                continue  # Already handled above

            # Get current processor value
            current_value = getattr(self, f"_{processor_name}", None)
            if current_value is None:
                # Use factory function for lazy initialization
                new_instance = factory_func()
                setattr(self, f"_{processor_name}", new_instance)
                logger.debug(f"Lazy-initialized {proc_type.__name__}")

    def _validate(self) -> None:
        """Validate builder state before building with detailed error reporting.

        Performs comprehensive validation of builder configuration with helpful
        error messages and suggestions for fixing issues.

        Raises
        ------
        BuilderValidationError
            If required dependencies are missing or invalid with details.
        """
        missing_critical: list[str] = []
        warnings: list[str] = []

        # Validation rules
        # Example: if self._config is None:
        #     missing_critical.append("config")

        # Check for common incomplete configurations
        if self._config is None:
            warnings.append(
                "No config provided - using default FaciesCorrelationConfig"
            )

        if self._plotter is None:
            warnings.append("No plotter configured - will use default visualization")

        if missing_critical:
            error_msg = (
                f"Cannot build analyzer. Missing required dependencies: {missing_critical}\n"
                f"Suggestions:\n"
            )
            for dep in missing_critical:
                error_msg += f"  - Add .with_{dep}(...) to your builder chain\n"
            error_msg += "\nCall builder.debug_info() for detailed configuration status"
            raise BuilderValidationError(error_msg, missing_critical)

        # Log warnings
        for warning in warnings:
            logger.warning(f"Builder validation warning: {warning}")

        logger.debug("Builder validation passed")

    def reset(self) -> "AnalyzerBuilder":
        """Reset builder to initial state.

        Clears all configured dependencies, allowing the builder to be
        reused for creating a new analyzer from scratch.

        Returns
        -------
        AnalyzerBuilder
            Self for method chaining.
        """
        # Reinitialize all attributes
        self._resampler_factory = None
        self._select_cache_files = None
        self._cache_loader = None
        self._velocity_model_class = None
        self._plotter = None
        self._config = None
        self._boundary_detector = None
        self._cube_aligner = None
        self._boundary_amp_extractor = None
        self._gradient_calculator = None
        self._interface_analyzer = None
        self._facies_discriminator = None
        self._domain_handler_factory = None
        self._processor_registry = {
            "boundary_detector": (
                BoundaryDetector,
                self._create_boundary_detector,
                None,
            ),
            "cube_aligner": (CubeAligner, self._create_cube_aligner, None),
            "gradient_calculator": (
                GradientCorrelationCalculator,
                self._create_gradient_calculator,
                None,
            ),
            "interface_analyzer": (
                InterfaceReflectionAnalyzer,
                self._create_interface_analyzer,
                None,
            ),
            "facies_discriminator": (
                FaciesDiscriminationCalculator,
                self._create_facies_discriminator,
                None,
            ),
            "domain_handler_factory": (
                DomainHandlerFactory,
                self._create_domain_handler_factory,
                None,
            ),
            "boundary_amp_extractor": (
                BoundaryAmplitudeExtractor,
                self._create_boundary_amp_extractor,
                None,
            ),
        }
        object.__setattr__(self, "_is_frozen", False)
        logger.debug("Builder reset to initial state")
        return self

    def copy(self) -> "AnalyzerBuilder":
        """Create a shallow copy of the current builder state.

        Useful for creating variants of a configured builder without
        affecting the original builder's state.

        Returns
        -------
        AnalyzerBuilder
            A new builder with the same configuration.
        """
        new_builder = shallow_copy(self)
        logger.debug("Builder copied")
        return new_builder

    @classmethod
    def from_existing_builder(cls, existing: "AnalyzerBuilder") -> "AnalyzerBuilder":
        """Create a new builder as a copy of an existing builder.

        Useful for cloning configuration from another builder instance.

        Parameters
        ----------
        existing
            Existing builder to clone from.

        Returns
        -------
        AnalyzerBuilder
            A new builder with cloned configuration.

        Example
        -------
        >>> base_builder = AnalyzerFactory.builder().with_config(config)
        >>> variant_builder = AnalyzerBuilder.from_existing_builder(base_builder)
        >>> variant_builder.with_plotter(custom_plotter).build()
        """
        cloned = existing.copy()
        logger.info("Created new builder from existing builder")
        return cloned

    @classmethod
    def with_state_snapshot(cls, snapshot: Dict[str, object]) -> "AnalyzerBuilder":
        """Create a builder from a previously saved state snapshot.

        Allows restoring builder configuration from a saved state.

        Parameters
        ----------
        snapshot
            Dictionary containing builder state (from state_snapshot()).

        Returns
        -------
        AnalyzerBuilder
            Builder with restored state.

        Raises
        ------
        ValueError
            If snapshot format is invalid.
        """
        builder = cls()
        if not isinstance(snapshot, dict):
            raise ValueError("Snapshot must be a dictionary")

        # Restore configuration
        if "resampler_factory" in snapshot:
            builder._resampler_factory = cast(
                Optional[ResamplerFactory], snapshot["resampler_factory"]
            )
        if "select_cache_files" in snapshot:
            builder._select_cache_files = cast(
                Optional[Callable[[str, str], Optional[str]]],
                snapshot["select_cache_files"],
            )
        if "cache_loader" in snapshot:
            builder._cache_loader = cast(
                Optional[CacheLoaderProtocol], snapshot["cache_loader"]
            )
        if "velocity_model_class" in snapshot:
            builder._velocity_model_class = cast(
                Optional[Type[VelocityModel]], snapshot["velocity_model_class"]
            )
        if "plotter" in snapshot:
            builder._plotter = cast(Optional[PlotterProtocol], snapshot["plotter"])
        if "config" in snapshot:
            builder._config = cast(
                Optional[FaciesCorrelationConfig], snapshot["config"]
            )

        logger.info("Created builder from state snapshot")
        return builder

    def state_snapshot(self) -> Dict[str, object]:
        """Save builder state for later restoration.

        Returns
        -------
        Dict[str, object]
            Snapshot of current builder state.
        """
        return {
            "resampler_factory": self._resampler_factory,
            "select_cache_files": self._select_cache_files,
            "cache_loader": self._cache_loader,
            "velocity_model_class": self._velocity_model_class,
            "plotter": self._plotter,
            "config": self._config,
        }

    def configured_processor_count(self) -> int:
        """Count how many processors have been explicitly configured.

        Returns
        -------
        int
            Number of processors that are not None.
        """
        count = 0
        for processor_name, (_, _, _) in self._processor_registry.items():
            if getattr(self, f"_{processor_name}", None) is not None:
                count += 1
        return count

    def debug_info(self) -> str:
        """Get detailed debugging information about builder configuration.

        Returns
        -------
        str
            Formatted string showing configuration state, missing dependencies,
            and processor status.

        Example
        -------
        >>> builder = AnalyzerFactory.builder().with_config(config)
        >>> print(builder.debug_info())
        """
        lines = []
        lines.append("=" * 70)
        lines.append("ANALYZER BUILDER DEBUG INFO")
        lines.append("=" * 70)

        # Builder state
        lines.append("\nBuilder State:")
        lines.append(f"  Frozen: {self.is_frozen()}")
        lines.append(f"  Configured Processors: {self.configured_processor_count()}/7")

        # Configuration dependencies
        lines.append("\nConfiguration Dependencies:")
        config_deps = {
            "resampler_factory": self._resampler_factory,
            "cache_file_selector": self._select_cache_files,
            "cache_loader": self._cache_loader,
            "velocity_model": self._velocity_model_class,
            "plotter": self._plotter,
            "config": self._config,
        }
        for name, value in config_deps.items():
            status = "✓" if value is not None else "✗"
            type_str = type(value).__name__ if value is not None else "None"
            lines.append(f"  {status} {name:25} : {type_str}")

        # Processors
        lines.append("\nProcessors:")
        for proc_name, (proc_type, _, _) in sorted(self._processor_registry.items()):
            value = getattr(self, f"_{proc_name}", None)
            status = "✓" if value is not None else "○"
            config_status = (
                "Configured" if value is not None else "Will be lazy-initialized"
            )
            lines.append(f"  {status} {proc_name:30} : {config_status}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def build(self) -> "FaciesCorrelationAnalyzer":
        """Build and return the configured analyzer.

        This method performs:
        1. Lazy initialization of all processors
        2. Validation of dependencies
        3. Construction of FaciesCorrelationAnalyzer
        4. Logging of the build operation

        Returns
        -------
        FaciesCorrelationAnalyzer
            Fully configured analyzer instance.

        Raises
        ------
        BuilderValidationError
            If validation fails before building.
        """
        was_frozen = self.is_frozen()
        if was_frozen:
            logger.warning("Unfreezing builder for processor initialization")
            self.unfreeze()

        self._validate()
        self._initialize_processors()

        # Re-freeze if it was frozen before
        if was_frozen:
            self.freeze()

        # Import here to avoid circular dependency issues
        from src.analysis.facies import FaciesCorrelationAnalyzer

        analyzer = FaciesCorrelationAnalyzer(
            resampler_factory=self._resampler_factory,
            select_cache_files=self._select_cache_files,
            cache_loader=self._cache_loader,
            velocity_model_class=self._velocity_model_class,
            plotter=self._plotter,
            config=self._config,
            # Inject processors
            boundary_detector=self._boundary_detector,
            cube_aligner=self._cube_aligner,
            boundary_amp_extractor=self._boundary_amp_extractor,
            gradient_calculator=self._gradient_calculator,
            interface_analyzer=self._interface_analyzer,
            facies_discriminator=self._facies_discriminator,
            domain_handler_factory=self._domain_handler_factory,
        )

        logger.info(f"Built FaciesCorrelationAnalyzer: {repr(self)}")
        return analyzer
