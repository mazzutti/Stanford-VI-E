"""Processor registration setup for facies domain.

This module handles registration of all facies domain processors with the
central ProcessorRegistry, enabling plugin-based discovery and instantiation.

Processors are registered with metadata including domain, version, and tags
for flexible discovery and filtering.
"""

import logging

from src.analysis.processors import (
    BoundaryAmplitudeExtractor,
    BoundaryDetector,
    CubeAligner,
    FaciesDiscriminationCalculator,
    GradientCorrelationCalculator,
    InterfaceReflectionAnalyzer,
)
from src.analysis.processors.management import (
    ProcessorRegistry,
    get_default_processor_registry,
)

logger = logging.getLogger(__name__)


def get_facies_processor_registry() -> ProcessorRegistry:
    """Get or initialize the facies processor registry.

    Lazily stored as a function attribute to avoid `global` usage.

    Returns
    -------
    ProcessorRegistry
        Shared registry instance for facies processors.
    """
    inst = getattr(get_facies_processor_registry, "_registry", None)
    if inst is None:
        inst = get_default_processor_registry()
        setattr(get_facies_processor_registry, "_registry", inst)
    return inst


def _create_boundary_detector() -> BoundaryDetector:
    """Factory function for BoundaryDetector."""
    return BoundaryDetector()


def _create_cube_aligner() -> CubeAligner:
    """Factory function for CubeAligner."""
    return CubeAligner()


def _create_boundary_amp_extractor() -> BoundaryAmplitudeExtractor:
    """Factory function for BoundaryAmplitudeExtractor."""
    return BoundaryAmplitudeExtractor()


def _create_gradient_calculator() -> GradientCorrelationCalculator:
    """Factory function for GradientCorrelationCalculator."""
    return GradientCorrelationCalculator()


def _create_interface_analyzer() -> InterfaceReflectionAnalyzer:
    """Factory function for InterfaceReflectionAnalyzer."""
    return InterfaceReflectionAnalyzer()


def _create_facies_discriminator() -> FaciesDiscriminationCalculator:
    """Factory function for FaciesDiscriminationCalculator."""
    return FaciesDiscriminationCalculator()


def register_facies_processors() -> None:
    """Register all facies domain processors with the central registry.

    Registers the following processors:
    - boundary_detector: Detects facies boundaries in 3D cubes
    - cube_aligner: Aligns and crops 3D cubes to common shape
    - boundary_amp_extractor: Extracts amplitudes at boundaries
    - gradient_calculator: Calculates gradient correlations
    - interface_analyzer: Analyzes interface reflections
    - facies_discriminator: Calculates facies discrimination

    Each processor is registered with:
    - domain: "facies" (domain identifier)
    - version: "1.0" (processor version)
    - tags: Descriptive tags for filtering/discovery

    This function is idempotent and can be called multiple times safely.

    Raises
    ------
    RuntimeError
        If any processor registration fails.

    Examples
    --------
    >>> register_facies_processors()
    >>> registry = get_facies_processor_registry()
    >>> procs = registry.list_processors(domain="facies")
    >>> print(len(procs))  # Should be 6
    """
    registry = get_facies_processor_registry()

    try:
        # Register boundary detector
        registry.register(
            name="boundary_detector",
            factory=_create_boundary_detector,
            domain="facies",
            version="1.0",
            tags=["detection", "boundary", "segmentation"],
            description="Detects facies boundaries in 3D seismic cubes",
        )

        # Register cube aligner
        registry.register(
            name="cube_aligner",
            factory=_create_cube_aligner,
            domain="facies",
            version="1.0",
            tags=["alignment", "preprocessing", "cropping"],
            description="Aligns and crops 3D cubes to common shape",
        )

        # Register boundary amplitude extractor
        registry.register(
            name="boundary_amp_extractor",
            factory=_create_boundary_amp_extractor,
            domain="facies",
            version="1.0",
            tags=["amplitude", "boundary", "extraction"],
            description="Extracts amplitudes at and away from facies boundaries",
        )

        # Register gradient correlation calculator
        registry.register(
            name="gradient_calculator",
            factory=_create_gradient_calculator,
            domain="facies",
            version="1.0",
            tags=["correlation", "gradient", "analysis"],
            description="Calculates gradient correlation between seismic and facies",
        )

        # Register interface reflection analyzer
        registry.register(
            name="interface_analyzer",
            factory=_create_interface_analyzer,
            domain="facies",
            version="1.0",
            tags=["interface", "reflection", "analysis"],
            description="Analyzes interface reflection patterns and transitions",
        )

        # Register facies discriminator
        registry.register(
            name="facies_discriminator",
            factory=_create_facies_discriminator,
            domain="facies",
            version="1.0",
            tags=["discrimination", "classification", "separation"],
            description="Calculates facies discrimination and separation metrics",
        )

        logger.debug("Successfully registered all facies processors")

    except Exception as exc:
        # Catching broad exceptions here because registration may fail for many
        # reasons (user-provided processors, plugins, or unexpected runtime
        # errors). We wrap and re-raise as a RuntimeError to provide a
        # consistent failure mode to callers while still logging the original
        # exception for diagnostics.
        logger.error("Failed to register facies processors: %s", exc)
        raise RuntimeError(f"Facies processor registration failed: {exc}") from exc


def verify_facies_processors_registered() -> bool:
    """Verify that all expected facies processors are registered.

    Returns
    -------
    bool
        True if all 6 expected processors are registered, False otherwise.

    Examples
    --------
    >>> register_facies_processors()
    >>> if verify_facies_processors_registered():
    ...     print("All processors ready!")
    """
    registry = get_facies_processor_registry()
    registered = registry.list_processors(domain="facies")

    expected_processors = {
        "boundary_detector",
        "cube_aligner",
        "boundary_amp_extractor",
        "gradient_calculator",
        "interface_analyzer",
        "facies_discriminator",
    }

    registered_names = set(registered)

    if registered_names == expected_processors:
        logger.info("✓ All %s facies processors verified", len(expected_processors))
        return True
    missing = expected_processors - registered_names
    extra = registered_names - expected_processors
    if missing:
        logger.warning("Missing processors: %s", missing)
    if extra:
        logger.warning("Extra processors: %s", extra)
    return False


def list_facies_processors() -> list[str]:
    """Get list of registered facies processor names.

    Returns
    -------
    list[str]
        List of processor names registered for facies domain.

    Examples
    --------
    >>> register_facies_processors()
    >>> names = list_facies_processors()
    >>> print(names)
    ['boundary_detector', 'cube_aligner', ...]
    """
    registry = get_facies_processor_registry()
    registered = registry.list_processors(domain="facies")
    return list(registered)
