"""Processors package for facies correlation analysis.

This package provides specialized processor classes for different aspects
of the analysis pipeline, following the Single Responsibility Principle.

Note: This package is a transitional refactoring of the monolithic processors.py.
During this transition, imports are re-exported from the original module.
Individual modules will be created gradually.

Public API
----------
Main processor classes:
- BoundaryDetector: Detects facies boundaries in 3D cubes
- CubeAligner: Aligns and crops multiple 3D cubes to common shape
- BoundaryAmplitudeExtractor: Extracts amplitudes at/away from boundaries
- GradientCorrelationCalculator: Correlates seismic gradients with boundaries
- InterfaceReflectionAnalyzer: Analyzes reflection amplitudes at interfaces
- FaciesDiscriminationCalculator: Measures facies discrimination capability

Configuration classes:
- ProcessorConfig: Immutable configuration for processor operations
- BoundaryComputationConfig: Boundary-specific configuration
- NeighborDirection: Enum for neighbor directions in boundary detection

Utilities and validators:
- ProcessorUtils: Common operations for processors
- ArrayValidator: Validation logic for array inputs
- ValidationResult: Structured validation results

Exceptions:
- ProcessorError: Base exception for processor operations
- ValidationError: Raised when array validation fails
- CorrelationError: Raised when correlation computation fails
- ReshapeError: Raised when array reshaping fails

Example
-------
>>> import numpy as np
from numpy.typing import NDArray
>>> from src.analysis.processors import BoundaryDetector, CubeAligner
>>> facies_cube = np.random.randint(0, 5, (10, 10, 20))
>>> detector = BoundaryDetector()
>>> boundaries = detector(facies_cube)  # Returns bool mask via BaseProcessor
>>> aligner = CubeAligner()
>>> seismic = np.random.randn(10, 10, 20)
>>> seismic_aligned, facies_aligned = aligner.align(seismic, facies_cube)
"""

from src.core.processors import BaseProcessor, Processor  # noqa: F401

from .amplitude import BoundaryAmplitudeExtractor  # noqa: F401

# Phase 3 modules (processor implementations)
from .boundary import NeighborDirection  # noqa: F401
from .boundary import BoundaryDetector, CubeAligner  # noqa: F401

# Phase 2 modules (utilities and helpers)
from .decorators import ProcessorDecorators  # noqa: F401
from .discrimination import FaciesDiscriminationCalculator  # noqa: F401
from .exceptions import ProcessorError  # noqa: F401
from .exceptions import CorrelationError, ReshapeError, ValidationError
from .gradient import GradientCorrelationCalculator  # noqa: F401
from .interface import InterfaceReflectionAnalyzer  # noqa: F401
from .management import BoundaryComputationConfig  # noqa: F401
from .management import (
    DilationConfig,
    PadConfig,
    ProcessorConfig,
    ProcessorMetadata,
    ProcessorRegistry,
    ValidationResult,
    align_and_reshape,
    compute_amplitude_stats,
    compute_quartiles,
    compute_vertical_gradient,
    convert_numpy_scalars_to_float,
    create_processor,
    extract_amplitude_subset,
    filter_finite_values,
    flatten_and_filter_finite,
    get_default_processor_registry,
    get_default_statistics_strategy,
    register_processor,
    reshape_3d_to_2d,
    set_default_statistics_strategy,
)
from .operations import AlignmentOps, ExtractionOps, ReshapeOps, StatsOps  # noqa: F401
from .types import CorrelationFunction  # noqa: F401
from .types import (
    BoolArray,
    CorrelationResult,
    FilterResult,
    Float64Array,
    Int64Array,
    OptionalArrayPair,
)
from .validators import ArrayValidator, Validatable, ValidationHelpers  # noqa: F401

__all__ = [
    # Processor classes
    "BoundaryDetector",
    "CubeAligner",
    "BoundaryAmplitudeExtractor",
    "GradientCorrelationCalculator",
    "InterfaceReflectionAnalyzer",
    "FaciesDiscriminationCalculator",
    # Base classes
    "Processor",
    "BaseProcessor",
    # Configuration
    "ProcessorConfig",
    "BoundaryComputationConfig",
    "NeighborDirection",
    "ValidationResult",
    "PadConfig",
    "DilationConfig",
    # Utility functions
    "convert_numpy_scalars_to_float",
    "compute_quartiles",
    "filter_finite_values",
    "flatten_and_filter_finite",
    "reshape_3d_to_2d",
    "align_and_reshape",
    "compute_vertical_gradient",
    "extract_amplitude_subset",
    "compute_amplitude_stats",
    "set_default_statistics_strategy",
    "get_default_statistics_strategy",
    # Array validation
    "ArrayValidator",
    "ValidationHelpers",
    "Validatable",
    # Operations (Phase 6c)
    "AlignmentOps",
    "ReshapeOps",
    "ExtractionOps",
    "StatsOps",
    # Decorators
    "ProcessorDecorators",
    # Registry
    "ProcessorRegistry",
    "ProcessorMetadata",
    "get_default_processor_registry",
    "register_processor",
    "create_processor",
    # Exceptions
    "ProcessorError",
    "ValidationError",
    "CorrelationError",
    "ReshapeError",
    # Type aliases
    "CorrelationFunction",
    "CorrelationResult",
    "FilterResult",
    "OptionalArrayPair",
    "Float64Array",
    "Int64Array",
    "BoolArray",
]
