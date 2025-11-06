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
>>> boundaries = detector.detect(facies_cube)  # Returns bool mask
>>> aligner = CubeAligner()
>>> seismic = np.random.randn(10, 10, 20)
>>> seismic_aligned, facies_aligned = aligner.align(seismic, facies_cube)
"""

# Phase 2 modules (utilities and helpers)
from .decorators import ProcessorDecorators  # noqa: F401
from .registry import (  # noqa: F401
    ProcessorRegistry,
    ProcessorMetadata,
    get_default_processor_registry,
    register_processor,
    create_processor,
)
from .exceptions import (  # noqa: F401
    CorrelationError,
    ProcessorError,
    ReshapeError,
    ValidationError,
)
from .validators import (  # noqa: F401
    ArrayValidator,
    ValidationHelpers,
    Validatable,
    _ValidationErrors,
)
from .utils import ProcessorUtils  # noqa: F401
from .types import (  # noqa: F401
    BoolArray,
    CorrelationFunction,
    CorrelationResult,
    FilterResult,
    Float64Array,
    Int64Array,
    OptionalArrayPair,
    StatsType,
    T,
)
from .config import (  # noqa: F401
    BoundaryComputationConfig,
    NeighborDirection,
    ProcessorConfig,
    ValidationResult,
)
from .base import BaseProcessor, Processor  # noqa: F401

# Phase 3 modules (processor implementations)
from .boundary import BoundaryDetector, CubeAligner  # noqa: F401
from .amplitude import BoundaryAmplitudeExtractor  # noqa: F401
from .gradient import GradientCorrelationCalculator  # noqa: F401
from .interface import InterfaceReflectionAnalyzer  # noqa: F401
from .discrimination import FaciesDiscriminationCalculator  # noqa: F401

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
    # Utilities
    "ProcessorUtils",
    "ArrayValidator",
    "ValidationHelpers",
    "Validatable",
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
