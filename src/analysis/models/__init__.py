"""Analysis data models package.

This package organizes structured result types and configurations used by
analysis modules.

Organization:
- base: Utilities, validation, and abstract base classes
- config: Configuration and core domain models (Transition, FaciesCorrelationConfig)
- cache: Cached data and display results
- facies: Facies-specific models (FaciesStats)
- statistics: Statistical analysis results (Gradient, Boundary, Discrimination, Interface, AvoAnalysis)
- avo: AVO-specific models (AvoStats, AvoResults, TechniqueComparison)
"""

from __future__ import annotations

# Base utilities and abstract classes
from .base import (
    ValidationConfig,
    ModelUtilities,
    StatisticalResult,
    STATS_REPR_PRECISION,
    STR_PRECISION,
    SUMMARY_PRECISION,
    ANALYSIS_COMPONENTS_COUNT,
)

# Configuration and domain models
from .config import (
    Transition,
    FaciesCorrelationConfig,
)

# Cache-related models
from .cache import (
    CacheLoadResult,
    DisplayCubesResult,
)

# Facies models
from .facies import (
    FaciesStats,
)

# Statistical result models
from .statistics import (
    GradientCorrelationResult,
    BoundaryAmpsResult,
    FaciesDiscriminationResult,
    InterfaceReflectionResult,
    AvoAnalysisResult,
)

# AVO-specific models
from .avo import (
    TechniqueComparison,
    AvoStats,
    AvoResults,
)

__all__ = [
    # Base utilities
    "ValidationConfig",
    "ModelUtilities",
    "StatisticalResult",
    "STATS_REPR_PRECISION",
    "STR_PRECISION",
    "SUMMARY_PRECISION",
    "ANALYSIS_COMPONENTS_COUNT",
    # Configuration
    "Transition",
    "FaciesCorrelationConfig",
    # Cache
    "CacheLoadResult",
    "DisplayCubesResult",
    # Facies
    "FaciesStats",
    # Statistical results
    "GradientCorrelationResult",
    "BoundaryAmpsResult",
    "FaciesDiscriminationResult",
    "InterfaceReflectionResult",
    "AvoAnalysisResult",
    # AVO
    "TechniqueComparison",
    "AvoStats",
    "AvoResults",
]
