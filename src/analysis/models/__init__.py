"""Analysis data models package.

This package organizes structured result types and configurations used by
analysis modules.

- Organization:
- base: Utilities, validation, and abstract base classes
- config: Configuration and core domain models (Transition,
    FaciesCorrelationConfig)
- cache: Cached data and display results
- facies: Facies-specific models (FaciesStats)
- statistics: Statistical analysis results (Gradient, Boundary, Discrimination,
    Interface, AvoAnalysis)
- avo: AVO-specific models (AvoStats, AvoResults, TechniqueComparison)
"""

from __future__ import annotations

# This package aggregates many public symbols and long import lines for
# convenience; silence long-line warnings at module scope.

# Duplicate `__all__` exports across small modules are intentional for
# explicit public API assembly; silence duplicate-code here.
# pylint: disable=duplicate-code

# AVO-specific models
from .avo import AvoResults, AvoStats, TechniqueComparison

# Base utilities and abstract classes
from .base import (
    ANALYSIS_COMPONENTS_COUNT,
    STATS_REPR_PRECISION,
    STR_PRECISION,
    SUMMARY_PRECISION,
    ModelUtilities,
    StatisticalResult,
    ValidationConfig,
)

# Cache-related models
from .cache import CacheLoadResult, DisplayCubesResult

# Configuration and domain models
from .config import FaciesCorrelationConfig, Transition

# Facies models
from .facies import FaciesStats

# Statistical result models
from .statistics import (
    AvoAnalysisResult,
    BoundaryAmpsResult,
    FaciesDiscriminationResult,
    GradientCorrelationResult,
    InterfaceReflectionResult,
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

# This package file assembles many imports and long names for public export.
# Lines may exceed the configured column width due to grouped imports and
# documented __all__ entries. Silence line-too-long here for readability.
