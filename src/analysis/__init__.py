"""Analysis pipelines package (formerly `src.regenerate`).

This package contains multi-step pipelines for seismic and rock-physics
analysis. Invoke with e.g. `python -m src.analysis.seismograms`.

New modules (integrated):
    - exceptions: Structured exception hierarchy for error handling
    - validators: Reusable validation utilities
    - cache.extractors: Data extraction strategies
"""

import logging

from .exceptions import (
    AnalysisException,
    CacheError,
    CacheLoadingError,
    CacheSelectionError,
    CacheExtractionError,
    ValidationError,
    DomainError,
    ProcessingError,
    ConfigurationError,
)
from .validators import (
    RangeValidator,
    CountValidator,
    QuantileValidator,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Modules
    "common_imports",
    "common",
    "header",
    "rock_physics",
    "seismograms",
    # Exceptions (new)
    "AnalysisException",
    "CacheError",
    "CacheLoadingError",
    "CacheSelectionError",
    "CacheExtractionError",
    "ValidationError",
    "DomainError",
    "ProcessingError",
    "ConfigurationError",
    # Validators (new)
    "RangeValidator",
    "CountValidator",
    "QuantileValidator",
]
