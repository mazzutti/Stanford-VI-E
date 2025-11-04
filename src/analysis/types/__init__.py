"""Type definitions, protocols, and abstractions for the analysis module.

This package consolidates type contracts and computational abstractions used
throughout the analysis system, including:

- Type variables and generic constraints
- Protocols for domain abstraction (resampling, caching, factories)
- Base computational components (Computer, AnalysisSchema, ComputationResult)
"""

# Re-export commonly-used types and protocols
from src.analysis.types.base import (
    T,
    T_In,
    T_Out,
    ResamplePlan,
    Resampler,
    ResamplerFactory,
    TimeResampler,
    CacheLoaderProtocol,
    CacheProtocol,
    SelectorProtocol,
    ArchiveExtractorProtocol,
    DatasetManagerFactory,
    PlotterProtocol,
    Computer,
    AnalysisSchema,
    ComputationResult,
)

from src.analysis.domain.enum import Domain

__all__ = [
    "T",
    "T_In",
    "T_Out",
    "Domain",
    "ResamplePlan",
    "Resampler",
    "ResamplerFactory",
    "TimeResampler",
    "CacheLoaderProtocol",
    "CacheProtocol",
    "SelectorProtocol",
    "ArchiveExtractorProtocol",
    "DatasetManagerFactory",
    "PlotterProtocol",
    "Computer",
    "AnalysisSchema",
    "ComputationResult",
]
