"""Type definitions, protocols, and abstractions for the analysis module.

This package consolidates type contracts and computational abstractions used
throughout the analysis system, including:

- Type variables and generic constraints (from protocols.py)
- Protocols for domain abstraction (from protocols.py)
- Base computational components (from base.py)
- Domain enumeration (from domain.enum)
"""

# Import domain from canonical location
from src.analysis.domain.enum import Domain

# Import computational abstractions from canonical module
from src.analysis.types.base import (
    AnalysisSchema,
    ComputationResult,
    Computer,
    T_In,
    T_Out,
)

# Import protocols from canonical module
from src.analysis.types.protocols import (
    ArchiveExtractorProtocol,
    CacheLoaderProtocol,
    CacheProtocol,
    DatasetManagerFactory,
    PlotterProtocol,
    ResamplePlan,
    Resampler,
    ResamplerFactory,
    SelectorProtocol,
    T,
    TimeResampler,
)

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
