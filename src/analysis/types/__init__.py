"""Type protocols and type variables for analysis workflows.

This module defines structural types (Protocols) and type variables used
across analysis modules. These protocols enable duck-typing and flexible
implementations while maintaining static type safety through Python's
Protocol (Structural Subtyping) system.

Design Principles:
    - Protocols define interfaces without mandating inheritance
    - Type variables (T) enable generic, reusable protocols
    - TYPE_CHECKING imports prevent circular dependencies
    - All types are composable and reusable across modules
    - This is a types-only module—no runtime implementations

Protocol Categories:
    Resampling:
        - ResamplePlan: Marker protocol for resampling plans
        - Resampler: Domain conversion (time ↔ depth)
        - ResamplerFactory: Creates Resampler instances
        - TimeResampler: Uniform time resampling

    Caching:
        - CacheLoaderProtocol: File selection and loading
        - CacheProtocol[T]: Generic cache interface
        - SelectorProtocol: Custom file selection strategy
        - ArchiveExtractorProtocol: Archive handling strategy

    Factories:
        - ResamplerFactory: Resampler creation
        - DatasetManagerFactory: Dataset manager creation

    Domain & Visualization:
        - Domain: Enum for 'depth' and 'time' domains
        - PlotterProtocol: Summary plot generation

Usage Example:
    >>> from src.analysis.types import CacheProtocol, Domain
    >>> # Domain enum with helper methods
    >>> assert Domain.DEPTH.is_depth()
    >>> # Generic cache protocol
    >>> class ArrayCache(CacheProtocol[NDArray]): ...
"""

from .constants import T, Domain
from .protocols import (
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
)

__all__ = [
    "T",
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
]
