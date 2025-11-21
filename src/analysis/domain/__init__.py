"""Domain-specific analysis handling subpackage.

This subpackage provides polymorphic domain handlers for depth/time processing,
eliminating string-based conditionals and following the Strategy pattern.

Public API:
    - Domain: Enum for depth/time domain selection
    - DomainHandler: Abstract base for domain handling
    - DepthDomainHandler: Depth-domain strategy implementation
    - TimeDomainHandler: Time-domain strategy implementation
    - DomainHandlerFactory: Factory for creating domain handlers
    - DomainHandlerRegistry: Registry for available handlers
    - DisplayCubes: Named tuple for display cube results
    - HandlerStatistics: Statistics for handler operations

Example:
    >>> from src.analysis.domain import DomainHandlerFactory, Domain
    >>> handler = DomainHandlerFactory.get_handler(Domain.DEPTH)
    >>> avo_display, facies_display = handler.prepare_display_cubes(...)
"""

from .enum import Domain
from .handlers import (
    CubeProcessor,
    DepthDomainHandler,
    DisplayCubes,
    DomainHandler,
    DomainHandlerFactory,
    DomainHandlerRegistry,
    HandlerStatistics,
    TimeDomainHandler,
)

__all__ = [
    "Domain",
    "DomainHandler",
    "DepthDomainHandler",
    "TimeDomainHandler",
    "DomainHandlerFactory",
    "DomainHandlerRegistry",
    "DisplayCubes",
    "HandlerStatistics",
    "CubeProcessor",
]
