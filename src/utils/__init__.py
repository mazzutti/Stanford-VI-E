"""Utilities package with clean OOP design.

Provides:
- Quantity: Unit-aware array wrapper with composition-based conversions
- UnitRegistry & Converters: OOP strategy pattern for unit conversions
- LRU caching: Thread-safe cache implementations
- Type utilities: Protocols and type definitions
"""

import logging

from src.utils.units import (
    Converter,
    VelocityConverter,
    DensityConverter,
    TimeConverter,
    LengthConverter,
    UnitRegistry,
    get_unit_registry,
    unit_registry,
)
from src.utils.quantity import Quantity
from src.utils.lru import LRUCache, ShardedLRUCache
from src.utils.types import ProcessManagerProtocol
from src.utils.normalizers import UnitNormalizer
from src.utils.converters import UnitConverter

logger = logging.getLogger(__name__)

__all__ = [
    # Converters (OOP strategy pattern)
    "Converter",
    "VelocityConverter",
    "DensityConverter",
    "TimeConverter",
    "LengthConverter",
    "UnitConverter",
    # Registry
    "UnitRegistry",
    "get_unit_registry",
    "unit_registry",
    # Quantity
    "Quantity",
    # Caching
    "LRUCache",
    "ShardedLRUCache",
    # Protocols
    "ProcessManagerProtocol",
    # Utilities
    "UnitNormalizer",
]
