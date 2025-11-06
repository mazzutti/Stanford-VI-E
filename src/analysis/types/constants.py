"""Type variables for analysis workflows."""

from typing import TypeVar

__all__ = ["T"]


# ============================================================================
# Type Variables
# ============================================================================

# Type variables enable flexible, reusable protocols that work with any type:
#   - CacheProtocol[T] works with any value type
#   - Preserves type safety in generic operations
#   - Essential for creating reusable abstractions
T = TypeVar("T")
